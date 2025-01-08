'''
This file is part of SmallchatGPT.

SmallchatGPT is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

SmallchatGPT is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with SmallchatGPT. If not, see <https://www.gnu.org/licenses/>.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import os
from data_utils import JSONDataset
from utils import save_checkpoint, interrupted

# --- Model Definition ---


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(
            n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(
                1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)

            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# --- Hyperparameters ---
file_path = 'data.json'  # Update with your JSON file path
model_save_path = 'trained_model.pt'
block_size = 512
batch_size = 16
n_embd = 384
n_head = 8
n_layer = 6
dropout = 0.1
num_epochs = 15
learning_rate = 1e-4

# Create dataset and split into training and validation sets
dataset = JSONDataset(file_path, block_size)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

vocab_size = len(dataset.vocab)

# Initialize model
model = GPTLanguageModel(vocab_size, n_embd, block_size,
                         n_head, n_layer, dropout)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, weight_decay=1e-1)

# Create a SummaryWriter instance
writer = SummaryWriter()


# --- Training Loop ---


def train(model, data_loader, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0
    num_batches = len(data_loader)
    for batch_idx, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Check for interruption signal after each batch
        if interrupted:
            print("Saving model and exiting...")
            save_checkpoint(model, optimizer, epoch,
                            model_save_path, batch_idx, best_val_loss)
            return  # Exit the training function

        if batch_idx % 10 == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.4f}")
            # Log training loss to TensorBoard
            writer.add_scalar(
                "Loss/train", loss.item(), epoch * num_batches + batch_idx)

    return total_loss / num_batches

# --- Validation Loop ---


def validate(model, data_loader, device, epoch, writer):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            total_loss += loss.item()

    avg_val_loss = total_loss / len(data_loader)
    # Log validation loss to TensorBoard
    writer.add_scalar("Loss/val", avg_val_loss, epoch)
    return avg_val_loss


# --- Check for Saved Model ---
if os.path.exists(model_save_path):
    # Load the saved model
    print("Loading saved model...")
    checkpoint = torch.load(model_save_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']

    # Handle case where these were not saved in the checkpoint (older version)
    # Default to -1 (end of epoch) if not found
    start_batch = checkpoint.get('batch_idx', -1)
    # Default to infinity if not found
    best_val_loss = checkpoint.get('val_loss', float('inf'))

    model.to(device)  # Make sure to move the loaded model to the correct device
    print(f"Model loaded! Resuming training from epoch {start_epoch + 1}")
else:
    # Train a new model
    print("No saved model found. Starting training...")
    start_epoch = 0
    start_batch = -1
    best_val_loss = float('inf')

# Train a new model
for epoch in range(start_epoch, num_epochs):
    data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    train_loss = train(model, train_loader, optimizer, device, epoch, writer)
    val_loss = validate(model, val_loader, device, epoch, writer)
    print(
        f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Save the model if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print("Saving model...")
        save_checkpoint(model, optimizer, epoch,
                        model_save_path, val_loss=best_val_loss)
        print("Model saved!")

# --- Chat Loop ---
model.eval()
with torch.no_grad():
    while True:
        context = input("You: ")
        if context.lower() == "quit":
            break

        indexed_context = [dataset.word2idx['<s>']] + [dataset.word2idx.get(
            word, dataset.word2idx['<pad>']) for word in context]
        x = torch.tensor(indexed_context, dtype=torch.long,
                         device=device).unsqueeze(0)

        generated_text = model.generate(
            x, max_new_tokens=50, temperature=0.7, do_sample=True)[0].tolist()

        decoded_text = ''.join(
            [dataset.idx2word[idx] for idx in generated_text if idx != dataset.word2idx['<pad>']])

        # Find the start of the response in the decoded text
        response_start_index = decoded_text.find(
            '<s>') + len('<s>')  # Skip the <s> token
        if response_start_index != -1:  # Find the end of the response (</s>)
            response_end_index = decoded_text.find(
                '</s>', response_start_index)
            if response_end_index != -1:
                response = decoded_text[response_start_index:response_end_index]
            else:
                # Fallback: take everything after <s>
                response = decoded_text[response_start_index:]
        else:
            response = decoded_text

        # Remove leading/trailing whitespace
        response = response.strip()

        print(f"GPT: {response}")
