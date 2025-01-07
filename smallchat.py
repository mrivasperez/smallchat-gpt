import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

# --- Data Preprocessing ---


class TextDataset(Dataset):
    def __init__(self, file_path, block_size):
        self.block_size = block_size
        self.vocab = set()
        self.data = []

        # Read the file and create the vocabulary
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                question, answer = line.strip().split('\t')
                self.vocab.update(list(question))
                self.vocab.update(list(answer))
                self.data.append((question, answer))

        # Add special tokens
        self.vocab.add('<pad>')  # Padding token
        self.vocab.add('<s>')    # Start of sequence token
        self.vocab.add('</s>')   # End of sequence token

        self.word2idx = {word: i for i, word in enumerate(sorted(self.vocab))}
        self.idx2word = {i: word for word, i in self.word2idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, answer = self.data[idx]
        combined_text = '<s>' + question + '\t' + answer + '</s>'

        # Convert text to indices, handling unknown characters
        indexed_text = [self.word2idx.get(
            word, self.word2idx['<pad>']) for word in combined_text]

        # Pad or truncate to block_size
        if len(indexed_text) < self.block_size:
            indexed_text += [self.word2idx['<pad>']] * \
                (self.block_size - len(indexed_text))
        else:
            indexed_text = indexed_text[:self.block_size]

        # Input: all but last token
        x = torch.tensor(indexed_text[:-1], dtype=torch.long)
        # Target: shift by one (next token prediction)
        y = torch.tensor(indexed_text[1:], dtype=torch.long)
        return x, y

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
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# --- Hyperparameters ---
file_path = 'data.txt'
model_save_path = 'trained_model.pt'  # Path to save/load the model
block_size = 64
batch_size = 32
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.1
num_epochs = 10
learning_rate = 1e-3

# Create dataset
dataset = TextDataset(file_path, block_size)
vocab_size = len(dataset.vocab)

# Initialize model
model = GPTLanguageModel(vocab_size, n_embd, block_size,
                         n_head, n_layer, dropout)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# --- Check for Saved Model ---
if os.path.exists(model_save_path):
    # Load the saved model
    print("Loading saved model...")
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.to(device)  # Make sure to move the loaded model to the correct device
    print("Model loaded!")
else:
    # Train a new model
    print("No saved model found. Starting training...")
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        for batch_idx, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx +
                      1}/{len(data_loader)}, Loss: {loss.item():.4f}")

    # Save the trained model
    print("Training finished. Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_save_path)
    print("Model saved!")

# --- Chat Loop ---
model.eval()
with torch.no_grad():
    while True:
        context = input("You: ")
        if context.lower() == "quit":
            break

        indexed_context = [dataset.word2idx.get(
            word, dataset.word2idx['<pad>']) for word in context]
        x = torch.tensor(indexed_context, dtype=torch.long,
                         device=device).unsqueeze(0)

        generated_text = model.generate(x, max_new_tokens=50)[0].tolist()
        decoded_text = ''.join(
            [dataset.idx2word[idx] for idx in generated_text if idx != dataset.word2idx['<pad>']])

        response_end_index = decoded_text.find('</s>')
        if response_end_index != -1:
            response = decoded_text[len(context):response_end_index]
        else:
            response = decoded_text[len(context):]

        print(f"GPT: {response}")
