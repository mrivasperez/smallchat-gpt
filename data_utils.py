'''
This file is part of SmallchatGPT.

SmallchatGPT is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

SmallchatGPT is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with SmallchatGPT. If not, see <https://www.gnu.org/licenses/>.
'''


import torch
from torch.utils.data import Dataset
import json


class JSONDataset(Dataset):
    def __init__(self, file_path, block_size):
        self.block_size = block_size
        self.vocab = set()
        self.data = []

        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        for item in json_data["data"]:
            story = item["story"]
            questions = item["questions"]
            answers = item["answers"]
            qa_pairs = []

            # Add story to vocabulary
            self.vocab.update(list(story))

            for i in range(len(questions)):
                question = questions[i]["input_text"]
                answer = answers[i]["input_text"]
                self.vocab.update(list(question))
                self.vocab.update(list(answer))
                qa_pairs.append((question, answer))

            self.data.append((story, qa_pairs))

        # Add special tokens
        self.vocab.add('<pad>')  # Padding token
        self.vocab.add('<s>')    # Start of sequence token
        self.vocab.add('</s>')   # End of sequence token

        self.word2idx = {word: i for i, word in enumerate(sorted(self.vocab))}
        self.idx2word = {i: word for word, i in self.word2idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        story, qa_pairs = self.data[idx]

        indexed_context = [self.word2idx['<s>']]

        # Add story to the sequence
        indexed_context.extend(
            [self.word2idx.get(word, self.word2idx['<pad>']) for word in list(story)])
        indexed_context.append(self.word2idx['</s>'])

        # Add question/answer pairs to the sequence
        for question, answer in qa_pairs:
            indexed_context.extend(
                [self.word2idx.get(word, self.word2idx['<pad>']) for word in list(question)])
            indexed_context.append(self.word2idx['</s>'])
            indexed_context.extend(
                [self.word2idx.get(word, self.word2idx['<pad>']) for word in list(answer)])
            indexed_context.append(self.word2idx['</s>'])

        # Pad or truncate the context
        if len(indexed_context) >= self.block_size:
            indexed_context = indexed_context[:self.block_size]
        else:
            padding_length = self.block_size - len(indexed_context)
            indexed_context += [self.word2idx['<pad>']] * padding_length

        # Create input and target tensors
        x = torch.tensor(indexed_context[:-1], dtype=torch.long)
        y = torch.tensor(indexed_context[1:], dtype=torch.long)

        return x, y
