"""Model definitions for the BiLSTM sentiment classifier."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_outputs):
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy, dim=1)
        context = torch.sum(encoder_outputs * weights, dim=1)
        return context, weights


class BiLSTM(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, hidden_dim, num_layers, num_classes, pad_value=0):
        super(BiLSTM, self).__init__()
        self.vocab_dim = vocab_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.pad_value = float(pad_value)

        self.emb = nn.Embedding(vocab_dim, embedding_dim, padding_idx=int(pad_value))
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = SelfAttention(self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, self.num_classes)

    def update_embedding(self, vectors):
        with torch.no_grad():
            self.emb.weight.copy_(vectors)
        self.emb.weight.requires_grad = True  # Allow fine-tuning of embeddings during training

    def dropout(self, v):
        return F.dropout(v, p=0.5, training=self.training)

    def forward(self, x, lengths):
        x = self.emb(x)
        x = self.dropout(x)

        lengths_cpu = torch.tensor(lengths, dtype=torch.int64, device="cpu")
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed_x)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        out = out[:, :, :self.hidden_dim] + out[:, :, self.hidden_dim:]

        context, attn_weights = self.attention(out)
        outputs = self.fc(context)

        return outputs, attn_weights
    