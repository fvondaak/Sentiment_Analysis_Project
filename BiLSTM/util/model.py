"""BiLSTM and self-attention model definitions."""

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
    def __init__(
        self, vocab_dim, embedding_dim, hidden_dim, num_layers, num_classes=1, pad_value=0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb = nn.Embedding(vocab_dim, embedding_dim, padding_idx=int(pad_value))
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = SelfAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def update_embedding(self, vectors):
        self.emb.weight.data.copy_(vectors)
        self.emb.weight.requires_grad = False

    def forward(self, input_ids, lengths):
        embedded = F.dropout(self.emb(input_ids), p=0.5, training=self.training)
        lengths_cpu = torch.as_tensor(lengths, dtype=torch.int64, device="cpu")
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        output = output[:, :, : self.hidden_dim] + output[:, :, self.hidden_dim :]
        context, attention_weights = self.attention(output)
        return self.fc(context), attention_weights
