"""Tokenization, dataset, and dataloader utilities."""

import re

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .vokab import BOS_TOKEN, EOS_TOKEN


def load_data(data_path, val_size=0.1, seed=42):
    dataframe = pd.read_csv(data_path)
    full_train_df = dataframe[dataframe["split"] == "train"].reset_index(drop=True)
    test_df = dataframe[dataframe["split"] == "test"].reset_index(drop=True)
    train_df, val_df = train_test_split(
        full_train_df,
        test_size=val_size,
        random_state=seed,
        stratify=full_train_df["label"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df


def get_tokenizer():
    def tokenizer(text):
        return re.findall(r"\b\w+\b", text.lower())

    return tokenizer


class IMDbDataset(Dataset):
    def __init__(self, dataframe, tokenizer, vocab, max_seq_len=200, pad_value=0):
        bos_idx = vocab[BOS_TOKEN]
        eos_idx = vocab[EOS_TOKEN]
        sequences = []
        for text in dataframe["text"].values:
            indices = vocab.lookup_indices(tokenizer(str(text)))[: max_seq_len - 2]
            sequences.append(torch.tensor([bos_idx, *indices, eos_idx], dtype=torch.long))

        self.input_ids = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=pad_value
        )
        self.labels = torch.tensor(dataframe["label"].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.input_ids[index], self.labels[index]


def create_dataloader(
    dataframe, tokenizer, vocab, batch_size, max_seq_len=200, pad_value=0, shuffle=True
):
    dataset = IMDbDataset(dataframe, tokenizer, vocab, max_seq_len, pad_value)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def create_embedding_vectors(vocab, embedding_dim):
    # Import lazily so inference does not require gensim to initialize.
    import gensim.downloader as gensim_api

    glove_model = gensim_api.load(f"glove-wiki-gigaword-{embedding_dim}")
    vectors = torch.zeros(len(vocab), embedding_dim)
    for index, token in enumerate(vocab.get_itos()):
        if token in glove_model:
            vectors[index] = torch.tensor(glove_model[token])
    return vectors


def get_lengths(input_ids, pad_value=0):
    return input_ids.ne(pad_value).sum(dim=1).cpu().tolist()
