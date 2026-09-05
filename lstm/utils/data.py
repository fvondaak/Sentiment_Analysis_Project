'''Data utilities for handling datasets and data loaders.'''

from pathlib import Path

import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

from .vocab import BOS_TOKEN, EOS_TOKEN

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_DATA_PATH = PROJECT_ROOT / "dataset" / "train.csv"
DEFAULT_TEST_DATA_PATH = PROJECT_ROOT / "dataset" / "test.csv"
MAX_SEQ_LEN = 400

class IMDBDataset(Dataset):
    def __init__(self, df, tokenizer, vocab, max_seq_len=MAX_SEQ_LEN, pad_value=0):
        super(IMDBDataset, self).__init__()
        all_text = [t for t in df["text"].values]
        labels = df["label"].values

        self.vocab = vocab
        self.max_seq_len = max_seq_len

        proc_seq = []
        bos_idx = vocab[BOS_TOKEN]
        eos_idx = vocab[EOS_TOKEN]

        for t in all_text:
            tokens = tokenizer(str(t))
            indices = [vocab[tok] for tok in tokens]
            indices = indices[:(max_seq_len - 2)]
            formatted_seq = [bos_idx] + indices + [eos_idx]
            proc_seq.append(torch.tensor(formatted_seq, dtype=torch.long))

        self.input_ids = torch.nn.utils.rnn.pad_sequence(
            proc_seq, batch_first=True, padding_value=pad_value
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]


def load_data(
    train_data_path=DEFAULT_TRAIN_DATA_PATH,
    test_data_path=DEFAULT_TEST_DATA_PATH,
    val_size=0.1,
    seed=42,
):
    full_train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path).reset_index(drop=True)

    train_df, val_df = train_test_split(
        full_train_df,
        test_size=val_size,
        random_state=seed,
        stratify=full_train_df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    return train_df, val_df, test_df

def create_dataloader(df, tokenizer, vocab, batch_size, max_seq_len=MAX_SEQ_LEN, pad_value=0, shuffle=True):
    dataset = IMDBDataset(df, tokenizer, vocab, max_seq_len, pad_value)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_length(x, pad_value=0):
    length = []
    for i in x.cpu().tolist():
        length.append(len(i) - i.count(pad_value))
    return length
