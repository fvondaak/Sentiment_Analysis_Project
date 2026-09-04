"""Vocabulary utilities for the BiLSTM sentiment classifier."""

import os
import pickle


PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"

DEFAULT_VOCAB_PATH = os.path.join("own_nn_artifacts", "vocab.pkl")


class CustomVocabulary:
    def __init__(self, token_to_idx, idx_to_token, unknown_token):
        self.token_to_idx = token_to_idx
        self.idx_to_token = idx_to_token
        self.unknown_token = unknown_token
        self.unk_idx = token_to_idx[unknown_token]

    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.unk_idx)

    def __contains__(self, token):
        return token in self.token_to_idx

    def __len__(self):
        return len(self.idx_to_token)

    def lookup_indices(self, tokens):
        return [self.__getitem__(token) for token in tokens]

    def get_itos(self):
        return self.idx_to_token


def create_vocabulary(
    dataframe,
    tokenizer,
    pad_token=PAD_TOKEN,
    unknown_token=UNK_TOKEN,
    bos_token=BOS_TOKEN,
    eos_token=EOS_TOKEN,
):
    idx_to_token = [pad_token, unknown_token, bos_token, eos_token]
    token_to_idx = {pad_token: 0, unknown_token: 1, bos_token: 2, eos_token: 3}

    unique_words = set()
    for sentence in dataframe["text"].values:
        tokens = tokenizer(str(sentence))
        for token in tokens:
            unique_words.add(token)

    for token in sorted(unique_words):
        if token not in token_to_idx:
            token_to_idx[token] = len(idx_to_token)
            idx_to_token.append(token)

    return CustomVocabulary(token_to_idx, idx_to_token, unknown_token)


def save_vocab(vocab, path=DEFAULT_VOCAB_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(
            {
                "token_to_idx": vocab.token_to_idx,
                "idx_to_token": vocab.idx_to_token,
                "unknown_token": vocab.unknown_token,
            },
            file,
        )


def load_vocab(path=DEFAULT_VOCAB_PATH):
    with open(path, "rb") as file:
        data = pickle.load(file)
    return CustomVocabulary(
        data["token_to_idx"], data["idx_to_token"], data["unknown_token"]
    )
