"""Vocabulary creation and persistence utilities."""

import pickle
from pathlib import Path


PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"


class CustomVocabulary:
    def __init__(self, token_to_idx, idx_to_token, unknown_token=UNK_TOKEN):
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
        return [self[token] for token in tokens]

    def get_itos(self):
        return self.idx_to_token


def create_vocabulary(dataframe, tokenizer):
    special_tokens = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
    idx_to_token = list(special_tokens)
    token_to_idx = {token: index for index, token in enumerate(special_tokens)}

    words = {
        token
        for sentence in dataframe["text"].values
        for token in tokenizer(str(sentence))
    }
    for token in sorted(words):
        if token not in token_to_idx:
            token_to_idx[token] = len(idx_to_token)
            idx_to_token.append(token)

    return CustomVocabulary(token_to_idx, idx_to_token)


def save_vocab(vocab, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(
            {
                "token_to_idx": vocab.token_to_idx,
                "idx_to_token": vocab.idx_to_token,
                "unknown_token": vocab.unknown_token,
            },
            file,
        )


def load_vocab(path):
    with Path(path).open("rb") as file:
        data = pickle.load(file)
    return CustomVocabulary(
        data["token_to_idx"], data["idx_to_token"], data["unknown_token"]
    )
