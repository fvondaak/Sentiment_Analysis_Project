"""Unit tests for the BiLSTM vocabulary."""

import pandas as pd
import pytest

from own_nn import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    create_vocabulary,
    get_tokenizer,
)


@pytest.fixture
def dataframe():
    return pd.DataFrame(
        {
            "text": [
                "Good movie",
                "Bad movie",
                "GOOD acting",
            ]
        }
    )


@pytest.fixture
def tokenizer():
    return get_tokenizer()


@pytest.fixture
def vocab(dataframe, tokenizer):
    return create_vocabulary(dataframe, tokenizer)


def test_special_token_indices(vocab):
    assert vocab[PAD_TOKEN] == 0
    assert vocab[UNK_TOKEN] == 1
    assert vocab[BOS_TOKEN] == 2
    assert vocab[EOS_TOKEN] == 3


def test_training_tokens_are_unique_and_sorted(vocab):
    assert vocab.get_itos() == [
        PAD_TOKEN,
        UNK_TOKEN,
        BOS_TOKEN,
        EOS_TOKEN,
        "acting",
        "bad",
        "good",
        "movie",
    ]
    assert len(vocab) == 8


def test_known_and_unknown_token_lookup(vocab):
    assert "good" in vocab
    assert vocab["good"] != vocab[UNK_TOKEN]

    assert "excellent" not in vocab
    assert vocab["excellent"] == vocab[UNK_TOKEN]


def test_token_and_index_mappings_are_consistent(vocab):
    for token, index in vocab.token_to_idx.items():
        assert vocab.idx_to_token[index] == token

    indices = list(vocab.token_to_idx.values())
    assert len(indices) == len(set(indices))

 
def test_lookup_indices(vocab):
    assert vocab.lookup_indices(["good", "missing", "movie"]) == [
        vocab["good"],
        vocab[UNK_TOKEN],
        vocab["movie"],
    ]


def test_vocabulary_is_deterministic(dataframe, tokenizer, vocab):
    shuffled = dataframe.sample(frac=1, random_state=17).reset_index(drop=True)
    second_vocab = create_vocabulary(shuffled, tokenizer)

    assert vocab.token_to_idx == second_vocab.token_to_idx
    assert vocab.idx_to_token == second_vocab.idx_to_token
