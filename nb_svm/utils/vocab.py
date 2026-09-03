"""Vocabulary for NB-SVM unigram and bigram features."""


class NBSVMVocabulary:
    """Map observed unigrams and bigrams to one feature index space."""

    def __init__(self, token_to_idx, idx_to_token, bigram_to_idx, idx_to_bigram):
        self.token_to_idx = token_to_idx
        self.idx_to_token = idx_to_token
        self.bigram_to_idx = bigram_to_idx
        self.idx_to_bigram = idx_to_bigram

    @classmethod
    def from_dataframe(cls, dataframe, tokenizer):
        """Build a vocabulary from the text column of a training dataframe."""
        tokens = set()
        bigrams = set()

        for text in dataframe["text"].values:
            document_tokens = tokenizer(str(text))
            tokens.update(document_tokens)  # unigrams
            bigrams.update(zip(document_tokens, document_tokens[1:]))  # Create bigrams from two consecutive tokens

        sorted_tokens = sorted(tokens)
        sorted_bigrams = sorted(bigrams)

        token_to_idx = {token: idx for idx, token in enumerate(sorted_tokens)}  # sorted
        idx_to_token = {idx: token for token, idx in token_to_idx.items()}

        bigram_offset = len(token_to_idx)  # first tokens, then bigrams
        bigram_to_idx = {
            bigram: bigram_offset + idx
            for idx, bigram in enumerate(sorted_bigrams)
        }
        idx_to_bigram = {
            idx: bigram for bigram, idx in bigram_to_idx.items()
        }

        return cls(token_to_idx, idx_to_token, bigram_to_idx, idx_to_bigram)

    def lookup_token(self, token):
        """Return a unigram index or None when it is unknown."""
        return self.token_to_idx.get(token)

    def lookup_bigram(self, bigram):
        """Return a bigram index or None when it is unknown."""
        return self.bigram_to_idx.get(bigram)

    def __len__(self):
        return len(self.token_to_idx) + len(self.bigram_to_idx)
