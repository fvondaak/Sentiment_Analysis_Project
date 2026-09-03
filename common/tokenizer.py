"""Shared tokenizer used by the sentiment-analysis models."""

import html
import re


HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

IRREGULAR_CONTRACTIONS = {
    "won't": "will not",
    "can't": "can not",
    "shan't": "shall not",
    "ain't": "is not",
}
NT_CONTRACTION_PATTERN = re.compile(r"\b(\w+)n't\b")


def get_tokenizer():
    def tokenizer(text):
        text = HTML_TAG_PATTERN.sub(" ", str(text))
        text = html.unescape(text)
        text = text.lower()

        text = text.replace("’", "'")
        for contraction, expansion in IRREGULAR_CONTRACTIONS.items():
            text = text.replace(contraction, expansion)
        text = NT_CONTRACTION_PATTERN.sub(r"\1 not", text)

        tokens = re.findall(r"\b\w+\b", text)
        return tokens
    return tokenizer
