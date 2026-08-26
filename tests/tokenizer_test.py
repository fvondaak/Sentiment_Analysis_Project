"""Print the current BiLSTM tokenizer output for several edge cases."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from own_nn import get_tokenizer


tokenizer = get_tokenizer()


def print_tokens(case, text):
    print(f"{case}: {text!r}")
    print(f"Tokens: {tokenizer(text)}\n")


def normal_text():
    print_tokens(
        "Normal text",
        "This movie has a wonderful story and great acting.",
    )


def contractions():
    print_tokens("Contractions", "I don't think it wasn't bad.")


def html():
    print_tokens(
        "IMDb HTML",
        "Great movie!<br /><br />Would watch again &amp; recommend.",
    )


def capitalization_and_punctuation():
    print_tokens("Capitalization and punctuation", "AWFUL!!! Really? Yes: awful.")


def hyphenated_words():
    print_tokens("Hyphenated words", "A well-written, thought-provoking film.")


def numbers_and_underscores():
    print_tokens(
        "Numbers and underscores",
        "A 10/10 movie from 2024; user_score was 9.5.",
    )


def unicode_characters():
    print_tokens("Unicode characters", "Café, naïve, façade—über!")


def empty_text():
    print_tokens("Empty text", "")
    print_tokens("Whitespace and punctuation only", "  \n\t...!?  ")


if __name__ == "__main__":
    normal_text()
    contractions()
    html()
    capitalization_and_punctuation()
    hyphenated_words()
    numbers_and_underscores()
    unicode_characters()
    empty_text()
