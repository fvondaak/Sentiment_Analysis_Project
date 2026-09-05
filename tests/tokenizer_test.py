"""Unit tests for the BiLSTM tokenizer.

Run ``pytest tests/tokenizer_test.py`` to check the assertions, or run the file
directly to print the tokenization of every case for manual inspection.
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.tokenizer import get_tokenizer


tokenizer = get_tokenizer()


def print_tokens(case, text):
    print(f"{case}: {text!r}")
    print(f"Tokens: {tokenizer(text)}\n")


def test_normal_text():
    assert tokenizer("This movie has a wonderful story and great acting.") == [
        "this", "movie", "has", "a", "wonderful", "story", "and", "great", "acting",
    ]


def test_contractions_expand_negation():
    # "wasn't bad" is positive, "bad" is negative -- splitting the contraction
    # into ["wasn", "t", "bad"] would hide the negation from the model.
    assert tokenizer("I don't think it wasn't bad.") == [
        "i", "do", "not", "think", "it", "was", "not", "bad",
    ]


def test_irregular_contractions():
    # The general "<word>n't" rule would mangle these into "wo not" / "ca not".
    assert tokenizer("It won't disappoint and I can't complain.") == [
        "it", "will", "not", "disappoint", "and", "i", "can", "not", "complain",
    ]


def test_typographic_apostrophe():
    # Text copied from the web often uses U+2019 instead of a plain quote.
    assert tokenizer("It wasn’t good.") == ["it", "was", "not", "good"]


def test_html_markup_is_removed():
    assert tokenizer(
        "Great movie!<br /><br />Would watch again &amp; recommend."
    ) == ["great", "movie", "would", "watch", "again", "recommend"]


def test_html_tags_do_not_glue_words_together():
    # Tags are replaced by a space, otherwise "plot<br />was" becomes "plotwas".
    assert tokenizer("The plot<br />was thin.") == ["the", "plot", "was", "thin"]


def test_html_entities_are_unescaped():
    assert tokenizer("A &quot;great&quot; film &mdash; truly.") == [
        "a", "great", "film", "truly",
    ]


def test_capitalization_and_punctuation():
    # Emphasis (caps, exclamation marks) is deliberately discarded; VADER uses
    # it as a signal, the BiLSTM does not see it.
    assert tokenizer("AWFUL!!! Really? Yes: awful.") == [
        "awful", "really", "yes", "awful",
    ]


def test_hyphenated_words_are_split():
    # Splitting is intentional: "well" and "written" both have GloVe vectors,
    # while "well-written" would become an out-of-vocabulary token.
    assert tokenizer("A well-written, thought-provoking film.") == [
        "a", "well", "written", "thought", "provoking", "film",
    ]


def test_numbers_and_underscores():
    assert tokenizer("A 10/10 movie from 2024; user_score was 9.5.") == [
        "a", "10", "10", "movie", "from", "2024", "user_score", "was", "9", "5",
    ]


def test_unicode_characters_are_kept():
    assert tokenizer("Café, naïve, façade—über!") == [
        "café", "naïve", "façade", "über",
    ]


def test_empty_and_punctuation_only_text():
    assert tokenizer("") == []
    assert tokenizer("  \n\t...!?  ") == []


if __name__ == "__main__":
    print_tokens("Normal text", "This movie has a wonderful story and great acting.")
    print_tokens("Contractions", "I don't think it wasn't bad.")
    print_tokens("Irregular contractions", "It won't disappoint and I can't complain.")
    print_tokens("Typographic apostrophe", "It wasn’t good.")
    print_tokens("IMDb HTML", "Great movie!<br /><br />Would watch again &amp; recommend.")
    print_tokens("HTML between words", "The plot<br />was thin.")
    print_tokens("HTML entities", "A &quot;great&quot; film &mdash; truly.")
    print_tokens("Capitalization and punctuation", "AWFUL!!! Really? Yes: awful.")
    print_tokens("Hyphenated words", "A well-written, thought-provoking film.")
    print_tokens("Numbers and underscores", "A 10/10 movie from 2024; user_score was 9.5.")
    print_tokens("Unicode characters", "Café, naïve, façade—über!")
    print_tokens("Empty text", "")
    print_tokens("Whitespace and punctuation only", "  \n\t...!?  ")
