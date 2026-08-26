"""Convert the Kaggle IMDb CSV to the format used by all three models.

The source dataset downloaded by ``download_dataset.sh`` contains the columns
``review`` and ``sentiment``.  The model code expects ``text``, ``label`` and
``split``, where labels are 0 (negative) or 1 (positive).

The generated ``train``/``test`` split is deterministic and stratified.  The
BiLSTM creates its validation set later from the generated training partition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = REPO_DIR / "data" / "imdb_reviews.csv"
DEFAULT_OUTPUT_PATH = REPO_DIR / "imdb_sentiment_dataset.csv"

LABEL_MAP = {
    "negative": 0,
    "positive": 1,
    "0": 0,
    "1": 1,
}


def normalize_labels(labels: pd.Series) -> pd.Series:
    """Return IMDb sentiment labels as integer values 0 and 1."""
    normalized = labels.astype(str).str.strip().str.lower().map(LABEL_MAP)
    if normalized.isna().any():
        invalid = sorted(labels[normalized.isna()].astype(str).unique())
        raise ValueError(
            "Unsupported sentiment label(s): "
            f"{invalid}. Expected 'negative'/'positive' or 0/1."
        )
    return normalized.astype("int64")


def preprocess_dataset(
    input_path: Path,
    output_path: Path,
    test_size: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Read, validate, split, and save an IMDb dataset."""
    dataframe = pd.read_csv(input_path)

    # Accept both the downloaded Kaggle names and the model-ready names.  Only
    # these two source fields are retained; columns such as file_name are not
    # consumed by any model in this repository.
    text_column = "review" if "review" in dataframe.columns else "text"
    label_column = "sentiment" if "sentiment" in dataframe.columns else "label"
    missing = [
        name
        for name, present in (
            ("review (or text)", text_column in dataframe.columns),
            ("sentiment (or label)", label_column in dataframe.columns),
        )
        if not present
    ]
    if missing:
        raise ValueError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Found: {list(dataframe.columns)}"
        )

    prepared = pd.DataFrame(
        {
            "text": dataframe[text_column],
            "label": normalize_labels(dataframe[label_column]),
        }
    )
    if prepared["text"].isna().any():
        raise ValueError("The input dataset contains missing review text.")
    prepared["text"] = prepared["text"].astype(str)

    train_df, test_df = train_test_split(
        prepared,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=prepared["label"],
    )
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    test_df["split"] = "test"

    result = pd.concat([train_df, test_df], ignore_index=True)
    result = result[["text", "label", "split"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the IMDb CSV for the BiLSTM, RoBERTa, and VADER models."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"source CSV (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"destination CSV (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.5,
        help="fraction assigned to the test split (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed used for the split (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = preprocess_dataset(args.input, args.output, args.test_size, args.seed)

    counts = result.groupby(["split", "label"]).size().unstack(fill_value=0)
    print(f"Saved {len(result):,} rows to '{args.output}'.")
    print("Columns: text, label, split")
    print(counts.rename(columns={0: "negative", 1: "positive"}).to_string())


if __name__ == "__main__":
    main()
