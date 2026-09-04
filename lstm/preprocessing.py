"""Convert the raw IMDb CSV into the format expected by the LSTM."""

from pathlib import Path
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "dataset" / "IMDB_Dataset.csv"
OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "dataset_preprocessed"
    / "imdb_sentiment_dataset.csv"
)

LABEL_MAP = {"negative": 0, "positive": 1}
TEST_SIZE = 0.5
SEED = 42


def preprocess_dataset(
    input_path=INPUT_PATH,
    output_path=OUTPUT_PATH,
    test_size=TEST_SIZE,
    seed=SEED,
):
    dataframe = pd.read_csv(input_path)
    dataframe = dataframe.rename(columns={"review": "text", "sentiment": "label"})
    dataframe["label"] = dataframe["label"].map(LABEL_MAP)

    train_df, test_df = train_test_split(
        dataframe[["text", "label"]],
        test_size=test_size,
        random_state=seed,
        stratify=dataframe["label"],
    )

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    test_df["split"] = "test"

    output_df = pd.concat([train_df, test_df], ignore_index=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"Saved {len(output_df)} rows to {output_path}")
    return output_df


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Preprocess the IMDb dataset.")
    args.add_argument("--input_path", type=str, default=INPUT_PATH, help="Path to the input CSV file."
    )
    args.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility.")
    args.add_argument("--split", type=float, default=TEST_SIZE, help="Proportion of the dataset to include in the test split.")

    args = args.parse_args()
    preprocess_dataset(
        input_path=args.input_path,
        output_path=OUTPUT_PATH,
        test_size=TEST_SIZE,
        seed=args.seed,
    )
