#!/usr/bin/env python3
"""Download the official IMDb train and test splits as CSV files."""

from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = PROJECT_ROOT / "dataset"


if __name__ == "__main__":
    dataset = load_dataset("stanfordnlp/imdb")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        output_path = OUTPUT_DIRECTORY / f"{split}.csv"
        dataset[split].to_csv(output_path, index=False)
        print(f"Saved {len(dataset[split])} rows to '{output_path}'.")
