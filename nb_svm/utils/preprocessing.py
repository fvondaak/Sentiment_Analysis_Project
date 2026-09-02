
# Script to preprocess the raw IMDB dataset. Texts are transformed into a unigram-bigram sparse matricies, and stored as npz in the output directory. Labels are stored as npy.
# The vocabulary is built from the training data only, and stored as a pickle file. Metadata about the preprocessing is stored as a JSON file.

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.model_selection import train_test_split

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from common.tokenizer import get_tokenizer
from nb_svm.utils.vocab import NBSVMVocabulary


DEFAULT_INPUT_PATH = REPO_DIR / "common_dataset" / "IMDB_Dataset.csv"
DEFAULT_OUTPUT_DIR = REPO_DIR / "nb_svm" / "data_preprocessed"
LABEL_MAP = {"negative": 0, "positive": 1}


def load_dataset(input_path):
    """Load and validate the raw IMDb dataset."""
    dataframe = pd.read_csv(input_path)
    required_columns = {"review", "sentiment"}
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing}")
    if dataframe[["review", "sentiment"]].isna().any().any():
        raise ValueError("The dataset contains missing reviews or labels.")

    labels = dataframe["sentiment"].astype(str).str.strip().str.lower()
    unsupported_labels = sorted(set(labels).difference(LABEL_MAP))
    if unsupported_labels:
        raise ValueError(f"Unsupported sentiment label(s): {unsupported_labels}")

    return pd.DataFrame(
        {
            "text": dataframe["review"].astype(str),  # all reviews as strings
            "label": labels.map(LABEL_MAP).astype(np.int8),  # 1 corresponds to positive, 0 to negative
        }
    )


def split_dataset(dataframe, test_size=0.5, seed=42):
    """Create a deterministic stratified train/test split."""
    train_df, test_df = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=dataframe["label"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def transform_texts(texts, vocabulary, tokenizer):
    """Transform texts into a binary unigram-bigram CSR matrix. CSR matrix as most of the features will be sparse."""
    indices = []
    indptr = [0]

    for text in texts:
        tokens = tokenizer(str(text))
        feature_indices = {
            index
            for token in tokens
            if (index := vocabulary.lookup_token(token)) is not None
        }
        feature_indices.update(
            index
            for bigram in zip(tokens, tokens[1:])
            if (index := vocabulary.lookup_bigram(bigram)) is not None
        )
        indices.extend(sorted(feature_indices))
        indptr.append(len(indices))

    data = np.ones(len(indices), dtype=np.int8)
    return csr_matrix(
        (
            data,
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=(len(indptr) - 1, len(vocabulary)),
        dtype=np.int8,
    )


def save_preprocessed_data(
    output_dir,
    X_train,
    X_test,
    y_train,
    y_test,
    vocabulary,
    metadata,
):
    """Save matrices, labels, vocabulary, and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_npz(output_dir / "X_train.npz", X_train)
    save_npz(output_dir / "X_test.npz", X_test)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_test.npy", y_test)

    with (output_dir / "vocabulary.pkl").open("wb") as file:
        pickle.dump(vocabulary, file)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def preprocess(input_path, output_dir, test_size=0.5, seed=42):
    """Run the complete NB-SVM preprocessing pipeline."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    dataframe = load_dataset(input_path)
    train_df, test_df = split_dataset(dataframe, test_size, seed)  # 50/50 train/test split (by default)
    tokenizer = get_tokenizer()
    vocabulary = NBSVMVocabulary.from_dataframe(train_df, tokenizer)  # Build vocab from training data only

    X_train = transform_texts(train_df["text"], vocabulary, tokenizer)  # Transform texts into sparse matrices, are not binary yet!!!
    X_test = transform_texts(test_df["text"], vocabulary, tokenizer)
    y_train = train_df["label"].to_numpy(dtype=np.int8)
    y_test = test_df["label"].to_numpy(dtype=np.int8)

    metadata = {
        "source_path": str(input_path.resolve()),
        "split_method": "deterministic stratified random split",
        "official_split_available": False,
        "seed": seed,
        "test_size": test_size,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "unigram_features": len(vocabulary.token_to_idx),
        "bigram_features": len(vocabulary.bigram_to_idx),
        "total_features": len(vocabulary),
        "labels": {"negative": 0, "positive": 1},
    }

    save_preprocessed_data(
        output_dir,
        X_train,
        X_test,
        y_train,
        y_test,
        vocabulary,
        metadata,
    )
    return metadata


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    args.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args.add_argument("--test-size", type=float, default=0.5)
    args.add_argument("--seed", type=int, default=42)
    args = args.parse_args()

    metadata = preprocess(args.input, args.output_dir, args.test_size, args.seed)
    print(json.dumps(metadata, indent=2))
