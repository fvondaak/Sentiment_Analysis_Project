
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

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from common.tokenizer import get_tokenizer
from nb_svm.utils.vocab import NBSVMVocabulary


DEFAULT_TRAIN_INPUT_PATH = REPO_DIR / "dataset" / "train.csv"
DEFAULT_TEST_INPUT_PATH = REPO_DIR / "dataset" / "test.csv"
DEFAULT_OUTPUT_DIR = REPO_DIR / "nb_svm" / "data_preprocessed"


def load_dataset(input_path):
    """Load and validate an official IMDb dataset split."""
    dataframe = pd.read_csv(input_path)
    required_columns = {"text", "label"}
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing}")
    if dataframe[["text", "label"]].isna().any().any():
        raise ValueError("The dataset contains missing texts or labels.")
    if not dataframe["label"].isin([0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1.")

    dataframe["text"] = dataframe["text"].astype(str)
    dataframe["label"] = dataframe["label"].astype(np.int8)
    return dataframe.reset_index(drop=True)


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


def preprocess(train_input_path, test_input_path, output_dir):
    """Run the complete NB-SVM preprocessing pipeline."""
    train_input_path = Path(train_input_path)
    test_input_path = Path(test_input_path)
    output_dir = Path(output_dir)
    train_df = load_dataset(train_input_path)
    test_df = load_dataset(test_input_path)
    tokenizer = get_tokenizer()
    vocabulary = NBSVMVocabulary.from_dataframe(train_df, tokenizer)  # Build vocab from training data only

    X_train = transform_texts(train_df["text"], vocabulary, tokenizer)  # Transform texts into sparse matrices
    X_test = transform_texts(test_df["text"], vocabulary, tokenizer)
    y_train = train_df["label"].to_numpy(dtype=np.int8)
    y_test = test_df["label"].to_numpy(dtype=np.int8)

    metadata = {
        "train_source_path": str(train_input_path.resolve()),
        "test_source_path": str(test_input_path.resolve()),
        "split_method": "official IMDb train/test split",
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


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument(
        "--train-input",
        type=Path,
        default=DEFAULT_TRAIN_INPUT_PATH,
    )
    args.add_argument(
        "--test-input",
        type=Path,
        default=DEFAULT_TEST_INPUT_PATH,
    )
    args.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = args.parse_args()

    metadata = preprocess(args.train_input, args.test_input, args.output_dir)
    print(json.dumps(metadata, indent=2))
