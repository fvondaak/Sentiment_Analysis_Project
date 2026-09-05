# Script to evaluate nb_svm model

import argparse
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import scipy.sparse as sp
from sklearn.metrics import accuracy_score

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

from nb_svm.utils.preprocessing import load_dataset
from nb_svm.nb_svm_train import binarize_features


NB_SVM_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = NB_SVM_DIR / "data_preprocessed"
DEFAULT_MODEL_PATH = NB_SVM_DIR / "trained_models" / "nb_svm.pkl"
DEFAULT_RESULTS_PATH = NB_SVM_DIR / "nb_svm_results.csv"


def load_model(model_path):
    """Load a classifier and its NB log-count ratio."""
    with Path(model_path).open("rb") as file:
        model = pickle.load(file)
    if not isinstance(model, dict) or not {
        "classifier",
        "log_count_ratio",
    }.issubset(model):
        raise ValueError("Model artifact does not contain classifier and log_count_ratio.")
    return model["classifier"], np.asarray(model["log_count_ratio"])


def load_test_dataframe(data_dir, y_test):
    """Load the official test dataframe recorded during preprocessing."""
    data_dir = Path(data_dir)
    with (data_dir / "metadata.json").open(encoding="utf-8") as file:
        metadata = json.load(file)

    test_df = load_dataset(metadata["test_source_path"])
    if not np.array_equal(test_df["label"].to_numpy(), y_test):
        raise ValueError("Official test labels do not match y_test.npy.")
    return test_df



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Naive Bayes SVM model.")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing preprocessed data.")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the trained model file.")
    parser.add_argument("--results_path", type=Path, default=DEFAULT_RESULTS_PATH, help="Path to save the evaluation results CSV.")
    args = parser.parse_args()

    # Load preprocessed data
    print(f"Loading preprocessed data from {args.data_dir}...")
    X_test = sp.load_npz(Path(args.data_dir) / "X_test.npz")
    y_test = np.load(Path(args.data_dir) / "y_test.npy")

    # Load the trained model
    print(f"Loading trained model from {args.model_path}...")
    classifier, log_count_ratio = load_model(args.model_path)

    if X_test.shape[1] != len(log_count_ratio):
        raise ValueError("Test feature count does not match the saved log-count ratio.")
    X_test_weighted = binarize_features(X_test).multiply(log_count_ratio).tocsr()

    # Make predictions on the test set
    print("Making predictions on the test set...")
    y_pred = classifier.predict(X_test_weighted)

    test_df = load_test_dataframe(args.data_dir, y_test)
    results = pd.DataFrame(
        {
            "text": test_df["text"].to_numpy(),
            "label": test_df["label"].to_numpy(),
            "prediction": np.asarray(y_pred, dtype=np.int8),
        }
    )

    print(f"Saving evaluation results to {args.results_path}...")
    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
