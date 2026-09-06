"""Train and evaluate an NB-weighted linear SVM."""

# C = 1, alpha = 1.0 from paper, might optimize this with a hyperparameter search later,
# but with 91,5 % accuracy on test set and some trials with different parameters seems quite optimal


import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
from sklearn.svm import LinearSVC

import pickle

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data_preprocessed"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "trained_models" / "nb_svm.pkl"


def load_preprocessed_data(data_dir):
    """Load sparse features and labels produced by preprocessing."""
    data_dir = Path(data_dir)
    X_train = load_npz(data_dir / "X_train.npz").tocsr()
    X_val = load_npz(data_dir / "X_val.npz").tocsr()
    X_test = load_npz(data_dir / "X_test.npz").tocsr()
    y_train = np.load(data_dir / "y_train.npy")
    y_val = np.load(data_dir / "y_val.npy")
    y_test = np.load(data_dir / "y_test.npy")
    return X_train, X_val, X_test, y_train, y_val, y_test


def binarize_features(features):
    """Return a binary CSR copy of a sparse feature matrix."""
    binary = features.tocsr(copy=True)
    binary.sum_duplicates()
    binary.eliminate_zeros()
    binary.data = np.ones(binary.nnz, dtype=np.int8)
    return binary


def compute_nb_weighted_features(X_train, X_test, y_train, alpha=1.0):
    """Compute the training log-count ratio and weight both feature matrices."""
    if alpha <= 0:
        raise ValueError("alpha must be greater than zero.")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train and test matrices must have the same feature count.")
    if X_train.shape[0] != len(y_train):
        raise ValueError("Training features and labels must have the same length.")

    labels = np.asarray(y_train)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Training labels must contain only 0 and 1.")
    if not np.any(labels == 0) or not np.any(labels == 1):
        raise ValueError("Training labels must contain both classes.")

    X_train_binary = binarize_features(X_train)
    X_test_binary = binarize_features(X_test)

    positive_counts = alpha + np.asarray(
        X_train_binary[labels == 1].sum(axis=0)
    ).ravel()
    negative_counts = alpha + np.asarray(
        X_train_binary[labels == 0].sum(axis=0)
    ).ravel()

    positive_probabilities = positive_counts / positive_counts.sum()
    negative_probabilities = negative_counts / negative_counts.sum()
    log_count_ratio = np.log(positive_probabilities / negative_probabilities)

    X_train_weighted = X_train_binary.multiply(log_count_ratio).tocsr()
    X_test_weighted = X_test_binary.multiply(log_count_ratio).tocsr()
    return X_train_weighted, X_test_weighted, log_count_ratio


def train_classifier(X_train, y_train, C=1.0):
    """Fit and return a linear SVM classifier."""
    classifier = LinearSVC(
        C=C,
        penalty="l2",
        loss="squared_hinge",
        dual=True,
        max_iter=10_000,
    )
    print("training classifier...")
    classifier.fit(X_train, y_train)
    print("classifier trained")
    return classifier


def grid_search(X_train, X_val, y_train, y_val, alphas, Cs):
    """Select alpha and C using validation accuracy."""
    best = None
    for params in ParameterGrid({"alpha": alphas, "C": Cs}):
        X_train_weighted, X_val_weighted, _ = compute_nb_weighted_features(
            X_train, X_val, y_train, params["alpha"]
        )
        classifier = train_classifier(X_train_weighted, y_train, params["C"])
        accuracy = accuracy_score(y_val, classifier.predict(X_val_weighted))
        print(
            f"alpha={params['alpha']}, C={params['C']}, "
            f"validation accuracy={accuracy:.4f}"
        )
        if best is None or accuracy > best["validation_accuracy"]:
            best = {
                **params,
                "validation_accuracy": accuracy,
            }
    return best


def save_model(classifier, log_count_ratio, model_path):
    """Save the classifier and its NB log-count ratio."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as file:
        pickle.dump(
            {
                "classifier": classifier,
                "log_count_ratio": log_count_ratio,
            },
            file,
        )



if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4])
    args.add_argument("--Cs", nargs="+", type=float, default=[0.05, 0.1, 0.15 ,0.2])
    args = args.parse_args()

    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data(args.data_dir)
    best = grid_search(X_train, X_val, y_train, y_val, args.alphas, args.Cs)
    X_train_weighted, X_test_weighted, log_count_ratio = (
        compute_nb_weighted_features(X_train, X_test, y_train, best["alpha"])
    )
    classifier = train_classifier(X_train_weighted, y_train, best["C"])
    save_model(classifier, log_count_ratio, args.model_path)
    X_val_weighted = binarize_features(X_val).multiply(log_count_ratio).tocsr()
    validation_accuracy = accuracy_score(y_val, classifier.predict(X_val_weighted))
    print(f"Best alpha: {best['alpha']}")
    print(f"Best C: {best['C']}")
    print(f"Validation Accuracy: {validation_accuracy:.4f}")
    print(f"model saved to {args.model_path}")
