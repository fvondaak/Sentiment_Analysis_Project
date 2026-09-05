"""Compare NB-SVM, RoBERTa, and BiLSTM results on the IMDb test set."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parent
RESULT_PATHS = {
    "NB-SVM": PROJECT_ROOT / "nb_svm" / "nb_svm_results.csv",
    "RoBERTa": PROJECT_ROOT / "roberta" / "roberta_results.csv",
    "BiLSTM": PROJECT_ROOT / "lstm" / "lstm_results.csv",
}
ACCURACY_PLOT_PATH = PROJECT_ROOT / "model_comparison.png"
CONFUSION_PLOT_PATH = PROJECT_ROOT / "model_confusion_matrices.png"
REQUIRED_COLUMNS = {"text", "label", "prediction"}


def load_results(result_paths=RESULT_PATHS):
    """Load model results and verify that they use the same test examples."""
    all_results = {}

    for model_name, result_path in result_paths.items():
        if not result_path.exists():
            raise FileNotFoundError(
                f"Missing results for {model_name}: '{result_path}'. "
                "Run that model's evaluation script first."
            )

        dataframe = pd.read_csv(result_path)
        missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
        if dataframe.empty:
            raise ValueError(f"Results for {model_name} are empty.")

        all_results[model_name] = dataframe

    return all_results


def compute_metrics(all_results):
    """Calculate accuracy and confusion matrix for every model."""
    metrics = {}

    for model_name, dataframe in all_results.items():
        labels = dataframe["label"].to_numpy()
        predictions = dataframe["prediction"].to_numpy()
        metrics[model_name] = {
            "accuracy": accuracy_score(labels, predictions),
            "confusion_matrix": confusion_matrix(
                labels,
                predictions,
                labels=[0, 1],
            ),
        }

    return metrics


def plot_accuracies(metrics, output_path=ACCURACY_PLOT_PATH):
    """Save a bar chart containing each model's test accuracy."""
    model_names = list(metrics)
    accuracies = [metrics[name]["accuracy"] for name in model_names]

    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(
        model_names,
        accuracies,
        color=["#3498db", "#e67e22", "#2ecc71"],
    )
    axis.set_ylabel("Test accuracy")
    axis.set_ylim(0, 1)
    axis.set_title("IMDb sentiment model comparison")
    axis.bar_label(bars, labels=[f"{value:.2%}" for value in accuracies], padding=3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_confusion_matrices(metrics, output_path=CONFUSION_PLOT_PATH):
    """Save the three confusion matrices in one figure."""
    figure, axes = plt.subplots(1, len(metrics), figsize=(15, 4))

    for axis, (model_name, model_metrics) in zip(axes, metrics.items()):
        sns.heatmap(
            model_metrics["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            ax=axis,
        )
        axis.set_title(model_name)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main():
    all_results = load_results()
    metrics = compute_metrics(all_results)

    print(f"Evaluating {len(next(iter(all_results.values())))} test samples...")
    for model_name, model_metrics in metrics.items():
        print(f"{model_name} test accuracy: {model_metrics['accuracy']:.4f}")

    plot_accuracies(metrics)
    plot_confusion_matrices(metrics)
    print(f"Accuracy plot saved to '{ACCURACY_PLOT_PATH}'.")
    print(f"Confusion matrices saved to '{CONFUSION_PLOT_PATH}'.")


if __name__ == "__main__":
    main()
