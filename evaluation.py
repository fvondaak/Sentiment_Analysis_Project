"""Compare NB-SVM, RoBERTa, and BiLSTM results on the IMDb test set."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parent
PLOTS_DIRECTORY = PROJECT_ROOT / "plots"
RESULT_PATHS = {
    "NB-SVM": PROJECT_ROOT / "nb_svm" / "nb_svm_results.csv",
    "RoBERTa": PROJECT_ROOT / "roberta" / "roberta_results.csv",
    "BiLSTM": PROJECT_ROOT / "lstm" / "lstm_results.csv",
}
ACCURACY_PLOT_PATH = PLOTS_DIRECTORY / "model_comparison.png"
CONFUSION_PLOT_PATH = PLOTS_DIRECTORY / "model_confusion_matrices.png"

#length analysis
SUMMARY_PATH = PLOTS_DIRECTORY / "accuracy_by_review_length.csv"
PLOT_PATH = PLOTS_DIRECTORY / "accuracy_by_review_length.png"

LENGTH_BINS = [0, 100, 200, 300, 400, float("inf")]
LENGTH_LABELS = [
    "1-100 words",
    "101-200 words",
    "201-300 words",
    "301-400 words",
    "More than 400 words",
]



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


#ĺength analysis helper functions
def load_result_dataframe(result_path):
    """Load and validate one model's result dataframe."""
    dataframe = pd.read_csv(result_path)
    required_columns = {"text", "label", "prediction"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing}")
    if dataframe.empty:
        raise ValueError(f"Results file is empty: '{result_path}'.")

    dataframe = dataframe.copy()
    dataframe["word_count"] = dataframe["text"].fillna("").str.split().str.len()
    dataframe["length_bucket"] = pd.cut(
        dataframe["word_count"],
        bins=LENGTH_BINS,
        labels=LENGTH_LABELS,
        include_lowest=True,
    )
    return dataframe


def build_accuracy_dataframe(result_paths=RESULT_PATHS):
    """Build one accuracy summary row for every model and length bucket."""
    summaries = []
    for model_name, result_path in result_paths.items():
        dataframe = load_result_dataframe(result_path)
        grouped = dataframe.groupby("length_bucket", observed=False)
        summary = grouped.apply(
            lambda group: pd.Series(
                {
                    "samples": len(group),
                    "accuracy": (group["label"] == group["prediction"]).mean(),
                }
            ),
            include_groups=False,
        ).reset_index()
        summary.insert(0, "model", model_name)
        summaries.append(summary)

    result = pd.concat(summaries, ignore_index=True)
    result["length_bucket"] = pd.Categorical(
        result["length_bucket"], categories=LENGTH_LABELS, ordered=True
    )
    return result.sort_values(["length_bucket", "model"]).reset_index(drop=True)


def plot_accuracy_by_length(summary, output_path=PLOT_PATH):
    """Save grouped accuracy bars for all models and length buckets."""
    plot_data = summary.pivot(
        index="length_bucket", columns="model", values="accuracy"
    ).reindex(LENGTH_LABELS)

    axis = plot_data.plot(
        kind="bar",
        figsize=(14, 7),
        width=0.85,
        color={"NB-SVM": "#3498db", "BiLSTM": "#2ecc71", "RoBERTa": "#e67e22"},
    )
    axis.set_xlabel("Review length")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0, 1.05)
    axis.set_title("Model accuracy by review length")
    axis.legend(title="Model")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    axis.set_xticklabels(LENGTH_LABELS, rotation=20, ha="right")
    for bars in axis.containers:
        axis.bar_label(
            bars,
            labels=[f"{bar.get_height():.2%}" for bar in bars],
            padding=3,
            fontsize=8,
        )
    axis.figure.tight_layout()
    axis.figure.savefig(output_path, dpi=200)
    plt.close(axis.figure)

def main():
    PLOTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    all_results = load_results()
    metrics = compute_metrics(all_results)

    print(f"Evaluating {len(next(iter(all_results.values())))} test samples...")
    for model_name, model_metrics in metrics.items():
        print(f"{model_name} test accuracy: {model_metrics['accuracy']:.4f}")

    plot_accuracies(metrics)
    plot_confusion_matrices(metrics)
    print(f"Accuracy plot saved to '{ACCURACY_PLOT_PATH}'.")
    print(f"Confusion matrices saved to '{CONFUSION_PLOT_PATH}'.")

    summary = build_accuracy_dataframe()
    summary.to_csv(SUMMARY_PATH, index=False)
    plot_accuracy_by_length(summary)

    print(summary.to_string(index=False))
    print(f"Summary saved to '{SUMMARY_PATH}'.")
    print(f"Plot saved to '{PLOT_PATH}'.")    


if __name__ == "__main__":
    main()
