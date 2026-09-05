"""Plot the BiLSTM training history."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LSTM_DIR = Path(__file__).resolve().parent
HISTORY_PATH = LSTM_DIR / "training_history.csv"
PLOT_PATH = LSTM_DIR / "lstm_train_plot.png"
REQUIRED_COLUMNS = {"epoch", "train_loss", "val_loss", "val_accuracy"}


if __name__ == "__main__":
    history = pd.read_csv(HISTORY_PATH)
    missing_columns = REQUIRED_COLUMNS.difference(history.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Training history is missing columns: {missing}")
    if history.empty:
        raise ValueError("Training history does not contain any epochs.")

    figure, loss_axis = plt.subplots(figsize=(10, 6))
    accuracy_axis = loss_axis.twinx()

    train_loss_line = loss_axis.plot(
        history["epoch"],
        history["train_loss"],
        color="tab:blue",
        linewidth=2,
        label="Training loss",
    )[0]
    val_loss_line = loss_axis.plot(
        history["epoch"],
        history["val_loss"],
        color="tab:orange",
        linewidth=2,
        label="Validation loss",
    )[0]
    val_accuracy_line = accuracy_axis.plot(
        history["epoch"],
        history["val_accuracy"],
        color="tab:green",
        linewidth=2,
        label="Validation accuracy",
    )[0]

    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    accuracy_axis.set_ylabel("Validation accuracy")
    accuracy_axis.set_ylim(0, 1)
    loss_axis.set_title("BiLSTM Training History")
    loss_axis.grid(alpha=0.25)

    lines = [train_loss_line, val_loss_line, val_accuracy_line]
    loss_axis.legend(lines, [line.get_label() for line in lines], loc="best")

    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved training plot to '{PLOT_PATH}'.")
