"""Evaluate the trained BiLSTM model on the IMDb test split."""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from common.tokenizer import get_tokenizer
from .utils.data import DEFAULT_DATA_PATH, create_dataloader, get_length, load_data
from .utils.model import BiLSTM
from .utils.vocab import PAD_TOKEN, load_vocab


LSTM_DIR = Path(__file__).resolve().parent
VOCAB_PATH = LSTM_DIR / "vocab" / "vocab.pkl"
MODEL_PATH = LSTM_DIR / "trained_models" / "BiLSTM.pt"
RESULTS_PATH = LSTM_DIR / "lstm_results.csv"

BATCH_SIZE = 64
EMBEDDING_DIM = 100
HIDDEN_DIM = 200
NUM_LAYERS = 1
NUM_CLASSES = 1
MAX_SEQ_LEN = 400


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the trained BiLSTM model on the test dataset."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size used during evaluation.",
    )
    return parser.parse_args()


def evaluate_test_set(model, test_dataloader, loss_function, device):
    """Return predictions, average loss, and accuracy for the test set."""
    model.eval()
    predictions = []
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for input_ids, labels in test_dataloader:
            lengths = get_length(input_ids, model.pad_value)
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            outputs, _ = model(input_ids, lengths)
            logits = outputs.view(-1)
            loss = loss_function(logits, labels.float())
            batch_predictions = (torch.sigmoid(logits) >= 0.5).long()

            batch_size = labels.numel()
            total_loss += loss.item() * batch_size
            total_correct += (batch_predictions == labels).sum().item()
            total_samples += batch_size
            predictions.extend(batch_predictions.cpu().tolist())

    if total_samples == 0:
        raise ValueError("Test dataloader must contain at least one sample.")

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return predictions, average_loss, accuracy


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vocab = load_vocab(VOCAB_PATH)
    pad_value = vocab[PAD_TOKEN]
    tokenizer = get_tokenizer()
    _, _, test_df = load_data(DEFAULT_DATA_PATH)

    test_dataloader = create_dataloader(
        test_df,
        tokenizer,
        vocab,
        args.batch_size,
        max_seq_len=MAX_SEQ_LEN,
        pad_value=pad_value,
        shuffle=False,
    )

    model = BiLSTM(
        len(vocab),
        EMBEDDING_DIM,
        HIDDEN_DIM,
        NUM_LAYERS,
        NUM_CLASSES,
        pad_value,
    )
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)

    loss_function = nn.BCEWithLogitsLoss()
    predictions, average_loss, accuracy = evaluate_test_set(
        model,
        test_dataloader,
        loss_function,
        device,
    )

    results = pd.DataFrame(
        {
            "text": test_df["text"].values,
            "label": test_df["label"].values,
            "prediction": predictions,
        }
    )
    results["correct"] = (results["label"] == results["prediction"]).astype(int)
    results.to_csv(RESULTS_PATH, index=False)

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Average test loss: {average_loss:.4f}")
    print(f"Saved {len(results)} predictions to '{RESULTS_PATH}'.")

