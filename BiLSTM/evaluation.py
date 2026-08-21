"""Inference and test-set evaluation for the trained BiLSTM model."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from .train import DATA_PATH, DEVICE, FINAL_MODEL_PATH, META_PATH, VOCAB_PATH, train_and_save
    from .util.data import get_tokenizer
    from .util.model import BiLSTM
    from .util.vokab import BOS_TOKEN, EOS_TOKEN, load_vocab
except ImportError:  # Allows: python BiLSTM/evaluation.py
    from train import DATA_PATH, DEVICE, FINAL_MODEL_PATH, META_PATH, VOCAB_PATH, train_and_save
    from util.data import get_tokenizer
    from util.model import BiLSTM
    from util.vokab import BOS_TOKEN, EOS_TOKEN, load_vocab


RESULTS_PATH = Path(__file__).resolve().parent / "own_nn_results.csv"
NUM_EVAL_SAMPLES = 1000


def load_resources():
    required = (FINAL_MODEL_PATH, VOCAB_PATH, META_PATH)
    if not all(path.exists() for path in required):
        raise FileNotFoundError("No trained BiLSTM found. Run `python -m BiLSTM.train` first.")

    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    vocab = load_vocab(VOCAB_PATH)
    model = BiLSTM(
        metadata["vocab_size"], metadata["embedding_dim"], metadata["hidden_dim"],
        metadata["num_layers"], pad_value=metadata["pad_value"]
    )
    model.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()
    return {"model": model, "vocab": vocab, "tokenizer": get_tokenizer(), "meta": metadata}


def text_to_tensor(text, resources):
    vocab = resources["vocab"]
    tokens = resources["tokenizer"](str(text))
    indices = vocab.lookup_indices(tokens)[: resources["meta"]["max_seq_len"] - 2]
    return torch.tensor(
        [vocab[BOS_TOKEN], *indices, vocab[EOS_TOKEN]], dtype=torch.long
    ).unsqueeze(0)


def predict_single_text(text, resources):
    tensor = text_to_tensor(text, resources).to(DEVICE)
    with torch.no_grad():
        outputs, _ = resources["model"](tensor, [tensor.shape[1]])
        probability = torch.sigmoid(outputs.view(-1)).item()
    return int(probability >= 0.5), probability


def predict_batch(texts, resources):
    predictions, probabilities = zip(*(predict_single_text(text, resources) for text in texts))
    return np.asarray(predictions), np.asarray(probabilities)


def run_evaluation(resources=None, num_samples=NUM_EVAL_SAMPLES, save_path=RESULTS_PATH):
    if resources is None:
        try:
            resources = load_resources()
        except FileNotFoundError:
            train_and_save()
            resources = load_resources()

    dataframe = pd.read_csv(DATA_PATH)
    test_df = dataframe[dataframe["split"] == "test"].reset_index(drop=True).head(num_samples)
    predictions, probabilities = predict_batch(test_df["text"].values, resources)
    results = test_df[["text", "label"]].copy()
    results["prediction"] = predictions
    results["probability_positive"] = probabilities
    results["correct"] = (results["label"] == results["prediction"]).astype(int)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(save_path, index=False)
    print(f"Saved results for {len(results)} samples to '{save_path}'.")
    return results


if __name__ == "__main__":
    run_evaluation()
