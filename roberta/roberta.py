"""
RoBERTa Sentiment-Analyse für den IMDb-Datensatz.

Nutzt ein vortrainiertes RoBERTa-Modell (Standard: cardiffnlp/twitter-roberta-
base-sentiment-latest) zur Sentiment-Klassifikation. Liest den Test-Split ein,
wertet ihn aus und speichert die Ergebnisse als CSV für evaluation.py.
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

# ==========================================
# 1. KONFIGURATION
# ==========================================
REPO_DIR = Path(__file__).resolve().parent.parent
ROBERTA_DIR = Path(__file__).resolve().parent
DATA_PATH = REPO_DIR / "dataset" / "test.csv"
RESULTS_PATH = ROBERTA_DIR / "roberta_results.csv"
BATCH_SIZE = 16
MAX_LENGTH = 256

# Falls du ein lokal feingetuntes Modell hast, trage hier den Ordnerpfad ein.
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. RESSOURCEN LADEN
# ==========================================
def load_resources(model_name=MODEL_NAME):
    """Lädt Tokenizer und Modell einmalig."""
    print(f"Loading Tokenizer and model ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    return {"tokenizer": tokenizer, "model": model}


# ==========================================
# 3. VORHERSAGE-FUNKTIONEN
# ==========================================
def _predict_probs(texts, resources, batch_size=BATCH_SIZE, max_length=MAX_LENGTH):
    tokenizer = resources["tokenizer"]
    model = resources["model"]

    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = [str(t) for t in texts[i:i + batch_size]]
            encoding = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            logits = model(**encoding).logits
            probs = torch.softmax(logits, dim=1)

            if probs.shape[1] == 3:
                # 0=Negativ, 1=Neutral, 2=Positiv. Nur Negativ gegen Positiv
                # vergleichen, sonst wird die neutral-Masse als negativ gewertet.
                p_neg, p_pos = probs[:, 0], probs[:, 2]
                pos_probs = (p_pos / (p_neg + p_pos + 1e-9)).cpu().numpy()
            else:
                # 0=Negativ, 1=Positiv
                pos_probs = probs[:, 1].cpu().numpy()

            all_probs.extend(pos_probs)

    return np.array(all_probs)


def predict_single_text(text, resources):
    """Wertet einen einzelnen Text aus.
    Gibt (label, wahrscheinlichkeit_positiv) zurück, label ist 0 oder 1.
    """
    prob = _predict_probs([text], resources)[0]
    label = int(prob >= 0.5)
    return label, float(prob)


def predict_batch(texts, resources):
    """Wertet eine Liste von Texten aus."""
    probs = _predict_probs(list(texts), resources)
    preds = (probs >= 0.5).astype(int)
    return preds, probs


# ==========================================
# 4. AUSWERTUNG AUF DEM TESTSET
# ==========================================
def run_evaluation(resources=None, save_path=RESULTS_PATH):
    """Evaluate RoBERTa on the complete test split and save the results."""
    if resources is None:
        resources = load_resources()

    print(f"Loading data from {DATA_PATH}...")
    test_df = pd.read_csv(DATA_PATH)

    print(f"Starting RoBERTa evaluation on {len(test_df)} samples...")
    preds, probs = predict_batch(test_df["text"].values, resources)

    result_df = pd.DataFrame({
        "text": test_df["text"].values,
        "label": test_df["label"].values,
        "prediction": preds,
        "probability_positive": probs,
    })
    result_df["correct"] = (result_df["label"] == result_df["prediction"]).astype(int)

    result_df.to_csv(save_path, index=False)
    print(f"RoBERTa: Results for {len(result_df)} examples saved to '{save_path}'.")
    return result_df


if __name__ == "__main__":
    resources = load_resources()
    run_evaluation(resources)
