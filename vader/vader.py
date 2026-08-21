"""
VADER Sentiment-Analyse für den IMDb-Datensatz.

Dieses Skript liest den Test-Split des Datensatzes ein, wertet ihn mit
VADER (Valence Aware Dictionary and sEntiment Reasoner) aus und speichert
die Ergebnisse als CSV, damit sie später von evaluation.py eingelesen und
mit den anderen Modellen verglichen werden können.
"""

import numpy as np
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# ==========================================
# 1. KONFIGURATION
# ==========================================
DATA_PATH = "imdb_sentiment_dataset.csv"
RESULTS_PATH = "vader_results.csv"
NUM_EVAL_SAMPLES = 1000


# ==========================================
# 2. RESSOURCEN LADEN
# ==========================================
def _ensure_lexicon():
    """Lädt das VADER-Lexikon herunter, falls es noch nicht vorhanden ist."""
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")


def load_resources():
    """Initialisiert den VADER-Analyzer."""
    _ensure_lexicon()
    sia = SentimentIntensityAnalyzer()
    return {"sia": sia}


# ==========================================
# 3. VORHERSAGE-FUNKTIONEN
# ==========================================
def predict_single_text(text, resources):
    """Wertet einen einzelnen Text aus.
    Gibt (label, wahrscheinlichkeit_positiv) zurück, label ist 0 oder 1.
    """
    sia = resources["sia"]
    compound = sia.polarity_scores(str(text))["compound"]

    # VADER liefert einen "compound"-Score zwischen -1 (sehr negativ) und
    # +1 (sehr positiv). Wir bilden daraus eine Pseudo-Wahrscheinlichkeit
    # zwischen 0 und 1, damit sie mit den anderen Modellen vergleichbar ist.
    probability_positive = (compound + 1.0) / 2.0
    label = int(compound >= 0)
    return label, probability_positive


def predict_batch(texts, resources):
    """Wertet eine Liste von Texten aus."""
    preds, probs = [], []
    for text in texts:
        label, prob = predict_single_text(text, resources)
        preds.append(label)
        probs.append(prob)
    return np.array(preds), np.array(probs)


# ==========================================
# 4. AUSWERTUNG AUF DEM TESTSET
# ==========================================
def run_evaluation(resources=None, num_samples=NUM_EVAL_SAMPLES, save_path=RESULTS_PATH):
    """Wertet VADER auf `num_samples` Beispielen aus dem Test-Split aus und
    speichert das Ergebnis als CSV."""
    if resources is None:
        resources = load_resources()

    df = pd.read_csv(DATA_PATH)
    test_df = df[df["split"] == "test"].reset_index(drop=True).head(num_samples)

    preds, probs = predict_batch(test_df["text"].values, resources)

    result_df = pd.DataFrame({
        "text": test_df["text"].values,
        "label": test_df["label"].values,
        "prediction": preds,
        "probability_positive": probs,
    })
    result_df["correct"] = (result_df["label"] == result_df["prediction"]).astype(int)

    result_df.to_csv(save_path, index=False)
    print(f"VADER: Ergebnisse für {len(result_df)} Beispiele gespeichert unter '{save_path}'.")
    return result_df


if __name__ == "__main__":
    resources = load_resources()
    run_evaluation(resources)