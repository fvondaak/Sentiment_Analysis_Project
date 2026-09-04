"""
Eigenes neuronales Netz (BiLSTM + Self-Attention) für Sentiment-Analyse auf
dem IMDb-Datensatz.

Dieses Skript kann sowohl trainieren als auch (nach dem Training) für die
Auswertung und für die Live-Vorhersage einzelner Texte genutzt werden:

- Beim ersten Aufruf wird das Modell trainiert und unter own_nn_artifacts/
  gespeichert (Modell-Checkpoint, Vokabular, Meta-Informationen).
- Bei weiteren Aufrufen wird das bereits trainierte Modell geladen, sodass
  nicht jedes Mal neu trainiert werden muss.
- Die Auswertung auf dem Test-Split wird als CSV gespeichert, damit
  evaluation.py sie einlesen und mit den anderen Modellen vergleichen kann.
"""

import os
import re
import html
import json
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import gensim.downloader as gensim_api

from .utils.model import BiLSTM, SelfAttention
from common.tokenizer import get_tokenizer
from .utils.data import IMDBDataset, load_data, create_dataloader, get_length
from .utils.vocab import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    CustomVocabulary,
    create_vocabulary,
    load_vocab,
    save_vocab,
)

# ==========================================
# 1. KONFIGURATION
# ==========================================
DEFAULT_DATA_PATH = "imdb_sentiment_dataset.csv"
RESULTS_PATH = "own_nn_results.csv"
NUM_EVAL_SAMPLES = 1000

MODEL_DIR = "own_nn_artifacts"
CKPT_PATH = os.path.join(MODEL_DIR, "best_model.ckpt")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.pkl")
META_PATH = os.path.join(MODEL_DIR, "meta.json")

BATCH_SIZE = 64
EMBEDDING_DIM = 100
HIDDEN_DIM = 200
NUM_LAYERS = 1
NUM_EPOCHS = 20
MAX_SEQ_LEN = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_embedding_vectors(vocab, embedding_dim):
    glove_model = gensim_api.load(f"glove-wiki-gigaword-{embedding_dim}")
    vectors = torch.randn(len(vocab), embedding_dim) * 0.01  # Random initialization instead of zeros
    for idx, token in enumerate(vocab.get_itos()):
        if token in glove_model:
            vectors[idx] = torch.tensor(glove_model[token])
        if token == PAD_TOKEN:
            vectors[idx] = torch.zeros(embedding_dim)  # Ensure PAD_TOKEN has a zero vector
    return vectors

# ==========================================
# 6. TRAINING
# ==========================================
def train_model(model, train_loader, val_loader, num_epochs, pad_value=0, save_path=CKPT_PATH):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam([
    {
        "params": model.emb.parameters(),
        "lr": 0.0001,
    },
    {
        "params": model.lstm.parameters(),
        "lr": 0.001,
    },
    {
        "params": model.attention.parameters(),
        "lr": 0.001,
    },
    {
        "params": model.fc.parameters(),
        "lr": 0.001,
    },
])  # smaller learning rate for embedding layer
    total_step = len(train_loader)
    best_val_loss = float("inf")

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        for i, (input_ids, labels) in enumerate(train_loader):
            lengths = get_length(input_ids, pad_value)
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            outputs, _ = model(input_ids, lengths)
            loss = criterion(outputs.view(-1), labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Train Loss: {:.8f}".format(
                        epoch + 1, num_epochs, i + 1, total_step, loss.item()
                    )
                )

        model.eval()
        val_loss = 0
        for i, (input_ids, labels) in enumerate(val_loader):
            lengths = get_length(input_ids, pad_value)
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs, _ = model(input_ids, lengths)
            val_loss += criterion(outputs.view(-1), labels.float()).item()
        epoch_val_loss = val_loss / (i + 1)
        print("\nEpoch [{}/{}] Validation Loss: {:.4f}\n".format(epoch + 1, num_epochs, epoch_val_loss))

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), save_path)
            print("--> Modell verbessert und unter '{}' gespeichert.\n".format(save_path))

    model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device)
    return model


def save_meta(vocab_size, pad_value, path=META_PATH):
    meta = {
        "vocab_size": vocab_size,
        "pad_value": pad_value,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "max_seq_len": MAX_SEQ_LEN,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f)


def load_meta(path=META_PATH):
    with open(path) as f:
        return json.load(f)


def train_and_save():
    """Volle Trainings-Pipeline: Daten laden, Vokabular bauen, Modell
    trainieren und Modell + Vokabular + Meta-Infos speichern."""
    print("Own-NN: Lade Trainingsdaten und baue Vokabular auf...")
    train_df, val_df, test_df = load_data(DEFAULT_DATA_PATH)
    tokenizer = get_tokenizer()

    vocab = create_vocabulary(train_df, tokenizer)
    pad_value = vocab[PAD_TOKEN]
    vocab_size = len(vocab.get_itos())

    print("Own-NN: Lade GloVe-Embeddings (das kann beim ersten Mal etwas dauern)...")
    vectors = create_embedding_vectors(vocab, EMBEDDING_DIM)

    train_loader = create_dataloader(train_df, tokenizer, vocab, BATCH_SIZE,
                                      max_seq_len=MAX_SEQ_LEN, pad_value=pad_value, shuffle=True)
    val_loader = create_dataloader(val_df, tokenizer, vocab, BATCH_SIZE,
                                    max_seq_len=MAX_SEQ_LEN, pad_value=pad_value, shuffle=False)

    model = BiLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, 1, pad_value)
    model.update_embedding(vectors)  # embedding layer is initialized with GloVe vectors
    model.to(device)

    print("Own-NN: Starte Training...")
    model = train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS,
                         pad_value=pad_value, save_path=CKPT_PATH)

    save_vocab(vocab)
    save_meta(vocab_size, pad_value)
    print("Own-NN: Training abgeschlossen. Modell, Vokabular und Meta-Infos gespeichert.")


# ==========================================
# 7. INFERENZ (für Auswertung & eigene Texte)
# ==========================================
def load_resources():
    """Lädt Vokabular + trainiertes Modell für die Inferenz. Löst
    FileNotFoundError aus, falls noch nicht trainiert wurde."""
    if not (os.path.exists(CKPT_PATH) and os.path.exists(VOCAB_PATH) and os.path.exists(META_PATH)):
        raise FileNotFoundError(
            "Kein trainiertes Own-NN Modell gefunden. Bitte zuerst 'python own_nn.py' "
            "ausführen (oder train_and_save() aufrufen), um das Modell zu trainieren."
        )

    meta = load_meta()
    vocab = load_vocab()
    model = BiLSTM(meta["vocab_size"], meta["embedding_dim"], meta["hidden_dim"],
                   meta["num_layers"], 1, meta["pad_value"])
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.to(device)
    model.eval()
    tokenizer = get_tokenizer()

    return {"model": model, "vocab": vocab, "tokenizer": tokenizer, "meta": meta}


def _text_to_tensor(text, resources):
    tokenizer = resources["tokenizer"]
    vocab = resources["vocab"]
    meta = resources["meta"]

    bos_idx = vocab[BOS_TOKEN]
    eos_idx = vocab[EOS_TOKEN]

    tokens = tokenizer(str(text))
    indices = vocab.lookup_indices(tokens)
    indices = indices[: meta["max_seq_len"] - 2]
    formatted = [bos_idx] + indices + [eos_idx]

    return torch.tensor(formatted, dtype=torch.long).unsqueeze(0)


def predict_single_text(text, resources):
    """Wertet einen einzelnen Text aus.
    Gibt (label, wahrscheinlichkeit_positiv) zurück, label ist 0 oder 1.
    """
    model = resources["model"]
    tensor = _text_to_tensor(text, resources).to(device)
    length = [tensor.shape[1]]

    with torch.no_grad():
        outputs, _ = model(tensor, length)
        prob = torch.sigmoid(outputs.view(-1)).item()

    label = int(prob >= 0.5)
    return label, prob


def predict_batch(texts, resources):
    """Wertet eine Liste von Texten aus (Text für Text, da variable Länge)."""
    preds, probs = [], []
    for text in texts:
        label, prob = predict_single_text(text, resources)
        preds.append(label)
        probs.append(prob)
    return np.array(preds), np.array(probs)


# ==========================================
# 8. AUSWERTUNG AUF DEM TESTSET
# ==========================================
def run_evaluation(resources=None, num_samples=NUM_EVAL_SAMPLES, save_path=RESULTS_PATH):
    """Wertet Own-NN auf `num_samples` Beispielen aus dem Test-Split aus und
    speichert das Ergebnis als CSV. Trainiert das Modell automatisch, falls
    noch kein trainiertes Modell vorhanden ist."""
    if resources is None:
        try:
            resources = load_resources()
        except FileNotFoundError:
            print("Own-NN: Noch kein trainiertes Modell gefunden - starte Training...")
            train_and_save()
            resources = load_resources()

    df = pd.read_csv(DEFAULT_DATA_PATH)
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
    print(f"Own-NN: Ergebnisse für {len(result_df)} Beispiele gespeichert unter '{save_path}'.")
    return result_df


if __name__ == "__main__":
    print(f"device is {device}")
    if os.path.exists(CKPT_PATH) and os.path.exists(VOCAB_PATH) and os.path.exists(META_PATH):
        print("Own-NN: Bereits trainiertes Modell gefunden - Training wird übersprungen.")
        resources = load_resources()
    else:
        train_and_save()
        resources = load_resources()

    run_evaluation(resources)
