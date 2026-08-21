"""Training entry point for the BiLSTM sentiment classifier."""

import json
from pathlib import Path

import torch
import torch.nn as nn

try:
    from .util.data import create_dataloader, create_embedding_vectors, get_lengths, get_tokenizer, load_data
    from .util.model import BiLSTM
    from .util.vokab import PAD_TOKEN, create_vocabulary, save_vocab
except ImportError:  # Allows: python BiLSTM/train.py
    from util.data import create_dataloader, create_embedding_vectors, get_lengths, get_tokenizer, load_data
    from util.model import BiLSTM
    from util.vokab import PAD_TOKEN, create_vocabulary, save_vocab


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parent
DATA_PATH = REPO_DIR / "imdb_sentiment_dataset.csv"
MODEL_DIR = PACKAGE_DIR / "trained_models"
CHECKPOINT_PATH = MODEL_DIR / "best_model.ckpt"
FINAL_MODEL_PATH = MODEL_DIR / "final_model.pt"
VOCAB_PATH = MODEL_DIR / "vocab.pkl"
META_PATH = MODEL_DIR / "meta.json"

BATCH_SIZE = 32
EMBEDDING_DIM = 100
HIDDEN_DIM = 200
NUM_LAYERS = 1
NUM_EPOCHS = 10
MAX_SEQ_LEN = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, pad_value=0):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_val_loss = float("inf")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        for step, (input_ids, labels) in enumerate(train_loader, start=1):
            lengths = get_lengths(input_ids, pad_value)
            outputs, _ = model(input_ids.to(DEVICE), lengths)
            loss = criterion(outputs.view(-1), labels.to(DEVICE).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{step}/{len(train_loader)}], Train Loss: {loss.item():.8f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_ids, labels in val_loader:
                lengths = get_lengths(input_ids, pad_value)
                outputs, _ = model(input_ids.to(DEVICE), lengths)
                val_loss += criterion(outputs.view(-1), labels.to(DEVICE).float()).item()
        val_loss /= len(val_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"Model checkpoint saved to '{CHECKPOINT_PATH}'.")

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    return model


def save_meta(vocab_size, pad_value):
    metadata = {
        "vocab_size": vocab_size,
        "pad_value": pad_value,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "max_seq_len": MAX_SEQ_LEN,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def train_and_save(data_path=DATA_PATH):
    train_df, val_df, _ = load_data(data_path)
    tokenizer = get_tokenizer()
    vocab = create_vocabulary(train_df, tokenizer)
    pad_value = vocab[PAD_TOKEN]
    vectors = create_embedding_vectors(vocab, EMBEDDING_DIM)
    train_loader = create_dataloader(train_df, tokenizer, vocab, BATCH_SIZE, MAX_SEQ_LEN, pad_value, True)
    val_loader = create_dataloader(val_df, tokenizer, vocab, BATCH_SIZE, MAX_SEQ_LEN, pad_value, False)

    model = BiLSTM(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, pad_value=pad_value)
    model.update_embedding(vectors)
    model.to(DEVICE)
    train_model(model, train_loader, val_loader, pad_value=pad_value)
    save_vocab(vocab, VOCAB_PATH)
    save_meta(len(vocab), pad_value)
    print(f"Training complete. Artifacts saved in '{MODEL_DIR}'.")
    return model


if __name__ == "__main__":
    train_and_save()
