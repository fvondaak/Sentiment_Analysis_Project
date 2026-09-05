"""Train a fresh BiLSTM sentiment model."""

import argparse
from pathlib import Path

import gensim.downloader as gensim_api
import torch
import torch.nn as nn

from common.tokenizer import get_tokenizer
from .utils.data import DEFAULT_DATA_PATH, create_dataloader, load_data
from .utils.model import BiLSTM
from .utils.training import evaluate_model, train_one_epoch
from .utils.vocab import PAD_TOKEN, create_vocabulary, load_vocab, save_vocab


LSTM_DIR = Path(__file__).resolve().parent
VOCAB_PATH = LSTM_DIR / "own_nn_artifacts" / "vocab.pkl"
MODEL_PATH = LSTM_DIR / "trained_models" / "BiLSTM.pt"

BATCH_SIZE = 64
EMBEDDING_DIM = 100
HIDDEN_DIM = 200
NUM_LAYERS = 1
NUM_CLASSES = 1
NUM_EPOCHS = 20
MAX_SEQ_LEN = 200


def parse_args():
    parser = argparse.ArgumentParser(description="Train a fresh BiLSTM model.")
    parser.add_argument(
        "--use-existing-vocab",
        action="store_true",
        help=f"Load the vocabulary from {VOCAB_PATH} instead of rebuilding it.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Number of epochs to train the model.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--lr_embedding",
        type=float,
        default=0.0001,
        help="Learning rate for the embedding layer.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 penalty) for the optimizer.",
    )
    return parser.parse_args()


def create_embedding_vectors(vocab, embedding_dim):
    """Create an embedding matrix initialized with pretrained GloVe vectors."""
    glove_model = gensim_api.load(f"glove-wiki-gigaword-{embedding_dim}")
    vectors = torch.randn(len(vocab), embedding_dim) * 0.01

    for index, token in enumerate(vocab.get_itos()):
        if token in glove_model:
            vectors[index] = torch.tensor(glove_model[token])
        if token == PAD_TOKEN:
            vectors[index] = torch.zeros(embedding_dim)

    return vectors


def train_model(model, train_dataloader, val_dataloader, num_epochs, save_path, args):
    """Train the model and save the state with the lowest validation loss."""
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.emb.parameters(), "lr": args.lr_embedding},
            {"params": model.lstm.parameters(), "lr": args.lr},
            {"params": model.attention.parameters(), "lr": args.lr},
            {"params": model.fc.parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    best_val_loss = float("inf")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        train_one_epoch(
            model,
            train_dataloader,
            loss_function,
            epoch,
            optimizer,
        )
        val_loss, _ = evaluate_model(
            model,
            val_dataloader,
            loss_function,
            epoch,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved improved model to '{save_path}'.")


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df, val_df, _ = load_data(DEFAULT_DATA_PATH)
    tokenizer = get_tokenizer()

    if args.use_existing_vocab:
        print(f"Loading vocabulary from '{VOCAB_PATH}'...")
        vocab = load_vocab(VOCAB_PATH)
    else:
        print("Building vocabulary from the training dataset...")
        vocab = create_vocabulary(train_df, tokenizer)
        save_vocab(vocab, VOCAB_PATH)
        print(f"Saved vocabulary to '{VOCAB_PATH}'.")

    pad_value = vocab[PAD_TOKEN]
    vocab_size = len(vocab)
 
    print("Loading GloVe embeddings...")
    embedding_vectors = create_embedding_vectors(vocab, EMBEDDING_DIM)

    train_dataloader = create_dataloader(
        train_df,
        tokenizer,
        vocab,
        args.batch_size,
        max_seq_len=MAX_SEQ_LEN,
        pad_value=pad_value,
        shuffle=True,
    )
    val_dataloader = create_dataloader(
        val_df,
        tokenizer,
        vocab,
        batch_size=args.batch_size,
        max_seq_len=MAX_SEQ_LEN,
        pad_value=pad_value,
        shuffle=False,
    )

    model = BiLSTM(
        vocab_size,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        NUM_LAYERS,
        NUM_CLASSES,
        pad_value,
    )
    model.update_embedding(embedding_vectors)
    model.to(device)

    print("Starting training...")
    train_model(
        model,
        train_dataloader,
        val_dataloader,
        args.epochs,
        MODEL_PATH,
        args
    )
    print(f"Training complete. Best model exported to '{MODEL_PATH}'.")
