"""Utilities for training the BiLSTM model."""

import torch

from .data import get_length


def train_one_epoch(
    model,
    train_dataloader,
    loss_function,
    epoch,
    optimizer,
):
    """Train the model for one complete pass over the training dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training model on {device}")
    model.train()
    total_loss = 0.0

    for input_ids, labels in train_dataloader:
        lengths = get_length(input_ids, model.pad_value)
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs, _ = model(input_ids, lengths)
        loss = loss_function(outputs.view(-1), labels.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch} - Average training loss: {average_loss:.4f}")
    return average_loss


def evaluate_model(model, val_dataloader, loss_function, epoch):
    """Return mean validation loss and accuracy (0–1) for binary logits.

    The loss function should return a mean loss for each batch, as
    BCEWithLogitsLoss does by default.
    """
    print("evaluating model..")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for input_ids, labels in val_dataloader:
            lengths = get_length(input_ids, model.pad_value)
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            outputs, _ = model(input_ids, lengths)
            logits = outputs.view(-1)
            loss = loss_function(logits, labels.float())
            predictions = torch.sigmoid(logits) >= 0.5

            batch_size = labels.numel()
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Validation dataloader must contain at least one sample.")

    val_loss = total_loss / total_samples
    val_accuracy = total_correct / total_samples
    print(
        f"Epoch {epoch} - Validation loss: {val_loss:.4f}, "
        f"Validation accuracy: {val_accuracy:.4f}"
    )
    return val_loss, val_accuracy
