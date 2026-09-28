import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from data import get_dataloaders
from model import CNN


def evaluate(model, loader, criterion, device):
    """Runs the model in eval mode over a loader, no gradient tracking, returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # clear gradients from previous step
        outputs = model(images)        # forward pass
        loss = criterion(outputs, labels)
        loss.backward()                # backprop: compute gradients
        optimizer.step()               # update weights using those gradients

        total_loss += loss.item() * images.size(0)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def run_training(model, run_name, epochs=8, lr=1e-3, weight_decay=0.0, batch_size=64, device=None):
    """Trains `model`, logs per-epoch metrics, saves history + checkpoint under runs/<run_name>/."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    start = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"[{run_name}] epoch {epoch}/{epochs} | "
              f"train_loss {train_loss:.4f} train_acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} val_acc {val_acc:.4f}")

    elapsed = time.time() - start
    print(f"[{run_name}] training took {elapsed:.1f}s on {device}")

    os.makedirs(f"runs/{run_name}", exist_ok=True)
    torch.save(model.state_dict(), f"runs/{run_name}/model.pth")
    with open(f"runs/{run_name}/history.json", "w") as f:
        json.dump({"history": history, "elapsed_seconds": elapsed, "device": str(device)}, f, indent=2)

    return history, elapsed


if __name__ == "__main__":
    model = CNN()
    run_training(model, run_name="l2", epochs=8, weight_decay=1e-4)
