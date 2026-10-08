"""Re-measure the train/val gap with both sides in eval mode, from saved checkpoints.

The train accuracy logged during training is computed in train mode (dropout active,
batch norm on per-batch stats) and averaged over a moving model, so it isn't
comparable to val accuracy, which is computed in eval mode on end-of-epoch weights.
"""
import json

import torch
import torch.nn as nn

from data import get_dataloaders
from inspect_failures import VARIANTS, load_model
from train import evaluate

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_dataloaders()
    criterion = nn.CrossEntropyLoss()

    results = {}
    header = f"{'variant':10s} | {'logged train':>12s} | {'eval train':>10s} | {'val':>7s} | {'old gap':>8s} | {'true gap':>8s}"
    print(header)
    print("-" * len(header))
    for name, kwargs in VARIANTS.items():
        with open(f"runs/{name}/history.json") as f:
            logged_train_acc = json.load(f)["history"]["train_acc"][-1]

        model = load_model(name, kwargs, device)
        train_loss, train_acc = evaluate(model, train_loader, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        results[name] = {
            "logged_train_acc": logged_train_acc,
            "eval_train_acc": train_acc, "eval_train_loss": train_loss,
            "val_acc": val_acc, "val_loss": val_loss,
        }
        print(f"{name:10s} | {logged_train_acc:12.4f} | {train_acc:10.4f} | {val_acc:7.4f} | "
              f"{(logged_train_acc - val_acc) * 100:+7.2f}pt | {(train_acc - val_acc) * 100:+7.2f}pt")

    with open("runs/gap_check.json", "w") as f:
        json.dump(results, f, indent=2)
