import json

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from data import get_dataloaders, MNIST_MEAN, MNIST_STD
from model import CNN
from train import evaluate

VARIANTS = {
    "baseline": dict(),
    "dropout": dict(dropout_p=0.4),
    "batchnorm": dict(use_batchnorm=True),
    "l2": dict(),
    "combined": dict(dropout_p=0.4, use_batchnorm=True),
}


def load_model(name, kwargs, device):
    model = CNN(**kwargs).to(device)
    model.load_state_dict(torch.load(f"runs/{name}/model.pth", map_location=device))
    model.eval()
    return model


def collect_examples(model, loader, device, want_correct, limit=None):
    """Scans the loader; limit=None scans everything so len() is the true total."""
    examples = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            mask = (preds == labels) if want_correct else (preds != labels)
            for img, true, pred in zip(images[mask], labels[mask], preds[mask]):
                examples.append((img.cpu(), true.item(), pred.item()))
            if limit is not None and len(examples) >= limit:
                break
    return examples if limit is None else examples[:limit]


def plot_grid(examples, title, out_path):
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for ax, (img, true, pred) in zip(axes.flat, examples):
        img = img.squeeze(0) * MNIST_STD + MNIST_MEAN  # undo normalization for display
        ax.imshow(img, cmap="gray")
        ax.set_title(f"true={true} pred={pred}", fontsize=10)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_dataloaders()
    criterion = nn.CrossEntropyLoss()

    print("--- Final test set results (touched once, here) ---")
    models, test_accs = {}, {}
    for name, kwargs in VARIANTS.items():
        model = load_model(name, kwargs, device)
        models[name] = model
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        test_accs[name] = test_acc
        print(f"{name:10s} test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    # Select by validation loss, never test. Use the final epoch's val loss because
    # model.pth holds the final-epoch weights; a best-epoch value would describe weights we didn't save.
    final_val_losses = {}
    for name in VARIANTS:
        with open(f"runs/{name}/history.json") as f:
            final_val_losses[name] = json.load(f)["history"]["val_loss"][-1]
    best_name = min(final_val_losses, key=final_val_losses.get)
    print(f"\nSelected by final-epoch validation loss: {best_name} (val_loss={final_val_losses[best_name]:.4f})")
    print(f"Inspecting {best_name} model's predictions")
    model = models[best_name]

    failures = collect_examples(model, test_loader, device, want_correct=False)
    n_test = len(test_loader.dataset)
    print(f"Total misclassified: {len(failures)} / {n_test} test images (showing first 16)")
    plot_grid(failures[:16], f"Misclassified test examples ({best_name} model)", "runs/failures.png")

    successes = collect_examples(model, test_loader, device, want_correct=True, limit=16)
    plot_grid(successes, f"Correctly classified test examples ({best_name} model)", "runs/successes.png")
