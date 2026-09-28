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
}


def load_model(name, kwargs, device):
    model = CNN(**kwargs).to(device)
    model.load_state_dict(torch.load(f"runs/{name}/model.pth", map_location=device))
    model.eval()
    return model


def collect_examples(model, loader, device, want_correct, limit=16):
    examples = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            mask = (preds == labels) if want_correct else (preds != labels)
            for img, true, pred in zip(images[mask], labels[mask], preds[mask]):
                examples.append((img.cpu(), true.item(), pred.item()))
            if len(examples) >= limit:
                break
    return examples[:limit]


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

    best_name = max(test_accs, key=test_accs.get)
    print(f"\nInspecting {best_name} model's predictions")
    model = models[best_name]

    failures = collect_examples(model, test_loader, device, want_correct=False)
    print(f"Found {len(failures)} misclassified examples (showing up to 16)")
    plot_grid(failures, f"Misclassified test examples ({best_name} model)", "runs/failures.png")

    successes = collect_examples(model, test_loader, device, want_correct=True)
    plot_grid(successes, f"Correctly classified test examples ({best_name} model)", "runs/successes.png")
