import json
import os

import matplotlib.pyplot as plt

RUNS_DIR = "runs"


def load_histories():
    histories = {}
    for name in sorted(os.listdir(RUNS_DIR)):
        if name.startswith("timing_"):
            continue  # timing benchmark runs, not part of the variant comparison
        path = os.path.join(RUNS_DIR, name, "history.json")
        if os.path.exists(path):
            with open(path) as f:
                histories[name] = json.load(f)
    return histories


def plot(histories, out_path="runs/comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for color, (name, data) in zip(colors, histories.items()):
        h = data["history"]
        epochs = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(epochs, h["train_loss"], linestyle="--", color=color, label=f"{name} (train)")
        axes[0].plot(epochs, h["val_loss"], linestyle="-", color=color, label=f"{name} (val)")
        axes[1].plot(epochs, h["train_acc"], linestyle="--", color=color, label=f"{name} (train)")
        axes[1].plot(epochs, h["val_acc"], linestyle="-", color=color, label=f"{name} (val)")

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    histories = load_histories()
    print(f"Found runs: {list(histories.keys())}")
    plot(histories)
