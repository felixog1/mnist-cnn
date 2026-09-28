import json

import torch

from model import CNN
from train import run_training

EPOCHS = 3  # timing only, not accuracy comparison -- keep short

if __name__ == "__main__":
    results = {}

    for device_name in ["cpu", "cuda"]:
        if device_name == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, skipping.")
            continue
        device = torch.device(device_name)
        model = CNN()
        _, elapsed = run_training(
            model, run_name=f"timing_{device_name}", epochs=EPOCHS, device=device
        )
        results[device_name] = elapsed

    print("\n--- Timing summary ---")
    for device_name, elapsed in results.items():
        print(f"{device_name}: {elapsed:.1f}s for {EPOCHS} epochs "
              f"({elapsed / EPOCHS:.1f}s/epoch)")

    if "cpu" in results and "cuda" in results:
        speedup = results["cpu"] / results["cuda"]
        print(f"GPU speedup: {speedup:.1f}x")

    with open("runs/timing_summary.json", "w") as f:
        json.dump({"epochs": EPOCHS, "results": results}, f, indent=2)
