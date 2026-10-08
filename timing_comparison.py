"""CPU vs GPU training-throughput benchmark.

Three setups, all using the same train step (forward, backward, optimizer step,
no per-batch .item() logging, no validation pass):
  1. dataloader_bs64: the DataLoader pipeline used for the real runs (num_workers=0)
  2. preloaded_bs64:  whole train set preloaded on the device as one tensor -> pure compute
  3. preloaded_bs512: same, bigger batches -> more parallel work per step

Each setup runs 1 untimed warmup epoch, then TIMED_EPOCHS timed epochs.
"""
import json
import statistics
import time

import torch
import torch.nn as nn
import torch.optim as optim

from data import MNIST_MEAN, MNIST_STD, get_dataloaders
from model import CNN

TIMED_EPOCHS = 3


def sync(device):
    # CUDA kernels launch asynchronously; wait for them so the timer measures finished work.
    if device.type == "cuda":
        torch.cuda.synchronize()


def train_step(model, images, labels, criterion, optimizer):
    optimizer.zero_grad()
    loss = criterion(model(images), labels)
    loss.backward()
    optimizer.step()


def dataloader_epoch(model, loader, device, criterion, optimizer):
    for images, labels in loader:
        train_step(model, images.to(device), labels.to(device), criterion, optimizer)


def preloaded_epoch(model, x, y, batch_size, criterion, optimizer):
    perm = torch.randperm(len(x), device=x.device)
    for i in range(0, len(x), batch_size):
        idx = perm[i:i + batch_size]
        train_step(model, x[idx], y[idx], criterion, optimizer)


def preload(train_set, device):
    """Same pixels/normalization as ToTensor+Normalize, done once for the whole subset."""
    idx = torch.tensor(train_set.indices)
    x = train_set.dataset.data[idx].float().div(255).sub(MNIST_MEAN).div(MNIST_STD).unsqueeze(1)
    y = train_set.dataset.targets[idx]
    return x.to(device), y.to(device)


def time_setup(run_epoch, device):
    torch.manual_seed(0)
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    run_epoch(model, criterion, optimizer)  # warmup: cuDNN algorithm selection, allocator, caches
    times = []
    for _ in range(TIMED_EPOCHS):
        sync(device)
        start = time.perf_counter()
        run_epoch(model, criterion, optimizer)
        sync(device)
        times.append(time.perf_counter() - start)
    return times


if __name__ == "__main__":
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    train_loader, _, _ = get_dataloaders(batch_size=64)
    results = {}

    for device in devices:
        x, y = preload(train_loader.dataset, device)
        setups = {
            "dataloader_bs64": lambda m, c, o: dataloader_epoch(m, train_loader, device, c, o),
            "preloaded_bs64": lambda m, c, o: preloaded_epoch(m, x, y, 64, c, o),
            "preloaded_bs512": lambda m, c, o: preloaded_epoch(m, x, y, 512, c, o),
        }
        for setup_name, run_epoch in setups.items():
            times = time_setup(run_epoch, device)
            results.setdefault(setup_name, {})[device.type] = times
            print(f"{setup_name:16s} {device.type:4s} "
                  f"{statistics.mean(times):6.2f}s/epoch (+/- {statistics.stdev(times):.2f})  {times}")

    print("\n--- Summary (mean s/epoch over timed epochs) ---")
    summary = {}
    for setup_name, by_device in results.items():
        cpu = statistics.mean(by_device["cpu"])
        row = {"cpu_s_per_epoch": cpu}
        line = f"{setup_name:16s} cpu {cpu:6.2f}s"
        if "cuda" in by_device:
            gpu = statistics.mean(by_device["cuda"])
            row.update(gpu_s_per_epoch=gpu, speedup=cpu / gpu)
            line += f" | gpu {gpu:6.2f}s | speedup {cpu / gpu:5.1f}x"
        summary[setup_name] = row
        print(line)

    with open("runs/timing_summary.json", "w") as f:
        json.dump({
            "timed_epochs": TIMED_EPOCHS,
            "cpu_threads": torch.get_num_threads(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "raw_epoch_times": results,
            "summary": summary,
        }, f, indent=2)
