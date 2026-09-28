# MNIST CNN — Regularization Comparison

A small CNN trained on MNIST to study how dropout, batch normalization, and L2
weight decay each affect training dynamics and generalization, starting from
one shared baseline architecture.

## Setup

- **Data**: MNIST, split into 55k train / 5k val (from the original 60k train
  set) / 10k test (held out, untouched until final evaluation). Inputs
  normalized to the dataset's known mean/std (0.1307, 0.3081).
- **Architecture**: conv(1→32, 3x3) → [bn] → relu → maxpool(2x2) →
  conv(32→64, 3x3) → [bn] → relu → maxpool(2x2) → flatten → [dropout] →
  fc(3136→128) → relu → [dropout] → fc(128→10).
- **Training**: CrossEntropyLoss, Adam, lr=1e-3, batch size 64, 8 epochs,
  same random seed for the train/val split across all runs.
- All four variants share this architecture and training config, changing
  exactly one thing at a time, so differences in the curves are attributable
  to that one change.

## Variants and results (epoch 8)

| Variant | Train Acc | Val Acc | Train/Val Gap | Notes |
|---|---|---|---|---|
| Baseline | 99.65% | 98.86% | 0.79 pt | Val loss rises after epoch 6 — mild overfitting |
| + Dropout (p=0.4) | 98.91% | 98.98% | -0.07 pt | Slowest to converge, smallest gap, best val loss overall |
| + BatchNorm | 99.55% | 99.02% | 0.53 pt | Fastest early convergence, modest gap reduction |
| + L2 (weight_decay=1e-4) | 99.51% | 99.00% | 0.51 pt | Same convergence shape as baseline, uniformly smaller gap |

See `runs/comparison.png` for train/val loss and accuracy curves across all
four variants (one color per variant, dashed = train, solid = val).

### What each addition changed, and why

- **Dropout** randomly zeroes a fraction of activations each forward pass
  during training, preventing neurons from co-adapting and approximating an
  ensemble of subnetworks at inference time. This produced the clearest
  overfitting fix here: training accuracy actually *dropped* (the model can't
  fit the training set as tightly), but validation performance held or
  slightly improved, and the train/val gap essentially disappeared.
- **Batch normalization** re-normalizes each layer's pre-activations to
  zero-mean/unit-variance (with a learned scale/shift), keeping gradient
  scale stable across layers as weights update. Its main claimed benefit is
  faster convergence — visible here in epoch-1/2 val accuracy being slightly
  higher than baseline — but the effect was modest, since this network is
  shallow and already trained with Adam, which itself adapts per-parameter
  step sizes and mitigates part of what batch norm addresses.
- **L2 weight decay** penalizes large weights, biasing the model toward
  smoother functions that are less able to memorize training-specific noise.
  At `weight_decay=1e-4` the effect was a modest, uniform reduction in the
  train/val gap without changing the shape of the training curve — same
  convergence speed as baseline, just consistently less overfit.

## CPU vs GPU timing

Same baseline config, 3 epochs, single run each on CPU and GPU
(NVIDIA RTX 4070 Laptop GPU):

| Device | Time (3 epochs) | Time/epoch |
|---|---|---|
| CPU | 60.7s | 20.2s |
| GPU (CUDA) | 34.2s | 11.4s |

**~1.8x speedup on GPU.** This is a modest speedup compared to the large
numbers often quoted for GPU training — expected here because the model is
small (2 conv layers, ~400K params) and batch size (64) is small, so there
isn't enough parallel work per batch to fully utilize the GPU, and per-batch
overhead (kernel launches, host↔device transfers) eats into the win. Larger
models, larger batches, and larger inputs are where GPU speedups grow much
more dramatically.

## Files

- `data.py` — MNIST loading, train/val/test split, normalization.
- `model.py` — configurable `CNN` class (dropout_p, use_batchnorm flags).
- `train.py` — training loop, per-epoch logging, saves history + checkpoint
  per run under `runs/<name>/`.
- `plot_runs.py` — reads all saved runs and plots loss/accuracy comparison.
- `timing_comparison.py` — CPU vs GPU wall-clock comparison.
- `runs/` — one folder per run (`baseline`, `dropout`, `batchnorm`, `l2`),
  each with `history.json` (metrics) and `model.pth` (weights).
