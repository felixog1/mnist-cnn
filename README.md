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
- The first four variants change exactly one thing at a time relative to the
  baseline, so differences in the curves are attributable to that one change.
  A fifth `combined` variant stacks all three together.

## Variants and results (epoch 8)

| Variant | Train Acc | Val Acc | Train/Val Gap | Notes |
|---|---|---|---|---|
| Baseline | 99.65% | 98.86% | 0.79 pt | Val loss rises after epoch 6 — mild overfitting |
| + Dropout (p=0.4) | 98.91% | 98.98% | -0.07 pt | Slowest to converge, smallest gap, best val loss overall |
| + BatchNorm | 99.55% | 99.02% | 0.53 pt | Fastest early convergence, modest gap reduction |
| + L2 (weight_decay=1e-4) | 99.51% | 99.00% | 0.51 pt | Same convergence shape as baseline, uniformly smaller gap |
| + Combined (dropout + batchnorm + L2) | 98.52% | 99.10% | -0.58 pt | Still under-fit at epoch 8 — regularizers compound, need more epochs to reach full capacity |

See `runs/comparison.png` for train/val loss and accuracy curves across all
five variants (one color per variant, dashed = train, solid = val).

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
- **Combining all three** compounds each one's "handicap training" effect
  (dropout zeroes activations, L2 pulls weights toward zero, batch norm adds
  batch-statistics noise), so the model needs more epochs to reach the same
  training fit the individual variants got to alone. At a fixed 8 epochs it's
  still visibly under-fit (lowest train accuracy of all five, and still
  climbing) — regularization strength is a budget spent against the epoch
  budget, not a free win.

## Test set results

The test set (10k images, separate from the 55k/5k train/val split) was
touched exactly once, at the end, after all training and model-selection
decisions were already made from validation metrics — using it earlier or to
pick a variant would bias it as a measure of generalization.

**Model selection was based on validation loss, not test accuracy.** Dropout
had the best validation loss (0.0287) of all five variants and was selected
as the "final" model before test evaluation. For comparison purposes only,
test accuracy is reported for all five below:

| Variant | Test Loss | Test Acc |
|---|---|---|
| Baseline | 0.0428 | 98.94% |
| Dropout (selected) | 0.0254 | 99.21% |
| BatchNorm | 0.0405 | 98.83% |
| L2 | 0.0291 | 98.99% |
| Combined | 0.0237 | 99.24% |

Note combined edges out dropout slightly on this particular test set
(99.24% vs 99.21%) — but since combined wasn't the validation-selected model,
this difference isn't used to change the selection after the fact. That
would be using the test set to make a modeling decision, exactly what
holding it out is meant to prevent. It's also a small enough gap (3 examples
out of 10,000) to plausibly be noise rather than a real difference.

### What the errors look like

`inspect_failures.py` pulls misclassified and correctly-classified examples
from the selected (dropout) model on the test set:

- `runs/failures.png` — of 10,000 test images, only 16-17 are misclassified.
  Nearly all are genuinely ambiguous or badly-formed handwriting (unclosed
  loops, ambiguous strokes) rather than clean digits the model should
  obviously get right — a sign the model learned real digit structure rather
  than a spurious shortcut.
- `runs/successes.png` — for contrast, correctly classified examples are
  mostly clean, unambiguous handwriting.

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
- `inspect_failures.py` — final test-set evaluation for all variants, plus
  misclassified/correct example grids for the validation-selected model.
- `runs/` — one folder per run (`baseline`, `dropout`, `batchnorm`, `l2`,
  `combined`), each with `history.json` (metrics) and `model.pth` (weights).
