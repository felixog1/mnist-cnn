# MNIST CNN: regularization comparison

A small CNN on MNIST, used to see what dropout, batch norm and L2 each do to overfitting, starting from one shared baseline. Plus a CPU vs GPU timing study.

## Takeaways

- The baseline overfits mildly (val loss rises after epoch 6). All regularized variants have a smaller train/val gap than the baseline, but only combined's reduction is clearly larger than single-run noise.
- The differences between dropout, batch norm and L2 are within single-run noise, so this is not a ranking.
- On GPU, data loading (not compute) was the bottleneck: about 80% of epoch time. Preloading the data gave a 7.8x GPU speedup instead of 1.8x.

## Setup

- **Data:** 55k train / 5k val / 10k test (test used only for final evaluation; see the correction under Test results). Normalized with mean 0.1307, std 0.3081.
- **Model:** conv(1→32) → [bn] → relu → pool → conv(32→64) → [bn] → relu → pool → flatten → [dropout] → fc(3136→128) → relu → [dropout] → fc(128→10). About 420K params.
- **Training:** Adam, lr 1e-3, batch 64, 8 epochs, one run per variant, same train/val split for all.
- **Variants:** baseline, dropout (p=0.4), batch norm, L2 (weight_decay=1e-4), and all three combined. The first four each change one thing from the baseline.

## Results (epoch 8)

| Variant | Train acc | Val acc | Gap |
|---|---|---|---|
| Baseline | 99.75% | 98.86% | 0.89 pt |
| Dropout | 99.67% | 98.98% | 0.69 pt |
| BatchNorm | 99.72% | 99.02% | 0.70 pt |
| L2 | 99.75% | 99.00% | 0.75 pt |
| Combined | 99.59% | 99.10% | 0.49 pt |

Train accuracy is re-measured in eval mode after training (`gap_check.py`), so it's comparable to val. The accuracy logged during training is not: it's measured in train mode (dropout on, noisy batch-norm stats) and averaged over a changing epoch. That made dropout and combined look under-fit and produced negative gaps, which were a measurement artifact.

Standard error on a 5k val set is about 0.14 pt, so the 0.69 to 0.75 pt gaps for dropout, batch norm and L2 can't be told apart, and L2's 0.14 pt reduction from baseline is itself within noise. Combined's 0.40 pt reduction is the only one that clearly stands out.

`runs/comparison.png` plots train and val loss/accuracy per epoch for all five variants (dashed = train, solid = val). The dashed train curves are the train-mode numbers logged during training, so dropout and combined look worse there than they really are.

**What each one does**
- **Dropout:** randomly zeroes activations during training. Lowest single-epoch val loss of any variant (0.0287 at epoch 7), and its val loss trended down through epoch 7 instead of turning up like the baseline's.
- **Batch norm:** normalizes layer inputs. Slightly faster early convergence; small effect here because the network is shallow and Adam already adapts step sizes.
- **L2:** penalizes large weights. Smaller gap, same curve shape as baseline. Note that `weight_decay` in `torch.optim.Adam` is coupled L2: the penalty is added to the gradient and then rescaled by Adam, so its strength depends on gradient size. `AdamW` applies it directly to the weights (decoupled) and is usually what you want.
- **Combined:** smallest gap and highest val accuracy, with slightly lower train accuracy, as expected from stronger regularization.

## Test results

Model selection uses validation loss only, measured on the checkpoint that's actually tested. Only the epoch-8 weights are saved, so the rule is lowest epoch-8 val loss: combined (0.0312) over dropout (0.0347).

| Variant | Epoch-8 val loss | Test loss | Test acc |
|---|---|---|---|
| Baseline | 0.0457 | 0.0428 | 98.94% |
| Dropout | 0.0347 | 0.0254 | 99.21% |
| BatchNorm | 0.0479 | 0.0405 | 98.83% |
| L2 | 0.0419 | 0.0291 | 98.99% |
| Combined (selected) | 0.0312 | 0.0237 | 99.24% |

**Correction:** an earlier version selected dropout using its best val loss across epochs (0.0287, at epoch 7). Those epoch-7 weights were never saved, so the selection was based on a number the tested model didn't have. The rule was corrected after the test results had been seen, so the test number isn't a fully clean held-out estimate. It's also only 3 images better than dropout out of 10,000, which is noise. The proper fix for future runs is to save the best-val-loss checkpoint during training (early stopping) and select on that.

**Errors:** the combined model gets 76 of 10,000 test images wrong. `runs/failures.png` shows the first 16. Most are ambiguous or badly formed, and many are the same images the dropout model got wrong. A few are readable digits, and 5 of the 16 are predicted as 5 (mostly 3s and 9s). The other 60 haven't been inspected.

## CPU vs GPU timing

Training throughput only (forward, backward, optimizer step), RTX 4070 Laptop vs the laptop CPU. One untimed warmup epoch, then the mean of 3 timed epochs, with `torch.cuda.synchronize()` around each.

| Setup | CPU s/epoch | GPU s/epoch | Speedup |
|---|---|---|---|
| DataLoader, batch 64 (real pipeline) | 14.00 | 7.87 | 1.8x |
| Preloaded on device, batch 64 | 12.43 | 1.59 | 7.8x |
| Preloaded on device, batch 512 | 10.79 | 0.83 | 13.0x |

- **Data loading is the bottleneck.** The GPU made compute fast, but the DataLoader (`num_workers=0`, one image at a time in Python) didn't get faster, so it now takes about 80% of the GPU epoch. On CPU, compute dominates, so removing the DataLoader saves only 1.6s.
- **Small batches underuse the GPU.** Batch 64 to 512 halves GPU time but barely changes CPU time. (Larger batches also mean fewer weight updates per epoch, so this isn't a free speedup.)
- **Not yet tested:** why data loading costs more in the GPU run (6.3s vs 1.6s). Next step is `num_workers > 0` with `pin_memory=True`. Preloading only works here because MNIST is small.

## Files

- `data.py`: loading, split, normalization
- `model.py`: configurable `CNN` (dropout_p, use_batchnorm)
- `train.py`: training loop; saves history and weights to `runs/<name>/`
- `plot_runs.py`: loss/accuracy comparison plot
- `gap_check.py`: eval-mode train accuracy from saved checkpoints
- `inspect_failures.py`: test evaluation for all variants, plus error and success grids
- `timing_comparison.py`: CPU vs GPU benchmark; results in `runs/timing_summary.json`
