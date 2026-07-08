# EXP-007 Depth-Heldout Final Claim

Generated: 2026-07-07. Scope: single-well depth-heldout EXP-007 on `array_03`.

## Claim

EXP-007 1D percentage label regression is trainable under a contiguous depth-heldout split, but it does not provide a cleaner thesis fallback than EXP-008.

## Evidence

Best completed run: `train_v002`.

| metric | value |
| --- | ---: |
| test MAE | 0.811672 |
| test RMSE | 2.832056 |
| test R2 | -0.011278 |
| test Pearson | 0.198048 |
| test Spearman | 0.413335 |
| test samples | 423 |

Baseline comparison:

- Better than zero baseline by RMSE/R2, but worse by MAE.
- Better than train-mean and train-median by MAE/RMSE/R2.
- Correlation is lower than train-mean/train-median baselines, so correlation should not be used as the main comparison.

## Thesis Role

Recommended thesis role: `fallback / limitation comparison`.

EXP-008 remains the method innovation mainline because:

- FFT magnitude label is more aligned with the azimuth-invariance research question.
- EXP-008 depth-heldout R2 is positive (`0.106634`), while EXP-007 best R2 remains slightly negative (`-0.011278`).
- Both EXP-008 and EXP-007 fail to beat zero baseline by MAE, so sparse-label limitations must be stated in the results chapter.

## Required Limitations

- single-well only
- depth-heldout within `array_03`, not multi-well generalization
- sparse label distribution makes zero baseline hard to beat by MAE
- severity group details are `needs_verification` until remote `severity_group_metrics.csv` is copied/read
- v003 was not run because SSH/SCP escalation was rejected by the execution environment usage limit
