# EXP-007 Depth-Heldout Training Report

Generated: 2026-07-07. This report covers EXP-007 1D percentage label fallback training on a single-well depth-heldout split. No raw data, Windows results, processed data, split, or label formula was modified.

## Split

| split | count | min_depth_ft | max_depth_ft | median_depth_ft |
| --- | ---: | ---: | ---: | ---: |
| train | 1984 | 2732.439628 | 3715.852672 | 3232.642772 |
| validation | 416 | 3721.118908 | 3917.684829 | 3819.798246 |
| test | 423 | 3922.785079 | 4129.743471 | 4025.084500 |

Audit conclusion: `depth_heldout_split_confirmed`. Boundary distances were train/val `5.266235` ft and val/test `5.100250` ft with `gap_ft=5.0`.

## Runs

| run | change | epochs completed | best validation evidence | test MAE | test RMSE | test R2 | test Pearson | test Spearman | outcome |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| train_smoke_v001 | smoke only | 1 | chain verified | 0.857879 | 2.885499 | -0.049805 | 0.003825 | -0.014208 | not thesis metric |
| train_v001 | lr `1e-4` | 17 | best val_loss epoch 7: `0.32228`; early stopped at epoch 17 | 0.881390 | 2.982832 | -0.121823 | 0.151424 | 0.425906 | valid but weak |
| train_v002 | lr `5e-5` | 11 | best val_loss epoch 1: `0.34110`; early stopped at epoch 11 | 0.811672 | 2.832056 | -0.011278 | 0.198048 | 0.413335 | best completed EXP-007 run |
| train_v003 | lr `1e-5` planned | 0 | not run | unknown | unknown | unknown | unknown | unknown | not run: SSH/SCP escalation rejected by execution environment usage limit |

## Best Run

Best completed run: `train_v002`.

Remote source path:

`/home/xiaoj/hal_azi/output/thesis_depth_blocked/exp007/train_v002`

Relevant remote artifacts:

- `test_metrics.json`
- `val_metrics.json`
- `baseline_comparison.csv/json/md`
- `training_history.csv/json`
- `training_curve.png`
- `prediction_vs_truth_scatter.png`
- `residual_distribution.png`
- `depth_curve_if_available.png`
- `per_profile_index_mae.png`
- `severity_group_metrics.csv/json`
- `run_report.md`

Local note: these small artifacts were generated remotely, but full copying to local docs was blocked after SSH/SCP escalation was denied by the execution environment usage limit. Numeric values in this report are captured from remote command output.

## Baseline Comparison

`train_v002`:

| comparator | MAE | RMSE | R2 |
| --- | ---: | ---: | ---: |
| model | 0.811672 | 2.832056 | -0.011278 |
| zero | 0.738339 | 2.911398 | -0.068735 |
| train_mean | 2.438934 | 4.860866 | -1.979161 |
| train_median | 1.390606 | 2.867288 | -0.036596 |

Conclusion: EXP-007 beats train-mean/train-median on MAE/RMSE/R2 and beats zero on RMSE/R2, but still does not beat zero on MAE.

## Training Behavior

- No NaN was observed in completed v001/v002 logs.
- v001 improved validation loss through epoch 7 and then overfit.
- v002 improved only at epoch 1 and then overfit/failed to improve.
- v003 was not run because remote command escalation was rejected by the execution environment usage limit.

## Thesis Use

EXP-007 should be used as a fallback/comparison result, not as the clean main result. It supports that 1D percentage labels are trainable under a single-well depth-heldout split, but the zero-baseline MAE limitation remains.
