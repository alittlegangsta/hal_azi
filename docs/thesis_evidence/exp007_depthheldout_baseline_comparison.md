# EXP-007 Depth-Heldout Baseline Comparison

Generated: 2026-07-07. Scope: single-well depth-heldout EXP-007 on `array_03`.

Remote source root:

`/home/xiaoj/hal_azi/output/thesis_depth_blocked/exp007`

Local note: metrics were captured from remote command output. Full remote small artifacts were not copied after SSH/SCP escalation was denied by the execution environment usage limit.

## Test Metrics

| run | comparator | MAE | RMSE | R2 | Pearson | Spearman |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train_v001 | model | 0.881390 | 2.982832 | -0.121823 | 0.151424 | 0.425906 |
| train_v001 | zero | 0.738339 | 2.911398 | -0.068735 |  |  |
| train_v001 | train_mean | 2.438934 | 4.860866 | -1.979161 | 0.414551 | 0.499721 |
| train_v001 | train_median | 1.390606 | 2.867288 | -0.036596 | 0.414366 | 0.507353 |
| train_v002 | model | 0.811672 | 2.832056 | -0.011278 | 0.198048 | 0.413335 |
| train_v002 | zero | 0.738339 | 2.911398 | -0.068735 |  |  |
| train_v002 | train_mean | 2.438934 | 4.860866 | -1.979161 | 0.414551 | 0.499721 |
| train_v002 | train_median | 1.390606 | 2.867288 | -0.036596 | 0.414366 | 0.507353 |

## Interpretation

- `train_v002` is the best EXP-007 run among completed attempts by test RMSE and R2.
- `train_v002` beats zero baseline by RMSE/R2 but not by MAE.
- `train_v002` beats train-mean and train-median baselines by MAE/RMSE/R2.
- Correlation metrics are weaker than train-mean/train-median baselines because those baselines encode the train profile shape; use correlation comparisons cautiously.
- EXP-007 does not provide a cleaner fallback than EXP-008 on zero-baseline MAE.
