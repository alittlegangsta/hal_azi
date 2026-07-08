# Final Thesis Metrics Table

Generated: 2026-07-08. This table freezes the metrics that are safe to use in the thesis result chapter. It separates old random-split exploratory evidence from the new single-well depth-heldout evidence.

## Thesis Position

| role | experiment | final use |
| --- | --- | --- |
| method innovation | EXP-008 severity + FFT magnitude label + EfficientNet | Main method result, with metric-qualified limitations. |
| fallback / limitation comparison | EXP-007 1D percentage label + EfficientNet | Simpler label fallback; useful for comparison, not stronger than EXP-008. |
| baseline / background | EXP-006 CNN binary classification | Demonstrates CWT-label learnability under random validation only. |

## Main Metrics

| experiment | split scope | comparator | test MAE | test RMSE | test R2 | Pearson | Spearman | thesis-safe reading |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| EXP-008 | single-well depth-heldout `array_03` | model | 0.079025 | 0.277019 | 0.106634 | 0.351950 | 0.480519 | Supports FFT-severity label learnability under single-well depth holdout. |
| EXP-008 | single-well depth-heldout `array_03` | zero baseline | 0.069751 | 0.301272 | -0.056638 | unknown | unknown | Zero has lower MAE; model is better by RMSE/R2. |
| EXP-008 | single-well depth-heldout `array_03` | train-mean baseline | 0.142358 | 0.316129 | -0.163426 | 0.418723 | 0.503310 | Model is better by MAE/RMSE/R2, but correlations are baseline-sensitive. |
| EXP-008 | single-well depth-heldout `array_03` | train-median baseline | 0.119024 | 0.278963 | 0.094049 | 0.419976 | 0.503310 | Model is slightly better by RMSE/R2 and better by MAE. |
| EXP-007 | single-well depth-heldout `array_03` | model `train_v002` | 0.811672 | 2.832056 | -0.011278 | 0.198048 | 0.413335 | Trainable fallback, but not cleaner than EXP-008. |
| EXP-007 | single-well depth-heldout `array_03` | zero baseline | 0.738339 | 2.911398 | -0.068735 | unknown | unknown | Zero has lower MAE; model is better by RMSE/R2. |
| EXP-007 | single-well depth-heldout `array_03` | train-mean baseline | 2.438934 | 4.860866 | -1.979161 | 0.414551 | 0.499721 | Model is better by MAE/RMSE/R2, but lower correlation. |
| EXP-007 | single-well depth-heldout `array_03` | train-median baseline | 1.390606 | 2.867288 | -0.036596 | 0.414366 | 0.507353 | Model is better by MAE/RMSE/R2, but lower correlation. |

EXP-006 is not included in the same performance comparison table because it is a different binary task and uses random validation with adjacent-depth leakage risk. It may be reported separately as exploratory background: validation AUC `0.95361` and validation accuracy `0.885366` from `result.txt.txt` / `training_history.pkl`.

## Error Structure

| experiment | evidence | observation | thesis use |
| --- | --- | --- | --- |
| EXP-008 | `exp008_depthheldout_baseline_comparison.csv`; `exp008_depthheldout_error_structure.*` | Low-frequency FFT coefficients have higher absolute error than high-frequency coefficients; high-severity samples tend to be underestimated. | Limitation and future work. |
| EXP-007 | `train_v002/test_metrics.json` remote path captured in evidence docs | Shallow/mid profile indices have larger MAE than deep indices; deep low error likely reflects sparse/near-zero labels. | Limitation; severity-group details remain `needs_verification` until remote CSV/JSON are copied/read. |

## Frozen Interpretation

The thesis should present EXP-008 as the primary method evidence because it matches the azimuth-invariant FFT-label research question and gives positive depth-heldout R2. EXP-007 should be presented after EXP-008 as a fallback showing that the simpler 1D profile label is trainable but not stronger. EXP-006 belongs in the baseline/background section only.
