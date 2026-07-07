# EXP-008 Depth-Heldout Training Report

Generated: 2026-07-07. This report archives the first full EXP-008 depth-heldout training run. It does not modify Windows results, raw data, or processed data.

## Run Identity

| item | value |
| --- | --- |
| experiment | EXP-008 severity + FFT magnitude/log label + EfficientNet |
| remote output | `/home/xiaoj/hal_azi/output/thesis_depth_blocked/exp008/train_v001` |
| local archive | `docs/thesis_evidence/remote_exp008_depth_blocked_train/` |
| split dir | `/home/xiaoj/hal_azi/output/thesis_depth_blocked/exp008/split_v001` |
| training script | `scripts/thesis_train_exp008_depth_blocked.py` |
| remote branch | `feature/thesis-depth-blocked-exp008` |
| remote git status during run | `?? requirements.txt; ?? scripts/thesis_train_exp008_depth_blocked.py` |
| EXP-008 source branch evidence | `origin/percentage_label+FFT@7ba021cfa6eacd148247258ee28b8527dbbc6c92` |
| Python/TensorFlow evidence | remote `hall`, TensorFlow 2.9.1 from prior audit |

## Split Evidence

| split | count | depth min ft | depth max ft | median ft |
| --- | ---: | ---: | ---: | ---: |
| train | 1984 | 2732.439628 | 3715.852672 | 3232.642772 |
| validation | 416 | 3721.118908 | 3917.684829 | 3819.798246 |
| test | 423 | 3922.785079 | 4129.743471 | 4025.084500 |

Audit conclusion: `depth_heldout_split_confirmed`. Record sets are mutually exclusive: `True`. Boundary distances are train/val `5.266235` ft and val/test `5.100250` ft with gap_ft `5.0`.

## Training Configuration

| item | value |
| --- | --- |
| epochs requested | 80 |
| epochs completed | 12 |
| early stop epoch | 12 |
| best val_loss epoch | 2 |
| batch size | 8 |
| learning rate | 0.0001 |
| patience | 10 |
| dropout | 0.5 |
| optimizer | Adam |
| loss | tf.keras.losses.Huber |
| pretrained EfficientNetV2B0 weights loaded | True |
| no random split | True |
| no validation_split | True |

## Metrics

| split | loss | MAE | RMSE | R2 | Pearson | Spearman | samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 0.015220 | 0.042817 | 0.187167 | 0.010882 | 0.200575 | 0.266137 | 416 |
| test | 0.032994 | 0.079025 | 0.277019 | 0.106634 | 0.351950 | 0.480519 | 423 |

Additional test metrics: DC MAE `0.142721`, DC RMSE `0.467645`, low-frequency coefficient 0-4 MAE `0.131411`, low-frequency coefficient 0-4 RMSE `0.431368`.

## Training Behavior

- Smoke training succeeded before full training.
- Full training v001 completed successfully and produced metrics/plots.
- No NaN was detected by the script.
- Training loss decreased from `0.050934` to `0.020418`.
- Best validation loss occurred at epoch `2`: `0.015220`.
- Best validation MAE occurred at epoch `7`: `0.036285`.
- EarlyStopping restored epoch 2 weights before evaluation.

## Archived Files

Local archive excludes checkpoint, TFRecords, raw data, and `predictions_test.npz`. Archived small files include metrics JSON, history CSV/JSON, figures, prediction summary CSV, run report, train log, split manifest JSON, and leakage audit JSON.
