# EXP-008 Training Code Inspection

Generated: 2026-07-07. This inspection covers code paths needed for the EXP-008 depth-heldout training pipeline. No training was run during this inspection.

## Method Identity

| Item | Evidence | Finding |
| --- | --- | --- |
| Experiment | `EXP-008` in `docs/thesis_evidence/experiment_inventory.md` | EfficientNet FFT severity regression |
| Branch | `origin/percentage_label+FFT` | Remote/local inventory maps EXP-008 to this branch |
| Commit | `7ba021cfa6eacd148247258ee28b8527dbbc6c92` | final verified EXP-008 branch commit |
| Source TFRecord | `data/processed/fft_regression/array_03/tfrecords/fft_regression_data.tfrecord` | read-only source for split builder |
| Depth index | `data/processed/fft_regression/array_03/tfrecords/fft_regression_data.tfrecord.idx.pkl` | maps TFRecord record order to `processed_waveforms.pkl['sonic_depths']` |

## Old EXP-008 Code

| Role | Code path | Key behavior |
| --- | --- | --- |
| config | `origin/percentage_label+FFT:config.py` | `TASK_TYPE='fft_regression'`; `INPUT_SHAPE=(150,400,8)`; `MAX_PATH_DEPTH_POINTS=70`; `FFT_COEFFICIENTS=30`; `BATCH_SIZE=128`; `EPOCHS=100`; `LEARNING_RATE=1e-4` |
| label generation | `origin/percentage_label+FFT:src/data_processing/create_tfrecords.py` | computes severity `max(0, 2.5 - Zc)`, FFT magnitude, first 30 coefficients, `log(1 + coefficients)`; writes `fft_regression_data.tfrecord`; saves `.idx.pkl` |
| parse function | `origin/percentage_label+FFT:src/modeling/train.py` | parses serialized `feature` and `label`; reshapes feature to `(150,400,8)`; masks first 30 time steps; reshapes label to `(70,30)` for `fft_regression` |
| dataset | `origin/percentage_label+FFT:src/modeling/dataset.py` | reads one TFRecord, maps parse function, shuffles if training, batches and prefetches |
| model | `origin/percentage_label+FFT:src/modeling/model.py` | `Conv2D(1x1)` channel adapter from 8 to 3 channels; EfficientNetV2B0 backbone; global average pooling; dropout 0.5; dense output reshaped to `(70,30)` |
| optimizer/loss/metrics | `origin/percentage_label+FFT:src/modeling/train.py` | Adam `1e-4`; Huber loss; MAE metric |
| callbacks | `origin/percentage_label+FFT:src/modeling/train.py` | TensorBoard; ModelCheckpoint by `val_loss`; EarlyStopping by `val_loss`; ReduceLROnPlateau |
| analysis | `origin/percentage_label+FFT:src/interpretation/run_analysis_regressor.py` | Grad-CAM, scatter plot, depth log comparison, FFT spectrum comparison |

## Old Split Limitation

The old training entry does not accept explicit train/val/test TFRecord files. It creates one dataset from `fft_regression_data.tfrecord`, shuffles it, computes batch count, then uses `take(train_size)` and `skip(train_size)` for validation. This is the source of `random_split_depth_leakage_risk` for EXP-008.

Code evidence:

```text
origin/percentage_label+FFT:src/modeling/train.py
full_dataset = create_dataset(..., is_training=True)
val_size = max(1, int(0.2 * dataset_size))
train_dataset = full_dataset.take(train_size)
val_dataset = full_dataset.skip(train_size)
```

## New Depth-Heldout Training Script

Implemented:

```text
scripts/thesis_train_exp008_depth_blocked.py
```

This script uses the same EXP-008 method components where practical:

| Component | Reuse status |
| --- | --- |
| TFRecord feature keys | reused: `feature`, `label` |
| input shape | reused: `(150,400,8)` |
| label shape | reused: `(70,30)` |
| artifact mask | reused: first 30 time steps masked |
| augmentation | reused: noise + frequency mask on train set only |
| model family | reused: EfficientNetV2B0 with 1x1 channel adapter and `(70,30)` regression head |
| loss/metric | reused: Huber loss and MAE |
| callbacks | reused conceptually: TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau |

It intentionally changes only the split interface:

- reads `split_dir/train.tfrecord`;
- reads `split_dir/val.tfrecord`;
- reads `split_dir/test.tfrecord`;
- never calls `validation_split`;
- never calls `train_test_split`;
- never splits a shuffled full dataset with `take/skip`.

## New Outputs

For each run directory `output/thesis_depth_blocked/exp008/train_v*/`, the script writes:

- `run_config.json`
- split evidence copies: `split_manifest.json`, `leakage_audit.json`
- `training_history.csv`
- `training_history.json`
- `models/best_model.h5`
- `val_metrics.json`
- `test_metrics.json`
- `prediction_summary.csv`
- `predictions_test.npz`
- `training_curve.png`
- `prediction_vs_truth_scatter.png`
- `depth_curve_if_available.png`
- `error_summary.md`
- `run_report.md`
- `failure_report.*` if an exception occurs

Large artifacts such as `models/best_model.h5` and `predictions_test.npz` should remain on the remote server and should not be copied into `docs/thesis_evidence`.

## Pretrained Weight Handling

The old EXP-008 model tried to load EfficientNetV2B0 no-top weights via `tf.keras.utils.get_file`. The new script does the same by default. If the remote environment cannot fetch or find the cached weights, the script records the failure in `run_config.json` and falls back to random initialization unless `--pretrained-required` is used.

To force no pretrained weights:

```bash
python scripts/thesis_train_exp008_depth_blocked.py --no-pretrained ...
```

This should only be used for recovery/debugging, because it differs from the old EXP-008 training intent.

## Pending Verification

- Smoke training must confirm TensorFlow 2.9.1 can build the scripted EfficientNetV2B0 model in `hall`.
- Full train_v001 must confirm memory fits with selected batch size.
- Test metrics must be interpreted as single-well depth-heldout evidence, not multi-well generalization.
