# EXP-007 Artifact And Code Inspection

Generated: 2026-07-07. Scope: EXP-007 1D percentage label fallback, single-well `array_03`. This inspection did not modify raw data, `data/processed`, Windows results, or old result files.

## Remote Artifact Evidence

Remote repo: `/home/xiaoj/hal_azi` on `cement-server`.

| artifact | remote path | observed evidence |
| --- | --- | --- |
| CWT HDF5 | `data/processed/image_translation/array_03/cwt_images.h5` | key `cwt_images`, shape `(2846, 150, 400, 8)`, dtype `float32` |
| waveform/depth metadata | `data/processed/image_translation/array_03/processed_waveforms.pkl` | keys `sonic_depths`, `waveforms`; `sonic_depths_len=2846`; depth range `2732.439628` to `4131.742090`; `waveforms_shape=(2846, 8, 400)` |
| ground truth HDF5 | `data/processed/image_translation/array_03/ground_truth_db_array_03.h5` | keys `interpolated_zc_full`, `path_data`, `unified_depth_axis`; `path_data_len=2842` |
| TFRecord | `data/processed/image_translation/array_03/tfrecords/profile_regression_data.tfrecord` | `2842` records |
| `.idx.pkl` | `data/processed/image_translation/array_03/tfrecords/profile_regression_data.tfrecord.idx.pkl` | not present |

## Record-Depth Mapping

EXP-007 old TFRecord generation did not save `.idx.pkl`. Mapping was reconstructed from code evidence:

- Code path: `origin/1D+percentage_Label:src/data_processing/create_tfrecords.py`
- Logic: iterate `processed_waveforms.pkl['sonic_depths']` in order; create key `str(current_sonic_depth).replace('.', '_')`; write a record only if the key exists in `ground_truth_db_array_03.h5['path_data']`.
- Remote check: `2842` sonic depths matched `path_data`; the last 4 sonic depths did not match and were not written.
- Result: TFRecord record count `2842` equals reconstructed mapping count `2842`.

Conclusion: record-depth mapping is reliable enough for EXP-007 depth-heldout split generation.

## Label And Input Shape

Remote TensorFlow parse of three source TFRecord examples:

| field | value |
| --- | --- |
| feature shape | `(150, 400, 8)` |
| label shape | `(70,)` |
| label meaning | 70-point depth profile of channeling percentage |
| label construction code | `process_zc_slice_to_label`: `(zc_slice < 2.5)` mean along azimuth axis, multiplied by `100.0`, padded/truncated to `MAX_PATH_DEPTH_POINTS=70` |

## Old EXP-007 Training Code

| role | code evidence |
| --- | --- |
| training entry | `origin/1D+percentage_Label:src/modeling/train.py` |
| model | `origin/1D+percentage_Label:src/modeling/model.py` |
| dataset | `origin/1D+percentage_Label:src/modeling/dataset.py` |
| model body | EfficientNetV2B0, 1x1 channel adapter from 8 CWT channels to 3 channels, global average pooling, dropout `0.5`, Dense `70` ReLU output |
| loss/metric | Huber loss, MAE |
| old split flaw | shuffled dataset then `take/skip`; random adjacent-depth leakage risk |

## Supplement Implementation Decision

Use split-specific TFRecords:

- `output/thesis_depth_blocked/exp007/split_v001/train.tfrecord`
- `output/thesis_depth_blocked/exp007/split_v001/val.tfrecord`
- `output/thesis_depth_blocked/exp007/split_v001/test.tfrecord`

Training script:

- `scripts/thesis_train_exp007_depth_blocked.py`
- explicitly reads train/val/test TFRecords
- does not use random split or `validation_split`
- saves only new outputs under `output/thesis_depth_blocked/exp007/train_v*`

Limitations:

- single-well depth-heldout only
- no multi-well evidence
- remote output files were not fully copied back after the SSH/SCP approval limit was hit; local docs therefore cite remote output paths and captured command output.
