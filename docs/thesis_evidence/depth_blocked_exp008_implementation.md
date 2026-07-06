# EXP-008 Depth-Blocked Split Tooling Implementation P1

Generated: 2026-07-06. This stage implemented split/audit tooling only. No model training was run, no raw data was modified, no CWT/preprocessing was rerun, and `/mnt/c/Users/Administrator/Desktop/Hal/results` was not accessed for writes.

## Scope

This P1 stage prepares EXP-008 for a future depth-heldout training run. It creates scripts that can split the existing EXP-008 TFRecord by contiguous sonic depth and audit leakage risk. It does not modify the old training code and does not start training.

## EXP-008 Evidence And Required Inputs

| Item | Path / evidence | Status |
| --- | --- | --- |
| method branch | `origin/percentage_label+FFT` | verified from Git inventory |
| branch commit | `7ba021cfa6eacd148247258ee28b8527dbbc6c92` | verified |
| old training entry | `origin/percentage_label+FFT:src/modeling/train.py` | verified |
| old model definition | `origin/percentage_label+FFT:src/modeling/model.py` | verified |
| old TFRecord creation | `origin/percentage_label+FFT:src/data_processing/create_tfrecords.py` | verified |
| input shape | `(150, 400, 8)` | from `config.py` / `_parse_regression_tfrecord_fn` |
| label shape | `(70, 30)` for `TASK_TYPE=fft_regression` | from `config.py` and train/model code |
| model | EfficientNetV2B0 backbone + regression head | from `build_advanced_profile_regressor` |
| loss/metric in old code | Huber loss, MAE | from `train.py` |
| old split limitation | `create_dataset(... is_training=True)` shuffles, then `take/skip` splits batches | verified; not depth-heldout |

Remote processed inputs expected for real split:

```text
data/processed/fft_regression/array_03/
  cwt_images.h5
  ground_truth_db_array_03.h5
  processed_waveforms.pkl
  tfrecords/fft_regression_data.tfrecord
  tfrecords/fft_regression_data.tfrecord.idx.pkl
```

Remote metadata from the prior read-only audit:

| Artifact | Observed metadata |
| --- | --- |
| `processed_waveforms.pkl` | keys `sonic_depths`, `waveforms`; `sonic_depths_len=2846`; depth range `2732.4396282122984` to `4131.742089956898`; `waveforms_shape=(2846, 8, 400)` |
| `cwt_images.h5` | `cwt_images` shape `(2846, 150, 400, 8)` |
| `ground_truth_db_array_03.h5` | `path_data_len=2842`; `unified_depth_axis_shape=(14000,)` |
| `fft_regression_data.tfrecord.idx.pkl` | `processed_indices` length `2842`, index range `0` to `2841` |

The local clone does not need to contain these processed files for this P1 implementation. The scripts are intended to run on the remote server where these artifacts were verified.

## Added Scripts

| Script | Purpose | Training? | TensorFlow required? |
| --- | --- | --- | --- |
| `scripts/thesis_make_exp008_depth_split.py` | Build train/val/test TFRecords by sonic depth, write manifest and leakage audit | no | no |
| `scripts/thesis_check_exp008_tfrecord_split.py` | Count split TFRecords and optionally parse one sample shape | no | only for `--parse-one`; count checks are pure Python |

The split builder copies serialized TFRecord records at the TFRecord container level. It does not parse or rewrite tensors, and it does not load full CWT arrays into memory.

## Split Builder Behavior

Default command behavior:

- source TFRecord: `data/processed/fft_regression/array_03/tfrecords/fft_regression_data.tfrecord`
- source index: `data/processed/fft_regression/array_03/tfrecords/fft_regression_data.tfrecord.idx.pkl`
- depth file: `data/processed/fft_regression/array_03/processed_waveforms.pkl`
- output dir: `output/thesis_depth_blocked/exp008/split_v001`
- split preset: `depth_heldout_simple`
- train/val/test fractions: `70% / 15% / 15%` after sorting records by depth
- default gap: `5.0 ft`

Supported split presets:

| Preset | Use |
| --- | --- |
| `depth_heldout_simple` | default; train lower-depth interval, validation middle interval, test upper-depth interval |
| `tail_holdout` | explicit tail-heldout alias for the same lower/middle/tail layout |
| `middle_holdout_optional` | optional stress test with middle interval held out |

Supported safety flags:

- `--dry-run`: writes manifest/audit/overview only; does not write `train.tfrecord`, `val.tfrecord`, or `test.tfrecord`.
- `--max-records-for-smoke`: uses the first N records for a small smoke split.
- `--overwrite`: required to replace existing split outputs.

The builder refuses to write inside `/mnt/c/Users/Administrator/Desktop/Hal/results` and refuses to write split outputs into the source TFRecord directory.

## Leakage Audit Logic

The builder writes `leakage_audit.json` and `leakage_audit.md`. The audit checks:

- train/val/test record and sample index mutual exclusivity;
- train/val/test count;
- train/val/test min/max/median depth;
- nearest kept depth distance between split intervals;
- depth range overlap;
- configured gap usage;
- dropped buffer count;
- conclusion.

Expected successful conclusion:

```text
depth_heldout_split_confirmed
```

If any split is empty, record/sample overlap exists, depth intervals overlap, or the configured gap is not respected, the conclusion becomes:

```text
needs_manual_verification
```

## Training Integration Status

The old EXP-008 `train.py` does not currently accept explicit `train.tfrecord` and `val.tfrecord` paths. It still constructs one dataset from `fft_regression_data.tfrecord`, shuffles it, and then uses `take/skip`.

Therefore, this P1 stage is ready for split generation and smoke checks, but not yet ready for approved training. The next implementation stage should add a small training entry that:

- reads `train.tfrecord` and `val.tfrecord` explicitly;
- does not call `take/skip` on a shuffled full dataset;
- saves checkpoint/history under `output/thesis_depth_blocked/exp008/train_v001/`;
- evaluates only `test.tfrecord`;
- copies `split_manifest.csv` into the train output directory.

No full training command should be run until the user explicitly approves the training stage.

## Validation Performed In P1

Local validation used fake temporary TFRecords only:

```text
python3 -m py_compile scripts/thesis_make_exp008_depth_split.py scripts/thesis_check_exp008_tfrecord_split.py tests/test_exp008_depth_split.py
python3 -m unittest tests.test_exp008_depth_split
```

The tests confirm:

- the builder creates manifest/audit and split TFRecords on fake data;
- `--dry-run` does not write split TFRecords;
- the smoke checker can count split records without TensorFlow.
