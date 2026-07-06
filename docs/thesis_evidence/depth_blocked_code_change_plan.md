# Depth-Blocked Code Change Plan

Generated: 2026-07-06. This plan describes future minimal code changes. No code changes were made in this task.

## Current Limitation

Current branch snapshots split data after `dataset.shuffle(...)` and after batching:

- EXP-008: `origin/percentage_label+FFT:src/modeling/dataset.py`, `origin/percentage_label+FFT:src/modeling/train.py`
- EXP-007: `origin/1D+percentage_Label:src/modeling/dataset.py`, `origin/1D+percentage_Label:src/modeling/train.py`
- EXP-006: `origin/test_relativity:src/modeling/dataset.py`, `origin/test_relativity:src/modeling/train.py`

This creates random validation at the batch/sample level and does not preserve contiguous depth holdout. The old TFRecord examples do not include depth as a feature, so depth split must be applied before training by creating split-specific TFRecords or by adding a depth-aware dataset filter.

## Minimal Design Choice

Use split-specific TFRecords. This requires the fewest changes to the training code and preserves old TFRecord contents. The split script should copy serialized examples from the existing source TFRecord into train/val/test files based on depth metadata.

Benefits:

- does not rerun raw preprocessing
- does not recompute CWT
- does not modify old TFRecords
- does not touch Windows results
- makes split manifest auditable
- avoids changing old random-split behavior

## New Script

Add:

```text
scripts/thesis_create_depth_split_tfrecords.py
```

Required behavior:

1. Read source TFRecord as raw serialized examples.
2. Resolve record depth:
   - if `--source-idx-pkl` is present, read `processed_indices` and map each record to `processed_waveforms.pkl['sonic_depths'][processed_index]`;
   - otherwise reconstruct processed indices by iterating `processed_waveforms.pkl['sonic_depths']` and keeping only depths whose `str(depth).replace('.', '_')` key exists in `ground_truth_db_array_03.h5['path_data']`.
3. Assign each record to train, val, test, or guard/excluded according to depth ranges.
4. Write only non-excluded serialized examples to:
   - `train.tfrecord`
   - `val.tfrecord`
   - `test.tfrecord`
5. Write:
   - `split_manifest.csv`
   - `split_manifest.json`
   - `leakage_audit.json`
   - `split_summary.md`

Required validations:

- number of TFRecord records equals number of resolved record depths;
- train/val/test depth ranges do not overlap;
- guard gap is non-empty if configured;
- no sample index appears in more than one split;
- train/val/test counts are nonzero;
- write source paths and Git branch/commit into manifest.

Suggested arguments:

```text
--experiment-id
--task-type
--source-tfrecord
--source-idx-pkl
--processed-waveforms-pkl
--ground-truth-h5
--output-dir
--split-mode depth_heldout_simple|depth_blocked_kfold
--train-depth-min
--train-depth-max
--val-depth-min
--val-depth-max
--test-depth-min
--test-depth-max
--guard-gap-ft
--kfold-blocks
--test-block
--val-block
```

## Dataset Code

Modify:

```text
src/modeling/dataset.py
```

Minimal changes:

- add `seed=None` and `reshuffle_each_iteration=True` arguments;
- support `is_training=False` without shuffle;
- keep old default behavior for random-split runs;
- for depth-heldout runs, use pre-split TFRecord files and shuffle only `train.tfrecord`.

Suggested signature:

```python
def create_dataset(
    tfrecord_path,
    batch_size,
    shuffle_buffer_size=1024,
    is_training=True,
    parse_fn=None,
    seed=None,
    reshuffle_each_iteration=True,
):
    ...
```

## Training Code

Modify:

```text
src/modeling/train.py
```

Minimal changes:

- add argparse options:
  - `--task-type`
  - `--split-mode random|depth_heldout`
  - `--train-tfrecord`
  - `--val-tfrecord`
  - `--split-manifest`
  - `--output-dir`
  - `--run-name`
  - `--seed`
- preserve old behavior when `--split-mode random` or no arguments are supplied;
- for `--split-mode depth_heldout`, load train/val datasets from separate TFRecords and do not use `take/skip`;
- copy or summarize the split manifest into the output directory;
- save `train_config.json` with branch/commit, command, task type, seed, and source TFRecord paths;
- save `training_history.pkl` under the new output directory.

For EXP-008:

- branch: `origin/percentage_label+FFT`
- parse function: existing `_parse_regression_tfrecord_fn`
- model: existing `build_advanced_profile_regressor`
- output head: `(MAX_PATH_DEPTH_POINTS, FFT_COEFFICIENTS)` when `task_type=fft_regression`
- loss: existing Huber loss
- metric: MAE

For EXP-007:

- branch: preferred exact `origin/1D+percentage_Label`
- parse function: existing `_parse_regression_tfrecord_fn`
- model: existing `build_advanced_profile_regressor`
- output head: `(MAX_PATH_DEPTH_POINTS,)`
- loss: existing Huber loss
- metric: MAE

## Config Code

Modify:

```text
config.py
```

Minimal changes:

- allow `TASK_TYPE` to be read from environment:

```python
TASK_TYPE = os.getenv("TASK_TYPE", "fft_regression")
```

- optionally allow output root override:

```python
OUTPUT_BASE_DIR = os.getenv("OUTPUT_BASE_DIR", "output")
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, TASK_TYPE, f"array_{str(ARRAY_ID).zfill(2)}")
```

For supplement runs, prefer explicit `--output-dir` in `train.py` so old output paths are never overwritten.

## Evaluation Code

Modify:

```text
src/interpretation/run_analysis_regressor.py
```

Minimal changes:

- add argparse options:
  - `--task-type`
  - `--model-path`
  - `--eval-tfrecord`
  - `--split-manifest`
  - `--output-dir`
  - `--metrics-output`
- evaluate only records from `test.tfrecord`;
- merge predictions with `split_manifest.csv` by `record_index` or evaluation order;
- save:
  - `test_metrics.json`
  - `test_metrics.csv`
  - `severity_group_metrics.csv`
  - `predictions_with_depth.csv`
  - `depth_log_comparison.png`
  - `prediction_scatter.png`
  - for EXP-008, `fft_coefficient_error.csv` and `fft_spectrum_image_comparison.png`

Do not use validation-set samples for final figures after depth-heldout evaluation exists.

## Optional Classification Code

For EXP-006, modify the `origin/test_relativity` branch analogously:

- `src/modeling/train.py`: accept train/val TFRecords and output directory;
- `src/interpretation/run_analysis_classification.py`: evaluate test TFRecord only and save AUC, accuracy, precision, recall, F1, confusion matrix, and classification report;
- `src/data_processing/create_tfrecords.py`: optionally save `.idx.pkl` for classification TFRecord generation.

This is optional and should not delay EXP-008.

## Non-Goals

Do not implement in the supplement:

- GAN/two-channel retraining
- dual-channel metadata fusion retraining
- eccentricity pre-correction retraining
- sample weights/asymmetric loss rerun
- frequency-weighted loss rerun
- large refactor of the data pipeline
- changes to raw data or Windows result evidence

## Acceptance Criteria

The supplement code is acceptable when:

- old random-split command still works;
- `split_mode=depth_heldout` never calls `take/skip` on a shuffled full dataset;
- train/val/test TFRecords are distinct files;
- `split_manifest.csv` records source path, sample index, record index, depth, and split;
- `leakage_audit.json` confirms no depth overlap and reports guard gaps;
- output path is under `/home/xiaoj/hal_azi/output/thesis_depth_blocked/...`;
- final test metrics are computed on `test.tfrecord` only.
