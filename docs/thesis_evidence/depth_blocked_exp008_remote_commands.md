# EXP-008 Depth-Blocked Remote Commands

Generated: 2026-07-06. These commands are prepared for a future remote run. This task did not execute training and did not push.

## 0. Remote Branch Preparation

Run after the implementation commit is available on the remote or after the user explicitly approves pushing/fetching this branch.

```bash
ssh -o BatchMode=yes cement-server

cd /home/xiaoj/hal_azi
git fetch origin
git switch feature/thesis-depth-blocked-exp008
git pull --ff-only
git status -sb
```

If the branch is not on `origin` yet, do not improvise with rsync of large data. Either push the code branch after approval or apply the committed patch through Git.

## 1. Activate Environment

```bash
cd /home/xiaoj/hal_azi
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate hall
python --version
python - <<'PY'
import tensorflow as tf
print(tf.__version__)
PY
```

Expected environment evidence from prior audit:

```text
Python 3.10.18
TensorFlow 2.9.1
```

## 2. Preflight Existing EXP-008 Artifacts

```bash
cd /home/xiaoj/hal_azi

ls -lh data/processed/fft_regression/array_03/tfrecords/fft_regression_data.tfrecord
ls -lh data/processed/fft_regression/array_03/tfrecords/fft_regression_data.tfrecord.idx.pkl
ls -lh data/processed/fft_regression/array_03/processed_waveforms.pkl
ls -lh data/processed/fft_regression/array_03/cwt_images.h5
ls -lh data/processed/fft_regression/array_03/ground_truth_db_array_03.h5
```

## 3. Dry-Run Split

Dry-run writes only manifest/audit/overview files. It does not write train/val/test TFRecords.

```bash
cd /home/xiaoj/hal_azi

python scripts/thesis_make_exp008_depth_split.py \
  --dry-run \
  --output-dir output/thesis_depth_blocked/exp008/split_v001_dry_run \
  --split-preset depth_heldout_simple \
  --gap-ft 5.0
```

Inspect:

```bash
sed -n '1,220p' output/thesis_depth_blocked/exp008/split_v001_dry_run/leakage_audit.md
cat output/thesis_depth_blocked/exp008/split_v001_dry_run/depth_split_overview.csv
```

Expected audit conclusion for a usable split:

```text
depth_heldout_split_confirmed
```

## 4. Optional Small Smoke Split

This writes tiny split TFRecords under a dedicated smoke directory. It is not a thesis result.

```bash
cd /home/xiaoj/hal_azi

python scripts/thesis_make_exp008_depth_split.py \
  --output-dir output/thesis_depth_blocked/exp008/split_smoke_128 \
  --split-preset depth_heldout_simple \
  --gap-ft 0 \
  --max-records-for-smoke 128

python scripts/thesis_check_exp008_tfrecord_split.py \
  --split-dir output/thesis_depth_blocked/exp008/split_smoke_128 \
  --parse-one
```

If `--parse-one` reports TensorFlow shape parsing failure, stop and inspect before creating the full split.

## 5. Real Split

This creates split TFRecords and audit files. It still does not train.

```bash
cd /home/xiaoj/hal_azi

python scripts/thesis_make_exp008_depth_split.py \
  --output-dir output/thesis_depth_blocked/exp008/split_v001 \
  --split-preset depth_heldout_simple \
  --gap-ft 5.0
```

Do not use `--overwrite` unless intentionally replacing a previous failed split directory.

## 6. Smoke Check Real Split

```bash
cd /home/xiaoj/hal_azi

python scripts/thesis_check_exp008_tfrecord_split.py \
  --split-dir output/thesis_depth_blocked/exp008/split_v001 \
  --parse-one \
  --require-parse
```

Expected shape check:

```text
feature_shape = [150, 400, 8]
label_shape = [70, 30]
```

Expected count check:

- `train.tfrecord`, `val.tfrecord`, and `test.tfrecord` counts match `split_manifest.csv`;
- `leakage_audit.json` conclusion is `depth_heldout_split_confirmed`.

## 7. Future Training Command Placeholder

Do not run this in P1. It requires a follow-up training integration stage and explicit user approval.

```bash
# REQUIRES USER APPROVAL. DO NOT RUN IN P1.
python scripts/thesis_train_exp008_depth_blocked.py \
  --train-tfrecord output/thesis_depth_blocked/exp008/split_v001/train.tfrecord \
  --val-tfrecord output/thesis_depth_blocked/exp008/split_v001/val.tfrecord \
  --test-tfrecord output/thesis_depth_blocked/exp008/split_v001/test.tfrecord \
  --split-manifest output/thesis_depth_blocked/exp008/split_v001/split_manifest.csv \
  --output-dir output/thesis_depth_blocked/exp008/train_v001
```

The future training entry must not use random `validation_split`, `train_test_split`, or `take/skip` on a shuffled full dataset.
