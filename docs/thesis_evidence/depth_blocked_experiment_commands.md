# Depth-Blocked Supplement Experiment Commands

Generated: 2026-07-06. These are future command drafts. They were not executed in this task.

## Environment

Run on the remote server unless a local GPU/TensorFlow 2.9.1 environment is restored.

Evidence:

- remote repo: `/home/xiaoj/hal_azi`
- conda env: `hall`
- Python: `3.10.18`
- TensorFlow: `2.9.1`
- h5py: `3.14.0`
- source: `docs/thesis_evidence/remote_server_code_inventory.md`

Suggested shell preflight:

```bash
ssh -o BatchMode=yes cement-server 'cd /home/xiaoj/hal_azi && git status -sb && /usr/local/anaconda3/envs/hall/bin/python --version'
```

## Branch Preparation

Use a future implementation branch. Do not edit the thesis evidence branch for training code.

```bash
cd /home/xiaoj/hal_azi
git switch -c thesis/depth-blocked-supplement origin/percentage_label+FFT
```

If the branch already exists:

```bash
cd /home/xiaoj/hal_azi
git switch thesis/depth-blocked-supplement
```

## EXP-008 Must-Run: FFT Severity Regression

After implementing `scripts/thesis_create_depth_split_tfrecords.py` and split-aware train/eval options, create split TFRecords from the existing FFT regression TFRecord.

```bash
cd /home/xiaoj/hal_azi
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate hall

python scripts/thesis_create_depth_split_tfrecords.py \
  --experiment-id EXP-008 \
  --task-type fft_regression \
  --source-tfrecord data/processed/fft_regression/array_03/tfrecords/fft_regression_data.tfrecord \
  --source-idx-pkl data/processed/fft_regression/array_03/tfrecords/fft_regression_data.tfrecord.idx.pkl \
  --processed-waveforms-pkl data/processed/fft_regression/array_03/processed_waveforms.pkl \
  --ground-truth-h5 data/processed/fft_regression/array_03/ground_truth_db_array_03.h5 \
  --output-dir data/processed/thesis_depth_blocked/EXP-008/array_03/tfrecords \
  --split-mode depth_heldout_simple \
  --train-depth-min 2732.44 \
  --train-depth-max 3707.00 \
  --val-depth-min 3712.00 \
  --val-depth-max 3916.00 \
  --test-depth-min 3921.00 \
  --test-depth-max 4131.75 \
  --guard-gap-ft 5.0
```

Train:

```bash
cd /home/xiaoj/hal_azi
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate hall

ARRAY_ID=3 TASK_TYPE=fft_regression SPLIT_MODE=depth_heldout \
python src/modeling/train.py \
  --task-type fft_regression \
  --split-mode depth_heldout \
  --train-tfrecord data/processed/thesis_depth_blocked/EXP-008/array_03/tfrecords/train.tfrecord \
  --val-tfrecord data/processed/thesis_depth_blocked/EXP-008/array_03/tfrecords/val.tfrecord \
  --split-manifest data/processed/thesis_depth_blocked/EXP-008/array_03/tfrecords/split_manifest.csv \
  --output-dir output/thesis_depth_blocked/EXP-008/array_03 \
  --run-name depth_heldout_simple
```

Evaluate heldout test only:

```bash
cd /home/xiaoj/hal_azi
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate hall

ARRAY_ID=3 TASK_TYPE=fft_regression SPLIT_MODE=depth_heldout \
python src/interpretation/run_analysis_regressor.py \
  --task-type fft_regression \
  --model-path output/thesis_depth_blocked/EXP-008/array_03/models/best_fft_regressor_model.h5 \
  --eval-tfrecord data/processed/thesis_depth_blocked/EXP-008/array_03/tfrecords/test.tfrecord \
  --split-manifest data/processed/thesis_depth_blocked/EXP-008/array_03/tfrecords/split_manifest.csv \
  --output-dir output/thesis_depth_blocked/EXP-008/array_03/results/test_depth_heldout
```

## EXP-007 Recommended: 1D Percentage Regression

Preferred exact-code route: start from `origin/1D+percentage_Label` and backport only split manifest support. This best preserves the original EXP-007 method.

```bash
cd /home/xiaoj/hal_azi
git switch -c thesis/depth-blocked-EXP-007 origin/1D+percentage_Label
```

Create split TFRecords from the existing profile regression TFRecord. Because the old EXP-007 branch did not save `.idx.pkl`, the split creation script must reconstruct processed record order from `processed_waveforms.pkl['sonic_depths']` and `ground_truth_db_array_03.h5['path_data']`.

```bash
cd /home/xiaoj/hal_azi
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate hall

python scripts/thesis_create_depth_split_tfrecords.py \
  --experiment-id EXP-007 \
  --task-type profile_regression \
  --source-tfrecord data/processed/image_translation/array_03/tfrecords/profile_regression_data.tfrecord \
  --processed-waveforms-pkl data/processed/image_translation/array_03/processed_waveforms.pkl \
  --ground-truth-h5 data/processed/image_translation/array_03/ground_truth_db_array_03.h5 \
  --output-dir data/processed/thesis_depth_blocked/EXP-007/array_03/tfrecords \
  --split-mode depth_heldout_simple \
  --train-depth-min 2732.44 \
  --train-depth-max 3707.00 \
  --val-depth-min 3712.00 \
  --val-depth-max 3916.00 \
  --test-depth-min 3921.00 \
  --test-depth-max 4131.75 \
  --guard-gap-ft 5.0
```

Train:

```bash
cd /home/xiaoj/hal_azi
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate hall

ARRAY_ID=3 TASK_TYPE=image_translation SPLIT_MODE=depth_heldout \
python src/modeling/train.py \
  --task-type profile_regression \
  --split-mode depth_heldout \
  --train-tfrecord data/processed/thesis_depth_blocked/EXP-007/array_03/tfrecords/train.tfrecord \
  --val-tfrecord data/processed/thesis_depth_blocked/EXP-007/array_03/tfrecords/val.tfrecord \
  --split-manifest data/processed/thesis_depth_blocked/EXP-007/array_03/tfrecords/split_manifest.csv \
  --output-dir output/thesis_depth_blocked/EXP-007/array_03 \
  --run-name depth_heldout_simple
```

Evaluate:

```bash
cd /home/xiaoj/hal_azi
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate hall

ARRAY_ID=3 TASK_TYPE=image_translation SPLIT_MODE=depth_heldout \
python src/interpretation/run_analysis_regressor.py \
  --task-type profile_regression \
  --model-path output/thesis_depth_blocked/EXP-007/array_03/models/best_advanced_regressor_model.h5 \
  --eval-tfrecord data/processed/thesis_depth_blocked/EXP-007/array_03/tfrecords/test.tfrecord \
  --split-manifest data/processed/thesis_depth_blocked/EXP-007/array_03/tfrecords/split_manifest.csv \
  --output-dir output/thesis_depth_blocked/EXP-007/array_03/results/test_depth_heldout
```

Alternative unified-code route: implement env-configurable `TASK_TYPE` on `origin/percentage_label+FFT` and run `TASK_TYPE=image_translation`. This is simpler operationally, but if the artifact-mask logic differs from the original EXP-007 branch, record it as `EXP-007-compatible supplement` rather than exact EXP-007 reproduction.

## EXP-006 Optional: Binary CWT Learnability Baseline

Run only if the thesis needs a depth-heldout baseline table. The remote current processed listing did not show `classification_data.tfrecord`, so this path may require generating a classification TFRecord from existing CWT/HDF5/PKL on `origin/test_relativity`.

```bash
cd /home/xiaoj/hal_azi
git switch -c thesis/depth-blocked-EXP-006 origin/test_relativity
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate hall

python scripts/thesis_create_depth_split_tfrecords.py \
  --experiment-id EXP-006 \
  --task-type binary_classification \
  --generate-labels-from-existing-cwt \
  --cwt-h5 data/processed/image_translation/array_03/cwt_images.h5 \
  --processed-waveforms-pkl data/processed/image_translation/array_03/processed_waveforms.pkl \
  --ground-truth-h5 data/processed/image_translation/array_03/ground_truth_db_array_03.h5 \
  --output-dir data/processed/thesis_depth_blocked/EXP-006/array_03/tfrecords \
  --split-mode depth_heldout_simple \
  --train-depth-min 2732.44 \
  --train-depth-max 3707.00 \
  --val-depth-min 3712.00 \
  --val-depth-max 3916.00 \
  --test-depth-min 3921.00 \
  --test-depth-max 4131.75 \
  --guard-gap-ft 5.0
```

Do not run the old `src/data_processing/create_tfrecords.py` for EXP-006 unless it has first been changed to accept a safe output directory. The optional baseline should not overwrite or add files inside the old random-split processed directory.

Training and evaluation commands should mirror EXP-008/EXP-007 but compile classification metrics: AUC, accuracy, precision, recall, F1, and confusion matrix.

## Optional K-Fold Commands

Only after EXP-008 simple heldout is complete:

```bash
python scripts/thesis_create_depth_split_tfrecords.py \
  --experiment-id EXP-008 \
  --task-type fft_regression \
  --source-tfrecord data/processed/fft_regression/array_03/tfrecords/fft_regression_data.tfrecord \
  --source-idx-pkl data/processed/fft_regression/array_03/tfrecords/fft_regression_data.tfrecord.idx.pkl \
  --processed-waveforms-pkl data/processed/fft_regression/array_03/processed_waveforms.pkl \
  --ground-truth-h5 data/processed/fft_regression/array_03/ground_truth_db_array_03.h5 \
  --output-dir data/processed/thesis_depth_blocked/EXP-008_kfold/fold_0/array_03/tfrecords \
  --split-mode depth_blocked_kfold \
  --kfold-blocks 5 \
  --test-block 0 \
  --val-block 1 \
  --guard-gap-ft 5.0
```

Repeat for folds only if the first heldout result is thesis-critical and time allows.
