# Depth-Blocked Expected Outputs

Generated: 2026-07-06. This file lists expected artifacts only. No training was run, and it does not contain expected metric values.

## Directory Layout

Recommended safe output layout on the remote server:

```text
/home/xiaoj/hal_azi/
  data/processed/thesis_depth_blocked/
    EXP-008/array_03/tfrecords/
      train.tfrecord
      val.tfrecord
      test.tfrecord
      split_manifest.csv
      split_manifest.json
      leakage_audit.json
      split_summary.md
    EXP-007/array_03/tfrecords/
      train.tfrecord
      val.tfrecord
      test.tfrecord
      split_manifest.csv
      split_manifest.json
      leakage_audit.json
      split_summary.md
  output/thesis_depth_blocked/
    EXP-008/array_03/
      logs/
      models/
      results/test_depth_heldout/
      train_config.json
      source_evidence.json
    EXP-007/array_03/
      logs/
      models/
      results/test_depth_heldout/
      train_config.json
      source_evidence.json
```

Do not write supplement artifacts under `/mnt/c/Users/Administrator/Desktop/Hal/results`.

## Split Manifest Outputs

Each experiment should produce:

| File | Required content |
| --- | --- |
| `split_manifest.csv` | one row per source TFRecord record; `experiment_id`, `record_index`, `sample_index`, `sonic_depth`, `split`, `source_tfrecord`, `source_processed_waveforms_pkl`, `source_idx_pkl`, `guard_gap_ft`, `branch_or_commit` |
| `split_manifest.json` | same metadata plus run configuration and source checksums if cheap to compute |
| `leakage_audit.json` | train/val/test depth min/max, counts, excluded guard count, depth overlap status, adjacent-depth leakage risk status |
| `split_summary.md` | human-readable summary for thesis methods section |

Required leakage status after successful split:

```text
depth_blocked_split_confirmed
```

If the script cannot map TFRecord records to depths:

```text
needs_manual_verification
```

## EXP-008 Expected Outputs

Source evidence:

- branch: `origin/percentage_label+FFT`
- source TFRecord: `data/processed/fft_regression/array_03/tfrecords/fft_regression_data.tfrecord`
- source index: `data/processed/fft_regression/array_03/tfrecords/fft_regression_data.tfrecord.idx.pkl`
- source depth file: `data/processed/fft_regression/array_03/processed_waveforms.pkl`

Expected model outputs:

| Path | Notes |
| --- | --- |
| `output/thesis_depth_blocked/EXP-008/array_03/models/best_fft_regressor_model.h5` | best validation model |
| `output/thesis_depth_blocked/EXP-008/array_03/logs/training_history_fft.pkl` | train/val curve history |
| `output/thesis_depth_blocked/EXP-008/array_03/logs/train/` | TensorBoard train events |
| `output/thesis_depth_blocked/EXP-008/array_03/logs/validation/` | TensorBoard validation events |
| `output/thesis_depth_blocked/EXP-008/array_03/train_config.json` | command, seed, branch, split manifest path |

Expected heldout test outputs:

| Path | Notes |
| --- | --- |
| `results/test_depth_heldout/test_metrics.json` | MAE, RMSE, Huber loss, R2/Pearson/Spearman if computed |
| `results/test_depth_heldout/test_metrics.csv` | long-form metric table |
| `results/test_depth_heldout/predictions_with_depth.csv` | record index, sample index, sonic depth, true severity, predicted severity, error |
| `results/test_depth_heldout/severity_group_metrics.csv` | group-wise MAE/RMSE/count |
| `results/test_depth_heldout/fft_coefficient_error.csv` | coefficient-wise error summary |
| `results/test_depth_heldout/prediction_scatter.png` | heldout scatter plot |
| `results/test_depth_heldout/depth_log_comparison.png` | heldout depth curve |
| `results/test_depth_heldout/fft_spectrum_image_comparison.png` | heldout FFT spectrum comparison |

Thesis use if successful:

- main quantitative result for severity + FFT magnitude/log label + EfficientNet;
- old random-split EXP-008 remains exploratory development evidence;
- report depth-heldout split as the primary performance table.

## EXP-007 Expected Outputs

Source evidence:

- branch: `origin/1D+percentage_Label`
- source TFRecord: `data/processed/image_translation/array_03/tfrecords/profile_regression_data.tfrecord`
- source depth file: `data/processed/image_translation/array_03/processed_waveforms.pkl`
- source ground truth H5: `data/processed/image_translation/array_03/ground_truth_db_array_03.h5`

Expected model outputs:

| Path | Notes |
| --- | --- |
| `output/thesis_depth_blocked/EXP-007/array_03/models/best_advanced_regressor_model.h5` | best validation model |
| `output/thesis_depth_blocked/EXP-007/array_03/logs/training_history_advanced.pkl` | train/val curve history |
| `output/thesis_depth_blocked/EXP-007/array_03/train_config.json` | command, seed, branch, split manifest path |

Expected heldout test outputs:

| Path | Notes |
| --- | --- |
| `results/test_depth_heldout/test_metrics.json` | MAE, RMSE, Huber loss, R2/Pearson/Spearman if computed |
| `results/test_depth_heldout/test_metrics.csv` | long-form metric table |
| `results/test_depth_heldout/predictions_with_depth.csv` | record index, sample index, sonic depth, true profile severity, predicted profile severity, error |
| `results/test_depth_heldout/severity_group_metrics.csv` | group-wise MAE/RMSE/count |
| `results/test_depth_heldout/prediction_scatter.png` | heldout scatter plot |
| `results/test_depth_heldout/depth_log_comparison.png` | heldout depth profile comparison |
| `results/test_depth_heldout/error_distribution_by_category.png` | heldout group error plot |

Thesis use:

- fallback mainline if EXP-008 depth-heldout is weak;
- conservative depth-profile result if FFT route is too unstable;
- still does not recover azimuthal localization.

## EXP-006 Optional Outputs

Source evidence:

- branch: `origin/test_relativity`
- source processed artifacts: `data/processed/image_translation/array_03/cwt_images.h5`, `processed_waveforms.pkl`, and `ground_truth_db_array_03.h5`
- generated split TFRecords should be written under `data/processed/thesis_depth_blocked/EXP-006/array_03/tfrecords/`, not under the old random-split processed directory

Expected outputs:

| Path | Notes |
| --- | --- |
| `output/thesis_depth_blocked/EXP-006/array_03/models/best_classifier_model.h5` | best validation AUC model |
| `output/thesis_depth_blocked/EXP-006/array_03/results/test_depth_heldout/classification_metrics.json` | AUC, accuracy, precision, recall, F1 |
| `output/thesis_depth_blocked/EXP-006/array_03/results/test_depth_heldout/confusion_matrix.csv` | confusion matrix |
| `output/thesis_depth_blocked/EXP-006/array_03/results/test_depth_heldout/classification_report.txt` | sklearn-style report if available |

Thesis use:

- optional depth-heldout baseline for CWT learnability;
- not a replacement for severity regression results.

## Evidence Updates After Future Runs

After the supplement is actually run, update:

- `docs/thesis_evidence/unified_metrics_table.csv`
- `docs/thesis_evidence/metric_source_traceability.csv`
- `docs/thesis_evidence/split_forensic_audit.csv`
- `docs/thesis_evidence/leakage_risk_report.md`
- `docs/thesis_evidence/experiment_inventory.md`
- `docs/thesis_evidence/thesis_claims_traceability.md`
- `docs/thesis_evidence/figure_shortlist_for_thesis.csv`

Do not replace old exploratory metrics. Add new rows with:

```text
split_type=depth_heldout_simple
leakage_risk=depth_blocked_split_confirmed
thesis_use=main_result
```
