# EXP-007 Depth Split Leakage Audit

Generated remotely: `2026-07-07`. Local archive reconstructed from remote command output because later SSH/SCP escalation was denied by the execution environment usage limit.

Conclusion: `depth_heldout_split_confirmed`

## Record-Depth Mapping

- method: `sonic_depths_order_filtered_by_ground_truth_path_data_key`
- source TFRecord: `/home/xiaoj/hal_azi/data/processed/image_translation/array_03/tfrecords/profile_regression_data.tfrecord`
- source metadata: `/home/xiaoj/hal_azi/data/processed/image_translation/array_03/processed_waveforms.pkl`
- source ground truth: `/home/xiaoj/hal_azi/data/processed/image_translation/array_03/ground_truth_db_array_03.h5`
- sonic_depths_len: `2846`
- path_data_key_count: `2842`
- matched_count: `2842`
- missing_count: `4`

## Split Counts And Depth Ranges

| split | count | min_depth_ft | max_depth_ft | median_depth_ft |
| --- | ---: | ---: | ---: | ---: |
| train | 1984 | 2732.4396282122984 | 3715.852672292335 | 3232.6427716114194 |
| val | 416 | 3721.118907594681 | 3917.68482944096 | 3819.798245939437 |
| test | 423 | 3922.7850794766823 | 4129.7434705787255 | 4025.08450001953 |

## Boundary Distances

| boundary | nearest_kept_depth_distance_ft |
| --- | ---: |
| train_val_ft | 5.266235302346104 |
| val_test_ft | 5.100250035722183 |
| train_test_ft | 206.93240718434754 |

## Dropped Records

- dropped_buffer_count: `19`
- dropped_other_count: `0`

## Warnings

- none

## Smoke Check

Remote smoke check status: `pass`.

| split | TFRecord count | manifest count | parsed feature shape | parsed label shape |
| --- | ---: | ---: | --- | --- |
| train | 1984 | 1984 | `(150, 400, 8)` | `(70,)` |
| val | 416 | 416 | `(150, 400, 8)` | `(70,)` |
| test | 423 | 423 | `(150, 400, 8)` | `(70,)` |
