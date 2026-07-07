# EXP-008 Depth Split Leakage Audit

Generated: `2026-07-06T09:39:44+00:00`
Conclusion: `depth_heldout_split_confirmed`
Dry run: `True`

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
