# Leakage Risk Report

Generated: 2026-07-06.

## Bottom Line

EXP-008, EXP-007, EXP-006, EXP-002, EXP-003, and EXP-014 are marked `random_split_depth_leakage_risk`. This is stronger than the prior `split_unknown`: the split code is visible and it is not depth-blocked.

## Evidence

| evidence | source | detail |
| --- | --- | --- |
| shuffle before split | src/modeling/dataset.py on relevant branches | `dataset.shuffle(buffer_size=1024)` is applied when `is_training=True`. |
| take/skip split | src/modeling/train.py on relevant branches | `train_dataset = full_dataset.take(train_size)` and `val_dataset = full_dataset.skip(train_size)`. |
| no seed/grouping | rg scan of repo and branch snapshots | No `GroupKFold`, depth block, heldout depth, train_idx/val_idx/test_idx artifact, or fixed shuffle seed found. |
| depth metadata not present in results | find results/repo for idx/depth/split artifacts | Only image depth plots were found; no usable depth array/index split artifact was located. |

## Thesis Implications

| experiment_id | implication |
| --- | --- |
| EXP-008 | Can be thesis mainline method candidate, but not a final generalization result until depth-blocked split is added or recovered. |
| EXP-007 | Can be fallback/exploratory mainline and severity-error analysis; still needs depth-blocked validation for final performance claims. |
| EXP-006 | Safe to claim CWT contains learnable channeling signal under random validation; unsafe to claim depth-heldout generalization. |
| EXP-002/EXP-003/EXP-014 | Use as ablation/failed-attempt evidence only; metrics are exploratory under random split. |
| EXP-004/EXP-005 | Train-only/failed route; do not present as validation/test metric. |
