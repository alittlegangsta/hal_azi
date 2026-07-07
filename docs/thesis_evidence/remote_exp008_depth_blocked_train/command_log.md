# EXP-008 Remote Command Log

## Smoke

```bash
python scripts/thesis_train_exp008_depth_blocked.py --split-dir output/thesis_depth_blocked/exp008/split_v001 --output-dir output/thesis_depth_blocked/exp008/train_smoke_v001 --epochs 1 --batch-size 8 --learning-rate 1e-4 --patience 1 --smoke --max-train-batches 2 --max-val-batches 1 --max-test-batches 1
```

Result: success, exit code 0.

## Full train_v001

```bash
python scripts/thesis_train_exp008_depth_blocked.py --split-dir output/thesis_depth_blocked/exp008/split_v001 --output-dir output/thesis_depth_blocked/exp008/train_v001 --epochs 80 --batch-size 8 --learning-rate 1e-4 --patience 10
```

Result: success, exit code 0. Training stopped at epoch 12 and restored epoch 2 weights.
