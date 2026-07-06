# TensorBoard Event Parse Report

Generated: 2026-07-06.

## Result

TensorBoard parsing was not performed because local Python cannot import `tensorboard.backend.event_processing.event_accumulator.EventAccumulator`.

## What Was Generated

- `scripts/thesis_parse_tensorboard_events.py`: read-only parser to run after activating an environment with TensorBoard.
- `docs/thesis_evidence/tensorboard_event_files_to_parse.csv`: 72 event files with experiment mapping and train/validation folder inference.
- `docs/thesis_evidence/tensorboard_scalar_metrics.csv`: header-only table; no metrics guessed.

## Command To Run Later

```bash
python scripts/thesis_parse_tensorboard_events.py --event-list docs/thesis_evidence/tensorboard_event_files_to_parse.csv --output docs/thesis_evidence/tensorboard_scalar_metrics.csv
```

## Event Files By Experiment

| experiment_id | experiment_name | run_split | event_file_count |
| --- | --- | --- | --- |
| EXP-002 | FFT log-label image translation | train | 1 |
| EXP-002 | FFT log-label image translation | validation | 1 |
| EXP-003 | FFT high-frequency weighted loss | train | 1 |
| EXP-003 | FFT high-frequency weighted loss | validation | 1 |
| EXP-005 | Two-channel binary label and focal-loss/overfit test | train | 3 |
| EXP-005 | Two-channel binary label and focal-loss/overfit test | validation | 1 |
| EXP-006 | CNN binary classification: CWT-label relationship test | train | 3 |
| EXP-006 | CNN binary classification: CWT-label relationship test | validation | 1 |
| EXP-007 | 1D percentage label profile regression | train | 18 |
| EXP-007 | 1D percentage label profile regression | validation | 12 |
| EXP-008 | EfficientNet FFT severity regression | train | 8 |
| EXP-008 | EfficientNet FFT severity regression | validation | 8 |
| EXP-014 | 1D percentage label + sample weights + asymmetric loss failed attempt | train | 8 |
| EXP-014 | 1D percentage label + sample weights + asymmetric loss failed attempt | validation | 6 |
