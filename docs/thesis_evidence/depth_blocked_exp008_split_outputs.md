# EXP-008 Depth Split Outputs

Generated: 2026-07-06. This document describes outputs produced by the split tooling. It does not contain model performance metrics.

## Default Output Directory

```text
output/thesis_depth_blocked/exp008/split_v001/
```

The split builder refuses to write under:

```text
/mnt/c/Users/Administrator/Desktop/Hal/results
```

## Output Files

| File | Created in dry-run | Created in real split | Purpose |
| --- | --- | --- | --- |
| `train.tfrecord` | no | yes | depth-heldout training records |
| `val.tfrecord` | no | yes | contiguous validation records |
| `test.tfrecord` | no | yes | final heldout test records |
| `split_manifest.csv` | yes | yes | one row per source record with record index, sample index, depth, split, and evidence paths |
| `split_manifest.json` | yes | yes | full manifest and run metadata |
| `leakage_audit.json` | yes | yes | machine-readable leakage audit |
| `leakage_audit.md` | yes | yes | thesis-readable leakage audit summary |
| `depth_split_overview.csv` | yes | yes | split counts and depth min/max/median |
| `smoke_check.json` | no | after smoke checker | count and optional TensorFlow shape check results |

## Manifest Columns

`split_manifest.csv` fields:

| Field | Meaning |
| --- | --- |
| `experiment_id` | fixed as `EXP-008` unless overridden |
| `record_index` | zero-based index in source `fft_regression_data.tfrecord` |
| `sample_index` | original sample index from `fft_regression_data.tfrecord.idx.pkl` |
| `depth_ft` | sonic depth in ft from `processed_waveforms.pkl['sonic_depths']` |
| `split` | `train`, `val`, `test`, `dropped_gap`, or another dropped reason |
| `split_reason` | rule used to assign or drop the record |
| `source_tfrecord` | source EXP-008 TFRecord path |
| `source_idx_pkl` | source `.idx.pkl` path |
| `source_processed_waveforms_pkl` | source depth mapping path |
| `created_from_branch_or_commit` | default `origin/percentage_label+FFT@7ba021c...` |

## Leakage Audit Fields

`leakage_audit.json` includes:

- `stats.train`, `stats.val`, `stats.test`: count, min depth, max depth, median depth;
- `boundary_distances`: nearest kept depth distance between split depth intervals;
- `record_index_intersections`: sample overlap checks between split pairs;
- `mutually_exclusive_record_sets`;
- `dropped_buffer_count`;
- `dropped_other_count`;
- `warnings`;
- `conclusion`.

Valid final split conclusion:

```text
depth_heldout_split_confirmed
```

If the conclusion is `needs_manual_verification`, do not train until the warning has been reviewed.

## Smoke Check Output

`scripts/thesis_check_exp008_tfrecord_split.py` writes:

```text
output/thesis_depth_blocked/exp008/split_v001/smoke_check.json
```

It checks:

- each split TFRecord exists;
- train/val/test TFRecord record counts;
- counts match `split_manifest.csv`;
- `leakage_audit.json` can be loaded;
- optionally, one sample per split parses with TensorFlow and has:
  - feature shape `[150, 400, 8]`;
  - label shape `[70, 30]`.

The shape parse is optional because local documentation environments may not have TensorFlow. On the remote `hall` environment, run with:

```bash
python scripts/thesis_check_exp008_tfrecord_split.py \
  --split-dir output/thesis_depth_blocked/exp008/split_v001 \
  --parse-one \
  --require-parse
```

## Interpretation

These outputs are split evidence only. They are not model performance evidence. After a future approved training run, add new depth-heldout metrics to:

- `docs/thesis_evidence/unified_metrics_table.csv`
- `docs/thesis_evidence/metric_source_traceability.csv`
- `docs/thesis_evidence/split_forensic_audit.csv`
- `docs/thesis_evidence/leakage_risk_report.md`
- `docs/thesis_evidence/thesis_claims_traceability.md`

Do not replace old random-split EXP-008 metrics; add new rows with `split_type=depth_heldout_simple`.

<!-- EXP008_DEPTH_HELDOUT_ARCHIVE_START -->
## Remote Split Evidence Archive (2026-07-07)

Small split evidence files from remote `split_v001` were archived under:

```text
docs/thesis_evidence/remote_exp008_split_v001/
```

Archived files include `leakage_audit.md/json`, `depth_split_overview.csv`, `split_manifest.json/csv`, `smoke_check.json`, and dry-run audit/overview files. The full train/val/test TFRecords were not copied. The confirmed split has train/val/test counts 1984/416/423 and audit conclusion `depth_heldout_split_confirmed`.
<!-- EXP008_DEPTH_HELDOUT_ARCHIVE_END -->
