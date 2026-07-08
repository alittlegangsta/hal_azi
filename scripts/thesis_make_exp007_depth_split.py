#!/usr/bin/env python3
"""Build depth-heldout TFRecord splits for EXP-007 without training.

EXP-007 uses the 1D percentage label profile saved in
``profile_regression_data.tfrecord``. The original branch did not save an
``.idx.pkl`` file, so this script reconstructs record order with the exact
old TFRecord creation rule: iterate ``processed_waveforms.pkl['sonic_depths']``
in order and keep only depths whose string key exists in
``ground_truth_db_array_03.h5['path_data']``.

The script never modifies source TFRecords, raw data, processed data, or the
Windows result evidence directory. It only writes split outputs under the
requested output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import statistics
import struct
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_SOURCE_TFRECORD = Path("data/processed/image_translation/array_03/tfrecords/profile_regression_data.tfrecord")
DEFAULT_PROCESSED_WAVEFORMS_PKL = Path("data/processed/image_translation/array_03/processed_waveforms.pkl")
DEFAULT_GROUND_TRUTH_H5 = Path("data/processed/image_translation/array_03/ground_truth_db_array_03.h5")
DEFAULT_OUTPUT_DIR = Path("output/thesis_depth_blocked/exp007/split_v001")
WINDOWS_RESULTS_ROOT = Path("/mnt/c/Users/Administrator/Desktop/Hal/results")
DEFAULT_GIT_EVIDENCE = "origin/1D+percentage_Label@e9739c8c3fc4d53e1af63dafa84e29e00b42e9c4"

MANIFEST_FIELDS = [
    "experiment_id",
    "record_index",
    "sample_index",
    "depth_ft",
    "split",
    "split_reason",
    "source_tfrecord",
    "source_processed_waveforms_pkl",
    "source_ground_truth_h5",
    "record_depth_mapping_method",
    "created_from_branch_or_commit",
]


@dataclass
class RecordMeta:
    record_index: int
    sample_index: int
    depth_ft: float
    split: str = "unassigned"
    split_reason: str = ""


@dataclass
class SplitPlan:
    records: list[RecordMeta]
    boundary_depths: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_non_strict(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def ensure_safe_output_dir(output_dir: Path, source_tfrecord: Path) -> None:
    resolved_output = resolve_non_strict(output_dir)
    resolved_results = resolve_non_strict(WINDOWS_RESULTS_ROOT)
    resolved_source_parent = resolve_non_strict(source_tfrecord).parent
    if resolved_output == resolved_results or resolved_results in resolved_output.parents:
        raise ValueError(f"Refusing to write inside Windows results evidence directory: {resolved_output}")
    if resolved_output == resolved_source_parent:
        raise ValueError(f"Refusing to write split outputs into source TFRecord directory: {resolved_output}")


def read_exact(handle, size: int, context: str) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"Unexpected EOF while reading {context}: expected {size} bytes, got {len(data)}")
    return data


def iter_tfrecord_full_records(path: Path, max_records: int | None = None) -> Iterable[tuple[int, bytes]]:
    """Yield ``(record_index, full_record_bytes)`` without parsing TensorFlow examples."""
    with path.open("rb") as handle:
        record_index = 0
        while True:
            length_bytes = handle.read(8)
            if not length_bytes:
                break
            if len(length_bytes) != 8:
                raise ValueError(f"Corrupt TFRecord length header at record {record_index}")
            length = struct.unpack("<Q", length_bytes)[0]
            length_crc = read_exact(handle, 4, f"length crc for record {record_index}")
            payload = read_exact(handle, length, f"payload for record {record_index}")
            data_crc = read_exact(handle, 4, f"data crc for record {record_index}")
            yield record_index, length_bytes + length_crc + payload + data_crc
            record_index += 1
            if max_records is not None and record_index >= max_records:
                break


def count_tfrecord_records(path: Path, max_records: int | None = None) -> int:
    return sum(1 for _ in iter_tfrecord_full_records(path, max_records=max_records))


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def depth_to_path_key(depth_ft: float) -> str:
    return str(float(depth_ft)).replace(".", "_")


def reconstruct_processed_indices(processed_waveforms_pkl: Path, ground_truth_h5: Path) -> tuple[list[int], list[float], dict]:
    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"h5py is required to inspect {ground_truth_h5}: {type(exc).__name__}: {exc}") from exc

    waveform_data = load_pickle(processed_waveforms_pkl)
    if not isinstance(waveform_data, dict):
        raise TypeError(f"{processed_waveforms_pkl} did not contain a dict")
    if "sonic_depths" not in waveform_data:
        raise KeyError(f"`sonic_depths` not found in {processed_waveforms_pkl}")
    sonic_depths = [float(value) for value in waveform_data["sonic_depths"]]

    with h5py.File(ground_truth_h5, "r") as handle:
        if "path_data" not in handle:
            raise KeyError(f"`path_data` group not found in {ground_truth_h5}")
        path_keys = set(handle["path_data"].keys())

    processed_indices: list[int] = []
    missing_examples: list[dict[str, str | int | float]] = []
    for sample_index, depth_ft in enumerate(sonic_depths):
        key = depth_to_path_key(depth_ft)
        if key in path_keys:
            processed_indices.append(sample_index)
        elif len(missing_examples) < 10:
            missing_examples.append({"sample_index": sample_index, "depth_ft": depth_ft, "path_key": key})

    mapping_info = {
        "method": "sonic_depths_order_filtered_by_ground_truth_path_data_key",
        "sonic_depths_len": len(sonic_depths),
        "path_data_key_count": len(path_keys),
        "matched_count": len(processed_indices),
        "missing_count": len(sonic_depths) - len(processed_indices),
        "first_processed_indices": processed_indices[:5],
        "last_processed_indices": processed_indices[-5:],
        "first_missing_examples": missing_examples,
    }
    return processed_indices, sonic_depths, mapping_info


def load_record_metadata(
    source_tfrecord: Path,
    processed_waveforms_pkl: Path,
    ground_truth_h5: Path,
    max_records: int | None,
) -> tuple[list[RecordMeta], int, dict]:
    tfrecord_count = count_tfrecord_records(source_tfrecord, max_records=max_records)
    processed_indices, sonic_depths, mapping_info = reconstruct_processed_indices(processed_waveforms_pkl, ground_truth_h5)
    if max_records is not None:
        processed_indices = processed_indices[:max_records]
        mapping_info = {**mapping_info, "max_records_for_smoke_applied": max_records}

    if len(processed_indices) != tfrecord_count:
        raise ValueError(
            "TFRecord record count and reconstructed depth mapping length do not match: "
            f"records={tfrecord_count}, reconstructed_indices={len(processed_indices)}"
        )

    records: list[RecordMeta] = []
    for record_index, sample_index in enumerate(processed_indices):
        if sample_index < 0 or sample_index >= len(sonic_depths):
            raise IndexError(
                f"reconstructed_indices[{record_index}]={sample_index} is outside sonic_depths length {len(sonic_depths)}"
            )
        records.append(
            RecordMeta(
                record_index=record_index,
                sample_index=sample_index,
                depth_ft=float(sonic_depths[sample_index]),
            )
        )
    return records, tfrecord_count, mapping_info


def checked_counts(total: int, train_frac: float, val_frac: float) -> tuple[int, int, int]:
    if total < 3:
        raise ValueError("At least three records are required to create train/val/test splits")
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("--train-frac and --val-frac must be positive and sum to less than 1")
    train_count = max(1, int(total * train_frac))
    val_count = max(1, int(total * val_frac))
    test_count = total - train_count - val_count
    if test_count <= 0:
        test_count = 1
        if val_count > 1:
            val_count -= 1
        else:
            train_count -= 1
    if min(train_count, val_count, test_count) <= 0:
        raise ValueError(f"Invalid split counts: train={train_count}, val={val_count}, test={test_count}")
    return train_count, val_count, test_count


def base_order_assignments(total: int, preset: str, train_frac: float, val_frac: float) -> list[str]:
    train_count, val_count, _ = checked_counts(total, train_frac, val_frac)
    if preset != "depth_heldout_simple":
        raise ValueError(f"Unknown split preset for EXP-007: {preset}")
    assignments = ["test"] * total
    for i in range(0, train_count):
        assignments[i] = "train"
    for i in range(train_count, train_count + val_count):
        assignments[i] = "val"
    return assignments


def transition_boundaries(sorted_records: list[RecordMeta], assignments: list[str]) -> list[float]:
    boundaries: list[float] = []
    for left, right in zip(range(len(assignments) - 1), range(1, len(assignments))):
        if assignments[left] != assignments[right]:
            boundaries.append((sorted_records[left].depth_ft + sorted_records[right].depth_ft) / 2.0)
    return boundaries


def assign_by_order(
    sorted_records: list[RecordMeta],
    preset: str,
    train_frac: float,
    val_frac: float,
    gap_ft: float,
) -> SplitPlan:
    assignments = base_order_assignments(len(sorted_records), preset, train_frac, val_frac)
    boundaries = transition_boundaries(sorted_records, assignments)
    half_gap = max(0.0, gap_ft) / 2.0
    plan = SplitPlan(records=sorted_records, boundary_depths=boundaries)
    for i, record in enumerate(sorted_records):
        if half_gap > 0 and any(abs(record.depth_ft - boundary) < half_gap for boundary in boundaries):
            record.split = "dropped_gap"
            record.split_reason = f"within_{gap_ft:g}_ft_gap_of_split_boundary"
        else:
            record.split = assignments[i]
            record.split_reason = f"{preset}_depth_order"
    return plan


def build_split_plan(args: argparse.Namespace, records: list[RecordMeta]) -> SplitPlan:
    sorted_records = sorted(records, key=lambda record: (record.depth_ft, record.record_index))
    return assign_by_order(sorted_records, args.split_preset, args.train_frac, args.val_frac, args.gap_ft)


def split_depths(records: list[RecordMeta], split: str) -> list[float]:
    return [record.depth_ft for record in records if record.split == split]


def split_sample_indices(records: list[RecordMeta], split: str) -> set[int]:
    return {record.sample_index for record in records if record.split == split}


def describe_depths(depths: list[float]) -> dict[str, float | int | None]:
    if not depths:
        return {"count": 0, "min_depth_ft": None, "max_depth_ft": None, "median_depth_ft": None}
    return {
        "count": len(depths),
        "min_depth_ft": min(depths),
        "max_depth_ft": max(depths),
        "median_depth_ft": statistics.median(depths),
    }


def interval_distance(a: dict[str, float | int | None], b: dict[str, float | int | None]) -> float | None:
    if not a["count"] or not b["count"]:
        return None
    a_min = float(a["min_depth_ft"])
    a_max = float(a["max_depth_ft"])
    b_min = float(b["min_depth_ft"])
    b_max = float(b["max_depth_ft"])
    if a_max <= b_min:
        return b_min - a_max
    if b_max <= a_min:
        return a_min - b_max
    return -min(a_max, b_max) + max(a_min, b_min)


def make_audit(records: list[RecordMeta], args: argparse.Namespace, mapping_info: dict, warnings: list[str]) -> dict:
    stats = {split: describe_depths(split_depths(records, split)) for split in ("train", "val", "test")}
    dropped_gap = [record for record in records if record.split == "dropped_gap"]
    dropped_other = [record for record in records if record.split.startswith("dropped_") and record.split != "dropped_gap"]
    sample_sets = {split: split_sample_indices(records, split) for split in ("train", "val", "test")}
    intersections = {
        "train_val": sorted(sample_sets["train"] & sample_sets["val"]),
        "train_test": sorted(sample_sets["train"] & sample_sets["test"]),
        "val_test": sorted(sample_sets["val"] & sample_sets["test"]),
    }
    boundary_distances = {
        "train_val_ft": interval_distance(stats["train"], stats["val"]),
        "val_test_ft": interval_distance(stats["val"], stats["test"]),
        "train_test_ft": interval_distance(stats["train"], stats["test"]),
    }
    audit_warnings = list(warnings)
    for split, stat in stats.items():
        if not stat["count"]:
            audit_warnings.append(f"{split} split has zero records")
    if any(intersections.values()):
        audit_warnings.append("record/sample index overlap detected between splits")
    for name, distance in boundary_distances.items():
        if distance is not None and distance < 0:
            audit_warnings.append(f"depth range overlap detected for {name}: {distance:.6g} ft")
        if args.gap_ft > 0 and distance is not None and distance >= 0 and distance < args.gap_ft:
            audit_warnings.append(
                f"nearest kept depth distance for {name} is {distance:.6g} ft, below configured gap_ft={args.gap_ft:g}"
            )

    mapping_confirmed = mapping_info.get("matched_count") == mapping_info.get("path_data_key_count")
    if not mapping_confirmed:
        audit_warnings.append("reconstructed mapping count does not equal path_data key count")

    confirmed = (
        mapping_confirmed
        and all(stats[split]["count"] for split in ("train", "val", "test"))
        and not any(intersections.values())
        and all(distance is None or distance >= 0 for distance in boundary_distances.values())
        and not any("below configured gap_ft" in warning for warning in audit_warnings)
    )

    return {
        "generated_at": utc_now(),
        "experiment_id": args.experiment_id,
        "split_preset": args.split_preset,
        "dry_run": bool(args.dry_run),
        "source_tfrecord": str(args.source_tfrecord),
        "source_processed_waveforms_pkl": str(args.processed_waveforms_pkl),
        "source_ground_truth_h5": str(args.ground_truth_h5),
        "output_dir": str(args.output_dir),
        "gap_ft": args.gap_ft,
        "max_records_for_smoke": args.max_records_for_smoke,
        "record_depth_mapping": mapping_info,
        "stats": stats,
        "boundary_distances": boundary_distances,
        "record_index_intersections": {key: value[:20] for key, value in intersections.items()},
        "mutually_exclusive_record_sets": not any(intersections.values()),
        "dropped_buffer_count": len(dropped_gap),
        "dropped_other_count": len(dropped_other),
        "warnings": audit_warnings,
        "conclusion": "depth_heldout_split_confirmed" if confirmed else "needs_manual_verification",
    }


def manifest_row(record: RecordMeta, args: argparse.Namespace) -> dict[str, str]:
    return {
        "experiment_id": args.experiment_id,
        "record_index": str(record.record_index),
        "sample_index": str(record.sample_index),
        "depth_ft": f"{record.depth_ft:.12g}",
        "split": record.split,
        "split_reason": record.split_reason,
        "source_tfrecord": str(args.source_tfrecord),
        "source_processed_waveforms_pkl": str(args.processed_waveforms_pkl),
        "source_ground_truth_h5": str(args.ground_truth_h5),
        "record_depth_mapping_method": "sonic_depths_order_filtered_by_ground_truth_path_data_key",
        "created_from_branch_or_commit": args.git_evidence,
    }


def write_manifest(records: list[RecordMeta], args: argparse.Namespace, output_dir: Path, audit: dict) -> None:
    with (output_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.record_index):
            writer.writerow(manifest_row(record, args))

    manifest_json = {
        "generated_at": utc_now(),
        "experiment_id": args.experiment_id,
        "source_tfrecord": str(args.source_tfrecord),
        "source_processed_waveforms_pkl": str(args.processed_waveforms_pkl),
        "source_ground_truth_h5": str(args.ground_truth_h5),
        "git_evidence": args.git_evidence,
        "command": " ".join(sys.argv),
        "dry_run": bool(args.dry_run),
        "split_preset": args.split_preset,
        "gap_ft": args.gap_ft,
        "max_records_for_smoke": args.max_records_for_smoke,
        "record_depth_mapping": audit["record_depth_mapping"],
        "stats": audit["stats"],
        "warnings": audit["warnings"],
        "records": [manifest_row(record, args) for record in sorted(records, key=lambda item: item.record_index)],
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest_json, ensure_ascii=False, indent=2), encoding="utf-8")


def write_overview(audit: dict, output_dir: Path) -> None:
    fields = ["split", "count", "min_depth_ft", "max_depth_ft", "median_depth_ft"]
    with (output_dir / "depth_split_overview.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split, stats in audit["stats"].items():
            writer.writerow({"split": split, **stats})


def write_audit(audit: dict, output_dir: Path) -> None:
    (output_dir / "leakage_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# EXP-007 Depth Split Leakage Audit",
        "",
        f"Generated: `{audit['generated_at']}`",
        f"Conclusion: `{audit['conclusion']}`",
        f"Dry run: `{audit['dry_run']}`",
        "",
        "## Record-Depth Mapping",
        "",
        f"- method: `{audit['record_depth_mapping'].get('method')}`",
        f"- sonic_depths_len: `{audit['record_depth_mapping'].get('sonic_depths_len')}`",
        f"- path_data_key_count: `{audit['record_depth_mapping'].get('path_data_key_count')}`",
        f"- matched_count: `{audit['record_depth_mapping'].get('matched_count')}`",
        f"- missing_count: `{audit['record_depth_mapping'].get('missing_count')}`",
        "",
        "## Split Counts And Depth Ranges",
        "",
        "| split | count | min_depth_ft | max_depth_ft | median_depth_ft |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split, stats in audit["stats"].items():
        lines.append(
            f"| {split} | {stats['count']} | {stats['min_depth_ft']} | "
            f"{stats['max_depth_ft']} | {stats['median_depth_ft']} |"
        )
    lines.extend(["", "## Boundary Distances", "", "| boundary | nearest_kept_depth_distance_ft |", "| --- | ---: |"])
    for name, distance in audit["boundary_distances"].items():
        lines.append(f"| {name} | {distance} |")
    lines.extend(
        [
            "",
            "## Dropped Records",
            "",
            f"- dropped_buffer_count: `{audit['dropped_buffer_count']}`",
            f"- dropped_other_count: `{audit['dropped_other_count']}`",
            "",
            "## Warnings",
            "",
        ]
    )
    if audit["warnings"]:
        lines.extend(f"- {warning}" for warning in audit["warnings"])
    else:
        lines.append("- none")
    lines.append("")
    (output_dir / "leakage_audit.md").write_text("\n".join(lines), encoding="utf-8")


def output_paths(output_dir: Path) -> list[Path]:
    return [
        output_dir / "train.tfrecord",
        output_dir / "val.tfrecord",
        output_dir / "test.tfrecord",
        output_dir / "split_manifest.csv",
        output_dir / "split_manifest.json",
        output_dir / "leakage_audit.json",
        output_dir / "leakage_audit.md",
        output_dir / "depth_split_overview.csv",
    ]


def ensure_no_overwrite(output_dir: Path, overwrite: bool) -> None:
    existing = [path for path in output_paths(output_dir) if path.exists()]
    if existing and not overwrite:
        existing_list = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing split outputs without --overwrite. Existing files:\n"
            f"{existing_list}"
        )


def write_split_tfrecords(records: list[RecordMeta], args: argparse.Namespace, output_dir: Path) -> dict[str, int]:
    wanted = {record.record_index: record.split for record in records if record.split in {"train", "val", "test"}}
    handles = {
        "train": (output_dir / "train.tfrecord").open("wb"),
        "val": (output_dir / "val.tfrecord").open("wb"),
        "test": (output_dir / "test.tfrecord").open("wb"),
    }
    counts = {"train": 0, "val": 0, "test": 0}
    try:
        for record_index, full_record in iter_tfrecord_full_records(args.source_tfrecord, max_records=args.max_records_for_smoke):
            split = wanted.get(record_index)
            if split:
                handles[split].write(full_record)
                counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default="EXP-007")
    parser.add_argument("--source-tfrecord", type=Path, default=DEFAULT_SOURCE_TFRECORD)
    parser.add_argument("--processed-waveforms-pkl", type=Path, default=DEFAULT_PROCESSED_WAVEFORMS_PKL)
    parser.add_argument("--ground-truth-h5", type=Path, default=DEFAULT_GROUND_TRUTH_H5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-preset", choices=("depth_heldout_simple",), default="depth_heldout_simple")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--gap-ft", type=float, default=5.0)
    parser.add_argument("--max-records-for-smoke", type=int, default=None)
    parser.add_argument("--git-evidence", default=DEFAULT_GIT_EVIDENCE)
    parser.add_argument("--dry-run", action="store_true", help="Write manifest/audit only; do not write split TFRecords.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting files in the output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_safe_output_dir(args.output_dir, args.source_tfrecord)
    if args.gap_ft < 0:
        raise ValueError("--gap-ft must be non-negative")
    if args.max_records_for_smoke is not None and args.max_records_for_smoke < 3:
        raise ValueError("--max-records-for-smoke must be at least 3")

    ensure_no_overwrite(args.output_dir, overwrite=args.overwrite)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records, tfrecord_count, mapping_info = load_record_metadata(
        args.source_tfrecord,
        args.processed_waveforms_pkl,
        args.ground_truth_h5,
        max_records=args.max_records_for_smoke,
    )
    plan = build_split_plan(args, records)
    audit = make_audit(plan.records, args, mapping_info, plan.warnings)

    if not args.dry_run:
        written_counts = write_split_tfrecords(plan.records, args, args.output_dir)
        for split in ("train", "val", "test"):
            expected = audit["stats"][split]["count"]
            if written_counts[split] != expected:
                raise RuntimeError(f"{split} write count mismatch: wrote {written_counts[split]}, expected {expected}")

    write_manifest(plan.records, args, args.output_dir, audit)
    write_overview(audit, args.output_dir)
    write_audit(audit, args.output_dir)

    print(
        "exp007_depth_split "
        f"records={tfrecord_count} "
        f"train={audit['stats']['train']['count']} "
        f"val={audit['stats']['val']['count']} "
        f"test={audit['stats']['test']['count']} "
        f"dropped_gap={audit['dropped_buffer_count']} "
        f"dry_run={args.dry_run} "
        f"conclusion={audit['conclusion']} "
        f"output_dir={args.output_dir}"
    )
    if audit["warnings"]:
        print("warnings:")
        for warning in audit["warnings"]:
            print(f"- {warning}")
    return 0 if audit["conclusion"] == "depth_heldout_split_confirmed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
