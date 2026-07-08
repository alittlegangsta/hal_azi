#!/usr/bin/env python3
"""Smoke-check EXP-007 depth-heldout TFRecord split outputs without training."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from thesis_make_exp007_depth_split import count_tfrecord_records


DEFAULT_SPLIT_DIR = Path("output/thesis_depth_blocked/exp007/split_v001")
DEFAULT_FEATURE_SHAPE = (150, 400, 8)
DEFAULT_LABEL_SHAPE = (70,)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_shape(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def load_manifest_counts(manifest_path: Path) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            split = row.get("split")
            if split in counts:
                counts[split] += 1
    return counts


def parse_one_record_with_tensorflow(
    path: Path,
    expected_feature_shape: tuple[int, ...],
    expected_label_shape: tuple[int, ...],
) -> dict:
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "status": "tensorflow_missing",
            "error": f"{type(exc).__name__}: {exc}",
            "feature_shape": None,
            "label_shape": None,
            "shape_ok": None,
        }

    dataset = tf.data.TFRecordDataset(str(path))
    feature_description = {
        "feature": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.string),
    }
    for raw_record in dataset.take(1):
        parsed = tf.io.parse_single_example(raw_record, feature_description)
        feature = tf.io.parse_tensor(parsed["feature"], out_type=tf.float32)
        label = tf.io.parse_tensor(parsed["label"], out_type=tf.float32)
        feature_shape = tuple(int(dim) for dim in feature.shape)
        label_shape = tuple(int(dim) for dim in label.shape)
        shape_ok = feature_shape == expected_feature_shape and label_shape == expected_label_shape
        return {
            "status": "parsed",
            "feature_shape": list(feature_shape),
            "label_shape": list(label_shape),
            "expected_feature_shape": list(expected_feature_shape),
            "expected_label_shape": list(expected_label_shape),
            "shape_ok": shape_ok,
        }
    return {
        "status": "empty_tfrecord",
        "feature_shape": None,
        "label_shape": None,
        "shape_ok": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--train-tfrecord", type=Path, default=None)
    parser.add_argument("--val-tfrecord", type=Path, default=None)
    parser.add_argument("--test-tfrecord", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--expected-feature-shape", default="150,400,8")
    parser.add_argument("--expected-label-shape", default="70")
    parser.add_argument("--parse-one", action="store_true", help="Parse one record from each split with TensorFlow if available.")
    parser.add_argument("--require-parse", action="store_true", help="Fail if TensorFlow parse is unavailable or shapes mismatch.")
    parser.add_argument("--dry-run", action="store_true", help="Report planned checks only; do not read TFRecord contents or write JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split_dir = args.split_dir
    paths = {
        "train": args.train_tfrecord or split_dir / "train.tfrecord",
        "val": args.val_tfrecord or split_dir / "val.tfrecord",
        "test": args.test_tfrecord or split_dir / "test.tfrecord",
    }
    manifest_path = args.manifest or split_dir / "split_manifest.csv"
    audit_path = args.audit_json or split_dir / "leakage_audit.json"
    output_json = args.output_json or split_dir / "smoke_check.json"
    expected_feature_shape = parse_shape(args.expected_feature_shape)
    expected_label_shape = parse_shape(args.expected_label_shape)

    planned = {
        "generated_at": utc_now(),
        "dry_run": bool(args.dry_run),
        "split_dir": str(split_dir),
        "tfrecords": {split: str(path) for split, path in paths.items()},
        "manifest": str(manifest_path),
        "audit_json": str(audit_path),
        "output_json": str(output_json),
        "parse_one": bool(args.parse_one),
    }
    if args.dry_run:
        print(json.dumps(planned, ensure_ascii=False, indent=2))
        return 0

    result = dict(planned)
    errors: list[str] = []
    counts: dict[str, int] = {}
    for split, path in paths.items():
        if not path.exists():
            errors.append(f"missing {split} tfrecord: {path}")
            counts[split] = 0
            continue
        counts[split] = count_tfrecord_records(path)
    result["tfrecord_counts"] = counts

    if manifest_path.exists():
        manifest_counts = load_manifest_counts(manifest_path)
        result["manifest_counts"] = manifest_counts
        for split in ("train", "val", "test"):
            if counts.get(split) != manifest_counts.get(split):
                errors.append(f"{split} count mismatch: tfrecord={counts.get(split)} manifest={manifest_counts.get(split)}")
    else:
        errors.append(f"missing manifest: {manifest_path}")

    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        result["audit_conclusion"] = audit.get("conclusion")
        result["audit_warnings"] = audit.get("warnings", [])
    else:
        errors.append(f"missing audit json: {audit_path}")

    parse_results = {}
    if args.parse_one:
        for split, path in paths.items():
            if path.exists() and counts.get(split, 0) > 0:
                parse_result = parse_one_record_with_tensorflow(path, expected_feature_shape, expected_label_shape)
                parse_results[split] = parse_result
                if args.require_parse and parse_result.get("shape_ok") is not True:
                    errors.append(f"{split} TensorFlow parse/shape check failed: {parse_result}")
    result["parse_results"] = parse_results
    result["errors"] = errors
    result["status"] = "pass" if not errors else "fail"

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
