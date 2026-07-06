#!/usr/bin/env python3
"""Parse TensorBoard scalar event files for thesis evidence inventory.

This script is intentionally read-only with respect to the HAL results tree.
It requires the optional `tensorboard` package:

    python scripts/thesis_parse_tensorboard_events.py

Outputs are written under docs/thesis_evidence by default.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


RESULTS_ROOT = Path("/mnt/c/Users/Administrator/Desktop/Hal/results")
DEFAULT_OUTPUT = Path("docs/thesis_evidence/tensorboard_scalar_metrics.csv")


FIELDS = [
    "source_path",
    "relative_path",
    "experiment_id",
    "run_split",
    "tag",
    "scalar_min",
    "scalar_max",
    "scalar_final",
    "best_step",
    "best_value",
    "final_step",
    "num_points",
    "extraction_method",
    "confidence",
    "notes",
]


def map_experiment(relative_path: str) -> str:
    path = relative_path.replace("\\", "/")
    if "样本权重+非对称损失" in path:
        return "EXP-014"
    if path.startswith("temp_result/test_relativity"):
        return "EXP-006"
    if path.startswith("temp_result/GaN+2Dlabel"):
        return "EXP-005"
    if path.startswith("temp_result/frequency-weighted_loss"):
        return "EXP-003"
    if path.startswith("temp_result/log_label"):
        return "EXP-002"
    if path.startswith("temp_result/baseline"):
        return "EXP-001"
    if path.startswith("temp_result/1D+percentage_Label"):
        return "EXP-007"
    if path.startswith("FFT_EfficientNet") and "/fft_regression/" in path:
        return "EXP-008"
    if path.startswith("FFT_EfficientNet") and "/image_translation/" in path:
        return "EXP-007"
    return "unknown"


def infer_run_split(relative_path: str) -> str:
    parts = relative_path.replace("\\", "/").split("/")
    if "validation" in parts:
        return "validation"
    if "train" in parts:
        return "train"
    return "unknown"


def higher_is_better(tag: str) -> bool:
    lowered = tag.lower()
    return any(token in lowered for token in ("auc", "accuracy", "precision", "recall", "f1", "r2"))


def rel_to_results(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def parse_event_file(path: Path, root: Path, event_accumulator_cls) -> list[dict[str, str]]:
    relative_path = rel_to_results(path, root)
    experiment_id = map_experiment(relative_path)
    run_split = infer_run_split(relative_path)
    accumulator = event_accumulator_cls(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    rows: list[dict[str, str]] = []
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        if not events:
            continue
        values = [float(event.value) for event in events]
        steps = [int(event.step) for event in events]
        final_value = values[-1]
        final_step = steps[-1]
        if higher_is_better(tag):
            best_value = max(values)
        else:
            best_value = min(values)
        best_step = steps[values.index(best_value)]
        rows.append(
            {
                "source_path": str(path),
                "relative_path": relative_path,
                "experiment_id": experiment_id,
                "run_split": run_split,
                "tag": tag,
                "scalar_min": f"{min(values):.12g}",
                "scalar_max": f"{max(values):.12g}",
                "scalar_final": f"{final_value:.12g}",
                "best_step": str(best_step),
                "best_value": f"{best_value:.12g}",
                "final_step": str(final_step),
                "num_points": str(len(values)),
                "extraction_method": "tensorboard_event_accumulator",
                "confidence": "strong",
                "notes": "Read-only TensorBoard scalar extraction; split/leakage status must be joined from split_forensic_audit.",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--event-list", type=Path, default=None)
    args = parser.parse_args()

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "TensorBoard is not importable. Install/activate an environment with tensorboard, "
            f"then rerun this script. Original error: {type(exc).__name__}: {exc}"
        ) from exc

    if args.event_list:
        with args.event_list.open(newline="", encoding="utf-8-sig") as handle:
            event_files = [Path(row["source_path"]) for row in csv.DictReader(handle)]
    else:
        event_files = sorted(args.results_root.rglob("events.out.tfevents*"))

    rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for event_file in event_files:
        try:
            rows.extend(parse_event_file(event_file, args.results_root, EventAccumulator))
        except Exception as exc:  # pragma: no cover - depends on event contents
            failures.append(
                {
                    "source_path": str(event_file),
                    "relative_path": rel_to_results(event_file, args.results_root),
                    "experiment_id": map_experiment(rel_to_results(event_file, args.results_root)),
                    "run_split": infer_run_split(rel_to_results(event_file, args.results_root)),
                    "tag": "parse_error",
                    "scalar_min": "",
                    "scalar_max": "",
                    "scalar_final": "",
                    "best_step": "",
                    "best_value": "",
                    "final_step": "",
                    "num_points": "0",
                    "extraction_method": "tensorboard_event_accumulator",
                    "confidence": "unknown",
                    "notes": f"{type(exc).__name__}: {exc}",
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows + failures)

    print(f"event_files={len(event_files)} scalar_rows={len(rows)} failures={len(failures)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
