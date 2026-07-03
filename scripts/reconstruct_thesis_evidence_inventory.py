#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reconstruct thesis evidence inventory for the old HAL/XSI-CAST project.

This script is intentionally read-only with respect to the Windows results
directory. It only writes generated inventory documents under docs/thesis_evidence.
"""

from __future__ import annotations

import ast
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path("/mnt/c/Users/Administrator/Desktop/Hal/results")
OUT = ROOT / "docs" / "thesis_evidence"
MEMO = RESULTS / "temp_result" / "改进memo.md"
MAX_TEXT_BYTES = 2 * 1024 * 1024

TEXT_SUFFIXES = {".md", ".txt", ".csv", ".json", ".yaml", ".yml"}
CODE_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".svg", ".pdf"}
MODEL_SUFFIXES = {".h5", ".keras", ".ckpt", ".pt", ".pth", ".onnx", ".pkl", ".index"}
NOTE_KEYWORDS = [
    "fft",
    "gradcam",
    "grad-cam",
    "cwt",
    "label",
    "history",
    "prediction",
    "truth",
    "baseline",
    "efficientnet",
    "resnet",
    "se-resnet",
    "metadata",
    "inclination",
    "pre",
    "correction",
    "channel",
    "severity",
    "scatter",
]


def run_git(args: list[str], cwd: Path = ROOT, check: bool = False) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=check,
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"ERROR: {exc}"
    if result.returncode != 0:
        stderr = result.stderr.strip()
        return f"ERROR[{result.returncode}]: {stderr}"
    return result.stdout.strip()


def run_cmd(args: list[str], cwd: Path = ROOT) -> dict[str, Any]:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return {
            "cmd": args,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except FileNotFoundError as exc:
        return {"cmd": args, "returncode": 127, "stdout": "", "stderr": str(exc)}


def ensure_out() -> None:
    if RESULTS in OUT.parents or OUT == RESULTS:
        raise RuntimeError("Refusing to write under the read-only results directory")
    OUT.mkdir(parents=True, exist_ok=True)


def rel_results(path: Path) -> str:
    try:
        return path.relative_to(RESULTS).as_posix()
    except ValueError:
        return path.as_posix()


def rel_repo(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def json_dump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def csv_dump(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: normalize_cell(row.get(k, "")) for k in fieldnames})


def normalize_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return "; ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def md_escape(value: Any) -> str:
    text = normalize_cell(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def md_table(rows: list[dict[str, Any]], cols: list[str], max_rows: int | None = None) -> str:
    if max_rows is not None:
        rows = rows[:max_rows]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(md_escape(row.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def safe_read_text(path: Path, max_bytes: int = MAX_TEXT_BYTES) -> tuple[str, str]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        return "", f"stat_error:{exc}"
    if size > max_bytes:
        return "", f"skipped_large_text:{size}"
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return "", f"read_error:{exc}"
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin-1"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8-replace"


def human_size(num: int) -> str:
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024
    return f"{num}B"


def modified_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().isoformat(timespec="seconds")


def likely_group(relative_path: str) -> str:
    parts = relative_path.split("/")
    if not parts:
        return "unknown"
    if parts[0] == "temp_result" and len(parts) > 1:
        return f"temp_result/{parts[1]}"
    return parts[0]


def classify_artifact(path: Path, relative_path: str) -> str:
    lower = relative_path.lower()
    suffix = path.suffix.lower()
    name = path.name.lower()
    if name.startswith("events.out.tfevents"):
        return "tensorboard_log"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in {".h5", ".keras", ".ckpt", ".pt", ".pth", ".onnx", ".index"}:
        return "model_checkpoint"
    if ".data-" in name and "ckpt" in name:
        return "model_checkpoint"
    if suffix == ".pkl":
        if "history" in lower or "logs" in lower:
            return "tensorboard_log"
        return "config" if "idx" in name else "unknown"
    if suffix == ".ipynb":
        return "notebook"
    if suffix == ".md":
        return "markdown_memo"
    if suffix in {".ppt", ".pptx"}:
        return "ppt"
    if suffix == ".csv":
        return "metric_csv"
    if suffix in {".json", ".yaml", ".yml", ".ini", ".cfg"}:
        return "config"
    if suffix == ".txt":
        return "markdown_memo" if "result" in lower or "memo" in lower else "unknown"
    return "unknown"


def short_note(relative_path: str, artifact_type: str) -> str:
    lower = relative_path.lower()
    notes: list[str] = []
    if "gradcam" in lower or "grad-cam" in lower:
        notes.append("Grad-CAM/attention visualization")
    if "truth_vs_prediction" in lower or "comparison" in lower:
        notes.append("prediction-vs-ground-truth figure")
    if "training_history" in lower or "history" in lower:
        notes.append("training history")
    if "fft" in lower:
        notes.append("FFT-related artifact")
    if "cwt" in lower or "scalogram" in lower:
        notes.append("CWT/scalogram evidence")
    if "label_generation" in lower or "final_label" in lower:
        notes.append("label construction visualization")
    if "performance" in lower or "metric" in lower:
        notes.append("performance/metric summary")
    if "metadata" in lower or "inclination" in lower:
        notes.append("metadata/eccentricity/inclination evidence")
    if artifact_type == "model_checkpoint":
        notes.append("model weight/checkpoint metadata only")
    if not notes:
        matched = [k for k in NOTE_KEYWORDS if k in lower]
        notes = matched[:3] if matched else ["metadata only"]
    return "; ".join(dict.fromkeys(notes))


def scan_results() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files: list[dict[str, Any]] = []
    first_dirs: set[str] = set()
    second_dirs: set[str] = set()
    dir_counts: Counter[str] = Counter()
    dir_sizes: Counter[str] = Counter()
    second_counts: Counter[str] = Counter()
    second_sizes: Counter[str] = Counter()
    artifact_counts: Counter[str] = Counter()
    suffix_counts: Counter[str] = Counter()
    keyword_counts: Counter[str] = Counter()
    total_size = 0

    for dirpath, _dirnames, filenames in os.walk(RESULTS, followlinks=False):
        dpath = Path(dirpath)
        try:
            rel_dir = dpath.relative_to(RESULTS)
        except ValueError:
            rel_dir = Path(".")
        parts = rel_dir.parts
        if len(parts) >= 1 and parts[0] != ".":
            first_dirs.add(parts[0])
        if len(parts) >= 2:
            second_dirs.add("/".join(parts[:2]))
        for filename in filenames:
            path = dpath / filename
            try:
                st = path.stat()
            except OSError:
                continue
            relative_path = rel_results(path)
            group = likely_group(relative_path)
            artifact = classify_artifact(path, relative_path)
            row = {
                "relative_path": relative_path,
                "file_name": filename,
                "suffix": path.suffix.lower(),
                "size_bytes": st.st_size,
                "modified_time": modified_iso(st.st_mtime),
                "likely_experiment_group": group,
                "likely_artifact_type": artifact,
                "short_note": short_note(relative_path, artifact),
            }
            files.append(row)
            total_size += st.st_size
            artifact_counts[artifact] += 1
            suffix_counts[path.suffix.lower() or "<none>"] += 1
            parts_rel = relative_path.split("/")
            top = parts_rel[0] if parts_rel else "."
            second = "/".join(parts_rel[:2]) if len(parts_rel) > 2 else top
            dir_counts[top] += 1
            dir_sizes[top] += st.st_size
            second_counts[second] += 1
            second_sizes[second] += st.st_size
            lower = relative_path.lower()
            for keyword in NOTE_KEYWORDS:
                if keyword in lower:
                    keyword_counts[keyword] += 1

    files.sort(key=lambda r: r["relative_path"])
    summary = {
        "results_root": RESULTS.as_posix(),
        "file_count": len(files),
        "total_size_bytes": total_size,
        "total_size_human": human_size(total_size),
        "first_level_dirs": sorted(first_dirs),
        "second_level_dirs": sorted(second_dirs),
        "counts_by_first_level": [
            {"path": k, "file_count": dir_counts[k], "size_bytes": dir_sizes[k], "size_human": human_size(dir_sizes[k])}
            for k in sorted(dir_counts)
        ],
        "counts_by_second_level": [
            {"path": k, "file_count": second_counts[k], "size_bytes": second_sizes[k], "size_human": human_size(second_sizes[k])}
            for k in sorted(second_counts)
        ],
        "artifact_type_counts": dict(sorted(artifact_counts.items())),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "keyword_counts": dict(sorted(keyword_counts.items())),
    }
    return files, summary


def generate_preflight(result_summary: dict[str, Any]) -> dict[str, Any]:
    python_cmd = run_cmd(["python", "--version"])
    python3_cmd = run_cmd(["python3", "--version"])
    memo_exists = MEMO.is_file()
    preflight = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "repo_path": ROOT.as_posix(),
        "current_branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "current_commit": run_git(["rev-parse", "HEAD"]),
        "git_status_short": run_git(["status", "--short", "--branch"]),
        "python_version": {
            "python": python_cmd,
            "python3": python3_cmd,
            "selected": python3_cmd["stdout"] or python3_cmd["stderr"],
        },
        "git_branches": run_git(["branch", "--all", "--verbose", "--no-abbrev"]).splitlines(),
        "git_remotes": run_git(["remote", "--verbose"]).splitlines(),
        "git_tags": run_git(["tag", "--list"]).splitlines(),
        "results_root": RESULTS.as_posix(),
        "results_readable": os.access(RESULTS, os.R_OK),
        "memo_path": MEMO.as_posix(),
        "memo_exists": memo_exists,
        "results_file_count": result_summary["file_count"],
        "results_total_size_bytes": result_summary["total_size_bytes"],
        "results_total_size_human": result_summary["total_size_human"],
        "first_level_dirs": result_summary["first_level_dirs"],
        "second_level_dirs": result_summary["second_level_dirs"],
        "note": "Results directory was scanned read-only; generated files are under docs/thesis_evidence.",
    }
    json_dump(OUT / "preflight.json", preflight)
    md = [
        "# Preflight",
        "",
        f"- Generated at: `{preflight['generated_at']}`",
        f"- Repo path: `{preflight['repo_path']}`",
        f"- Current branch: `{preflight['current_branch']}`",
        f"- Current commit: `{preflight['current_commit']}`",
        f"- Selected Python: `{preflight['python_version']['selected']}`",
        f"- `python` command: return code `{python_cmd['returncode']}`, stdout `{python_cmd['stdout']}`, stderr `{python_cmd['stderr']}`",
        f"- Results root readable: `{preflight['results_readable']}`",
        f"- Memo exists: `{preflight['memo_exists']}` at `{preflight['memo_path']}`",
        f"- Results size: `{preflight['results_total_size_human']}` (`{preflight['results_total_size_bytes']}` bytes)",
        f"- Results file count: `{preflight['results_file_count']}`",
        "",
        "## Git Status",
        "",
        "```text",
        preflight["git_status_short"],
        "```",
        "",
        "## Git Remotes",
        "",
        "```text",
        "\n".join(preflight["git_remotes"]) or "(none)",
        "```",
        "",
        "## Git Branches",
        "",
        "```text",
        "\n".join(preflight["git_branches"]) or "(none)",
        "```",
        "",
        "## Results Tree: Level 1",
        "",
        md_table(result_summary["counts_by_first_level"], ["path", "file_count", "size_human"]),
        "",
        "## Results Tree: Level 2",
        "",
        md_table(result_summary["counts_by_second_level"], ["path", "file_count", "size_human"], max_rows=200),
    ]
    (OUT / "preflight.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return preflight


def write_result_inventory(files: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    fields = [
        "relative_path",
        "file_name",
        "suffix",
        "size_bytes",
        "modified_time",
        "likely_experiment_group",
        "likely_artifact_type",
        "short_note",
    ]
    csv_dump(OUT / "result_file_inventory.csv", files, fields)
    json_dump(OUT / "result_tree_inventory.json", {"summary": summary, "files": files})
    top_sample = [
        {
            "relative_path": f["relative_path"],
            "size_human": human_size(int(f["size_bytes"])),
            "likely_experiment_group": f["likely_experiment_group"],
            "likely_artifact_type": f["likely_artifact_type"],
            "short_note": f["short_note"],
        }
        for f in sorted(files, key=lambda r: int(r["size_bytes"]), reverse=True)[:60]
    ]
    md = [
        "# Result Tree Inventory",
        "",
        "This file is a read-only metadata inventory of `/mnt/c/Users/Administrator/Desktop/Hal/results`.",
        "",
        "## Summary",
        "",
        f"- File count: `{summary['file_count']}`",
        f"- Total size: `{summary['total_size_human']}` (`{summary['total_size_bytes']}` bytes)",
        "",
        "## Artifact Types",
        "",
        md_table(
            [{"artifact_type": k, "file_count": v} for k, v in summary["artifact_type_counts"].items()],
            ["artifact_type", "file_count"],
        ),
        "",
        "## Keyword Hits",
        "",
        md_table(
            [{"keyword": k, "file_count": v} for k, v in sorted(summary["keyword_counts"].items())],
            ["keyword", "file_count"],
        ),
        "",
        "## Directory Groups",
        "",
        md_table(summary["counts_by_first_level"], ["path", "file_count", "size_human"]),
        "",
        "## Second-Level Groups",
        "",
        md_table(summary["counts_by_second_level"], ["path", "file_count", "size_human"], max_rows=200),
        "",
        "## Largest Files",
        "",
        md_table(top_sample, ["relative_path", "size_human", "likely_experiment_group", "likely_artifact_type", "short_note"]),
        "",
        "Full per-file details are in `result_file_inventory.csv` and `result_tree_inventory.json`.",
    ]
    (OUT / "result_tree_inventory.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def find_text_files(files: list[dict[str, Any]]) -> list[Path]:
    paths: list[Path] = []
    for row in files:
        rel = row["relative_path"]
        path = RESULTS / rel
        if path.suffix.lower() in TEXT_SUFFIXES and "/.git/" not in f"/{rel}":
            paths.append(path)
    return paths


def repo_text_files() -> list[Path]:
    targets: list[Path] = []
    for base in [ROOT]:
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            rel = rel_repo(path)
            if rel.startswith(".git/") or rel.startswith("docs/thesis_evidence/"):
                continue
            if rel == "scripts/reconstruct_thesis_evidence_inventory.py":
                continue
            if any(part in {".git", "__pycache__", ".pytest_cache"} for part in path.parts):
                continue
            if path.suffix.lower() in CODE_SUFFIXES or path.name.lower().startswith("readme"):
                targets.append(path)
    return sorted(targets)


def keywords_from_text(text: str) -> list[str]:
    lower = text.lower()
    patterns = {
        "FFT": ["fft", "傅里叶"],
        "CWT": ["cwt", "时频", "scalogram"],
        "Grad-CAM": ["grad-cam", "gradcam", "热力"],
        "EfficientNetV2B0": ["efficientnetv2b0", "efficientnet"],
        "SE-ResNet": ["se-resnet", "resnet"],
        "dual-channel": ["双通道", "two-channel", "metadata"],
        "eccentricity/pre-correction": ["偏心", "预校正", "correction", "inclination"],
        "log label": ["log", "对数"],
        "severity": ["严重性", "severity", "2.5 -"],
        "1D percentage": ["百分比", "percentage", "profile"],
        "AUC": ["auc"],
        "SSIM/PSNR": ["ssim", "psnr"],
        "GAN/GaN": ["gan", "gan+"],
    }
    found = [name for name, keys in patterns.items() if any(k in lower for k in keys)]
    return found


def parse_memo_claims(text: str) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    sections: list[tuple[str, list[str]]] = []
    current_title = "preamble"
    current_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("# "):
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = line.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_lines))

    for idx, (title, lines) in enumerate(sections, start=1):
        body = "\n".join(lines).strip()
        if not body:
            continue
        lower = body.lower()
        result_lines = [ln.strip(" -*\t") for ln in lines if any(k in ln.lower() for k in ["result", "auc", "accuracy", "预测", "崩塌", "attention", "热力", "不足", "val_"])]
        failure_lines = [ln.strip(" -*\t") for ln in lines if any(k in ln for k in ["很差", "崩塌", "不足", "失败", "学不到", "不理想", "低估"])]
        method = "; ".join(keywords_from_text(title + "\n" + body)) or "unknown"
        if "baseline" in title.lower():
            result_hint = "temp_result/baseline"
        elif "对数" in title or "log" in lower:
            result_hint = "temp_result/log_label"
        elif "高频" in title or "weighted" in lower:
            result_hint = "temp_result/frequency-weighted_loss"
        elif "gan" in title.lower() or "gan+" in lower:
            result_hint = "temp_result/GaN+2Dlabel"
        elif "分类" in title:
            result_hint = "temp_result/test_relativity"
        elif "一维" in title or "百分比" in title:
            result_hint = "temp_result/1D+percentage_Label; FFT_EfficientNet; FFT_EfficientNet_1"
        elif "ppt" in title.lower():
            result_hint = "presentation outline, no direct result dir"
        else:
            result_hint = "unknown"
        code_hint = "unknown"
        if "efficientnet" in lower:
            code_hint = "origin/1D+percentage_Label:src/modeling/model.py"
        elif "分类" in title:
            code_hint = "origin/test_relativity:src/modeling/model.py; origin/test_relativity:src/data_processing/create_tfrecords.py"
        elif "fft" in lower or "对数" in title or "baseline" in title.lower():
            code_hint = "src/data_processing/create_tfrecords.py; src/modeling/model.py; src/interpretation/run_analysis.py"
        if "高频" in title:
            code_hint = "origin/frequency-weighted_loss:src/modeling/train.py; origin/frequency-weighted_loss:src/interpretation/run_analysis.py"
        claims.append(
            {
                "claim_id": f"memo-{idx:02d}",
                "evidence_path": MEMO.as_posix(),
                "section": title,
                "experiment_name": title,
                "method": method,
                "model": infer_model_from_text(title + "\n" + body),
                "input_feature": infer_input_from_text(title + "\n" + body),
                "target_label": infer_label_from_text(title + "\n" + body),
                "training_strategy": infer_training_from_text(title + "\n" + body),
                "evaluation_metrics": infer_metrics_from_text(title + "\n" + body),
                "result_conclusion": " ".join(result_lines[:6]) or "unknown",
                "failure_or_limitation": " ".join(failure_lines[:4]) or "unknown",
                "possible_result_dir": result_hint,
                "possible_code_file": code_hint,
                "confidence": "strong" if result_lines or failure_lines else "medium",
                "notes": "Extracted from memo section; claims require source path citation in thesis.",
            }
        )
    return claims


def infer_model_from_text(text: str) -> str:
    lower = text.lower()
    if "efficientnetv2b0" in lower or "efficientnet" in lower:
        return "EfficientNetV2B0"
    if "se-resnet" in lower:
        return "SE-ResNet"
    if "unet" in lower or "u-net" in lower or "a²inet" in lower or "a2inet" in lower:
        return "Attention U-Net / A2INet"
    if "cnn" in lower:
        return "CNN classifier"
    if "gan" in lower:
        return "GAN"
    return "unknown"


def infer_input_from_text(text: str) -> str:
    lower = text.lower()
    parts = []
    if "cwt" in lower or "时频" in lower:
        parts.append("8-channel CWT time-frequency image")
    if "waveform" in lower or "声波" in lower:
        parts.append("sonic waveform")
    if "metadata" in lower or "inclination" in lower or "偏心" in text:
        parts.append("metadata/inclination/eccentricity features")
    return "; ".join(parts) or "unknown"


def infer_label_from_text(text: str) -> str:
    lower = text.lower()
    labels = []
    if "二元" in text or "binary" in lower or "分类" in text:
        labels.append("binary channeling label/mask")
    if "百分比" in text or "percentage" in lower or "profile" in lower:
        labels.append("1D channeling percentage profile")
    if "fft" in lower:
        labels.append("FFT magnitude label")
    if "严重性" in text or "severity" in lower or "2.5 - zc" in lower:
        labels.append("severity transform max(0, 2.5 - Zc)")
    if "zc" in lower:
        labels.append("CAST Zc slice")
    return "; ".join(dict.fromkeys(labels)) or "unknown"


def infer_training_from_text(text: str) -> str:
    lower = text.lower()
    parts = []
    if "early stopping" in lower or "提前停止" in text:
        parts.append("early stopping")
    if "focal" in lower or "焦点" in text:
        parts.append("focal loss")
    if "过拟合" in text or "single" in lower or "单一样本" in text:
        parts.append("single-sample overfit test")
    if "高频" in text or "weighted" in lower:
        parts.append("frequency-weighted loss")
    if "混合损失" in text or "hybrid" in lower:
        parts.append("hybrid MSE + gradient loss")
    return "; ".join(parts) or "unknown"


def infer_metrics_from_text(text: str) -> str:
    lower = text.lower()
    metrics = []
    for token in ["auc", "accuracy", "val_auc", "val_accuracy", "loss", "val_loss", "ssim", "psnr", "mae"]:
        if token in lower:
            metrics.append(token)
    if "85%" in text:
        metrics.append("accuracy≈85%")
    if "0.95361" in text:
        metrics.append("AUC=0.95361")
    return "; ".join(dict.fromkeys(metrics)) or "unknown"


def generate_text_evidence(files: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result_text_paths = find_text_files(files)
    repo_paths = repo_text_files()
    evidence_files: list[dict[str, Any]] = []
    snippets: list[dict[str, Any]] = []
    keyword_re = re.compile(
        r"(FFT|CWT|Grad-?CAM|EfficientNet|SE-?ResNet|AUC|SSIM|PSNR|log|对数|百分比|严重性|偏心|预校正|双通道|baseline|窜槽|Zc|训练|标签|模型)",
        re.IGNORECASE,
    )

    all_paths: list[tuple[str, Path]] = [("results", p) for p in result_text_paths] + [("repo", p) for p in repo_paths]
    for source_type, path in all_paths:
        text, status = safe_read_text(path)
        rel = rel_results(path) if source_type == "results" else rel_repo(path)
        keys = keywords_from_text(text) if text else []
        evidence_files.append(
            {
                "source_type": source_type,
                "path": path.as_posix(),
                "relative_path": rel,
                "size_bytes": path.stat().st_size if path.exists() else "unknown",
                "read_status": status,
                "keywords": keys,
                "line_count": text.count("\n") + 1 if text else 0,
            }
        )
        if text:
            for lineno, line in enumerate(text.splitlines(), start=1):
                if keyword_re.search(line):
                    cleaned = line.strip()
                    if cleaned:
                        snippets.append(
                            {
                                "source_type": source_type,
                                "path": path.as_posix(),
                                "relative_path": rel,
                                "line": lineno,
                                "snippet": cleaned[:500],
                                "keywords": keywords_from_text(cleaned),
                            }
                        )
                    if sum(1 for s in snippets if s["path"] == path.as_posix()) >= 40:
                        break

    memo_text, memo_status = safe_read_text(MEMO)
    memo_claims = parse_memo_claims(memo_text) if memo_text else []
    text_json = {
        "summary": {
            "result_text_file_count": len(result_text_paths),
            "repo_text_file_count": len(repo_paths),
            "memo_status": memo_status,
            "memo_claim_count": len(memo_claims),
            "snippet_count": len(snippets),
        },
        "text_files": evidence_files,
        "snippets": snippets,
        "memo_claims": memo_claims,
    }
    json_dump(OUT / "text_evidence_extraction.json", text_json)
    csv_dump(
        OUT / "memo_experiment_claims.csv",
        memo_claims,
        [
            "claim_id",
            "evidence_path",
            "section",
            "experiment_name",
            "method",
            "model",
            "input_feature",
            "target_label",
            "training_strategy",
            "evaluation_metrics",
            "result_conclusion",
            "failure_or_limitation",
            "possible_result_dir",
            "possible_code_file",
            "confidence",
            "notes",
        ],
    )
    md = [
        "# Text Evidence Extraction",
        "",
        "Readable text evidence was extracted from results text files and repository code/config/docs.",
        "",
        "## Summary",
        "",
        f"- Result text files: `{len(result_text_paths)}`",
        f"- Repo text/code files: `{len(repo_paths)}`",
        f"- Memo extraction status: `{memo_status}`",
        f"- Memo claims: `{len(memo_claims)}`",
        f"- Keyword snippets: `{len(snippets)}`",
        "",
        "## Memo Experiment Claims",
        "",
        md_table(
            memo_claims,
            [
                "claim_id",
                "section",
                "method",
                "model",
                "target_label",
                "evaluation_metrics",
                "result_conclusion",
                "failure_or_limitation",
                "possible_result_dir",
            ],
        ),
        "",
        "## Text Files With Keywords",
        "",
        md_table(
            [r for r in evidence_files if r["keywords"]],
            ["source_type", "relative_path", "read_status", "keywords", "line_count"],
            max_rows=120,
        ),
        "",
        "## Representative Snippets",
        "",
        md_table(snippets, ["source_type", "relative_path", "line", "keywords", "snippet"], max_rows=120),
        "",
        "Full extraction is in `text_evidence_extraction.json`; memo claims are in `memo_experiment_claims.csv`.",
    ]
    (OUT / "text_evidence_extraction.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return memo_claims, evidence_files


def py_functions_classes(text: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        names = re.findall(r"^\s*(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", text, flags=re.MULTILINE)
        return names[:30]
    names = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
    return names[:40]


def infer_code_role(path: str, text: str) -> str:
    lower = (path + "\n" + text).lower()
    roles = []
    if "main_preprocess" in lower:
        roles.append("preprocessing script")
    if "cwt" in lower and ("pywt" in lower or "wavelet" in lower or "scales" in lower):
        roles.append("CWT transformation")
    if "tfrecord" in lower or "hdf5" in lower or "h5py" in lower:
        roles.append("TFRecord/HDF5 generation")
    if "process_zc_slice_to_label" in lower or "label" in lower:
        roles.append("label construction")
    if "fft" in lower:
        roles.append("FFT label/script")
    if "2.5 - zc" in lower or "severity" in lower or "严重性" in text:
        roles.append("severity transform")
    if "efficientnetv2b0" in lower:
        roles.append("EfficientNetV2B0 model")
    if "se-resnet" in lower or "seresnet" in lower:
        roles.append("SE-ResNet model")
    if "metadata" in lower or "dual" in lower or "双通道" in text:
        roles.append("dual-input/metadata model")
    if "correction" in lower or "inclination" in lower or "偏心" in text or "预校正" in text:
        roles.append("eccentricity correction / pre-correction")
    if "grad_cam" in lower or "grad-cam" in lower or "heatmap" in lower:
        roles.append("Grad-CAM analysis")
    if "model.fit" in lower or "compile(" in lower or "earlystopping" in lower:
        roles.append("training logic")
    if "metric" in lower or "ssim" in lower or "psnr" in lower or "auc" in lower:
        roles.append("analysis/evaluation/metric computation")
    if "savefig" in lower or "save_pickle" in lower or "model.save" in lower:
        roles.append("result saving logic")
    if "mask" in lower or "artifact" in lower:
        roles.append("artifact masking / binary mask")
    if "validation_split" in lower or ".take(" in lower or ".skip(" in lower or "split" in lower:
        roles.append("train/test split logic")
    return "; ".join(dict.fromkeys(roles)) or "unknown"


def infer_experiment_group_from_code(path: str, text: str) -> str:
    lower = (path + "\n" + text).lower()
    if "test_relativity" in lower or "classification" in lower or "binary classification" in lower:
        return "CWT-label binary classification"
    if "1d+percentage" in lower or "percentage" in lower or "profile_regression" in lower:
        return "1D percentage label / EfficientNet"
    if "frequency-weighted" in lower or "weighted_mse" in lower:
        return "FFT high-frequency weighted loss"
    if "log1p" in lower or "log scaling" in lower:
        return "FFT log label"
    if "gan" in lower:
        return "GAN/severity map"
    if "fft_regression" in lower:
        return "FFT severity label"
    if "grad" in lower and "cam" in lower:
        return "Grad-CAM interpretability"
    if "cwt" in lower:
        return "CWT preprocessing"
    return "unknown"


def extract_io_refs(text: str) -> str:
    refs = re.findall(r"['\"]([^'\"]+\.(?:h5|keras|pkl|tfrecord|png|csv|json|yaml|yml|txt|npy|npz|mat))['\"]", text)
    refs = list(dict.fromkeys(refs))[:30]
    return "; ".join(refs) or "unknown"


def code_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add_entry(path_label: str, text: str, source_ref: str) -> None:
        key = (path_label, str(hash(text)))
        if key in seen:
            return
        seen.add(key)
        role = infer_code_role(path_label, text)
        rows.append(
            {
                "path": path_label,
                "source_ref": source_ref,
                "role": role,
                "key_functions_classes": py_functions_classes(text) if path_label.endswith(".py") else [],
                "input_output_files": extract_io_refs(text),
                "likely_experiment_group": infer_experiment_group_from_code(path_label, text),
                "thesis_relevance": thesis_relevance_from_role(role),
                "confidence": "strong" if role != "unknown" else "weak",
                "evidence_note": "Repository code or tracked branch snapshot; no training executed.",
            }
        )

    for path in sorted(ROOT.rglob("*.py")):
        rel = rel_repo(path)
        if rel.startswith(".git/") or rel.startswith("docs/thesis_evidence/"):
            continue
        if rel == "scripts/reconstruct_thesis_evidence_inventory.py":
            continue
        text, status = safe_read_text(path)
        if text:
            add_entry(rel, text, f"working tree:{rel}; read_status={status}")

    branch_refs = [
        "origin/master",
        "origin/log_scaling",
        "origin/frequency-weighted_loss",
        "origin/GaN",
        "origin/test_relativity",
        "origin/1D+percentage_Label",
        "origin/percentage_label+FFT",
    ]
    key_paths = [
        "config.py",
        "main.py",
        "src/data_processing/create_tfrecords.py",
        "src/data_processing/main_preprocess.py",
        "src/cwt_transformation/main_transform_translation.py",
        "src/modeling/model.py",
        "src/modeling/train.py",
        "src/modeling/dataset.py",
        "src/interpretation/run_analysis.py",
        "src/interpretation/run_analysis_regressor.py",
        "src/interpretation/run_analysis_classification.py",
        "src/interpretation/grad_cam.py",
        "src/interpretation/debug_data_visualization.py",
        "src/visualization/visualize_data_pipeline.py",
        "src/visualization/visualize_training_history.py",
    ]
    for branch in branch_refs:
        for file_path in key_paths:
            output = run_git(["show", f"{branch}:{file_path}"])
            if output.startswith("ERROR[") or output.startswith("ERROR:"):
                continue
            add_entry(f"{branch}:{file_path}", output, f"git show {branch}:{file_path}")

    rows.sort(key=lambda r: r["path"])
    return rows


def thesis_relevance_from_role(role: str) -> str:
    if role == "unknown":
        return "unknown"
    relevance = []
    if "label" in role or "severity" in role or "FFT" in role:
        relevance.append("label construction and method rationale")
    if "model" in role or "training" in role:
        relevance.append("model/training method")
    if "Grad-CAM" in role or "evaluation" in role:
        relevance.append("evaluation and interpretability")
    if "split" in role:
        relevance.append("data leakage risk review")
    if "CWT" in role or "preprocessing" in role:
        relevance.append("data pipeline")
    return "; ".join(dict.fromkeys(relevance)) or "background"


def write_code_inventory(rows: list[dict[str, Any]]) -> None:
    fields = [
        "path",
        "source_ref",
        "role",
        "key_functions_classes",
        "input_output_files",
        "likely_experiment_group",
        "thesis_relevance",
        "confidence",
        "evidence_note",
    ]
    csv_dump(OUT / "code_method_inventory.csv", rows, fields)
    json_dump(OUT / "code_method_inventory.json", {"files": rows})
    md = [
        "# Code Method Inventory",
        "",
        "This inventory includes the current working tree plus selected tracked branch snapshots via `git show`.",
        "",
        f"- Code evidence entries: `{len(rows)}`",
        "",
        "## Key Code Evidence",
        "",
        md_table(rows, ["path", "role", "key_functions_classes", "likely_experiment_group", "thesis_relevance", "confidence"], max_rows=160),
    ]
    (OUT / "code_method_inventory.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def git_inventory() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    branch_lines = run_git(["for-each-ref", "--format=%(refname:short)%09%(objectname)%09%(committerdate:iso-strict)%09%(subject)", "refs/heads", "refs/remotes"]).splitlines()
    branches = []
    for line in branch_lines:
        parts = line.split("\t", 3)
        if len(parts) == 4:
            branches.append({"branch": parts[0], "commit": parts[1], "date": parts[2], "subject": parts[3]})
    log_raw = run_git(["log", "--all", "--date=iso-strict", "--pretty=format:%H%x09%ad%x09%D%x09%s", "-n", "120"]).splitlines()
    commits: list[dict[str, Any]] = []
    for line in log_raw:
        parts = line.split("\t", 3)
        if len(parts) == 4:
            files = run_git(["show", "--name-only", "--pretty=format:", parts[0]]).splitlines()
            commits.append({"commit": parts[0], "date": parts[1], "refs": parts[2], "subject": parts[3], "changed_files": [f for f in files if f.strip()]})
    graph_summary = run_git(["log", "--all", "--graph", "--decorate", "--oneline", "--date=short", "-n", "80"])
    tags = run_git(["tag", "--list"]).splitlines()
    selected_branches = [
        "origin/master",
        "origin/log_scaling",
        "origin/frequency-weighted_loss",
        "origin/GaN",
        "origin/test_relativity",
        "origin/1D+percentage_Label",
        "origin/percentage_label+FFT",
    ]
    diffs = []
    for branch in selected_branches:
        stat = run_git(["diff", "--stat", "origin/master", branch])
        diffs.append({"base": "origin/master", "compare": branch, "diff_stat": stat})
    branch_mapping = build_branch_mapping(branches)
    git_json = {
        "branches": branches,
        "tags": tags,
        "commits": commits,
        "graph_summary": graph_summary,
        "selected_branch_diffs_vs_origin_master": diffs,
        "needs_remote_server_or_github_fetch": [
            "No local branch/commit conclusively maps CSI+SE-ResNet result directory to code.",
            "No local branch/commit conclusively maps 双通道学习 result directory to code.",
            "No local branch/commit conclusively maps 预校正 result directory to code.",
            "Remote server `/home/xiaoj/hal_azi` may contain unpushed branch state and conda environment `hall`.",
        ],
    }
    return git_json, branch_mapping


def build_branch_mapping(branches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    known = {
        "origin/test_relativity": ("CWT-label binary classification", "CNN classifier validates CWT-label relation", "strong"),
        "origin/GaN": ("GAN/severity map", "GaN+2Dlabel and GAN training attempt", "medium"),
        "origin/1D+percentage_Label": ("1D percentage label", "EfficientNetV2B0 profile regressor", "strong"),
        "origin/percentage_label+FFT": ("FFT regression", "Severity transform plus FFT magnitude/log label", "strong"),
        "origin/frequency-weighted_loss": ("FFT high-frequency weighted loss", "Weighted loss for higher FFT coefficients", "strong"),
        "origin/log_scaling": ("FFT log label", "log(1+magnitude) FFT label route", "strong"),
        "origin/master": ("FFT log label", "master points at log_scaling commit", "medium"),
        "master": ("FFT log label", "local master points at log_scaling commit", "medium"),
    }
    rows = []
    branch_names = {b["branch"] for b in branches}
    for b in branches:
        if b["branch"] in known:
            family, reason, confidence = known[b["branch"]]
        else:
            family, reason, confidence = ("unknown", "No direct mapping from local branch name/message", "weak")
        rows.append(
            {
                "branch": b["branch"],
                "latest_commit": b["commit"],
                "latest_date": b["date"],
                "latest_subject": b["subject"],
                "likely_experiment_group": family,
                "mapping_reason": reason,
                "confidence": confidence,
            }
        )
    expected = {
        "baseline": "needs_remote_server_or_github_fetch",
        "dual-channel": "needs_remote_server_or_github_fetch",
        "pre-correction": "needs_remote_server_or_github_fetch",
        "CSI+SE-ResNet": "needs_remote_server_or_github_fetch",
        "Grad-CAM": "partial: present in repo files and multiple branches, not a standalone branch",
    }
    for family, status in expected.items():
        if family == "Grad-CAM" or not any(family.lower() in r["likely_experiment_group"].lower() for r in rows):
            rows.append(
                {
                    "branch": status,
                    "latest_commit": "unknown",
                    "latest_date": "unknown",
                    "latest_subject": "unknown",
                    "likely_experiment_group": family,
                    "mapping_reason": "No conclusive local branch mapping; verify remote server/GitHub if thesis needs exact order.",
                    "confidence": "unknown",
                }
            )
    return rows


def write_git_inventory(git_json: dict[str, Any], branch_mapping: list[dict[str, Any]]) -> None:
    json_dump(OUT / "git_branch_timeline.json", git_json)
    csv_dump(
        OUT / "branch_experiment_mapping.csv",
        branch_mapping,
        ["branch", "latest_commit", "latest_date", "latest_subject", "likely_experiment_group", "mapping_reason", "confidence"],
    )
    commit_rows = [
        {
            "commit": c["commit"][:12],
            "date": c["date"],
            "refs": c["refs"],
            "subject": c["subject"],
            "changed_files": "; ".join(c["changed_files"][:8]),
        }
        for c in git_json["commits"][:60]
    ]
    md = [
        "# Git Branch Timeline",
        "",
        "Local Git information only; no fetch/network access was used.",
        "",
        "## Branch Mapping",
        "",
        md_table(branch_mapping, ["branch", "latest_date", "latest_subject", "likely_experiment_group", "confidence"], max_rows=80),
        "",
        "## Commit Timeline",
        "",
        md_table(commit_rows, ["commit", "date", "refs", "subject", "changed_files"], max_rows=80),
        "",
        "## Graph Summary",
        "",
        "```text",
        git_json["graph_summary"],
        "```",
        "",
        "## Branch Diff Stat vs origin/master",
        "",
    ]
    for diff in git_json["selected_branch_diffs_vs_origin_master"]:
        md.extend(
            [
                f"### {diff['compare']}",
                "",
                "```text",
                diff["diff_stat"] or "(no diff or unavailable)",
                "```",
                "",
            ]
        )
    md.extend(
        [
            "## Needs Verification",
            "",
            "\n".join(f"- `{item}`" for item in git_json["needs_remote_server_or_github_fetch"]),
        ]
    )
    (OUT / "git_branch_timeline.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def group_paths(files: list[dict[str, Any]], group: str, limit: int = 12, contains: str | None = None) -> list[str]:
    rows = [f for f in files if f["relative_path"].startswith(group)]
    if contains:
        rows = [f for f in rows if contains.lower() in f["relative_path"].lower()]
    rows = sorted(rows, key=lambda r: (r["likely_artifact_type"] != "image", r["relative_path"]))
    return [(RESULTS / r["relative_path"]).as_posix() for r in rows[:limit]]


def experiment_registry(files: list[dict[str, Any]], memo_claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def paths(*groups: str, limit: int = 16) -> str:
        collected: list[str] = []
        for group in groups:
            collected.extend(group_paths(files, group, limit=limit))
        return "; ".join(collected[:limit]) or "unknown"

    memo_path = MEMO.as_posix()
    rows = [
        {
            "experiment_id": "EXP-001",
            "experiment_name": "Baseline FFT magnitude image translation",
            "likely_stage_order": "1",
            "method_family": "baseline",
            "model": "Attention U-Net / A2INet",
            "input_feature": "8-channel CWT time-frequency image",
            "target_label": "FFT magnitude of CAST Zc slice",
            "label_transform": "none or unknown; memo says no extra high-frequency penalty",
            "azimuth_handling": "FFT magnitude used to reduce azimuth/phase mismatch; phase discarded",
            "eccentricity_handling": "unknown",
            "train_split": "code uses VALIDATION_SPLIT and dataset take/skip; depth leakage not verified",
            "loss": "hybrid MSE + gradient difference loss",
            "metrics_available": "loss/val_loss/mae in logs; memo qualitative result",
            "main_result_summary": "Memo states Grad-CAM was scattered and prediction collapsed toward an overall average with vertical stripe patterns.",
            "main_failure_or_limitation": "Failed to learn higher-structure FFT coefficients; kept as baseline/failed attempt.",
            "result_paths": paths("temp_result/baseline"),
            "code_paths": "src/data_processing/create_tfrecords.py; src/modeling/model.py; src/modeling/train.py; src/interpretation/run_analysis.py",
            "branch_or_commit": "initial/current history around 54b0182..1ee68a; exact baseline branch unknown",
            "thesis_use": "baseline",
            "evidence_strength": "strong",
            "notes": f"Evidence: memo `{memo_path}` plus result directory metadata.",
        },
        {
            "experiment_id": "EXP-002",
            "experiment_name": "FFT log-label image translation",
            "likely_stage_order": "2",
            "method_family": "FFT log label",
            "model": "Attention U-Net / A2INet",
            "input_feature": "8-channel CWT time-frequency image",
            "target_label": "FFT magnitude label",
            "label_transform": "log(1 + magnitude)",
            "azimuth_handling": "FFT magnitude; phase discarded for rotation/azimuth invariance",
            "eccentricity_handling": "unknown",
            "train_split": "VALIDATION_SPLIT with dataset take/skip; depth-blocked split not verified",
            "loss": "hybrid MSE + gradient difference loss",
            "metrics_available": "loss/val_loss/mae; qualitative Grad-CAM and prediction plots",
            "main_result_summary": "Memo says Grad-CAM became more concentrated around 0.5-0.7 ms and 25-30 kHz, but predictions remained poor with horizontal stripe patterns.",
            "main_failure_or_limitation": "Prediction quality insufficient despite improved attention localization.",
            "result_paths": paths("temp_result/log_label"),
            "code_paths": "src/data_processing/create_tfrecords.py; src/interpretation/run_analysis.py",
            "branch_or_commit": "origin/log_scaling / 1ee68a62763c091097683e326c3a187c991a85c3",
            "thesis_use": "ablation",
            "evidence_strength": "strong",
            "notes": f"Evidence: memo `{memo_path}`, current code uses `np.log1p`, result directory `temp_result/log_label`.",
        },
        {
            "experiment_id": "EXP-003",
            "experiment_name": "FFT high-frequency weighted loss",
            "likely_stage_order": "3",
            "method_family": "FFT severity label",
            "model": "Attention U-Net / A2INet",
            "input_feature": "8-channel CWT time-frequency image",
            "target_label": "FFT magnitude/log label",
            "label_transform": "log label plus frequency weighting in loss",
            "azimuth_handling": "FFT magnitude; phase discarded",
            "eccentricity_handling": "unknown",
            "train_split": "same split logic as image translation; depth leakage not verified",
            "loss": "weighted MSE over FFT coefficients + gradient difference loss",
            "metrics_available": "SSIM/PSNR mentioned in memo/code; tensorboard logs and plots available",
            "main_result_summary": "Memo states Grad-CAM was more concentrated near 0.75-0.85 ms and 25-28 kHz, but predictions still fit low FFT coefficients.",
            "main_failure_or_limitation": "High-frequency penalty did not solve low-coefficient collapse.",
            "result_paths": paths("temp_result/frequency-weighted_loss"),
            "code_paths": "origin/frequency-weighted_loss:src/modeling/train.py; origin/frequency-weighted_loss:src/interpretation/run_analysis.py",
            "branch_or_commit": "origin/frequency-weighted_loss / 9899283c351712b791f9837b0be9a96b7003f96f",
            "thesis_use": "failed_attempt",
            "evidence_strength": "strong",
            "notes": f"Evidence: memo `{memo_path}` and branch code snapshot.",
        },
        {
            "experiment_id": "EXP-004",
            "experiment_name": "GAN + severity transform / 2D label",
            "likely_stage_order": "4",
            "method_family": "other",
            "model": "GAN / generator-discriminator route",
            "input_feature": "8-channel CWT time-frequency image",
            "target_label": "severity map max(0, 2.5 - Zc), then FFT/log variants",
            "label_transform": "severity transform and possible FFT/log",
            "azimuth_handling": "partly FFT; exact final label variant needs verification",
            "eccentricity_handling": "unknown",
            "train_split": "unknown",
            "loss": "GAN generator/discriminator loss; details need branch verification",
            "metrics_available": "result.txt contains epoch losses; plots/checkpoints available",
            "main_result_summary": "Memo states model collapse; result.txt shows generator loss stayed around 307-325 through epoch 100.",
            "main_failure_or_limitation": "Model collapse; not suitable as main quantitative result.",
            "result_paths": paths("temp_result/GaN+2Dlabel"),
            "code_paths": "origin/GaN:src/modeling/train.py; origin/GaN:src/data_processing/create_tfrecords.py",
            "branch_or_commit": "origin/GaN / c08d695779b3fbb666663a7439b16fbe00e1d61d",
            "thesis_use": "failed_attempt",
            "evidence_strength": "medium",
            "notes": "Spelling is `GaN` in branch/result path; likely intended GAN. Treat exact architecture as needs_verification.",
        },
        {
            "experiment_id": "EXP-005",
            "experiment_name": "Two-channel binary label and focal-loss/overfit test",
            "likely_stage_order": "5",
            "method_family": "other",
            "model": "GAN/U-Net route, exact code needs verification",
            "input_feature": "8-channel CWT time-frequency image",
            "target_label": "two-channel binary mask: channeling vs good bonding",
            "label_transform": "Zc < 2.5 channeling mask; Zc >= 2.5 good-bonding mask",
            "azimuth_handling": "direct mask; no reliable azimuth alignment",
            "eccentricity_handling": "unknown",
            "train_split": "single-sample overfit test mentioned; exact split unknown",
            "loss": "focal loss mentioned in memo; exact implementation needs verification",
            "metrics_available": "qualitative memo and result images",
            "main_result_summary": "Memo states model collapse.",
            "main_failure_or_limitation": "Even simplified binary mask route did not become a usable image-generation result.",
            "result_paths": paths("temp_result/GaN+2Dlabel"),
            "code_paths": "origin/GaN and surrounding commits; exact file needs verification",
            "branch_or_commit": "origin/GaN / c08d695779b3fbb666663a7439b16fbe00e1d61d; earlier commits 3b1eecc/92d670d likely relevant",
            "thesis_use": "failed_attempt",
            "evidence_strength": "medium",
            "notes": f"Evidence primarily from memo `{memo_path}`; implementation mapping needs remote/server verification.",
        },
        {
            "experiment_id": "EXP-006",
            "experiment_name": "CNN binary classification: CWT-label relationship test",
            "likely_stage_order": "6",
            "method_family": "baseline",
            "model": "CNN classifier",
            "input_feature": "8-channel CWT time-frequency image",
            "target_label": "binary label: channeling exists if >1% pixels have Zc < 2.5",
            "label_transform": "threshold at 1% channeling pixels",
            "azimuth_handling": "reduces task to presence/absence, avoiding azimuth alignment",
            "eccentricity_handling": "unknown",
            "train_split": "VALIDATION_SPLIT; exact depth-blocked separation not verified",
            "loss": "binary classification loss; exact compile options in branch code",
            "metrics_available": "AUC=0.95361, accuracy/val_accuracy≈85%, loss/val_loss from memo/result.txt",
            "main_result_summary": "Validated that CWT contains learnable information about channeling existence.",
            "main_failure_or_limitation": "Only presence/absence; not the final depth/severity profile target.",
            "result_paths": paths("temp_result/test_relativity"),
            "code_paths": "origin/test_relativity:src/modeling/model.py; origin/test_relativity:src/data_processing/create_tfrecords.py; origin/test_relativity:src/interpretation/run_analysis_classification.py",
            "branch_or_commit": "origin/test_relativity / 5e59652a4259c59e1d22b069271e2a67d863dadd",
            "thesis_use": "background",
            "evidence_strength": "strong",
            "notes": "AUC/accuracy claim is traceable to memo and `temp_result/test_relativity/result.txt.txt`.",
        },
        {
            "experiment_id": "EXP-007",
            "experiment_name": "1D percentage label profile regression",
            "likely_stage_order": "7",
            "method_family": "1D percentage label",
            "model": "EfficientNetV2B0 with regression head",
            "input_feature": "8-channel CWT time-frequency image",
            "target_label": "depth-wise channeling percentage profile",
            "label_transform": "mean(Zc<2.5) over azimuth per depth, expressed as 0-100%",
            "azimuth_handling": "collapses azimuth to depth profile; avoids azimuth mismatch",
            "eccentricity_handling": "unknown",
            "train_split": "VALIDATION_SPLIT; exact depth-blocked split not verified",
            "loss": "regression loss; exact branch compile settings need review",
            "metrics_available": "training history images; memo qualitative conclusion",
            "main_result_summary": "Memo states attention consistently focused on 0.5-1.0 ms and 23-28 kHz, with stronger attention for severe channeling.",
            "main_failure_or_limitation": "Quantitative severity is conservatively underestimated, especially for severe channeling.",
            "result_paths": paths("temp_result/1D+percentage_Label", "FFT_EfficientNet", "FFT_EfficientNet_1"),
            "code_paths": "origin/1D+percentage_Label:src/modeling/model.py; origin/1D+percentage_Label:src/data_processing/create_tfrecords.py; origin/1D+percentage_Label:src/interpretation/run_analysis_regressor.py",
            "branch_or_commit": "origin/1D+percentage_Label / e9739c8c3fc4d53e1af63dafa84e29e00b42e9c4",
            "thesis_use": "main_result",
            "evidence_strength": "strong",
            "notes": "Strongest confirmed learnable/severity-oriented route, but it sacrifices azimuth resolution.",
        },
        {
            "experiment_id": "EXP-008",
            "experiment_name": "EfficientNet FFT severity regression",
            "likely_stage_order": "8",
            "method_family": "FFT severity label",
            "model": "EfficientNetV2B0 or branch-specific regressor",
            "input_feature": "8-channel CWT time-frequency image",
            "target_label": "severity transform and FFT magnitude/log coefficients",
            "label_transform": "severity max(0, 2.5-Zc) + FFT + log(1+coefficients)",
            "azimuth_handling": "FFT magnitude discards phase to handle azimuth rotation/mismatch",
            "eccentricity_handling": "unknown",
            "train_split": "unknown; branch needs exact verification",
            "loss": "regression loss, exact compile settings needs_verification",
            "metrics_available": "label-generation and training-history figures; no confirmed final numeric metric in text inventory",
            "main_result_summary": "Evidence exists for label generation/training history; final quantitative outcome is needs_verification.",
            "main_failure_or_limitation": "Potential FFT frequency resolution issue noted in PPT outline: 8 receivers and low-frequency discriminability may limit angle-resolved separation.",
            "result_paths": paths("FFT_EfficientNet", "FFT_EfficientNet_1"),
            "code_paths": "origin/percentage_label+FFT:src/data_processing/create_tfrecords.py; origin/percentage_label+FFT:src/interpretation/run_analysis_regressor.py",
            "branch_or_commit": "origin/percentage_label+FFT / 7ba021cfa6eacd148247258ee28b8527dbbc6c92",
            "thesis_use": "main_result",
            "evidence_strength": "medium",
            "notes": "Recommended thesis route by task framing, but needs metric/split verification before making final performance claims.",
        },
        {
            "experiment_id": "EXP-009",
            "experiment_name": "CSI + CNN visual/Grad-CAM analysis",
            "likely_stage_order": "unknown",
            "method_family": "Grad-CAM interpretability",
            "model": "CNN or CSI-specific model, code mapping unknown",
            "input_feature": "sonic/CSI-derived images",
            "target_label": "channeling class/distribution, exact label unknown",
            "label_transform": "unknown",
            "azimuth_handling": "unknown",
            "eccentricity_handling": "unknown",
            "train_split": "unknown",
            "loss": "unknown",
            "metrics_available": "distribution/Grad-CAM/statistics figures",
            "main_result_summary": "Directory contains Grad-CAM statistics, signal examples, CSI/channeling distributions, and filtering comparisons.",
            "main_failure_or_limitation": "Exact code/metrics missing locally.",
            "result_paths": paths("CSI+CNN"),
            "code_paths": "needs_remote_server_or_github_fetch",
            "branch_or_commit": "unknown",
            "thesis_use": "figure/background",
            "evidence_strength": "medium",
            "notes": "Use as visual/material evidence only unless code and metrics are recovered.",
        },
        {
            "experiment_id": "EXP-010",
            "experiment_name": "CSI + SE-ResNet azimuth matching / classification",
            "likely_stage_order": "unknown",
            "method_family": "SE-ResNet azimuth matching",
            "model": "SE-ResNet",
            "input_feature": "CSI/CWT-derived images, exact input unknown",
            "target_label": "class/quality labels visible in candidate filenames; exact label definition unknown",
            "label_transform": "unknown",
            "azimuth_handling": "likely azimuth matching; needs verification",
            "eccentricity_handling": "unknown",
            "train_split": "unknown",
            "loss": "unknown",
            "metrics_available": "performance_summary_plots, attention analysis, candidate comparisons",
            "main_result_summary": "Result directory contains multi-array outputs, model checkpoints, performance summaries, attention analysis, and candidate comparisons.",
            "main_failure_or_limitation": "No local code branch conclusively mapped to these results.",
            "result_paths": paths("CSI+SE-ResNet"),
            "code_paths": "needs_remote_server_or_github_fetch",
            "branch_or_commit": "unknown",
            "thesis_use": "baseline",
            "evidence_strength": "medium",
            "notes": "Strong artifact evidence but code provenance needs server/GitHub verification.",
        },
        {
            "experiment_id": "EXP-011",
            "experiment_name": "Dual-channel metadata fusion",
            "likely_stage_order": "unknown",
            "method_family": "dual-channel metadata fusion",
            "model": "dual-input model, exact architecture unknown",
            "input_feature": "CWT plus metadata/inclination-related channel",
            "target_label": "unknown, likely channeling/severity class or profile",
            "label_transform": "unknown",
            "azimuth_handling": "unknown",
            "eccentricity_handling": "metadata fusion; exact use unknown",
            "train_split": "unknown",
            "loss": "unknown",
            "metrics_available": "performance_summary_plots, metadata-vs-label validation, CWT-vs-label validation",
            "main_result_summary": "Result directory contains validation plots for inclination, CWT-vs-label, and metadata-vs-label.",
            "main_failure_or_limitation": "Task framing says dual-channel failed; treat as explicitly marked inference until metrics/code are recovered.",
            "result_paths": paths("双通道学习"),
            "code_paths": "needs_remote_server_or_github_fetch",
            "branch_or_commit": "unknown",
            "thesis_use": "failed_attempt",
            "evidence_strength": "weak",
            "notes": "Explicitly marked inference: failure status comes from user task framing plus lack of strong result text.",
        },
        {
            "experiment_id": "EXP-012",
            "experiment_name": "Eccentricity pre-correction",
            "likely_stage_order": "unknown",
            "method_family": "eccentricity correction",
            "model": "correction model, exact architecture unknown",
            "input_feature": "CWT/metadata, exact input unknown",
            "target_label": "unknown",
            "label_transform": "unknown",
            "azimuth_handling": "attempted pre-correction for eccentricity/azimuth mismatch",
            "eccentricity_handling": "pre-correction",
            "train_split": "unknown",
            "loss": "unknown",
            "metrics_available": "performance_summary_plots and attention plots",
            "main_result_summary": "Result directory contains a model checkpoint and attention/performance summary plots.",
            "main_failure_or_limitation": "Task framing says eccentricity correction was poor; treat as explicitly marked inference until metrics/code are recovered.",
            "result_paths": paths("预校正"),
            "code_paths": "needs_remote_server_or_github_fetch",
            "branch_or_commit": "unknown",
            "thesis_use": "failed_attempt",
            "evidence_strength": "weak",
            "notes": "Explicitly marked inference: poor result status comes from user task framing plus missing mapped code/text.",
        },
        {
            "experiment_id": "EXP-013",
            "experiment_name": "Grad-CAM interpretability across routes",
            "likely_stage_order": "cross-cutting",
            "method_family": "Grad-CAM interpretability",
            "model": "CNN, A2INet, EfficientNet/SE-ResNet depending on route",
            "input_feature": "CWT/scalogram or route-specific input image",
            "target_label": "route-specific",
            "label_transform": "route-specific",
            "azimuth_handling": "interpretability only",
            "eccentricity_handling": "interpretability only",
            "train_split": "route-specific",
            "loss": "route-specific",
            "metrics_available": "Grad-CAM plots/statistics and memo attention-frequency claims",
            "main_result_summary": "Memo and result plots repeatedly locate sensitive regions in high-frequency CWT bands around roughly 22-30 kHz and 0.5-1.3 ms, depending on route.",
            "main_failure_or_limitation": "Needs batch-level statistics and consistent plotting for final thesis claims.",
            "result_paths": "; ".join(
                p
                for p in [
                    *group_paths(files, "CSI+CNN", contains="gradcam", limit=8),
                    *group_paths(files, "temp_result", contains="gradcam", limit=10),
                    *group_paths(files, "CSI+SE-ResNet", contains="attention", limit=8),
                ][:20]
            )
            or "unknown",
            "code_paths": "src/interpretation/run_analysis.py; src/interpretation/grad_cam.py; branch-specific run_analysis scripts",
            "branch_or_commit": "multiple branches",
            "thesis_use": "main_result",
            "evidence_strength": "strong",
            "notes": "Use for interpretability chapter, but avoid overclaiming without a batch statistic.",
        },
    ]
    return rows


def write_experiment_registry(rows: list[dict[str, Any]]) -> None:
    fields = [
        "experiment_id",
        "experiment_name",
        "likely_stage_order",
        "method_family",
        "model",
        "input_feature",
        "target_label",
        "label_transform",
        "azimuth_handling",
        "eccentricity_handling",
        "train_split",
        "loss",
        "metrics_available",
        "main_result_summary",
        "main_failure_or_limitation",
        "result_paths",
        "code_paths",
        "branch_or_commit",
        "thesis_use",
        "evidence_strength",
        "notes",
    ]
    csv_dump(OUT / "experiment_inventory.csv", rows, fields)
    json_dump(OUT / "experiment_inventory.json", {"experiments": rows})
    md = [
        "# Experiment Inventory",
        "",
        "One row is one reconstructed experiment or method version. Unknown fields are intentionally left as `unknown` or `needs_verification` rather than inferred as facts.",
        "",
        f"- Experiment rows: `{len(rows)}`",
        "",
        md_table(
            rows,
            [
                "experiment_id",
                "experiment_name",
                "likely_stage_order",
                "method_family",
                "model",
                "target_label",
                "metrics_available",
                "main_result_summary",
                "thesis_use",
                "evidence_strength",
            ],
        ),
        "",
        "Full path-level evidence is in `experiment_inventory.csv` and `experiment_inventory.json`.",
    ]
    (OUT / "experiment_inventory.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_method_taxonomy(experiments: list[dict[str, Any]]) -> None:
    taxonomy = {
        "research_problem": [
            {
                "claim": "XSI acoustic waveform and CAST Zc image have azimuth mismatch in vertical wells.",
                "evidence": "memo PPT section in results/temp_result/改进memo.md; explicitly thesis framing",
                "status": "supported_by_memo",
            },
            {
                "claim": "Relative Bearing is unreliable for direct point-to-point azimuth supervision.",
                "evidence": "memo PPT section and user task framing",
                "status": "supported_by_memo_plus_inference",
            },
            {
                "claim": "Direct azimuth-resolved point supervision is not the strongest thesis route with current evidence.",
                "evidence": "FFT baseline/log/weighted-loss failures in memo and result plots",
                "status": "supported_by_memo",
            },
        ],
        "data_construction": [
            "XSI waveform -> CWT scalogram/time-frequency image (repo config.py and src/cwt_transformation/main_transform_translation.py)",
            "CAST Zc slices from ground_truth_db HDF5 (repo config.py and create_tfrecords.py)",
            "Severity transform max(0, 2.5 - Zc) appears in memo and origin/percentage_label+FFT create_tfrecords.py",
            "Depth window/path label uses MAX_PATH_DEPTH_POINTS=70 and target depth range in config.py",
        ],
        "label_routes": [
            "1D percentage label: mean(Zc < 2.5) over azimuth for each depth, from origin/1D+percentage_Label create_tfrecords.py.",
            "FFT magnitude label: apply FFT along azimuth axis and take magnitude, from current create_tfrecords.py and memo.",
            "Log transform: log(1 + magnitude), from memo and current code.",
            "Phase discarded for rotation/azimuth invariance: memo PPT section; use as rationale, not as proven performance claim.",
        ],
        "model_routes": [
            "SE-ResNet route has result artifacts but code mapping is missing locally.",
            "EfficientNetV2B0 route is present in origin/1D+percentage_Label model.py.",
            "Dual-channel metadata fusion has result artifacts but code/metrics need verification.",
            "Correction/pre-correction model has result artifacts but code/metrics need verification.",
        ],
        "interpretability": [
            "Grad-CAM plots and memo identify sensitive time-frequency regions, especially high-frequency bands around 22-30 kHz and sub-ms to ~1.3 ms windows.",
            "Use Grad-CAM as qualitative interpretability unless batch statistics are regenerated from existing artifacts or verified.",
        ],
        "failed_routes": [
            "Azimuth matching mismatch and FFT image translation collapse: supported by memo and baseline/log/weighted-loss result artifacts.",
            "Eccentricity correction poor: explicitly marked inference from user task framing plus result directory existence; needs code/metric verification.",
            "Dual-channel failed: explicitly marked inference from user task framing plus result directory existence; needs code/metric verification.",
            "High-severity prediction poor for 1D percentage regression: supported by memo.",
        ],
        "recommended_mainline": [
            "Use severity + FFT magnitude/log label with CWT + EfficientNet as the intended angle-mismatch-aware thesis mainline, but mark quantitative performance as needs_verification until metrics/split are recovered.",
            "Use 1D percentage EfficientNet as the strongest confirmed learnability/severity evidence and a practical fallback main result.",
            "Use artifact masking and Grad-CAM interpretability as explanation/supporting evidence only when tied to specific result files.",
        ],
    }
    json_dump(OUT / "method_taxonomy.json", taxonomy)
    md = [
        "# Method Taxonomy",
        "",
        "## 1. Research Problem",
        "",
        "\n".join(f"- {r['claim']} Evidence: `{r['evidence']}`. Status: `{r['status']}`." for r in taxonomy["research_problem"]),
        "",
        "## 2. Data Construction",
        "",
        "\n".join(f"- {item}" for item in taxonomy["data_construction"]),
        "",
        "## 3. Label Routes",
        "",
        "\n".join(f"- {item}" for item in taxonomy["label_routes"]),
        "",
        "## 4. Model Routes",
        "",
        "\n".join(f"- {item}" for item in taxonomy["model_routes"]),
        "",
        "## 5. Interpretability Route",
        "",
        "\n".join(f"- {item}" for item in taxonomy["interpretability"]),
        "",
        "## 6. Failed Routes",
        "",
        "\n".join(f"- {item}" for item in taxonomy["failed_routes"]),
        "",
        "## 7. Recommended Thesis Mainline",
        "",
        "\n".join(f"- {item}" for item in taxonomy["recommended_mainline"]),
        "",
        "## Experiment Families",
        "",
        md_table(experiments, ["experiment_id", "method_family", "experiment_name", "thesis_use", "evidence_strength"]),
    ]
    (OUT / "method_taxonomy.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def image_candidate_category(relative_path: str) -> tuple[str, str, str, str]:
    lower = relative_path.lower()
    if "real_original_sonic" in lower or "raw_waveform" in lower or "csi_distribution" in lower or "channeling_distribution" in lower:
        return ("数据示意图", "第2章 数据与问题定义", "展示原始声波/CSI/窜槽分布证据", "no")
    if "cwt" in lower or "scalogram" in lower:
        return ("CWT 示例图", "第2章 数据构建", "展示CWT时频图输入", "no")
    if "final_label" in lower or "severity" in lower:
        return ("severity map", "第3章 标签构造", "展示严重性标签或最终标签形态", "maybe")
    if "fft" in lower and "label" in lower:
        return ("FFT label map", "第3章 标签构造", "展示FFT标签生成流程", "maybe")
    if "model_architecture" in lower:
        return ("model architecture", "第4章 模型结构", "展示模型结构图", "maybe")
    if "scatter" in lower:
        return ("scatter plot", "第5章 实验结果", "展示预测与真值散点关系", "no")
    if "profile_label" in lower or "training_history_regression" in lower:
        return ("depth prediction curve", "第5章 实验结果", "展示深度剖面标签或训练过程", "maybe")
    if "truth_vs_prediction" in lower:
        return ("FFT prediction map", "第5章 实验结果", "展示FFT/重构图预测与真值对比", "no")
    if "error" in lower or "performance" in lower or "summary" in lower:
        return ("error map", "第5章 实验结果", "展示性能或误差概况", "maybe")
    if "gradcam" in lower or "attention" in lower:
        return ("Grad-CAM heatmap", "第6章 可解释性分析", "展示模型敏感时频区域", "no")
    if "baseline" in lower:
        return ("baseline comparison", "第5章 对比实验", "展示baseline结果或失败形态", "maybe")
    if "frequency-weighted" in lower or "log_label" in lower:
        return ("ablation", "第5章 消融实验", "展示log/weighted loss变体", "maybe")
    if "gan" in lower or "双通道" in relative_path or "预校正" in relative_path:
        return ("failed attempt", "附录或讨论章", "展示失败路线证据", "yes")
    return ("limitation evidence", "第7章 讨论", "展示限制或补充证据", "maybe")


def build_figure_candidates(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    images = [f for f in files if f["likely_artifact_type"] == "image"]
    category_counts: Counter[str] = Counter()
    rows = []
    for f in images:
        category, chapter, useful, redraw = image_candidate_category(f["relative_path"])
        if category_counts[category] >= 12:
            continue
        # Keep more of strong categories but avoid thousands of repetitive candidates.
        if category in {"limitation evidence"} and category_counts[category] >= 10:
            continue
        category_counts[category] += 1
        size = int(f["size_bytes"])
        quality = "high" if size > 200_000 else "medium" if size > 50_000 else "low"
        figure_id = f"FIG-{len(rows)+1:03d}"
        rows.append(
            {
                "figure_id": figure_id,
                "category": category,
                "source_path": (RESULTS / f["relative_path"]).as_posix(),
                "caption_draft_cn": caption_for_category(category, f["relative_path"]),
                "thesis_chapter": chapter,
                "why_useful": useful,
                "quality": quality,
                "needs_redraw": redraw,
                "notes": f"Artifact type: {f['likely_artifact_type']}; size={human_size(size)}; evidence path only, image not modified.",
            }
        )
    rows.sort(key=lambda r: (r["category"], r["figure_id"]))
    # Re-number after sorting.
    for idx, row in enumerate(rows, start=1):
        row["figure_id"] = f"FIG-{idx:03d}"
    return rows


def caption_for_category(category: str, rel: str) -> str:
    captions = {
        "数据示意图": "原始声波信号及窜槽分布示意图",
        "CWT 示例图": "声波信号连续小波变换后的时频图示例",
        "severity map": "基于 Zc 阈值构建的窜槽严重性标签示例",
        "FFT label map": "方位维 FFT 幅值标签生成示例",
        "model architecture": "模型结构示意图",
        "scatter plot": "预测值与真值散点对比图",
        "depth prediction curve": "深度方向窜槽比例标签或预测曲线示例",
        "FFT prediction map": "FFT 标签预测与真值对比图",
        "error map": "模型性能或误差分布概览图",
        "Grad-CAM heatmap": "Grad-CAM 显示的敏感时频区域",
        "baseline comparison": "基线方法预测结果与局限性示例",
        "ablation": "标签或损失函数改进的消融实验结果示例",
        "failed attempt": "失败路线的实验结果或诊断图",
        "limitation evidence": "方法局限性或补充诊断图",
    }
    return captions.get(category, "实验结果图候选") + f"（来源：{rel}）"


def write_figure_tables(rows: list[dict[str, Any]]) -> None:
    fields = [
        "figure_id",
        "category",
        "source_path",
        "caption_draft_cn",
        "thesis_chapter",
        "why_useful",
        "quality",
        "needs_redraw",
        "notes",
    ]
    csv_dump(OUT / "figure_candidates.csv", rows, fields)
    md = [
        "# Figure Candidates",
        "",
        "Candidate figures were selected from existing result images. No image was modified.",
        "",
        f"- Candidate figures: `{len(rows)}`",
        "",
        md_table(rows, ["figure_id", "category", "source_path", "caption_draft_cn", "thesis_chapter", "quality", "needs_redraw"], max_rows=180),
    ]
    (OUT / "figure_candidates.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    table_rows = [
        {
            "table_id": "TAB-001",
            "table_name": "实验台账总表",
            "source": "docs/thesis_evidence/experiment_inventory.csv",
            "chapter": "第5章 实验设计与结果",
            "notes": "作为论文实验路线总览表。",
        },
        {
            "table_id": "TAB-002",
            "table_name": "方法路线与标签构造对比表",
            "source": "docs/thesis_evidence/method_taxonomy.md; code_method_inventory.csv",
            "chapter": "第3章 方法",
            "notes": "对比 1D percentage、FFT magnitude、log、severity transform。",
        },
        {
            "table_id": "TAB-003",
            "table_name": "Git 分支与实验映射表",
            "source": "docs/thesis_evidence/branch_experiment_mapping.csv",
            "chapter": "附录或实验复现说明",
            "notes": "用于证明旧实验版本来源。",
        },
        {
            "table_id": "TAB-004",
            "table_name": "图件候选与证据路径表",
            "source": "docs/thesis_evidence/figure_candidates.csv",
            "chapter": "写作素材管理",
            "notes": "写论文时按章节挑图。",
        },
        {
            "table_id": "TAB-005",
            "table_name": "缺失证据与最小补充计划",
            "source": "docs/thesis_evidence/missing_evidence.md; minimal_supplement_plan.md",
            "chapter": "第7章 讨论/后续工作",
            "notes": "标记必须补、可选补和不建议补。",
        },
    ]
    md2 = ["# Table Candidates", "", md_table(table_rows, ["table_id", "table_name", "source", "chapter", "notes"])]
    (OUT / "table_candidates.md").write_text("\n".join(md2) + "\n", encoding="utf-8")


def write_missing_evidence() -> None:
    missing = [
        {
            "priority": "A 必须补",
            "item": "防止数据泄漏的 split 说明或 depth-blocked 验证",
            "reason": "现有代码多处使用 VALIDATION_SPLIT + dataset take/skip，未证明同一井深邻近窗口不会跨 train/val。",
            "minimal_action": "不训练；先从 TFRecord/索引/脚本恢复样本深度顺序，写出 split 说明。如需补，运行轻量 depth-blocked 评估脚本。",
        },
        {
            "priority": "A 必须补",
            "item": "baseline 对比",
            "reason": "论文主线需要明确比 baseline 好；现有 baseline 多为定性失败说明。",
            "minimal_action": "从现有 results 和日志提取可比指标；缺失则标为 missing，不重训。",
        },
        {
            "priority": "A 必须补",
            "item": "消融实验",
            "reason": "log、frequency weighted、severity/FFT、1D percentage 已有路线，但指标不统一。",
            "minimal_action": "整理已有图和日志，统一表述为路线对比；只在必要时做轻量评估，不做训练。",
        },
        {
            "priority": "A 必须补",
            "item": "Grad-CAM 批量统计",
            "reason": "现有 Grad-CAM 多为样本图/定性 memo，论文需要稳定性证据。",
            "minimal_action": "优先使用现有 comprehensive_gradcam_statistics.png 和 attention plots；若缺统计脚本，写读取已保存热图/图像的轻量汇总。",
        },
        {
            "priority": "A 必须补",
            "item": "严重度分组误差分析",
            "reason": "memo 提到严重窜槽预测偏保守，但缺统一误差表。",
            "minimal_action": "从现有 prediction/result artifacts 中寻找已保存数组/CSV；缺失则列 missing。",
        },
        {
            "priority": "A 必须补",
            "item": "旧实验顺序验证",
            "reason": "memo、结果目录、Git 分支能形成大体顺序，但 CSI+SE-ResNet、双通道、预校正缺本地分支映射。",
            "minimal_action": "检查远程服务器 `/home/xiaoj/hal_azi` 和 GitHub 分支，不训练。",
        },
        {
            "priority": "B 可选补",
            "item": "更多模型/超参数/多井/高级信号处理",
            "reason": "可提高论文完整度，但不是最短毕业路线。",
            "minimal_action": "仅作为后续工作，不进入当前主线。",
        },
        {
            "priority": "C 不建议补",
            "item": "大规模重训、新版本复杂弱标签、STC/APES、多目标人工审核",
            "reason": "会扩大风险并偏离旧项目证据重建目标。",
            "minimal_action": "明确不做，除非导师要求并单独批准。",
        },
    ]
    md = [
        "# Missing Evidence",
        "",
        md_table(missing, ["priority", "item", "reason", "minimal_action"]),
    ]
    (OUT / "missing_evidence.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    plan = [
        {
            "step": "1",
            "task": "只读核对远程服务器/GitHub 分支",
            "output": "更新 branch_experiment_mapping.csv 和 git_branch_timeline.md",
            "training": "no",
        },
        {
            "step": "2",
            "task": "从现有 logs/history/result.txt/pkl 中抽取统一指标",
            "output": "新增 metrics_summary.csv（当前未生成，因为需确认 pkl 读取范围）",
            "training": "no",
        },
        {
            "step": "3",
            "task": "补 depth-blocked split 说明",
            "output": "split_verification.md 或 missing_evidence 更新",
            "training": "no or lightweight evaluation only",
        },
        {
            "step": "4",
            "task": "挑选并重绘少量论文图",
            "output": "论文图编号与 captions",
            "training": "no",
        },
        {
            "step": "5",
            "task": "写论文主线：问题 -> CWT -> severity/FFT/1D 标签 -> EfficientNet -> Grad-CAM -> 局限",
            "output": "thesis chapter draft",
            "training": "no",
        },
    ]
    (OUT / "minimal_supplement_plan.md").write_text(
        "# Minimal Supplement Plan\n\n" + md_table(plan, ["step", "task", "output", "training"]) + "\n",
        encoding="utf-8",
    )

    risks = [
        {
            "risk_id": "R-001",
            "risk": "把 memo 定性结论写成未证实的定量结论",
            "impact": "high",
            "mitigation": "所有结论引用 memo/result/code/git；没有指标则写 unknown/needs_verification。",
        },
        {
            "risk_id": "R-002",
            "risk": "train/validation 深度泄漏",
            "impact": "high",
            "mitigation": "补 depth-blocked split 说明或承认为限制。",
        },
        {
            "risk_id": "R-003",
            "risk": "结果目录与代码分支映射错误",
            "impact": "medium",
            "mitigation": "用 Git commit、result modified_time、服务器目录核对。",
        },
        {
            "risk_id": "R-004",
            "risk": "FFT 主线性能不足",
            "impact": "medium",
            "mitigation": "把 1D percentage EfficientNet 作为强证据主结果，FFT 作为角度不匹配方法探索/局限。",
        },
        {
            "risk_id": "R-005",
            "risk": "为了补证据触发大规模重训",
            "impact": "medium",
            "mitigation": "当前阶段只做整理和轻量验证；训练需单独审批。",
        },
    ]
    (OUT / "risk_register.md").write_text(
        "# Risk Register\n\n" + md_table(risks, ["risk_id", "risk", "impact", "mitigation"]) + "\n",
        encoding="utf-8",
    )


def write_thesis_outline(experiments: list[dict[str, Any]], figures: list[dict[str, Any]]) -> None:
    title_suggestions = [
        "基于声波时频特征与CAST胶结成像标签的水泥窜槽识别方法研究",
        "面向方位失配井段的XSI-CAST水泥窜槽弱监督表征与识别研究",
        "基于CWT-EfficientNet与FFT方位不变标签的水泥窜槽严重度预测研究",
    ]
    outline_rows = [
        {
            "chapter": "第1章 绪论",
            "experiments": "EXP-006; EXP-007; EXP-008 as motivation",
            "figures": "数据示意图; limitation evidence",
            "existing_evidence": "memo PPT section; result tree inventory",
            "missing_evidence": "工程背景文字和引用文献需要另补",
            "minimal_experiment": "none",
        },
        {
            "chapter": "第2章 数据与问题定义",
            "experiments": "all data construction code",
            "figures": "数据示意图; CWT 示例图; severity map",
            "existing_evidence": "config.py; CWT scripts; result visualization plots",
            "missing_evidence": "raw data provenance and split/depth-blocked validation",
            "minimal_experiment": "split audit only",
        },
        {
            "chapter": "第3章 标签构造与方位失配处理",
            "experiments": "EXP-001; EXP-002; EXP-003; EXP-007; EXP-008",
            "figures": "FFT label map; severity map; depth prediction curve",
            "existing_evidence": "create_tfrecords code across branches; memo",
            "missing_evidence": "FFT route final metric verification",
            "minimal_experiment": "existing-artifact metric extraction",
        },
        {
            "chapter": "第4章 模型方法",
            "experiments": "EXP-006; EXP-007; EXP-008; EXP-010",
            "figures": "model architecture",
            "existing_evidence": "model.py across branches; model_architecture images",
            "missing_evidence": "SE-ResNet code mapping",
            "minimal_experiment": "server branch inventory",
        },
        {
            "chapter": "第5章 实验结果与消融",
            "experiments": "EXP-001 to EXP-008; EXP-011; EXP-012",
            "figures": "scatter plot; FFT prediction map; baseline comparison; ablation; failed attempt",
            "existing_evidence": "result plots/logs/memo",
            "missing_evidence": "unified baseline/ablation numeric table",
            "minimal_experiment": "metric extraction from saved artifacts only",
        },
        {
            "chapter": "第6章 可解释性分析",
            "experiments": "EXP-013; EXP-006; EXP-007",
            "figures": "Grad-CAM heatmap; CWT example",
            "existing_evidence": "Grad-CAM plots/statistics and memo sensitive-region claims",
            "missing_evidence": "batch-level Grad-CAM statistics if not already sufficient",
            "minimal_experiment": "aggregate existing Grad-CAM artifacts",
        },
        {
            "chapter": "第7章 讨论与结论",
            "experiments": "all failed routes and mainline limitations",
            "figures": "limitation evidence",
            "existing_evidence": "memo failure records; missing_evidence.md; risk_register.md",
            "missing_evidence": "导师确认最终主线取舍",
            "minimal_experiment": "none",
        },
    ]
    claims = [
        {
            "claim_id": "C-001",
            "claim": "CWT 时频图与窜槽存在性之间存在可学习关系。",
            "evidence": "EXP-006; temp_result/test_relativity/result.txt.txt; memo AUC=0.95361",
            "status": "supported",
        },
        {
            "claim_id": "C-002",
            "claim": "1D 窜槽百分比标签规避方位失配并得到较稳定敏感区域。",
            "evidence": "EXP-007; origin/1D+percentage_Label code; memo section 6",
            "status": "supported, metric table still needed",
        },
        {
            "claim_id": "C-003",
            "claim": "FFT 幅值/丢弃相位是处理方位旋转不变性的合理标签路线。",
            "evidence": "memo PPT section; create_tfrecords FFT code",
            "status": "method rationale supported, performance needs verification",
        },
        {
            "claim_id": "C-004",
            "claim": "log transform improves Grad-CAM localization but does not solve prediction quality.",
            "evidence": "EXP-002; memo section 1; temp_result/log_label plots",
            "status": "supported by memo/result artifacts",
        },
        {
            "claim_id": "C-005",
            "claim": "frequency-weighted FFT loss did not solve low-coefficient collapse.",
            "evidence": "EXP-003; memo section 2; origin/frequency-weighted_loss train.py",
            "status": "supported by memo/code",
        },
        {
            "claim_id": "C-006",
            "claim": "GAN/two-channel image-generation routes collapsed.",
            "evidence": "EXP-004; EXP-005; memo sections 3-4; result.txt losses",
            "status": "supported for GAN, two-channel implementation mapping needs verification",
        },
        {
            "claim_id": "C-007",
            "claim": "Dual-channel and pre-correction failed/poor.",
            "evidence": "result directories 双通道学习 and 预校正; user task framing",
            "status": "explicitly_marked_inference_needs_verification",
        },
        {
            "claim_id": "C-008",
            "claim": "Sensitive CWT region is concentrated in high-frequency bands roughly 22-30 kHz and 0.5-1.3 ms depending on route.",
            "evidence": "memo sections 1/2/6; temp_result/test_relativity/result.txt.txt; Grad-CAM result files",
            "status": "supported qualitatively; batch statistics recommended",
        },
    ]
    md = [
        "# Thesis Outline Draft",
        "",
        "## Title Suggestions",
        "",
        "\n".join(f"- {title}" for title in title_suggestions),
        "",
        "## Abstract Core Logic",
        "",
        "垂直井中 XSI 声波与 CAST 胶结图像存在方位失配，直接做点对点方位监督容易失败。旧实验显示，CWT 时频图能够稳定学习窜槽存在性；进一步将 CAST Zc 构造成深度方向百分比标签或方位 FFT 幅值标签，可以分别形成实用的深度严重度预测路线和方位不变的角度失配处理路线。论文主线建议以 CWT + EfficientNet + severity/FFT 或 1D percentage 标签为核心，用 baseline/log/weighted/GAN/dual-channel/pre-correction 作为对比和失败路线，并用 Grad-CAM 解释模型关注的高频时频区域。",
        "",
        "## Chapter Plan",
        "",
        md_table(outline_rows, ["chapter", "experiments", "figures", "existing_evidence", "missing_evidence", "minimal_experiment"]),
        "",
        "## Suggested Figure Categories",
        "",
        md_table(figures[:40], ["figure_id", "category", "source_path", "thesis_chapter", "quality", "needs_redraw"], max_rows=40),
    ]
    (OUT / "thesis_outline.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    trace_md = [
        "# Thesis Claims Traceability",
        "",
        "Every claim below is tied to at least one result path, code path, git branch/commit, memo path, or explicit inference marker.",
        "",
        md_table(claims, ["claim_id", "claim", "evidence", "status"]),
    ]
    (OUT / "thesis_claims_traceability.md").write_text("\n".join(trace_md) + "\n", encoding="utf-8")


def lightweight_checks() -> dict[str, Any]:
    checks = {
        "python_compile_script": run_cmd(["python3", "-m", "py_compile", rel_repo(Path(__file__).resolve())]),
        "json_files_valid": {},
        "csv_files_present": {},
    }
    for path in sorted(OUT.glob("*.json")):
        try:
            json.loads(path.read_text(encoding="utf-8"))
            checks["json_files_valid"][path.name] = True
        except Exception as exc:
            checks["json_files_valid"][path.name] = f"ERROR: {exc}"
    for path in sorted(OUT.glob("*.csv")):
        checks["csv_files_present"][path.name] = path.stat().st_size
    json_dump(OUT / "generation_checks.json", checks)
    return checks


def main() -> int:
    ensure_out()
    files, result_summary = scan_results()
    preflight = generate_preflight(result_summary)
    write_result_inventory(files, result_summary)
    memo_claims, _text_files = generate_text_evidence(files)
    code_rows = code_inventory()
    write_code_inventory(code_rows)
    git_json, branch_mapping = git_inventory()
    write_git_inventory(git_json, branch_mapping)
    experiments = experiment_registry(files, memo_claims)
    write_experiment_registry(experiments)
    write_method_taxonomy(experiments)
    figures = build_figure_candidates(files)
    write_figure_tables(figures)
    write_missing_evidence()
    write_thesis_outline(experiments, figures)
    checks = lightweight_checks()
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results_file_count": len(files),
        "experiment_count": len(experiments),
        "figure_candidate_count": len(figures),
        "code_inventory_count": len(code_rows),
        "memo_claim_count": len(memo_claims),
        "current_branch": preflight["current_branch"],
        "current_commit_before_commit": preflight["current_commit"],
        "checks": checks,
    }
    json_dump(OUT / "inventory_generation_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
