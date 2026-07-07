#!/usr/bin/env python3
"""Audit EXP-008 depth-heldout train_v001 without training.

The script reads existing split TFRecords and train_v001 outputs, computes
simple train-based baselines, summarizes error structure, and redraws small
thesis figures. It does not modify raw data, processed data, split files, or
Windows result evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


LABEL_SHAPE = (70, 30)
DEFAULT_RUN_DIR = Path("output/thesis_depth_blocked/exp008/train_v001")
DEFAULT_SPLIT_DIR = Path("output/thesis_depth_blocked/exp008/split_v001")
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "audit_v001"
WINDOWS_RESULTS_ROOT = Path("/mnt/c/Users/Administrator/Desktop/Hal/results")


@dataclass
class MetricSet:
    overall_mae: float
    overall_mse: float
    overall_rmse: float
    overall_r2: float | None
    overall_pearson: float | None
    overall_spearman: float | None
    dc_mae: float
    dc_rmse: float
    low_freq_0_5_mae: float
    low_freq_0_5_rmse: float
    mid_freq_6_14_mae: float
    mid_freq_6_14_rmse: float
    high_freq_15_29_mae: float
    high_freq_15_29_rmse: float
    num_samples: int
    num_label_values: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_non_strict(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def ensure_safe_output_dir(output_dir: Path) -> None:
    resolved_output = resolve_non_strict(output_dir)
    resolved_results = resolve_non_strict(WINDOWS_RESULTS_ROOT)
    if resolved_output == resolved_results or resolved_results in resolved_output.parents:
        raise ValueError(f"Refusing to write inside Windows results evidence directory: {resolved_output}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def parse_label_only(example_proto):
    import tensorflow as tf  # type: ignore

    feature_description = {
        "label": tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    label = tf.io.parse_tensor(parsed["label"], out_type=tf.float32)
    return tf.reshape(label, LABEL_SHAPE)


def load_labels_from_tfrecord(path: Path) -> np.ndarray:
    import tensorflow as tf  # type: ignore

    if not path.exists():
        raise FileNotFoundError(path)
    labels: list[np.ndarray] = []
    dataset = tf.data.TFRecordDataset(str(path)).map(parse_label_only)
    for label in dataset:
        labels.append(label.numpy())
    if not labels:
        raise ValueError(f"No labels loaded from {path}")
    return np.stack(labels, axis=0).astype(np.float32)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> MetricSet:
    diff = y_pred.astype(np.float64) - y_true.astype(np.float64)
    abs_diff = np.abs(diff)
    squared = diff * diff
    flat_true = y_true.astype(np.float64).ravel()
    flat_pred = y_pred.astype(np.float64).ravel()
    denom = float(np.sum((flat_true - np.mean(flat_true)) ** 2))
    r2 = None if denom <= 0 else 1.0 - float(np.sum((flat_true - flat_pred) ** 2)) / denom

    pearson = None
    if np.std(flat_true) > 0 and np.std(flat_pred) > 0:
        pearson = float(np.corrcoef(flat_true, flat_pred)[0, 1])

    spearman = None
    try:
        from scipy.stats import spearmanr  # type: ignore

        stat = spearmanr(flat_true, flat_pred)
        spearman = safe_float(stat.statistic)
    except Exception:
        spearman = None

    def subset(a: np.ndarray, b: np.ndarray, start: int, stop: int) -> tuple[float, float]:
        local_diff = b[:, :, start:stop] - a[:, :, start:stop]
        return float(np.mean(np.abs(local_diff))), float(np.sqrt(np.mean(local_diff * local_diff)))

    low_mae, low_rmse = subset(y_true, y_pred, 0, 6)
    mid_mae, mid_rmse = subset(y_true, y_pred, 6, 15)
    high_mae, high_rmse = subset(y_true, y_pred, 15, 30)
    dc_true = y_true[:, :, 0]
    dc_pred = y_pred[:, :, 0]

    return MetricSet(
        overall_mae=float(np.mean(abs_diff)),
        overall_mse=float(np.mean(squared)),
        overall_rmse=float(np.sqrt(np.mean(squared))),
        overall_r2=safe_float(r2),
        overall_pearson=safe_float(pearson),
        overall_spearman=safe_float(spearman),
        dc_mae=float(np.mean(np.abs(dc_pred - dc_true))),
        dc_rmse=float(np.sqrt(np.mean((dc_pred - dc_true) ** 2))),
        low_freq_0_5_mae=low_mae,
        low_freq_0_5_rmse=low_rmse,
        mid_freq_6_14_mae=mid_mae,
        mid_freq_6_14_rmse=mid_rmse,
        high_freq_15_29_mae=high_mae,
        high_freq_15_29_rmse=high_rmse,
        num_samples=int(y_true.shape[0]),
        num_label_values=int(y_true.size),
    )


def metric_rows(metrics_by_predictor: dict[str, MetricSet]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for predictor, metrics in metrics_by_predictor.items():
        for metric_name, value in asdict(metrics).items():
            rows.append({"predictor": predictor, "metric_name": metric_name, "metric_value": value})
    return rows


def make_constant_prediction(template: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return np.broadcast_to(vector.astype(np.float32), template.shape).copy()


def artifact_audit(run_dir: Path, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[list[dict[str, Any]], list[str]]:
    required = [
        "run_config.json",
        "training_history.csv",
        "training_history.json",
        "val_metrics.json",
        "test_metrics.json",
        "prediction_summary.csv",
        "prediction_vs_truth_scatter.png",
        "training_curve.png",
        "run_report.md",
    ]
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for name in required:
        path = run_dir / name
        rows.append(
            {
                "artifact": name,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else "",
            }
        )
        if not path.exists():
            warnings.append(f"missing_artifact:{name}")

    test_metrics_path = run_dir / "test_metrics.json"
    if test_metrics_path.exists():
        stored = read_json(test_metrics_path)
        recomputed = asdict(regression_metrics(y_true, y_pred))
        for key in ("overall_mae", "overall_rmse", "overall_r2", "overall_pearson", "overall_spearman", "dc_mae"):
            stored_value = safe_float(stored.get(key))
            recomputed_value = safe_float(recomputed.get(key))
            if stored_value is None or recomputed_value is None:
                warnings.append(f"metric_compare_missing:{key}")
                continue
            if abs(stored_value - recomputed_value) > 1e-6:
                warnings.append(f"metric_mismatch:{key}:stored={stored_value}:recomputed={recomputed_value}")

    prediction_summary_path = run_dir / "prediction_summary.csv"
    if prediction_summary_path.exists():
        summary_rows = load_csv_rows(prediction_summary_path)
        if len(summary_rows) != int(y_true.shape[0]):
            warnings.append(f"prediction_summary_count_mismatch:{len(summary_rows)}!={y_true.shape[0]}")

    history_path = run_dir / "training_history.csv"
    val_metrics_path = run_dir / "val_metrics.json"
    if history_path.exists() and val_metrics_path.exists():
        history_rows = load_csv_rows(history_path)
        val_metrics = read_json(val_metrics_path)
        if history_rows:
            best_val_loss_row = min(history_rows, key=lambda row: float(row["val_loss"]))
            if abs(float(best_val_loss_row["val_loss"]) - float(val_metrics["keras_loss"])) > 1e-6:
                warnings.append("val_loss_does_not_match_best_epoch_by_val_loss")

    return rows, warnings


def sample_integrated_severity_from_dc(label: np.ndarray) -> np.ndarray:
    dc = np.clip(label[:, :, 0], -20.0, 20.0)
    severity = np.exp(dc) - 1.0
    severity = np.where(severity > 0.01, severity, 0.0)
    return np.mean(severity, axis=1)


def severity_bin(value: float) -> str:
    if value <= 0:
        return "zero"
    if value < 1:
        return "low_0_1"
    if value < 5:
        return "mid_1_5"
    return "high_ge_5"


def build_error_structure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_predictions: dict[str, np.ndarray],
    prediction_summary_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    abs_model = np.abs(y_pred - y_true)
    rows: list[dict[str, Any]] = []

    per_coeff_model = np.mean(abs_model, axis=(0, 1))
    for coeff in range(y_true.shape[2]):
        row: dict[str, Any] = {
            "analysis_type": "per_fft_coefficient",
            "index": coeff,
            "label": f"k={coeff}",
            "model_mae": float(per_coeff_model[coeff]),
            "target_mean_abs": float(np.mean(np.abs(y_true[:, :, coeff]))),
        }
        for name, pred in baseline_predictions.items():
            row[f"{name}_mae"] = float(np.mean(np.abs(pred[:, :, coeff] - y_true[:, :, coeff])))
        rows.append(row)

    per_depth_model = np.mean(abs_model, axis=(0, 2))
    for depth_idx in range(y_true.shape[1]):
        row = {
            "analysis_type": "per_depth_within_window",
            "index": depth_idx,
            "label": f"depth_index={depth_idx}",
            "model_mae": float(per_depth_model[depth_idx]),
            "target_mean_abs": float(np.mean(np.abs(y_true[:, depth_idx, :]))),
        }
        for name, pred in baseline_predictions.items():
            row[f"{name}_mae"] = float(np.mean(np.abs(pred[:, depth_idx, :] - y_true[:, depth_idx, :])))
        rows.append(row)

    flat_abs = abs_model.ravel()
    quantile_levels = [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    quantiles = {str(level): float(np.quantile(flat_abs, level)) for level in quantile_levels}
    for level, value in quantiles.items():
        rows.append(
            {
                "analysis_type": "absolute_error_quantile",
                "index": level,
                "label": f"q={level}",
                "model_mae": value,
                "target_mean_abs": "",
            }
        )

    sample_mae = np.mean(abs_model, axis=(1, 2))
    sample_rmse = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=(1, 2)))
    true_integrated = sample_integrated_severity_from_dc(y_true)
    pred_integrated = sample_integrated_severity_from_dc(y_pred)
    top_indices = np.argsort(sample_mae)[-20:][::-1]
    top_samples: list[dict[str, Any]] = []
    for rank, idx in enumerate(top_indices, start=1):
        meta = prediction_summary_rows[idx] if idx < len(prediction_summary_rows) else {}
        item = {
            "rank": rank,
            "eval_index": int(idx),
            "record_index": meta.get("record_index", ""),
            "sample_index": meta.get("sample_index", ""),
            "depth_ft": meta.get("depth_ft", ""),
            "sample_mae": float(sample_mae[idx]),
            "sample_rmse": float(sample_rmse[idx]),
            "true_integrated_severity": float(true_integrated[idx]),
            "pred_integrated_severity": float(pred_integrated[idx]),
        }
        top_samples.append(item)
        rows.append(
            {
                "analysis_type": "top_error_sample",
                "index": rank,
                "label": f"eval_index={idx}",
                "model_mae": item["sample_mae"],
                "target_mean_abs": float(np.mean(np.abs(y_true[idx]))),
                "record_index": item["record_index"],
                "sample_index": item["sample_index"],
                "depth_ft": item["depth_ft"],
                "sample_rmse": item["sample_rmse"],
                "true_integrated_severity": item["true_integrated_severity"],
                "pred_integrated_severity": item["pred_integrated_severity"],
            }
        )

    bin_names = [severity_bin(float(value)) for value in true_integrated]
    for bin_name in sorted(set(bin_names)):
        idx = np.array([name == bin_name for name in bin_names])
        if not np.any(idx):
            continue
        local_true = y_true[idx]
        local_pred = y_pred[idx]
        local_diff = local_pred - local_true
        rows.append(
            {
                "analysis_type": "integrated_severity_bin",
                "index": bin_name,
                "label": bin_name,
                "model_mae": float(np.mean(np.abs(local_diff))),
                "model_rmse": float(np.sqrt(np.mean(local_diff * local_diff))),
                "sample_count": int(np.sum(idx)),
                "target_mean_abs": float(np.mean(np.abs(local_true))),
            }
        )

    summary = {
        "generated_at": utc_now(),
        "label_shape": list(y_true.shape[1:]),
        "overall_label_mae": float(np.mean(abs_model)),
        "coefficient_mae": [float(value) for value in per_coeff_model],
        "best_coefficients_by_mae": [
            {"coefficient": int(idx), "mae": float(per_coeff_model[idx])}
            for idx in np.argsort(per_coeff_model)[:5]
        ],
        "worst_coefficients_by_mae": [
            {"coefficient": int(idx), "mae": float(per_coeff_model[idx])}
            for idx in np.argsort(per_coeff_model)[-5:][::-1]
        ],
        "depth_index_mae": [float(value) for value in per_depth_model],
        "best_depth_indices_by_mae": [
            {"depth_index": int(idx), "mae": float(per_depth_model[idx])}
            for idx in np.argsort(per_depth_model)[:5]
        ],
        "worst_depth_indices_by_mae": [
            {"depth_index": int(idx), "mae": float(per_depth_model[idx])}
            for idx in np.argsort(per_depth_model)[-5:][::-1]
        ],
        "absolute_error_quantiles": quantiles,
        "top_error_samples": top_samples,
        "severity_bin_definitions": {
            "zero": "mean integrated severity <= 0",
            "low_0_1": "0 < mean integrated severity < 1",
            "mid_1_5": "1 <= mean integrated severity < 5",
            "high_ge_5": "mean integrated severity >= 5",
        },
    }
    return rows, summary


def plot_outputs(
    output_dir: Path,
    history_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prediction_summary_rows: list[dict[str, str]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    figures_dir = output_dir / "figures_redraw"
    figures_dir.mkdir(parents=True, exist_ok=True)

    epochs = [int(row["epoch"]) for row in history_rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [float(row["loss"]) for row in history_rows], label="train loss")
    axes[0].plot(epochs, [float(row["val_loss"]) for row in history_rows], label="validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Huber loss")
    axes[0].set_title("EXP-008 Depth-Heldout Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, [float(row["mae"]) for row in history_rows], label="train MAE")
    axes[1].plot(epochs, [float(row["val_mae"]) for row in history_rows], label="validation MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("EXP-008 Depth-Heldout MAE")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "training_curve_redraw.png", dpi=180)
    plt.close(fig)

    if prediction_summary_rows:
        true_vals = np.array([float(row["true_mean_integrated_severity"]) for row in prediction_summary_rows])
        pred_vals = np.array([float(row["pred_mean_integrated_severity"]) for row in prediction_summary_rows])
    else:
        true_vals = sample_integrated_severity_from_dc(y_true)
        pred_vals = sample_integrated_severity_from_dc(y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.scatter(true_vals, pred_vals, s=14, alpha=0.65, edgecolors="none")
    max_val = float(max(np.max(true_vals), np.max(pred_vals), 1.0))
    ax.plot([0, max_val], [0, max_val], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("True mean integrated severity")
    ax.set_ylabel("Predicted mean integrated severity")
    ax.set_title("EXP-008 Depth-Heldout Test")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "prediction_vs_truth_scatter_redraw.png", dpi=180)
    plt.close(fig)

    metric_lookup = {
        (row["predictor"], row["metric_name"]): float(row["metric_value"])
        for row in baseline_rows
        if row["metric_value"] not in ("", None)
    }
    predictors = [name for name in ("model", "zero", "train_mean", "train_median", "val_mean_oracle_analysis") if (name, "overall_mae") in metric_lookup]
    x = np.arange(len(predictors))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.bar(x - width / 2, [metric_lookup[(p, "overall_mae")] for p in predictors], width, label="MAE")
    ax1.bar(x + width / 2, [metric_lookup[(p, "overall_rmse")] for p in predictors], width, label="RMSE")
    ax1.set_xticks(x)
    ax1.set_xticklabels(predictors, rotation=20, ha="right")
    ax1.set_ylabel("Label units")
    ax1.set_title("EXP-008 Depth-Heldout Test Baselines")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "baseline_comparison_bar.png", dpi=180)
    plt.close(fig)

    coeff_rows = [row for row in error_rows if row["analysis_type"] == "per_fft_coefficient"]
    coeff_rows.sort(key=lambda row: int(row["index"]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    coeffs = [int(row["index"]) for row in coeff_rows]
    ax.plot(coeffs, [float(row["model_mae"]) for row in coeff_rows], marker="o", label="model")
    for name in ("zero", "train_mean", "train_median"):
        key = f"{name}_mae"
        if key in coeff_rows[0]:
            ax.plot(coeffs, [float(row[key]) for row in coeff_rows], linestyle="--", label=name)
    ax.set_xlabel("FFT coefficient k")
    ax.set_ylabel("MAE")
    ax.set_title("Per-Coefficient MAE on Depth-Heldout Test")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "per_fft_coefficient_mae.png", dpi=180)
    plt.close(fig)

    sample_mae = np.mean(np.abs(y_pred - y_true), axis=(1, 2))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(sample_mae, bins=32, color="#4c78a8", alpha=0.85)
    ax.set_xlabel("Sample MAE")
    ax.set_ylabel("Count")
    ax.set_title("EXP-008 Depth-Heldout Test Error Distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "error_distribution.png", dpi=180)
    plt.close(fig)

    true_low = np.mean(y_true[:, :, :6], axis=(1, 2))
    pred_low = np.mean(y_pred[:, :, :6], axis=(1, 2))
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.scatter(true_low, pred_low, s=14, alpha=0.65, edgecolors="none")
    max_val = float(max(np.max(true_low), np.max(pred_low), 1.0))
    min_val = float(min(np.min(true_low), np.min(pred_low), 0.0))
    ax.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("True low-frequency label mean (k=0-5)")
    ax.set_ylabel("Predicted low-frequency label mean (k=0-5)")
    ax.set_title("Low-Frequency Label Summary")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "target_vs_prediction_lowfreq_summary.png", dpi=180)
    plt.close(fig)

    captions = """# EXP-008 Depth-Heldout Redrawn Figure Captions

All figures are single-well depth-heldout EXP-008 evidence from `array_03`. Do not describe them as multi-well generalization or deployment performance.

| file | caption_draft_cn |
| --- | --- |
| training_curve_redraw.png | EXP-008 在显式 depth-heldout 划分下的训练与验证损失/MAE 曲线。模型在第 12 轮早停并恢复第 2 轮最佳验证损失权重。 |
| prediction_vs_truth_scatter_redraw.png | EXP-008 depth-heldout 测试集平均 integrated severity 的预测-真值散点图，用于说明单井连续深度留出段上的可学习关系。 |
| baseline_comparison_bar.png | EXP-008 模型与 zero、train-mean、train-median 简单基线在 depth-heldout 测试集上的 MAE/RMSE 对比。 |
| per_fft_coefficient_mae.png | EXP-008 depth-heldout 测试集按 FFT 系数统计的 MAE，用于分析低频、中频和高频标签维度的误差结构。 |
| error_distribution.png | EXP-008 depth-heldout 测试样本级 MAE 分布，用于展示误差长尾和异常样本风险。 |
| target_vs_prediction_lowfreq_summary.png | EXP-008 depth-heldout 测试集中低频 FFT 系数均值的预测-真值关系，用于补充说明低频标签维度表现。 |
"""
    (figures_dir / "figure_captions.md").write_text(captions, encoding="utf-8")


def write_markdown_reports(
    output_dir: Path,
    artifact_rows: list[dict[str, Any]],
    artifact_warnings: list[str],
    baseline_metrics: dict[str, MetricSet],
    error_summary: dict[str, Any],
) -> None:
    artifact_lines = [
        "# EXP-008 Depth-Heldout Artifact Audit",
        "",
        f"Generated: {utc_now()}",
        "",
        "| artifact | exists | size_bytes |",
        "| --- | --- | ---: |",
    ]
    for row in artifact_rows:
        artifact_lines.append(f"| {row['artifact']} | {row['exists']} | {row['size_bytes']} |")
    artifact_lines.extend(["", "## Consistency Warnings", ""])
    if artifact_warnings:
        artifact_lines.extend(f"- {warning}" for warning in artifact_warnings)
    else:
        artifact_lines.append("- none")
    artifact_lines.extend(
        [
            "",
            "All checks are for single-well depth-heldout EXP-008 train_v001 artifacts.",
        ]
    )
    (output_dir / "exp008_depthheldout_artifact_audit.md").write_text("\n".join(artifact_lines) + "\n", encoding="utf-8")

    baseline_lines = [
        "# EXP-008 Depth-Heldout Baseline Comparison",
        "",
        "Baselines were computed without training. Train-based baselines use labels from `split_v001/train.tfrecord`; test labels and model predictions use train_v001 artifacts.",
        "",
        "| predictor | MAE | RMSE | R2 | Pearson | Spearman | DC MAE | low k=0-5 MAE | mid k=6-14 MAE | high k=15-29 MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in baseline_metrics.items():
        baseline_lines.append(
            f"| {name} | {metrics.overall_mae:.6f} | {metrics.overall_rmse:.6f} | "
            f"{'' if metrics.overall_r2 is None else f'{metrics.overall_r2:.6f}'} | "
            f"{'' if metrics.overall_pearson is None else f'{metrics.overall_pearson:.6f}'} | "
            f"{'' if metrics.overall_spearman is None else f'{metrics.overall_spearman:.6f}'} | "
            f"{metrics.dc_mae:.6f} | {metrics.low_freq_0_5_mae:.6f} | "
            f"{metrics.mid_freq_6_14_mae:.6f} | {metrics.high_freq_15_29_mae:.6f} |"
        )
    baseline_lines.extend(
        [
            "",
            "Use `train_mean` and `train_median` as thesis-safe baselines. `val_mean_oracle_analysis` is an analysis-only comparator and should not be presented as a deployable predictor.",
        ]
    )
    (output_dir / "exp008_depthheldout_baseline_comparison.md").write_text("\n".join(baseline_lines) + "\n", encoding="utf-8")

    best_coeffs = ", ".join(f"k={item['coefficient']} ({item['mae']:.6f})" for item in error_summary["best_coefficients_by_mae"])
    worst_coeffs = ", ".join(f"k={item['coefficient']} ({item['mae']:.6f})" for item in error_summary["worst_coefficients_by_mae"])
    best_depths = ", ".join(f"{item['depth_index']} ({item['mae']:.6f})" for item in error_summary["best_depth_indices_by_mae"])
    worst_depths = ", ".join(f"{item['depth_index']} ({item['mae']:.6f})" for item in error_summary["worst_depth_indices_by_mae"])
    error_lines = [
        "# EXP-008 Depth-Heldout Error Structure",
        "",
        "This is single-well depth-heldout EXP-008 analysis. It summarizes label-space error structure from existing train_v001 predictions.",
        "",
        f"- overall_label_mae: `{error_summary['overall_label_mae']:.6f}`",
        f"- most learnable FFT coefficients by MAE: {best_coeffs}",
        f"- worst FFT coefficients by MAE: {worst_coeffs}",
        f"- lowest-error depth indices: {best_depths}",
        f"- highest-error depth indices: {worst_depths}",
        "",
        "## Error Quantiles",
        "",
        "| quantile | absolute_error |",
        "| --- | ---: |",
    ]
    for quantile, value in error_summary["absolute_error_quantiles"].items():
        error_lines.append(f"| {quantile} | {value:.6f} |")
    error_lines.extend(
        [
            "",
            "## Top Error Samples",
            "",
            "| rank | eval_index | depth_ft | sample_mae | true_integrated_severity | pred_integrated_severity |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in error_summary["top_error_samples"][:10]:
        error_lines.append(
            f"| {item['rank']} | {item['eval_index']} | {item['depth_ft']} | "
            f"{item['sample_mae']:.6f} | {item['true_integrated_severity']:.6f} | "
            f"{item['pred_integrated_severity']:.6f} |"
        )
    (output_dir / "exp008_depthheldout_error_structure.md").write_text("\n".join(error_lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    ensure_safe_output_dir(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = args.run_dir / "predictions_test.npz"
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)
    predictions = np.load(predictions_path)
    y_true = predictions["y_true"].astype(np.float32)
    y_pred = predictions["y_pred"].astype(np.float32)

    train_labels = load_labels_from_tfrecord(args.split_dir / "train.tfrecord")
    val_labels = load_labels_from_tfrecord(args.split_dir / "val.tfrecord")
    test_labels = load_labels_from_tfrecord(args.split_dir / "test.tfrecord")
    label_match_max_abs = float(np.max(np.abs(test_labels - y_true)))

    zero_pred = np.zeros_like(y_true)
    train_mean = np.mean(train_labels, axis=0)
    train_median = np.median(train_labels, axis=0)
    val_mean = np.mean(val_labels, axis=0)
    baseline_predictions = {
        "zero": zero_pred,
        "train_mean": make_constant_prediction(y_true, train_mean),
        "train_median": make_constant_prediction(y_true, train_median),
        "val_mean_oracle_analysis": make_constant_prediction(y_true, val_mean),
    }
    all_predictions = {"model": y_pred, **baseline_predictions}
    baseline_metrics = {name: regression_metrics(y_true, pred) for name, pred in all_predictions.items()}

    artifact_rows, artifact_warnings = artifact_audit(args.run_dir, y_true, y_pred)
    if label_match_max_abs > 1e-6:
        artifact_warnings.append(f"test_tfrecord_label_mismatch_max_abs={label_match_max_abs}")

    prediction_summary_rows = load_csv_rows(args.run_dir / "prediction_summary.csv") if (args.run_dir / "prediction_summary.csv").exists() else []
    error_rows, error_summary = build_error_structure(y_true, y_pred, baseline_predictions, prediction_summary_rows)
    error_summary.update(
        {
            "test_tfrecord_label_match_max_abs": label_match_max_abs,
            "source_run_dir": str(args.run_dir),
            "source_split_dir": str(args.split_dir),
        }
    )

    baseline_json = {
        "generated_at": utc_now(),
        "source_run_dir": str(args.run_dir),
        "source_split_dir": str(args.split_dir),
        "baseline_scope": "single-well depth-heldout EXP-008",
        "metrics": {name: asdict(metrics) for name, metrics in baseline_metrics.items()},
        "notes": {
            "zero": "predicts all 70x30 labels as zero",
            "train_mean": "predicts every test sample as the train-label mean tensor",
            "train_median": "predicts every test sample as the train-label median tensor",
            "val_mean_oracle_analysis": "analysis-only comparator; do not present as deployable thesis baseline",
        },
    }
    write_json(args.output_dir / "exp008_depthheldout_baseline_comparison.json", baseline_json)
    write_csv(
        args.output_dir / "exp008_depthheldout_baseline_comparison.csv",
        metric_rows(baseline_metrics),
        ["predictor", "metric_name", "metric_value"],
    )
    write_json(args.output_dir / "exp008_depthheldout_error_structure.json", error_summary)
    write_csv(args.output_dir / "exp008_depthheldout_error_structure.csv", error_rows)
    write_json(
        args.output_dir / "exp008_depthheldout_artifact_audit.json",
        {
            "generated_at": utc_now(),
            "source_run_dir": str(args.run_dir),
            "source_split_dir": str(args.split_dir),
            "artifact_rows": artifact_rows,
            "warnings": artifact_warnings,
            "test_tfrecord_label_match_max_abs": label_match_max_abs,
        },
    )

    history_rows = load_csv_rows(args.run_dir / "training_history.csv")
    plot_outputs(args.output_dir, history_rows, metric_rows(baseline_metrics), error_rows, y_true, y_pred, prediction_summary_rows)
    write_markdown_reports(args.output_dir, artifact_rows, artifact_warnings, baseline_metrics, error_summary)

    print(f"audit_output_dir={args.output_dir}")
    print(json.dumps({"baseline_metrics": baseline_json["metrics"], "artifact_warnings": artifact_warnings}, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
