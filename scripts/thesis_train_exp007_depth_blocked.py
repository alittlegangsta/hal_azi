#!/usr/bin/env python3
"""Train/evaluate EXP-007 on an explicit depth-heldout TFRecord split.

This script is intentionally scoped to EXP-007. It does not create splits,
does not modify raw data, does not touch Windows result evidence, and never
uses validation_split or train_test_split. Training/validation/test inputs are
three explicit TFRecord files created by thesis_make_exp007_depth_split.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


INPUT_SHAPE = (150, 400, 8)
MAX_PATH_DEPTH_POINTS = 70
LABEL_SHAPE = (MAX_PATH_DEPTH_POINTS,)
DEFAULT_SPLIT_DIR = Path("output/thesis_depth_blocked/exp007/split_v001")
DEFAULT_OUTPUT_DIR = Path("output/thesis_depth_blocked/exp007/train_v001")
WINDOWS_RESULTS_ROOT = Path("/mnt/c/Users/Administrator/Desktop/Hal/results")
EXP007_GIT_EVIDENCE = "origin/1D+percentage_Label@e9739c8c3fc4d53e1af63dafa84e29e00b42e9c4"


@dataclass
class PretrainedStatus:
    requested: bool
    loaded: bool
    policy: str
    weights_path: str | None = None
    error: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_non_strict(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def ensure_safe_output_dir(output_dir: Path) -> None:
    resolved_output = resolve_non_strict(output_dir)
    resolved_results = resolve_non_strict(WINDOWS_RESULTS_ROOT)
    if resolved_output == resolved_results or resolved_results in resolved_output.parents:
        raise ValueError(f"Refusing to write inside Windows results evidence directory: {resolved_output}")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_git_info() -> dict[str, str]:
    def run_git(args: list[str]) -> str:
        try:
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"

    return {
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": run_git(["rev-parse", "HEAD"]),
        "status_short": run_git(["status", "--short"]),
    }


def set_reproducibility(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # type: ignore

        tf.random.set_seed(seed)
    except Exception:
        pass


def parse_exp007_example(example_proto):
    import tensorflow as tf  # type: ignore

    feature_description = {
        "feature": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.string),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    feature_tensor = tf.io.parse_tensor(parsed_example["feature"], out_type=tf.float32)
    label_tensor = tf.io.parse_tensor(parsed_example["label"], out_type=tf.float32)
    feature_tensor = tf.reshape(feature_tensor, INPUT_SHAPE)
    label_tensor = tf.reshape(label_tensor, LABEL_SHAPE)
    return feature_tensor, label_tensor


def augment_cwt(image, label):
    import tensorflow as tf  # type: ignore

    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.01, dtype=tf.float32)
    image = image + noise
    freq_mask_size = tf.random.uniform(shape=[], minval=5, maxval=20, dtype=tf.int32)
    freq_mask_pos = tf.random.uniform(shape=[], minval=0, maxval=INPUT_SHAPE[0] - freq_mask_size, dtype=tf.int32)
    mask_start = freq_mask_pos
    mask_end = freq_mask_pos + freq_mask_size
    mask_part1 = tf.ones([mask_start, INPUT_SHAPE[1], INPUT_SHAPE[2]], dtype=tf.float32)
    mask_part2 = tf.zeros([mask_end - mask_start, INPUT_SHAPE[1], INPUT_SHAPE[2]], dtype=tf.float32)
    mask_part3 = tf.ones([INPUT_SHAPE[0] - mask_end, INPUT_SHAPE[1], INPUT_SHAPE[2]], dtype=tf.float32)
    freq_mask = tf.concat([mask_part1, mask_part2, mask_part3], axis=0)
    return image * freq_mask, label


def create_dataset(
    tfrecord_path: Path,
    batch_size: int,
    *,
    training: bool,
    seed: int,
    shuffle_buffer_size: int,
    max_batches: int | None,
    augment: bool,
):
    import tensorflow as tf  # type: ignore

    if not tfrecord_path.exists():
        raise FileNotFoundError(f"TFRecord not found: {tfrecord_path}")
    dataset = tf.data.TFRecordDataset(str(tfrecord_path))
    dataset = dataset.map(parse_exp007_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(
            buffer_size=shuffle_buffer_size,
            seed=seed,
            reshuffle_each_iteration=True,
        )
        if augment:
            dataset = dataset.map(augment_cwt, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    if max_batches is not None:
        dataset = dataset.take(max_batches)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def build_exp007_model(
    *,
    learning_rate: float,
    dropout: float,
    pretrained: bool,
    pretrained_required: bool,
    gradient_clipnorm: float | None,
) -> tuple[Any, PretrainedStatus]:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras.applications import EfficientNetV2B0  # type: ignore
    from tensorflow.keras.layers import Conv2D, Dense, Dropout, GlobalAveragePooling2D, Input  # type: ignore
    from tensorflow.keras.models import Model  # type: ignore
    from tensorflow.keras.utils import get_file  # type: ignore

    inputs = Input(shape=INPUT_SHAPE, name="cwt_input")
    x = Conv2D(3, (1, 1), padding="same", name="channel_adapter")(inputs)
    base_model = EfficientNetV2B0(include_top=False, weights=None, input_tensor=x)

    status = PretrainedStatus(requested=pretrained, loaded=False, policy="required" if pretrained_required else "optional")
    if pretrained:
        weights_url = "https://storage.googleapis.com/tensorflow/keras-applications/efficientnet_v2/efficientnetv2-b0_notop.h5"
        try:
            weights_path = get_file("efficientnetv2-b0_notop.h5", weights_url, cache_subdir="models")
            base_model.load_weights(weights_path, by_name=True, skip_mismatch=True)
            status.loaded = True
            status.weights_path = str(weights_path)
        except Exception as exc:
            status.error = f"{type(exc).__name__}: {exc}"
            if pretrained_required:
                raise

    base_model.trainable = True
    y = base_model.output
    y = GlobalAveragePooling2D()(y)
    y = Dropout(dropout)(y)
    outputs = Dense(MAX_PATH_DEPTH_POINTS, activation="relu", name="profile_output")(y)
    model = Model(inputs, outputs, name="EXP007_DepthHeldout_EfficientNetV2B0")

    optimizer_kwargs: dict[str, Any] = {"learning_rate": learning_rate}
    if gradient_clipnorm is not None:
        optimizer_kwargs["clipnorm"] = gradient_clipnorm
    optimizer = tf.keras.optimizers.Adam(**optimizer_kwargs)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=["mae"])
    return model, status


def history_to_rows(history: dict[str, list[float]]) -> list[dict[str, Any]]:
    max_len = max((len(values) for values in history.values()), default=0)
    rows: list[dict[str, Any]] = []
    for epoch in range(max_len):
        row: dict[str, Any] = {"epoch": epoch + 1}
        for key, values in history.items():
            row[key] = values[epoch] if epoch < len(values) else ""
        rows.append(row)
    return rows


def write_history(output_dir: Path, history: dict[str, list[float]]) -> None:
    write_json(output_dir / "training_history.json", history)
    rows = history_to_rows(history)
    fieldnames = sorted({field for row in rows for field in row.keys()}, key=lambda item: (item != "epoch", item))
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_predictions(model, dataset) -> tuple[np.ndarray, np.ndarray]:
    true_batches: list[np.ndarray] = []
    pred_batches: list[np.ndarray] = []
    for features, labels in dataset:
        preds = model.predict(features, verbose=0)
        true_batches.append(labels.numpy())
        pred_batches.append(preds)
    if not true_batches:
        raise ValueError("No records available for prediction collection")
    return np.concatenate(true_batches, axis=0), np.concatenate(pred_batches, axis=0)


def collect_labels(dataset) -> np.ndarray:
    label_batches: list[np.ndarray] = []
    for _, labels in dataset:
        label_batches.append(labels.numpy())
    if not label_batches:
        raise ValueError("No records available for label collection")
    return np.concatenate(label_batches, axis=0)


def safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int | None]:
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

    return {
        "overall_mae": float(np.mean(abs_diff)),
        "overall_mse": float(np.mean(squared)),
        "overall_rmse": float(np.sqrt(np.mean(squared))),
        "overall_r2": safe_float(r2),
        "overall_pearson": safe_float(pearson),
        "overall_spearman": safe_float(spearman),
        "profile_mean_mae": float(np.mean(np.abs(np.mean(y_pred, axis=1) - np.mean(y_true, axis=1)))),
        "profile_max_mae": float(np.mean(np.abs(np.max(y_pred, axis=1) - np.max(y_true, axis=1)))),
        "shallow_profile_0_9_mae": float(np.mean(abs_diff[:, :10])),
        "mid_profile_10_34_mae": float(np.mean(abs_diff[:, 10:35])),
        "deep_profile_35_69_mae": float(np.mean(abs_diff[:, 35:])),
        "num_samples": int(y_true.shape[0]),
        "num_label_values": int(y_true.size),
    }


def baseline_predictions(y_train: np.ndarray, y_eval: np.ndarray) -> dict[str, np.ndarray]:
    train_mean = np.mean(y_train, axis=0)
    train_median = np.median(y_train, axis=0)
    return {
        "zero": np.zeros_like(y_eval),
        "train_mean": np.broadcast_to(train_mean, y_eval.shape).astype(np.float32),
        "train_median": np.broadcast_to(train_median, y_eval.shape).astype(np.float32),
    }


def write_baseline_comparison(output_dir: Path, model_metrics: dict[str, Any], baseline_metrics: dict[str, dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for name, metrics in {"model": model_metrics, **baseline_metrics}.items():
        rows.append(
            {
                "experiment_id": "EXP-007",
                "comparator": name,
                "dataset_split": "test",
                "mae": metrics.get("overall_mae"),
                "rmse": metrics.get("overall_rmse"),
                "r2": metrics.get("overall_r2"),
                "pearson": metrics.get("overall_pearson"),
                "spearman": metrics.get("overall_spearman"),
                "profile_mean_mae": metrics.get("profile_mean_mae"),
                "profile_max_mae": metrics.get("profile_max_mae"),
                "num_samples": metrics.get("num_samples"),
                "notes": "single-well depth-heldout EXP-007",
            }
        )
    fields = list(rows[0].keys())
    with (output_dir / "baseline_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / "baseline_comparison.json", {"rows": rows})

    lines = [
        "# EXP-007 Depth-Heldout Baseline Comparison",
        "",
        "Scope: single-well depth-heldout EXP-007 on `array_03`. Baselines use train labels only.",
        "",
        "| comparator | MAE | RMSE | R2 | Pearson | Spearman |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['comparator']} | {row['mae']} | {row['rmse']} | {row['r2']} | "
            f"{row['pearson']} | {row['spearman']} |"
        )
    lines.append("")
    (output_dir / "baseline_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def load_split_manifest_rows(path: Path, split: str) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") == split:
                rows.append(row)
    return rows


def write_prediction_summary(
    output_dir: Path,
    split_rows: list[dict[str, str]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    fields = [
        "eval_index",
        "record_index",
        "sample_index",
        "depth_ft",
        "true_profile_mean_percent",
        "pred_profile_mean_percent",
        "true_profile_max_percent",
        "pred_profile_max_percent",
        "sample_mae",
        "sample_rmse",
        "sample_bias_mean",
    ]
    with (output_dir / "prediction_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(y_true.shape[0]):
            row = split_rows[i] if i < len(split_rows) else {}
            diff = y_pred[i] - y_true[i]
            writer.writerow(
                {
                    "eval_index": i,
                    "record_index": row.get("record_index", ""),
                    "sample_index": row.get("sample_index", ""),
                    "depth_ft": row.get("depth_ft", ""),
                    "true_profile_mean_percent": float(np.mean(y_true[i])),
                    "pred_profile_mean_percent": float(np.mean(y_pred[i])),
                    "true_profile_max_percent": float(np.max(y_true[i])),
                    "pred_profile_max_percent": float(np.max(y_pred[i])),
                    "sample_mae": float(np.mean(np.abs(diff))),
                    "sample_rmse": float(np.sqrt(np.mean(diff * diff))),
                    "sample_bias_mean": float(np.mean(diff)),
                }
            )


def severity_group(value: float) -> str:
    if value <= 0:
        return "zero"
    if value <= 5:
        return "low_0_5"
    if value <= 20:
        return "medium_5_20"
    return "high_gt_20"


def write_severity_group_metrics(output_dir: Path, y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, Any]]:
    true_mean = np.mean(y_true, axis=1)
    rows: list[dict[str, Any]] = []
    for group in ("zero", "low_0_5", "medium_5_20", "high_gt_20"):
        indices = [i for i, value in enumerate(true_mean) if severity_group(float(value)) == group]
        if not indices:
            rows.append({"severity_group": group, "count": 0, "mae": None, "rmse": None, "bias_mean": None})
            continue
        yt = y_true[indices]
        yp = y_pred[indices]
        diff = yp - yt
        rows.append(
            {
                "severity_group": group,
                "count": len(indices),
                "mae": float(np.mean(np.abs(diff))),
                "rmse": float(np.sqrt(np.mean(diff * diff))),
                "bias_mean": float(np.mean(diff)),
            }
        )
    fields = ["severity_group", "count", "mae", "rmse", "bias_mean"]
    with (output_dir / "severity_group_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / "severity_group_metrics.json", {"rows": rows})
    return rows


def write_profile_error(output_dir: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    abs_error = np.abs(y_pred - y_true)
    rows = [
        {
            "profile_index": i,
            "mae": float(np.mean(abs_error[:, i])),
            "rmse": float(np.sqrt(np.mean((y_pred[:, i] - y_true[:, i]) ** 2))),
            "true_mean": float(np.mean(y_true[:, i])),
            "pred_mean": float(np.mean(y_pred[:, i])),
        }
        for i in range(y_true.shape[1])
    ]
    with (output_dir / "profile_index_error.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_training_curve(output_dir: Path, history: dict[str, list[float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.get("loss", []), label="train_loss")
    axes[0].plot(history.get("val_loss", []), label="val_loss")
    axes[0].set_title("EXP-007 loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(history.get("mae", []), label="train_mae")
    axes[1].plot(history.get("val_mae", []), label="val_mae")
    axes[1].set_title("EXP-007 MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curve.png", dpi=160)
    plt.close(fig)


def load_prediction_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_prediction_scatter(output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    rows = load_prediction_summary(output_dir / "prediction_summary.csv")
    true_vals = np.array([float(row["true_profile_mean_percent"]) for row in rows], dtype=float)
    pred_vals = np.array([float(row["pred_profile_mean_percent"]) for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_vals, pred_vals, s=12, alpha=0.65)
    max_val = float(max(np.max(true_vals), np.max(pred_vals), 1.0))
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1)
    ax.set_xlabel("True mean channeling percentage")
    ax.set_ylabel("Predicted mean channeling percentage")
    ax.set_title("EXP-007 Depth-Heldout Test")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_vs_truth_scatter.png", dpi=160)
    plt.close(fig)


def plot_residual_distribution(output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    rows = load_prediction_summary(output_dir / "prediction_summary.csv")
    residuals = np.array(
        [float(row["pred_profile_mean_percent"]) - float(row["true_profile_mean_percent"]) for row in rows],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=30, alpha=0.8)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Predicted - true mean channeling percentage")
    ax.set_ylabel("Sample count")
    ax.set_title("EXP-007 Depth-Heldout Residuals")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "residual_distribution.png", dpi=160)
    plt.close(fig)


def plot_depth_curve(output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    rows = [row for row in load_prediction_summary(output_dir / "prediction_summary.csv") if row.get("depth_ft")]
    if not rows:
        return
    depths = np.array([float(row["depth_ft"]) for row in rows], dtype=float)
    true_vals = np.array([float(row["true_profile_mean_percent"]) for row in rows], dtype=float)
    pred_vals = np.array([float(row["pred_profile_mean_percent"]) for row in rows], dtype=float)
    order = np.argsort(depths)
    fig, ax = plt.subplots(figsize=(7, 10))
    ax.plot(true_vals[order], depths[order], label="true", linewidth=1)
    ax.plot(pred_vals[order], depths[order], label="pred", linewidth=1, alpha=0.75)
    ax.invert_yaxis()
    ax.set_xlabel("Mean channeling percentage")
    ax.set_ylabel("Sonic depth ft")
    ax.set_title("EXP-007 Depth-Heldout Test Depth Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "depth_curve_if_available.png", dpi=160)
    plt.close(fig)


def plot_profile_error(output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    rows: list[dict[str, str]]
    with (output_dir / "profile_index_error.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    x = np.array([int(row["profile_index"]) for row in rows])
    mae = np.array([float(row["mae"]) for row in rows])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, mae, marker="o", linewidth=1)
    ax.set_xlabel("Profile depth index within 70-point label")
    ax.set_ylabel("MAE")
    ax.set_title("EXP-007 Per-Profile-Index MAE")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "per_profile_index_mae.png", dpi=160)
    plt.close(fig)


def copy_split_evidence(split_dir: Path, output_dir: Path) -> None:
    for name in ("split_manifest.json", "leakage_audit.json"):
        src = split_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def create_callbacks(output_dir: Path, patience: int):
    import tensorflow as tf  # type: ignore

    models_dir = output_dir / "models"
    logs_dir = output_dir / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.TensorBoard(log_dir=str(logs_dir / "tensorboard")),
        tf.keras.callbacks.CSVLogger(str(logs_dir / "keras_epoch_log.csv")),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(models_dir / "best_model.h5"),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            mode="min",
            verbose=1,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=max(2, patience // 2),
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def write_reports(
    output_dir: Path,
    run_config: dict[str, Any],
    history: dict[str, list[float]],
    val_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    baseline_metrics: dict[str, dict[str, Any]],
    severity_rows: list[dict[str, Any]],
) -> None:
    best_val_loss = min(history.get("val_loss", []) or [math.nan])
    final_val_loss = (history.get("val_loss", []) or [math.nan])[-1]
    final_train_loss = (history.get("loss", []) or [math.nan])[-1]
    report = [
        "# EXP-007 Depth-Heldout Training Run Report",
        "",
        f"Generated: `{utc_now()}`",
        f"Output dir: `{output_dir}`",
        f"Run mode: `{'smoke' if run_config.get('smoke') else 'full'}`",
        "",
        "## Split",
        "",
        f"- split_dir: `{run_config.get('split_dir')}`",
        "- split_type: `depth_heldout_exp007`",
        "- leakage claim: uses precomputed `train.tfrecord`, `val.tfrecord`, and `test.tfrecord`; no random split or validation_split.",
        "",
        "## Training Summary",
        "",
        f"- epochs_requested: `{run_config.get('epochs')}`",
        f"- epochs_completed: `{len(history.get('loss', []))}`",
        f"- batch_size: `{run_config.get('batch_size')}`",
        f"- learning_rate: `{run_config.get('learning_rate')}`",
        f"- best_val_loss: `{best_val_loss}`",
        f"- final_train_loss: `{final_train_loss}`",
        f"- final_val_loss: `{final_val_loss}`",
        f"- pretrained_status: `{run_config.get('pretrained_status')}`",
        "",
        "## Heldout Metrics",
        "",
        "| metric | validation | test |",
        "| --- | ---: | ---: |",
    ]
    for key in sorted(set(val_metrics) | set(test_metrics)):
        report.append(f"| {key} | {val_metrics.get(key)} | {test_metrics.get(key)} |")
    report.extend(
        [
            "",
            "## Test Baselines",
            "",
            "| comparator | MAE | RMSE | R2 | Pearson | Spearman |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            f"| model | {test_metrics.get('overall_mae')} | {test_metrics.get('overall_rmse')} | {test_metrics.get('overall_r2')} | {test_metrics.get('overall_pearson')} | {test_metrics.get('overall_spearman')} |",
        ]
    )
    for name, metrics in baseline_metrics.items():
        report.append(
            f"| {name} | {metrics.get('overall_mae')} | {metrics.get('overall_rmse')} | "
            f"{metrics.get('overall_r2')} | {metrics.get('overall_pearson')} | {metrics.get('overall_spearman')} |"
        )
    report.extend(
        [
            "",
            "## Severity Groups",
            "",
            "| group | count | MAE | RMSE | bias_mean |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in severity_rows:
        report.append(f"| {row['severity_group']} | {row['count']} | {row['mae']} | {row['rmse']} | {row['bias_mean']} |")
    report.extend(
        [
            "",
            "## Thesis Use",
            "",
            "These metrics are EXP-007 single-well depth-heldout evidence. They should not be described as multi-well generalization.",
        ]
    )
    (output_dir / "run_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    warnings = []
    if any(value is not None and (math.isnan(float(value)) or math.isinf(float(value))) for value in [final_train_loss, final_val_loss]):
        warnings.append("NaN or infinite final loss detected.")
    if history.get("val_loss") and len(history["val_loss"]) >= 2 and min(history["val_loss"]) >= history["val_loss"][0]:
        warnings.append("Validation loss did not improve below the first epoch.")
    high_rows = [row for row in severity_rows if row["severity_group"] == "high_gt_20" and row["count"]]
    if high_rows and high_rows[0].get("bias_mean") is not None and float(high_rows[0]["bias_mean"]) < 0:
        warnings.append("High-severity group is underpredicted on average.")
    if not warnings:
        warnings.append("No automatic NaN/flat-validation warning detected. Interpret overfitting from curves and metrics.")
    (output_dir / "error_summary.md").write_text(
        "# EXP-007 Error Summary\n\n" + "\n".join(f"- {warning}" for warning in warnings) + "\n",
        encoding="utf-8",
    )


def run_training(args: argparse.Namespace) -> int:
    import tensorflow as tf  # type: ignore

    ensure_safe_output_dir(args.output_dir)
    if args.output_dir.exists() and not args.overwrite:
        existing_paths = list(args.output_dir.iterdir())
        allowed_precreated = {"train.log", "command.log", "stdout.log", "stderr.log"}
        blocking_paths = [path for path in existing_paths if path.name not in allowed_precreated]
        if blocking_paths:
            blocking_names = ", ".join(path.name for path in blocking_paths[:8])
            raise FileExistsError(f"Output directory already contains run artifacts: {args.output_dir} ({blocking_names})")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_dir = args.split_dir
    train_tfrecord = split_dir / "train.tfrecord"
    val_tfrecord = split_dir / "val.tfrecord"
    test_tfrecord = split_dir / "test.tfrecord"
    manifest_csv = split_dir / "split_manifest.csv"
    split_manifest_json = split_dir / "split_manifest.json"
    leakage_audit_json = split_dir / "leakage_audit.json"
    for path in (train_tfrecord, val_tfrecord, test_tfrecord, manifest_csv):
        if not path.exists():
            raise FileNotFoundError(f"Required split artifact missing: {path}")

    set_reproducibility(args.seed)
    copy_split_evidence(split_dir, args.output_dir)

    max_train_batches = args.max_train_batches
    max_val_batches = args.max_val_batches
    max_test_batches = args.max_test_batches

    train_ds = create_dataset(
        train_tfrecord,
        args.batch_size,
        training=True,
        seed=args.seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
        max_batches=max_train_batches,
        augment=not args.no_augment,
    )
    val_ds = create_dataset(
        val_tfrecord,
        args.batch_size,
        training=False,
        seed=args.seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
        max_batches=max_val_batches,
        augment=False,
    )
    test_ds = create_dataset(
        test_tfrecord,
        args.batch_size,
        training=False,
        seed=args.seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
        max_batches=max_test_batches,
        augment=False,
    )
    train_labels_ds = create_dataset(
        train_tfrecord,
        args.batch_size,
        training=False,
        seed=args.seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
        max_batches=max_train_batches if args.smoke else None,
        augment=False,
    )

    model, pretrained_status = build_exp007_model(
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        pretrained=not args.no_pretrained,
        pretrained_required=args.pretrained_required,
        gradient_clipnorm=args.gradient_clipnorm,
    )
    callbacks = create_callbacks(args.output_dir, args.patience)

    run_config = {
        "generated_at": utc_now(),
        "command": " ".join(sys.argv),
        "git": get_git_info(),
        "exp007_git_evidence": EXP007_GIT_EVIDENCE,
        "split_dir": str(split_dir),
        "train_tfrecord": str(train_tfrecord),
        "val_tfrecord": str(val_tfrecord),
        "test_tfrecord": str(test_tfrecord),
        "split_manifest_json_exists": split_manifest_json.exists(),
        "leakage_audit_json_exists": leakage_audit_json.exists(),
        "output_dir": str(args.output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "dropout": args.dropout,
        "seed": args.seed,
        "smoke": args.smoke,
        "max_train_batches": max_train_batches,
        "max_val_batches": max_val_batches,
        "max_test_batches": max_test_batches,
        "input_shape": list(INPUT_SHAPE),
        "label_shape": list(LABEL_SHAPE),
        "loss": "tf.keras.losses.Huber",
        "metric": "mae",
        "optimizer": "Adam",
        "gradient_clipnorm": args.gradient_clipnorm,
        "pretrained_status": asdict(pretrained_status),
        "no_random_split": True,
        "no_validation_split": True,
    }
    write_json(args.output_dir / "run_config.json", run_config)

    history_obj = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=2,
    )
    history = {key: [float(v) for v in values] for key, values in history_obj.history.items()}
    write_history(args.output_dir, history)
    plot_training_curve(args.output_dir, history)

    val_eval = model.evaluate(val_ds, verbose=0, return_dict=True)
    test_eval = model.evaluate(test_ds, verbose=0, return_dict=True)
    y_val_true, y_val_pred = collect_predictions(model, val_ds)
    y_test_true, y_test_pred = collect_predictions(model, test_ds)
    y_train_labels = collect_labels(train_labels_ds)

    val_metrics = {**{f"keras_{k}": safe_float(v) for k, v in val_eval.items()}, **regression_metrics(y_val_true, y_val_pred)}
    test_metrics = {**{f"keras_{k}": safe_float(v) for k, v in test_eval.items()}, **regression_metrics(y_test_true, y_test_pred)}
    write_json(args.output_dir / "val_metrics.json", val_metrics)
    write_json(args.output_dir / "test_metrics.json", test_metrics)

    baseline_metrics = {
        name: regression_metrics(y_test_true, pred)
        for name, pred in baseline_predictions(y_train_labels, y_test_true).items()
    }
    write_baseline_comparison(args.output_dir, test_metrics, baseline_metrics)

    test_rows = load_split_manifest_rows(manifest_csv, "test")
    if max_test_batches is not None:
        test_rows = test_rows[: y_test_true.shape[0]]
    write_prediction_summary(args.output_dir, test_rows, y_test_true, y_test_pred)
    severity_rows = write_severity_group_metrics(args.output_dir, y_test_true, y_test_pred)
    write_profile_error(args.output_dir, y_test_true, y_test_pred)

    plot_prediction_scatter(args.output_dir)
    plot_residual_distribution(args.output_dir)
    plot_depth_curve(args.output_dir)
    plot_profile_error(args.output_dir)

    np.savez_compressed(args.output_dir / "predictions_test.npz", y_true=y_test_true, y_pred=y_test_pred)
    write_reports(args.output_dir, run_config, history, val_metrics, test_metrics, baseline_metrics, severity_rows)
    if args.save_final_model:
        model.save(args.output_dir / "models" / "final_model.h5")

    print(f"EXP-007 depth-heldout training complete: output_dir={args.output_dir}")
    print(f"test_metrics={json.dumps(test_metrics, sort_keys=True)}")
    print(f"baseline_metrics={json.dumps(baseline_metrics, sort_keys=True)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--shuffle-buffer-size", type=int, default=1024)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--pretrained-required", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--gradient-clipnorm", type=float, default=None)
    parser.add_argument("--save-final-model", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run_training(args)
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "generated_at": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "command": " ".join(sys.argv),
        }
        write_json(args.output_dir / "failure_report.json", failure)
        (args.output_dir / "failure_report.md").write_text(
            "# EXP-007 Depth-Heldout Training Failure\n\n"
            f"- error_type: `{failure['error_type']}`\n"
            f"- error: `{failure['error']}`\n\n"
            "```text\n"
            f"{failure['traceback']}\n"
            "```\n",
            encoding="utf-8",
        )
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
