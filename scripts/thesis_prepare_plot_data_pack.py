#!/usr/bin/env python3
"""Prepare data tables and manual design briefs for thesis figures.

This script does not generate final-style figures. It copies/filters small
archived evidence files into docs/thesis_plot_data for manual drawing in
Origin/PPT/AI/Figma.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "thesis_evidence"
OUT = ROOT / "docs" / "thesis_plot_data"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_csv(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def num(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def prepare_exp008_training() -> None:
    rows = read_csv(EVIDENCE / "remote_exp008_depth_blocked_train" / "training_history.csv")
    out = []
    for r in rows:
        out.append(
            {
                "epoch": r["epoch"],
                "train_loss": r["loss"],
                "val_loss": r["val_loss"],
                "train_mae": r["mae"],
                "val_mae": r["val_mae"],
                "learning_rate": r["lr"],
                "source_path": "docs/thesis_evidence/remote_exp008_depth_blocked_train/training_history.csv",
                "notes": "EXP-008 train_v001 single-well depth-heldout; early stopping restored best epoch 2",
            }
        )
    write_csv(OUT / "exp008_training_curve.csv", out)


def prepare_exp008_scatter() -> None:
    src = EVIDENCE / "remote_exp008_depth_blocked_train" / "prediction_summary.csv"
    if not src.exists():
        (OUT / "exp008_prediction_scatter_missing.md").write_text(
            "# EXP-008 prediction scatter data missing\n\nNo archived per-sample prediction summary was found.\n",
            encoding="utf-8",
        )
        return
    rows = read_csv(src)
    out = []
    for r in rows:
        out.append(
            {
                "eval_index": r["eval_index"],
                "record_index": r["record_index"],
                "depth_ft": r["depth_ft"],
                "true_mean_integrated_severity": r["true_mean_integrated_severity"],
                "pred_mean_integrated_severity": r["pred_mean_integrated_severity"],
                "sample_mae": r["sample_mae"],
                "sample_rmse": r["sample_rmse"],
                "sample_dc_mae": r["sample_dc_mae"],
                "source_path": "docs/thesis_evidence/remote_exp008_depth_blocked_train/prediction_summary.csv",
                "notes": "single-well depth-heldout test sample; not multi-well generalization",
            }
        )
    write_csv(OUT / "exp008_prediction_scatter.csv", out)


def prepare_baselines() -> None:
    copy_csv(EVIDENCE / "exp008_depthheldout_baseline_comparison.csv", OUT / "exp008_baseline_comparison.csv")
    copy_csv(EVIDENCE / "exp007_depthheldout_baseline_comparison.csv", OUT / "exp007_baseline_comparison.csv")


def prepare_fft_error() -> None:
    rows = read_csv(EVIDENCE / "exp008_depthheldout_error_structure.csv")
    keep = []
    for r in rows:
        if r.get("analysis_type") == "per_fft_coefficient":
            keep.append(
                {
                    "fft_coefficient": r["index"],
                    "label": r["label"],
                    "model_mae": r["model_mae"],
                    "zero_mae": r["zero_mae"],
                    "train_mean_mae": r["train_mean_mae"],
                    "train_median_mae": r["train_median_mae"],
                    "val_mean_oracle_analysis_mae": r["val_mean_oracle_analysis_mae"],
                    "target_mean_abs": r["target_mean_abs"],
                    "source_path": "docs/thesis_evidence/exp008_depthheldout_error_structure.csv",
                    "notes": "per FFT coefficient MAE; single-well depth-heldout EXP-008",
                }
            )
    write_csv(OUT / "exp008_per_fft_coefficient_error.csv", keep)


def prepare_high_severity() -> None:
    src = EVIDENCE / "remote_exp008_depth_blocked_train" / "prediction_summary.csv"
    rows = read_csv(src)
    parsed = []
    for r in rows:
        true = num(r["true_mean_integrated_severity"])
        pred = num(r["pred_mean_integrated_severity"])
        depth = num(r["depth_ft"])
        if true is not None and pred is not None and depth is not None:
            parsed.append((r, true, pred, depth))
    high = [(r, true, pred, depth) for r, true, pred, depth in parsed if true >= 5.0]
    out = []
    if high:
        d0 = min(depth for _, _, _, depth in high)
        d1 = max(depth for _, _, _, depth in high)
        for r, true, pred, depth in parsed:
            if d0 - 3 <= depth <= d1 + 3:
                out.append(
                    {
                        "eval_index": r["eval_index"],
                        "record_index": r["record_index"],
                        "depth_ft": r["depth_ft"],
                        "true_mean_integrated_severity": r["true_mean_integrated_severity"],
                        "pred_mean_integrated_severity": r["pred_mean_integrated_severity"],
                        "underestimation": true - pred,
                        "high_severity_flag": "yes" if true >= 5.0 else "context",
                        "sample_mae": r["sample_mae"],
                        "sample_rmse": r["sample_rmse"],
                        "source_path": "docs/thesis_evidence/remote_exp008_depth_blocked_train/prediction_summary.csv",
                        "notes": "EXP-008 high-severity context window; single-well depth-heldout",
                    }
                )
    write_csv(OUT / "high_severity_underestimation.csv", out)

    summary = {
        "source_path": "docs/thesis_evidence/remote_exp008_depth_blocked_train/prediction_summary.csv",
        "threshold_for_high_severity_context": 5.0,
        "num_high_severity_rows": len(high),
        "num_context_rows_exported": len(out),
        "depth_min_ft": min((depth for _, _, _, depth in high), default=None),
        "depth_max_ft": max((depth for _, _, _, depth in high), default=None),
        "limitation": "This is a plotting subset for EXP-008 single-well depth-heldout only; not multi-well generalization.",
    }
    (OUT / "high_severity_underestimation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_manual_briefs() -> None:
    brief = """# Manual Figure Design Brief

Generated: 2026-07-08. This directory is for manual redrawing in Origin/PPT/AI/Figma. It intentionally does not contain final thesis-style generated figures.

## Global Rules

- Distinguish `random split exploratory` from `single-well depth-heldout`.
- Do not claim multi-well generalization or industrial deployment.
- Do not use values not present in the CSV files in this directory or the cited evidence paths.
- For method schematics, use them as conceptual diagrams only; do not add performance numbers.

## 1. XSI/CAST 方位失配示意图

图题：XSI 与 CAST 方位失配问题示意图

必须包含：
- 左侧 XSI 声波接收器/波形或 CWT 输入。
- 右侧 CAST Zc 方位图。
- 深度方向可对齐，但方位零点/旋转角未知。
- 标注“直接点对点方位监督不可靠”。
- 引出两条弱标签路线：1D percentage 和 FFT magnitude。

推荐布局：
- 左右对照布局：XSI 在左，CAST 在右，中间画深度对齐箭头和方位旋转偏移箭头。
- 底部放“weak label strategy”小结。

推荐颜色逻辑：
- XSI/CWT 用蓝色。
- CAST/Zc 用红色或橙色。
- 不确定方位偏移用灰色虚线或红色旋转箭头。

禁止表达：
- 不要写 Relative Bearing 永远无用。
- 不要写方位问题已经完全解决。
- 不要放任何模型性能指标。

## 2. 数据处理流程图

图题：XSI-CWT 与 CAST 弱标签数据构建流程

必须包含：
- XSI waveform -> CWT `150 x 400 x 8`。
- CAST Zc -> severity `max(0, 2.5-Zc)` -> FFT magnitude label。
- CAST Zc -> binary mask `Zc < 2.5` -> azimuth mean -> 1D percentage label。
- Explicit train/val/test TFRecord split。
- EfficientNetV2B0 regression。

推荐布局：
- 上支路为 XSI 特征，下支路为 CAST 标签，中间在 TFRecord split 处汇合。
- EXP-008 主线用实线或强调色，EXP-007 fallback 用次级色。

禁止表达：
- 不要写 validation_split。
- 不要暗示 EXP-006 是最终性能主线。

## 3. Percentage Label 构造图

图题：一维窜槽百分比标签构造流程

必须包含：
- CAST Zc depth-window slice。
- 阈值 `Zc < 2.5` 的二值掩膜。
- 沿方位维求平均，得到每个深度位置的窜槽百分比。
- 输出 70 点 profile。

推荐布局：
- 四步横向流程：Zc slice -> mask -> azimuth mean -> 1D profile。

禁止表达：
- 不要写该标签保留完整方位信息；它丢弃了方位结构。
- 不要把 EXP-007 写成比 EXP-008 更强。

## 4. FFT Severity Label 构造图

图题：FFT severity magnitude 标签构造流程

必须包含：
- CAST Zc。
- severity transform: `severity = max(0, 2.5 - Zc)`。
- 沿方位维做 FFT。
- 取 magnitude/log magnitude，phase discarded。
- 输出 70 x 30 FFT label map。

推荐布局：
- 横向流程，最后用频谱条形或热图表示低频到高频系数。
- 明确标注“phase discarded for rotation invariance”。

禁止表达：
- 不要写 FFT magnitude 保留完整方位相位信息。
- 不要写旋转不变标签已完全解决所有方位问题。

## 5. CWT-EfficientNet 架构图

图题：CWT-EfficientNet 回归模型结构示意

必须包含：
- CWT input `150 x 400 x 8`。
- 1x1 Conv channel adapter `8 -> 3`。
- EfficientNetV2B0 backbone。
- Global Average Pooling。
- Dropout。
- Dense regression head。
- 两种输出：EXP-008 `70 x 30` FFT label；EXP-007 `70` percentage profile。
- 训练输入为 explicit train/val/test TFRecord，不使用 validation_split。

推荐布局：
- 主干横向网络结构，输出处分叉到 EXP-008 和 EXP-007。

禁止表达：
- 不要加入未核验参数量、FLOPs 或部署速度。
- 不要把架构图作为性能证明。
"""
    (OUT / "manual_figure_design_brief.md").write_text(brief, encoding="utf-8")

    captions = """# Figure Caption Reference

Generated: 2026-07-08.

| figure_id | suggested_title_cn | caption_cn | data_source | caution |
| --- | --- | --- | --- | --- |
| FIG-M01 | XSI 与 CAST 方位失配问题示意图 | 示意 XSI 声波接收器阵列与 CAST Zc 方位图之间存在未知方位偏移，直接点对点方位监督不可靠，因此本文采用一维百分比标签与 FFT magnitude 弱监督标签路线。 | `figure_redraw_plan.md`; `thesis_safe_claims.md` | 方法示意，不含性能。 |
| FIG-M02 | XSI-CWT 与 CAST 弱标签数据构建流程 | 展示从 XSI 波形到 CWT 输入、从 CAST Zc 到 severity/percentage/FFT 标签、再到显式 depth-heldout TFRecord 与 EfficientNet 回归模型的流程。 | `manual_figure_design_brief.md`; code inventory | 不暗示多井泛化。 |
| FIG-M03 | 一维窜槽百分比标签构造流程 | 将 CAST Zc 以 2.5 为阈值生成窜槽掩膜，并沿方位方向求平均得到深度方向的一维窜槽百分比标签。 | `exp007_artifact_and_code_inspection.md` | 该标签丢弃方位结构。 |
| FIG-M04 | FFT severity magnitude 标签构造流程 | 将 CAST Zc 转换为 severity=max(0,2.5-Zc)，再沿方位维计算 FFT magnitude，丢弃 phase 以降低对直接方位匹配的依赖。 | `exp008_training_code_inspection.md` | 不说保留完整方位信息。 |
| FIG-M05 | CWT-EfficientNet 回归模型结构示意 | CWT 输入经过 1x1 通道适配、EfficientNetV2B0、全局池化、Dropout 与 Dense 回归头，分别输出 EXP-008 FFT 标签或 EXP-007 percentage profile。 | `scripts/thesis_train_exp008_depth_blocked.py`; `scripts/thesis_train_exp007_depth_blocked.py` | 架构图不是性能证明。 |
| FIG-R01 | EXP-008 depth-heldout 训练曲线 | 基于单井 depth-heldout train_v001 的训练/验证 loss 与 MAE，显示早期过拟合。 | `exp008_training_curve.csv` | 单井 `array_03`。 |
| FIG-R02 | EXP-008 depth-heldout 测试预测散点图 | 展示 EXP-008 测试样本 integrated severity 的预测-真值关系。 | `exp008_prediction_scatter.csv` | 不代表多井泛化。 |
| FIG-R03 | EXP-008 与简单基线对比 | 对比模型、zero、train-mean、train-median 的 MAE/RMSE/R2。模型优于 zero 的 RMSE/R2，但 MAE 不优于 zero。 | `exp008_baseline_comparison.csv` | 必须写 MAE caveat。 |
| FIG-R04 | EXP-008 按 FFT 系数的误差分布 | 展示不同 FFT 系数的 MAE，低频系数绝对误差较高。 | `exp008_per_fft_coefficient_error.csv` | 高频低误差可能受目标幅值影响。 |
| FIG-R05 | EXP-007 fallback 与简单基线对比 | 对比 EXP-007 train_v002 与 zero/train-mean/train-median。EXP-007 优于 train-based baselines，但 MAE 不优于 zero。 | `exp007_baseline_comparison.csv` | fallback/limitation comparison。 |
| FIG-R06 | EXP-008 高严重度样本低估 | 展示 EXP-008 测试集中高 integrated severity 深度段的真值和预测，突出高严重度峰值低估。 | `high_severity_underestimation.csv` | limitation evidence only。 |
"""
    (OUT / "figure_caption_reference.md").write_text(captions, encoding="utf-8")

    design_json = {
        "generated": "2026-07-08",
        "scope": "manual design reference only; no final-style figures",
        "figures": [
            {
                "figure_id": "FIG-M01",
                "title_cn": "XSI 与 CAST 方位失配问题示意图",
                "required_elements": [
                    "XSI sonic receiver / waveform / CWT input",
                    "CAST Zc azimuth map",
                    "depth alignment arrow",
                    "unknown azimuth offset / rotation arrow",
                    "weak label strategy: 1D percentage and FFT magnitude",
                ],
                "layout": "left-right comparison, XSI left and CAST right, mismatch arrows in the middle",
                "color_logic": "blue for XSI/CWT, red/orange for CAST/Zc, gray dashed or red arrow for azimuth uncertainty",
                "do_not_say": [
                    "Relative Bearing is always useless",
                    "azimuth mismatch is fully solved",
                    "any model performance value",
                ],
            },
            {
                "figure_id": "FIG-M02",
                "title_cn": "XSI-CWT 与 CAST 弱标签数据构建流程",
                "required_elements": [
                    "XSI waveform -> CWT 150 x 400 x 8",
                    "CAST Zc -> severity max(0, 2.5-Zc) -> FFT magnitude label",
                    "CAST Zc -> binary mask Zc < 2.5 -> azimuth mean -> 1D percentage label",
                    "explicit train/val/test TFRecord split",
                    "EfficientNetV2B0 regression",
                ],
                "layout": "two-branch pipeline; XSI features on top, CAST labels below, merge at TFRecord/model",
                "color_logic": "EXP-008 main route emphasized; EXP-007 fallback in secondary color",
                "do_not_say": ["validation_split", "EXP-006 is final performance mainline"],
            },
            {
                "figure_id": "FIG-M03",
                "title_cn": "一维窜槽百分比标签构造流程",
                "required_elements": ["CAST Zc slice", "Zc < 2.5 mask", "azimuth average", "70-point percentage profile"],
                "layout": "four-step horizontal process",
                "color_logic": "Zc heatmap red/orange, mask red/gray, profile line blue or teal",
                "do_not_say": ["label preserves full azimuth information", "EXP-007 is stronger than EXP-008"],
            },
            {
                "figure_id": "FIG-M04",
                "title_cn": "FFT severity magnitude 标签构造流程",
                "required_elements": [
                    "CAST Zc",
                    "severity = max(0, 2.5 - Zc)",
                    "azimuth FFT",
                    "magnitude/log magnitude",
                    "phase discarded",
                    "70 x 30 target map",
                ],
                "layout": "horizontal process ending with spectrum bars or target heatmap",
                "color_logic": "severity heatmap red/orange, FFT magnitude teal/blue, phase discarded shown in gray",
                "do_not_say": ["FFT magnitude preserves phase", "rotation-invariant label fully solves all azimuth issues"],
            },
            {
                "figure_id": "FIG-M05",
                "title_cn": "CWT-EfficientNet 回归模型结构示意",
                "required_elements": [
                    "CWT input 150 x 400 x 8",
                    "1x1 Conv 8 -> 3",
                    "EfficientNetV2B0 backbone",
                    "Global Average Pooling",
                    "Dropout",
                    "Dense regression head",
                    "EXP-008 output 70 x 30",
                    "EXP-007 output 70",
                    "explicit train/val/test TFRecords",
                ],
                "layout": "horizontal model backbone with output branch for EXP-008 and EXP-007",
                "color_logic": "input teal, backbone purple/blue, outputs teal for EXP-008 and amber for EXP-007",
                "do_not_say": ["unverified parameter counts", "FLOPs", "deployment speed", "architecture proves performance"],
            },
        ],
    }
    (OUT / "manual_figure_design_elements.json").write_text(
        json.dumps(design_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_inventory() -> None:
    files = [
        ("exp008_training_curve.csv", "EXP-008 training/validation loss and MAE by epoch", "remote_exp008_depth_blocked_train/training_history.csv"),
        ("exp008_prediction_scatter.csv", "EXP-008 per-sample true/pred integrated severity scatter data", "remote_exp008_depth_blocked_train/prediction_summary.csv"),
        ("exp008_baseline_comparison.csv", "EXP-008 model and simple baselines", "exp008_depthheldout_baseline_comparison.csv"),
        ("exp008_per_fft_coefficient_error.csv", "EXP-008 per FFT coefficient MAE", "exp008_depthheldout_error_structure.csv"),
        ("exp007_baseline_comparison.csv", "EXP-007 train_v001/v002 baseline comparison", "exp007_depthheldout_baseline_comparison.csv"),
        ("high_severity_underestimation.csv", "EXP-008 high severity context rows from prediction summary", "remote_exp008_depth_blocked_train/prediction_summary.csv"),
        ("high_severity_underestimation_summary.json", "Summary of high severity export", "remote_exp008_depth_blocked_train/prediction_summary.csv"),
        ("manual_figure_design_brief.md", "Manual drawing elements/layout/colors/cautions", "final thesis evidence docs"),
        ("manual_figure_design_elements.json", "Structured manual drawing element brief", "final thesis evidence docs"),
        ("figure_caption_reference.md", "Chinese title/caption reference", "final thesis evidence docs"),
        ("figure_data_inventory.md", "Inventory of this data pack", "generated by this script"),
    ]
    lines = [
        "# Figure Data Inventory",
        "",
        "Generated: 2026-07-08. This pack contains data and manual design notes only. It does not contain final thesis-style generated figures.",
        "",
        "| file | purpose | source evidence | status |",
        "| --- | --- | --- | --- |",
    ]
    for filename, purpose, source in files:
        path = OUT / filename
        status = "available" if path.exists() or filename == "figure_data_inventory.md" else "missing"
        source_text = f"`docs/thesis_evidence/{source}`" if source.startswith(("remote_", "exp", "final_")) else source
        lines.append(f"| `{filename}` | {purpose} | {source_text} | {status} |")
    lines.extend(
        [
            "",
            "## Missing/Not Exported",
            "",
            "- EXP-007 prediction scatter data was not exported because local archived evidence does not include per-sample EXP-007 prediction summary or prediction arrays.",
            "- No checkpoint, SavedModel, TFRecord, large NPZ, raw data, processed data, or Windows results file is included.",
            "- Existing PNG redraws under `docs/thesis_figures_redraw/` are not copied here because this pack is for manual design data only.",
            "",
            "## Scope Labels",
            "",
            "- EXP-008 and EXP-007 depth-heldout data: `single-well depth-heldout array_03`.",
            "- EXP-006 data, if cited from final metrics: `random split exploratory baseline`.",
        ]
    )
    (OUT / "figure_data_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prepare_exp008_training()
    prepare_exp008_scatter()
    prepare_baselines()
    prepare_fft_error()
    prepare_high_severity()
    write_manual_briefs()
    write_inventory()
    print(f"wrote thesis plot data pack to {OUT}")


if __name__ == "__main__":
    main()
