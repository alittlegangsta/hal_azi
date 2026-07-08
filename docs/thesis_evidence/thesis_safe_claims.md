# Thesis Safe Claims

Generated: 2026-07-08. Use these formulations in the thesis. Each claim is scoped to available evidence and avoids overclaiming.

## Mainline

- 本文最终主线是 CWT + EfficientNetV2B0 + FFT severity magnitude label，用于缓解 XSI 与 CAST 方位失配下的弱监督标签构造问题。
- 在 `array_03` 单井连续深度留出设置下，EXP-008 显示 FFT severity 标签具有一定可学习性。
- EXP-008 在测试集上取得 RMSE `0.277019`、R2 `0.106634`、Pearson `0.351950`、Spearman `0.480519`。
- EXP-008 优于 train-mean 和 train-median baselines 的 MAE/RMSE/R2，也优于 zero baseline 的 RMSE/R2。
- EXP-008 没有优于 zero baseline 的 MAE，因此结果必须表述为 metric-qualified learnability，而不是全面优于基线。

## Fallback

- EXP-007 的 1D percentage label 是一个直观的 fallback 标签路线。
- 在 `array_03` 单井 depth-heldout 设置下，EXP-007 train_v002 可完成训练并取得正 Spearman 相关。
- EXP-007 优于 train-mean 和 train-median baselines 的 MAE/RMSE/R2，也优于 zero baseline 的 RMSE/R2。
- EXP-007 没有优于 zero baseline 的 MAE，且 R2 略为负，因此不应替代 EXP-008 作为主线。

## Baseline

- EXP-006 的随机划分二分类结果说明 CWT 时频图中存在与窜槽存在性相关的可学习信号。
- EXP-006 可以作为 baseline/background，不应写成 depth-heldout 性能证据。

## Data And Split

- 旧随机划分实验存在 adjacent-depth leakage risk，因此只作为 exploratory evidence。
- EXP-008 和 EXP-007 的补充实验使用 explicit train/val/test TFRecord 和 continuous depth-heldout split。
- 当前 depth-heldout 证据只覆盖单井 `array_03`，不能外推到多井或工业部署。

## Error And Limitation

- EXP-008 的主要误差来自低频 FFT 系数和高严重度样本低估。
- EXP-007 的误差集中在 shallow/mid profile index 区间，deep 区间低误差可能来自标签稀疏。
- Grad-CAM 图可作为模型关注区域的定性解释，不能作为因果证明。

## Recommended Final Thesis Claim

在单井连续深度留出实验中，基于 CWT 的 EfficientNet 模型能够从 XSI 声波时频特征中学习到与 CAST 窜槽弱标签相关的结构。其中，FFT severity magnitude 标签更贴合方位失配问题，适合作为本文方法主线；1D percentage label 可作为 fallback 对照。现有结果支持方法可行性和标签路线比较，但不构成多井泛化或工程部署性能证明。
