# Figure Caption Reference

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
