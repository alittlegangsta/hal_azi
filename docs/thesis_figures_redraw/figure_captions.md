# Thesis Figure Captions

Generated: 2026-07-08. Figures are derived only from existing `docs/thesis_evidence` files and archived small artifacts. No raw data, processed data, Windows results, TFRecord, checkpoint, or large NPZ file was copied or modified.

| figure | status | 中文图题 | 图注草稿 |
| --- | --- | --- | --- |
| fig_01_xsi_cast_azimuth_mismatch_schematic.png | generated | XSI 与 CAST 方位失配问题示意图 | 示意 XSI 声波接收器阵列与 CAST Zc 方位图之间存在未知方位偏移，直接点对点方位监督不可靠，因此本文采用 1D percentage 与 FFT magnitude 弱监督标签路线。该图为方法示意，不含性能结论。 |
| fig_02_data_pipeline.png | generated | XSI-CWT 与 CAST 弱标签数据构建流程 | 展示从 XSI 波形到 CWT 输入、从 CAST Zc 到 severity/percentage/FFT 标签、再到显式 depth-heldout TFRecord 与 EfficientNet 回归模型的整体流程。 |
| fig_03_percentage_label_construction.png | generated | 一维窜槽百分比标签构造流程 | 根据 EXP-007 代码证据，将 CAST Zc 以 2.5 为阈值生成窜槽掩膜，再沿方位求平均得到深度方向 percentage profile。该图为标签构造示意，不使用图中数值作为实验结果。 |
| fig_04_fft_severity_label_construction.png | generated | FFT severity magnitude 标签构造流程 | 根据 EXP-008 方法路线，将 CAST Zc 转换为 severity=max(0,2.5-Zc)，再沿方位维计算 FFT magnitude，丢弃相位以降低对方位匹配的依赖。 |
| fig_05_cwt_efficientnet_regression_architecture.png | generated | CWT-EfficientNet 回归模型结构示意 | 展示 150x400x8 CWT 输入经过 1x1 通道适配、EfficientNetV2B0 backbone、全局池化、dropout 与 Dense 回归头输出 EXP-008 70x30 或 EXP-007 70 维标签。 |
| fig_06_exp008_training_curve.png | generated | EXP-008 depth-heldout 训练曲线 | 基于 `remote_exp008_depth_blocked_train/training_history.csv` 重画训练/验证 loss 与 MAE，显示 train_v001 在单井 depth-heldout 设置下早期过拟合并由 EarlyStopping 恢复较早 epoch 权重。 |
| fig_07_exp008_prediction_scatter.png | generated | EXP-008 depth-heldout 测试预测散点图 | 基于 `prediction_summary.csv` 展示测试集 integrated severity 的预测-真值关系。结果只代表 `array_03` 单井 depth-heldout，不代表多井泛化。 |
| fig_08_exp008_baseline_comparison.png | generated | EXP-008 与简单基线对比 | 基于 `exp008_depthheldout_baseline_comparison.csv` 展示模型、zero、train-mean、train-median 的 MAE/RMSE/R2。图注必须说明模型未优于 zero baseline 的 MAE。 |
| fig_09_exp008_per_fft_coefficient_error.png | generated | EXP-008 按 FFT 系数的误差分布 | 基于 `exp008_depthheldout_error_structure.csv` 展示不同 FFT 系数的 MAE，低频系数绝对误差更高，是主要 limitation。 |
| fig_10_exp007_prediction_scatter.png | missing_data_no_plot | EXP-007 depth-heldout 测试预测散点图 | 本地图件包未生成该图，因为缺少 EXP-007 train_v002 逐样本 prediction summary 或预测数组。见 `fig_10_exp007_prediction_scatter_missing.md`。 |
| fig_11_exp007_baseline_comparison.png | generated | EXP-007 fallback 与简单基线对比 | 基于 `exp007_depthheldout_baseline_comparison.csv` 展示 EXP-007 train_v002 与 zero/train-mean/train-median 的 MAE/RMSE/R2。EXP-007 优于 train-based baselines，但未优于 zero baseline 的 MAE。 |
| fig_12_limitation_high_severity_underestimation.png | generated | EXP-008 高严重度样本低估现象 | 基于 EXP-008 测试集 prediction summary 中高 integrated severity 深度段绘制真值与预测趋势，展示高严重度峰值被模型低估，是本文主要限制之一。 |
