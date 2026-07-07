# EXP-008 Depth-Heldout Redrawn Figure Captions

All figures are single-well depth-heldout EXP-008 evidence from `array_03`. Do not describe them as multi-well generalization or deployment performance.

| file | caption_draft_cn |
| --- | --- |
| training_curve_redraw.png | EXP-008 在显式 depth-heldout 划分下的训练与验证损失/MAE 曲线。模型在第 12 轮早停并恢复第 2 轮最佳验证损失权重。 |
| prediction_vs_truth_scatter_redraw.png | EXP-008 depth-heldout 测试集平均 integrated severity 的预测-真值散点图，用于说明单井连续深度留出段上的可学习关系。 |
| baseline_comparison_bar.png | EXP-008 模型与 zero、train-mean、train-median 简单基线在 depth-heldout 测试集上的 MAE/RMSE 对比。 |
| per_fft_coefficient_mae.png | EXP-008 depth-heldout 测试集按 FFT 系数统计的 MAE，用于分析低频、中频和高频标签维度的误差结构。 |
| error_distribution.png | EXP-008 depth-heldout 测试样本级 MAE 分布，用于展示误差长尾和异常样本风险。 |
| target_vs_prediction_lowfreq_summary.png | EXP-008 depth-heldout 测试集中低频 FFT 系数均值的预测-真值关系，用于补充说明低频标签维度表现。 |
