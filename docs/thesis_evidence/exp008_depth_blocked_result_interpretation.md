# EXP-008 Depth-Heldout Result Interpretation

Generated: 2026-07-07.

## Direct Answers

| question | answer | evidence |
| --- | --- | --- |
| EXP-008 depth-heldout 是否成功训练 | yes | `docs/thesis_evidence/remote_exp008_depth_blocked_train/run_report.md`; train exit code 0 in `train.log` |
| depth-heldout val/test 指标是多少 | validation MAE `0.042817`, RMSE `0.187167`, R2 `0.010882`; test MAE `0.079025`, RMSE `0.277019`, R2 `0.106634` | `val_metrics.json`; `test_metrics.json` |
| 与旧 random split 指标相比是否下降 | validation MAE is comparable to the old random-split best val_mae 0.043332, but heldout test MAE `0.079025` is higher; this is expected under stricter depth-heldout evaluation. | old row in `unified_metrics_table.csv`; new `test_metrics.json` |
| 是否仍可支撑“可学习性” | yes, cautiously: test Pearson `0.351950` and Spearman `0.480519` are positive, and R2 `0.106634` is above zero. | `test_metrics.json` |
| 是否可支撑“泛化性能” | only as single-well depth-heldout performance, not multi-well generalization. | split evidence covers one `array_03` continuous depth interval only |
| 是否需要跑 EXP-007 fallback | not immediately required; EXP-008 v001 succeeded. EXP-007 can be reserved if thesis needs a fallback label route or if advisors require a second depth-heldout comparison. | this report; EXP-007 remains random-split only |
| 论文中如何安全表述 | “在 array_03 的连续深度留出验证中，FFT severity + EfficientNet 仍表现出可学习关系；测试集 MAE 为 `0.079025`，Spearman 为 `0.480519`。该结果证明方法可行性和单井 depth-heldout 能力，但不代表多井泛化。” | `test_metrics.json`; `leakage_audit.json` |

## Interpretation

The strict heldout test result is weaker than historical random-split validation metrics. This should be treated as a strength of the new audit, not as a failure: it exposes the earlier adjacent-depth leakage risk and gives a more defensible thesis result. The positive rank correlation on the heldout tail interval supports learnability of FFT severity coefficients from CWT input, while the modest R2 indicates that final claims should avoid strong predictive-accuracy language.

## Thesis Placement

Use this result in the main experiment chapter as the primary EXP-008 performance table. Keep older EXP-008 random-split figures as exploratory/ablation background only. Use the new training curve, scatter plot, and depth curve from `docs/thesis_evidence/remote_exp008_depth_blocked_train/` as the preferred thesis figures for EXP-008.
