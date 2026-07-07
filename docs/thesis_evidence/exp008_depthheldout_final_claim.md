# EXP-008 Depth-Heldout Final Claim After Result Audit P3

Generated: 2026-07-07. Scope: single-well depth-heldout EXP-008 on `array_03`. This document does not claim multi-well generalization or industrial deployment performance.

## Direct Answers

| question | answer | evidence |
| --- | --- | --- |
| EXP-008 是否优于 zero baseline | Mixed. It is **not** better by overall MAE: model `0.079025` vs zero `0.069751`. It **is** better by RMSE and R2: model RMSE `0.277019`, R2 `0.106634` vs zero RMSE `0.301272`, R2 `-0.056638`. | `exp008_depthheldout_baseline_comparison.csv` |
| EXP-008 是否优于 train-mean baseline | Yes by MAE/RMSE/R2: model MAE `0.079025`, RMSE `0.277019`, R2 `0.106634` vs train-mean MAE `0.142358`, RMSE `0.316129`, R2 `-0.163426`. | `exp008_depthheldout_baseline_comparison.csv` |
| 哪些 FFT 频率/标签维度最可学习 | By absolute MAE, high-frequency coefficients k=25-29 have the lowest error, but this likely reflects smaller/sparser target magnitude. Low-frequency coefficients k=0-4 are the most difficult and most important error source. | `exp008_depthheldout_error_structure.json` |
| 主要误差来自哪里 | DC/low-frequency FFT coefficients, depth-within-window indices 12-14 and 18-19, and high-severity samples where the model often underpredicts integrated severity. | `exp008_depthheldout_error_structure.md` |
| 是否仍建议不跑 EXP-007 | No longer a hard “do not run”. EXP-008 succeeded, but failure to beat zero baseline on MAE is a thesis limitation. If time allows or if the thesis needs a stronger absolute-error result, run EXP-007 depth-heldout as fallback/comparison in a later stage. | baseline audit P3 |
| 论文结果章节如何表述 | State that EXP-008 under single-well depth-heldout split improves over train-mean/train-median baselines and over zero baseline in RMSE/R2, with positive test correlation, but MAE is worse than the zero predictor due sparse labels. | this audit |
| 哪些结论必须加 limitation | Any performance claim, any generalization claim, and any “model is better than baseline” statement must specify metric-dependent comparison and single-well depth-heldout scope. | `risk_register.md` update |

## Safe Thesis Claim

A safe wording is:

> Under a deterministic single-well depth-heldout split on `array_03`, the EXP-008 CWT + EfficientNetV2B0 model for FFT severity labels achieved test RMSE `0.277019`, R2 `0.106634`, Pearson `0.351950`, and Spearman `0.480519`. It outperformed train-mean and train-median baselines on MAE/RMSE/R2 and outperformed the zero baseline on RMSE/R2, but did not outperform the zero baseline on MAE because the target labels are highly sparse. Therefore, the result supports metric-qualified method feasibility rather than strong absolute-error or multi-well generalization claims.

## Required Limitations

- Single-well only: `array_03` continuous depth-heldout.
- Metric-dependent baseline result: MAE favors the zero baseline.
- Label sparsity: 75% of label-value absolute errors are zero, so aggregate MAE can reward all-zero predictions.
- High-severity underprediction remains visible in top-error samples.
- EXP-007 depth-heldout remains a reasonable fallback/comparison if final thesis evidence needs stronger baseline dominance.
