# Thesis Overclaim Blacklist

Generated: 2026-07-08. Do not use these claims unless new evidence is added.

## Prohibited Performance Claims

- 禁止写“模型已经实现可靠泛化”。
- 禁止写“模型已满足工业部署要求”。
- 禁止写“EXP-008 全面优于所有 baseline”。
- 禁止写“EXP-007 是更稳的主线方法”。
- 禁止写“EXP-006 证明了 depth-heldout 泛化能力”。
- 禁止把 random split 指标和 depth-heldout 指标混在同一性能排名中。
- 禁止只报告 RMSE/R2 而隐瞒 zero baseline MAE 更低这一事实。

## Prohibited Scope Claims

- 禁止写“多井泛化已验证”。
- 禁止写“不同仪器/不同地层条件下稳定有效”。
- 禁止写“可以直接用于现场实时判断”。
- 禁止写“方位问题已经完全解决”。
- 禁止写“Relative Bearing 完全无用”。可以写旧证据未能支持可靠点对点方位监督。

## Prohibited Method Claims

- 禁止写“FFT 标签保留了完整方位信息”。本文使用 magnitude/log magnitude，phase 被丢弃。
- 禁止写“1D percentage 标签不损失信息”。它做了方位平均，丢弃方位结构。
- 禁止写“Grad-CAM 证明了物理机理”。它只能作为定性可解释性证据。
- 禁止写“高频误差低说明模型更懂高频结构”。高频目标幅值更小，绝对误差低可能来自幅值分布。

## Prohibited Evidence Handling

- 禁止从图片 OCR 或目测读取未抽取的精确数值。
- 禁止把 `val_mean_oracle_analysis` 写成可部署 baseline。
- 禁止把未复制/未读取的 EXP-007 severity-group CSV 写成最终结论。
- 禁止把 v003 写成训练失败；它是未运行，原因是执行环境 SSH/SCP 审批额度被拒绝。

## Safer Replacements

| unsafe | use instead |
| --- | --- |
| 模型泛化性能良好 | 单井 depth-heldout 结果支持一定可学习性 |
| 全面优于基线 | 在 RMSE/R2 上优于 zero baseline，但 MAE 未优于 zero baseline |
| 证明方位问题解决 | 通过 FFT magnitude 标签降低了对直接方位匹配的依赖 |
| 高严重度预测准确 | 高严重度样本仍存在低估，是主要限制 |
| Grad-CAM 证明物理原因 | Grad-CAM 定性显示模型关注某些时频区域 |
