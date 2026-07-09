# Outdated Or Conflicting Claims

Generated: 2026-07-09. Do not use these reference-material claims without rewriting.

## Performance Overclaims

| source | risky wording / meaning | why outdated or conflicting | safe replacement |
| --- | --- | --- | --- |
| `final_project_report.pptx` | “model performance is good” for EfficientNet/FFT route | Final EXP-008 depth-heldout result is metric-qualified: RMSE/R2 improve over zero, but MAE is worse than zero. | “EXP-008 在单井 depth-heldout 下显示一定可学习性，但存在 zero-baseline MAE 限制。” |
| `proposal_report_form.docx` | “MAE误差控制在5%以内，能够准确预测窜槽百分比” | Final EXP-007 train_v002 MAE is `0.811672` in percentage-label units and not better than zero baseline MAE. | “EXP-007 可训练，但作为 fallback/limitation comparison。” |
| `proposal_defense.pptx` | “模型预测性能验证” without split caveat | Old figures may be random-split/exploratory. | Always distinguish random split exploratory vs single-well depth-heldout. |

## Scope Overclaims

| source | risky wording / meaning | issue | safe replacement |
| --- | --- | --- | --- |
| `proposal_report_form.docx` | multi-well data coverage and broad deployment implications | Final performance evidence is only `array_03` single-well depth-heldout. | “本研究在单井连续深度留出条件下验证方法可行性。” |
| `proposal_report_form.docx` | “低成本、全井段、超声级洞察力” | No final deployment or all-well validation evidence. | Use as motivation only, not result. |
| `proposal_report_form.docx` | “精准反演” | Final metrics do not support precision claim. | “弱监督严重度/结构标签预测的可学习性探索。” |

## Method Overclaims

| source | risky wording / meaning | issue | safe replacement |
| --- | --- | --- | --- |
| proposal materials | “精确物理对应关系” / “像素级对齐” | Final mainline avoids direct pointwise azimuth supervision because azimuth is unreliable. | “构造声波-Cast 弱监督样本对/深度窗口对应关系。” |
| proposal materials | “FFT 解决方位失配问题” | FFT magnitude reduces phase dependence but does not solve every source of mismatch. | “FFT magnitude 降低对绝对方位匹配的依赖。” |
| proposal materials | “幅度谱保留结构信息” without caveat | Magnitude discards phase and loses specific azimuth position. | “保留部分环向结构频率信息，同时丢弃相位。” |
| proposal materials | “Grad-CAM 验证物理规律/物理因果” | Grad-CAM is qualitative interpretability, not causal proof. | “Grad-CAM 定性显示模型关注某些时频区域。” |

## Timeline / Plan Items No Longer Current

- “继续优化 FFT 系数截断长度、loss 权重” is future work; no further training is planned.
- “大规模 Grad-CAM 批量统计” remains optional/future work.
- “发表高水平论文 1 篇” is an expected outcome, not a thesis result.
- “软件代码库/现场工具” should not be described as deployed.

## Required Rule

When using reference materials in thesis prose, check each sentence against:

- `docs/thesis_evidence/thesis_safe_claims.md`
- `docs/thesis_evidence/thesis_overclaim_blacklist.md`
- `docs/thesis_evidence/final_thesis_metrics_table.md`
