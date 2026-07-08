# EXP-007 Depth-Heldout Result Interpretation

Generated: 2026-07-07.

## Direct Answers

| question | answer | evidence |
| --- | --- | --- |
| EXP-007 是否比 zero baseline 更稳 | Mixed. Best run `train_v002` beats zero by RMSE/R2 (`2.832056` vs `2.911398`; `-0.011278` vs `-0.068735`) but not by MAE (`0.811672` vs `0.738339`). | `exp007_depthheldout_baseline_comparison.csv` |
| EXP-007 是否比 train-mean/train-median baseline 更稳 | Yes by MAE/RMSE/R2 for best run `train_v002`. | `exp007_depthheldout_baseline_comparison.csv` |
| EXP-007 是否比 EXP-008 更适合作为实证 fallback | No. EXP-007 improves over train-based baselines but still fails zero-baseline MAE and has negative R2. EXP-008 remains stronger by positive R2 and clearer method novelty. | `exp007_vs_exp008_depthheldout_comparison.csv` |
| 是否需要再训练 EXP-006 | Not recommended for this thesis path unless a binary detection safety net is required. EXP-007 did not solve the zero-MAE issue, so additional regression tuning is unlikely to be the shortest path. | explicitly marked inference from EXP-007/EXP-008 depth-heldout results |
| 是否可以停止实验进入论文写作 | Yes, with cautious framing. EXP-008 is main method feasibility; EXP-007 is fallback/negative comparison; both have single-well and sparse-label limitations. | this report |

## Error Structure

Captured profile metrics for best run `train_v002`:

| metric | value |
| --- | ---: |
| shallow_profile_0_9_mae | 2.790083 |
| mid_profile_10_34_mae | 1.151847 |
| deep_profile_35_69_mae | 0.003429 |
| profile_mean_mae | 0.744563 |
| profile_max_mae | 5.947414 |

Interpretation:

- Most error is concentrated in the shallow/mid part of the 70-point label profile.
- Deep profile indices are nearly zero in the test labels, so low deep-profile MAE mostly reflects label sparsity.
- Severity group metrics were generated remotely, but were not copied/read locally after SSH/SCP escalation was denied. Detailed severity group conclusions are therefore `needs_verification`.

## Safe Thesis Wording

Use:

> In the single-well depth-heldout EXP-007 fallback experiment, the 1D percentage-label EfficientNet model improved over train-mean and train-median baselines in MAE/RMSE/R2 and improved over the zero baseline in RMSE/R2, but did not beat the zero baseline in MAE. This indicates that sparse percentage labels remain difficult under contiguous depth holdout and should be presented as a fallback/limitation comparison rather than the main performance claim.

Avoid:

- claiming robust generalization
- claiming EXP-007 is clearly superior to all simple baselines
- using EXP-007 as stronger evidence than EXP-008
