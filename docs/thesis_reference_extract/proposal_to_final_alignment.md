# Proposal To Final Alignment

Generated: 2026-07-09.

## Alignment Summary

| proposal-stage plan | final thesis status | evidence | writing action |
| --- | --- | --- | --- |
| Build XSI-CAST supervised dataset | Partially aligned. Final work uses archived processed artifacts and explicit depth-heldout splits, but does not claim perfect pixel-level alignment. | `docs/thesis_evidence/depth_blocked_*`; `docs/thesis_plot_data/*` | Write as “构建弱监督样本对/标签”， not “精确像素级对齐”. |
| Use CWT features from sonic waveforms | Aligned. | `docs/thesis_plot_data/manual_figure_design_brief.md`; code inventory | Use in method chapter. |
| Solve azimuth mismatch via 1D percentage and FFT magnitude labels | Aligned, with final hierarchy changed. | EXP-008 final mainline; EXP-007 fallback | Present EXP-008 as mainline and EXP-007 as fallback. |
| EfficientNetV2B0 regression | Aligned. | `scripts/thesis_train_exp008_depth_blocked.py`; `scripts/thesis_train_exp007_depth_blocked.py` | Use code-derived architecture description. |
| Artifact masking | Partially aligned / needs exact run-specific caution. | final code/evidence docs | Mention only as method design where verified; avoid universal claim. |
| Grad-CAM physical interpretation | Partially aligned. Qualitative interpretability remains useful, but final evidence does not prove physical causality. | Grad-CAM result files; `thesis_safe_claims.md` | Write as qualitative interpretation and future work. |
| Improve FFT regression performance with hyperparameters/loss weights | Not pursued after evidence freeze. | final freeze docs | State as future work, not completed result. |
| Large-scale Grad-CAM statistics | Not completed as final quantitative evidence. | final missing evidence | Use existing Grad-CAM qualitatively. |
| Publish high-level software/tool or deployment capability | Not final thesis claim. | no deployment evidence | Remove or keep as future expectation only. |

## Final Experiment Roles

| experiment | proposal relation | final role |
| --- | --- | --- |
| EXP-008 severity + FFT magnitude + EfficientNet | Directly matches proposal’s FFT magnitude route. | Method innovation mainline; single-well depth-heldout feasibility. |
| EXP-007 1D percentage + EfficientNet | Matches proposal’s 1D profile route. | Fallback / limitation comparison. |
| EXP-006 CNN binary baseline | Proposal baseline verification. | Random-split exploratory baseline only. |
| SE-ResNet / dual-channel / eccentricity correction / GAN / sample weights | Exploration not retained as mainline. | Discussion or appendix failure routes. |

## Final Result Correction To Proposal

Proposal-stage materials imply strong performance and precision. Final evidence is more cautious:

- EXP-008 depth-heldout test: positive R2 and correlations, better than zero baseline by RMSE/R2, but worse than zero baseline by MAE.
- EXP-007 depth-heldout fallback: trainable, better than train-based baselines by MAE/RMSE/R2, but R2 is slightly negative and zero baseline MAE is lower.
- EXP-006 is random-split only and cannot be used for depth-heldout generalization.

## Recommended Thesis Narrative

1. The proposal’s motivation remains valid: azimuth mismatch blocks direct pointwise supervision.
2. The final thesis narrows the claim: use FFT magnitude weak labels and CWT-EfficientNet to demonstrate single-well depth-heldout learnability.
3. The final thesis is stronger because it explicitly audits split leakage and reports limitations rather than relying on optimistic random-split performance.
