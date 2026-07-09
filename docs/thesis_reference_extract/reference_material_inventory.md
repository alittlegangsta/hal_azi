# Reference Material Inventory

Generated: 2026-07-09. Scope: writing-reference ingestion only. These files are not final performance evidence. If a reference statement conflicts with `docs/thesis_evidence/` or `docs/thesis_plot_data/`, the final evidence wins.

## Files Scanned

| file | type | pages/slides/sections | main themes | reusable content | outdated or risky content | relation to final thesis mainline |
| --- | --- | ---: | --- | --- | --- | --- |
| `docs/thesis_reference_raw/final_project_report.pptx` | PPTX | 11 slides | Project retrospective; azimuth mismatch; eccentricity condition; hybrid input/pre-correction failures; non-eccentric route; Grad-CAM attention observations. | Useful for framing the project evolution from SE-ResNet/eccentricity correction to EfficientNet/FFT severity labels; useful failure-route wording for dual-input and eccentricity correction. | Mentions “model performance is good” without final depth-heldout caveats; predates the final EXP-008/EXP-007 evidence freeze. Treat performance wording as outdated. | Supports why EXP-008 became final method mainline and why dual-channel/pre-correction belong in discussion/appendix. |
| `docs/thesis_reference_raw/proposal_defense.pptx` | PPTX | 16 slides | Proposal defense; research background; domestic/international context; CWT feature extraction; FFT magnitude label; EfficientNetV2-B0; Grad-CAM; research plan. | Useful for thesis introduction, research significance, problem definition, CWT/FFT/EfficientNet/Grad-CAM method wording, and chapter-level technical route. | Uses planned/expected wording such as “验证物理规律”, “预测性能验证”, “高精度/精准反演”; must be softened to final single-well, metric-qualified evidence. | Closely aligned with final method route, but results must be updated to EXP-008 single-well depth-heldout and EXP-007 fallback. |
| `docs/thesis_reference_raw/proposal_report_form.docx` | DOCX | 161 useful paragraphs after paragraph merge; major sections include research content, basis/significance, research plan/methods, feasibility, schedule, expected outcomes. | Richest reusable text for background, significance, problem definition, research objectives, literature narrative, method descriptions, and feasibility. | Contains over-strong claims: “精确物理对应关系”, “像素级对齐”, “精准反演”, “MAE误差控制在5%以内”, “物理因果性验证”, “超声级洞察力”, multi-well/data-coverage claims. These require rewriting or deletion. | Strongly supports thesis framing, but final evidence narrows claims to single-well depth-heldout feasibility and method comparison. |

## Evidence Priority

| priority | source | use |
| --- | --- | --- |
| 1 | `docs/thesis_evidence/final_thesis_metrics_table.*`; `docs/thesis_evidence/exp008_*`; `docs/thesis_evidence/exp007_*` | Final metrics, split scope, final experiment roles, limitations. |
| 2 | `docs/thesis_plot_data/*` | Data for manual figure drawing and figure captions. |
| 3 | `docs/thesis_reference_raw/*` | Background wording, motivation, planned method language, literature framing. |

## High-Level Reusable Themes

- Cement channeling is a safety-critical cement-bonding problem.
- Sonic logging is lower-cost and broader-coverage than CAST/ultrasonic imaging, but its wavefield is complex and lower-resolution.
- CAST provides high-resolution supervisory information, but direct pointwise azimuth matching with XSI is unreliable in vertical wells.
- CWT converts non-stationary sonic waveforms into time-frequency representations suitable for CNN/EfficientNet models.
- FFT magnitude labels provide an azimuth-rotation-invariant weak supervision route by discarding phase.
- Grad-CAM can be used for qualitative interpretability of CWT-sensitive regions.

## Main Conflicts To Carry Forward

- Proposal-stage “precise/pixel-level alignment” conflicts with final decision to avoid direct azimuth supervision.
- Proposal-stage performance optimism conflicts with final depth-heldout results, where EXP-008/EXP-007 do not beat zero baseline by MAE.
- Proposal-stage multi-well/data coverage language is not supported by final performance evidence, which is single-well `array_03`.
- Proposal-stage physical-causality claims from Grad-CAM must be softened to qualitative interpretability.
