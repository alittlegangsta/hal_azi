# Reference To Chapter Mapping

Generated: 2026-07-09.

## Chapter Mapping

| thesis chapter | reusable reference material | how to use | caveat |
| --- | --- | --- | --- |
| 第1章 绪论 | `proposal_report_form.docx` paragraphs on cementing/channeling risk; `proposal_defense.pptx` slides 3-5 | Use for background, engineering significance, and research motivation. | Remove “精准/全井段/部署” overclaims. |
| 第1章 国内外研究现状 | `proposal_report_form.docx` literature paragraphs; `proposal_defense.pptx` slide 4 | Use as literature-outline skeleton: borehole acoustics, deep learning, CWT, FFT, Grad-CAM. | Verify bibliographic details separately before final submission. |
| 第2章 数据与问题定义 | `proposal_defense.pptx` slides 3, 6-8; `proposal_report_form.docx` problem-definition paragraphs | Explain XSI/CAST complementarity, CWT feature construction, and azimuth mismatch. | Do not claim perfect alignment. |
| 第3章 标签构造 | `proposal_defense.pptx` slide 9; `proposal_report_form.docx` FFT/percentage label paragraphs | Use for 1D percentage and FFT severity magnitude label descriptions. | State phase discarded and information loss; EXP-008 mainline, EXP-007 fallback. |
| 第4章 模型方法 | `proposal_defense.pptx` slides 8-11; `proposal_report_form.docx` EfficientNet/CWT/Huber/Grad-CAM method paragraphs | Use for method language and architecture explanation. | Do not add unverified parameter/deployment claims. |
| 第5章 实验结果 | Reference materials only for historical context; final evidence from `docs/thesis_evidence` | Use `final_thesis_metrics_table.md`, not proposal performance wording. | PPT/DOCX performance claims are outdated unless matched by final evidence. |
| 第6章 可解释性分析 | `proposal_defense.pptx` slide 11; `proposal_report_form.docx` Grad-CAM paragraphs; `final_project_report.pptx` slides 9-10 | Use for qualitative Grad-CAM motivation and sensitive time-frequency region discussion. | Do not write physical causality proof. |
| 第7章 讨论与限制 | `final_project_report.pptx` slides 2, 5-6; `outdated_or_conflicting_claims.md` | Use to explain failure-route evolution and why EXP-008 was retained. | Keep failed routes in discussion/appendix, not main performance claims. |

## Figure/Diagram Mapping

| manual figure | reference source | final drawing data |
| --- | --- | --- |
| XSI/CAST 方位失配示意图 | proposal problem definition; final project review slides | `docs/thesis_plot_data/manual_figure_design_brief.md` |
| 数据处理流程图 | proposal technical route slides | `docs/thesis_plot_data/manual_figure_design_elements.json` |
| 1D percentage label 构造图 | proposal/report label-route text | `docs/thesis_plot_data/manual_figure_design_brief.md`; final evidence EXP-007 |
| FFT severity label 构造图 | proposal slide 9 and report label engineering section | `docs/thesis_plot_data/manual_figure_design_brief.md`; final evidence EXP-008 |
| CWT-EfficientNet 架构图 | proposal slide 10 and code-derived final figure pack | `docs/thesis_plot_data/manual_figure_design_brief.md`; final code scripts |

## Recommended Citation Workflow

1. Use reference materials for narrative and motivation.
2. Use final evidence files for all experiment roles and metrics.
3. Check every strong claim against `outdated_or_conflicting_claims.md`.
4. If a sentence uses “验证/证明/精准/全井段/工程部署”, rewrite it unless final evidence directly supports it.
