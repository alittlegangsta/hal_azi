# Table Shortlist For Thesis

Generated: 2026-07-06.

| table_id | table_name_cn | thesis_chapter | source | must_use | notes |
| --- | --- | --- | --- | --- | --- |
| TAB-S01 | 实验路线与证据强度总表 | 第5章 实验设计与结果 | docs/thesis_evidence/experiment_inventory.csv | yes | 按 EXP-001–EXP-014 汇总方法、标签、结果用途和证据强度。 |
| TAB-S02 | 统一指标表（仅探索性/随机划分） | 第5章 实验结果 | docs/thesis_evidence/unified_metrics_table.csv | yes | 只报告可追踪指标，明确 random_split_depth_leakage_risk；不能写最终泛化。 |
| TAB-S03 | split forensic 与泄漏风险表 | 第7章 讨论与限制 | docs/thesis_evidence/split_forensic_audit.csv; leakage_risk_report.md | yes | 解释 EXP-008/007/006 为什么需要 depth-blocked split。 |
| TAB-S04 | 标签构造路线对比表 | 第3章 标签构造 | docs/thesis_evidence/method_taxonomy.md; code_method_inventory.csv | yes | 比较 1D percentage、FFT magnitude、log transform、frequency weighting。 |
| TAB-S05 | 图件短名单与重画计划表 | 写作管理/附录 | docs/thesis_evidence/figure_shortlist_for_thesis.csv; figure_redraw_plan.md | yes | 保证论文图件均有 source_path 和风险标签。 |
| TAB-S06 | 失败路线汇总表 | 第7章 讨论 | experiment_inventory.csv; memo_experiment_claims.csv; metric_source_traceability.csv | yes | GAN、双通道、预校正、样本权重等只作 failed_attempt/appendix。 |
| TAB-S07 | Git 分支与实验映射表 | 附录：复现与证据链 | docs/thesis_evidence/branch_experiment_mapping.csv; git_branch_timeline.md | useful | 用于证明旧项目版本来源，正文可简化。 |
| TAB-S08 | 缺失证据与最小补充计划表 | 第7章 讨论与后续工作 | docs/thesis_evidence/missing_evidence.md; minimal_supplement_plan.md | yes | 明确必须补 depth-blocked split/TensorBoard dependency/图件重画。 |
