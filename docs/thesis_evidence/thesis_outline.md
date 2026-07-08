# Thesis Outline Draft

## Title Suggestions

- 基于声波时频特征与CAST胶结成像标签的水泥窜槽识别方法研究
- 面向方位失配井段的XSI-CAST水泥窜槽弱监督表征与识别研究
- 基于CWT-EfficientNet与FFT方位不变标签的水泥窜槽严重度预测研究

## Abstract Core Logic

垂直井中 XSI 声波与 CAST 胶结图像存在方位失配，直接做点对点方位监督容易失败。旧实验显示，CWT 时频图能够稳定学习窜槽存在性；进一步将 CAST Zc 构造成深度方向百分比标签或方位 FFT 幅值标签，可以分别形成实用的深度严重度预测路线和方位不变的角度失配处理路线。论文主线建议以 CWT + EfficientNet + severity/FFT 或 1D percentage 标签为核心，用 baseline/log/weighted/GAN/dual-channel/pre-correction 作为对比和失败路线，并用 Grad-CAM 解释模型关注的高频时频区域。

## Chapter Plan

| chapter | experiments | figures | existing_evidence | missing_evidence | minimal_experiment |
| --- | --- | --- | --- | --- | --- |
| 第1章 绪论 | EXP-006; EXP-007; EXP-008 as motivation | 数据示意图; limitation evidence | memo PPT section; result tree inventory | 工程背景文字和引用文献需要另补 | none |
| 第2章 数据与问题定义 | all data construction code | 数据示意图; CWT 示例图; severity map | config.py; CWT scripts; result visualization plots | raw data provenance and split/depth-blocked validation | split audit only |
| 第3章 标签构造与方位失配处理 | EXP-001; EXP-002; EXP-003; EXP-007; EXP-008 | FFT label map; severity map; depth prediction curve | create_tfrecords code across branches; memo | FFT route final metric verification | existing-artifact metric extraction |
| 第4章 模型方法 | EXP-006; EXP-007; EXP-008; EXP-010 | model architecture | model.py across branches; model_architecture images | SE-ResNet code mapping | server branch inventory |
| 第5章 实验结果与消融 | EXP-001 to EXP-008; EXP-011; EXP-012 | scatter plot; FFT prediction map; baseline comparison; ablation; failed attempt | result plots/logs/memo | unified baseline/ablation numeric table | metric extraction from saved artifacts only |
| 第6章 可解释性分析 | EXP-013; EXP-006; EXP-007 | Grad-CAM heatmap; CWT example | Grad-CAM plots/statistics and memo sensitive-region claims | batch-level Grad-CAM statistics if not already sufficient | aggregate existing Grad-CAM artifacts |
| 第7章 讨论与结论 | all failed routes and mainline limitations | limitation evidence | memo failure records; missing_evidence.md; risk_register.md | 导师确认最终主线取舍 | none |

## Suggested Figure Categories

| figure_id | category | source_path | thesis_chapter | quality | needs_redraw |
| --- | --- | --- | --- | --- | --- |
| FIG-001 | CWT 示例图 | /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+CNN/sample_scalograms.png | 第2章 数据构建 | high | no |
| FIG-002 | CWT 示例图 | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/GaN+2Dlabel/output/visualization_plots/150_02_cwt_result.png | 第2章 数据构建 | medium | no |
| FIG-003 | CWT 示例图 | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/GaN+2Dlabel/output/visualization_plots/2500_02_cwt_result.png | 第2章 数据构建 | medium | no |
| FIG-004 | CWT 示例图 | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/baseline/results/debug_plots/debug_cwt_array.png | 第2章 数据构建 | low | no |
| FIG-005 | CWT 示例图 | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/baseline/visualization_plots/02_cwt_result.png | 第2章 数据构建 | medium | no |
| FIG-006 | CWT 示例图 | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/log_label/output/visualization_plots/02_cwt_result.png | 第2章 数据构建 | medium | no |
| FIG-007 | CWT 示例图 | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/test_relativity/output/visualization_plots/150_02_cwt_result.png | 第2章 数据构建 | medium | no |
| FIG-008 | CWT 示例图 | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/test_relativity/output/visualization_plots/2500_02_cwt_result.png | 第2章 数据构建 | medium | no |
| FIG-009 | CWT 示例图 | /mnt/c/Users/Administrator/Desktop/Hal/results/双通道学习/array_03/plots/validate_pipeline_02_CWT_vs_Label.png | 第2章 数据构建 | high | no |
| FIG-010 | FFT label map | /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet/output/visualization_plots/06_profile_label_generation.png | 第3章 标签构造 | medium | maybe |
| FIG-011 | FFT label map | /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet/output/visualization_plots/fft_regression/06_label_generation_fft_regression.png | 第3章 标签构造 | medium | maybe |
| FIG-012 | FFT label map | /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet_1/06_label_generation_fft_regression.png | 第3章 标签构造 | medium | maybe |
| FIG-013 | FFT label map | /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet_1/output/visualization_plots/06_profile_label_generation.png | 第3章 标签构造 | medium | maybe |
| FIG-014 | FFT label map | /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet_1/output/visualization_plots/fft_regression/06_label_generation_fft_regression.png | 第3章 标签构造 | medium | maybe |
| FIG-015 | FFT prediction map | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/baseline/results/analysis_plots_with_gradcam/sample_0_truth_vs_prediction.png | 第5章 实验结果 | medium | no |
| FIG-016 | FFT prediction map | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/baseline/results/analysis_plots_with_gradcam/sample_1_truth_vs_prediction.png | 第5章 实验结果 | medium | no |
| FIG-017 | FFT prediction map | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/baseline/results/analysis_plots_with_gradcam/sample_2_truth_vs_prediction.png | 第5章 实验结果 | medium | no |
| FIG-018 | FFT prediction map | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/baseline/results/analysis_plots_with_gradcam/sample_3_truth_vs_prediction.png | 第5章 实验结果 | medium | no |
| FIG-019 | FFT prediction map | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/baseline/results/analysis_plots_with_gradcam/sample_4_truth_vs_prediction.png | 第5章 实验结果 | medium | no |
| FIG-020 | FFT prediction map | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/baseline/results/analysis_plots_with_gradcam/sample_5_truth_vs_prediction.png | 第5章 实验结果 | medium | no |
| FIG-021 | FFT prediction map | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/baseline/results/analysis_plots_with_gradcam/sample_6_truth_vs_prediction.png | 第5章 实验结果 | medium | no |
| FIG-022 | FFT prediction map | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/baseline/results/analysis_plots_with_gradcam/sample_7_truth_vs_prediction.png | 第5章 实验结果 | medium | no |
| FIG-023 | FFT prediction map | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/baseline/results/analysis_plots_with_gradcam/sample_8_truth_vs_prediction.png | 第5章 实验结果 | medium | no |
| FIG-024 | FFT prediction map | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/baseline/results/analysis_plots_with_gradcam/sample_9_truth_vs_prediction.png | 第5章 实验结果 | medium | no |
| FIG-025 | FFT prediction map | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/frequency-weighted_loss/output/image_translation/array_03/results/analysis_plots_with_gradcam/sample_0_truth_vs_prediction.png | 第5章 实验结果 | medium | no |
| FIG-026 | FFT prediction map | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/frequency-weighted_loss/output/image_translation/array_03/results/analysis_plots_with_gradcam/sample_1_truth_vs_prediction.png | 第5章 实验结果 | medium | no |
| FIG-027 | Grad-CAM heatmap | /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+CNN/comprehensive_gradcam_statistics.png | 第6章 可解释性分析 | high | no |
| FIG-028 | Grad-CAM heatmap | /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+CNN/gradcam_analysis_optimized.png | 第6章 可解释性分析 | high | no |
| FIG-029 | Grad-CAM heatmap | /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+SE-ResNet/outputs/array_03/plots/attention_mean_variance.png | 第6章 可解释性分析 | high | no |
| FIG-030 | Grad-CAM heatmap | /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+SE-ResNet/outputs/array_03/plots/class_conditional_attention_analysis.png | 第6章 可解释性分析 | high | no |
| FIG-031 | Grad-CAM heatmap | /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+SE-ResNet/outputs/array_07/plots/attention_mean_variance.png | 第6章 可解释性分析 | high | no |
| FIG-032 | Grad-CAM heatmap | /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+SE-ResNet/outputs/array_07/plots/class_conditional_attention_analysis.png | 第6章 可解释性分析 | high | no |
| FIG-033 | Grad-CAM heatmap | /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+SE-ResNet/outputs/array_11/plots/attention_mean_variance.png | 第6章 可解释性分析 | high | no |
| FIG-034 | Grad-CAM heatmap | /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+SE-ResNet/outputs/array_11/plots/class_conditional_attention_analysis.png | 第6章 可解释性分析 | high | no |
| FIG-035 | Grad-CAM heatmap | /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet/output/image_translation/array_03/results/final_analysis_plots/_overall_mean_attention_map.png | 第6章 可解释性分析 | medium | no |
| FIG-036 | Grad-CAM heatmap | /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet/output/image_translation/array_03/results/final_analysis_plots/gradcam_sample_312.png | 第6章 可解释性分析 | medium | no |
| FIG-037 | Grad-CAM heatmap | /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet/output/image_translation/array_03/results/final_analysis_plots/gradcam_sample_313.png | 第6章 可解释性分析 | medium | no |
| FIG-038 | Grad-CAM heatmap | /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet/output/image_translation/array_03/results/final_analysis_plots/gradcam_sample_314.png | 第6章 可解释性分析 | medium | no |
| FIG-039 | ablation | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/log_label/output/visualization_plots/01_filtering_effect.png | 第5章 消融实验 | medium | maybe |
| FIG-040 | ablation | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/log_label/output/visualization_plots/03_label_generation.png | 第5章 消融实验 | medium | maybe |
<!-- FIGURE_SHORTLIST_START -->
## Figure Shortlist Arrangement Update (2026-07-06)

- 本轮从 110 个 figure candidates 中筛选 34 张论文候选图。
- EXP-008/EXP-007/EXP-006 相关结果图在图注中统一标注 `exploratory_only` 或 `random_split_depth_leakage_risk`，不用于最终泛化性能声明。
- 候选集中未找到 EfficientNet 主线架构图，需根据代码另画。

### Chapter Figure Plan

| chapter | shortlisted_figures | must_use | note |
| --- | --- | --- | --- |
| 第2章 数据与问题定义 | FIG-106, FIG-104, FIG-105, FIG-074, FIG-075 | FIG-106, FIG-104, FIG-074 | 方法/数据图为主 |
| 第2章 数据构建 | FIG-001, FIG-009, FIG-007 | FIG-001 | 方法/数据图为主 |
| 第3章 标签构造 | FIG-049, FIG-011, FIG-092 | FIG-049, FIG-011 | 方法/数据图为主 |
| 第4章 模型方法 | FIG-084 | none | 方法/数据图为主 |
| 第5章 实验结果 | FIG-046, FIG-085, FIG-048, FIG-089, FIG-057 | FIG-085, FIG-089, FIG-057 | 结果图必须标注 exploratory/random split 风险 |
| 第5章 对比实验 | FIG-045 | none | 结果图必须标注 exploratory/random split 风险 |
| 第5章 消融实验 | FIG-041, FIG-055 | none | 结果图必须标注 exploratory/random split 风险 |
| 第6章 可解释性分析 | FIG-027, FIG-028, FIG-035, FIG-036, FIG-029, FIG-030 | FIG-027, FIG-035 | 方法/数据图为主 |
| 第7章 讨论 | FIG-060, FIG-061, FIG-062, FIG-072, FIG-090, FIG-059, FIG-076, FIG-052 | none | 失败路线和 split 风险讨论 |

### Table Plan

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
<!-- FIGURE_SHORTLIST_END -->

<!-- EXP007_DEPTH_HELDOUT_FALLBACK_START -->
## EXP-007 Depth-Heldout Fallback Update (2026-07-07)

### Updated Thesis Result Logic

The recommended result narrative after EXP-007 fallback training is:

1. Use EXP-008 as the method innovation mainline: CWT + EfficientNetV2B0 + FFT severity magnitude label for azimuth-invariant weak supervision.
2. Use EXP-007 as a simpler fallback/limitation comparison: CWT + EfficientNetV2B0 + 1D percentage label confirms that a profile label is trainable under a single-well depth-heldout split, but it is not stronger than EXP-008.
3. Do not present either EXP-008 or EXP-007 as multi-well generalization. Both are `array_03` single-well depth-heldout evidence.
4. Do not run EXP-006 for the shortest thesis path unless the thesis committee explicitly requires a binary-detection fallback.

### Updated Chapter Placement

| chapter | added_exp007_depthheldout_use | evidence | limitation |
| --- | --- | --- | --- |
| 第3章 标签构造 | 1D percentage label as fallback label route | `docs/thesis_evidence/exp007_artifact_and_code_inspection.md`; old `create_tfrecords.py` evidence from `origin/1D+percentage_Label` | It removes azimuth matching but produces sparse labels. |
| 第5章 实验结果 | Report EXP-007 `train_v002` after EXP-008, as fallback comparison | `docs/thesis_evidence/exp007_depthheldout_training_report.md`; `exp007_depthheldout_baseline_comparison.csv`; `unified_metrics_table.csv` | Test MAE `0.811672`, RMSE `2.832056`, R2 `-0.011278`; better than zero by RMSE/R2 but not by MAE. |
| 第5章 实验结果 | Compare EXP-007 and EXP-008 depth-heldout results | `docs/thesis_evidence/exp007_vs_exp008_depthheldout_comparison.md` | Metrics are not directly scale-comparable because labels differ; comparison is about thesis role and evidence strength. |
| 第7章 讨论 | Sparse-label baseline limitation and early overfitting | `docs/thesis_evidence/exp007_depthheldout_final_claim.md`; `risk_register.md` | Severity-group details remain `needs_verification` until remote `severity_group_metrics.csv/json` can be copied/read. |

### Updated Figure/Table Placement

| item | chapter | source | use |
| --- | --- | --- | --- |
| EXP-007 split audit table | 第5章 / 第7章 | `docs/thesis_evidence/remote_exp007_split_v001/leakage_audit.json` | Show `depth_heldout_split_confirmed` and continuous train/val/test ranges. |
| EXP-007 baseline comparison table | 第5章 | `docs/thesis_evidence/exp007_depthheldout_baseline_comparison.csv` | Show model vs zero/train-mean/train-median; highlight MAE caveat. |
| EXP-007 vs EXP-008 comparison table | 第5章 / 第7章 | `docs/thesis_evidence/exp007_vs_exp008_depthheldout_comparison.csv` | Justify retaining EXP-008 as mainline and EXP-007 as fallback/limitation comparison. |
| EXP-007 training/prediction figures | 第5章 | Remote `/home/xiaoj/hal_azi/output/thesis_depth_blocked/exp007/train_v002/*.png` | Use after small figures are copied locally; captions must say single-well depth-heldout. |

### Stop/Continue Recommendation

Current evidence is sufficient to move into thesis writing if the result chapter is framed as single-well method feasibility plus limitations. The next shortest useful task is not another training run, but copying the already generated EXP-007 small figure/CSV artifacts from remote once SSH/SCP execution is available again, then redrawing final thesis figures.
<!-- EXP007_DEPTH_HELDOUT_FALLBACK_END -->
