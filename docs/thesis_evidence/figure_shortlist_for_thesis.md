# Figure Shortlist For Thesis

Generated: 2026-07-06. Shortlisted `34` figures from `figure_candidates.csv` (110 candidates). Results directory remains read-only; no OCR and no image modification were performed.

## Selection Rules

- Mainline EXP-008/EXP-007/EXP-006 result figures are labeled `exploratory_only` because split audit confirmed `random_split_depth_leakage_risk`.
- Figures with image-only metrics are not used to invent numbers; captions point to qualitative or illustrative use only.
- `needs_redraw=yes` means redraw/crop/recompose for thesis style; source_path remains the evidence source.

## Counts

### evidence_role

| evidence_role | count |
| --- | --- |
| appendix | 1 |
| baseline_result | 2 |
| exploratory_result | 5 |
| failed_attempt | 6 |
| interpretability | 6 |
| limitation | 5 |
| method_illustration | 9 |

### use_priority

| use_priority | count |
| --- | --- |
| appendix | 10 |
| must_use | 11 |
| useful | 13 |

### needs_redraw

| needs_redraw | count |
| --- | --- |
| no | 13 |
| yes | 21 |

### chapter

| chapter | count |
| --- | --- |
| 第2章 数据与问题定义 | 5 |
| 第2章 数据构建 | 3 |
| 第3章 标签构造 | 3 |
| 第4章 模型方法 | 1 |
| 第5章 实验结果 | 5 |
| 第5章 对比实验 | 1 |
| 第5章 消融实验 | 2 |
| 第6章 可解释性分析 | 6 |
| 第7章 讨论 | 8 |

## Shortlist

| figure_id | source_path | thesis_chapter | thesis_section | figure_title_cn | caption_draft_cn | evidence_role | use_priority | needs_redraw | redraw_reason | related_experiment_id | related_claim | split_risk_note | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FIG-106 | /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+CNN/real_original_sonic_signals.png | 第2章 数据与问题定义 | 2.1 XSI 声波原始数据 | 原始 XSI 声波多接收器波形示例 | 展示同一深度处多接收器声波波形形态，用于说明 CWT 输入来自多通道声波响应；该图仅用于数据形态说明，不包含模型性能结论。 | method_illustration | must_use | no |  | other | XSI waveform is the model input before CWT. | not_applicable_method_illustration | 原始声波图分辨率较高，适合放在数据章节。 |
| FIG-104 | /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+CNN/channeling_distribution.png | 第2章 数据与问题定义 | 2.2 CAST 窜槽标签分布 | CAST/CSI 窜槽分布统计示意 | 展示窜槽样本分布或类别占比，用于说明数据不平衡与后续严重度建模动机；图中数值不在本文中重新读取。 | method_illustration | must_use | no |  | other | CAST/CSI distribution motivates severity labels. | not_applicable_method_illustration | 不 OCR，若正文需要精确数值应引用原始表或重画。 |
| FIG-105 | /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+CNN/csi_distribution_analysis.png | 第2章 数据与问题定义 | 2.2 CAST/CSI 数据分布 | CSI 指标分布与窜槽样本统计 | 展示 CSI/窜槽相关分布特征，用于引出胶结质量与窜槽标签构造；仅作为数据探索图。 | method_illustration | useful | no |  | other | CSI distribution supports problem definition. | not_applicable_method_illustration | 适合与 FIG-104 合并为一张多子图数据概览。 |
| FIG-074 | /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+CNN/alignment_results.png | 第2章 数据与问题定义 | 2.3 方位失配问题 | XSI 与 CAST 方位/深度对齐结果示意 | 用于说明 XSI 与 CAST 之间存在对齐和方位匹配问题，是放弃直接点对点方位监督的重要背景。 | limitation | must_use | yes | 建议重画为简洁的 XSI-CAST 深度/方位关系示意，避免依赖工程调试图样式。 | other | Relative Bearing / azimuth matching is unreliable. | not_applicable_method_illustration | 作为问题动机图，不作为性能证据。 |
| FIG-075 | /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+CNN/depth_range_csi_analysis.png | 第2章 数据与问题定义 | 2.3 深度窗口与数据范围 | 研究井段深度范围与 CSI 分析示意 | 展示研究井段、深度范围或 CSI 分布，用于说明样本来自连续深度井段；也用于解释相邻深度泄漏风险。 | limitation | useful | yes | 建议重画为论文风格的深度轴示意，并标注后续需要 depth-blocked split。 | other | Samples are adjacent-depth windows, so random split is risky. | not_applicable_method_illustration | 可与 split 风险讨论联动。 |
| FIG-001 | /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+CNN/sample_scalograms.png | 第2章 数据构建 | 2.4 CWT 时频特征 | 多通道声波 CWT 时频图示例 | 展示声波波形经连续小波变换后的时频表示，是所有 CNN/EfficientNet 路线的核心输入形式。 | method_illustration | must_use | no |  | other | CWT scalogram is the shared input feature. | not_applicable_method_illustration | 高分辨率，适合直接使用或裁剪。 |
| FIG-009 | /mnt/c/Users/Administrator/Desktop/Hal/results/双通道学习/array_03/plots/validate_pipeline_02_CWT_vs_Label.png | 第2章 数据构建 | 2.5 CWT 与标签对应关系检查 | CWT 输入与标签构造管线校验图 | 展示 CWT 输入与标签之间的管线对应关系，用于说明特征-标签样本构建流程；不作为双通道方法有效性证据。 | method_illustration | useful | no |  | EXP-011 | CWT-to-label data construction was visually checked. | split_unknown_for_dual_channel_route | 来源于双通道目录，但此处只作管线示意。 |
| FIG-007 | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/test_relativity/output/visualization_plots/150_02_cwt_result.png | 第2章 数据构建 | 2.4 二分类 baseline 输入示例 | 二分类可学习性实验的 CWT 输入示例 | 展示 EXP-006 二分类 baseline 使用的 CWT 输入样例；正文只表述随机验证下存在可学习关系，不表述泛化性能。 | baseline_result | useful | no |  | EXP-006 | CWT contains learnable information for channeling existence under random validation. | random_split_depth_leakage_risk; exploratory_only | 不是 baseline 性能图，性能应放表格。 |
| FIG-049 | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/1D+percentage_Label/output/visualization_plots/06_profile_label_generation.png | 第3章 标签构造 | 3.2 一维窜槽百分比标签 | 1D 窜槽百分比深度剖面标签生成示意 | 展示将 CAST Zc 切片转化为深度方向窜槽百分比剖面的流程，是 EXP-007 fallback 主线的标签构造依据。 | method_illustration | must_use | yes | 建议重画为“Zc 切片 -> 阈值掩膜 -> 方位平均百分比 -> 深度剖面”的四步流程图。 | EXP-007 | 1D percentage label avoids direct azimuth matching. | random_split_depth_leakage_risk only affects reported metrics, not label definition | 现图可作为证据，正式论文建议重画。 |
| FIG-011 | /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet/output/visualization_plots/fft_regression/06_label_generation_fft_regression.png | 第3章 标签构造 | 3.3 FFT 幅值标签 | FFT 严重度幅值标签生成示意 | 展示严重度图经方位维 FFT 后取幅值的标签构造流程，用于说明丢弃相位以获得旋转不变表征。 | method_illustration | must_use | yes | 建议重画为“Zc -> severity=max(0,2.5-Zc) -> azimuth FFT -> magnitude/log magnitude”的方法图。 | EXP-008 | FFT magnitude label is the azimuth-invariant method candidate. | random_split_depth_leakage_risk only affects reported metrics, not label definition | 主线方法图，建议重画到论文统一风格。 |
| FIG-092 | /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet/output/fft_regression/array_03/results/fft_model_analysis_plots/combo_plots_High_Severity/combo_sample_1717.png | 第3章 标签构造 | 3.4 严重度标签样例 | 高严重度样本的 severity/FFT 标签与预测组合示例 | 展示高严重度样本中标签、预测或重构图的组合形态，用于说明主线标签的空间/频域表现；不读取图中数值。 | method_illustration | useful | yes | 组合图信息多，建议裁剪或重排为标签真值、预测、误差三个子图。 | EXP-008 | High-severity examples reveal what the FFT-severity label represents. | random_split_depth_leakage_risk; exploratory_only if interpreted as result | 可在方法章节作为标签形态示例，若讨论预测则必须标 exploratory。 |
| FIG-084 | /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+SE-ResNet/outputs/array_03/plots/model_architecture_array_03.png | 第4章 模型方法 | 4.2 模型结构 | 现有 SE-ResNet 模型结构图（主线 EfficientNet 结构需另画） | 展示旧项目中保存的模型结构图；由于论文主线是 EfficientNet/CWT 回归，该图仅作为已有结构证据或附录，正式主线结构图应重画。 | method_illustration | appendix | yes | 该图过高且对应 SE-ResNet，不是 EXP-008 EfficientNet 主线；需另画 EfficientNetV2B0 + 回归头/FFT标签流程结构图。 | EXP-010 | Existing model-architecture artifact confirms SE-ResNet route, not mainline EfficientNet. | split_unknown_for_se_resnet_route | 候选集中没有 EfficientNet 架构图。 |
| FIG-046 | /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet/output/visualization_plots/05_training_history_regression.png | 第5章 实验结果 | 5.2 EXP-008 训练过程 | FFT severity + EfficientNet 训练曲线 | 展示 EXP-008 训练/验证损失或 MAE 的变化趋势，只能作为随机划分下的训练过程证据。 | exploratory_result | useful | yes | 建议用 unified_metrics_table 中 PKL 数值重画训练曲线，并在图注标注 random split。 | EXP-008 | FFT regression has saved training-history metrics. | random_split_depth_leakage_risk; exploratory_only | 不作为最终泛化性能图。 |
| FIG-085 | /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet/output/fft_regression/array_03/results/fft_model_analysis_plots/_overall_performance_scatter.png | 第5章 实验结果 | 5.2 EXP-008 探索性结果 | FFT severity 回归预测-真值散点图 | 展示 EXP-008 在随机验证设置下预测与真值的散点关系，用于方法可行性讨论；由于 split 存在相邻深度泄漏风险，不表述为最终泛化性能。 | exploratory_result | must_use | no |  | EXP-008 | FFT severity regression is the mainline method candidate. | random_split_depth_leakage_risk; exploratory_only | 主线结果图，但图注必须避免“测试集泛化”。 |
| FIG-048 | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/1D+percentage_Label/output/visualization_plots/05_training_history_regression.png | 第5章 实验结果 | 5.3 EXP-007 训练过程 | 1D percentage + EfficientNet 训练曲线 | 展示 EXP-007 训练过程，是 fallback 主线的训练证据；受随机划分限制，只用于探索性分析。 | exploratory_result | useful | yes | 建议用 PKL 数值重画并合并训练/验证 MAE、loss，标注 random split。 | EXP-007 | 1D percentage route is fallback/exploratory mainline. | random_split_depth_leakage_risk; exploratory_only | 可与 FIG-046 并列对比。 |
| FIG-089 | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/1D+percentage_Label/output/image_translation/array_03/results/final_analysis_plots/_overall_performance_scatter.png | 第5章 实验结果 | 5.3 EXP-007 探索性结果 | 1D percentage 回归预测-真值散点图 | 展示 1D percentage 路线预测与真值之间的总体关系；仅作为随机划分下的探索性结果。 | exploratory_result | must_use | no |  | EXP-007 | 1D percentage label provides a fallback route but needs depth-blocked validation. | random_split_depth_leakage_risk; exploratory_only | 不能写成最终性能。 |
| FIG-057 | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/1D+percentage_Label/output/image_translation/array_03/results/final_analysis_plots/_error_distribution_by_category.png | 第5章 实验结果 | 5.4 严重度分组误差 | 1D percentage 严重度分组误差分布 | 展示不同严重度分组下误差差异，用于支撑高严重度预测偏差/误差增大的讨论；图中数值应以 result.txt/metrics 表为准。 | limitation | must_use | yes | 建议根据 result.txt 中 MAE/RMSE 表重画，突出 High Severity 误差最大。 | EXP-007 | High-severity error increases in the 1D percentage route. | random_split_depth_leakage_risk; exploratory_only | 该图与 result.txt 数值表配合使用。 |
| FIG-027 | /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+CNN/comprehensive_gradcam_statistics.png | 第6章 可解释性分析 | 6.1 Grad-CAM 统计 | CSI+CNN Grad-CAM 统计图 | 展示模型关注区域的统计性热力图，用于定性说明声波时频图中存在可解释敏感区域；不读取图中数值。 | interpretability | must_use | no |  | EXP-013 | Grad-CAM suggests sensitive time-frequency regions. | split_unknown_or_route_specific; qualitative_only | 适合作为可解释性总览。 |
| FIG-028 | /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+CNN/gradcam_analysis_optimized.png | 第6章 可解释性分析 | 6.1 Grad-CAM 样例 | CSI+CNN 优化 Grad-CAM 分析图 | 展示单样本或多样本 Grad-CAM 热力图，用于说明模型关注的时频区域；作为定性解释证据。 | interpretability | useful | no |  | EXP-013 | Grad-CAM supports qualitative feature localization. | split_unknown_or_route_specific; qualitative_only | 可与 FIG-027 合并。 |
| FIG-035 | /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet/output/image_translation/array_03/results/final_analysis_plots/_overall_mean_attention_map.png | 第6章 可解释性分析 | 6.2 EfficientNet 注意力区域 | EfficientNet 平均注意力图 | 展示 EfficientNet 路线整体关注的 CWT 区域，用于和 memo 中 0.5–1.0 ms、23–28 kHz 敏感区结论对照；不作为定量统计。 | interpretability | must_use | no |  | EXP-013 | EfficientNet/CWT route attends to sensitive time-frequency regions. | random_split_depth_leakage_risk for associated model; qualitative_only | 图来自 FFT_EfficientNet image_translation 路线，注意与 EXP-008 FFT regression 区分。 |
| FIG-036 | /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet/output/image_translation/array_03/results/final_analysis_plots/gradcam_sample_312.png | 第6章 可解释性分析 | 6.2 单样本 Grad-CAM | EfficientNet 单样本 Grad-CAM 示例 | 展示具体样本的 CWT 输入与热力区域，用于说明模型关注区域可视化方式。 | interpretability | useful | no |  | EXP-013 | Single-sample Grad-CAM gives qualitative interpretability. | random_split_depth_leakage_risk for associated model; qualitative_only | 只选一张代表样本，避免堆图。 |
| FIG-029 | /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+SE-ResNet/outputs/array_03/plots/attention_mean_variance.png | 第6章 可解释性分析 | 6.3 SE-ResNet 注意力统计 | SE-ResNet 注意力均值与方差图 | 展示 SE-ResNet 路线的注意力统计，可作为对比解释路线或附录证据。 | interpretability | appendix | no |  | EXP-010 | SE-ResNet route has attention artifacts but code/result mapping remains weaker. | split_unknown; exploratory_only | 建议放附录或讨论，不作为主线解释。 |
| FIG-030 | /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+SE-ResNet/outputs/array_03/plots/class_conditional_attention_analysis.png | 第6章 可解释性分析 | 6.3 类条件注意力 | SE-ResNet 类条件注意力分析图 | 展示不同类别条件下的注意力差异，用于支持模型解释性讨论；由于路线非主线，建议附录。 | interpretability | appendix | yes | 图幅较高，建议裁剪为类别均值热力图和简短说明。 | EXP-010 | Class-conditional attention is available but not mainline evidence. | split_unknown; exploratory_only | 不作为主线结果。 |
| FIG-045 | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/baseline/visualization_plots/04_training_history.png | 第5章 对比实验 | 5.1 早期 FFT baseline | 早期 FFT baseline 训练曲线 | 展示早期 FFT 图像翻译 baseline 的训练过程，用于说明旧路线存在预测塌缩/效果不足；不作为主线性能。 | baseline_result | useful | yes | 建议重画为小图或表格化，配合 memo 中“只能预测整体平均”的失败说明。 | EXP-001 | Baseline FFT image-translation route was weak. | split_unknown_or_unverified; exploratory_only | 作为失败/对比背景。 |
| FIG-041 | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/log_label/output/visualization_plots/04_training_history.png | 第5章 消融实验 | 5.1 log label 消融 | log label 训练曲线 | 展示 log label 变体的训练曲线，用于说明标签变换带来训练/注意力变化；受随机 split 风险限制。 | exploratory_result | useful | yes | 建议用 PKL 指标重画，并标注为随机划分下消融。 | EXP-002 | Log transform was an ablation route, not final performance evidence. | random_split_depth_leakage_risk; exploratory_only | 配合 memo qualitative claim。 |
| FIG-055 | /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet/output/image_translation/array_03/results/final_analysis_plots/_error_distribution_by_category.png | 第5章 消融实验 | 5.1 旧回归路线误差分组 | 旧 EfficientNet 回归路线误差分布 | 展示旧 image_translation/1D 路线按类别的误差分布，用于对比主线与 fallback 的局限。 | limitation | useful | yes | 建议与 FIG-057 合并重画，避免重复并统一类别名称。 | EXP-007 | Error distribution by severity/category is available as image-only evidence. | random_split_depth_leakage_risk; exploratory_only | 图中数值需人工读图或用已有文本指标替代。 |
| FIG-060 | /mnt/c/Users/Administrator/Desktop/Hal/results/双通道学习/array_03/plots/performance_summary_plots.png | 第7章 讨论 | 7.2 双通道路线失败 | 双通道 metadata fusion 性能概要图 | 展示双通道/元数据融合路线的性能概要，用于讨论失败尝试；没有可抽取数值，不能作为主线。 | failed_attempt | appendix | yes | 图中数值未 OCR，建议只保留代表性子图或重画为“失败路线概览”示意。 | EXP-011 | Dual-channel metadata fusion is failed/weak evidence. | split_unknown; exploratory_only | 附录或讨论章。 |
| FIG-061 | /mnt/c/Users/Administrator/Desktop/Hal/results/预校正/array_03/plots/performance_summary_plots.png | 第7章 讨论 | 7.2 偏心预校正失败 | 偏心预校正性能概要图 | 展示 eccentricity pre-correction 路线的性能概要，用于说明预校正未形成可靠主线。 | failed_attempt | appendix | yes | 图中数值未 OCR，建议与双通道路线合并为失败路线对比图。 | EXP-012 | Eccentricity pre-correction is failed/weak evidence. | split_unknown; exploratory_only | 附录或讨论章。 |
| FIG-062 | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/GaN+2Dlabel/output/image_translation/array_03/results/classification_analysis_plots/sample_0_mask_comparison.png | 第7章 讨论 | 7.3 GAN/二通道标签失败 | GAN 二通道掩膜预测对比示例 | 展示 GAN/二通道二值标签路线的预测掩膜与真值对比，用于作为失败尝试证据。 | failed_attempt | appendix | yes | 建议选 1–2 个样本重排为附录图，并配合 train-only loss 说明。 | EXP-004 | GAN/two-channel image-generation route collapsed or failed. | needs_manual_verification; train_only_or_failed_route | 不作为验证集性能。 |
| FIG-072 | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/GaN+2Dlabel/output/visualization_plots/04_training_history.png | 第7章 讨论 | 7.3 GAN 训练过程 | GAN 路线训练曲线 | 展示 GAN 路线训练过程，用于说明 generator/discriminator loss 未形成稳定有效结果。 | failed_attempt | appendix | yes | 建议用 result.txt 中 epoch loss 重画简洁曲线；当前图不读取数值。 | EXP-004 | GAN route is a failed attempt; losses are train-only. | needs_manual_verification; train_only_metric | 附录图。 |
| FIG-090 | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/1D+percentage_Label/样本权重+非对称损失/output/image_translation/array_03/results/final_analysis_plots/_overall_performance_scatter.png | 第7章 讨论 | 7.4 样本权重失败路线 | 样本权重+非对称损失散点图 | 展示样本权重与非对称损失路线的预测-真值关系，用于说明该补救策略仍效果较差；仅作失败尝试。 | failed_attempt | appendix | yes | 建议与 FIG-089 对照重画，突出未解决高严重度误差问题。 | EXP-014 | Sample weights + asymmetric loss was a failed attempt. | random_split_depth_leakage_risk; exploratory_only | 远程 commit 和本地 result.txt 均支持 failed attempt。 |
| FIG-059 | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/1D+percentage_Label/样本权重+非对称损失/output/image_translation/array_03/results/weighted_model_analysis_plots/_error_distribution_by_category_weighted.png | 第7章 讨论 | 7.4 样本权重误差分布 | 样本权重+非对称损失严重度误差分布 | 展示样本权重策略下的误差分布，用于说明其未改善高严重度预测问题；作为附录/讨论证据。 | failed_attempt | appendix | yes | 建议根据可用 PKL/text 指标重画或仅作为附录图。 | EXP-014 | Weighted/asymmetric route remained poor. | random_split_depth_leakage_risk; exploratory_only | 不作为主线结果。 |
| FIG-076 | /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+CNN/filtering_effect_comparison.png | 第7章 讨论 | 7.1 信号处理限制 | 滤波前后信号对比图 | 展示信号预处理和滤波影响，用于讨论声波数据质量、CWT 构造和高频敏感区域的物理合理性。 | limitation | useful | yes | 建议重画为简洁的原始/滤波波形及频带说明。 | other | Signal preprocessing affects CWT features. | not_applicable_method_illustration | 讨论章节可用。 |
| FIG-052 | /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+SE-ResNet/outputs/array_03/plots/performance_summary_plots.png | 第7章 讨论 | 7.2 SE-ResNet 对比路线 | CSI+SE-ResNet 性能概要图 | 展示 SE-ResNet 方位匹配/分类路线的性能概要，用于对比早期方位匹配路线和主线标签路线。 | appendix | appendix | yes | 图中数值未 OCR，建议只作附录或重画为路线对比示意。 | EXP-010 | SE-ResNet route is not the final mainline and split/code mapping remains weaker. | split_unknown; exploratory_only | 不作为主线结论。 |

<!-- EXP008_DEPTH_HELDOUT_TRAINING_START -->
## EXP-008 Depth-Heldout Figure Addendum (2026-07-07)

| figure_id | source_path | thesis_chapter | thesis_section | figure_title_cn | caption_draft_cn | evidence_role | use_priority | needs_redraw | related_experiment_id | split_risk_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FIG-DH-001 | docs/thesis_evidence/remote_exp008_depth_blocked_train/training_curve.png | 第5章 实验结果 | 5.2 EXP-008 depth-heldout 训练 | EXP-008 depth-heldout 训练曲线 | 展示显式 depth-heldout split 下训练/验证 loss 与 MAE 曲线；EarlyStopping 在第 12 轮停止并恢复第 2 轮权重。 | exploratory_result | must_use | no | EXP-008 | depth_heldout_split_confirmed; single-well only |
| FIG-DH-002 | docs/thesis_evidence/remote_exp008_depth_blocked_train/prediction_vs_truth_scatter.png | 第5章 实验结果 | 5.2 EXP-008 depth-heldout 测试 | EXP-008 depth-heldout 测试集预测-真值散点图 | 展示 heldout test 样本的平均 integrated severity 预测关系；只作为 array_03 单井 depth-heldout 结果，不表述为多井泛化。 | exploratory_result | must_use | no | EXP-008 | depth_heldout_split_confirmed; single-well only |
| FIG-DH-003 | docs/thesis_evidence/remote_exp008_depth_blocked_train/depth_curve_if_available.png | 第5章 实验结果 | 5.2 EXP-008 depth-heldout 深度曲线 | EXP-008 depth-heldout 测试深度曲线 | 展示 heldout tail depth interval 上真值与预测的深度趋势，用于说明模型在连续深度留出段上的趋势捕捉与误差。 | limitation | useful | yes | EXP-008 | depth_heldout_split_confirmed; single-well only |
<!-- EXP008_DEPTH_HELDOUT_TRAINING_END -->

<!-- EXP008_RESULT_AUDIT_P3_START -->
## EXP-008 Result Audit P3 Figure Addendum (2026-07-07)

| figure_id | source_path | thesis_chapter | thesis_section | figure_title_cn | caption_draft_cn | evidence_role | use_priority | needs_redraw | related_experiment_id | split_risk_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FIG-DH-P3-001 | docs/thesis_evidence/remote_exp008_depth_blocked_train/figures_redraw/training_curve_redraw.png | 第5章 实验结果 | 5.2 EXP-008 depth-heldout | EXP-008 depth-heldout 训练曲线重画图 | 展示 train_v001 的 loss/MAE 曲线和早停行为，用于说明早期过拟合与 best epoch。 | exploratory_result | must_use | no | EXP-008 | depth_heldout_single_well |
| FIG-DH-P3-002 | docs/thesis_evidence/remote_exp008_depth_blocked_train/figures_redraw/baseline_comparison_bar.png | 第5章 实验结果 | 5.2 基线对照 | EXP-008 与简单基线对照 | 展示模型、zero、train-mean、train-median 的 MAE/RMSE 对照；图注必须说明 MAE 不优于 zero baseline。 | limitation | must_use | no | EXP-008 | depth_heldout_single_well |
| FIG-DH-P3-003 | docs/thesis_evidence/remote_exp008_depth_blocked_train/figures_redraw/per_fft_coefficient_mae.png | 第5章 实验结果 | 5.3 标签维度误差 | EXP-008 按 FFT 系数的 MAE | 展示低频系数 k=0-4 误差最大，高频系数绝对误差较低但目标幅值也较小。 | limitation | must_use | no | EXP-008 | depth_heldout_single_well |
| FIG-DH-P3-004 | docs/thesis_evidence/remote_exp008_depth_blocked_train/figures_redraw/error_distribution.png | 第7章 讨论 | 7.1 误差分布 | EXP-008 测试样本误差分布 | 展示样本级误差长尾，用于说明高严重度低估和标签稀疏带来的指标风险。 | limitation | useful | no | EXP-008 | depth_heldout_single_well |
<!-- EXP008_RESULT_AUDIT_P3_END -->

<!-- EXP007_DEPTH_HELDOUT_FALLBACK_START -->
## EXP-007 Depth-Heldout Fallback Figure Addendum (2026-07-07)

These figures were generated by the remote EXP-007 training script under `/home/xiaoj/hal_azi/output/thesis_depth_blocked/exp007/train_v002`. They were not copied into local docs during this run because SSH/SCP escalation was rejected by the execution environment usage limit. Use the remote paths as traceable sources until the small image files can be copied in a later approved session.

| figure_id | source_path | thesis_chapter | thesis_section | figure_title_cn | caption_draft_cn | evidence_role | use_priority | needs_redraw | related_experiment_id | split_risk_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FIG-DH-EXP007-001 | `/home/xiaoj/hal_azi/output/thesis_depth_blocked/exp007/train_v002/training_curve.png` | 第5章 实验结果 | 5.4 EXP-007 depth-heldout fallback | EXP-007 depth-heldout 训练曲线 | 展示 1D percentage label fallback 在 single-well depth-heldout split 下的训练/验证 loss 与 MAE；train_v002 best validation evidence 位于第 1 轮，后续未继续改善。 | exploratory_result | useful | yes | EXP-007 | depth_heldout_split_confirmed; single-well only |
| FIG-DH-EXP007-002 | `/home/xiaoj/hal_azi/output/thesis_depth_blocked/exp007/train_v002/prediction_vs_truth_scatter.png` | 第5章 实验结果 | 5.4 EXP-007 depth-heldout fallback | EXP-007 depth-heldout 测试预测-真值散点图 | 展示 heldout test 样本的 1D percentage 预测关系；图注必须说明模型优于 train-mean/train-median 基线，但未优于 zero baseline 的 MAE。 | exploratory_result | useful | yes | EXP-007 | depth_heldout_split_confirmed; single-well only |
| FIG-DH-EXP007-003 | `/home/xiaoj/hal_azi/output/thesis_depth_blocked/exp007/train_v002/residual_distribution.png` | 第7章 讨论 | 7.1 稀疏标签与误差分布 | EXP-007 depth-heldout 残差分布 | 展示 1D percentage 标签下 residual 分布，用于说明稀疏标签使 zero baseline 在 MAE 上较强。 | limitation | useful | yes | EXP-007 | depth_heldout_split_confirmed; single-well only |
| FIG-DH-EXP007-004 | `/home/xiaoj/hal_azi/output/thesis_depth_blocked/exp007/train_v002/per_profile_index_mae.png` | 第5章 实验结果 | 5.4 深度剖面误差 | EXP-007 按剖面位置的 MAE | 展示 70 点 1D percentage 标签各位置的误差分布；当前结论只可写成位置维度误差结构，严重度分组细节仍需复制远程 `severity_group_metrics.csv/json` 后核验。 | limitation | useful | yes | EXP-007 | depth_heldout_split_confirmed; single-well only |
| FIG-DH-EXP007-005 | `/home/xiaoj/hal_azi/output/thesis_depth_blocked/exp007/train_v002/depth_curve_if_available.png` | 第7章 讨论 | 7.1 连续深度留出段表现 | EXP-007 depth-heldout 测试深度曲线 | 展示 heldout tail depth interval 上的真值与预测趋势；仅用于单井连续深度留出段行为分析，不能写成多井泛化。 | limitation | appendix | yes | EXP-007 | depth_heldout_split_confirmed; single-well only |
<!-- EXP007_DEPTH_HELDOUT_FALLBACK_END -->
