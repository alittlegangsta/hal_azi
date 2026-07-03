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
