# Metric Extraction Report

Generated: 2026-07-06.

## Scope And Constraints

- Read-only evidence root: `/mnt/c/Users/Administrator/Desktop/Hal/results`.
- Extracted only from existing `.txt`, `.md`, `.csv`, `.json`, `.yaml`, `.yml`, `.pkl`, and metadata for TensorBoard/image files.
- No OCR was performed. Metrics visible only in figures are marked `image_only_needs_manual_read`.
- No training, no feature recomputation, no raw-data modification, no results-file modification.
- Local Python lacks TensorBoard/TensorFlow; TensorBoard event files are traced but scalar values were not read.
- Keras `training_history*.pkl` files were read with a narrow fake-NumPy scalar adapter to recover dictionaries of recorded floats; no model code was executed.

## Counts

| item | count |
| --- | --- |
| metric_rows_total | 256 |
| pkl_metric_rows | 163 |
| text_or_memo_metric_rows | 48 |
| image_only_rows | 45 |
| traceability_sources | 144 |
| training_history_pkl_files | 15 |
| tensorboard_event_files_traced_unparsed | 72 |

## Main Thesis Metrics Summary

| category | experiment_id | available_metrics | source | caveat |
| --- | --- | --- | --- | --- |
| mainline_candidate | EXP-008 | FFT regression history: val_loss min 0.011499; val_mae min 0.043332; final val_mae 0.051927; image scatter/depth/FFT maps need manual read | /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet/output/fft_regression/array_03/logs/training_history_fft.pkl | split_unknown; leakage_risk_unknown; target unit normalized/FFT-label units; duplicate FFT_EfficientNet_1 copy exists |
| fallback_mainline | EXP-007 | 1D severity-group MAE/RMSE from result.txt; val_auc/val_accuracy history files also exist but appear duplicated with binary-classification history | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/1D+percentage_Label/result.txt | all-samples severity table, not verified independent test split; high severity MAE/RMSE is worst |
| baseline | EXP-006 | CNN binary baseline val_auc 0.95361; pkl val_auc max 0.953608; val_accuracy max 0.885366; early stop epoch 72, best epoch 57 | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/test_relativity/result.txt.txt | classification baseline only; split_unknown and leakage_risk_unknown |
| ablation | EXP-002/EXP-003 | log label and frequency-weighted loss histories contain loss/MAE/val_loss/val_mae; memo conclusions are qualitative | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/log_label/.../training_history.pkl; /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/frequency-weighted_loss/.../training_history.pkl | metric units depend on label transform; no verified test split |
| failed_attempt | EXP-004/EXP-005/EXP-011/EXP-012/EXP-014 | GAN generator/discriminator losses; sample-weight pkl history; qualitative very poor result; dual-channel/pre-correction image-only summaries | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/GaN+2Dlabel/result.txt.txt; /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/1D+percentage_Label/样本权重+非对称损失/result.txt | retain as failed attempts or appendix, not mainline recommendation |
| appendix | EXP-009/EXP-010/EXP-013 | CSI+CNN, CSI+SE-ResNet and Grad-CAM mostly image-only analysis artifacts | /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+CNN; /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+SE-ResNet; /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/test_relativity/.../classification_gradcam_plots | numeric metrics require manual reading/redraw from images or locating original logs |

## Metrics Sufficient For Direct Thesis Text

| experiment_id | claim | metrics | source | limits |
| --- | --- | --- | --- | --- |
| EXP-006 | CNN binary baseline can distinguish channeling/non-channeling in validation evidence | val_auc 0.95361 from text; pkl val_auc max 0.9536079; val_accuracy max 0.8853658 | temp_result/test_relativity/result.txt.txt; temp_result/test_relativity/.../training_history.pkl | split_unknown; leakage_risk_unknown |
| EXP-007 | 1D percentage model error increases with severity | MAE/RMSE: Negligible 0.010/0.150; Low 2.895/3.839; Medium 4.782/6.035; High 6.555/9.597 | temp_result/1D+percentage_Label/result.txt | source says all samples; split_unknown |
| EXP-008 | FFT regression mainline has validation training-history metrics | val_loss min 0.011499; val_mae min 0.043332; final val_mae 0.051927 | FFT_EfficientNet/output/fft_regression/array_03/logs/training_history_fft.pkl | not a held-out/depth-blocked test metric; label unit needs description |
| EXP-014 | sample weights + asymmetric loss should be appendix/failed attempt | weighted pkl history exists; result.txt says 结果很差 | temp_result/1D+percentage_Label/样本权重+非对称损失/result.txt; training_history_weighted.pkl | failed attempt; do not use as mainline |

## Image-Only Metrics Needing Manual Read Or Redraw

| experiment_id | experiment_name | image_metric_sources_count | example_sources | status |
| --- | --- | --- | --- | --- |
| EXP-001 | Baseline FFT magnitude image translation | 1 | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/baseline/visualization_plots/04_training_history.png | image_only_needs_manual_read |
| EXP-002 | FFT log-label image translation | 1 | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/log_label/output/visualization_plots/04_training_history.png | image_only_needs_manual_read |
| EXP-005 | Two-channel binary label and focal-loss/overfit test | 1 | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/GaN+2Dlabel/output/visualization_plots/04_training_history.png | image_only_needs_manual_read |
| EXP-007 | 1D percentage label profile regression | 9 | /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet/output/image_translation/array_03/results/final_analysis_plots/_error_distribution_by_category.png; /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet/output/image_translation/array_03/results/final_analysis_plots/_overall_performance_scatter.png; /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet/output/image_translation/array_03/results/original_model_analysis_plots/_virtual_log_comparison.png; /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet_1/output/image_translation/array_03/results/final_analysis_plots/_error_distribution_by_category.png; /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet_1/output/image_translation/array_03/results/final_analysis_plots/_overall_performance_scatter.png | image_only_needs_manual_read |
| EXP-008 | EfficientNet FFT severity regression | 5 | /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet/output/fft_regression/array_03/results/fft_model_analysis_plots/_overall_performance_scatter.png; /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet/output/fft_regression/array_03/results/fft_model_analysis_plots/_virtual_log_comparison.png; /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet_1/output/fft_regression/array_03/results/fft_model_analysis_plots/_depth_log_comparison.png; /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet_1/output/fft_regression/array_03/results/fft_model_analysis_plots/_fft_spectrum_image_comparison.png; /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet_1/output/fft_regression/array_03/results/fft_model_analysis_plots/_overall_performance_scatter.png | image_only_needs_manual_read |
| EXP-009 | CSI + CNN visual/Grad-CAM analysis | 3 | /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+CNN/comprehensive_gradcam_statistics.png; /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+CNN/filtering_effect_comparison.png; /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+CNN/gradcam_analysis_optimized.png | image_only_needs_manual_read |
| EXP-010 | CSI + SE-ResNet azimuth matching / classification | 9 | /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+SE-ResNet/outputs/array_03/plots/attention_mean_variance.png; /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+SE-ResNet/outputs/array_03/plots/class_conditional_attention_analysis.png; /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+SE-ResNet/outputs/array_03/plots/performance_summary_plots.png; /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+SE-ResNet/outputs/array_07/plots/attention_mean_variance.png; /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+SE-ResNet/outputs/array_07/plots/class_conditional_attention_analysis.png | image_only_needs_manual_read |
| EXP-011 | Dual-channel metadata fusion | 2 | /mnt/c/Users/Administrator/Desktop/Hal/results/双通道学习/array_03/plots/class_conditional_attention_analysis.png; /mnt/c/Users/Administrator/Desktop/Hal/results/双通道学习/array_03/plots/performance_summary_plots.png | image_only_needs_manual_read |
| EXP-012 | Eccentricity pre-correction | 2 | /mnt/c/Users/Administrator/Desktop/Hal/results/预校正/array_03/plots/class_conditional_attention_analysis.png; /mnt/c/Users/Administrator/Desktop/Hal/results/预校正/array_03/plots/performance_summary_plots.png | image_only_needs_manual_read |
| EXP-013 | Grad-CAM interpretability across routes | 5 | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/test_relativity/output/image_translation/array_03/results/classification_gradcam_plots/gradcam_sample_0.png; /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/test_relativity/output/image_translation/array_03/results/classification_gradcam_plots/gradcam_sample_1.png; /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/test_relativity/output/image_translation/array_03/results/classification_gradcam_plots/gradcam_sample_2.png; /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/test_relativity/output/image_translation/array_03/results/classification_gradcam_plots/gradcam_sample_3.png; /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/test_relativity/output/image_translation/array_03/results/classification_gradcam_plots/gradcam_sample_4.png | image_only_needs_manual_read |
| EXP-014 | 1D percentage label + sample weights + asymmetric loss failed attempt | 5 | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/1D+percentage_Label/样本权重+非对称损失/output/image_translation/array_03/results/final_analysis_plots/_error_distribution_by_category.png; /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/1D+percentage_Label/样本权重+非对称损失/output/image_translation/array_03/results/final_analysis_plots/_overall_performance_scatter.png; /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/1D+percentage_Label/样本权重+非对称损失/output/image_translation/array_03/results/weighted_model_analysis_plots/_error_distribution_by_category_weighted.png; /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/1D+percentage_Label/样本权重+非对称损失/output/image_translation/array_03/results/weighted_model_analysis_plots/_overall_performance_scatter_weighted.png; /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/1D+percentage_Label/样本权重+非对称损失/output/visualization_plots/05_training_history_regression.png | image_only_needs_manual_read |
| unknown | unknown | 2 | /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet/output/visualization_plots/05_training_history_regression.png; /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet_1/output/visualization_plots/05_training_history_regression.png | image_only_needs_manual_read |

## Missing Split Information

All extracted experiments currently remain `split_unknown` for thesis-grade reporting. Validation metrics are labeled by source key (`val_*`) but the split construction is not proven depth-blocked or leak-free in the result artifacts.

EXP-001, EXP-002, EXP-003, EXP-004, EXP-005, EXP-006, EXP-007, EXP-008, EXP-009, EXP-010, EXP-011, EXP-012, EXP-013, EXP-014, unknown


## Leakage Risk Unknown

The metric table deliberately marks `leakage_risk_unknown` for extracted rows because adjacent-depth leakage/depth-blocked validation evidence is not present in the metric sources.

EXP-001, EXP-002, EXP-003, EXP-004, EXP-005, EXP-006, EXP-007, EXP-008, EXP-009, EXP-010, EXP-011, EXP-012, EXP-013, EXP-014, unknown


## Duplicate Training Histories

| hash_prefix | count | example_paths |
| --- | --- | --- |
| 5e7990a1ce089f30 | 2 | FFT_EfficientNet/output/fft_regression/array_03/logs/training_history_fft.pkl; FFT_EfficientNet_1/output/fft_regression/array_03/logs/training_history_fft.pkl |
| 673fb5e8b058b415 | 6 | FFT_EfficientNet/output/image_translation/array_03/logs/training_history.pkl; FFT_EfficientNet_1/output/image_translation/array_03/logs/training_history.pkl; temp_result/1D+percentage_Label/output/image_translation/array_03/logs/training_history.pkl; temp_result/1D+percentage_Label/样本权重+非对称损失/output/image_translation/array_03/logs/training_history.pkl; temp_result/GaN+2Dlabel/output/image_translation/array_03/logs/training_history.pkl |
| cc58fc273535b8e2 | 4 | FFT_EfficientNet/output/image_translation/array_03/logs/training_history_advanced.pkl; FFT_EfficientNet_1/output/image_translation/array_03/logs/training_history_advanced.pkl; temp_result/1D+percentage_Label/output/image_translation/array_03/logs/training_history_advanced.pkl; temp_result/1D+percentage_Label/样本权重+非对称损失/output/image_translation/array_03/logs/training_history_advanced.pkl |

## Traceability Status Counts

| status | count |
| --- | --- |
| image_only_needs_manual_read | 45 |
| memo_claim_metric_claim | 1 |
| memo_claim_no_metric | 1 |
| memo_claim_qualitative_only | 5 |
| metrics_extracted | 20 |
| tensorboard_event_unparsed_local_tensorboard_missing | 72 |
