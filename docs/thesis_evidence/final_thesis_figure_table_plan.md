# Final Thesis Figure And Table Plan

Generated: 2026-07-08. This plan freezes the figure/table arrangement for writing. Existing Windows results remain read-only; no OCR or image modification was performed.

## Must-Use Tables

| table | chapter | source | purpose | limitation note |
| --- | --- | --- | --- | --- |
| 实验路线与证据强度表 | 第5章 | `docs/thesis_evidence/experiment_inventory.csv`; `experiment_inventory.md` | Show how EXP-006/007/008 and failed routes fit the thesis. | Mark old random-split metrics as exploratory. |
| 最终论文指标表 | 第5章 | `docs/thesis_evidence/final_thesis_metrics_table.csv` | Main frozen result table. | Separate EXP-006 random split from EXP-007/008 depth-heldout. |
| EXP-008 基线对照表 | 第5章 | `docs/thesis_evidence/exp008_depthheldout_baseline_comparison.csv` | Show model vs zero/train-mean/train-median. | Explicitly state MAE is worse than zero. |
| EXP-007 fallback 基线对照表 | 第5章 | `docs/thesis_evidence/exp007_depthheldout_baseline_comparison.csv` | Show fallback performance and limitations. | Explicitly state MAE is worse than zero. |
| EXP-007 vs EXP-008 对比表 | 第5章 / 第7章 | `docs/thesis_evidence/exp007_vs_exp008_depthheldout_comparison.csv` | Justify final thesis organization. | Labels differ, so compare role/evidence strength rather than absolute scale only. |
| Split audit 表 | 第5章 / 第7章 | `remote_exp008_split_v001/*`; `remote_exp007_split_v001/*`; `split_forensic_audit.csv` | Explain why old random-split results were not used as final performance. | Single-well depth-heldout is still not multi-well generalization. |
| 失败路线汇总表 | 第7章 / 附录 | `experiment_inventory.md`; `memo_experiment_claims.csv`; `branch_experiment_mapping.csv` | Summarize GAN, dual-channel, eccentricity correction, weighted/asymmetric loss. | Do not over-discuss failed routes in the main result chapter. |

## Must-Use Figures

| figure | chapter | source_path | use | redraw |
| --- | --- | --- | --- | --- |
| 原始 XSI 多接收器波形 | 第2章 | `/mnt/c/Users/Administrator/Desktop/Hal/results/CSI+CNN/real_original_sonic_signals.png` | Data input illustration. | optional crop only |
| CWT 时频图示例 | 第2章 | `/mnt/c/Users/Administrator/Desktop/Hal/results/CSI+CNN/sample_scalograms.png` | Explain CWT feature representation. | optional crop only |
| XSI-CAST 方位/深度失配示意 | 第2章 | `/mnt/c/Users/Administrator/Desktop/Hal/results/CSI+CNN/alignment_results.png` | Motivate weak-label/azimuth-invariant label design. | must redraw in thesis style |
| 1D percentage 标签构造 | 第3章 | `/mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/1D+percentage_Label/output/visualization_plots/06_profile_label_generation.png` | Explain EXP-007 fallback label. | must redraw |
| FFT severity 标签构造 | 第3章 | `/mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet/output/visualization_plots/fft_regression/06_label_generation_fft_regression.png` | Explain EXP-008 main label. | must redraw |
| EfficientNetV2B0 + CWT + regression head 方法图 | 第4章 | code-derived from `scripts/thesis_train_exp008_depth_blocked.py`; `scripts/thesis_train_exp007_depth_blocked.py` | Main model architecture. | must draw new figure; no suitable existing mainline architecture figure found |
| EXP-008 depth-heldout training curve | 第5章 | `docs/thesis_evidence/remote_exp008_depth_blocked_train/figures_redraw/training_curve_redraw.png` | Show early overfitting and best epoch. | already redrawn |
| EXP-008 baseline comparison | 第5章 | `docs/thesis_evidence/remote_exp008_depth_blocked_train/figures_redraw/baseline_comparison_bar.png` | Show metric-qualified baseline comparison. | already redrawn |
| EXP-008 per-FFT-coefficient MAE | 第5章 / 第7章 | `docs/thesis_evidence/remote_exp008_depth_blocked_train/figures_redraw/per_fft_coefficient_mae.png` | Support low-frequency error limitation. | already redrawn |
| EXP-007 depth-heldout training/scatter/residual figures | 第5章 / 第7章 | remote `/home/xiaoj/hal_azi/output/thesis_depth_blocked/exp007/train_v002/*.png` | Fallback/limitation comparison. | must copy small files later and redraw if needed |
| Grad-CAM summary | 第6章 | `/mnt/c/Users/Administrator/Desktop/Hal/results/CSI+CNN/comprehensive_gradcam_statistics.png`; `/mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet/output/image_translation/array_03/results/final_analysis_plots/_overall_mean_attention_map.png` | Qualitative interpretability. | optional redraw/crop; do not extract numeric claims from image |

## Appendix Figures

| route | representative source | appendix role |
| --- | --- | --- |
| EXP-006 random baseline | `/mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/test_relativity/output/visualization_plots/150_02_cwt_result.png` | Random-split learnability background. |
| SE-ResNet azimuth matching | `/mnt/c/Users/Administrator/Desktop/Hal/results/CSI+SE-ResNet/outputs/array_03/plots/performance_summary_plots.png` | Early route / non-final method. |
| dual-channel metadata fusion | `/mnt/c/Users/Administrator/Desktop/Hal/results/双通道学习/array_03/plots/performance_summary_plots.png` | Failed attempt. |
| eccentricity pre-correction | `/mnt/c/Users/Administrator/Desktop/Hal/results/预校正/array_03/plots/performance_summary_plots.png` | Failed attempt. |
| GAN / two-channel binary label | `/mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/GaN+2Dlabel/output/image_translation/array_03/results/classification_analysis_plots/sample_0_mask_comparison.png` | Failed attempt. |
| sample weights + asymmetric loss | `/mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/1D+percentage_Label/样本权重+非对称损失/output/image_translation/array_03/results/final_analysis_plots/_overall_performance_scatter.png` | Failed attempt / limitation. |

## Redraw Priority

1. Method pipeline: XSI waveform -> CWT -> EfficientNet -> FFT severity label / 1D percentage label.
2. EXP-008 label construction: Zc -> severity -> azimuth FFT magnitude -> model target.
3. EXP-008 result panel: training curve, baseline bar, per-FFT error, prediction scatter.
4. EXP-007 fallback panel after remote small figures are copied locally.
5. Failure route overview as one compact appendix figure rather than many separate plots.
