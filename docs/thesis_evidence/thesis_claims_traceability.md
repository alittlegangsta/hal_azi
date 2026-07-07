# Thesis Claims Traceability

Every claim below is tied to at least one result path, code path, git branch/commit, memo path, or explicit inference marker.

| claim_id | claim | evidence | status |
| --- | --- | --- | --- |
| C-001 | CWT 时频图与窜槽存在性之间存在可学习关系。 | EXP-006; temp_result/test_relativity/result.txt.txt; memo AUC=0.95361 | supported |
| C-002 | 1D 窜槽百分比标签规避方位失配并得到较稳定敏感区域。 | EXP-007; origin/1D+percentage_Label code; memo section 6 | supported, metric table still needed |
| C-003 | FFT 幅值/丢弃相位是处理方位旋转不变性的合理标签路线。 | memo PPT section; create_tfrecords FFT code | method rationale supported, performance needs verification |
| C-004 | log transform improves Grad-CAM localization but does not solve prediction quality. | EXP-002; memo section 1; temp_result/log_label plots | supported by memo/result artifacts |
| C-005 | frequency-weighted FFT loss did not solve low-coefficient collapse. | EXP-003; memo section 2; origin/frequency-weighted_loss train.py | supported by memo/code |
| C-006 | GAN/two-channel image-generation routes collapsed. | EXP-004; EXP-005; memo sections 3-4; result.txt losses | supported for GAN, two-channel implementation mapping needs verification |
| C-007 | Dual-channel and pre-correction failed/poor. | result directories 双通道学习 and 预校正; user task framing | explicitly_marked_inference_needs_verification |
| C-008 | Sensitive CWT region is concentrated in high-frequency bands roughly 22-30 kHz and 0.5-1.3 ms depending on route. | memo sections 1/2/6; temp_result/test_relativity/result.txt.txt; Grad-CAM result files | supported qualitatively; batch statistics recommended |
<!-- METRIC_EXTRACTION_AUTO_START -->
## Metric-Based Claim Traceability Update (2026-07-06)

| claim | evidence | source | status |
| --- | --- | --- | --- |
| CNN binary baseline learned CWT/channeling correspondence | EXP-006 val_auc 0.95361; val_accuracy max 0.885366 | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/test_relativity/result.txt.txt and training_history.pkl | usable with caveat split_unknown/leakage_risk_unknown |
| 1D percentage label errors increase with severity | MAE grows from 0.010 Negligible to 6.555 High Severity; RMSE from 0.150 to 9.597 | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/1D+percentage_Label/result.txt | usable as all-samples severity analysis, not verified held-out test |
| FFT magnitude regression is best-supported mainline candidate | val_loss min 0.011499 and val_mae min 0.043332 in FFT regression history; FFT scatter/depth figures exist | /mnt/c/Users/Administrator/Desktop/Hal/results/FFT_EfficientNet/output/fft_regression/array_03/logs/training_history_fft.pkl and image analysis folders | usable as training/validation evidence; final thesis metric needs split verification |
| sample weights + asymmetric loss is failed attempt | weighted history exists; result.txt says 结果很差; remote commit message says effect was very poor | /mnt/c/Users/Administrator/Desktop/Hal/results/temp_result/1D+percentage_Label/样本权重+非对称损失/result.txt; remote branch commit b0825f3e from remote verification docs | appendix/failed_attempt only |
| CSI+CNN, CSI+SE-ResNet, dual-channel and pre-correction metrics are not directly extractable numerically | performance summary PNGs found but no text/PKL numeric metrics | /mnt/c/Users/Administrator/Desktop/Hal/results/CSI+CNN; CSI+SE-ResNet; 双通道学习; 预校正 | image_only_needs_manual_read |
<!-- METRIC_EXTRACTION_AUTO_END -->
<!-- TENSORBOARD_SPLIT_AUDIT_START -->
## TensorBoard And Split Risk Audit Update (2026-07-06)

| question | answer | evidence | risk |
| --- | --- | --- | --- |
| EXP-008 是否能作为主结果直接写入论文 | 可以作为方法主线/探索性主结果写入，但当前不能作为最终泛化性能直接写入。 | origin/percentage_label+FFT train/dataset split code; split_forensic_audit.csv | random_split_depth_leakage_risk |
| EXP-008 是否必须补 depth-blocked split | 是。若论文要报告最终性能或模型泛化，必须补 depth-blocked/depth-heldout split 或找到已存在证据。 | No depth-blocked code or split artifact found | high |
| EXP-007 是否能作为 fallback 主线 | 可以作为 fallback/exploratory 主线和严重度误差分析，但仍不能声称 depth-heldout 泛化。 | EXP-007 result.txt severity MAE/RMSE plus random split audit | random_split_depth_leakage_risk |
| EXP-006 baseline 的结论是否安全 | 安全范围是“随机验证下 CWT 与二分类标签存在可学习关系”；不安全范围是 depth-heldout/generalization。 | val_auc 0.95361; origin/test_relativity split code | random_split_depth_leakage_risk |
| 哪些指标只能写成 exploratory | EXP-008/007/006/002/003/014 的 validation metrics；EXP-004/005 train losses；所有 image-only metrics。 | unified_metrics_table.csv; split_forensic_audit.csv | requires caveat |
| 哪些实验必须标记 split_unknown | CSI+CNN, CSI+SE-ResNet, dual-channel metadata fusion, pre-correction, Grad-CAM image-only routes仍缺直接 split 代码/日志对应。 | result directories + no numeric/log split artifacts | split_unknown |
<!-- TENSORBOARD_SPLIT_AUDIT_END -->

<!-- EXP008_DEPTH_HELDOUT_TRAINING_START -->
## EXP-008 Depth-Heldout Training Update (2026-07-07)

| claim | evidence | status |
| --- | --- | --- |
| EXP-008 no longer relies only on random-split validation metrics. | `remote_exp008_depth_blocked_train/test_metrics.json`; `remote_exp008_depth_blocked_train/leakage_audit.json` | supported for single-well depth-heldout |
| FFT severity + EfficientNet remains learnable under depth-heldout split. | test MAE `0.079025`, RMSE `0.277019`, R2 `0.106634`, Pearson `0.351950`, Spearman `0.480519` | supported cautiously |
| EXP-008 can support final thesis method feasibility, but not multi-well generalization. | split is array_03 contiguous depth-heldout only; no second well evidence | supported with scope caveat |
| EXP-007 fallback is not immediately required. | EXP-008 train_v001 completed successfully and produced heldout metrics | recommendation |
<!-- EXP008_DEPTH_HELDOUT_TRAINING_END -->

<!-- EXP008_RESULT_AUDIT_P3_START -->
## EXP-008 Result Audit P3 Claim Update (2026-07-07)

| claim | evidence | status |
| --- | --- | --- |
| EXP-008 beats train-mean/train-median baselines on depth-heldout test by MAE/RMSE/R2. | model MAE `0.079025`, train-mean `0.142358`, train-median `0.119024`; model RMSE `0.277019`, train-median `0.278963` | supported |
| EXP-008 does not beat zero baseline by overall MAE. | model MAE `0.079025` vs zero MAE `0.069751` | limitation; must state |
| EXP-008 beats zero baseline by RMSE/R2 and has positive test correlation. | model RMSE `0.277019` vs zero `0.301272`; model R2 `0.106634` vs zero `-0.056638`; Spearman `0.480519` | supported with metric caveat |
| Main errors are low-frequency coefficients and high-severity underprediction. | worst coefficients k=0-4; top error samples true integrated severity around 2.4-4.6 with predictions near zero | supported |
<!-- EXP008_RESULT_AUDIT_P3_END -->
