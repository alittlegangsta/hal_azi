# Missing Evidence

| priority | item | reason | minimal_action |
| --- | --- | --- | --- |
| A 必须补 | 防止数据泄漏的 split 说明或 depth-blocked 验证 | 现有代码多处使用 VALIDATION_SPLIT + dataset take/skip，未证明同一井深邻近窗口不会跨 train/val。 | 不训练；先从 TFRecord/索引/脚本恢复样本深度顺序，写出 split 说明。如需补，运行轻量 depth-blocked 评估脚本。 |
| A 必须补 | baseline 对比 | 论文主线需要明确比 baseline 好；现有 baseline 多为定性失败说明。 | 从现有 results 和日志提取可比指标；缺失则标为 missing，不重训。 |
| A 必须补 | 消融实验 | log、frequency weighted、severity/FFT、1D percentage 已有路线，但指标不统一。 | 整理已有图和日志，统一表述为路线对比；只在必要时做轻量评估，不做训练。 |
| A 必须补 | Grad-CAM 批量统计 | 现有 Grad-CAM 多为样本图/定性 memo，论文需要稳定性证据。 | 优先使用现有 comprehensive_gradcam_statistics.png 和 attention plots；若缺统计脚本，写读取已保存热图/图像的轻量汇总。 |
| A 必须补 | 严重度分组误差分析 | memo 提到严重窜槽预测偏保守，但缺统一误差表。 | 从现有 prediction/result artifacts 中寻找已保存数组/CSV；缺失则列 missing。 |
| A 必须补 | 旧实验顺序验证 | memo、结果目录、Git 分支能形成大体顺序，但 CSI+SE-ResNet、双通道、预校正缺本地分支映射。 | 检查远程服务器 `/home/xiaoj/hal_azi` 和 GitHub 分支，不训练。 |
| B 可选补 | 更多模型/超参数/多井/高级信号处理 | 可提高论文完整度，但不是最短毕业路线。 | 仅作为后续工作，不进入当前主线。 |
| C 不建议补 | 大规模重训、新版本复杂弱标签、STC/APES、多目标人工审核 | 会扩大风险并偏离旧项目证据重建目标。 | 明确不做，除非导师要求并单独批准。 |
<!-- METRIC_EXTRACTION_AUTO_START -->
## Metric Extraction Missing Evidence Update (2026-07-06)

- Must still verify split construction and adjacent-depth leakage risk before claiming thesis final performance.
- Must manually read/redraw image-only metric figures if CSI+CNN, CSI+SE-ResNet, dual-channel, pre-correction, Grad-CAM, scatter plots, or depth curves are cited quantitatively.
- TensorBoard event scalars were not parsed locally because TensorBoard is unavailable; PKL histories were sufficient for many training/validation summaries.
- Duplicate `training_history*.pkl` hashes mean repeated folders should not be counted as independent runs without commit/run provenance verification.
- Failed attempts retained: GAN/two-channel binary FFT label, dual-channel metadata fusion, eccentricity pre-correction, and sample weights + asymmetric loss. Do not use as mainline recommendations.

### Experiments Still Missing Direct Numeric Metrics

| experiment_id | experiment_name | status | missing_or_manual_work |
| --- | --- | --- | --- |
| EXP-001 | Baseline FFT magnitude image translation | image_only_metrics | failed/baseline evidence should not be mainline unless thesis frames it as negative result; split_unknown; leakage_risk_unknown |
| EXP-009 | CSI + CNN visual/Grad-CAM analysis | image_only_metrics | numeric metrics not found in text/pkl; use images only after manual read/redraw; split_unknown; leakage_risk_unknown |
| EXP-010 | CSI + SE-ResNet azimuth matching / classification | image_only_metrics | numeric metrics not found in text/pkl; use images only after manual read/redraw; split_unknown; leakage_risk_unknown |
| EXP-011 | Dual-channel metadata fusion | image_only_metrics | numeric metrics not found in text/pkl; use images only after manual read/redraw; split_unknown; leakage_risk_unknown |
| EXP-012 | Eccentricity pre-correction | image_only_metrics | numeric metrics not found in text/pkl; use images only after manual read/redraw; split_unknown; leakage_risk_unknown |
| EXP-013 | Grad-CAM interpretability across routes | image_only_metrics | numeric metrics not found in text/pkl; use images only after manual read/redraw; split_unknown; leakage_risk_unknown |
<!-- METRIC_EXTRACTION_AUTO_END -->
<!-- TENSORBOARD_SPLIT_AUDIT_START -->
## TensorBoard And Split Audit Missing Evidence Update (2026-07-06)

- TensorBoard metrics remain missing locally: `needs_tensorboard_dependency`; no scalar values were guessed.
- EXP-008, EXP-007, EXP-006, EXP-002, EXP-003, and EXP-014 now have confirmed `random_split_depth_leakage_risk`, not merely unknown split.
- Minimum required supplement before final performance claims: construct deterministic depth-blocked/depth-heldout split and rerun only necessary evaluation/training if explicitly approved. If training is not approved, present old metrics as exploratory and list this as limitation.
- Still missing: train/val/test depth ranges, split index artifacts, TensorBoard scalar extraction, and split mapping for image-only CSI/dual-channel/pre-correction routes.
<!-- TENSORBOARD_SPLIT_AUDIT_END -->

<!-- EXP008_DEPTH_HELDOUT_TRAINING_START -->
## EXP-008 Depth-Heldout Training Missing Evidence Update (2026-07-07)

- Completed for EXP-008: deterministic depth-heldout split, smoke training, full train_v001, validation/test metrics, training curve, prediction scatter, depth curve, and split audit archive.
- No longer missing for EXP-008: train/val/test depth ranges and heldout test metrics.
- Still missing for final broad claims: multi-well validation, EXP-007 depth-heldout fallback comparison, baseline depth-heldout comparison, and TensorBoard historical scalar extraction.
- Current safest thesis claim: single-well depth-heldout method feasibility for EXP-008, with test MAE `0.079025` and Spearman `0.480519`.
<!-- EXP008_DEPTH_HELDOUT_TRAINING_END -->
