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
- Still missing for final broad claims: multi-well validation, baseline depth-heldout model comparison, and TensorBoard historical scalar extraction. EXP-007 depth-heldout fallback has now been run but is weak.
- Current safest thesis claim: single-well depth-heldout method feasibility for EXP-008, with test MAE `0.079025` and Spearman `0.480519`.
<!-- EXP008_DEPTH_HELDOUT_TRAINING_END -->

<!-- EXP008_RESULT_AUDIT_P3_START -->
## EXP-008 Result Audit P3 Missing Evidence Update (2026-07-07)

- Completed: artifact consistency audit, zero/train-mean/train-median baseline comparison, per-coefficient/per-depth error structure, top-error sample summary, and thesis-quality redraws.
- Still missing for a stronger performance chapter: EXP-007 depth-heldout fallback/comparison, baseline depth-heldout model comparison, and multi-well validation.
- New limitation to carry into thesis: EXP-008 does not beat zero baseline on overall MAE, although it improves RMSE/R2 and train-based baselines.
<!-- EXP008_RESULT_AUDIT_P3_END -->

<!-- EXP007_DEPTH_HELDOUT_TRAINING_START -->
## EXP-007 Depth-Heldout Fallback Missing Evidence Update (2026-07-07)

- Completed: EXP-007 record-depth mapping inspection, deterministic depth-heldout split, smoke check, smoke training, full training attempts v001/v002, baseline comparison, and EXP-007 vs EXP-008 comparison.
- Not completed: train_v003 was planned but not run because SSH/SCP escalation was rejected by the execution environment usage limit.
- Still missing locally: byte-for-byte copied remote small artifacts for EXP-007 v001/v002, especially `severity_group_metrics.csv/json`, `profile_index_error.csv`, figures, and full `run_report.md`.
- Current EXP-007 result is not a clean fallback: best run train_v002 beats zero by RMSE/R2 and train-based baselines by MAE/RMSE/R2, but zero still has lower MAE.
- Multi-well validation remains missing for both EXP-008 and EXP-007.
<!-- EXP007_DEPTH_HELDOUT_TRAINING_END -->

<!-- FINAL_EVIDENCE_FREEZE_START -->
## Final Evidence Freeze Missing Evidence Update (2026-07-08)

### No Longer Blocking For Thesis Draft

- EXP-008 depth-heldout split, training, metrics, baseline comparison, and error structure are available.
- EXP-007 depth-heldout fallback split, training attempts v001/v002, baseline comparison, and EXP-007 vs EXP-008 comparison are available.
- Final result/discussion writing drafts and safe/unsafe claim lists are available.

### Still Missing But Can Be Written As Limitation

| item | status | thesis handling |
| --- | --- | --- |
| Multi-well validation | missing | State as major limitation and future work. |
| EXP-006 depth-heldout baseline | not run by decision | Not needed for shortest thesis path; keep EXP-006 as random-split baseline/background. |
| EXP-007 remote small figures and severity-group CSV/JSON copied locally | partial/missing locally | Use remote path references and mark severity-group details `needs_verification` until copied/read. |
| TensorBoard historical scalar parsing | optional missing | Not required because PKL/text/depth-heldout metrics now cover thesis tables. |
| Final thesis-style redrawn method/result figures | writing task, not experiment | Redraw from frozen evidence; do not alter original results. |

### Final Recommendation

Stop experiments and enter thesis writing. Additional model runs are not required unless the advisor explicitly requests multi-well validation or a depth-heldout binary baseline.
<!-- FINAL_EVIDENCE_FREEZE_END -->
