# Risk Register

| risk_id | risk | impact | mitigation |
| --- | --- | --- | --- |
| R-001 | 把 memo 定性结论写成未证实的定量结论 | high | 所有结论引用 memo/result/code/git；没有指标则写 unknown/needs_verification。 |
| R-002 | train/validation 深度泄漏 | high | 补 depth-blocked split 说明或承认为限制。 |
| R-003 | 结果目录与代码分支映射错误 | medium | 用 Git commit、result modified_time、服务器目录核对。 |
| R-004 | FFT 主线性能不足 | medium | 把 1D percentage EfficientNet 作为强证据主结果，FFT 作为角度不匹配方法探索/局限。 |
| R-005 | 为了补证据触发大规模重训 | medium | 当前阶段只做整理和轻量验证；训练需单独审批。 |
<!-- TENSORBOARD_SPLIT_AUDIT_START -->
## TensorBoard And Split Audit Update (2026-07-06)

| risk_id | risk | impact | mitigation |
| --- | --- | --- | --- |
| R-006 | TensorBoard scalar metrics not parsed locally | medium | Run `scripts/thesis_parse_tensorboard_events.py` in an environment with TensorBoard; until then use PKL/text metrics only. |
| R-007 | EXP-008 mainline metrics come from random shuffled validation | high | Do not state final generalization; add/recover depth-blocked split before final thesis claims. |
| R-008 | take/skip after shuffle may produce unstable validation subset across independent iterations | high | Document as random split depth leakage risk; replace with deterministic split indices for any supplement. |
| R-009 | No depth/index artifacts found for historical runs | medium | Search remote data artifacts or regenerate only lightweight split index from existing TFRecord/depth metadata if approved; no training needed. |
<!-- TENSORBOARD_SPLIT_AUDIT_END -->

<!-- EXP008_DEPTH_HELDOUT_TRAINING_START -->
## EXP-008 Depth-Heldout Training Risk Update (2026-07-07)

| risk_id | risk | impact | mitigation |
| --- | --- | --- | --- |
| R-007 | EXP-008 mainline metrics come from random shuffled validation | reduced | New train_v001 uses explicit train/val/test TFRecords and `depth_heldout_split_confirmed`; keep historical metrics as exploratory only. |
| R-010 | EXP-008 depth-heldout result is single-well only | medium | State single-well depth-heldout scope; do not claim multi-well generalization. |
| R-011 | Heldout test weaker than historical random split | medium | Present as stricter validation evidence; report both with split labels and explain leakage risk. |
| R-012 | Remote run used an uncommitted copied training script | low | Commit local script and docs; remote `run_config.json` records `status_short`; no push or raw/result overwrite occurred. |
<!-- EXP008_DEPTH_HELDOUT_TRAINING_END -->

<!-- EXP008_RESULT_AUDIT_P3_START -->
## EXP-008 Result Audit P3 Risk Update (2026-07-07)

| risk_id | risk | impact | mitigation |
| --- | --- | --- | --- |
| R-013 | Zero baseline has lower overall MAE than EXP-008 model | high | Do not claim uniformly better performance; report RMSE/R2/correlation separately and explain label sparsity. |
| R-014 | High-frequency coefficients look best by MAE because target magnitude is small | medium | Avoid calling high-frequency coefficients physically most learnable; say they have lowest absolute error. |
| R-015 | High-severity samples are underpredicted | medium | Use top-error and depth-curve figures as limitation evidence; consider EXP-007 fallback if stronger result needed. |
<!-- EXP008_RESULT_AUDIT_P3_END -->

<!-- EXP007_DEPTH_HELDOUT_TRAINING_START -->
## EXP-007 Depth-Heldout Fallback Risk Update (2026-07-07)

| risk_id | risk | impact | mitigation |
| --- | --- | --- | --- |
| R-016 | EXP-007 fallback still does not beat zero baseline by MAE | high | Use EXP-007 as fallback/limitation comparison only; do not present as clean performance rescue. |
| R-017 | EXP-007 best R2 remains slightly negative | medium | Use RMSE/R2 improvement over zero and train-based baselines cautiously; keep EXP-008 as main method route. |
| R-018 | Remote v001/v002 artifacts were not fully copied locally | medium | Local docs cite remote paths and captured stdout; copy remote small files later when SSH/SCP approval is available. |
| R-019 | EXP-007 severity-group conclusions are incomplete locally | medium | Do not state severity-group conclusions until `/home/xiaoj/hal_azi/output/thesis_depth_blocked/exp007/train_v002/severity_group_metrics.csv` is copied/read. |
| R-020 | v003 could not be run after approval usage limit rejection | low | Record as not run; do not infer potential v003 performance. |
<!-- EXP007_DEPTH_HELDOUT_TRAINING_END -->

<!-- FINAL_EVIDENCE_FREEZE_START -->
## Final Evidence Freeze Risk Update (2026-07-08)

| risk_id | risk | impact | mitigation |
| --- | --- | --- | --- |
| R-021 | Mixing random-split exploratory metrics with depth-heldout metrics in one final ranking | high | Use `final_thesis_metrics_table.*`; label EXP-006 as random-split background and EXP-007/008 as single-well depth-heldout. |
| R-022 | Overstating EXP-008 as fully superior despite zero-baseline MAE caveat | high | Always report MAE/RMSE/R2 together and cite `thesis_overclaim_blacklist.md`. |
| R-023 | Treating EXP-007 as a successful rescue result | medium | Present EXP-007 as fallback/limitation comparison; state R2 is slightly negative and zero MAE is lower. |
| R-024 | Writing multi-well or deployment claims | high | Use `thesis_safe_claims.md`; explicitly state single-well `array_03` scope. |
| R-025 | Depending on remote EXP-007 figures that were not copied locally | medium | Use remote paths as traceable sources now; copy/redraw small figures in a later non-training writing task. |
| R-026 | Continuing experiments instead of writing | medium | Freeze evidence; treat additional training as optional future work only. |
<!-- FINAL_EVIDENCE_FREEZE_END -->
