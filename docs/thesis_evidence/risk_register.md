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
