# Experiment Inventory

One row is one reconstructed experiment or method version. Unknown fields are intentionally left as `unknown` or `needs_verification` rather than inferred as facts.

- Experiment rows: `13`

| experiment_id | experiment_name | likely_stage_order | method_family | model | target_label | metrics_available | main_result_summary | thesis_use | evidence_strength |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-001 | Baseline FFT magnitude image translation | 1 | baseline | Attention U-Net / A2INet | FFT magnitude of CAST Zc slice | loss/val_loss/mae in logs; memo qualitative result | Memo states Grad-CAM was scattered and prediction collapsed toward an overall average with vertical stripe patterns. | baseline | strong |
| EXP-002 | FFT log-label image translation | 2 | FFT log label | Attention U-Net / A2INet | FFT magnitude label | loss/val_loss/mae; qualitative Grad-CAM and prediction plots | Memo says Grad-CAM became more concentrated around 0.5-0.7 ms and 25-30 kHz, but predictions remained poor with horizontal stripe patterns. | ablation | strong |
| EXP-003 | FFT high-frequency weighted loss | 3 | FFT severity label | Attention U-Net / A2INet | FFT magnitude/log label | SSIM/PSNR mentioned in memo/code; tensorboard logs and plots available | Memo states Grad-CAM was more concentrated near 0.75-0.85 ms and 25-28 kHz, but predictions still fit low FFT coefficients. | failed_attempt | strong |
| EXP-004 | GAN + severity transform / 2D label | 4 | other | GAN / generator-discriminator route | severity map max(0, 2.5 - Zc), then FFT/log variants | result.txt contains epoch losses; plots/checkpoints available | Memo states model collapse; result.txt shows generator loss stayed around 307-325 through epoch 100. | failed_attempt | medium |
| EXP-005 | Two-channel binary label and focal-loss/overfit test | 5 | other | GAN/U-Net route, exact code needs verification | two-channel binary mask: channeling vs good bonding | qualitative memo and result images | Memo states model collapse. | failed_attempt | medium |
| EXP-006 | CNN binary classification: CWT-label relationship test | 6 | baseline | CNN classifier | binary label: channeling exists if >1% pixels have Zc < 2.5 | AUC=0.95361, accuracy/val_accuracy≈85%, loss/val_loss from memo/result.txt | Validated that CWT contains learnable information about channeling existence. | background | strong |
| EXP-007 | 1D percentage label profile regression | 7 | 1D percentage label | EfficientNetV2B0 with regression head | depth-wise channeling percentage profile | training history images; memo qualitative conclusion | Memo states attention consistently focused on 0.5-1.0 ms and 23-28 kHz, with stronger attention for severe channeling. | main_result | strong |
| EXP-008 | EfficientNet FFT severity regression | 8 | FFT severity label | EfficientNetV2B0 or branch-specific regressor | severity transform and FFT magnitude/log coefficients | label-generation and training-history figures; no confirmed final numeric metric in text inventory | Evidence exists for label generation/training history; final quantitative outcome is needs_verification. | main_result | medium |
| EXP-009 | CSI + CNN visual/Grad-CAM analysis | unknown | Grad-CAM interpretability | CNN or CSI-specific model, code mapping unknown | channeling class/distribution, exact label unknown | distribution/Grad-CAM/statistics figures | Directory contains Grad-CAM statistics, signal examples, CSI/channeling distributions, and filtering comparisons. | figure/background | medium |
| EXP-010 | CSI + SE-ResNet azimuth matching / classification | unknown | SE-ResNet azimuth matching | SE-ResNet | class/quality labels visible in candidate filenames; exact label definition unknown | performance_summary_plots, attention analysis, candidate comparisons | Result directory contains multi-array outputs, model checkpoints, performance summaries, attention analysis, and candidate comparisons. | baseline | medium |
| EXP-011 | Dual-channel metadata fusion | unknown | dual-channel metadata fusion | dual-input model, exact architecture unknown | unknown, likely channeling/severity class or profile | performance_summary_plots, metadata-vs-label validation, CWT-vs-label validation | Result directory contains validation plots for inclination, CWT-vs-label, and metadata-vs-label. | failed_attempt | weak |
| EXP-012 | Eccentricity pre-correction | unknown | eccentricity correction | correction model, exact architecture unknown | unknown | performance_summary_plots and attention plots | Result directory contains a model checkpoint and attention/performance summary plots. | failed_attempt | weak |
| EXP-013 | Grad-CAM interpretability across routes | cross-cutting | Grad-CAM interpretability | CNN, A2INet, EfficientNet/SE-ResNet depending on route | route-specific | Grad-CAM plots/statistics and memo attention-frequency claims | Memo and result plots repeatedly locate sensitive regions in high-frequency CWT bands around roughly 22-30 kHz and 0.5-1.3 ms, depending on route. | main_result | strong |

Full path-level evidence is in `experiment_inventory.csv` and `experiment_inventory.json`.

## Remote Verification Update

Date: `2026-07-06`

Remote target `cement-server:/home/xiaoj/hal_azi` could not be read because SSH authentication failed with `Permission denied (publickey,password)`. Therefore, no remote code, branch, reflog, uncommitted-file, or `hall` environment evidence was added.

Impact on experiment mapping:

| Experiment | Remote verification status | Impact |
| --- | --- | --- |
| EXP-001 baseline | needs_verification | Exact baseline branch/order remains unresolved. |
| EXP-010 CSI + SE-ResNet | needs_verification | Still lacks conclusive code/branch mapping. |
| EXP-011 dual-channel metadata fusion | needs_verification | Still lacks conclusive code/branch mapping. |
| EXP-012 eccentricity pre-correction | needs_verification | Still lacks conclusive code/branch mapping. |
| all other locally mapped experiments | no new remote evidence | Existing local evidence strength unchanged. |

Migration decision: `manual_review_required`. Do not rsync or migrate remote code until SSH authentication is restored and the read-only command list is collected.

## Remote Verification Retry

Date: `2026-07-06`

Required SSH probe failed:

```text
ssh -o BatchMode=yes cement-server 'echo remote_ssh_ok'
-> Permission denied (publickey,password).
```

No remote experiment-code evidence was collected in the retry. The following mappings remain `needs_verification`: EXP-001 exact baseline branch/order, EXP-010 CSI+SE-ResNet code provenance, EXP-011 dual-channel metadata fusion code provenance, and EXP-012 eccentricity pre-correction code provenance.

Migration decision remains: `manual_review_required`.

## Remote Verification Retry 2

Date: `2026-07-06`

Required SSH probe failed again:

```text
ssh -o BatchMode=yes cement-server 'echo remote_ssh_ok'
-> Permission denied (publickey,password).
```

No remote experiment-code evidence was collected in this retry. The following mappings remain `needs_verification`: EXP-001 exact baseline branch/order, EXP-010 CSI+SE-ResNet code provenance, EXP-011 dual-channel metadata fusion code provenance, and EXP-012 eccentricity pre-correction code provenance.

Migration decision remains: `manual_review_required`.

## Remote Verification Retry 3

Date: `2026-07-06`

Required SSH probe succeeded:

```text
ssh -o BatchMode=yes cement-server 'echo remote_ssh_ok'
-> remote_ssh_ok
```

Remote evidence update:

| Experiment / method | Remote evidence | Updated status |
| --- | --- | --- |
| EXP-007 1D percentage label | remote `1D+percentage_Label` confirms existing local mapping | unchanged strong |
| supplemental failed route | remote-only branch `1D+percentage_Label+Sample_weights+loss` at `b0825f3...`; modifies `src/modeling/train.py` and `src/interpretation/run_analysis_regressor.py`; commit says sample weights + asymmetric loss were very poor | add as failed attempt / appendix evidence |
| EXP-008 FFT severity regression | remote `percentage_label+FFT` confirms current checkout and chronology | mapping strengthened |
| EXP-005 two-channel binary label | remote `GaN` branch confirms dual-channel binary FFT label code | mapping strengthened |
| EXP-010 CSI + SE-ResNet | only weak early `SE-ResNet` visualization-script evidence; no conclusive code/result mapping | still needs_verification |
| EXP-011 dual-channel metadata fusion | no conclusive metadata-fusion code found across remote branches | still needs_verification |
| EXP-012 eccentricity pre-correction | no conclusive eccentricity/pre-correction code found across remote branches | still needs_verification |

Supplemental remote-only experiment candidate:

| experiment_id | experiment_name | method_family | model | target_label | main_result_summary | thesis_use | evidence_strength |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-014 | 1D percentage label + sample weighting + asymmetric loss | 1D percentage label | EfficientNetV2B0 regressor | depth-wise channeling percentage profile | Remote commit message states sample weighting + asymmetric loss produced very poor results; code adds `asymmetric_huber_loss`, dynamic `sample_weight`, weighted MAE, scatter/boxplot/representative Grad-CAM analysis. | failed_attempt / appendix | strong for code provenance, weak for numeric metric |

Updated migration decision: `fetch_missing_branches_only`. Fetching the missing branch would complete local Git evidence for EXP-014; do not rsync raw/output files.
<!-- METRIC_EXTRACTION_AUTO_START -->
## Metric Extraction Update (2026-07-06)

- Unified metric rows generated: `256` in `docs/thesis_evidence/unified_metrics_table.csv`.
- All extracted rows preserve `source_path`; all rows remain `split_unknown` and `leakage_risk_unknown` unless future verification proves otherwise.
- `EXP-008` remains the mainline candidate because FFT regression history and analysis figures exist, but thesis-grade final performance still needs split/leakage verification.
- `EXP-007` remains fallback mainline because `result.txt` contains direct severity-group MAE/RMSE evidence, with high severity error worst.
- `EXP-014` is a supplemental failed-attempt entry for `样本权重+非对称损失`; remote branch `1D+percentage_Label+Sample_weights+loss` commit `b0825f3e` plus local `result.txt` support appendix/negative-result use.

### Metric Availability By Experiment

| experiment_id | status | available_metric_names | missing_or_manual_work |
| --- | --- | --- | --- |
| EXP-001 | image_only_metrics | image_only_needs_manual_read | failed/baseline evidence should not be mainline unless thesis frames it as negative result; split_unknown; leakage_risk_unknown |
| EXP-002 | numeric_or_text_metrics_available | epoch_count, loss, loss_final, mae, mae_final, val_loss, val_loss_final, val_mae, val_mae_final | split_unknown; leakage_risk_unknown |
| EXP-003 | numeric_or_text_metrics_available | epoch_count, loss, loss_final, mae, mae_final, val_loss, val_loss_final, val_mae, val_mae_final | split_unknown; leakage_risk_unknown |
| EXP-004 | numeric_or_text_metrics_available | discriminator_loss, epoch_time, generator_loss | failed/baseline evidence should not be mainline unless thesis frames it as negative result; split_unknown; leakage_risk_unknown |
| EXP-005 | numeric_or_text_metrics_available | accuracy, accuracy_final, auc, auc_final, epoch_count, loss, loss_final, val_accuracy, val_accuracy_final, val_auc, val_auc_final, val_loss, val_loss_final | failed/baseline evidence should not be mainline unless thesis frames it as negative result; split_unknown; leakage_risk_unknown |
| EXP-006 | numeric_or_text_metrics_available | accuracy, accuracy_approx, accuracy_final, auc, auc_final, best_epoch_val_auc, early_stop_epoch, epoch_count, loss, loss_final, val_accuracy, val_accuracy_approx, val_accuracy_final, val_auc, val_auc_final, val_loss, val_loss_final | split_unknown; leakage_risk_unknown |
| EXP-007 | numeric_or_text_metrics_available | MAE, R2, RMSE, accuracy, accuracy_final, auc, auc_final, epoch_count, loss, loss_final, mae, mae_final, sample_count, val_accuracy, val_accuracy_final, val_auc, val_auc_final, val_loss, val_loss_final, val_mae | split_unknown; leakage_risk_unknown |
| EXP-008 | numeric_or_text_metrics_available | epoch_count, loss, loss_final, mae, mae_final, val_loss, val_loss_final, val_mae, val_mae_final | split_unknown; leakage_risk_unknown |
| EXP-009 | image_only_metrics | image_only_needs_manual_read | numeric metrics not found in text/pkl; use images only after manual read/redraw; split_unknown; leakage_risk_unknown |
| EXP-010 | image_only_metrics | image_only_needs_manual_read | numeric metrics not found in text/pkl; use images only after manual read/redraw; split_unknown; leakage_risk_unknown |
| EXP-011 | image_only_metrics | image_only_needs_manual_read | numeric metrics not found in text/pkl; use images only after manual read/redraw; split_unknown; leakage_risk_unknown |
| EXP-012 | image_only_metrics | image_only_needs_manual_read | numeric metrics not found in text/pkl; use images only after manual read/redraw; split_unknown; leakage_risk_unknown |
| EXP-013 | image_only_metrics | image_only_needs_manual_read | numeric metrics not found in text/pkl; use images only after manual read/redraw; split_unknown; leakage_risk_unknown |
| EXP-014 | numeric_or_text_metrics_available | accuracy, accuracy_final, auc, auc_final, epoch_count, loss, loss_final, mae, mae_final, qualitative_result, val_accuracy, val_accuracy_final, val_auc, val_auc_final, val_loss, val_loss_final, val_mae, val_mae_final, val_weighted_mae, val_weighted_mae_final | failed/baseline evidence should not be mainline unless thesis frames it as negative result; split_unknown; leakage_risk_unknown |
<!-- METRIC_EXTRACTION_AUTO_END -->

