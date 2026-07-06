# Experiments Missing Metrics

Generated: 2026-07-06. Missing means no numeric metric was extracted from existing text/CSV/JSON/PKL/log evidence. Images were not OCRed.

| experiment_id | experiment_name | status | available_metric_names | missing_or_manual_work |
| --- | --- | --- | --- | --- |
| EXP-001 | Baseline FFT magnitude image translation | image_only_metrics | image_only_needs_manual_read | failed/baseline evidence should not be mainline unless thesis frames it as negative result; split_unknown; leakage_risk_unknown |
| EXP-002 | FFT log-label image translation | numeric_or_text_metrics_available | epoch_count, loss, loss_final, mae, mae_final, val_loss, val_loss_final, val_mae, val_mae_final | split_unknown; leakage_risk_unknown |
| EXP-003 | FFT high-frequency weighted loss | numeric_or_text_metrics_available | epoch_count, loss, loss_final, mae, mae_final, val_loss, val_loss_final, val_mae, val_mae_final | split_unknown; leakage_risk_unknown |
| EXP-004 | GAN + severity transform / 2D label | numeric_or_text_metrics_available | discriminator_loss, epoch_time, generator_loss | failed/baseline evidence should not be mainline unless thesis frames it as negative result; split_unknown; leakage_risk_unknown |
| EXP-005 | Two-channel binary label and focal-loss/overfit test | numeric_or_text_metrics_available | accuracy, accuracy_final, auc, auc_final, epoch_count, loss, loss_final, val_accuracy, val_accuracy_final, val_auc, val_auc_final, val_loss, val_loss_final | failed/baseline evidence should not be mainline unless thesis frames it as negative result; split_unknown; leakage_risk_unknown |
| EXP-006 | CNN binary classification: CWT-label relationship test | numeric_or_text_metrics_available | accuracy, accuracy_approx, accuracy_final, auc, auc_final, best_epoch_val_auc, early_stop_epoch, epoch_count, loss, loss_final, val_accuracy, val_accuracy_approx, val_accuracy_final, val_auc, val_auc_final, val_loss, val_loss_final | split_unknown; leakage_risk_unknown |
| EXP-007 | 1D percentage label profile regression | numeric_or_text_metrics_available | MAE, R2, RMSE, accuracy, accuracy_final, auc, auc_final, epoch_count, loss, loss_final, mae, mae_final, sample_count, val_accuracy, val_accuracy_final, val_auc, val_auc_final, val_loss, val_loss_final, val_mae | split_unknown; leakage_risk_unknown |
| EXP-008 | EfficientNet FFT severity regression | numeric_or_text_metrics_available | epoch_count, loss, loss_final, mae, mae_final, val_loss, val_loss_final, val_mae, val_mae_final | split_unknown; leakage_risk_unknown |
| EXP-009 | CSI + CNN visual/Grad-CAM analysis | image_only_metrics | image_only_needs_manual_read | numeric metrics not found in text/pkl; use images only after manual read/redraw; split_unknown; leakage_risk_unknown |
| EXP-010 | CSI + SE-ResNet azimuth matching / classification | image_only_metrics | image_only_needs_manual_read | numeric metrics not found in text/pkl; use images only after manual read/redraw; split_unknown; leakage_risk_unknown |
| EXP-011 | Dual-channel metadata fusion | image_only_metrics | image_only_needs_manual_read | numeric metrics not found in text/pkl; use images only after manual read/redraw; split_unknown; leakage_risk_unknown |
| EXP-012 | Eccentricity pre-correction | image_only_metrics | image_only_needs_manual_read | numeric metrics not found in text/pkl; use images only after manual read/redraw; split_unknown; leakage_risk_unknown |
| EXP-013 | Grad-CAM interpretability across routes | image_only_metrics | image_only_needs_manual_read | numeric metrics not found in text/pkl; use images only after manual read/redraw; split_unknown; leakage_risk_unknown |
| EXP-014 | 1D percentage label + sample weights + asymmetric loss failed attempt | numeric_or_text_metrics_available | accuracy, accuracy_final, auc, auc_final, epoch_count, loss, loss_final, mae, mae_final, qualitative_result, val_accuracy, val_accuracy_final, val_auc, val_auc_final, val_loss, val_loss_final, val_mae, val_mae_final, val_weighted_mae, val_weighted_mae_final | failed/baseline evidence should not be mainline unless thesis frames it as negative result; split_unknown; leakage_risk_unknown |

## Required Manual Follow-Up

- Manually read or redraw image-only performance summaries for CSI+CNN, CSI+SE-ResNet, dual-channel metadata fusion, eccentricity pre-correction, and Grad-CAM if they will be cited quantitatively.
- Verify split construction and adjacent-depth leakage risk before presenting any validation metric as final thesis performance.
- Use remote `hall` or another environment with TensorBoard only if event scalar values are required; this extraction did not parse event files.
