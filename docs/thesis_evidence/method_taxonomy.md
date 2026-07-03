# Method Taxonomy

## 1. Research Problem

- XSI acoustic waveform and CAST Zc image have azimuth mismatch in vertical wells. Evidence: `memo PPT section in results/temp_result/改进memo.md; explicitly thesis framing`. Status: `supported_by_memo`.
- Relative Bearing is unreliable for direct point-to-point azimuth supervision. Evidence: `memo PPT section and user task framing`. Status: `supported_by_memo_plus_inference`.
- Direct azimuth-resolved point supervision is not the strongest thesis route with current evidence. Evidence: `FFT baseline/log/weighted-loss failures in memo and result plots`. Status: `supported_by_memo`.

## 2. Data Construction

- XSI waveform -> CWT scalogram/time-frequency image (repo config.py and src/cwt_transformation/main_transform_translation.py)
- CAST Zc slices from ground_truth_db HDF5 (repo config.py and create_tfrecords.py)
- Severity transform max(0, 2.5 - Zc) appears in memo and origin/percentage_label+FFT create_tfrecords.py
- Depth window/path label uses MAX_PATH_DEPTH_POINTS=70 and target depth range in config.py

## 3. Label Routes

- 1D percentage label: mean(Zc < 2.5) over azimuth for each depth, from origin/1D+percentage_Label create_tfrecords.py.
- FFT magnitude label: apply FFT along azimuth axis and take magnitude, from current create_tfrecords.py and memo.
- Log transform: log(1 + magnitude), from memo and current code.
- Phase discarded for rotation/azimuth invariance: memo PPT section; use as rationale, not as proven performance claim.

## 4. Model Routes

- SE-ResNet route has result artifacts but code mapping is missing locally.
- EfficientNetV2B0 route is present in origin/1D+percentage_Label model.py.
- Dual-channel metadata fusion has result artifacts but code/metrics need verification.
- Correction/pre-correction model has result artifacts but code/metrics need verification.

## 5. Interpretability Route

- Grad-CAM plots and memo identify sensitive time-frequency regions, especially high-frequency bands around 22-30 kHz and sub-ms to ~1.3 ms windows.
- Use Grad-CAM as qualitative interpretability unless batch statistics are regenerated from existing artifacts or verified.

## 6. Failed Routes

- Azimuth matching mismatch and FFT image translation collapse: supported by memo and baseline/log/weighted-loss result artifacts.
- Eccentricity correction poor: explicitly marked inference from user task framing plus result directory existence; needs code/metric verification.
- Dual-channel failed: explicitly marked inference from user task framing plus result directory existence; needs code/metric verification.
- High-severity prediction poor for 1D percentage regression: supported by memo.

## 7. Recommended Thesis Mainline

- Use severity + FFT magnitude/log label with CWT + EfficientNet as the intended angle-mismatch-aware thesis mainline, but mark quantitative performance as needs_verification until metrics/split are recovered.
- Use 1D percentage EfficientNet as the strongest confirmed learnability/severity evidence and a practical fallback main result.
- Use artifact masking and Grad-CAM interpretability as explanation/supporting evidence only when tied to specific result files.

## Experiment Families

| experiment_id | method_family | experiment_name | thesis_use | evidence_strength |
| --- | --- | --- | --- | --- |
| EXP-001 | baseline | Baseline FFT magnitude image translation | baseline | strong |
| EXP-002 | FFT log label | FFT log-label image translation | ablation | strong |
| EXP-003 | FFT severity label | FFT high-frequency weighted loss | failed_attempt | strong |
| EXP-004 | other | GAN + severity transform / 2D label | failed_attempt | medium |
| EXP-005 | other | Two-channel binary label and focal-loss/overfit test | failed_attempt | medium |
| EXP-006 | baseline | CNN binary classification: CWT-label relationship test | background | strong |
| EXP-007 | 1D percentage label | 1D percentage label profile regression | main_result | strong |
| EXP-008 | FFT severity label | EfficientNet FFT severity regression | main_result | medium |
| EXP-009 | Grad-CAM interpretability | CSI + CNN visual/Grad-CAM analysis | figure/background | medium |
| EXP-010 | SE-ResNet azimuth matching | CSI + SE-ResNet azimuth matching / classification | baseline | medium |
| EXP-011 | dual-channel metadata fusion | Dual-channel metadata fusion | failed_attempt | weak |
| EXP-012 | eccentricity correction | Eccentricity pre-correction | failed_attempt | weak |
| EXP-013 | Grad-CAM interpretability | Grad-CAM interpretability across routes | main_result | strong |
