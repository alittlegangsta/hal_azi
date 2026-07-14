# Unresolved items after confirmed-data correction

Audit date: 2026-07-14

## P0 - affects correctness

No open P0 item remains from the CAST threshold or negative-value questions.

Closed P0 items:

1. **Threshold identity:** 2.5 is confirmed as the project convention used by the project data provider for CAST image interpretation. It is not a threshold proposed by this thesis and is not presented as a universal industry standard.
2. **Negative-value impact:** all 14 negative `Zc` values are invalid physical values, but all lie outside 2732--4132 ft. Historical code did not mask them; nevertheless, none entered the target grid, severity map, one-dimensional label, FFT label, or training records used by the thesis.

## P1 - affects the core argument

Open P1 items:

1. Candidate Grad-CAM figures remain tied to the random-sample pilot output, but their exact checkpoint, target output, convolutional layer, and run cannot be uniquely recovered. They remain excluded from the thesis body.
2. Permission to publish the cooperation company's name remains undocumented. The thesis uses “项目数据提供方” or “某国际油田服务公司”.

Closed or bounded P1 items:

1. **Azimuth supervision boundary:** the available data cannot establish a unique transform between XSI and CAST absolute azimuth zeros. The thesis now treats this as the confirmed problem boundary, uses no absolute-position supervision, and makes no absolute-direction recovery claim.
2. **XSI endpoint values:** exact signed-24-bit endpoints are present. Historical preprocessing applies the high-pass filter without endpoint repair. The existence, counts, affected channels, and handling are now documented; only the hardware stage that produced the clipping remains unknown.
3. **CWT band identity:** 1--30 kHz is confirmed as the study's CWT analysis band, not the instrument's original bandwidth.

## P2 - affects completeness

1. `Zc` should be checked specifically for the unit MRayl, but neither the MAT fields nor the inspected report text explicitly states the unit. The thesis retains a TODO and does not present MRayl as confirmed.
2. The exact physical reference of `RelBearing` and the CAST azimuth-zero direction remain undocumented. This no longer blocks the method because the thesis explicitly avoids absolute azimuth supervision.
3. The acquisition documentation does not locate the XSI endpoint clipping at the ADC versus a later digital stage.
4. Detailed XSI analog frequency-response limits remain unavailable; the thesis states only the verified project sampling rate and study CWT band.

## P3 - presentation

1. Author, adviser, school, major, dates, and classification fields in `main.tex` remain author-supplied TODOs.
2. Final figure numbering may change after manual typography review.
