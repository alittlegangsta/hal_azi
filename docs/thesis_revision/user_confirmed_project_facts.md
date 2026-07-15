# User-confirmed project facts

Final confirmation date: 2026-07-15

These facts were supplied by a project participant for the final thesis correction. They supersede earlier provisional wording. No raw data, processed data, model output, or experiment metric was changed.

## XSI

- The data contain 13 axial receiver rings with eight circumferential receivers per ring, for 104 receiver channels.
- The thesis uses only the eight circumferential receivers of ring 03.
- Each raw channel waveform contains 1024 samples.
- The project sampling rate is 100 kHz, the interval is 10 microseconds, and a 1024-point record spans about 10.24 ms.
- The thesis selects the first 400 samples, corresponding to about 4 ms.
- The 1--30 kHz range is the thesis CWT analysis band, not the instrument's original bandwidth.
- The CWT input tensor is `150 x 400 x 8`.
- Raw waveforms contain the signed-24-bit endpoints `-8388608` and `8388607`, representing sparse acquisition saturation or digital clipping.
- The model input scope contains 156 endpoint samples in 75 records. Historical preprocessing did not repair, mask, or clip them.
- Project review determined that their aggregate effect on the reported model results can be neglected. They remain documented as an input-quality feature; no retraining or ablation is claimed.

## CAST

- `Zc` is the acoustic-impedance image and its unit is MRayl.
- `Zc` has shape `180 x 24750`; the main 2732--4132 ft interval contains 9,884 depth records.
- Every depth contains 180 samples at exact 2-degree intervals, covering the complete 360-degree circumference.
- `2.5 MRayl` is the project-specific interpretation threshold used by Halliburton for this CAST dataset. It was not proposed or optimized by the thesis and is not a universal industry standard.
- Fourteen negative `Zc` values in the complete data have no reasonable physical interpretation. Historical code did not mask, replace, or clip them.
- All 14 values are outside 2732--4132 ft. The target interpolation grid contains zero negative values, so none entered severity, FFT labels, or model training, validation, or testing.

## Main interval and azimuth relation

- Most thesis experiments use 2732--4132 ft.
- Median Inclination is about 0.4705 degrees, so the interval is near vertical overall.
- Inclination describes tilt magnitude and is not an azimuth-rotation correction.
- Relative Bearing changes frequently and sharply in the main interval; the high-side reference is weakly stable under near-vertical conditions.
- The available data cannot establish a unique transform between the XSI receiver azimuth zero and the CAST image azimuth zero.
- The thesis therefore does not use absolute CAST defect position as supervision.
- The 180-point CAST sequence is treated as periodic and transformed along azimuth. Fourier magnitude reduces sensitivity to a cyclic zero-point shift.
- The label preserves the spatial-frequency composition of circumferential structure while discarding phase and absolute azimuth. It cannot recover the absolute channeling direction.

## Project source

- The data originate from a collaborative project with Halliburton in the United States.
- Disclosure of the company name in the thesis is permitted.
- Chinese text uses “美国哈里伯顿公司” or “哈里伯顿公司”; the English abstract uses “Halliburton”.
- Promotional rankings or market-position claims are not used.
