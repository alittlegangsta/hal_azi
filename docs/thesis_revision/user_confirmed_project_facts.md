# User-confirmed project facts

Confirmation date: 2026-07-14

This file records project facts supplied for thesis correction. Where possible, the facts were cross-checked against raw MAT arrays, historical code, or existing experiment records. It does not alter raw data or frozen metrics.

## XSI

- The tool data contain 13 axial receiver rings, each with eight circumferential receivers.
- The thesis uses only the eight circumferential receivers of ring 03.
- Each raw channel waveform contains 1024 samples.
- The project sampling rate is 100 kHz.
- The first 400 samples selected for the thesis correspond to about 4 ms.
- The 1--30 kHz interval is the CWT analysis band used by the thesis, not the instrument's original bandwidth.
- The CWT input tensor is `150 x 400 x 8`.

Raw-data cross-check: `XSILMR03.mat` contains SideA--SideH matrices of shape `1024 x 7108`; the historical configuration sets `SAMPLING_RATE=1e5`, `TIME_STEPS=400`, `N_SCALES=150`, and `N_CHANNELS=8`.

## CAST

- `Zc` is the acoustic-impedance image.
- `Zc` has shape `180 x 24750`.
- The main 2732--4132 ft interval contains 9,884 CAST depth records.
- The 180 azimuth samples are 2 degrees apart and cover the full 360-degree circumference.
- The value 2.5 is the project convention used by the project data provider in this project's CAST image interpretation. It is not proposed by the thesis and is not a universal industry standard.
- Until publication permission is confirmed, the manuscript uses “项目数据提供方” or “某国际油田服务公司”.
- Fourteen negative `Zc` values lack reasonable physical meaning and are treated as invalid values for interpretation.

Unit status: the unit should be checked specifically as MRayl. The current MAT fields and inspected PPT text do not explicitly label it, so MRayl remains unconfirmed and the manuscript retains a TODO.

## Main interval and azimuth relation

- Most thesis experiments use 2732--4132 ft.
- Inclination is small in this interval, with median about 0.4705 degrees; the interval is near vertical overall.
- Relative Bearing changes frequently and sharply in this interval.
- Inclination describes tilt magnitude and is not a rotation-correction angle.
- The available data cannot establish a unique transform between XSI and CAST absolute azimuth zeros.
- The thesis therefore does not use absolute CAST azimuth position as supervision.
- The 180-point CAST circumferential sequence is transformed along azimuth with the FFT, and its magnitude is used to reduce the effect of a circular azimuth shift.
- The label preserves the frequency composition of circumferential structure while discarding phase and absolute azimuth position.
- The representation cannot recover the absolute channeling direction.

## Scope of this correction

- No model was trained and no experiment was rerun.
- No raw, processed, or result data were changed.
- No frozen metric was changed.
- No cooperation-company name was de-anonymized.
