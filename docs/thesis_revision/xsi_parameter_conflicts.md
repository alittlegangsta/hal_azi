# XSI parameter and endpoint audit

Audit date: 2026-07-14

## Confirmed acquisition and study facts

- The tool data contain 13 axial receiver rings with eight circumferential receivers per ring. This study uses only ring 03 and its eight SideA--SideH channels.
- Each ring-03 channel is `1024 x 7108` in `XSILMR03.mat`; each waveform therefore has 1024 raw samples.
- The project sampling rate is 100 kHz. The 1024 samples span 10.24 ms, while the first 400 samples selected by preprocessing span 4.00 ms.
- The 1--30 kHz interval is the study's CWT analysis band, not a claim about the instrument's original analog bandwidth.
- The CWT tensor is `150 x 400 x 8`.

## Fixed endpoint findings

All 58,228,736 values in the eight full ring-03 waveform matrices are finite `int32` values. No NaN or Inf was present. Two fixed extrema repeat exactly:

- `-8388608 = -2^23`: 3,443 occurrences in the full eight-channel matrices;
- `8388607 = 2^23 - 1`: 2,298 occurrences in the full eight-channel matrices.

These are the endpoints of a signed 24-bit range. Their exact repetition is evidence of digital range clipping. The available MAT fields and code do not identify whether clipping occurred in the ADC hardware or in an upstream digital stage, and no evidence identifies either endpoint as a missing-value code.

Within the actual thesis input scope (2732--4132 ft, first 400 samples, 2,846 unique ring-03 depths), 156 endpoint values occur in 75 records:

| Channel | `-8388608` | `8388607` | affected records | zero-based time index | affected depth range (ft) |
|---|---:|---:|---:|---|---|
| SideA | 0 | 0 | 0 | none | none |
| SideB | 0 | 0 | 0 | none | none |
| SideC | 20 | 1 | 14 | 62--78 | 2745.301902--3535.042253 |
| SideD | 83 | 44 | 63 | 60--86 | 2753.009278--3903.545614 |
| SideE | 5 | 0 | 3 | 63--72 | 2836.283808--3739.661091 |
| SideF | 0 | 0 | 0 | none | none |
| SideG | 0 | 0 | 0 | none | none |
| SideH | 1 | 2 | 3 | 71--74 | 4112.396974--4127.279466 |
| Total | 109 | 47 | 75 unique records | 60--86 | 2745.301902--4127.279466 |

## Historical handling

The EXP-008 preprocessing path at `7ba021cfa6eacd148247258ee28b8527dbbc6c92` performs the following operations:

1. reads the first 400 values from each SideA--SideH waveform;
2. applies a fourth-order, 1 kHz zero-phase Butterworth high-pass filter;
3. casts the stacked filtered waveform to `float32`;
4. computes CWT magnitudes.

No explicit saturation mask, endpoint replacement, artificial `clip`, missing-value substitution, `nan_to_num`, or interpolation is applied to the waveform amplitudes. The filtered outputs are finite and no longer equal the exact integer endpoints, but linear filtering is not a reconstruction of the clipped waveform. The thesis therefore records endpoint clipping as an input-quality limitation without claiming that preprocessing repaired it.

## Parameter conflicts and resolutions

| Item | Resolution in thesis |
|---|---|
| 400 samples | Selected preprocessing window from 1024 raw samples; not an instrument limit |
| 1--30 kHz | Study CWT band; not the raw instrument response |
| 13 by 8 geometry | Tool geometry; model uses ring 03 only and has eight channels |
| 100 kHz / 10 microseconds | Project sampling convention; 400 samples equal 4.00 ms |
| Fixed extrema | Confirmed signed-24-bit endpoint clipping evidence; hardware stage remains unknown |
| NaN/Inf | None found in ring-03 raw waveform matrices; no special replacement path was used |

## Remaining clarification

The acquisition-system documentation is still needed to determine whether the fixed endpoints arose at the ADC or in a later digital stage. This distinction does not change the historical preprocessing facts or the frozen experiment metrics.
