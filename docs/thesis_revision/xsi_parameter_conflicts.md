# XSI parameter conflict audit

## Confirmed facts

- The official tool paper reports 13 axial receiver rings with eight circumferential receivers per ring, spaced by 45 degrees, for 104 receivers total (Sun et al., 2016, OTC-26688-MS, p. 3).
- `XSILMR03.mat` contains `WaveRng03SideA` through `SideH`, each shaped 1024 by 7108. Receiver group 03 therefore contributes eight model channels; it is not a 104-channel model input.
- `Tad=10.0` and the processing code's 100 kHz sampling rate are consistent with a 10 microsecond sampling interval.
- A raw waveform contains 1024 samples (10.24 ms). The preprocessing selects the first 400 samples (4.00 ms).
- The model CWT uses complex Morlet `cmor1.5-1.0`, 150 log-spaced scales, and a 1--30 kHz analysis band, producing a 150 by 400 by 8 tensor.
- The processed depth grid is 0.1 ft; raw XSI group-03 depth spacing has a median near 0.493 ft in the target interval.

## Conflicts and conservative resolutions

| Item | Evidence conflict | Thesis treatment |
|---|---|---|
| 400 samples | Preprocessing window versus raw record length | State that 400 samples are selected from 1024; never describe 400 as an instrument limit |
| 1--30 kHz | Model analysis band versus instrument response | State that it is the study band; the raw instrument frequency response remains unresolved |
| 8 by 13 | Tool geometry versus model tensor | Explain 13 axial rings and eight side receivers; state that this study uses ring 03 only |
| Sampling time | `Tad` unit is not self-described in MAT | Use 10 microseconds only because code and 100 kHz configuration corroborate it |
| Saturation | Raw int32 values include extrema near +/-8388608 | Mark handling as unresolved; no explicit clipping/saturation repair was found |
| Missing/invalid samples | No NaNs in group-03 waveforms, but duplicate/reordered depths exist | Describe depth deduplication/resampling conservatively; do not invent waveform repair |

% TODO-equivalent P1: confirm the tool's original analog frequency response and the acquisition-system definition of Tad.
