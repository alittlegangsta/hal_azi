# Confirmed fact closure

Final confirmation date: 2026-07-15

| Item | Final thesis statement | Status |
|---|---|---|
| XSI endpoint identity | `-8388608` and `8388607` are sparse signed-24-bit acquisition saturation or digital clipping endpoints | resolved |
| XSI endpoint impact | 156 values in 75 input records; no dedicated historical repair; aggregate result impact can be neglected | resolved |
| CAST unit | Acoustic impedance `Zc` is expressed in MRayl | resolved |
| CAST threshold | `2.5 MRayl` is Halliburton's project-specific CAST interpretation convention, not a universal standard | resolved |
| CAST negative values | 14 invalid negatives are outside 2732--4132 ft; none entered severity, FFT labels, or model splits | resolved |
| CAST azimuth sampling | 180 samples at exact 2-degree intervals cover 360 degrees | resolved |
| Company disclosure | Chinese: 美国哈里伯顿公司/哈里伯顿公司; English: Halliburton | resolved |
| Absolute azimuth | FFT magnitude preserves circumferential frequency composition but discards phase and absolute azimuth; absolute channel direction is not recovered | resolved by scope |

The remaining Grad-CAM provenance issue is separate from these data facts and does not change the reported metrics or conclusions.
