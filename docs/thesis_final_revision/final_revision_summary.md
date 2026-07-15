# Final Revision Summary

## Scope

This revision incorporates the project participant's final confirmations into the thesis, closes previously unresolved data facts, audits the Grad-CAM evidence chain, checks terminology and conclusion boundaries, and completes a clean Windows TeX Live build. It does not retrain a model, alter an experiment metric, modify source data, or write to the external results directory.

## Confirmed facts incorporated

- XSI is described as 13 axial receiver rings with 8 circumferential receivers per ring, while the study uses the eight channels of ring 3. The waveform and CWT parameters are fixed at 1024 original samples, 100 kHz, 10 microseconds, approximately 10.24 ms, the first 400 samples, a study-selected 1-30 kHz CWT band, and a `150 x 400 x 8` input.
- Values `-8388608` and `8388607` are described as sparse signed 24-bit acquisition saturation or digital clipping. The model input contains 156 endpoint samples in 75 records. Historical preprocessing did not repair, mask or clip them; their aggregate effect on the reported experiments can be neglected but they remain documented as a data-quality phenomenon.
- CAST acoustic impedance `Zc` is uniformly reported in MRayl. The original matrix has `180 x 24750` values, the target interval has 9884 depth records, and 180 samples at exactly 2 degrees cover the full circumference.
- The `2.5 MRayl` threshold is identified as Halliburton's project-specific CAST interpretation convention. The resulting severity is a derived representation, not an absolute defect width, volume or directly measured ground truth.
- All 14 negative CAST values lie outside the 2732-4132 ft interval. None entered the target grid, severity calculation, FFT labels, or training, validation and test sets.
- The data source is named in Chinese as 美国哈里伯顿公司 and in English as Halliburton, without promotional wording.

## Azimuthal mismatch and method boundary

The thesis now connects the near-vertical interval, the approximately `0.4705 degree` median inclination, frequent Relative Bearing jumps, and weak high-side stability to the absence of a unique XSI-CAST absolute-zero transformation. The FFT magnitude target is presented as retaining circumferential spatial-frequency composition while discarding phase and absolute azimuth. The method therefore reduces the effect of circular azimuthal shifts but cannot recover the absolute direction of channeling.

## Grad-CAM decision

The candidate figures are classified as `partially_verified`. Their plotting code, `top_conv` layer and scalar all-output regression target can be traced, but a complete per-figure chain to a depth-blocked split, sample depth and exact checkpoint cannot. No Grad-CAM figure is admitted to the thesis body. The method description and interpretation limits remain, and missing final-model interpretability evidence is recorded as nonblocking future work.

## Quality and build result

- Closed or internal TODO wording was removed from the thesis body; only author-supplied cover fields and acknowledgements remain.
- All 48 bibliography entries are cited, with zero undefined citations and zero undefined references.
- The final build succeeded at 57 A4 pages with zero overfull boxes, two accepted underfull boxes and one inherited xeCJK template warning.
- Six verified experiment figures remain in Chapter 4; no Grad-CAM figure is used.
- The generated PDF and compilation cache were cleaned after inspection and are not committed.

## Remaining human actions

The author must complete the cover metadata and acknowledgements. A provenance-complete Grad-CAM rerun or manifest would be needed before a final-model heatmap could be added. Multi-well validation and larger-sample evaluation remain future research rather than requirements for the factual correctness of this revision.
