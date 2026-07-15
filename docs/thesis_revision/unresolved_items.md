# Unresolved items after final fact correction

Audit date: 2026-07-15

## P0 - affects correctness

No P0 item remains open.

The CAST threshold identity, CAST unit, CAST negative-value impact, XSI endpoint identity, XSI endpoint impact, circumferential sampling interval, cooperating-company disclosure, and absolute-azimuth recovery boundary are all resolved from project-participant confirmation plus the existing data/code audit.

## P1 - affects the core argument

No unresolved factual item blocks the thesis's main method or reported metrics.

One interpretability item remains open:

1. **Grad-CAM figure provenance:** the candidate figures are only partially verified. Their generating code can be identified, including the `top_conv` layer and the sum of all regression outputs used as the scalar target, but an individual figure cannot be uniquely bound to the checkpoint state, training run, sample depth, and split used when it was generated. The figures remain excluded from the thesis body. This does not affect the prediction metrics or the main conclusions.

## P2 - affects completeness but not the current conclusions

1. The formal external reference definitions of Relative Bearing and the CAST image zero are unavailable. The thesis does not infer an absolute conversion and does not use absolute-position supervision.
2. The precise acquisition stage at which the signed-24-bit endpoint clipping occurred is not identified. The thesis reports the observable endpoint clipping without assigning it to a specific hardware stage.
3. The XSI instrument's original analog transmit/receive bandwidth is not required by the analysis and is not claimed. The thesis reports only the confirmed 100 kHz sampling rate and the study's 1--30 kHz CWT band.
4. Multi-well generalization and larger-sample validation remain future research rather than missing evidence for the bounded single-well conclusions.

## P3 - author-supplied presentation items

1. Author, adviser, school, major, student number, classification fields, and dates in `main.tex` still require author input.
2. The acknowledgement remains author supplied.
3. Experiment one's complete optimizer and batch configuration is not stated because the available record is incomplete; its reported role remains limited to random-sample learnability validation.
