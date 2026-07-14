# Evidence-based thesis revision summary

## Literature

- Discovery records: 409 across Crossref, OpenAlex, Semantic Scholar, and arXiv
- DOI/title deduplicated candidates: 373
- Abstracts verified: 41
- Full texts verified from identity-checked Zotero attachments: 9
- Metadata-only records: 5
- References admitted and cited: 48
- Distinct references used in Chapter 1: 40

All admitted entries have unique semantic keys and recorded metadata sources. The bibliography contains no `nocite{*}`. Claims are limited to the verification level recorded in `citation_audit.csv`.

## Evidence and method

The revision separates the 13-by-8 published XSI receiver geometry from the eight-channel receiver-ring subset used by the model. It distinguishes the 1024-point raw record from the 400-point, 4.00 ms preprocessing window and describes 1--30 kHz only as the CWT analysis band.

CAST is described as a 180-by-24750 acoustic-impedance matrix. The 2.5 threshold is identified as a project convention, and CAST-derived labels are not treated as ground truth. Negative cells, physical units, and exact azimuth-bin spacing remain unresolved.

Relative Bearing has frequent circular jumps in the near-vertical target interval, whereas Inclination measures tilt magnitude and cannot be used as an azimuth correction. The available evidence does not define a unique XSI-to-CAST absolute azimuth transform. This establishes the physical motivation for a circular-shift-invariant Fourier-magnitude label while preserving the explicit loss of absolute direction.

## Manuscript

Chapters 1--6, the abstracts, the experiment appendix, and the bibliography were revised. The three experiments are named:

1. 基于 CWT 特征的窜槽二分类可学习性验证实验
2. 基于一维窜槽占比标签的回归对照实验
3. 基于方位旋转不变 FFT 幅值标签的窜槽结构特征反演实验

Six verified final depth-interval-held-out result figures were inserted. No Grad-CAM figure was inserted because the available candidate cannot be uniquely tied to the required model, checkpoint, target output, convolutional layer, and test split.

Chapter 4 now reports validation and test metrics, training and early stopping, all simple baselines, coefficient-wise error, severity distribution, high-value underestimation, and the evidence boundary between random-sample pilot results and single-well depth-interval-held-out results.

## Validation

Windows TeX Live 2026 compiled the manuscript successfully under the isolated `revision_validation` job name. The result has 55 pages, zero undefined citations or references, zero missing figures, and zero overfull boxes. Build products are excluded from Git and cleaned after validation.

Open P0/P1 issues are retained in `unresolved_items.md` and `revision_issues.csv`; no missing fact was guessed.
