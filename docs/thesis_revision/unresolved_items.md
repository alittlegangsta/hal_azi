# Unresolved items

## P0 - affects correctness

1. The CAST threshold 2.5 is traceable to project code and the final report, but no formal instrument manual or publication establishes it as a universal physical boundary. The thesis must call it a project data convention and request author confirmation.
2. `CAST.mat` contains 14 negative finite Zc values. The located label code applies `max(0, 2.5-Zc)` without an explicit invalid-value rule; this could inflate severity for those cells. Confirm whether upstream processing already defined them as valid or invalid before final submission.

## P1 - affects the core argument

1. The exact physical reference of `RelBearing` and the CAST azimuth-zero convention are absent. The thesis therefore cannot provide an absolute conversion formula.
2. XSI raw waveforms contain values at the apparent 24-bit extrema, but no explicit saturation-repair step was found. Confirm the acquisition encoding and preprocessing policy.
3. The official XSI paper supports receiver geometry but not the project's raw frequency-response limits. The 1--30 kHz range is documented only as this study's CWT band.
4. The candidate Grad-CAM figures are tied to the random-sample pilot output, but their exact checkpoint, target output, and run cannot be uniquely recovered. They are excluded from the thesis body.
5. The project documents confirm an external cooperative project but do not document permission to name Halliburton. The thesis uses “某国际油田服务公司” pending authorization.

## P2 - affects completeness

1. CAST Zc units are not encoded in the MAT file or confirmed in the report. Do not print a unit until source documentation is supplied.
2. CAST's 180 rows imply 2-degree bins only if they cover a full 360-degree scan; no explicit azimuth vector is stored.
3. Detailed acquisition hardware settings beyond the verified receiver geometry and project sampling convention remain incomplete.

## P3 - presentation

1. Author, adviser, school, major, dates, and classification fields in `main.tex` remain author-supplied TODOs.
2. Final figure numbering may change after manual typography review.
