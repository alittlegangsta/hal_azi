# UESTC Thesis LaTeX Draft

This directory is a ThesisUESTC-based LaTeX manuscript draft for the HAL/XSI-CAST thesis.

## Template Source

Copied into this directory from `third_party/ThesisUESTC/`:

- `thesis-uestc.cls`
- `thesis-uestc.bst`
- `latexmkrc`
- `pic/logo.pdf`
- `pic/bachelor_font.pdf`

The original template under `third_party/ThesisUESTC/` is not modified.

## Evidence Scope

This manuscript draft follows the frozen evidence:

- EXP-008 is the main method route: severity + FFT magnitude label + XSI-CWT + EfficientNetV2B0.
- EXP-007 is fallback / limitation comparison.
- EXP-006 is random-split exploratory baseline only.
- Failed routes are appendix/discussion material only.

Do not describe the current result as multi-well generalization, industrial deployment, or high-precision production performance.

## Manual TODO

- Fill cover information: author, student number, advisor, school, major, dates, classification number, UDC.
- Add verified bibliography entries to `refs/references.bib`.
- Replace figure placeholders with manually redrawn figures.
- Review English abstract manually.
- Check all claims against `docs/thesis_evidence/thesis_safe_claims.md`.
- Compile in an environment with TeX Live, XeLaTeX, and latexmk; this workspace currently does not provide `latexmk` or `xelatex`.
