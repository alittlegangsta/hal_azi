# Figures

This directory is reserved for manually redrawn thesis figures.

Current LaTeX draft uses placeholder boxes only. Do not add original result images, checkpoints, TFRecords, or large binary artifacts here.

Recommended workflow:

1. Redraw figures in Origin, PPT, AI, or Figma using `docs/thesis_plot_data/`.
2. Export publication figures as PDF or PNG.
3. Put final files in this directory.
4. Replace each `\placeholderfigure{...}{...}{...}` block with `\includegraphics`.

All figure captions must preserve the evidence scope: random-split exploratory evidence is not depth-heldout performance, and EXP-008/EXP-007 are single-well depth-heldout only.
