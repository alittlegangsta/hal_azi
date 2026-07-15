# Grad-CAM final decision

Decision date: 2026-07-15

## Decision

Final level: `partially_verified`

No existing Grad-CAM figure is admitted to the thesis body or appendix. Chapter 3 retains the method principle and interpretation boundary; Chapter 4 does not claim final-model Grad-CAM validation.

## Reconstructed chain

| Link | Finding | Verification |
|---|---|---|
| figure | Representative `combo_sample_1160.png` in `FFT_EfficientNet_1/output/fft_regression/array_03/results/fft_model_analysis_plots` has SHA256 `24a7f4a23e0aaf7e5c7888f8fc33650169aa8ed20a3e1fdfa02dff21f3e735f9` | exact file |
| generating script | `origin/percentage_label+FFT:src/interpretation/run_analysis_regressor.py` produces the exact `combo_sample_<index>.png` layout and output directory | strong code match |
| checkpoint path | the script loads `models/best_fft_regressor_model.h5`; the currently archived model has SHA256 `24401ee315197f746acb65e5d6ac8db80e671d26fbfcca3aaeb96e11dc762f97` | path and current file verified; generation-time file state not verified |
| split protocol | branch training shuffles the full dataset and takes/skips batches for train/validation; it is not the later single-well depth-blocked run | route verified |
| sample | filename records only an analysis-array index; the script randomly selects category members and does not persist sample depth or split in the filename | incomplete |
| regression target | Grad-CAM scalar is `tf.reduce_sum(final_preds)`, the sum of all `70 x 30` regression outputs, not a named depth/coefficient target | verified from code |
| convolution layer | script requests `top_conv`; the archived HDF5 model configuration contains `top_conv` | verified from code and model metadata |
| PPT | slide 10 contains candidate qualitative heatmaps but no checkpoint, run, depth, split, or scalar-target manifest | incomplete |

## Why the figure is not `verified_random_split_exploratory`

The output directory contains accumulated figures from repeated analyses, while the script writes a fixed checkpoint filename and does not write a run manifest. Multiple TensorBoard event files exist, and an individual image cannot be tied to the model-file state that existed when that image was generated. The analysis also predicts the complete TFRecord rather than a persisted held-out sample list. Consequently, the route is identifiable as an early random-sample experiment, but the per-figure provenance chain is incomplete.

## Thesis consequence

- No Grad-CAM image is inserted.
- No time-frequency hot-region claim is made from the candidate images.
- No candidate heatmap is attributed to the 423-sample depth-blocked test set.
- Grad-CAM is described only as a gradient-weighted response for a specified prediction target, not as model weights or causal physical evidence.
