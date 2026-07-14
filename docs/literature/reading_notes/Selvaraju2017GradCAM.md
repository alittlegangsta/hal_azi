# Bibliographic identity
Selvaraju et al. (2017), *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*, DOI 10.1109/ICCV.2017.74.

# Metadata verification
Crossref, OpenAlex, and Zotero item C9BQ5ZQ5 agree.

# Full-text status
fulltext_verified from the Zotero PDF attachment text cache.

# Research question
Generation of target-specific visual explanations for convolutional-network predictions.

# Data and sample
Image classification, captioning, and visual-question-answering experiments.

# Instrument or method
Gradients of a selected target output flowing into a convolutional layer are pooled as feature-map weights, and the weighted map is rectified to form a coarse localization.

# Main findings
The method creates target-dependent response maps without retraining the model.

# Limitations
The response is coarse, target- and layer-dependent, and is not a causal explanation or a direct view of model parameters.

# Relevance to this thesis
Defines the interpretation boundary for regression-output Grad-CAM on CWT inputs.

# Claims supported
A heatmap indicates gradient-weighted response regions for a specified prediction.

# Claims not supported
A heatmap does not prove a true physical mechanism or that a highlighted region causes channeling.

# Evidence page numbers
ICCV printed pp. 618--621 for method; pp. 622--626 for experiments and conclusion.

# BibTeX status
admitted as Selvaraju2017GradCAM.

# Human-review status
Needs final author review before submission.
