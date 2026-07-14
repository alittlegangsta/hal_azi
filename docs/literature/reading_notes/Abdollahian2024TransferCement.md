# Bibliographic identity
Abdollahian et al. (2024), *Transfer learning for acoustic cement bond evaluation*, DOI 10.1016/j.geoen.2024.212960.

# Metadata verification
Crossref, OpenAlex, and Zotero item 8V9EMQKN agree on title, authors, year, and venue.

# Full-text status
fulltext_verified from the Zotero PDF attachment text cache.

# Research question
Whether CWT images of VDL waveforms and transferred image classifiers can automate cement-isolation classification.

# Data and sample
Two wells are reported, including the public Volve 15/9-F-12 interval and a Chinese field well. The categorical task differs from the regression target in this thesis.

# Instrument or method
VDL waveforms are transformed with CWT and classified with pretrained Xception, VGG16, MobileNetV2, and ResNet50 backbones.

# Main findings
Time-frequency images contain cement-related information in the authors' datasets, and transfer learning can classify their cement-isolation categories.

# Limitations
The accuracy is specific to the paper's labels and splits. It does not validate CAST-derived FFT regression, absolute azimuth recovery, or cross-well generalization here.

# Relevance to this thesis
Supports CWT-based acoustic cement evaluation and cautious use of image backbones.

# Claims supported
CWT can convert VDL waveforms to time-frequency images for data-driven cement evaluation.

# Claims not supported
The paper does not prove that Grad-CAM heat regions are physical causes or that this thesis attains the reported classification accuracy.

# Evidence page numbers
PDF pp. 1--4 for problem, data, and workflow; results and discussion sections thereafter.

# BibTeX status
admitted as Abdollahian2024TransferCement.

# Human-review status
Needs final author review before submission.
