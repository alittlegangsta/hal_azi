# EXP-008 Depth-Heldout Baseline Comparison

Baselines were computed without training. Train-based baselines use labels from `split_v001/train.tfrecord`; test labels and model predictions use train_v001 artifacts.

| predictor | MAE | RMSE | R2 | Pearson | Spearman | DC MAE | low k=0-5 MAE | mid k=6-14 MAE | high k=15-29 MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| model | 0.079025 | 0.277019 | 0.106634 | 0.351950 | 0.480519 | 0.142721 | 0.128785 | 0.093920 | 0.050184 |
| zero | 0.069751 | 0.301272 | -0.056638 |  |  | 0.127209 | 0.114296 | 0.082648 | 0.044195 |
| train_mean | 0.142358 | 0.316129 | -0.163426 | 0.418723 | 0.503310 | 0.301496 | 0.254077 | 0.164485 | 0.084395 |
| train_median | 0.119024 | 0.278963 | 0.094049 | 0.419976 | 0.503310 | 0.236799 | 0.205667 | 0.140932 | 0.071222 |
| val_mean_oracle_analysis | 0.081989 | 0.276887 | 0.107488 | 0.420344 | 0.503445 | 0.148936 | 0.134225 | 0.097263 | 0.051931 |

Use `train_mean` and `train_median` as thesis-safe baselines. `val_mean_oracle_analysis` is an analysis-only comparator and should not be presented as a deployable predictor.
