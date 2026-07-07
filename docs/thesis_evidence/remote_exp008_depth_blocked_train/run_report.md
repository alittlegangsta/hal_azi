# EXP-008 Depth-Heldout Training Run Report

Generated: `2026-07-07T03:14:08+00:00`
Output dir: `output/thesis_depth_blocked/exp008/train_v001`
Run mode: `full`

## Split

- split_dir: `output/thesis_depth_blocked/exp008/split_v001`
- split_type: `depth_heldout_exp008`
- leakage claim: uses precomputed `train.tfrecord`, `val.tfrecord`, and `test.tfrecord`; no random split or validation_split.

## Training Summary

- epochs_requested: `80`
- epochs_completed: `12`
- batch_size: `8`
- learning_rate: `0.0001`
- best_val_loss: `0.01521982904523611`
- final_train_loss: `0.0204180795699358`
- final_val_loss: `0.015793336555361748`
- pretrained_status: `{'requested': True, 'loaded': True, 'policy': 'optional', 'weights_path': '/home/xiaoj/.keras/models/efficientnetv2-b0_notop.h5', 'error': None}`

## Heldout Metrics

| metric | validation | test |
| --- | ---: | ---: |
| dc_mae | 0.07076457142829895 | 0.1427205502986908 |
| dc_rmse | 0.3159470856189728 | 0.4676450788974762 |
| keras_loss | 0.01521982904523611 | 0.032993514090776443 |
| keras_mae | 0.04281659796833992 | 0.0790247768163681 |
| low_freq_0_4_mae | 0.06682131439447403 | 0.1314113438129425 |
| low_freq_0_4_rmse | 0.29153478145599365 | 0.43136832118034363 |
| num_label_values | 873600 | 888300 |
| num_samples | 416 | 423 |
| overall_mae | 0.04281659027890655 | 0.079024767050357 |
| overall_mse | 0.03503130481137632 | 0.07673956571576947 |
| overall_pearson | 0.20057479194367284 | 0.35194989306253516 |
| overall_r2 | 0.010882152947672585 | 0.1066341559863514 |
| overall_rmse | 0.18716651626660236 | 0.27701907103260864 |
| overall_spearman | 0.26613733903597164 | 0.4805185128927375 |

## Thesis Use

These metrics are EXP-008 single-well depth-heldout evidence. They should not be described as multi-well generalization.
