# Git Branch Timeline

Local Git information only; no fetch/network access was used.

## Branch Mapping

| branch | latest_date | latest_subject | likely_experiment_group | confidence |
| --- | --- | --- | --- | --- |
| docs/thesis-evidence-inventory | 2025-09-18T16:23:47+08:00 | 标签改为log变换，效果提升显著，但是预测结果不太好，需要补充结构优化 | unknown | weak |
| master | 2025-09-18T16:23:47+08:00 | 标签改为log变换，效果提升显著，但是预测结果不太好，需要补充结构优化 | FFT log label | medium |
| origin/1D+percentage_Label | 2025-11-10T10:36:08+08:00 | 分析脚本新增成像图功能 | 1D percentage label | strong |
| origin/GaN | 2025-09-26T14:27:11+08:00 | GaN+2Dlabel | GAN/severity map | medium |
| origin | 2025-09-18T16:23:47+08:00 | 标签改为log变换，效果提升显著，但是预测结果不太好，需要补充结构优化 | unknown | weak |
| origin/frequency-weighted_loss | 2025-09-23T16:03:42+08:00 | 改进了run_analysis.py，加入指标；并且新增了FFT高频系数惩罚 | FFT high-frequency weighted loss | strong |
| origin/log_scaling | 2025-09-18T16:23:47+08:00 | 标签改为log变换，效果提升显著，但是预测结果不太好，需要补充结构优化 | FFT log label | strong |
| origin/master | 2025-09-18T16:23:47+08:00 | 标签改为log变换，效果提升显著，但是预测结果不太好，需要补充结构优化 | FFT log label | medium |
| origin/percentage_label+FFT | 2025-12-08T17:05:29+08:00 | 删除部分注释 | FFT regression | strong |
| origin/test_relativity | 2025-09-25T17:07:26+08:00 | CNN+分类任务：证实CWT与Label存在对应关系 | CWT-label binary classification | strong |
| needs_remote_server_or_github_fetch | unknown | unknown | baseline | unknown |
| needs_remote_server_or_github_fetch | unknown | unknown | dual-channel | unknown |
| needs_remote_server_or_github_fetch | unknown | unknown | pre-correction | unknown |
| needs_remote_server_or_github_fetch | unknown | unknown | CSI+SE-ResNet | unknown |
| partial: present in repo files and multiple branches, not a standalone branch | unknown | unknown | Grad-CAM | unknown |

## Commit Timeline

| commit | date | refs | subject | changed_files |
| --- | --- | --- | --- | --- |
| 7ba021cfa6ea | 2025-12-08T17:05:29+08:00 | origin/percentage_label+FFT | 删除部分注释 | src/cwt_transformation/main_transform_translation.py; src/data_processing/create_tfrecords.py; src/data_processing/main_preprocess.py; src/interpretation/run_analysis_regressor.py; src/visualization/visualize_data_pipeline.py; src/visualization/visualize_training_history.py |
| a6892bdd45cc | 2025-11-18T08:51:41+08:00 |  | 最终更新 | src/data_processing/create_tfrecords.py; src/interpretation/run_analysis_regressor.py |
| 349f5a221e6c | 2025-11-14T19:56:37+08:00 |  | FFT大功告成，但是还需要整一下深度成像图以及如何展示出FFT的效果 | config.py; main.py; src/data_processing/create_tfrecords.py; src/interpretation/run_analysis_regressor.py; src/modeling/model.py; src/modeling/train.py; src/visualization/visualize_data_pipeline.py; src/visualization/visualize_training_history.py |
| e9739c8c3fc4 | 2025-11-10T10:36:08+08:00 | origin/1D+percentage_Label | 分析脚本新增成像图功能 | src/interpretation/run_analysis_regressor.py |
| 78170ef4e833 | 2025-09-29T10:48:26+08:00 |  | 删掉了无用的s.py | src/interpretation/s.py |
| 3ab1e7365d32 | 2025-09-28T17:19:40+08:00 |  | 一维窜槽百分比剖面图作为标签输入，采用预训练模型EfficientNetV2B0作为模型主干。目前存在窜槽程度越严重，模型预判误差越大模型性能越差，需要改进 | src/data_processing/create_tfrecords.py; src/interpretation/run_analysis_classification.py; src/interpretation/run_analysis_regressor.py; src/interpretation/s.py; src/modeling/model.py; src/modeling/train.py; src/visualization/visualize_data_pipeline.py; src/visualization/visualize_training_history.py |
| c08d695779b3 | 2025-09-26T14:27:11+08:00 | origin/GaN | GaN+2Dlabel | src/modeling/train.py |
| 5e59652a4259 | 2025-09-25T17:07:26+08:00 | origin/test_relativity | CNN+分类任务：证实CWT与Label存在对应关系 | src/data_processing/create_tfrecords.py; src/interpretation/run_analysis_classification.py; src/modeling/dataset.py; src/modeling/model.py; src/modeling/train.py |
| 3b1eeccf3511 | 2025-09-25T10:12:49+08:00 |  | 模型崩溃，以单样本循环多次训练证实目前模型无法学习到任何有帮助的特征。目前标签进行了二值化二通道输入 | src/data_processing/create_tfrecords.py; src/interpretation/debug_data_visualization.py; src/interpretation/generate_analysis_candidates.py; src/interpretation/grad_cam.py; src/interpretation/plot_manual_analysis.py; src/interpretation/run_analysis.py; src/interpretation/run_analysis_classification.py; src/interpretation/s.py |
| 92d670d102da | 2025-09-24T09:57:27+08:00 |  | 目前是模式崩溃 | src/modeling/model.py; src/modeling/train.py |
| 9899283c3517 | 2025-09-23T16:03:42+08:00 | origin/frequency-weighted_loss | 改进了run_analysis.py，加入指标；并且新增了FFT高频系数惩罚 | src/interpretation/run_analysis.py; src/modeling/train.py |
| 1ee68a62763c | 2025-09-18T16:23:47+08:00 | HEAD -> docs/thesis-evidence-inventory, origin/master, origin/log_scaling, origin/HEAD, master | 标签改为log变换，效果提升显著，但是预测结果不太好，需要补充结构优化 | src/data_processing/create_tfrecords.py; src/interpretation/run_analysis.py; src/visualization/visualize_data_pipeline.py |
| fc88e778d94b | 2025-09-15T17:07:30+08:00 |  | grad-cam画图问题解决 | src/interpretation/debug_data_visualization.py; src/interpretation/run_analysis.py |
| 31516b2036ae | 2025-09-12T14:42:35+08:00 |  | 问题依旧 | config.py; src/interpretation/run_analysis.py; src/modeling/model.py |
| 94e350de61ff | 2025-09-12T11:53:23+08:00 |  | 目前画图问题中第二幅图grad-cam热力图似乎有问题，有点窄，其他暂时 没问题，准备完整训练 | config.py; src/interpretation/run_analysis.py; src/modeling/model.py |
| 991d843b657f | 2025-09-11T17:29:02+08:00 |  | 新增可视化数据处理结果，目前grad-cam解释画图结果不行，需修改 | config.py; src/data_processing/main_preprocess.py; src/interpretation/run_analysis.py; src/visualization/__init__.py; src/visualization/visualize_data_pipeline.py; src/visualization/visualize_training_history.py |
| 54b0182df24b | 2025-09-11T16:31:33+08:00 |  | Initial commit: Add project source code and gitignore | .gitignore; config.py; main.py; src/__init__.py; src/cwt_transformation/__init__.py; src/cwt_transformation/main_transform_translation.py; src/data_processing/__init__.py; src/data_processing/create_tfrecords.py |

## Graph Summary

```text
* 7ba021c (origin/percentage_label+FFT) 删除部分注释
* a6892bd 最终更新
* 349f5a2 FFT大功告成，但是还需要整一下深度成像图以及如何展示出FFT的效果
* e9739c8 (origin/1D+percentage_Label) 分析脚本新增成像图功能
* 78170ef 删掉了无用的s.py
* 3ab1e73 一维窜槽百分比剖面图作为标签输入，采用预训练模型EfficientNetV2B0作为模型主干。目前存在窜槽程度越严重，模型预判误差越大模型性能越差，需要改进
* 5e59652 (origin/test_relativity) CNN+分类任务：证实CWT与Label存在对应关系
| * c08d695 (origin/GaN) GaN+2Dlabel
|/  
* 3b1eecc 模型崩溃，以单样本循环多次训练证实目前模型无法学习到任何有帮助的特征。目前标签进行了二值化二通道输入
* 92d670d 目前是模式崩溃
* 9899283 (origin/frequency-weighted_loss) 改进了run_analysis.py，加入指标；并且新增了FFT高频系数惩罚
* 1ee68a6 (HEAD -> docs/thesis-evidence-inventory, origin/master, origin/log_scaling, origin/HEAD, master) 标签改为log变换，效果提升显著，但是预测结果不太好，需要补充结构优化
* fc88e77 grad-cam画图问题解决
* 31516b2 问题依旧
* 94e350d 目前画图问题中第二幅图grad-cam热力图似乎有问题，有点窄，其他暂时 没问题，准备完整训练
* 991d843 新增可视化数据处理结果，目前grad-cam解释画图结果不行，需修改
* 54b0182 Initial commit: Add project source code and gitignore
```

## Branch Diff Stat vs origin/master

### origin/master

```text
(no diff or unavailable)
```

### origin/log_scaling

```text
(no diff or unavailable)
```

### origin/frequency-weighted_loss

```text
src/interpretation/run_analysis.py | 85 +++++++++++++++++++++++++++++---------
 src/modeling/train.py              | 63 +++++++++++++++++++++-------
 2 files changed, 112 insertions(+), 36 deletions(-)
```

### origin/GaN

```text
src/data_processing/create_tfrecords.py            |  87 +++-----
 src/interpretation/debug_data_visualization.py     |  65 ------
 src/interpretation/generate_analysis_candidates.py | 145 --------------
 src/interpretation/grad_cam.py                     |  50 -----
 src/interpretation/plot_manual_analysis.py         | 142 -------------
 src/interpretation/run_analysis.py                 | 161 ---------------
 src/interpretation/run_analysis_classification.py  | 132 ++++++++++++
 src/interpretation/s.py                            |  68 -------
 src/interpretation/visualize_model.py              |  77 -------
 src/modeling/dataset.py                            |  19 +-
 src/modeling/model.py                              |  69 ++++---
 src/modeling/train.py                              | 222 ++++++++++++---------
 src/visualization/visualize_data_pipeline.py       | 180 ++++++++++-------
 src/visualization/visualize_training_history.py    |  54 +++--
 14 files changed, 474 insertions(+), 997 deletions(-)
```

### origin/test_relativity

```text
src/data_processing/create_tfrecords.py            | 124 ++++++--------
 src/interpretation/debug_data_visualization.py     |  65 --------
 src/interpretation/generate_analysis_candidates.py | 145 -----------------
 src/interpretation/grad_cam.py                     |  50 ------
 src/interpretation/plot_manual_analysis.py         | 142 ----------------
 src/interpretation/run_analysis.py                 | 161 ------------------
 src/interpretation/run_analysis_classification.py  | 125 ++++++++++++++
 src/interpretation/s.py                            |  68 --------
 src/interpretation/visualize_model.py              |  77 ---------
 src/modeling/dataset.py                            |  83 ++--------
 src/modeling/model.py                              | 127 +++++----------
 src/modeling/train.py                              | 149 +++++++----------
 src/visualization/visualize_data_pipeline.py       | 180 ++++++++++++---------
 src/visualization/visualize_training_history.py    |  54 +++----
 14 files changed, 416 insertions(+), 1134 deletions(-)
```

### origin/1D+percentage_Label

```text
src/data_processing/create_tfrecords.py            | 104 +++-------
 src/interpretation/debug_data_visualization.py     |  65 ------
 src/interpretation/generate_analysis_candidates.py | 145 -------------
 src/interpretation/grad_cam.py                     |  50 -----
 src/interpretation/plot_manual_analysis.py         | 142 -------------
 src/interpretation/run_analysis.py                 | 161 ---------------
 src/interpretation/run_analysis_regressor.py       | 227 +++++++++++++++++++++
 src/interpretation/s.py                            |  68 ------
 src/interpretation/visualize_model.py              |  77 -------
 src/modeling/dataset.py                            |  83 +-------
 src/modeling/model.py                              | 137 +++++--------
 src/modeling/train.py                              | 182 ++++++++---------
 src/visualization/visualize_data_pipeline.py       | 146 +++++--------
 src/visualization/visualize_training_history.py    |  61 +++---
 14 files changed, 493 insertions(+), 1155 deletions(-)
```

### origin/percentage_label+FFT

```text
config.py                                          |  39 ++-
 main.py                                            |  38 +--
 .../main_transform_translation.py                  |   2 +-
 src/data_processing/create_tfrecords.py            | 136 ++++------
 src/data_processing/main_preprocess.py             |   2 +-
 src/interpretation/debug_data_visualization.py     |  65 -----
 src/interpretation/generate_analysis_candidates.py | 145 ----------
 src/interpretation/grad_cam.py                     |  50 ----
 src/interpretation/plot_manual_analysis.py         | 142 ----------
 src/interpretation/run_analysis.py                 | 161 -----------
 src/interpretation/run_analysis_regressor.py       | 299 +++++++++++++++++++++
 src/interpretation/s.py                            |  68 -----
 src/interpretation/visualize_model.py              |  77 ------
 src/modeling/dataset.py                            |  83 +-----
 src/modeling/model.py                              | 137 ++++------
 src/modeling/train.py                              | 196 +++++++-------
 src/visualization/visualize_data_pipeline.py       | 164 +++++------
 src/visualization/visualize_training_history.py    |  78 +++---
 18 files changed, 663 insertions(+), 1219 deletions(-)
```

## Needs Verification

- `No local branch/commit conclusively maps CSI+SE-ResNet result directory to code.`
- `No local branch/commit conclusively maps 双通道学习 result directory to code.`
- `No local branch/commit conclusively maps 预校正 result directory to code.`
- `Remote server `/home/xiaoj/hal_azi` may contain unpushed branch state and conda environment `hall`.`
