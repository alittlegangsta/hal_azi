# Remote Server Code Inventory

Verification date: `2026-07-06`

Remote target:

- SSH alias: `cement-server`
- Expected remote repo: `/home/xiaoj/hal_azi`
- Expected conda environment: `hall`
- Access mode requested: read-only

## Current Access Result

Current remote code inventory status: `verified_after_retry_3`

Latest decision: `fetch_missing_branches_only`

Historical SSH failures are retained below for traceability. The first successful probe was Retry Attempt 3 on `2026-07-06`.

## Initial Access Result

Initial remote code inventory status: `needs_verification`

No remote Git, file tree, or conda environment evidence could be collected in this run because SSH authentication failed before any remote command executed.

Observed local SSH configuration:

```text
host cement-server
user xiaoj
hostname 121.48.161.238
port 7070
identityfile ~/.ssh/id_ed25519_cement_server
identitiesonly yes
```

Authentication checks:

```text
ssh cement-server pwd
-> Permission denied (publickey,password). Also reported missing /usr/bin/ssh-askpass for password prompts.

ssh -o BatchMode=yes cement-server 'cd /home/xiaoj/hal_azi && pwd && git status -sb'
-> Permission denied (publickey,password).
```

Local SSH agent state:

```text
ssh-add -l
-> Could not open a connection to your authentication agent.
```

## Stage 1 Remote Git Information

Not collected. Each requested item remains `needs_verification`:

- `pwd`
- `git status -sb`
- `git remote -v`
- `git branch -a`
- `git tag`
- `git log --all --decorate --oneline --graph --max-count=200`
- `git reflog --date=iso --all --max-count=200`
- Latest commit hash/date/message for each remote branch

## Stage 2 Remote Code Structure

Not collected. The following remote-code mappings remain `needs_verification`:

| Target | Status |
| --- | --- |
| CSI+SE-ResNet | needs_verification |
| EfficientNet | needs_verification |
| FFT regression | needs_verification |
| 1D percentage label | needs_verification |
| dual-channel metadata fusion | needs_verification |
| eccentricity pre-correction | needs_verification |
| Grad-CAM | needs_verification |
| CWT | needs_verification |
| train/evaluate scripts | needs_verification |
| configs | needs_verification |

## Stage 3 Remote Environment

Not collected. The `hall` environment could not be activated or inspected.

| Item | Status |
| --- | --- |
| `which python` | needs_verification |
| `python --version` | needs_verification |
| `conda list` / `pip freeze` | needs_verification |
| TensorFlow version | needs_verification |
| PyTorch version | needs_verification |
| sklearn version | needs_verification |
| scipy version | needs_verification |
| h5py version | needs_verification |

## Conclusion

Decision: `manual_review_required`

Reason: remote SSH authentication failed, so this run cannot confirm whether the server has missing branches, uncommitted code, untracked scripts, or environment details that explain the remaining result directories.

## Retry Attempt

Retry date: `2026-07-06`

Required first command:

```text
ssh -o BatchMode=yes cement-server 'echo remote_ssh_ok'
```

Retry result: `failed`

Observed output:

```text
xiaoj@121.48.161.238: Permission denied (publickey,password).
```

No further remote commands were executed after this failure. Stage 1 Git inspection, Stage 2 code-structure inspection, and Stage 3 `hall` environment inspection remain `needs_verification`.

Updated decision: `manual_review_required`

## Retry Attempt 2

Retry date: `2026-07-06`

Required first command:

```text
ssh -o BatchMode=yes cement-server 'echo remote_ssh_ok'
```

Retry result: `failed`

Observed output:

```text
xiaoj@121.48.161.238: Permission denied (publickey,password).
```

No Stage 1, Stage 2, or Stage 3 remote commands were executed after the failed probe. Remote Git history, remote code structure, uncommitted/untracked remote state, and the `hall` environment remain `needs_verification`.

Updated decision: `manual_review_required`

## Retry Attempt 3

Retry date: `2026-07-06`

Required first command:

```text
ssh -o BatchMode=yes cement-server 'echo remote_ssh_ok'
```

Retry result: `success`

Observed output:

```text
remote_ssh_ok
```

All following evidence was collected with read-only SSH commands. No remote files were modified, no training was run, and no rsync/push was performed.

## Stage 1 Remote Git Information After Successful Retry

Remote repo path:

```text
/home/xiaoj/hal_azi
```

Remote Git status:

```text
## percentage_label+FFT...origin/percentage_label+FFT
?? requirements.txt
```

Remote Git remotes:

```text
origin git@github.com:alittlegangsta/hal_azi.git (fetch)
origin git@github.com:alittlegangsta/hal_azi.git (push)
```

Remote tags: none.

Remote branches and latest commits:

| Branch | Commit | Date | Message |
| --- | --- | --- | --- |
| `1D+percentage_Label` | `e9739c8c3fc4d53e1af63dafa84e29e00b42e9c4` | `2025-11-10T10:36:08+08:00` | 分析脚本新增成像图功能 |
| `1D+percentage_Label+Sample_weights+loss` | `b0825f3e274ea363fbb57814900139ff4de7df4a` | `2025-10-09T16:41:50+08:00` | 采用了样本权重+非对称损失后结果非常差 |
| `GaN` | `c08d695779b3fbb666663a7439b16fbe00e1d61d` | `2025-09-26T14:27:11+08:00` | GaN+2Dlabel |
| `frequency-weighted_loss` | `9899283c351712b791f9837b0be9a96b7003f96f` | `2025-09-23T16:03:42+08:00` | 改进了run_analysis.py，加入指标；并且新增了FFT高频系数惩罚 |
| `master` | `1ee68a62763c091097683e326c3a187c991a85c3` | `2025-09-18T16:23:47+08:00` | 标签改为log变换，效果提升显著，但是预测结果不太好，需要补充结构优化 |
| `percentage_label+FFT` | `7ba021cfa6eacd148247258ee28b8527dbbc6c92` | `2025-12-08T17:05:29+08:00` | 删除部分注释 |
| `test_relativity` | `5e59652a4259c59e1d22b069271e2a67d863dadd` | `2025-09-25T17:07:26+08:00` | CNN+分类任务：证实CWT与Label存在对应关系 |

Remote graph summary:

```text
* 7ba021c (HEAD -> percentage_label+FFT, origin/percentage_label+FFT) 删除部分注释
* a6892bd 最终更新
* 349f5a2 FFT大功告成，但是还需要整一下深度成像图以及如何展示出FFT的效果
* e9739c8 (origin/1D+percentage_Label, 1D+percentage_Label) 分析脚本新增成像图功能
| * b0825f3 (1D+percentage_Label+Sample_weights+loss) 采用了样本权重+非对称损失后结果非常差
|/
* 78170ef 删掉了无用的s.py
* 3ab1e73 一维窜槽百分比剖面图作为标签输入，采用预训练模型EfficientNetV2B0作为模型主干。目前存在窜槽程度越严重，模型预判误差越大模型性能越差，需要改进
* 5e59652 (origin/test_relativity, test_relativity) CNN+分类任务：证实CWT与Label存在对应关系
| * c08d695 (origin/GaN, GaN) GaN+2Dlabel
|/
* 3b1eecc 模型崩溃，以单样本循环多次训练证实目前模型无法学习到任何有帮助的特征。目前标签进行了二值化二通道输入
* 92d670d 目前是模式崩溃
* 9899283 (origin/frequency-weighted_loss, frequency-weighted_loss) 改进了run_analysis.py，加入指标；并且新增了FFT高频系数惩罚
* 1ee68a6 (origin/master, origin/log_scaling, master) 标签改为log变换，效果提升显著，但是预测结果不太好，需要补充结构优化
```

Remote reflog confirms the branch/order evidence above, including creation of `1D+percentage_Label+Sample_weights+loss` from `78170ef` on `2025-09-29T11:12:09+08:00` and commit `b0825f3` on `2025-10-09T16:41:50+08:00`.

## Stage 2 Remote Code Structure After Successful Retry

Remote `find /home/xiaoj/hal_azi -maxdepth 3 -type f` shows the checked-out tree contains:

- tracked project code: `config.py`, `main.py`, `src/cwt_transformation/main_transform_translation.py`, `src/data_processing/main_preprocess.py`, `src/data_processing/create_tfrecords.py`, `src/modeling/model.py`, `src/modeling/dataset.py`, `src/modeling/train.py`, `src/interpretation/run_analysis_regressor.py`, `src/visualization/*`, `src/utils/*`
- remote raw data paths: `data/raw/CAST.mat`, `data/raw/D2_XSI_RelBearing_Inclination.mat`
- processed/output directories: `data/processed/image_translation`, `data/processed/fft_regression`, `output/image_translation/array_03`, `output/fft_regression/array_03`, `output/visualization_plots`
- untracked environment file: `requirements.txt`

Method/code mapping from remote branch evidence:

| Target | Remote evidence | Status |
| --- | --- | --- |
| EfficientNet / 1D percentage label | `1D+percentage_Label`, `src/modeling/model.py`, `src/data_processing/create_tfrecords.py`, `src/interpretation/run_analysis_regressor.py` | verified |
| Sample weights + asymmetric loss | remote-only branch `1D+percentage_Label+Sample_weights+loss`, commit `b0825f3`, `src/modeling/train.py` | verified; failed attempt by commit message |
| FFT regression | `percentage_label+FFT`, commits `349f5a2`, `a6892bd`, `7ba021c` | verified |
| dual-channel binary/FFT label | `GaN:src/data_processing/create_tfrecords.py`, `GaN:src/modeling/dataset.py` | verified as GaN branch code |
| Grad-CAM | `src/interpretation/run_analysis*.py`, `src/interpretation/grad_cam.py`, branch grep hits | verified |
| CWT | `src/cwt_transformation/main_transform_translation.py`, `config.py` | verified |
| CSI+SE-ResNet | only weak/partial hits such as early `visualize_model.py` printing "SE-ResNet"; no conclusive full code mapping to result directory | needs_verification |
| dual-channel metadata fusion | no conclusive `metadata` fusion code found across remote branches | needs_verification |
| eccentricity pre-correction | no conclusive `eccentric` / `预校正` code found across remote branches | needs_verification |

## Stage 3 Remote Environment After Successful Retry

Remote conda env `hall` exists at:

```text
/usr/local/anaconda3/envs/hall
```

Python:

```text
/usr/local/anaconda3/envs/hall/bin/python
Python 3.10.18
```

Key package summary:

| Package | Version / Status |
| --- | --- |
| tensorflow | 2.9.1 |
| torch | missing |
| scikit-learn | 1.7.1 |
| scipy | 1.15.3 |
| h5py | 3.14.0 |
| numpy | 1.26.4 |
| pandas | 2.3.0 |
| matplotlib | 3.10.3 |
| keras | 2.9.0 |
| PyWavelets | 1.8.0 |
| seaborn | 0.13.2 |
| tqdm | 4.67.1 |

The remote untracked `requirements.txt` contains the same core package evidence, including `tensorflow==2.9.1`, `keras==2.9.0`, `h5py==3.14.0`, `scikit-learn==1.7.1`, `scipy==1.15.3`, `numpy==1.26.4`, `pandas==2.3.0`, and `PyWavelets==1.8.0`.

## Updated Conclusion After Successful Retry

Decision: `fetch_missing_branches_only`

Reason: remote verification found a committed remote-local branch absent from the local clone: `1D+percentage_Label+Sample_weights+loss` at `b0825f3e274ea363fbb57814900139ff4de7df4a`. Remote also has untracked `requirements.txt`, but this is an environment evidence file rather than experiment code; its contents have been recorded here. No remote code branch conclusively resolves `CSI+SE-ResNet`, `双通道学习` metadata fusion, or `预校正`.
