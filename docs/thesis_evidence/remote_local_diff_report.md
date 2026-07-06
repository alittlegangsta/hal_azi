# Remote Local Diff Report

Verification date: `2026-07-06`

## Summary

Current remote/local comparison status: `verified_after_retry_3`

Latest migration decision: `fetch_missing_branches_only`

Historical SSH failures are retained below for traceability. The first successful probe was Retry Attempt 3 on `2026-07-06`.

## Initial Failed Comparison

The remote server could not be read because SSH authentication failed. No remote files were listed, no Git metadata was read, and no environment commands were executed. Therefore, this report does not assert that the local repo is complete relative to `/home/xiaoj/hal_azi`; it only records the comparison questions that remain unresolved.

## Local Baseline

The local evidence inventory currently confirms these local or origin-tracking branches:

| Branch | Mapped experiment family | Confidence |
| --- | --- | --- |
| `origin/log_scaling` / `origin/master` | FFT log label | strong/medium |
| `origin/frequency-weighted_loss` | FFT high-frequency weighted loss | strong |
| `origin/test_relativity` | CWT-label binary classification | strong |
| `origin/GaN` | GAN/severity map | medium |
| `origin/1D+percentage_Label` | 1D percentage label / EfficientNetV2B0 | strong |
| `origin/percentage_label+FFT` | FFT regression / severity FFT route | strong |

## Initial Unresolved Remote Questions Before Retry 3

| Question | Status | Thesis impact |
| --- | --- | --- |
| Does remote `/home/xiaoj/hal_azi` contain local-only branches not present in this clone? | needs_verification | Could explain CSI+SE-ResNet, dual-channel, and pre-correction code provenance. |
| Does remote repo contain uncommitted tracked changes? | needs_verification | Could affect exact method definitions and branch chronology. |
| Does remote repo contain untracked scripts/configs? | needs_verification | Could contain experiment code not present in Git. |
| Does remote branch/reflog history provide older baseline or failed-attempt order? | needs_verification | Needed before making precise chronological claims. |
| Does remote `hall` environment define TensorFlow/Keras/scipy/h5py versions used in training? | needs_verification | Needed for reproducibility section. |

## Result Directory Mapping After Failed Remote Access

| Result directory | Current mapping status | Notes |
| --- | --- | --- |
| `temp_result/baseline` | partially mapped locally | Memo and local history support route, but exact baseline branch remains needs_verification. |
| `temp_result/log_label` | mapped locally | Supported by `origin/log_scaling` and current code. |
| `temp_result/frequency-weighted_loss` | mapped locally | Supported by `origin/frequency-weighted_loss`. |
| `temp_result/GaN+2Dlabel` | partially mapped locally | Supported by `origin/GaN`, but exact two-channel implementation details need verification. |
| `temp_result/test_relativity` | mapped locally | Supported by `origin/test_relativity` and result text. |
| `temp_result/1D+percentage_Label` | mapped locally | Supported by `origin/1D+percentage_Label`. |
| `FFT_EfficientNet` / `FFT_EfficientNet_1` | partially mapped locally | Supported by `origin/percentage_label+FFT`, but final metric provenance needs verification. |
| `CSI+SE-ResNet` | needs_verification | No conclusive local branch mapping. |
| `双通道学习` | needs_verification | No conclusive local branch mapping. |
| `预校正` | needs_verification | No conclusive local branch mapping. |

## Migration Decision

Decision: `manual_review_required`

Do not migrate or rsync anything yet. The safe next step is to restore SSH authentication or run the requested read-only commands manually on the server and paste/save their output. After that:

- Use `fetch_missing_branches_only` if remote has committed branches missing locally.
- Use `rsync_remote_untracked_code_only` only if remote has small, thesis-relevant untracked scripts/configs that are not in Git.
- Keep `no_migration_needed` only if remote branches, untracked files, and environment evidence add nothing beyond current local inventory.

## Retry Attempt

Retry date: `2026-07-06`

The required probe command failed:

```text
ssh -o BatchMode=yes cement-server 'echo remote_ssh_ok'
-> Permission denied (publickey,password).
```

Because the probe failed, no remote/local comparison data was collected in the retry. The comparison status remains `needs_verification`, and the migration decision remains `manual_review_required`.

## Retry Attempt 2

Retry date: `2026-07-06`

The required probe command failed again:

```text
ssh -o BatchMode=yes cement-server 'echo remote_ssh_ok'
-> Permission denied (publickey,password).
```

Because this probe failed, no remote/local comparison data was collected in this retry. The comparison status remains `needs_verification`, and the migration decision remains `manual_review_required`.

## Retry Attempt 3

Retry date: `2026-07-06`

The required probe command succeeded:

```text
ssh -o BatchMode=yes cement-server 'echo remote_ssh_ok'
-> remote_ssh_ok
```

Remote/local comparison was then completed with read-only commands.

## Remote vs Local Findings After Successful Retry

| Check | Finding | Status |
| --- | --- | --- |
| Remote current branch | `percentage_label+FFT`, tracking `origin/percentage_label+FFT` | verified |
| Remote uncommitted tracked changes | `git diff --stat` and `git diff --cached --stat` produced no tracked changes | verified |
| Remote untracked files | `requirements.txt` | verified |
| Remote branch missing locally | `1D+percentage_Label+Sample_weights+loss` at `b0825f3e274ea363fbb57814900139ff4de7df4a` | verified |
| Local object availability | local `git cat-file -t b0825f3...` failed: object absent | verified |
| Remote tags | none | verified |
| Remote raw/output data in repo working tree | `data/raw/*.mat`, `data/processed/*`, `output/*` exist on remote filesystem | path-only metadata recorded; not migrated |
| Remote `hall` env | `/usr/local/anaconda3/envs/hall`, Python 3.10.18, TensorFlow 2.9.1 | verified |

## Updated Result Directory Mapping

| Result directory / method | Remote mapping update |
| --- | --- |
| `temp_result/1D+percentage_Label` | remote confirms existing `1D+percentage_Label` plus a failed branch `1D+percentage_Label+Sample_weights+loss` for sample weighting and asymmetric loss |
| `FFT_EfficientNet` / `FFT_EfficientNet_1` | remote confirms `percentage_label+FFT` branch chronology and current checkout |
| `temp_result/GaN+2Dlabel` | remote confirms `GaN` branch contains two-channel binary FFT label code |
| `CSI+SE-ResNet` | still `needs_verification`; only weak early `SE-ResNet` visualization-script evidence, no conclusive code/result mapping |
| `双通道学习` metadata fusion | still `needs_verification`; remote branch grep did not find conclusive metadata-fusion code |
| `预校正` / eccentricity pre-correction | still `needs_verification`; remote branch grep did not find conclusive eccentricity/pre-correction code |

## Updated Migration Decision

Decision: `fetch_missing_branches_only`

Rationale: the only remote experiment-code evidence missing locally is a committed branch, `1D+percentage_Label+Sample_weights+loss`. The untracked `requirements.txt` is environment evidence, and its content has been captured in the inventory. No large files should be rsynced. A future fetch, if approved, should target only the missing branch from the remote repository.
