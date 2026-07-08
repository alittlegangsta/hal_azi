# EXP-007 Split Local Copy Status

The split was created remotely under:

`/home/xiaoj/hal_azi/output/thesis_depth_blocked/exp007/split_v001`

Remote dry-run, real split, and smoke check succeeded. Full `split_manifest.csv/json` was not copied to local docs because SSH/SCP escalation was later denied by the execution environment usage limit. The summary audit files in this folder are reconstructed from captured remote command output and should be treated as trace summaries, not byte-for-byte copies.

Do not copy `train.tfrecord`, `val.tfrecord`, or `test.tfrecord` into git.
