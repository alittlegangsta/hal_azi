# TensorBoard Parsing Preflight

Generated: 2026-07-06. Results root was read-only evidence: `/mnt/c/Users/Administrator/Desktop/Hal/results`.

| check | status | error |
| --- | --- | --- |
| python | 3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0] |  |
| results_root_readable | True |  |
| tensorboard import | False | ModuleNotFoundError: No module named 'tensorboard' |
| EventAccumulator import | False | ModuleNotFoundError: No module named 'tensorboard' |
| tensorflow import | False | ModuleNotFoundError: No module named 'tensorflow' |
| event files found | 72 |  |
| decision | needs_tensorboard_dependency |  |

No TensorBoard scalar values were inferred in this environment because `EventAccumulator` is unavailable. Use `scripts/thesis_parse_tensorboard_events.py` after activating an environment with `tensorboard`.
