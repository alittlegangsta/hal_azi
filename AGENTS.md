# AGENTS.md

## Thesis evidence inventory rules

This repository is being used to reconstruct the completed HAL azimuthal barrier / cement channeling research project for a master's thesis.

Hard constraints:

- Treat `/mnt/c/Users/Administrator/Desktop/Hal/results` as read-only evidence.
- Do not delete, rename, move, overwrite, or regenerate files under the results directory.
- Do not run expensive training unless explicitly approved.
- Do not modify raw data.
- Do not invent metrics, dates, branch order, or experimental results.
- Every thesis claim must be traceable to at least one of:
  - existing result file,
  - existing code/config,
  - Git history,
  - memo/report/PPT,
  - explicitly marked inference.
- If evidence is missing, mark it as missing rather than guessing.
- Prefer generating inventory tables and summaries first.
- Commit documentation and inventory files, but do not push unless instructed.

Expected outputs:

- `docs/thesis_evidence/experiment_inventory.csv`
- `docs/thesis_evidence/experiment_inventory.md`
- `docs/thesis_evidence/result_tree_inventory.md`
- `docs/thesis_evidence/git_branch_timeline.md`
- `docs/thesis_evidence/method_taxonomy.md`
- `docs/thesis_evidence/thesis_outline.md`
- `docs/thesis_evidence/missing_evidence.md`
- `docs/thesis_evidence/figure_candidates.md`
