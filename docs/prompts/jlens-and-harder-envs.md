# Task: J-lens and harder environments

Paste this into Claude Code in the repo checkout (branch `training`):

---

Read `docs/next.md` for where the project stands and where results live (W&B groups, `results/runs/`,
the cluster). Then read the newest entries of `docs/runlog.md`, our running record (newest first).
That's where the current best settings and the latest results are, and it gets updated as runs come
in, so check it again before starting anything new. This task has two parts:
- `docs/prompts/environments.md`: make `spec`/`rows` learnable and get Sudoku ready to train.
- `docs/prompts/jlens.md`: the Jacobian lens on our trained models. Shared notes are in
  `docs/interp.md`.

Read both, then start with the first step of `environments.md`. Measure the current warm start on
`spec`/`rows` so we have a baseline, and report what you find before changing anything. Ask before
submitting GPU jobs.
