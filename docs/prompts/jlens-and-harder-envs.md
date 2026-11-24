# Task: J-lens and harder environments

Paste this into Claude Code in the repo checkout (branch `training`):

---

Read `docs/next.md` for where the project stands. This task has two parts:
- `docs/prompts/environments.md`: make `spec`/`rows` learnable and get Sudoku ready to train.
- `docs/prompts/jlens.md`: the Jacobian lens on our trained models. Shared notes are in
  `docs/interp.md`.

Read both, then start with the first step of `environments.md`. Measure the current warm start on
`spec`/`rows` so we have a baseline, and report what you find before changing anything. Ask before
submitting GPU jobs.
