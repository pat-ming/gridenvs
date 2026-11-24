# Start here, Patrick

Paste this into Claude Code in the repo checkout (branch `training`):

---

Read `docs/next.md` for where the project stands and how the work is split. My part has two tracks:
`docs/prompts/environments.md` (make `spec`/`rows` learnable, get Sudoku ready to train) and
`docs/prompts/jlens.md` (the Jacobian lens on our trained models, with shared notes in `docs/interp.md`).
Read both, then start with the first step of `environments.md`. Measure the current warm start on
`spec`/`rows` so we know the baseline, and tell me what you find before changing anything. Ask before
submitting GPU jobs.
