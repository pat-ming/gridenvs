# Prompt: harder puzzles and Sudoku, the environment side

Paste everything below the line into a Claude Code session started in the repo checkout
(branch `training`).

---

I own the environments in our CSE 5100 project (`src/gridenvs/`). The training side (`src/fbrl/`) is
Motti's, and the GRPO runs go through him or the Claude session on the cluster. The project asks:
- H1: does training a model without the feedback line build independent violation detection?
- H2: does training with it build reliance on it?

Read these first:
- the proposal (`CSE 5100 Project Proposal - Moti & Patrick.tex`);
- `docs/next.md`;
- the top entry of `docs/runlog.md`;
- `docs/envs.md` and `docs/training.md`.

**Where things stand.** On the easy 4x4 sets (`distinct`, `ineq`) with Qwen3-0.6B-Base, GRPO works:
θ_C (trained with the true line) follows a false line about 100% of the time, and repairs as well as θ_A
with no line. We now need the same comparison where the line saves the model real work, and in a
second environment. Both need environment changes first.

**1. Make `spec` / `rows` learnable.** These are the proposal's sets. Their sum constraints mean the
line (and `residual` mode, which adds how far each sum is off) saves arithmetic. The problem: the SFT
warm start solves under 1% of them, so GRPO gets no signal.
- What counts as learnable: `mixed` (the share of instances whose samples disagree) of at least about
  20% on some (m, H) cell, measured with `fbrl.evaluate` on the warm start
  (`$FBRL/models/sft-qwen3-0.6b`).
- Design a ladder of variants in `src/gridenvs/grid4/instances.py` that goes from `distinct`/`ineq`
  difficulty up to `spec`. For example:
  - two row sums instead of four;
  - sums over a smaller value range;
  - row sums without the distinct columns;
  - `spec` at m=1 with extra slack.
- Measure each rung, and tell Motti or the cluster session which ones give a signal, so the three
  arms can run on the hardest rung that still learns.
- Keep each variant's constraint order stable, so index k means the same thing on every board. Tests
  go in `tests/test_grid4.py`.

**2. Sudoku.** It's wired into `fbrl` (`env.name: sudoku`, replies are `WRITE v r c`) but has never
been trained.
- Check the instance generator gives sensible difficulty at small m. Make sure the observation is
  short enough: a whole episode (every turn's observation and reply) has to fit the 4096-token vLLM
  context.
- Decide on constraint indices. They're fixed today (rows 0–8, columns 9–17, boxes 18–26), like
  `distinct`, where θ_C learned "violated: {3}" means column 3 without reading the constraint list. An
  option to shuffle indices per instance would force a real lookup and make a cleaner H2 test.
- Then hand it over for a warm start and runs.

**How we work.**
- Ask before submitting any GPU jobs. Evals of the warm start are short, one GPU each.
- Keep `src/fbrl/` changes to a minimum and tell Motti about them.
- Write what you tried and what came out as a dated entry in `docs/runlog.md`, and commit and push.
- Commit messages are one line. Plain prose, say what didn't work.
