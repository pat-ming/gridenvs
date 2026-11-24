# Prompt: GRPO on the harder sets (`spec`, `rows`)

For Motti. Paste everything below the line into a Claude Code session started in the repo checkout on
the cluster (`/engrfs/project/netanel.raviv/motti_k/fbrl/code`, branch `training`).

---

I'm working on our CSE 5100 project: GRPO on small constraint-repair puzzles under different feedback
conditions. The questions: does training without the feedback line build independent violation
detection (H1), and does training with it build reliance on it (H2)?

Before anything else, read:
- the proposal (`CSE 5100 Project Proposal - Moti & Patrick.tex`);
- `docs/next.md`;
- the top three entries of `docs/runlog.md`;
- `docs/training.md` and `docs/envs.md`.

Together they say what's been done, the recipe that works, and the mistakes we've already made.

**Where things stand.** On the easy 4x4 sets (`distinct`, `ineq`) with Qwen3-0.6B-Base, the recipe in
`configs/experiments/grpo_kl.yaml` trains stably: lr 1e-5, KL `beta` 0.04, 600 steps. Results so far:
- θ_C (trained with the true line) follows a false line about 100% of the time.
- θ_C repairs as well as θ_A with no line.

My job now is to get the same comparison on the proposal's sets, `spec` and `rows`. These have sum
constraints, so the line saves the model real work (arithmetic), and `residual` mode (the line plus how
far each sum is off) becomes meaningful.

**The problem.** The warm start (`$FBRL/models/sft-qwen3-0.6b`) solves under 1% of `spec`/`rows`, so
GRPO groups are almost all zeros and nothing is learned. The target is `mixed` (the share of instances
whose samples disagree, from `fbrl.evaluate`) of at least about 20% on some cell, in every arm.

Options, cheapest first. Measure each with `fbrl.evaluate` before training on it.
1. Restrict to m=1 with slack 1 or 2 (more turns than edits needed).
2. A curriculum: train on `distinct`/`ineq` plus a growing share of `spec`. Or start from a `grpo_kl`
   θ_A/θ_C/θ_D checkpoint instead of the warm start, keeping each arm on its own lineage.
3. A higher `demos.oracle` share in SFT. An earlier oracle-share study was invalidated by a stop-token
   bug, so it needs redoing. Watch for it teaching line-following from the demos alone.
4. Easier variants of the sets in `src/gridenvs/grid4/instances.py`, for example:
   - fewer sum constraints (two rows instead of four);
   - a smaller value range;
   - a set between `rows` and `spec`.
   
   `src/gridenvs/` is Patrick's code, so tell him what you change.

Once some option gives a signal, run `none`, `true`, `noise` and `residual` with the `grpo_kl` recipe,
3–4 seeds each. Check each run: does training success climb, and do `rollout/stopped` and entropy stay
healthy?

**How to judge results.**
- Eval success by mode (none / true / counterfactual), averaged over the last five evals.
- `python -m fbrl.placement <run dirs>`, which gives two shares of first edits:
  - `follow_false`: edits inside the constraint a false line names;
  - `find_alone`: edits inside a truly violated constraint, with no line.
- Compare arms over seeds, not single runs: on `ineq`, θ_A and θ_D seeds split.

**How we work.**
- Ask before submitting GPU jobs, and say how many GPUs and for how long. Keep the number of
  experiments small and each one aimed at a question.
- Two 0.6B runs fit on one 48 GB GPU: `--export=ALL,SWEEP_FILE=...,PER_GPU=2`. Never run GPU work on
  the login node.
- Submitting from inside an interactive job, first run
  `unset $(env | grep -oE "^(APPTAINER|SINGULARITY)_[A-Z_]*")`, or jobs die on a missing /opt/intel
  mount.
- Keep the "## In flight" section at the top of `docs/runlog.md` current: job IDs, what each is for,
  what to check when done.
- When results come in, write a dated entry (what ran, what came out, what it means), pull the small
  run files, then commit and push.
- Commit messages are one line. Talk to me in plain prose, say what didn't work, and don't oversell
  results.
