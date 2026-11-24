# Prompt: does the result hold on harder tasks, Sudoku, and a bigger model?

Paste everything below the line into a Claude Code session started in the repo checkout on the cluster
(`/engrfs/project/netanel.raviv/motti_k/fbrl/code`, branch `training`).

---

I'm working on our CSE 5100 project: GRPO on small constraint-repair puzzles under different feedback
conditions, to see whether a model trained with a feedback line comes to rely on it (H2) and whether that
costs it its own skill (H1). Read the proposal (`CSE 5100 Project Proposal - Moti & Patrick.tex`), `docs/next.md`, then the top three entries of `docs/runlog.md`, then
`docs/training.md` and `docs/envs.md`. Together they say what's been done, the recipe that works, and
the mistakes we've already made.

**Where things stand.** On the easy 4x4 sets (`distinct`, `ineq`) with Qwen3-0.6B-Base, the recipe in
`configs/experiments/grpo_kl.yaml` (lr 1e-5, KL `beta` 0.04, 600 steps) trains stably. H2 holds:
θ_C follows a false line about 100% of the time. H1 doesn't: θ_C is as good as θ_A without the line.
My job is to find out whether that holds where the task is harder, in a different environment, and at
a bigger model size.

**The work, in order.** Check each step against the yes/no question before going on.

1. **`spec` and `rows` at 0.6B.** The warm start (`$FBRL/models/sft-qwen3-0.6b`) solves under 1% of
   these, so GRPO groups are almost all zeros and nothing is learned. First get a signal: the target
   is `mixed` (share of instances whose samples disagree, from `fbrl.evaluate`) of at least about 20%
   on some cell. Options, cheapest first:
   - restrict to m=1 with slack 1;
   - a mode schedule;
   - a higher `demos.oracle` share in SFT (an earlier oracle-share study was invalidated by a
     stop-token bug, so it needs redoing);
   - easier variants of the sets in `src/gridenvs/grid4/instances.py`, e.g. fewer sum constraints or
     smaller values.

   Then run the three arms (`none`, `true`, `noise`, plus `residual`, which only means something on
   sum constraints) with the `grpo_kl` recipe, at least 3 seeds each.
   Yes/no: does training success climb in every arm, with `rollout/stopped` staying near 1?
2. **Sudoku.** It's wired in (`env.name: sudoku`, replies are `WRITE v r c`) but has never been trained.
   - SFT a warm start: `fbrl.train_sft` with `env.name=sudoku`, and an export path like
     `$FBRL/models/sft-qwen3-0.6b-sudoku`.
   - Eval it and check that `stopped` and `parse` are both near 1. The `<|im_end|>` fix in
     `train_sft.py` should apply automatically; confirm it logs `init_end_of_reply`.
   - Run the three arms with the recipe.
   
   One thing to keep in mind: Sudoku's constraint indices are fixed (rows 0–8, columns 9–17, boxes
   18–26), like `distinct`'s. On `distinct`, θ_C learned "violated: {3}" means column 3 without reading
   the constraint list. If we want to test real lookup, shuffling constraint indices per instance is
   the env change that does it.
3. **Qwen3-1.7B-Base.**
   - SFT a 1.7B warm start (`configs/models/qwen3-1.7b.yaml`).
   - Replicate `grpo_kl` on `distinct`/`ineq`, 3 seeds per arm, one run per GPU at first. Check memory
     before packing two (`PER_GPU=2`).
   - Then repeat whichever of 1–2 worked.
   
   Yes/no: same H2 and H1 picture as 0.6B? The lr may need retuning: try 5e-6 if 1e-5 is unstable.

**How to judge results.** Eval success by mode (none / true / counterfactual) averaged over the last
five evals. Then `python -m fbrl.placement <run dirs>`, which gives the share of first edits inside the
constraint a false line names (`follow_false`) and inside a truly violated one with no line
(`find_alone`). It's grid4-only; extend it to Sudoku with `edited_cell("sudoku", ...)`. Compare arms
over seeds, not single runs: θ_A and θ_D seeds split on `ineq`.

**How we work.**
- Ask before submitting GPU jobs, and say how many GPUs and for how long. Keep the number of
  experiments small and each one aimed at a question.
- Never run GPU work on the login node.
- Submitting from inside an interactive job: first
  `unset $(env | grep -oE "^(APPTAINER|SINGULARITY)_[A-Z_]*")`, or jobs die on a missing /opt/intel
  mount.
- Keep the "## In flight" section at the top of `docs/runlog.md` current: job IDs, what each is for,
  what to check when done. Write a dated entry (what ran, what came out, what it means) when results
  come in, pull small run files with `cluster/remote.sh pull` or the equivalent rsync, and commit and
  push.
- Commit messages are one line. Talk to me in plain prose, say what didn't work, and don't oversell
  results.
- Env code under `src/gridenvs/` is Patrick's. Training code under `src/fbrl/` is Motti's. Extend
  across that line rather than rewriting the other side, and mention the change in the runlog.
