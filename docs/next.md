# What's next

As of 2026-11-24. Where we are, how the remaining work splits, and the ideas we haven't run yet.
Ready-to-paste prompts for each track are in [prompts/](prompts/).

## Where we are

On the two easy 4x4 sets (`distinct`, `ineq`), with Qwen3-0.6B-Base and a working GRPO recipe (lr 1e-5,
KL `beta` 0.04, 600 steps, `configs/experiments/grpo_kl.yaml`), 4 seeds per arm:

- **H2 holds.** θ_C (trained with the true line) puts about 100% of its first edits where the line
  points, true or false, and fails when it's false. θ_A and θ_D ignore the line.
- **H1 doesn't.** With the line taken away, θ_C is as good as θ_A on `distinct` and better on `ineq`
  (18% against 8%). Training with the line didn't cost the model its own skill.

Details in the 2026-11-24 entry of [runlog.md](runlog.md). What we don't know yet is whether either
result survives a harder task, a different environment, or a bigger model, and what is going on inside.

## The split

**Track A, generality (Patrick).** Does the result hold where the task is harder and the line matters
more? Prompt: [prompts/generality.md](prompts/generality.md).
1. `spec` / `rows`, the proposal's sets. Today the warm start solves under 1%, so GRPO gets no signal.
   This needs an easier way in: a curriculum over m and H, a mode schedule, more oracle demos, or easier
   variants of the sets. Env design, so Patrick's call.
2. Sudoku. Already wired into `fbrl` (`env.name: sudoku`) but never trained: it needs its own SFT warm
   start, then the same three arms.
3. Qwen3-1.7B-Base on `distinct`/`ineq` first (a straight replication), then on whatever works from 1–2.

**Track B, inside the models (Motti, or Patrick once A is running).** How θ_C reads the line, and
whether its no-line path is the same as θ_A's. Notes and prompt: [prompts/interp.md](prompts/interp.md).
Starts now on the 0.6B checkpoints; repeat on 1.7B+ when Track A produces them.

**Track C, more experiments on the 4x4 sets (whoever has GPUs free; Claude on the cluster can run
these).** These are cheap because the recipe already works:
- *Take the line away mid-training.* `env.mode` takes a schedule, e.g. `[[0, "true"], [300, none]]`.
  Does θ_C drop its reliance once the line is gone, and how fast? This is the closest thing to the
  proposal's question of whether RL builds what the environment already runs.
- *Partly reliable line.* Show the true line with probability p and a false one otherwise. Where does
  reliance switch on? Needs a small env change (a mode between `true` and `noise`).
- *More θ_A seeds on `ineq`.* θ_A and θ_D split there (some seeds pick it up, some don't). Four more
  seeds per arm would say whether θ_C's H1 advantage is real or a seed effect.
- *`residual` on `spec`.* The line plus how far each sum is off. Only meaningful once `spec` has a
  signal (Track A, step 1).

## Practical notes for anyone starting

- Checkpoints: `$FBRL/runs/grpo_kl0.04_<arm>_s<seed>_*/checkpoints/checkpoint-{100..600}` and `final`,
  about 22 GB per run because each keeps optimizer state. The optimizer files can go once we're sure
  we won't resume; the model weights are what interp needs.
- Watch `rollout/stopped` and entropy on every new run. Both failures we've had (an untrained stop
  token, then drift with `beta` 0) showed up there first, while `parse` looked fine.
- Two runs fit on one 48 GB GPU at 0.6B: `--export=ALL,SWEEP_FILE=...,PER_GPU=2`. At 1.7B, start with
  one per GPU and check memory.
- `python -m fbrl.placement <run dirs>` gives where first edits land (follows a false line? finds the
  broken constraint alone?). It's grid4-only for now.
