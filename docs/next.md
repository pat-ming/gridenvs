# What's next

As of 2026-11-24. Where we are against the proposal (`CSE 5100 Project Proposal - Moti & Patrick.tex`),
how the remaining work splits, and the experiments we haven't run yet. Ready-to-paste prompts for each
track are in [prompts/](prompts/).

## Where we are

On the two easy 4x4 sets (`distinct`, `ineq`), with Qwen3-0.6B-Base, we have a working GRPO recipe
(lr 1e-5, KL `beta` 0.04, 600 steps, `configs/experiments/grpo_kl.yaml`) and 4 seeds per arm:

- **H2 (reliance) holds.** θ_C (trained with the true line) puts about 100% of its first edits where
  the line points, true or false, and fails when it's false. θ_A and θ_D ignore the line.
- **H1 isn't measured yet, as the proposal defines it.** The proposal's H1 is about *independent
  violation detection*, I(θ): can the model say which constraints are broken, with no line. What we
  have is two nearby numbers:
  - Repair success with no line: θ_C is level with θ_A on `distinct` and ahead on `ineq`, 18% vs 8%.
  - Whether the first edit lands in a broken constraint: about 26% in every arm.

  Both say training with the line didn't cost the model its own repair skill. Neither is I(θ).

Details in the 2026-11-24 entry of [runlog.md](runlog.md).

## Against the proposal's deliverables

| Deliverable | Status |
|---|---|
| (1) grid generator, diagnostics, shared warm start | done for grid4; Sudoku env exists, no warm start yet |
| (2) multi-seed RL comparison of E_A, E_C, E_D | done on `distinct`/`ineq` at 0.6B (4 seeds); not on `spec`/`rows`, Sudoku, or 1.7B |
| (3) independent-verification and reliance evals | reliance done (`fbrl.placement`, `cf_score`); I(θ) and the TV measure D(θ) not yet |
| (4) at least one causal intervention | not started |
| (5) recoverability analysis | not started |
| H3: tools costly, unreliable, unavailable | not started |
| Stretch: optional-verifier env E_T, SFT comparison, coding tasks | not started |

## The split

**Track A, generality (Patrick).** Does the result hold where the task is harder, in a second
environment, and at a bigger model size? Prompt: [prompts/generality.md](prompts/generality.md).
1. `spec` / `rows` (the proposal's sets). The warm start solves under 1% of these, so GRPO gets no
   signal; it needs an easier way in (curriculum over m and H, more oracle demos, or easier variants).
2. Sudoku: wired into `fbrl` (`env.name: sudoku`) but never trained. It needs a warm start, then the
   three arms.
3. Qwen3-1.7B-Base: replicate on `distinct`/`ineq`, then on whatever works from 1–2.
4. Later, the optional-verifier environment E_T (a `check_constraints()` action with a cost), since it's
   an env design job.

**Track B, measurement and mechanism (Motti).** The proposal's deliverables 3–5. Notes and prompt:
[prompts/interp.md](prompts/interp.md).
1. I(θ), so H1 can actually be answered. Two routes, ideally both:
   - Probe each violation bit from the residual stream on line-free states (the proposal's §5).
   - A behavioral readout: prompt each model to list the violated constraints.
2. D(θ): total-variation distance between the action distributions under true and false lines,
   computed from logits instead of sampled edits.
3. Where θ_C reads the line, and whether its line-free path matches θ_A's (activation patching, then a
   Δ_M ablation as the proposal's causal intervention).
4. Recoverability: can a probe tell which condition trained a model, across held-out seeds?

**Track C, H3 and other cheap 4x4 experiments (whoever has GPUs free; Claude on the cluster can run
these).** The recipe already works here, so each is a config or a small env mode:
- *Unreliable line (H3).* Show the true line with probability p and a false one otherwise. Where does
  reliance switch off, and does line-free skill grow as p drops? Needs a mode between `true` and `noise`.
- *Line taken away mid-training.* `env.mode` takes a schedule, e.g. `[[0, "true"], [300, none]]`.
  Does θ_C stop relying on the line, and how fast?
- *More θ_A seeds on `ineq`.* θ_A and θ_D split there (some seeds pick it up, some don't), so
  θ_C's lead could be a seed effect. Four more seeds per arm would settle it.
- *SFT comparison (stretch).* Action targets only, against diagnosis as input, against diagnosis as a
  generated target before the edit. That's a demos change in `fbrl.train_sft`.

## Practical notes for anyone starting

- Checkpoints: `$FBRL/runs/grpo_kl0.04_<arm>_s<seed>_*/checkpoints/checkpoint-{100..600}` and `final`.
  That's about 22 GB per run, because each keeps optimizer state. The optimizer files can go once we're
  sure we won't resume; the model weights are what interp needs.
- Watch `rollout/stopped` and entropy on every new run. Both failures we've had showed up there first,
  while `parse` looked fine. The first was an untrained stop token, the second was drift with `beta` 0.
- Two runs fit on one 48 GB GPU at 0.6B (`--export=ALL,SWEEP_FILE=...,PER_GPU=2`). At 1.7B, start with
  one per GPU and check memory.
- `python -m fbrl.placement <run dirs>` reports where first edits land: does the model follow a false
  line, and does it find the broken constraint alone? Grid4 only for now.
