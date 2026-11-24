# What's next

As of 2026-11-24. Where we are against the proposal (`CSE 5100 Project Proposal - Moti & Patrick.tex`),
how the remaining work splits, and the experiments we haven't run yet. Ready-to-paste prompts for each
track are in [prompts/](prompts/), and shared interp notes in [interp.md](interp.md).

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

Starter prompt for the J-lens and environments track: [prompts/jlens-and-harder-envs.md](prompts/jlens-and-harder-envs.md).

**Patrick: environments, and the J-lens.**
1. Make `spec` / `rows` (the proposal's sets) learnable. The warm start solves under 1%, so GRPO gets no
   signal. The fix is a ladder of easier variants in `src/gridenvs/` going up to `spec`, measured with
   `fbrl.evaluate`. Prompt: [prompts/environments.md](prompts/environments.md).
2. Get Sudoku ready to train. It's already wired into `fbrl` but has never been trained. The open
   choice is whether constraint indices should be shuffled per instance, which would force the model
   to look the line up instead of memorizing it.
3. The J-lens on our trained models. Does the model "say" which constraint is broken before acting,
   and what does θ_C carry at the line token? Prompt: [prompts/jlens.md](prompts/jlens.md).

Environment work is the fallback if the J-lens turns out not to read well on our task.

**Motti: inside the models, and model size.** The critical path to the proposal's deliverables.
1. I(θ) probes (this answers H1), activation patching to find where θ_C reads the line, a Δ_M ablation
   as the causal intervention, then neuron-level analysis of what patching finds. Prompt:
   [prompts/neurons.md](prompts/neurons.md).
2. The model-size ladder: 1.7B, then 4B, on `distinct`/`ineq`. Stop there unless something changes
   with size: past 4B needs multi-GPU training code we don't have. Prompt:
   [prompts/scale.md](prompts/scale.md).

**Claude on the cluster: runs and groundwork for both.**
- The shared interp groundwork in [interp.md](interp.md): fixed prompt set, model list, activation
  cache. Both interp sides start from it.
- Training runs once a recipe or env exists: the three arms on whatever `spec` rung Patrick finds
  learnable, Sudoku once it's ready, and each model size.
- *H3, an unreliable line:* show the true line with probability p and a false one otherwise. Where
  does reliance switch off, and does line-free skill grow as p drops? Needs a mode between `true` and
  `noise`.
- *Line taken away mid-training:* `env.mode` takes a schedule like `[[0, "true"], [300, none]]`.
- *More θ_A seeds on `ineq`*, to settle whether θ_C's lead there is a seed effect.

The two interp sides check each other: J-lens readouts against probes and the known V(s), and probes
against what the lens says the model represents.

**Later or stretch:** the optional-verifier environment E_T; the SFT comparison (actions only,
diagnosis as input, diagnosis as a target); coding tasks.

## Where the results live

- **W&B:** project [fbrl](https://forge.coreweave.com/wandb/cse-5100-fall-2026/fbrl) in team
  `cse-5100-fall-2026`. Filter by group:

  | Group | What it is |
  |---|---|
  | `grpo_kl` | the main result: 18 runs, `beta` 0.04 (4 seeds per arm) and 0.1 (2 seeds) |
  | `grpo_arms` | the same at `beta` 0; 3 of 6 broke |
  | `grpo_lr2` | the learning-rate check |

  Runs tagged `invalid-untrained-im-end` predate the stop-token fix; ignore them. Each run log entry
  links its runs individually.
- **In the repo** (`results/runs/<run>/`), for every run:
  - `config.yaml`: the full resolved config;
  - `events.jsonl`: every logged metric per step;
  - `records/eval_cells.jsonl`: the eval per cell per step;
  - `summary.json` and `meta.json`.
  
  That's enough to redo any table in the run log except the edit-placement columns.
- **On the cluster only** (`$FBRL/runs/<run>/`, readable by anyone on the cluster), because they're too
  big for git:
  - `checkpoints/` (about 22 GB a run);
  - `records/eval_episodes.jsonl`, every eval episode, which `fbrl.placement` reads (about 11 MB a run);
  - `records/rollouts.jsonl`, every training rollout.

  W&B also has rollout tables every 25 steps.
- **Models worth looking inside first:** θ_C and θ_A at seed 0 of `grpo_kl0.04`, and θ_D seed 1 of
  `grpo_kl0.1`. That θ_D is the only model that learned to find the broken column on its own (see the
  run log).

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
