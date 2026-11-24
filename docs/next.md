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

**Motti: harder puzzles, Sudoku, bigger models, and native neurons and activations.**
1. Get a GRPO signal on `spec` / `rows` (the proposal's sets), then run the arms there. Prompt:
   [prompts/harder.md](prompts/harder.md). Right now the warm start solves under 1% of these, so it
   needs an easier way in first: a curriculum, more oracle demos, or easier variants of the sets.
   Variants mean touching Patrick's `src/gridenvs/`, so tell him what changes.
2. Interp from the neuron and activation side: I(θ) probes (this answers H1), activation patching to
   find where θ_C reads the line, a Δ_M ablation as the proposal's causal intervention, then
   neuron-level analysis of the parts patching finds. Prompt: [prompts/neurons.md](prompts/neurons.md).
3. Sudoku (warm start, then the three arms), and the model-size ladder: 1.7B → 4B → 8B, optionally
   14B with LoRA as a behavior-only check. Up to 4B, full-weight GRPO fits on one 80 GB GPU. 8B needs
   the optimizer state split across GPUs, which our pipeline doesn't do yet. Prompt:
   [prompts/scale.md](prompts/scale.md).

**Patrick: the J-lens, and iterating on it.** The same questions from the readout side:
- Does the model "say" which constraint is broken before acting, and does θ_A do it more than θ_C?
- What does θ_C's residual stream carry at the line token?
- When does that change across checkpoints?

Prompt: [prompts/jlens.md](prompts/jlens.md). As the bigger models come in, the same readouts on
those.

The two interp sides share one prompt set and model list. The shared notes, the question list and
who owns what are in [interp.md](interp.md). Each side checks the other: J-lens readouts against
probes and known V(s), and probes against what the lens says the model represents.

**Claude on the cluster: runs that only need the existing recipe.**
- *H3, an unreliable line:* show the true line with probability p and a false one otherwise. Where
  does reliance switch off, and does line-free skill grow as p drops? Needs a mode between `true` and
  `noise`.
- *Line taken away mid-training:* `env.mode` takes a schedule like `[[0, "true"], [300, none]]`.
- *More θ_A seeds on `ineq`*, to settle whether θ_C's lead there is a seed effect.

**Later or stretch:** the optional-verifier environment E_T; the SFT comparison (actions only,
diagnosis as input, diagnosis as a target); coding tasks.

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
