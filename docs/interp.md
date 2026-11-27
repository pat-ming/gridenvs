# Inside the trained models

The behavioral results say *what* θ_C does: it edits wherever the line points, and without a line it
repairs as well as θ_A. This track covers the proposal's deliverables 3–5:
- the independent-verification measure I(θ), which H1 is actually about;
- the reliance measure D(θ);
- a causal intervention;
- recoverability.

Two people attack it from two sides: Motti with native neurons, activations and patching, and Patrick
with the Jacobian lens (J-lens). Several questions get both, which is the point: J-lens readouts need
checking against known answers (the proposal says so), and probes and patching supply them. The owner
of each question is marked below. Prompts: [prompts/neurons.md](prompts/neurons.md) and
[prompts/jlens.md](prompts/jlens.md).

Each question is useful on its own, and the first three need only forward passes over a few hundred
prompts.

## What we have

- Models: Qwen3-0.6B-Base after our SFT (`$FBRL/models/sft-qwen3-0.6b`, call it θ_0), and 12 GRPO
  runs from it, 4 seeds each of θ_A (`none`), θ_C (`true`) and θ_D (`noise`). Each run has
  `checkpoints/checkpoint-{100..600}` and `final` under `$FBRL/runs/grpo_kl0.04_<arm>_s<seed>_*/`.
  Six more at `beta` 0.1 (`grpo_kl0.1_*`, 2 seeds per arm). Among those, θ_D seed 1 is the only model
  that learned to find the broken column on its own: 97% of its line-free edits land there, against
  about 27% elsewhere. That makes it the natural positive control for the I(θ) probes and the J-lens.
  Final weights for all of these are also W&B model artifacts named after the run. Where everything
  else lives (W&B groups, repo vs cluster, how to fetch a model): `docs/next.md`.
- Prompts: `fbrl.grid` builds the exact token sequence the model saw. `ChatFormat(tok)` with
  `start(env, inst, fmt)` gives an `Episode` with `prompt_ids`. Instances come from
  `load_env("grid4")[1](set_name, m, seed)`; use seeds from `EVAL_SEED_START` up so nothing overlaps
  training. The env's `mode` decides the line: `none`, `true`, `noise`, or `counterfactual` (a false
  line the same size as the true one).
- On `distinct` the line is `violated: {k}` where k is the column, so true/false pairs differ in one
  token. On `ineq`, k names a pair of cells listed in the prompt (`c3: cell (2,1) < cell (0,2)`), and
  the model has to look it up.

## Shared groundwork

Build this once, before either side starts, so both work on the same prompts and models:
- A fixed prompt set from `EVAL_SEED_START` up: about 2000 states each for `distinct` and `ineq`, in
  each of the modes `none`, `true` and `counterfactual`, saved with V(s) and the shown line.
- The models in one list: θ_0, and `final` for θ_A, θ_C and θ_D at each seed.
- A loader that gives last-prompt-token residuals per layer, cached to `$FBRL/interp/`.

## Questions

1. **I(θ): does the model work out the broken constraints itself?** (Motti: probes. Patrick:
   J-lens readouts of the same states.) This is H1 as the proposal states it.
   - On line-free states, fit a linear probe per violation bit (one per constraint, from the residual
     stream at the last prompt token), layer by layer, for θ_0, θ_A, θ_C, θ_D across seeds.
   - Exact-match accuracy of the decoded set is the probe version of I(θ).
   - Then put a false line in: does θ_C's internal V get overwritten, or computed and then ignored?
   - Add a behavioral version too: prompt each model to list the violated constraints. The models
     were never trained to do this, so expect to need a few-shot prompt and to compare against θ_0.
2. **(Motti) D(θ): how much do actions move between a true and a false line?** Compute the total-variation
   distance between the next-edit distributions from the logits. The edit is three digits (r c v), so
   either chain them or use the (r, c) marginal. This replaces the sampled `follow_false` with the
   proposal's measure.
3. **(Motti) Where does θ_C read the line?** Take true/false prompt pairs that differ only in the line's index
   token. The metric is the logit difference between the true and the false column's digit at the
   edit's column position.
   - Activation patching: residual stream at the line token per layer, then head outputs, clean into
     corrupt.
   - θ_A and θ_D should show nothing, which checks the method.
   - Then ablate the heads you find and report Δ_M (the drop in repair success on held-out states).
     Controls: a norm-matched random ablation, and line-free repair, which should be unaffected if the
     heads are specific to the line.
4. **(Motti) Is θ_C's line-free path the same as θ_A's?** All the models are small fine-tunes of θ_0, so the
   residual streams should be comparable.
   - Train a probe from 1 on θ_A and test it on θ_C.
   - Patch line-free prompts with one grid cell corrupted, to find which heads carry the duplicate to
     the answer in each model.
5. **(Either, later) Recoverability.** Can a classifier on activations (or on behavior) tell which condition trained a
   model, when trained on some seeds and tested on held-out ones? Seeds are the unit here, so with 4
   per arm this is leave-one-seed-out.
6. **(Patrick; J-lens is cheap to sweep over checkpoints) When does it appear?** Run 1, 3 or 9
   across checkpoints 100–600 for one θ_C seed, against when following the line switches on in the eval records. Does the circuit form at the same time?
7. **(Either) `ineq` lookup.** θ_C follows the line there too, which needs two hops: index →
   constraint text → cells. Attention from the line's index token to the matching `cN:` line, then on to the cell
   coordinates, would show it.
8. **(Both, later) Bigger models.** Repeat 1–4 and 9 on the 1.7B runs once they exist (`docs/prompts/scale.md`).
9. **(Patrick) Does the model "say" the error before it acts?** The proposal's use of the J-lens:
   - Read out the residual stream at the last prompt token on line-free states.
   - Look for words naming the broken row, column or constraint before the model edits.
   - Compare θ_A against θ_C. If H1 holds, θ_A should verbalize the violation more.
   - At the line token in θ_C, does the readout carry the line's content?
10. **(Motti) Native neurons.** Once patching (3) has found the layers and heads that matter, look at
    the MLP neurons feeding them. Are there neurons that fire on "this column has a duplicate" or on
    the line's index? Test any candidate with a controlled input, since the proposal's footnote warns
    that neuron descriptions often fail those tests. Patch or ablate it and see whether behavior moves.

## Practical

- Use a GPU node interactively (`srun --gres=gpu:a40:1 --pty bash` on `general-gpu-long`), not the
  login node. A 0.6B forward pass on a few hundred prompts takes seconds.
- Check whether TransformerLens loads Qwen3. If it doesn't, plain PyTorch forward hooks on the HF model
  (or nnsight) are enough for patching and probes. The J-lens reference implementation
  (github.com/anthropics/jacobian-lens) has Qwen examples.
- Save results under `results/interp/` with a dated entry in `docs/runlog.md`, like the training runs.
