# Notes and prompt: inside the trained models

The behavioral results say *what* θ_C does: it edits wherever the line points, and without a line it
repairs as well as θ_A. This track covers the proposal's deliverables 3–5:
- the independent-verification measure I(θ), which H1 is actually about;
- the reliance measure D(θ);
- a causal intervention;
- recoverability.

Each question below is useful on its own. The first three need only forward passes over a few hundred
prompts.

## What we have

- Models: Qwen3-0.6B-Base after our SFT (`$FBRL/models/sft-qwen3-0.6b`, call it θ_0), and 12 GRPO
  runs from it, 4 seeds each of θ_A (`none`), θ_C (`true`) and θ_D (`noise`). Each run has
  `checkpoints/checkpoint-{100..600}` and `final` under `$FBRL/runs/grpo_kl0.04_<arm>_s<seed>_*/`.
- Prompts: `fbrl.grid` builds the exact token sequence the model saw. `ChatFormat(tok)` with
  `start(env, inst, fmt)` gives an `Episode` with `prompt_ids`. Instances come from
  `load_env("grid4")[1](set_name, m, seed)`; use seeds from `EVAL_SEED_START` up so nothing overlaps
  training. The env's `mode` decides the line: `none`, `true`, `noise`, or `counterfactual` (a false
  line the same size as the true one).
- On `distinct` the line is `violated: {k}` where k is the column, so true/false pairs differ in one
  token. On `ineq`, k names a pair of cells listed in the prompt (`c3: cell (2,1) < cell (0,2)`), and
  the model has to look it up.

## Questions

1. **I(θ): does the model work out the broken constraints itself?** This is H1 as the proposal states
   it.
   - On line-free states, fit a linear probe per violation bit (one per constraint, from the residual
     stream at the last prompt token), layer by layer, for θ_0, θ_A, θ_C, θ_D across seeds.
   - Exact-match accuracy of the decoded set is the probe version of I(θ).
   - Then put a false line in: does θ_C's internal V get overwritten, or computed and then ignored?
   - Add a behavioral version too: prompt each model to list the violated constraints. The models
     were never trained to do this, so expect to need a few-shot prompt and to compare against θ_0.
2. **D(θ): how much do actions move between a true and a false line?** Compute the total-variation
   distance between the next-edit distributions from the logits. The edit is three digits (r c v), so
   either chain them or use the (r, c) marginal. This replaces the sampled `follow_false` with the
   proposal's measure.
3. **Where does θ_C read the line?** Take true/false prompt pairs that differ only in the line's index
   token. The metric is the logit difference between the true and the false column's digit at the
   edit's column position.
   - Activation patching: residual stream at the line token per layer, then head outputs, clean into
     corrupt.
   - θ_A and θ_D should show nothing, which checks the method.
   - Then ablate the heads you find and report Δ_M (the drop in repair success on held-out states).
     Controls: a norm-matched random ablation, and line-free repair, which should be unaffected if the
     heads are specific to the line.
4. **Is θ_C's line-free path the same as θ_A's?** All the models are small fine-tunes of θ_0, so the
   residual streams should be comparable.
   - Train a probe from 1 on θ_A and test it on θ_C.
   - Patch line-free prompts with one grid cell corrupted, to find which heads carry the duplicate to
     the answer in each model.
5. **Recoverability.** Can a classifier on activations (or on behavior) tell which condition trained a
   model, when trained on some seeds and tested on held-out ones? Seeds are the unit here, so with 4
   per arm this is leave-one-seed-out.
6. **When does it appear?** Run 1–3 across checkpoints 100–600 for one θ_C seed, against when
   following the line switches on in the eval records. Does the circuit form at the same time?
7. **`ineq` lookup.** θ_C follows the line there too, which needs two hops: index → constraint text →
   cells. Attention from the line's index token to the matching `cN:` line, then on to the cell
   coordinates, would show it.
8. **Bigger models.** Repeat 1–4 on the 1.7B runs once Track A has them.

The proposal also mentions the Jacobian lens (github.com/anthropics/jacobian-lens, which has Qwen
examples) and neuron-level circuit work. Both are worth a try after the probes and patching give us
known answers to check them against.

## Practical

- Use a GPU node interactively (`srun --gres=gpu:a40:1 --pty bash` on `general-gpu-long`), not the
  login node. A 0.6B forward pass on a few hundred prompts takes seconds.
- Check whether TransformerLens loads Qwen3. If it doesn't, plain PyTorch forward hooks on the HF model
  (or nnsight) are enough for patching and probes.
- Save results under `results/interp/` with a dated entry in `docs/runlog.md`, like the training runs.

## Prompt

Paste everything below the line into a Claude Code session started in the repo checkout on the cluster.

---

I'm starting the mechanistic-interpretability part of our CSE 5100 project. Read `docs/prompts/interp.md`
(the notes above this prompt), `docs/next.md`, and the top two entries of `docs/runlog.md`, then look at
`src/fbrl/grid.py` for how prompts are built.

Start with question 1 (I(θ) probes) on `distinct` only. That's what answers H1.
- Collect last-prompt-token residuals for about 2000 line-free states from θ_0 and seed 0 of θ_A,
  θ_C and θ_D (`final`).
- Fit a logistic probe per column bit, per layer, with held-out instances.
- Sanity check first: the bits should be decodable from θ_0 well above the base rate at some layer. If
  nothing decodes anywhere, check the setup before reading anything into model differences.

Show me probe accuracy by layer per model. Then do question 3 (patching) the same way: check that the
metric separates θ_C from θ_A before patching, then show a layer × position heatmap and the top heads. Write analysis code as a
small module under `src/fbrl/` (or `interp/`), keep plots and numbers in `results/interp/`, and commit
as you go. Talk to me in plain prose, say what didn't work, and don't overclaim from one seed: once a
result looks real, repeat it on a second seed. Ask before submitting any batch jobs.
