# Notes and prompt: inside the trained models

The behavioral results say *what* θ_C does: it edits wherever the line points, and without a line it's
as good as θ_A. Interp asks *how*. The questions below are ordered so each one is useful on its own,
and the first two need only a forward pass and a few hundred prompts.

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

1. **Where does θ_C read the line?** Take true/false prompt pairs that differ only in the line's index
   token, and use the logit difference between the true and the false column's digit at the edit's
   column position as the metric.
   - Activation patching, residual stream at the line token per layer, then head outputs, clean into
     corrupt. Find the layers and heads that carry the index to the answer.
   - Do the same for θ_A and θ_D. We expect nothing there, which is a check the method works.
2. **Does the model still work out the broken constraint itself?** Fit a linear probe at the last
   prompt token for "which column is violated" (4-way on `distinct`), layer by layer, with no line in
   the prompt. Compare θ_0, θ_A, θ_C, θ_D.
   - H1 in mechanistic form: does θ_C still compute V(s) internally, or did it stop?
   - Then put the false line in: does θ_C's internal V get overwritten, or computed and ignored?
3. **Is θ_C's no-line path the same as θ_A's?** All the models are small fine-tunes of θ_0, so the
   residual streams should be comparable.
   - Train the probe from 2 on θ_A and test it on θ_C.
   - Patch on no-line prompts, corrupting one grid cell, to find the heads that carry the duplicate to
     the answer in each model.
4. **When does it appear?** Run 1 and 2 across checkpoints 100–600 for one θ_C seed. The eval logs
   already show when following the line switches on behaviorally (`fbrl.placement` on each
   checkpoint's eval, or the per-step eval records). Does the circuit form at the same time?
5. **`ineq` lookup.** θ_C follows the line there too, which needs two hops: index → constraint text →
   cells. Attention from the line's index token to the matching `cN:` line, then on to the cell
   coordinates, would show it.
6. **Bigger models.** Repeat 1–3 on the 1.7B runs once Track A has them (`docs/next.md`).

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

Start with question 1 on `distinct` only:
- Load θ_C seed 0 (`final`) and θ_A seed 0, and build 200 true/false prompt pairs.
- Check the metric before patching. θ_C's true-minus-false logit difference should be large and
  positive, and θ_A's near 0. If it isn't, stop and tell me.
- Then patch layer by layer, then head by head.

Show me a layer × position heatmap and the top heads before going further. Write analysis code as a
small module under `src/fbrl/` (or `interp/`), keep plots and numbers in `results/interp/`, and commit
as you go. Talk to me in plain prose, say what didn't work, and don't overclaim from one seed: once a
result looks real, repeat it on a second seed. Ask before submitting any batch jobs.
