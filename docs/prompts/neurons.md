# Prompt: native neurons, activations and patching

For Motti. Paste everything below the line into a Claude Code session started in the repo checkout on
the cluster.

---

I'm starting the mechanistic-interpretability part of our CSE 5100 project. My side is native neurons,
activations and patching. Patrick is doing the same questions with the Jacobian lens, so we can check
each other.

Read these first:
- `docs/interp.md`, the shared notes: what we have, the shared groundwork, the questions, and who owns
  each;
- the proposal tex in the repo root, §3 and §5;
- `docs/next.md` and the top two entries of `docs/runlog.md`;
- `src/fbrl/grid.py`, for how prompts are built.

1. **Groundwork.** If the shared groundwork in `docs/interp.md` doesn't exist yet, build it first as a
   small module under `src/fbrl/`, and tell Patrick it's there.
2. **Question 1, I(θ) probes, on `distinct` only.** This is what answers H1.
   - Use about 2000 line-free states from θ_0, and seed 0 (`final`) of θ_A, θ_C and θ_D.
   - Fit a logistic probe per column bit, per layer, with held-out instances.
   - Sanity check: the bits should be decodable from θ_0 well above the base rate at some layer. If
     nothing decodes anywhere, check the setup before reading anything into model differences.
   - Show me probe accuracy by layer per model.
3. **Question 3, patching.**
   - First check the metric: θ_C's true-minus-false logit difference should be large, θ_A's near 0.
   - Then patch and show a layer × position heatmap and the top heads.
4. **Then:**
   - question 10, the neurons feeding those heads;
   - question 3's ablation (Δ_M, with a random-ablation control);
   - question 2 (D(θ)).

Use a GPU node interactively, not the login node. Keep plots and numbers in `results/interp/`, write a
dated entry in `docs/runlog.md` when something comes out, and commit as you go. Talk to me in plain
prose, say what didn't work, and don't overclaim from one seed: once a result looks real, repeat it on a
second seed. Ask before submitting batch jobs.
