# Prompt: bigger models

For Motti. Paste everything below the line into a Claude Code session started in the repo checkout on
the cluster (`/engrfs/project/netanel.raviv/motti_k/fbrl/code`, branch `training`).

---

I'm working on our CSE 5100 project: GRPO on small constraint-repair puzzles under different feedback
conditions. The questions:
- H1: does training without the feedback line build independent violation detection?
- H2: does training with it build reliance on it?

Read these first:
- the proposal (`CSE 5100 Project Proposal - Moti & Patrick.tex`);
- `docs/next.md`;
- the top three entries of `docs/runlog.md`;
- `docs/training.md` and `docs/envs.md`.

**Where things stand.** On the easy 4x4 sets (`distinct`, `ineq`) with Qwen3-0.6B-Base, the recipe in
`configs/experiments/grpo_kl.yaml` trains stably: lr 1e-5, KL `beta` 0.04, 600 steps. Results so far:
- θ_C (trained with the true line) follows a false line about 100% of the time.
- θ_C repairs as well as θ_A with no line.

I want to know whether that holds as the model grows.

**Bigger models.** Go up the ladder one step at a time, and only move on when a step reproduces:
1. Qwen3-1.7B-Base (`configs/models/qwen3-1.7b.yaml`).
2. Qwen3-4B-Base.
3. Stop there unless 1.7B or 4B shows something 0.6B didn't. Going past 4B means new multi-GPU training
   code (see the memory notes below), which is days of infrastructure work.

At each step:
- SFT a warm start and check `stopped`/`parse`.
- Run `grpo_kl` on `distinct`/`ineq`, 3 seeds per arm.
- Compare against 0.6B with the same measures.

Expect to retune:
- lr: try 5e-6 if 1e-5 is unstable.
- Memory, at each size:
  - 1.7B: one run per 48 GB GPU, packed two only after checking memory.
  - 4B: one run per 80 GB A100.
  - 8B: the optimizer state no longer fits on one GPU (full-weight Adam at 8B is about 130 GB). It
    needs DeepSpeed ZeRO or FSDP across the 4-GPU a100-sxm4 nodes, which `cluster/job.sbatch` and
    `fbrl.train_grpo` don't do yet. That's an infrastructure job to scope before launching anything.
  - LoRA runs behave differently from full fine-tunes, so don't put them in the same comparison
    without saying so.

**How to judge results.**
- Eval success by mode (none / true / counterfactual), averaged over the last five evals.
- `python -m fbrl.placement <run dirs>`, which gives two shares of first edits:
  - `follow_false`: edits inside the constraint a false line names;
  - `find_alone`: edits inside a truly violated constraint, with no line.
- Compare arms over seeds, not single runs.

**How we work.**
- Ask before submitting GPU jobs, and say how many GPUs and for how long. Keep the number of
  experiments small and each one aimed at a question. Never run GPU work on the login node.
- Submitting from inside an interactive job, first run
  `unset $(env | grep -oE "^(APPTAINER|SINGULARITY)_[A-Z_]*")`.
- Keep the "## In flight" section at the top of `docs/runlog.md` current, and write a dated entry when
  results come in. Commit and push.
- Commit messages are one line. Plain prose, say what didn't work, and don't oversell.
