# Task: J-lens and harder environments

Paste this into Claude Code in the repo checkout (branch `training`):

---

Read `docs/next.md` for where the project stands. This task has two parts:
- `docs/prompts/environments.md`: make `spec`/`rows` learnable and get Sudoku ready to train.
- `docs/prompts/jlens.md`: the Jacobian lens on our trained models. Shared notes are in
  `docs/interp.md`.

What the 4x4 runs taught us, which should carry over to new sets and Sudoku:
- **Recipe that trains stably:** `configs/experiments/grpo_kl.yaml` on top of `grpo_easy_sets.yaml`.
  That's lr 1e-5 constant with 10 warmup steps, KL `beta` 0.04, 600 steps, and 8 generations per
  prompt.
- **Too high a learning rate breaks training:** 3e-5 collapsed within 50 steps. 1e-6 learned nothing.
- **The KL penalty matters:** with `beta` 0, 3 of 6 runs broke. Some lost the stop token and ran on to
  the length cap; others fell into gibberish and never recovered. With 0.04, all 12 were stable.
- **What to watch on any new run:**
  - `rollout/stopped` should stay near 1 and entropy shouldn't climb. Both problems showed up there
    first, while `parse` still looked fine.
  - `mixed` (the share of instances whose samples disagree) is the learning signal. Sets the warm
    start solves under about 1% of the time have almost none.
- **New warm starts:** Qwen base models ship an untrained `<|im_end|>`. `fbrl.train_sft` fixes it; check
  that the run logs `init_end_of_reply`.
- **Fixed constraint numbers get memorized:** on `distinct`, "violated: {3}" always means column 3, and
  the model learned that mapping without reading the constraint list.

Read both, then start with the first step of `environments.md`. Measure the current warm start on
`spec`/`rows` so we have a baseline, and report what you find before changing anything. Ask before
submitting GPU jobs.
