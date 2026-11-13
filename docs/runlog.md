# Run log

Newest first. Each entry says what ran, what came out, and what we think it means. Run directories are
in `results/runs/`, and everything is also in W&B (project `fbrl`).

## 2026-11-13 · Oracle share in the SFT warm start

Same recipe as the 11-10 warm start, but with 0.5 and 1.0 of the demo games played by the oracle
instead of 0.25. The question was whether stronger demos give `spec` a GRPO signal without teaching
reliance on the feedback line. Same eval as before. W&B tag `oracle-sweep`, SFT
[0.5](https://forge.coreweave.com/wandb/cse-5100-fall-2026/fbrl/runs/62qua5ra) /
[1.0](https://forge.coreweave.com/wandb/cse-5100-fall-2026/fbrl/runs/7cj7v58l), eval
[0.5](https://forge.coreweave.com/wandb/cse-5100-fall-2026/fbrl/runs/0uytujh1) /
[1.0](https://forge.coreweave.com/wandb/cse-5100-fall-2026/fbrl/runs/eg1rh6l5), checkpoints in
`fbrl/models/sft-qwen3-0.6b-oracle{0.5,1.0}`. Success in %, none / true / residual, averaged over the
four (m, H) cells, and the best cell's share of mixed groups.

| Oracle share | `spec` | `rows` | `ineq` | `distinct` |
|---|---|---|---|---|
| 0.25 | 0.2 / 0.1 / 0.3, 9% | 0.5 / 0.6 / 0.2, 12% | 5.2 / 4.0 / 5.5, 45% | 9.1 / 10.7 / 11.5, 75% |
| 0.5 | 0.1 / 0.2 / 0.6, 12% | 0.6 / 0.7 / 0.5, 12% | 4.7 / 4.6 / 4.5, 39% | 8.3 / 8.3 / 9.2, 62% |
| 1.0 | 0.3 / 1.6 / 1.8, 34% | 0.8 / 4.2 / 3.6, 48% | 4.9 / 4.2 / 3.8, 44% | 7.2 / 5.8 / 6.3, 47% |

- 0.5 changes nothing. It's within noise of 0.25 everywhere.
- 1.0 does open up `spec` and `rows`, but only with the line. On `spec` at m=1, H=1, success goes
  1.2% without the line to 4.9% with it (34% mixed groups vs 9%); `rows` at m=1, H=1 goes 2.0 to 8.0.
  Without the line, it's no better than 0.25. So the signal it buys is exactly the reliance we were
  worried about: all-oracle demos teach "read V(s), edit there."
- 1.0 also breaks parsing, but only when there's a line: 80–84% on `ineq` with true or residual,
  83–96% on `spec`, and 98–100% with none. Probably a line shape the demos never showed it; not checked yet.
- `distinct` gets worse as the oracle share goes up (9.1 → 7.2 with none, 75% → 47% mixed). Less
  random-edit coverage seems to cost exploration on the easy set.
- Train loss barely moves (1.63 / 1.62 / 1.57), so loss isn't a useful signal for picking the mix.

For H2 this makes 1.0 a bad warm start for the A-vs-C comparison, since the "true" arm starts out
reliant. Keep 0.25 (which `grpo_easy_sets` already uses), and if `spec` needs a signal, get it from a
curriculum rather than from the demos.

## 2026-11-10 · SFT warm start for Qwen3-0.6B-Base

One epoch on 4000 scripted games (25% played by the oracle, the rest random valid edits, modes split
evenly over none / true / noise / residual), loss on reply tokens only. Then the same eval as the
baselines. [SFT](https://forge.coreweave.com/wandb/cse-5100-fall-2026/fbrl/runs/toicvfm8),
[eval](https://forge.coreweave.com/wandb/cse-5100-fall-2026/fbrl/runs/zt3q86vr), checkpoint in
`fbrl/models/sft-qwen3-0.6b`. (The first attempt died on `warmup_ratio`, which transformers 5 dropped.)

| | `spec` | `rows` | `ineq` | `distinct` |
|---|---|---|---|---|
| success %, none / true / residual | 0.3 / 0.1 / 0.3 | 0.5 / 0.6 / 0.2 | 5.2 / 4.0 / 5.5 | 9.1 / 10.7 / 11.5 |
| mixed groups, best cell | 9% | 12% | 45% | 75% |

- The format is fixed: parse went from 0% to 98–100%.
- It lands about where the 0.6B instruct model is, so the warm start did its job and nothing more.
  `spec` still has almost no GRPO signal, while `distinct` and `ineq` have plenty.
- No sign that SFT taught reliance on the line: true vs none is +1.6 points on `distinct`, -1.2 on
  `ineq`, which is noise at this sample size. That was the worry with oracle demos, and it didn't show up.
- On-target edits on `spec` dropped (0.31 vs 0.44 for instruct at m=1, H=1). The 75% random demos
  taught it to edit anywhere, which is fine as a starting point but worth knowing.

## 2026-11-09 · GRPO smoke on `spec`

Qwen3-0.6B (instruct), 10 steps of 128 episodes, `spec` with m ∈ {1, 2}, E_C, eval every 5 steps.
About 25 s per step on an A6000. [W&B](https://forge.coreweave.com/wandb/cse-5100-fall-2026/fbrl/runs/mu6ujhjk),
`grpo_smoke_20261110_034424_372909`.

The pipeline works end to end: rollouts, env-masked loss, periodic eval, checkpoint. But 94–100% of
groups had identical rewards every step (`frac_reward_zero_std`), so there was nothing to learn from.
That's the cold baseline below, showing up in training.

## 2026-11-09 · Cold baselines across model sizes

64 held-out instances × 8 samples, temperature 1.0, H = m + {0, 2}, m ∈ {1, 2}. Success in %, averaged
over the four (m, H) cells, as `none → true`. W&B groups `baseline_1109` and `baseline_sizes`, runs `eval_<model>_…`.

| Model | `spec` | `rows` | `ineq` | `distinct` |
|---|---|---|---|---|
| Qwen3-0.6B | 0.1 → 0.2 | 0.7 → 1.1 | 4.5 → 4.6 | 7.0 → 8.9 |
| Qwen3-1.7B | 0.1 → 0.4 | 0.9 → 2.4 | 3.0 → 7.0 | 3.4 → 6.9 |
| Qwen3-4B | 0.3 → 0.3 | 0.7 → 3.2 | 10.4 → 12.7 | 21.0 → 23.2 |
| Qwen3-8B | 0.1 → 0.5 | 0.6 → 5.5 | 6.4 → 7.5 | 23.6 → 27.5 |

- `spec` (the proposal's set) is below 1% at every size, which is worse than a random policy (1.5% at
  m = 1). GRPO can't start on it cold. It needs the SFT warm start or a curriculum.
- `distinct` and `ineq` have enough signal for GRPO now: up to 66% of instances have mixed outcomes at 8B
  and about 50% at 0.6B.
- The feedback line starts to matter with size. On `rows`, E_C goes from about +0.4 points at 0.6B to
  +5 at 8B. So the bigger models already read the line before any training, which is worth keeping in
  mind for H2 (it's not a clean zero-reliance start for them).
- `residual` is within noise of `true` everywhere. The models don't use the extra arithmetic yet.
- 1.7B is worse than 0.6B on `distinct`, so size isn't monotonic here. Don't read much into one sample
  of 512 episodes per cell.

Base models (0.6B to 8B) parse 0–8% of replies and solve nothing. That's not a pipeline bug, just
base models without the format: 0.6B samples junk at temperature 1.0, and 8B-Base writes coherent text
but continues the conversation ("assistant… user… Constraints: …") or answers some other task. This is
the job of the SFT warm start, which is next.

## 2026-11-09 · Scripted policies on the 4x4 instances

How hard is the task, and what does knowing V(s) buy a policy that can't do arithmetic? 200 instances
per row, H = m.

| Set | m | Solving edits (of 160), m = 1 | Random | Knows V(s), random value | Greedy on \|V\| |
|---|---|---|---|---|---|
| `spec` | 1 | 1.4 | 1.5% | 4.5% | 100% |
| `spec` | 2 | | 0% | 2.0% | 95% |
| `rows` | 1 | 2.6 | 1.5% | 9.0% | 100% |
| `distinct` | 1 | 14.0 | 8.5% | 46.5% | 100% |
| `ineq` | 1 | 5.8 | 2.5% | 28.0% | 100% |

On `spec`, knowing which constraint is broken only triples a random policy's success. Most of the
difficulty is choosing the value, so E_C hands over less of the computation than it seems to. That's
the reason for adding `residual`.

## 2026-10-29 · Toy GRPO smoke (tool-calling)

The first plumbing check, on a placeholder env through TRL's tool-calling interface. 20 steps,
[W&B](https://forge.coreweave.com/wandb/cse-5100-fall-2026/fbrl/runs/rbc3oto1).
It's superseded by the plain-text `rollout_func` path, since the env feeds observations back
automatically rather than through tool calls.
