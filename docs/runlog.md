# Run log

Newest first. Each entry says what ran, what came out, and what we think it means. Run directories are
in `results/runs/`, and everything is also in W&B (project `fbrl`).

## 2026-11-09 · GRPO smoke on `spec`

Qwen3-0.6B (instruct), 10 steps of 128 episodes, `spec` with m ∈ {1, 2}, E_C, eval every 5 steps.
About 25 s per step on an A6000. [W&B](https://forge.coreweave.com/wandb/morevolution-washington-university-in-st-louis/fbrl/runs/mu6ujhjk),
`grpo_smoke_20261110_034424_372909`.

The pipeline works end to end: rollouts, env-masked loss, periodic eval, checkpoint. But 94–100% of
groups had identical rewards every step (`frac_reward_zero_std`), so there was nothing to learn from.
That's the cold baseline below, showing up in training.

## 2026-11-09 · Cold baselines across model sizes

64 held-out instances × 8 samples, temperature 1.0, H = m + {0, 2}, m ∈ {1, 2}. Success in %, averaged
over the four (m, H) cells, as `none → true`. W&B groups `baseline_1109` and `baseline_sizes`.

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

**Open: Base models don't work in this format yet.** Qwen3-0.6B-Base and 1.7B-Base parse 0–2% of
replies, and the replies are junk (random multilingual tokens, not near-misses). That's too broken to be
sampling noise. Next is checking the prompt they actually see (the empty `<think></think>` block, the
chat template) and trying a lower temperature or top-p. 4B-Base and 8B-Base are still running.

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
[W&B](https://forge.coreweave.com/wandb/morevolution-washington-university-in-st-louis/fbrl/runs/rbc3oto1).
It's superseded by the plain-text `rollout_func` path, since the env feeds observations back
automatically rather than through tool calls.
