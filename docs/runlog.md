# Run log

Newest first. Each entry says what ran, what came out, and what we think it means. Run directories are
in `results/runs/`, and everything is also in W&B (project `fbrl`).

## In flight

As of 2026-11-23. Nothing running. Proposed (not approved): rerun `true` s1 and `noise` s1 with a KL
penalty (`beta` 0.04) to see whether it keeps 600 steps stable, see below.

## 2026-11-23 · grpo_arms: a first H2 signal, no H1 signal, and half the runs broke

`grpo_arms` (array 383467, finished 11-15): A / C / D (`none`, `true`, `noise`) from the fixed warm start
on `distinct`/`ineq`, lr 1e-5 constant, `beta` 0, 600 steps, seeds 0 and 1. W&B: none
[s0](https://forge.coreweave.com/wandb/cse-5100-fall-2026/fbrl/runs/kx73lqk2) [s1](https://forge.coreweave.com/wandb/cse-5100-fall-2026/fbrl/runs/o2lrucpy), true [s0](https://forge.coreweave.com/wandb/cse-5100-fall-2026/fbrl/runs/pmryd7rp) [s1](https://forge.coreweave.com/wandb/cse-5100-fall-2026/fbrl/runs/ouf1ecbd), noise
[s0](https://forge.coreweave.com/wandb/cse-5100-fall-2026/fbrl/runs/wtpo99z3) [s1](https://forge.coreweave.com/wandb/cse-5100-fall-2026/fbrl/runs/w4w3jt7d).

Eval success in %, mean of the last five evals (steps 500–600), eval modes none / true / counterfactual,
and `cf_score`. Step 0 for every run is `distinct` 8.6 / 7.8 / 8.2, `ineq` 3.5 / 4.3 / 3.1.

| run | state at the end | `distinct` | cf | `ineq` | cf |
|---|---|---|---|---|---|
| none s0 | healthy | 14.9 / 15.8 / 16.0 | −0.00 | 17.0 / 17.0 / 18.7 | +0.01 |
| none s1 | healthy | 19.1 / 18.5 / 18.6 | −0.02 | 21.6 / 19.3 / 19.7 | −0.00 |
| true s0 | healthy (lost the stop token at steps 100–300, got it back) | 15.7 / **44.4** / **4.3** | **+0.63** | 22.7 / 24.6 / 20.3 | −0.01 |
| true s1 | gibberish from about step 120 | 1.5 / 1.3 / 1.1 | | 0.2 / 0.3 / 0.2 | |
| noise s0 | healthy to step 540 (25% train success), then gibberish | 9.3 / 10.5 / 10.2 | | 9.5 / 9.3 / 9.8 | |
| noise s1 | lost the stop token at step 200, never got it back | 14.0 / 14.0 / 16.8 | −0.03 | 3.0 / 4.0 / 4.1 | +0.03 |

**H2 has its first signal.** θ_C seed 0 solves 44% of `distinct` boards with the true line, 16% with no
line, and 4% with a false one. That's a model that reads the line and acts on it, including when the
line is wrong, which is exactly what `cf_score` was built to catch (+0.63, against about 0 for every
other run). It's one seed, and only on `distinct`; on `ineq` the same model ignores the line.

**H1 has no signal yet.** On the no-line eval, θ_A (15–22%) and the one usable θ_C (16–23%) end up in
the same place. So far training with the line didn't cost the model anything when the line is taken
away, and it didn't help either.

**The real problem is stability.** Three of six runs broke, in two ways:
- *Losing the stop token* (true s0 for a while, noise s1 for good), the same drift as 3e-5 in
  `grpo_lr2`, just slower.
- *Falling into gibberish* (true s1, noise s0): entropy jumps from about 0.2 to 3, replies become random
  tokens, nothing parses, every reward is 0, so the gradient is 0 and the run can never recover.

Both come from nothing anchoring the policy: `beta` is 0, so there's no pull back toward the warm start.
A KL penalty to the warm start (`beta` 0.04 is the usual default) is the standard fix and the first
thing to try. With half the runs unusable we can't compare arms at all, so this comes before more seeds.

## 2026-11-14 · GRPO learns at 1e-5 and collapses at 3e-5

`grpo_lr2` (array 382302): two runs on the `none` arm from the fixed warm start, seed 0, constant lr with
10 warmup steps, 300 steps, otherwise as `grpo_easy_sets`. W&B: [1e-5](https://forge.coreweave.com/wandb/cse-5100-fall-2026/fbrl/runs/v0jure8s),
[3e-5](https://forge.coreweave.com/wandb/cse-5100-fall-2026/fbrl/runs/06zxcuwu).

**1e-5 learns, slowly.** Rollout success goes from about 10% in the first 30 steps to about 15% in the
last 30. The held-out eval moves the same way. Success in %, mean of the last five evals (steps 200–300)
against step 0:

| | `distinct` none / true | `ineq` none / true |
|---|---|---|
| step 0 | 8.6 / 7.8 | 3.5 / 4.3 |
| steps 200–300 | 16.4 / 14.6 | 6.1 / 5.1 |

Each (set, mode) eval is 256 episodes (128 per m), so one eval has a standard error of about 2 points; the averages are
steadier than that. The run stays healthy: `stopped` is 1.0 throughout, `parse` is at least 98.5%, and
reply length stays at about 16 tokens over two turns. Entropy halves (0.66 to 0.32) and `mixed` falls
from about 50% to 42%, so the policy is sharpening, but by step 300 it still has plenty of signal left
and the curve hasn't flattened convincingly. `cf_score` stays within ±0.06 of zero.

**3e-5 collapses.** The early climb I saw in W&B didn't last. The model stops ending its replies within
50 steps (`stopped` 1.0 to 0.45 in steps 25–49, then 0), so every reply runs to the 32-token cap again.
By step 125 most replies are an `EDIT` followed by a run of `9`s. Entropy falls to about 0.006 by step
210, `mixed` to 0, and the gradient to 0: every rollout in a group is the same, so there's nothing left
to learn from. Eval success settles at exactly 20.3% on `distinct` in all three modes, which is one
fixed edit sequence that happens to solve that share of boards, not a skill.

What I take from it:
- 1e-5 is the step size to use. It's ten times the `grpo_easy_sets` lr, which didn't learn, and three
  times below where it falls apart.
- With `beta` at 0 nothing holds the stop token in place, so at larger steps the model can lose it
  again. The tail after the `EDIT` carries the same advantage as the edit, so it drifts freely. `stopped`
  is the first thing to watch on any new run.
- On the `none` arm, eval `none` ends up a point or two above `true`. That's the direction H1 predicts,
  but it's one seed of one arm and within noise; it needs the C and D arms to mean anything.
- Gains are small in absolute terms (`distinct` 16%, `ineq` 6%). Longer runs or more seeds may be worth
  more than further lr tuning.

## 2026-11-14 · Warm start with the `<|im_end|>` fix

Same 0.25 SFT as 11-10, but with `<|im_end|>` started as a copy of `<|endoftext|>` (2010aea), then the
usual eval. [SFT](https://forge.coreweave.com/wandb/cse-5100-fall-2026/fbrl/runs/ujchhj42), checkpoint in
`models/sft-qwen3-0.6b` (replacing the broken one). Final train loss 0.70, against 1.47 before.

The replies are clean now: 99.9% of 61,440 eval replies are exactly `EDIT r c v`, against 0.1% before.
About 60% end on `<|im_end|>` and 40% on `<|endoftext|>`. The two started out identical and SFT didn't
fully separate them. vLLM stops on both, so `stopped` now counts both (58c0578); the gate I'd set
(`stopped` > 0.95 on `<|im_end|>` alone) was the wrong test.

Success in %, none / true / residual, averaged over the (m, H) cells, and the best cell's mixed groups:

| | `spec` | `rows` | `ineq` | `distinct` |
|---|---|---|---|---|
| success | 0.3 / 0.1 / 0.4 | 0.5 / 0.6 / 0.4 | 5.4 / 4.2 / 4.5 | 9.8 / 9.2 / 9.8 |
| mixed, best cell | 12% | 11% | 45% | 59% |

- Success is about where the broken warm start was, so the junk wasn't what held success down at the
  start. It mattered for GRPO, where it diluted the update and sometimes overrode the edit.
- `distinct` has fewer mixed groups (59% vs 75%), but that's still plenty of signal for GRPO.
- The line still doesn't help (true vs none within a point everywhere), so no reliance going in.

## 2026-11-14 · The SFT warm start never learned to stop

While reading the `grpo_lr` sweep (array 382173, cancelled about halfway), the 3e-5 run's entropy fell
from about 3.5 to 0.09 at step 50. The rollouts showed why. Every reply is `EDIT r c v` followed by junk
until the 32-token cap, and at 3e-5 the junk collapsed into one repeated token. The edits themselves
stayed varied (7.8 distinct action sequences per group of 8).

The SFT checkpoint already did this: only 0.1% of its eval replies were a clean `EDIT r c v`. The
cause is the stop token. In Qwen3-0.6B-Base, `<|im_end|>` has an embedding norm of 0.35 against a
median of 0.97, so it was barely trained, and after our SFT it was 0.349. One epoch at 1e-5 moves a
weight by about 1e-3, which can't grow it. The demos did put `<|im_end|>` in the loss; the model just
couldn't learn to predict it.

What this broke:
- `parse` hid it. It only asks for an `EDIT` somewhere in the reply, so it read 98–100% throughout.
- The env takes the last `EDIT` in a reply, so junk that happens to contain another one replaces the
  model's edit with a random one.
- GRPO's update was spread over about 25 junk tokens per turn for every 5 that mattered, which may be
  why `grpo_easy_sets` didn't learn.
- `completions/clipped_ratio` (about 0.75) was real truncation. The earlier note calling it cosmetic
  was wrong.

So every result built on the warm start is invalid: the 11-10 SFT and its eval, the oracle-share
sweep, `grpo_easy_sets` and `grpo_lr`. Their run dirs are in `runs/_archive/untrained-im-end/` (and
`results/runs/_archive/…`), the checkpoints in `models/_archive/untrained-im-end/`, and the W&B runs
carry the tag `invalid-untrained-im-end`. The cold baselines and scripted policies are unaffected:
the instruct models' `<|im_end|>` is trained, and the base models were reported as not following the
format anyway.

The fix (2010aea): before SFT, if `<|im_end|>` is undertrained, copy the `<|endoftext|>` row into it,
since that's the token the base model already uses to stop. Starting it at the mean embedding doesn't
work, because the mean has an even smaller norm (0.29). Eval and rollouts now also log `stopped`, the
share of replies that end on the stop token, so this can't hide behind `parse` again.

## 2026-11-13 · `grpo_easy_sets`: first GRPO comparison across feedback modes

**Invalid**, see 2026-11-14: the warm start never learned to stop, so these numbers are off.

GRPO from the 0.25 warm start (`sft-qwen3-0.6b`) on `distinct` + `ineq` (m ∈ {1, 2}, slack 0 or 1),
one run per feedback mode and seed, 300 steps of 128 episodes, lr 1e-6 decaying linearly to 0, `beta` 0.
Eval every 25 steps on 32 held-out instances × 4 samples per cell. W&B group `grpo_easy_sets`,
run dirs `grpo_sft-qwen3-0.6b_<mode>_s<seed>_*`. (Six tasks of the first array died at step 0 on a
MASTER_PORT collision, fixed in 1010f09, and were rerun; their dirs say `failed`.)

GRPO didn't learn in any arm. Training success is about 7% at the start and end of every run, and
about 40% of groups stay mixed throughout, so there was signal to learn from. It just didn't move the
policy. Final eval, success in % averaged over the two (m, H) cells, seed 0 / seed 1:

| Mode | `distinct` none | `distinct` true | `ineq` none | `ineq` true | cf_score, `distinct` |
|---|---|---|---|---|---|
| step 0 | 10.2 | 9.8 | 1.6 | 5.9 | +0.08 |
| none | 12.9 / 9.4 | 9.8 / 11.3 | 5.1 / 2.0 | 5.1 / 2.3 | +0.02 / -0.05 |
| true | 12.1 / 9.4 | 9.8 / 10.9 | 4.7 / 3.1 | 3.1 / 3.1 | -0.00 / -0.02 |
| noise | 10.9 / 10.9 | 9.0 / 9.4 | 5.1 / 2.0 | 3.9 / 1.6 | +0.01 / -0.06 |

- With nothing learned, H1 and H2 can't be answered from this sweep. The arms differ by less than the
  seeds do, and cf_score sits around 0 everywhere.
- `residual` is left out because it's identical to `true` here, bit for bit, for the same seed.
  The residual line only adds the shortfall for sum constraints, and `distinct` and `ineq` have none.
  So the sweep was really three arms. Residual only means something on `spec` and `rows`.
- The likely reason it didn't learn is the step size. 1e-6, decaying to 0 over 300 steps, is small for
  a 0.6B model, and the policy barely moved (rollout success and eval never leave the step-0 band).
  Next would be a short lr sweep on one arm (say `none`, 3e-6 to 1e-5, constant lr) before rerunning the
  comparison. Not launched, because of the hold.
- `harness compare --group-by env.mode` counts the 0-step failed runs in `n` (they have a
  `summary.json`), so it reports n=3–4 for two real seeds. The means are still right, since failed
  runs contribute no values.

## 2026-11-13 · Oracle share in the SFT warm start

**Invalid**, see 2026-11-14: the warm start never learned to stop, so these numbers are off.

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

**Invalid**, see 2026-11-14: the warm start never learned to stop, so these numbers are off.

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
