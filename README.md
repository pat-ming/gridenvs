# GridEnvs

Environments and experiment code for *Does RL Build a Circuit the Environment Already Runs?*
Post-training here means taking a pretrained language model (Qwen3-0.6B) and continuing to train
it with RL (multi-turn GRPO). The model plays an environment and learns from the reward.

The environments are in `4x4Env/`. `SudokuEnv/` is an empty placeholder and `4x4Env/env.py` is an
unused scratch file. The experiments don't use either of them.

## Setup

```bash
conda activate 5100fin
pip install -r requirements.txt
cd 4x4Env                    # run every command below from here
python -m pytest tests -q    # needs the Qwen3 tokenizer in your Hugging Face cache
```

## The task

An instance is a 4x4 grid of integers 0–9 plus `K` constraints. The grid starts from a valid
solution corrupted by `m` random edits, so it can be solved in `m` edits or fewer.

- **Action:** one reply per turn, containing `EDIT r c v`. This sets `g[r][c] = v`. If a reply has
  several edits, the last one counts. An invalid reply uses up the turn and changes nothing.
- **Reward:** `R = 1` if every constraint holds after turn `H`, otherwise 0. There is no
  intermediate reward. Solving early and then breaking the grid scores 0, and episodes always run
  for all `H` turns.

## The environments

All of them are one class, `GridEnv(mode, H, seed, noise_sizes)`. Everything is shared except the
feedback line after the grid.

| Mode | Name | Used for | Line after the grid |
|---|---|---|---|
| `"none"` | E_A | Training θ_A | None |
| `"true"` | E_C | Training θ_C | `violated: {0, 4}`, the true V(s) |
| `"noise"` | E_D | Training θ_D (control) | `violated: {□, □}`, independent of the state |
| `"counterfactual"` | None | Evaluation only | `violated: {0, 7}`, false but the same size |

For example, a first prompt in E_C (with only two of the constraints shown):

```text
Constraints:
c0: row 0 sums to 10
c8: cell (0,1) < cell (2,3)
Values are integers 0-9. Rows and columns are numbered 0-3.
You have 4 edits. Reply with exactly one edit per turn: EDIT r c v

3 1 4 2
0 5 2 3
4 1 0 5
3 3 4 0
violated: {0, 8}
```

### How to use each one

**E_A (`none`): the policy has to work out V(s) itself.** Train with `--env A`. This is where H1
predicts a verifier circuit will form.

**E_C (`true`): the policy can read V(s).** Train with `--env C`. The line is a deterministic
function of the grid, so E_C contains no extra information. It only saves the computation. This is
where H2 predicts a reader rather than a verifier.

**E_D (`noise`): separates context length from information.** Train with `--env D`. Calibrate first
so the line length matches E_C, because the default (uniform over 0..K) doesn't. The training and
evaluation scripts refuse to run this mode without a calibration file.

```bash
python calibrate_noise.py --n 200 --out noise.json
```

The filler is `□` because it takes as many tokens as a digit under the Qwen3 tokenizer. With `#`
the line comes out shorter, because `#,` is one token. A test checks this. For the same reason
noise mode only allows `K <= 10`, which keeps every index to a single digit.

**Counterfactual: never train on it.** It swaps one true index for a false one, or adds or drops
an index when V is empty or full. Use `counterfactual_score` to see which way edits went: a positive
`score` means they leaned toward the false line, a negative one toward the true violations.

### Constraint sets

Pass one with `--set`. Indices `k` keep the same meaning within a set, which the per-index probes
rely on.

| Set | K | Contents | Notes |
|---|---|---|---|
| `spec` | 10 | 4 row sums, 4 distinct columns, 2 pairwise inequalities | **Default, and the formulation's task family** |
| `rows`, `cols` | 4 | Sums to 10 | |
| `rows+cols` | 8 | Both | |
| `distinct` | 4 | Columns all different | |
| `ineq` | 6 | Pairwise inequalities | |
| `desc` | 1 | Grid is non-increasing | Avoid for the main results: V stays `{0}` until solved |

Sum constraints have a catch for H2: the line says which row is wrong but not by how much, so even
a pure reader has to add up the row to choose a value. That could make V(s) decodable in θ_C for a
reason unrelated to verifying. Comparing against `distinct` and `ineq`, which don't need this, can
help separate the two explanations.

## Using an environment in your own loop

```python
from grid_env import GridEnv
from instances import make_instance

env = GridEnv(mode="true", H=4, seed=0)
obs = env.reset(make_instance("spec", m=2, seed=123))
done = False
while not done:
    reply = my_policy(obs)                        # any text; must contain EDIT r c v
    obs, reward, done, info = env.step(reply)     # reward is nonzero only on the last turn

info["log"]     # per turn: V(s_t), what the line claimed, the reply, the parsed action
info["solved"]  # whether all constraints hold now
```

`GridEnv` only handles strings, so it works with any model. For Qwen3, use `llm.rollout`, which
also handles these details that are easy to get wrong:

- **Thinking is off.** Otherwise the model can compute V(s) in text, where the probes can't see it.
- **Sampling is the plain softmax.** Qwen's defaults (`temperature=0.6, top_k=20`) sample from a
  different distribution than the one GRPO's log-probs describe.
- **Tokens are tracked rather than re-rendered.** `Conversation` records which tokens were
  generated and which belong to the feedback line. Training and head ablation both need this.

## Running the experiment

Use the same values for `--set`, `--m` and `--H` in every command, and the same training
arguments for A, C and D, so the optimization budget matches. The defaults are `spec`, `m` drawn
from 1, 2 and 3, and `H=4`.

**1. Calibrate the noise, then measure θ_0.** Check the baseline before spending compute on
training. A policy gradient on a terminal reward learns nothing from groups whose episodes all tie,
so if θ_0 almost never succeeds, start with an easier `--m` or a simpler `--set`.

```bash
python calibrate_noise.py --n 200 --out noise.json
python evaluate.py --model Qwen/Qwen3-0.6B --label theta_0 --noise-sizes noise.json --out eval/theta_0.json
```

**2. Train.** The formulation targets at least 3 seeds per environment.

```bash
for X in A C D; do for S in 0 1 2; do
  python train_grpo.py --env $X --seed $S --noise-sizes noise.json --out runs/$X/$S
done; done
```

Each run writes `metrics.jsonl` (success, parse rate, KL, timing) and a checkpoint in its folder.

**3. Evaluate under all four conditions.**

```bash
for X in A C D; do for S in 0 1 2; do
  python evaluate.py --model runs/$X/$S --label theta_$X --seed $S --noise-sizes noise.json --out eval/theta_$X-$S.json
done; done
python report.py table --evals 'eval/*.json'
```

In the table, `*` marks conditions the model didn't train on. Those are the primary ones.

**4. Fit probes.** Every model gets the same scripted, feedback-free prompts, so differences can't
come from models visiting different states.

```bash
python probes.py --model Qwen/Qwen3-0.6B --label theta_0 --out probes/theta_0
for X in A C D; do for S in 0 1 2; do
  python probes.py --model runs/$X/$S --label theta_$X --seed $S --out probes/theta_$X-$S
done; done
python report.py probes --probes 'probes/*/probes.json'
```

Probes read the residual stream at the last prompt token, just before the reply starts. `r`, `c`
and `v` are chosen a few tokens later, so a signal that only appears there won't be measured.

**5. Run the interventions.** Run each one on the matching θ_0 too, since that's a required control.

```bash
# H1: remove the probe directions from θ_A (compared with random directions and unrelated prompts)
python interventions.py project --model runs/A/0 --label theta_A --probes probes/theta_A-0 --out iv/project_A0.json

# Patch the probe component from a true-feedback run into a counterfactual run
python interventions.py patch --model runs/C/0 --label theta_C --probes probes/theta_C-0 --out iv/patch_C0.json

# H2: block attention from the violated line
python interventions.py heads --model runs/C/0 --label theta_C --out iv/heads_C0.json
```

`heads` reports two versions of "all heads". `decision` blocks only the edit position, as H2 states
it, but the line can still get through via the tokens between it and the edit.
`all_later_tokens` closes those routes too, so the decision can't depend on the line through
attention at all. A test verifies this.

**6. Test recoverability (H3).** This classifies θ_A against θ_C per instance, holding out one seed
at a time. It needs at least two seeds of each.

```bash
python report.py recover --evals 'eval/*.json' --probes 'probes/*/probes.json'
```

## Files in `4x4Env/`

| File | Purpose |
|---|---|
| `constraints.py` | Constraint types and `violations()`, the single source of truth for V(s) |
| `instances.py` | Instance generation and constraint sets |
| `grid_env.py` | `GridEnv` and `counterfactual_score` |
| `llm.py` | Model loading, `Conversation`, `rollout`, token log-probs |
| `calibrate_noise.py` | Records \|V\| from θ_0 for E_D |
| `train_grpo.py` | Multi-turn GRPO |
| `evaluate.py` | Success under the four conditions |
| `probes.py` | Per-layer linear probes |
| `interventions.py` | Projection, patching, head ablation and their controls |
| `report.py` | Condition table, probe table, recoverability |
| `tests/` | Environment and pipeline tests |

## Not included

- **The proof of Proposition 1**, which is written work rather than code.
- **The from-scratch transformer setting**, which the formulation lists as secondary, if compute
  permits.
