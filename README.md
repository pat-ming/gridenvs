# GridEnvs

Environments for *Does RL Build a Circuit the Environment Already Runs?* In LLM post-training, a
pretrained model plays an environment over several turns and learns from the reward. These
environments only handle text: your training code sends in the model's reply and gets back the
next observation, so they work with any model.

## Setup

```bash
pip install -e . pytest      # just the environments and harness
python -m pytest tests/test_grid4.py tests/test_sudoku.py -q
# or, with the training stack: uv sync && uv run pytest -q
```

The environments are the `gridenvs` package in `src/`. There are two: [`gridenvs.grid4`](#4x4env-the-task)
and [`gridenvs.sudoku`](#sudokuenv). Both offer
the same four modes and return observations the same way.

## 4x4Env: the task

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
You have 4 edits. Each turn, reply with only one line of the form EDIT r c v (row r, column c, new value v), with no other text. Example reply: EDIT 2 1 7

3 1 4 2
0 5 2 3
4 1 0 5
3 3 4 0
violated: {0, 8}
```

### How to use each one

**E_A (`none`): the policy has to work out V(s) itself.** Use it to train θ_A.

**E_C (`true`): the policy can read V(s).** Use it to train θ_C. The line is a deterministic
function of the grid, so E_C contains no extra information. It only saves the computation.

**E_D (`noise`): separates context length from information.** Use it to train θ_D. Pass
`noise_sizes`, a list of set sizes to sample from, such as the `len(shown)` values logged during
E_C rollouts. Without it, the size is uniform over 0..K, which doesn't match E_C's length. The size
is always drawn independently of the state, so the line doesn't leak |V|.

**Counterfactual: evaluation only.** It swaps one true index for a false one, or adds or drops an
index when V is empty or full. `counterfactual_score(info["log"], inst["constraints"])` shows which
way edits went: a positive `score` means they leaned toward the false line, a negative one toward
the true violations.

### Constraint sets

Pass one to `make_instance`. Indices `k` keep the same meaning within a set.

| Set | K | Contents | Notes |
|---|---|---|---|
| `spec` | 10 | 4 row sums, 4 distinct columns, 2 pairwise inequalities | The formulation's task family |
| `rows`, `cols` | 4 | Sums to 10 | |
| `rows+cols` | 8 | Both | |
| `distinct` | 4 | Columns all different | |
| `ineq` | 6 | Pairwise inequalities | |
| `desc` | 1 | Grid is non-increasing | V stays `{0}` until solved, so feedback says little |

Sum constraints have a catch for H2: the line says which row is wrong but not by how much, so even
a pure reader has to add up the row to choose a value. Comparing against `distinct` and `ineq` can
help separate the two explanations.

## Using an environment in a training loop

```python
from gridenvs.grid4.grid_env import GridEnv
from gridenvs.grid4.instances import make_instance

env = GridEnv(mode="true", H=4, seed=0)
inst = make_instance("spec", m=2, seed=123)
obs = env.reset(inst)              # first user message: constraints, instructions, grid
done = False
while not done:
    reply = my_policy(obs)                        # any text containing EDIT r c v
    obs, reward, done, info = env.step(reply)     # later observations hold only the grid and line

info["log"]     # per turn: V(s_t), what the line claimed, the reply, the parsed action
info["solved"]  # whether all constraints hold now
```

Keep held-out instance seeds separate from training seeds. `env.line` holds the exact feedback text
for the current observation, which helps if you need to locate those tokens.

### Things to watch when connecting a model

These came up when trying the environments on Qwen3-0.6B:

- **The prompt wording matters.** An earlier wording without an example reply got about 0% valid
  edits, because the model copied `EDIT r c v` literally. The current wording got 80–100%.
- **Turn thinking off.** Otherwise the model can compute V(s) in text instead of internally.
- **Set sampling explicitly.** Qwen's defaults (`temperature=0.6, top_k=20`) aren't the plain softmax
  that GRPO's log-probs assume.
- **`spec` is hard from a cold start.** With the current wording, Qwen3-0.6B had 0 successes in 64
  episodes per setting tried, with the feedback line on and off. A terminal reward gives GRPO nothing
  to learn from if every episode ties.

## Files in `src/gridenvs/grid4/`

| File | Purpose |
|---|---|
| `constraints.py` | Constraint types and `violations()`, the single source of truth for V(s) |
| `instances.py` | Instance generation and constraint sets |
| `grid_env.py` | `GridEnv` and `counterfactual_score` |
| `tests/test_grid4.py` | Environment tests |

## SudokuEnv

An instance is a solved 9x9 Sudoku board corrupted by `m` random writes, so it can be solved in `m`
writes or fewer. There's one constraint set, with `K = 27`: every row, column and box must hold each
digit 1–9 exactly once.

| Index `k` | Constraint |
|---|---|
| 0–8 | Rows 0–8 |
| 9–17 | Columns 0–8 |
| 18–26 | Boxes 0–8, left to right and then top to bottom |

- **Action:** one reply per turn, containing `WRITE v r c`, which writes digit `v` at row `r`, column
  `c`. Note the order: the value comes first, unlike `EDIT r c v` in the 4x4 grid. Any cell can be
  overwritten. If a reply has several writes, the last one counts. An invalid reply uses up the turn
  and changes nothing.
- **Reward:** the same as the 4x4 grid. `R = 1` if the whole board is valid after turn `H`, otherwise 0,
  with no intermediate reward. Use `H >= m`, or some instances can't be solved.

The four modes are the ones in [the table above](#the-environments), and you use them the same way.
There's one difference: indices can have two digits here, so a noise item repeats the filler to
match (`□` or `□□`), which keeps the line's token length matched to E_C.

```python
from gridenvs.sudoku.instances import make_instance
from gridenvs.sudoku.sudoku_env import SudokuEnv

env = SudokuEnv(mode="true", H=4, seed=0)
inst = make_instance(m=2, seed=123)
obs = env.reset(inst)              # first user message: constraints, instructions, board
done = False
while not done:
    reply = my_policy(obs)                        # any text containing WRITE v r c
    obs, reward, done, info = env.step(reply)     # later observations hold only the board and line

info["log"]     # per turn: V(s_t), what the line claimed, the reply, the parsed action
info["solved"]  # whether the board is valid now
```

The board looks like this at the end of a first prompt in E_C:

```text
8 5 4 6 7 9 2 1 3
7 9 6 1 3 4 5 4 8
3 2 1 4 8 5 9 6 7
1 3 5 9 4 8 7 2 6
6 7 2 5 1 3 8 9 4
4 8 9 2 6 7 3 5 1
2 6 3 8 5 1 4 7 9
5 1 8 7 9 4 6 3 2
9 4 7 3 2 6 3 8 5
violated: {1, 8, 14, 15, 19, 26}
```

Boards have no blank cells: every cell holds a digit, and some of them are wrong. That keeps finding
V(s) a real computation, as in the 4x4 grid, because a blank would mark the broken units on sight. This
environment hasn't been tried with a model yet, so the prompt wording may need the same tuning.

Files are in `src/gridenvs/sudoku/`.

| File | Purpose |
|---|---|
| `constraints.py` | `Row`, `Col`, `Box` and `violations()`, the single source of truth for V(s) |
| `instances.py` | Solved-board generation and corruption |
| `sudoku_env.py` | `SudokuEnv` and `counterfactual_score` |
| `tests/test_sudoku.py` | Environment tests |

## Training and evaluation

The training side is a uv project in `src/`:

```text
src/gridenvs/   the environments above
src/fbrl/       model <-> env play loop, eval, SFT warm start, GRPO
src/harness/    experiment tracking, sweeps, comparisons (see src/harness/README.md)
configs/        base configs, plus configs/models/ presets per model size
cluster/        scripts for shell.engr.wustl.edu
results/runs/   small files from every cluster run (config, meta, metrics, eval cells), via `cluster/remote.sh pull`
```

`fbrl.grid` plays an environment in plain text: the observation goes in as a user message, the model's
reply goes to `env.step`, for exactly `H` turns. An episode is one token sequence with a mask over the
model's reply tokens, so eval, SFT, GRPO and probing all see the same tokens. Evaluation instances use
seeds from `EVAL_SEED_START` up; SFT and GRPO seeds stay below it.

`fbrl.grid` also adds a `residual` mode to `GridEnv`: E_C's line plus how far each violated sum is from
its target (`violated: {0 (+3), 4}`). With `none`, `true` and `residual` the feedback offloads nothing,
the search for the broken constraint, or that plus the arithmetic.

| Module | What it does |
|---|---|
| `fbrl.evaluate` | success, `mixed` (share of instances whose samples disagree, i.e. GRPO signal), parse rate, on-target edits and counterfactual score, per (instance params, H, mode) |
| `fbrl.train_sft` | warm start on scripted episodes, modes spread evenly; `demos.oracle` sets the share of solved demos |
| `fbrl.train_grpo` | GRPO through TRL's `rollout_func`; `env.mode` can be a schedule like `[[0, "true"], [200, none]]`; runs `fbrl.evaluate` every `eval.every` steps |

```sh
uv sync
uv run pytest -q tests

cluster/remote.sh sync
cluster/remote.sh submit -- fbrl.evaluate --config configs/eval.yaml --set model.name=Qwen/Qwen3-1.7B
cluster/remote.sh submit -- fbrl.train_sft --config configs/sft.yaml --config configs/models/qwen3-1.7b.yaml \
    --set experiment.export='$FBRL/models/sft-qwen3-1.7b'
cluster/remote.sh sweep --gres=gpu:a40:1 -- fbrl.train_grpo --config configs/grpo.yaml \
    --grid env.mode=none,true,noise,residual --grid experiment.seed=0,1,2,3,4
cluster/remote.sh harness compare --where tracking.wandb.group=<sweep> --group-by env.mode \
    --metrics eval/spec_m1_H1_none/success,eval/spec_m1_H1_counterfactual/cf_score
```

Every run logs to W&B (project `fbrl`), and a sweep's runs share a W&B group named after the sweep. vLLM
is Linux only, so anything that generates runs on the cluster. Jobs default to one A100-SXM4 on
`general-gpu-long`; pass `--gres=gpu:a40:1` or `gpu:a6000:1` (48 GB) for anything that fits there.
`cluster/remote.sh` with no arguments lists its commands.
