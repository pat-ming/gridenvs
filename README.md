# GridEnvs

Environments for *Does RL Build a Circuit the Environment Already Runs?* In LLM post-training, a
pretrained model plays an environment over several turns and learns from the reward. These
environments only handle text: your training code sends in the model's reply and gets back the
next observation, so they work with any model.

## Setup

```bash
pip install -r requirements.txt
cd 4x4Env        # or SudokuEnv
python -m pytest tests -q
```

Run scripts from inside the environment's folder, because its modules import each other by name.
Both folders have a `constraints.py` and an `instances.py`, so use one environment per Python
process. Otherwise Python reuses whichever of those modules it imported first.

There are two environments: [`4x4Env`](#4x4env-the-task) and [`SudokuEnv`](#sudokuenv). Both offer
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

### Easier levels

`spec` and `rows` start too hard to train on (see below). `make_instance` takes two optional
settings that make them easier. Leaving both out gives exactly the same instances as before.

- `k`: keep only the first `k` constraints of the set. Each index keeps its meaning, so for example
  `spec` with `k=4` is its four row sums. Corrupting edits only touch cells those constraints check.
- `vmax`: cell values run from 0 to `vmax` instead of 0–9. The prompt and the action check follow
  it. Sum targets stay at 10, so sum sets need `vmax >= 5`.

`CURRICULUM` in `instances.py` lists levels that build up to each full set, easiest first, and
`make_level` builds one:

```python
from instances import make_level
inst, H = make_level("spec", level=0, seed=123)
env = GridEnv(mode="true", H=H, seed=0)
```

Success rates for Qwen3-0.6B before any training, in %, with the share of puzzles that had mixed
outcomes in parentheses. Mixed puzzles are what give GRPO something to learn from. Each cell is 64
puzzles × 8 samples, with thinking off, temperature 1 and `top_k=0`. `ineq` (m=2, H=4), which did
train, measured 5.1 (30) with the true line and 3.1 (19) without it under the same setup.

| Level | Settings | True line | No line |
|---|---|---|---|
| `rows` 0 | m=1, H=1, `vmax=5` | 4.5 (28) | 3.3 (19) |
| `rows` 1 | m=1, H=1 | 4.3 (23) | 0.8 (5) |
| `rows` 2 | m=2, H=4 (the full set) | about 0 | about 0 |
| `spec` 0 | `k=4` (row sums), m=1, H=1, `vmax=5` | 4.7 (28) | 3.1 (17) |
| `spec` 1 | `k=8` (+ distinct columns), m=1, H=1, `vmax=5` | 2.3 (16) | 0.8 (6) |
| `spec` 2 | all 10, m=1, H=1, `vmax=5` | 1.0 (8) | 0.2 (2) |
| `spec` 3 | m=1, H=1 | 1.0 (6) | 0.8 (2) |
| `spec` 4 | m=2, H=4 (the full set) | about 0 | about 0 |

Level 0 of both ladders starts about where `ineq` did. The later levels start lower, and the plan is
to reach them by training on the earlier ones, which hasn't been tested. The model starts close to
chance: with a sum constraint only one value per cell repairs a row, so the levels work by shrinking
the number of choices. Using fewer corruptions, more turns or ±1 corruptions didn't help. Extra
turns made it worse, because a later edit can undo a repair.

## Using an environment in a training loop

```python
from grid_env import GridEnv
from instances import make_instance

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
- **`spec` and `rows` are hard from a cold start.** With the current wording, Qwen3-0.6B solves
  under 1% of them, with the feedback line on and off. A terminal reward gives GRPO nothing to learn
  from if every episode ties. Start from the [easier levels](#easier-levels).
- **About a fifth of replies copy the example reply.** They count as ordinary edits. With a narrowed
  `vmax` the example's value is clamped into range, so the copies stay valid.

## Files in `4x4Env/`

| File | Purpose |
|---|---|
| `constraints.py` | Constraint types and `violations()`, the single source of truth for V(s) |
| `instances.py` | Instance generation, constraint sets and `CURRICULUM` |
| `grid_env.py` | `GridEnv` and `counterfactual_score` |
| `tests/test_env.py` | Environment tests |

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
  `c`. Note the order: the value comes first, unlike `EDIT r c v` in `4x4Env`. Any cell can be
  overwritten. If a reply has several writes, the last one counts. An invalid reply uses up the turn
  and changes nothing.
- **Reward:** the same as `4x4Env`. `R = 1` if the whole board is valid after turn `H`, otherwise 0,
  with no intermediate reward. Use `H >= m`, or some instances can't be solved.

The four modes are the ones in [the table above](#the-environments), and you use them the same way.
There's one difference: indices can have two digits here, so a noise item repeats the filler to
match (`□` or `□□`), which keeps the line's token length matched to E_C.

```python
from instances import make_instance
from sudoku_env import SudokuEnv

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
V(s) a real computation, as in `4x4Env`, because a blank would mark the broken units on sight.

### Smaller boards and levels

`make_instance` takes two optional settings. Leaving both out gives exactly the same 9x9 instances
as before.

- `n`: board size, 4 (2x2 boxes), 6 (boxes two rows tall and three columns wide) or 9. Indices
  follow the same layout: rows `0..n-1`, columns `n..2n-1`, boxes `2n..3n-1`. The prompt, the action
  check and the example reply follow the size.
- `k`: keep only the first `k` constraints, so `k = n` checks rows only and `k = 2n` checks rows and
  columns.

`CURRICULUM` in `instances.py` lists levels from a 4x4 board with rows only up to the full board,
and `make_level(level, seed)` returns `(instance, H)`. Measured the same way as the
[4x4 levels](#easier-levels):

| Level | Settings | True line | No line |
|---|---|---|---|
| 0 | `n=4, k=4` (rows only), m=1, H=1 | 1.6 (11) | 2.0 (11) |
| 1 | `n=4, k=8` (+ columns), m=1, H=1 | 1.0 (6) | 2.3 (13) |
| 2 | `n=4`, m=1, H=1 | 1.4 (5) | 1.6 (5) |
| 3 | `n=6`, m=1, H=1 | 0.2 (2) | 0.0 (0) |
| 4 | `n=9`, m=1, H=1 | 0.0 (0) | not run |
| 5 | `n=9`, m=2, H=4 | 0.0 (0) | not run |

**Even level 0 starts below `ineq`**, which measured 5.1 (30) and 3.1 (19). On a 4x4 board, one or
two of the 64 possible writes repair it, and the model starts at about that chance rate. Showing the
corrupted cell as a blank didn't raise it either (0–3% on every size). Level 0 might still train with
a larger GRPO group, since mixed puzzles become more common with more samples per puzzle. Otherwise
Sudoku probably needs help from outside the environment: a short supervised warm start on repairs
(each instance stores its `solution`), a larger model, or starting from a 4x4Env checkpoint.

On 9x9 boards, 20–30% of replies are invalid, mostly because the model writes a constraint index
such as 12 or 25 as a coordinate. About a fifth of replies on every size copy the example reply.

| File | Purpose |
|---|---|
| `constraints.py` | `Row`, `Col`, `Box` and `violations()`, the single source of truth for V(s) |
| `instances.py` | Solved-board generation, corruption and `CURRICULUM` |
| `sudoku_env.py` | `SudokuEnv` and `counterfactual_score` |
| `tests/test_env.py` | Environment tests |
