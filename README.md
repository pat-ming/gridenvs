# GridEnvs

Environments for *Does RL Build a Circuit the Environment Already Runs?* In LLM post-training, a
pretrained model plays an environment over several turns and learns from the reward. These
environments only handle text: your training code sends in the model's reply and gets back the
next observation, so they work with any model.

## Setup

```bash
pip install -r requirements.txt
cd 4x4Env
python -m pytest tests -q
```

Run scripts from inside the environment's folder, because its modules import each other by name.

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
- **`spec` is hard from a cold start.** With the current wording, Qwen3-0.6B had 0 successes in 64
  episodes per setting tried, with the feedback line on and off. A terminal reward gives GRPO nothing
  to learn from if every episode ties.

## Files in `4x4Env/`

| File | Purpose |
|---|---|
| `constraints.py` | Constraint types and `violations()`, the single source of truth for V(s) |
| `instances.py` | Instance generation and constraint sets |
| `grid_env.py` | `GridEnv` and `counterfactual_score` |
| `tests/test_env.py` | Environment tests |
