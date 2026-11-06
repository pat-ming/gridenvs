"""Multi-turn Sudoku environment.

One class, four observation modes. Everything except the feedback line is shared:
    "none"            E_A: board only
    "true"            E_C: board + "violated: {k, ...}" with the true V(s)
    "noise"           E_D: board + "violated: {□, ...}" whose size is random and independent of s
    "counterfactual"  eval: board + "violated: {...}" showing V' != V(s) with |V'| = |V(s)|

Usage:
    env = SudokuEnv(mode="true", H=4, seed=0)
    obs = env.reset(make_instance(m=2, seed=123))
    while True:
        obs, reward, done, info = env.step(model_reply)
        if done: break
"""
import re
import numpy as np

from constraints import N, render_constraints, violations
from instances import VMAX, VMIN

MODES = ("none", "true", "noise", "counterfactual")
# Under the Qwen3 tokenizer "□" takes as many tokens as one digit. Indices can have two digits here,
# so a noise item repeats it to the width of a real index. tests/test_env.py checks this.
NOISE_ITEM = "□"
ACTION_RE = re.compile(r"\bWRITE\s+([1-9])\s+([0-8])\s+([0-8])(?!\d)")


def format_feedback(items) -> str:
    return "violated: {" + ", ".join(str(x) for x in items) + "}"


class SudokuEnv:
    def __init__(self, mode: str, H: int, seed: int = 0, noise_sizes=None):
        """
        noise_sizes: optional list of set sizes to sample the noise line's length from
            (e.g. the empirical |V| distribution from E_C rollouts). Default: uniform 0..K.
            Whatever you pick, it is sampled independently of the current state.
        """
        assert mode in MODES, mode
        self.mode, self.H = mode, H
        self.rng = np.random.default_rng(seed)  # only used for noise / counterfactual lines
        self.noise_sizes = noise_sizes

    # ------------------------------------------------------------------ API
    def reset(self, inst: dict) -> str:
        self.inst = inst
        self.g = np.array(inst["grid"]).copy()
        self.cs = inst["constraints"]
        self.K = len(self.cs)
        self.t = 0
        self.log = []
        header = (
            "Constraints:\n" + render_constraints(self.cs) + "\n"
            f"Values are digits {VMIN}-{VMAX}. Rows and columns are numbered 0-{N - 1}. "
            f"Boxes are numbered 0-{N - 1}, left to right and then top to bottom.\n"
            f"You have {self.H} writes. Each turn, reply with only one line of the form WRITE v r c "
            "(value v, row r, column c), with no other text. Example reply: WRITE 7 2 1\n\n"
        )
        return header + self._obs()

    def step(self, text: str):
        assert self.t < self.H, "episode is over; call reset()"
        V_before = violations(self.g, self.cs)  # V(s_t): probe label at this decision
        shown_before = self.shown                # feedback the model saw before deciding

        matches = ACTION_RE.findall(text)  # the last write is the answer; earlier ones may be reasoning
        action = None
        if matches:
            v, r, c = map(int, matches[-1])
            action = (v, r, c)
            self.g[r, c] = v
        # invalid / unparseable action = no-op that still uses a turn

        self.t += 1
        V_after = violations(self.g, self.cs)
        done = self.t >= self.H
        solved = V_after == []
        reward = float(done and solved)

        obs = self._obs()
        self.log.append({
            "t": self.t - 1,
            "V": V_before,
            "shown": shown_before,   # what the feedback line claimed (None in "none" mode)
            "text": text,
            "action": action,
            "parse_ok": action is not None,
            "grid_after": self.g.copy(),
        })
        info = {"log": self.log, "V": V_after, "solved": solved}
        return obs, reward, done, info

    # ------------------------------------------------------------ rendering
    def _grid_text(self) -> str:
        return "\n".join(" ".join(str(int(x)) for x in row) for row in self.g)

    def _obs(self) -> str:
        self.line = self._feedback_line()  # exact feedback text, or None
        return self._grid_text() + ("" if self.line is None else "\n" + self.line)

    def _feedback_line(self):
        """Sets self.shown (the claimed violation set, or None) and returns the line."""
        V = violations(self.g, self.cs)

        if self.mode == "none":
            self.shown = None
            return None

        if self.mode == "true":
            self.shown = V
            return format_feedback(V)

        if self.mode == "noise":
            # Same header and token length as E_C; the filler carries no index information.
            # Size and widths are drawn independently of the state so length doesn't leak V.
            n = (int(self.rng.choice(self.noise_sizes)) if self.noise_sizes is not None
                 else int(self.rng.integers(0, self.K + 1)))
            indices = sorted(int(k) for k in self.rng.choice(self.K, size=n, replace=False))
            self.shown = None
            return format_feedback([NOISE_ITEM * len(str(k)) for k in indices])

        if self.mode == "counterfactual":
            Vp = self._counterfactual(V)
            self.shown = Vp
            return format_feedback(Vp)

    def _counterfactual(self, V: list[int]) -> list[int]:
        """V' != V, same size when possible: swap one violated index for one satisfied index.
        Edge cases (sizes necessarily change): V empty -> one random index; V = all -> drop one."""
        sat = [k for k in range(self.K) if k not in V]
        if not V:
            return [int(self.rng.choice(self.K))]
        if not sat:
            drop = int(self.rng.choice(V))
            return [k for k in V if k != drop]
        out_k = int(self.rng.choice(V))
        in_k = int(self.rng.choice(sat))
        return sorted([k for k in V if k != out_k] + [in_k])


def counterfactual_score(log: list[dict], constraints) -> dict | None:
    """Did writes follow the true violations or the false line? (one episode, H3's statistic)

    On each turn, compare the written cell with the constraints only the true set contains
    (V - V') and those only the shown set contains (V' - V). A cell belongs to a row, a column and
    a box, so the two rates are not exclusive. chance_* is the rate for a uniformly random cell,
    because scope unions differ in size. score > 0 means writes lean toward the false line.
    Returns None if no turn had a parseable write and a shown set differing from V.
    """
    totals = {"toward_true": 0.0, "toward_false": 0.0, "chance_true": 0.0, "chance_false": 0.0}
    n = 0
    for e in log:
        if e["shown"] is None or e["action"] is None or set(e["shown"]) == set(e["V"]):
            continue
        cell = e["action"][1:]
        true_only = set(e["V"]) - set(e["shown"])
        false_only = set(e["shown"]) - set(e["V"])
        for name, ks in (("true", true_only), ("false", false_only)):
            cells = frozenset().union(*(constraints[k].scope() for k in ks)) if ks else frozenset()
            totals[f"toward_{name}"] += cell in cells
            totals[f"chance_{name}"] += len(cells) / (N * N)
        n += 1
    if n == 0:
        return None
    out = {k: v / n for k, v in totals.items()}
    out["score"] = (out["toward_false"] - out["chance_false"]) - (out["toward_true"] - out["chance_true"])
    out["turns"] = n
    return out
