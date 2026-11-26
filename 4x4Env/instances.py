"""Instance generation: plant a solution, then corrupt it with m random edits."""
import numpy as np
from constraints import AllDistinctCol, FlatDescending, PairwiseLess, all_col_sums, all_row_sums, violations

VMIN, VMAX = 0, 9  # allowed cell values (inclusive)

# Constraint index k = position in the list. Keep the order fixed per experiment.
CONSTRAINT_SETS = {
    "desc": [FlatDescending()],
    "rows": all_row_sums(10),
    "cols": all_col_sums(10),
    "rows+cols": all_row_sums(10) + all_col_sums(10),
    "distinct": [AllDistinctCol(c) for c in range(4)],
    "ineq": [],
    # The formulation's task family. K = 10, so every index is one digit and the
    # feedback line's token count depends only on how many indices it lists.
    "spec": all_row_sums(10) + [AllDistinctCol(c) for c in range(4)],
    # Do NOT combine "desc" with any sum constraint: no integer grid satisfies both.
}
# Pairwise inequalities are sampled per instance from the planted solution and
# appended after the fixed constraints, so slot k keeps a stable meaning.
NUM_PAIRWISE = {"ineq": 6, "spec": 2}
NEEDS_DISTINCT_COLS = {"distinct", "spec"}


def solution_desc(rng, vmax: int = VMAX) -> np.ndarray:
    return np.sort(rng.integers(VMIN, vmax + 1, 16))[::-1].reshape(4, 4).copy()

def solution_sums(rng, steps: int = 200, vmax: int = VMAX) -> np.ndarray:
    """Random grid where every row and every column sums to 10."""
    g = np.array([[0, 1, 4, 5], [1, 0, 5, 4], [4, 5, 0, 1], [5, 4, 1, 0]])
    g = g[rng.permutation(4)][:, rng.permutation(4)]
    # Rectangle move: +1 at (r1,c1),(r2,c2), -1 at (r1,c2),(r2,c1).
    # Preserves every row sum and column sum.
    for _ in range(steps):
        r1, r2 = rng.choice(4, 2, replace=False)
        c1, c2 = rng.choice(4, 2, replace=False)
        g2 = g.copy()
        g2[r1, c1] += 1
        g2[r2, c2] += 1
        g2[r1, c2] -= 1
        g2[r2, c1] -= 1
        if g2.min() >= VMIN and g2.max() <= vmax:
            g = g2
    return g

def solution_distinct_sums(rng, vmax: int = VMAX) -> np.ndarray:
    """Row and column sums of 10, with four different values in every column."""
    while True:
        g = solution_sums(rng, vmax=vmax)
        if all(len(set(g[:, c].tolist())) == 4 for c in range(4)):
            return g

def sample_pairwise(g_star, n: int, rng) -> list[PairwiseLess]:
    """n distinct inequalities between cells that the planted solution satisfies."""
    out = []
    while len(out) < n:
        i, j = rng.choice(16, 2, replace=False)
        a, b = (int(i) // 4, int(i) % 4), (int(j) // 4, int(j) % 4)
        if g_star[a] == g_star[b]:
            continue
        if g_star[a] > g_star[b]:
            a, b = b, a
        c = PairwiseLess(a, b)
        if c not in out:
            out.append(c)
    return out

def make_instance(set_name: str, m: int, seed: int, k: int | None = None, vmax: int = VMAX) -> dict:
    """m = number of random corrupting edits; the instance is solvable in <= m edits.

    Optional difficulty settings (the defaults reproduce the original instances exactly):
        k:     keep only the first k constraints of the set. Index k keeps its meaning, so a prefix
               of "spec" is a sub-task of "spec". Corrupting edits only touch cells those k
               constraints can see.
        vmax:  cell values are 0..vmax instead of 0-9, which shrinks the action space. The sum
               target stays 10, so sum sets need vmax >= 5 (the seed grid's largest value).
    """
    assert VMIN < vmax <= VMAX and (set_name == "desc" or vmax >= 5), vmax
    rng = np.random.default_rng(seed)
    if set_name == "desc":
        g_star = solution_desc(rng, vmax)
    elif set_name in NEEDS_DISTINCT_COLS:
        g_star = solution_distinct_sums(rng, vmax)
    else:
        g_star = solution_sums(rng, vmax=vmax)
    cs = CONSTRAINT_SETS[set_name] + sample_pairwise(g_star, NUM_PAIRWISE.get(set_name, 0), rng)
    if k is not None:
        assert 1 <= k <= len(cs), f"k={k} but {set_name!r} has {len(cs)} constraints"
        cs = cs[:k]
    assert violations(g_star, cs) == []
    cells = None if k is None else sorted(frozenset().union(*(c.scope() for c in cs)))
    while True:
        g0 = g_star.copy()
        for _ in range(m):
            if cells is None:
                r, c = rng.integers(4, size=2)
            else:
                r, c = cells[rng.integers(len(cells))]
            g0[r, c] = rng.integers(VMIN, vmax + 1)
        if violations(g0, cs):  # start must be broken
            break
    return {"seed": seed, "set": set_name, "m": m, "k": k, "vmax": vmax,
            "grid": g0, "constraints": cs, "solution": g_star}


# Easier levels that build up to the full sets, easiest first. Each level is make_instance settings
# plus the horizon H to give GridEnv. Level 0 of each ladder starts near "ineq" for Qwen3-0.6B
# (see the README); later levels are meant to be reached by training on the earlier ones.
CURRICULUM = {
    "rows": [
        dict(set_name="rows", m=1, H=1, vmax=5),
        dict(set_name="rows", m=1, H=1),
        dict(set_name="rows", m=2, H=4),
    ],
    "spec": [
        dict(set_name="spec", k=4, m=1, H=1, vmax=5),  # the 4 row sums only
        dict(set_name="spec", k=8, m=1, H=1, vmax=5),  # + distinct columns
        dict(set_name="spec", m=1, H=1, vmax=5),       # + the 2 inequalities
        dict(set_name="spec", m=1, H=1),
        dict(set_name="spec", m=2, H=4),
    ],
}


def make_level(ladder: str, level: int, seed: int) -> tuple[dict, int]:
    """(instance, H) for one level of CURRICULUM[ladder]."""
    kw = dict(CURRICULUM[ladder][level])
    H = kw.pop("H")
    return make_instance(seed=seed, **kw), H
