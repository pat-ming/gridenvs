"""Instance generation: plant a solved board, then corrupt it with m random writes."""
import numpy as np
from constraints import N, SIZES, all_constraints, violations

VMIN, VMAX = 1, 9  # allowed cell values on the default 9x9 board (1..n in general)


def _shuffled_lines(rng, groups: int, size: int) -> list[int]:
    """Line order that shuffles the groups (bands or stacks) and the lines within each group."""
    return [size * band + i for band in rng.permutation(groups) for i in rng.permutation(size)]


def solution(rng, n: int = N) -> np.ndarray:
    """A random solved board: a fixed valid pattern under transformations that keep it valid."""
    bh, bw = SIZES[n]
    base = np.array([[(bw * (r % bh) + r // bh + c) % n for c in range(n)] for r in range(n)])
    digits = rng.permutation(np.arange(1, n + 1))
    g = digits[base][_shuffled_lines(rng, n // bh, bh)][:, _shuffled_lines(rng, n // bw, bw)]
    # Transposing swaps box height and width, so it only keeps square-box boards valid.
    return g.T.copy() if bh == bw and rng.random() < 0.5 else g.copy()


def make_instance(m: int, seed: int, n: int = N, k: int | None = None) -> dict:
    """m = number of random corrupting writes; the instance is solvable in <= m writes.

    Optional difficulty settings (the defaults reproduce the original 9x9 instances exactly):
        n: board size, 4, 6 or 9. Smaller boards have fewer cells, digits and constraints.
        k: keep only the first k constraints, a prefix of rows, then columns, then boxes, so
           k = n checks rows only and k = 2n checks rows and columns. Index k keeps its meaning.
           Every cell is in some row, so corrupting writes can land anywhere.
    """
    rng = np.random.default_rng(seed)
    cs = all_constraints(n)
    if k is not None:
        assert 1 <= k <= len(cs), f"k={k} but a {n}x{n} board has {len(cs)} constraints"
        cs = cs[:k]
    g_star = solution(rng, n)
    assert violations(g_star, cs) == []
    while True:
        g0 = g_star.copy()
        for _ in range(m):
            r, c = rng.integers(n, size=2)
            g0[r, c] = rng.integers(1, n + 1)
        if violations(g0, cs):  # start must be broken
            break
    return {"seed": seed, "m": m, "n": n, "k": k, "grid": g0, "constraints": cs, "solution": g_star}


# Smaller boards that build up to the full 9x9 one, easiest first. Each level is make_instance
# settings plus the horizon H to give SudokuEnv. Even level 0 starts below "ineq" for Qwen3-0.6B
# (see the README).
CURRICULUM = [
    dict(n=4, k=4, m=1, H=1),   # 4x4, rows only
    dict(n=4, k=8, m=1, H=1),   # + columns
    dict(n=4, m=1, H=1),        # + boxes
    dict(n=6, m=1, H=1),
    dict(n=9, m=1, H=1),
    dict(n=9, m=2, H=4),
]


def make_level(level: int, seed: int) -> tuple[dict, int]:
    """(instance, H) for one level of CURRICULUM."""
    kw = dict(CURRICULUM[level])
    H = kw.pop("H")
    return make_instance(seed=seed, **kw), H
