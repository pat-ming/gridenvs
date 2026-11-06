"""Instance generation: plant a solved board, then corrupt it with m random writes."""
import numpy as np
from constraints import BOX, N, all_constraints, violations

VMIN, VMAX = 1, 9  # allowed cell values (inclusive)


def _shuffled_lines(rng) -> list[int]:
    """Line order that shuffles the three bands and the lines within each band."""
    return [BOX * band + i for band in rng.permutation(BOX) for i in rng.permutation(BOX)]


def solution(rng) -> np.ndarray:
    """A random solved board: a fixed valid pattern under transformations that keep it valid."""
    base = np.array([[(BOX * r + r // BOX + c) % N for c in range(N)] for r in range(N)])
    digits = rng.permutation(np.arange(VMIN, VMAX + 1))
    g = digits[base][_shuffled_lines(rng)][:, _shuffled_lines(rng)]
    return g.T.copy() if rng.random() < 0.5 else g.copy()


def make_instance(m: int, seed: int) -> dict:
    """m = number of random corrupting writes; the instance is solvable in <= m writes."""
    rng = np.random.default_rng(seed)
    cs = all_constraints()
    g_star = solution(rng)
    assert violations(g_star, cs) == []
    while True:
        g0 = g_star.copy()
        for _ in range(m):
            r, c = rng.integers(N, size=2)
            g0[r, c] = rng.integers(VMIN, VMAX + 1)
        if violations(g0, cs):  # start must be broken
            break
    return {"seed": seed, "m": m, "grid": g0, "constraints": cs, "solution": g_star}
