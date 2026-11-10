"""Constraints for the 4x4 grid task.

Every constraint type provides:
    holds(g)  -> bool                  True if grid g (4x4 ints) satisfies it
    scope()   -> frozenset[(r, c)]     the cells that can affect holds()
    render()  -> str                   the text shown to the policy

violations(g, constraints) is the single source of truth for V(s).
"""
from dataclasses import dataclass

import numpy as np

N = 4
Cell = tuple[int, int]
REGISTRY: dict[str, type] = {}


def register(cls):
    """Add a constraint class to REGISTRY under its `name`."""
    REGISTRY[cls.name] = cls
    return cls


class Constraint:
    """Interface that every constraint type implements."""

    name = "base"

    def holds(self, g) -> bool:
        raise NotImplementedError

    def scope(self) -> frozenset[Cell]:
        raise NotImplementedError

    def render(self) -> str:
        raise NotImplementedError


ALL_CELLS: frozenset[Cell] = frozenset((r, c) for r in range(N) for c in range(N))


@register
@dataclass(frozen=True)
class FlatDescending(Constraint):
    strict: bool = False # ≤ or <
    name = "flat_descending"

    def holds(self, g) -> bool:
        d = np.diff(np.asarray(g).ravel())
        return bool(np.all(d < 0)) if self.strict else bool(np.all(d <= 0))

    def scope(self) -> frozenset[Cell]:
        return ALL_CELLS

    def render(self) -> str:
        kind = "strictly descending" if self.strict else "non-increasing"
        return f"grid read row by row is {kind}"


@register
@dataclass(frozen=True)
class RowSum(Constraint):
    """Row `row` sums to `target`."""

    row: int
    target: int = 10
    name = "row_sum"

    def holds(self, g) -> bool:
        return int(np.asarray(g)[self.row].sum()) == self.target

    def scope(self) -> frozenset[Cell]:
        return frozenset((self.row, c) for c in range(N))

    def render(self) -> str:
        return f"row {self.row} sums to {self.target}"


@register
@dataclass(frozen=True)
class ColSum(Constraint):
    """Column `col` sums to `target`."""

    col: int
    target: int = 10
    name = "col_sum"

    def holds(self, g) -> bool:
        return int(np.asarray(g)[:, self.col].sum()) == self.target

    def scope(self) -> frozenset[Cell]:
        return frozenset((r, self.col) for r in range(N))

    def render(self) -> str:
        return f"column {self.col} sums to {self.target}"


@register
@dataclass(frozen=True)
class PairwiseLess(Constraint):
    """Cell `a` is strictly less than cell `b`."""

    a: Cell
    b: Cell
    name = "pairwise_less"

    def holds(self, g) -> bool:
        g = np.asarray(g)
        return bool(g[self.a] < g[self.b])

    def scope(self) -> frozenset[Cell]:
        return frozenset((self.a, self.b))

    def render(self) -> str:
        return f"cell ({self.a[0]},{self.a[1]}) < cell ({self.b[0]},{self.b[1]})"


@register
@dataclass(frozen=True)
class AllDistinctCol(Constraint):
    """All values in column `col` are different."""

    col: int
    name = "all_distinct_col"

    def holds(self, g) -> bool:
        return len(set(np.asarray(g)[:, self.col].tolist())) == N

    def scope(self) -> frozenset[Cell]:
        return frozenset((r, self.col) for r in range(N))

    def render(self) -> str:
        return f"column {self.col} has all distinct values"


def all_row_sums(target: int = 10) -> list[RowSum]:
    """"Every row sums to `target`", as one constraint per row."""
    return [RowSum(r, target) for r in range(N)]

def all_col_sums(target: int = 10) -> list[ColSum]:
    """"Every column sums to `target`", as one constraint per column."""
    return [ColSum(c, target) for c in range(N)]

def violations(g, constraints: list[Constraint]) -> list[int]:
    """V(s): indices of the constraints that grid g violates."""
    return [k for k, c in enumerate(constraints) if not c.holds(g)]

def render_constraints(constraints: list[Constraint]) -> str:
    return "\n".join(f"c{k}: {c.render()}" for k, c in enumerate(constraints))
