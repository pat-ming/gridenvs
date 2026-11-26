"""Constraints for the Sudoku task: every row, column and box holds 1-n exactly once.

The board is n x n with n in SIZES: 9 (3x3 boxes) is the full task, and 4 (2x2 boxes) and 6
(2x3 boxes, two rows tall and three columns wide) are smaller versions of it.

Every constraint type provides:
    holds(g)  -> bool                  True if grid g (n x n ints) satisfies it
    scope()   -> frozenset[(r, c)]     the cells that can affect holds()
    render()  -> str                   the text shown to the policy

violations(g, constraints) is the single source of truth for V(s).
"""
from dataclasses import dataclass
import numpy as np

N = 9    # default board size
BOX = 3  # box side of the default board
SIZES = {4: (2, 2), 6: (2, 3), 9: (3, 3)}  # n -> (box height, box width)
DIGITS = list(range(1, N + 1))
Cell = tuple[int, int]


class Unit:
    """A group of n cells that must hold each digit 1-n once."""

    n: int

    def cells(self) -> list[Cell]:
        raise NotImplementedError

    def holds(self, g) -> bool:
        g = np.asarray(g)
        return sorted(int(g[cell]) for cell in self.cells()) == list(range(1, self.n + 1))

    def scope(self) -> frozenset[Cell]:
        return frozenset(self.cells())

    def render(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class Row(Unit):
    row: int
    n: int = N

    def cells(self) -> list[Cell]:
        return [(self.row, c) for c in range(self.n)]

    def render(self) -> str:
        return f"row {self.row} has each digit 1-{self.n} once"


@dataclass(frozen=True)
class Col(Unit):
    col: int
    n: int = N

    def cells(self) -> list[Cell]:
        return [(r, self.col) for r in range(self.n)]

    def render(self) -> str:
        return f"column {self.col} has each digit 1-{self.n} once"


@dataclass(frozen=True)
class Box(Unit):
    """Boxes are numbered 0 to n-1, left to right and then top to bottom."""

    box: int
    n: int = N

    def cells(self) -> list[Cell]:
        bh, bw = SIZES[self.n]
        per_band = self.n // bw  # boxes side by side
        top, left = bh * (self.box // per_band), bw * (self.box % per_band)
        return [(top + r, left + c) for r in range(bh) for c in range(bw)]

    def render(self) -> str:
        return f"box {self.box} has each digit 1-{self.n} once"


def all_constraints(n: int = N) -> list[Unit]:
    """Constraint index k = position: rows are 0..n-1, columns n..2n-1, boxes 2n..3n-1."""
    assert n in SIZES, n
    return [Row(i, n) for i in range(n)] + [Col(i, n) for i in range(n)] + [Box(i, n) for i in range(n)]


def violations(g, constraints: list[Unit]) -> list[int]:
    """V(s): indices of the constraints that grid g violates."""
    return [k for k, c in enumerate(constraints) if not c.holds(g)]


def render_constraints(constraints: list[Unit]) -> str:
    return "\n".join(f"c{k}: {c.render()}" for k, c in enumerate(constraints))
