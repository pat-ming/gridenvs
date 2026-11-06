"""Constraints for the 9x9 Sudoku task: every row, column and box holds 1-9 exactly once.

Every constraint type provides:
    holds(g)  -> bool                  True if grid g (9x9 ints) satisfies it
    scope()   -> frozenset[(r, c)]     the cells that can affect holds()
    render()  -> str                   the text shown to the policy

violations(g, constraints) is the single source of truth for V(s).
"""
from dataclasses import dataclass
import numpy as np

N = 9
BOX = 3
DIGITS = list(range(1, N + 1))
Cell = tuple[int, int]


class Unit:
    """A group of nine cells that must hold each digit once."""

    def cells(self) -> list[Cell]:
        raise NotImplementedError

    def holds(self, g) -> bool:
        g = np.asarray(g)
        return sorted(int(g[cell]) for cell in self.cells()) == DIGITS

    def scope(self) -> frozenset[Cell]:
        return frozenset(self.cells())

    def render(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class Row(Unit):
    row: int

    def cells(self) -> list[Cell]:
        return [(self.row, c) for c in range(N)]

    def render(self) -> str:
        return f"row {self.row} has each digit 1-9 once"


@dataclass(frozen=True)
class Col(Unit):
    col: int

    def cells(self) -> list[Cell]:
        return [(r, self.col) for r in range(N)]

    def render(self) -> str:
        return f"column {self.col} has each digit 1-9 once"


@dataclass(frozen=True)
class Box(Unit):
    """Boxes are numbered 0-8, left to right and then top to bottom."""

    box: int

    def cells(self) -> list[Cell]:
        top, left = BOX * (self.box // BOX), BOX * (self.box % BOX)
        return [(top + r, left + c) for r in range(BOX) for c in range(BOX)]

    def render(self) -> str:
        return f"box {self.box} has each digit 1-9 once"


def all_constraints() -> list[Unit]:
    """Constraint index k = position: rows are 0-8, columns 9-17, boxes 18-26."""
    return [Row(i) for i in range(N)] + [Col(i) for i in range(N)] + [Box(i) for i in range(N)]


def violations(g, constraints: list[Unit]) -> list[int]:
    """V(s): indices of the constraints that grid g violates."""
    return [k for k, c in enumerate(constraints) if not c.holds(g)]


def render_constraints(constraints: list[Unit]) -> str:
    return "\n".join(f"c{k}: {c.render()}" for k, c in enumerate(constraints))
