"""guard.py — Narrow tactical guard for searchless policy models (Maia-3).

Maia-3 is a pure searchless policy network. In crushing, won positions (e.g. multiple
promoted queens or heavy material advantage), it can wander aimlessly and miss obvious
mates-in-one or stumble into stalemates.

This guard:
1. Detects available mates-in-one and converts them according to human Elo error priors.
2. Prevents catastrophic stalemates in won positions.
3. Leaves all normal tactical/positional moves untouched.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Optional

import chess


# Prior probabilities of missing an available mate-in-1 by Elo rating
MATE_MISS_PRIOR: Sequence[tuple[int, float]] = (
    (1000, 0.120),
    (1300, 0.060),
    (1500, 0.035),
    (1800, 0.015),
    (2200, 0.005),
)


def _interp(table: Sequence[tuple[int, float]], x: float) -> float:
    """Piecewise linear interpolation over rating knots."""
    if x <= table[0][0]:
        return table[0][1]
    for (x0, y0), (x1, y1) in zip(table, table[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return table[-1][1]


def mates_in_one(board: chess.Board) -> list[chess.Move]:
    """Return all legal moves that deliver checkmate immediately."""
    mates: list[chess.Move] = []
    for m in board.legal_moves:
        board.push(m)
        if board.is_checkmate():
            mates.append(m)
        board.pop()
    return mates


def _visibility(board: chess.Board, m: chess.Move) -> float:
    """Visibility factor: >1 for moves humans overlook more often (discovered check, promotion)."""
    v = 1.0
    mover_sq = m.to_square
    board.push(m)
    checkers = list(board.checkers())
    board.pop()
    if mover_sq not in checkers:
        v *= 2.0  # Discovered check
    if m.promotion:
        v *= 1.3
    return v


@dataclass
class GuardResult:
    """Outcome of TacticalGuard evaluation."""
    move: chess.Move
    reason: str  # "ok" | "mate_in_1" | "missed_mate" | "stalemate_avoid"


class TacticalGuard:
    """Guards against tactical oversights (missed mate-in-1 and accidental stalemates)."""

    def __init__(
        self,
        elo: int = 1500,
        rng: Optional[random.Random] = None,
        mate_table: Sequence[tuple[int, float]] = MATE_MISS_PRIOR,
        stalemate_allow: float = 0.02,
    ) -> None:
        self.elo = elo
        self.rng = rng or random.Random()
        self.table = mate_table
        self.stalemate_allow = stalemate_allow
        self.missed_last: bool = False

    def p_miss(self, board: chess.Board, mate: chess.Move, my_clock_s: float) -> float:
        """Probability of overlooking this specific mate."""
        base = _interp(self.table, self.elo)
        pressure = 1.0 + 2.0 * max(0.0, (20.0 - my_clock_s) / 20.0)
        # If human missed it on move N, they almost certainly notice on move N+1
        decay = 0.15 if self.missed_last else 1.0
        return min(0.50, base * _visibility(board, mate) * pressure * decay)

    def apply(
        self,
        board: chess.Board,
        chosen: chess.Move,
        ranked: Iterable[tuple[chess.Move, float]] = (),
        my_clock_s: float = math.inf,
    ) -> GuardResult:
        """Evaluate chosen move. If a mate exists or stalemate threatened, adjust accordingly."""
        ranked_list = list(ranked)
        mates = mates_in_one(board)

        if mates:
            if chosen in mates:
                self.missed_last = False
                return GuardResult(chosen, "ok")

            # Human checks the most visually obvious mate first
            easiest = min(mates, key=lambda m: _visibility(board, m))
            if self.rng.random() < self.p_miss(board, easiest, my_clock_s):
                self.missed_last = True
                return GuardResult(chosen, "missed_mate")

            self.missed_last = False
            probs = dict(ranked_list)
            pick = max(mates, key=lambda m: probs.get(m, 0.0))
            return GuardResult(pick, "mate_in_1")

        self.missed_last = False

        # Stalemate prevention: policy nets sometimes trap kings accidentally
        board.push(chosen)
        is_stale = board.is_stalemate()
        board.pop()

        if is_stale and self.rng.random() > self.stalemate_allow:
            # Find the best non-stalemating alternative
            alternatives = [m for m, _ in ranked_list] or list(board.legal_moves)
            for m in alternatives:
                board.push(m)
                bad = board.is_stalemate()
                board.pop()
                if not bad:
                    return GuardResult(m, "stalemate_avoid")

        return GuardResult(chosen, "ok")
