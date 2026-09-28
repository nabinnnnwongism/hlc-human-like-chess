"""Core data types and interfaces for Human-Like Chess (HLC)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import chess


@dataclass(frozen=True)
class GameState:
    """Represents the complete game state passed to BotCore."""

    board: chess.Board
    move_history: list[chess.Move] = field(default_factory=list)
    clock_self: float = 180.0  # Remaining time for side to move (seconds)
    clock_opp: float = 180.0  # Remaining time for opponent (seconds)
    increment: float = 0.0  # Clock increment per move (seconds)
    self_elo: int = 1500
    opp_elo: int = 1500

    @property
    def turn(self) -> chess.Color:
        return self.board.turn

    @property
    def is_white(self) -> bool:
        return self.board.turn == chess.WHITE

    @property
    def ply_count(self) -> int:
        return len(self.move_history)

    def legal_moves(self) -> list[chess.Move]:
        return list(self.board.legal_moves)


@dataclass(frozen=True)
class CandidateMove:
    """Detailed information for a candidate move."""

    move: chess.Move
    probability: float
    wdl: tuple[int, int, int] | None = None  # (win, draw, loss) in permille (sums to 1000)


@dataclass
class MoveDistribution:
    """Probability distribution over legal moves in the current position."""

    moves: list[chess.Move]
    probabilities: list[float]
    candidate_details: list[CandidateMove] = field(default_factory=list)
    entropy: float = 0.0
    wdl: tuple[int, int, int] | None = None  # Current position WDL (win, draw, loss)

    def __post_init__(self) -> None:
        if len(self.moves) != len(self.probabilities):
            raise ValueError(
                f"Moves count ({len(self.moves)}) must match probabilities count ({len(self.probabilities)})"
            )
        if not self.moves:
            self.entropy = 0.0
            return

        # Compute Shannon entropy in nats
        total_p = sum(self.probabilities)
        if total_p > 0:
            norm_p = [p / total_p for p in self.probabilities]
            self.entropy = -sum(p * math.log(p) for p in norm_p if p > 1e-12)
        else:
            self.entropy = 0.0

    @property
    def top_move(self) -> chess.Move | None:
        if not self.moves:
            return None
        max_idx = max(range(len(self.probabilities)), key=lambda i: self.probabilities[i])
        return self.moves[max_idx]

    @property
    def top_probability(self) -> float:
        if not self.probabilities:
            return 0.0
        return max(self.probabilities)

    @property
    def legal_move_count(self) -> int:
        return len(self.moves)

    def sample(
        self,
        rng: random.Random | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> chess.Move:
        """Sample a move given temperature and nucleus top_p thresholds.

        temperature <= 0: deterministic argmax (top-1 move).
        top_p < 1.0: nucleus sampling keeping smallest set whose cumulative prob >= top_p.
        """
        if not self.moves:
            raise ValueError("Cannot sample from empty move distribution.")
        if len(self.moves) == 1 or temperature <= 0.0:
            return self.top_move  # type: ignore[return-value]

        r = rng if rng is not None else random.Random()

        # Apply temperature scaling to probabilities (convert back to log scale or p^(1/T))
        # Since p_i is softmax(z_i), p_i^(1/T) normalized is softmax(z_i / T)
        raw_weights = [math.pow(max(p, 1e-12), 1.0 / temperature) for p in self.probabilities]
        total_weight = sum(raw_weights)
        scaled_probs = [w / total_weight for w in raw_weights]

        # Nucleus (top-p) filtering
        if top_p < 1.0:
            sorted_indices = sorted(
                range(len(scaled_probs)), key=lambda i: scaled_probs[i], reverse=True
            )
            cumulative = 0.0
            kept_indices: list[int] = []
            for idx in sorted_indices:
                kept_indices.append(idx)
                cumulative += scaled_probs[idx]
                if cumulative >= top_p:
                    break

            kept_probs = [scaled_probs[i] for i in kept_indices]
            total_kept = sum(kept_probs)
            normalized_kept = [kp / total_kept for kp in kept_probs]

            # Sample from kept indices
            roll = r.random()
            cum = 0.0
            for k_idx, norm_p in zip(kept_indices, normalized_kept):
                cum += norm_p
                if roll <= cum:
                    return self.moves[k_idx]
            return self.moves[kept_indices[-1]]

        # Standard categorical sampling
        roll = r.random()
        cum = 0.0
        for move, p in zip(self.moves, scaled_probs):
            cum += p
            if roll <= cum:
                return move
        return self.moves[-1]


@dataclass(frozen=True)
class Decision:
    """The bot's complete output for a turn."""

    move: chess.Move
    delay_s: float
    debug: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MoveEngine(Protocol):
    """Protocol for move generation engines (e.g. Maia3)."""

    def get_distribution(self, state: GameState) -> MoveDistribution:
        """Compute the move probability distribution for the current position."""
        ...
