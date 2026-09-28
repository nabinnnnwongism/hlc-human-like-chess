"""BotCore: Coordinates MoveEngine, TimingModel, and Scheduler."""

from __future__ import annotations

import random
import time
from typing import Any

from hlc.types import Decision, GameState, MoveDistribution, MoveEngine


class BotCore:
    """Core brain of the Human-Like Chess bot.

    Orchestrates move selection, timing distribution prediction, and delay scheduling.
    """

    def __init__(
        self,
        move_engine: MoveEngine,
        timing_model: Any = None,
        scheduler: Any = None,
        rng: random.Random | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> None:
        self.move_engine = move_engine
        self.timing_model = timing_model
        self.scheduler = scheduler
        self.rng = rng if rng is not None else random.Random(42)
        self.temperature = temperature
        self.top_p = top_p

    def decide(self, state: GameState) -> Decision:
        """Process the game state, choose a move, and schedule its think delay."""
        t_start = time.perf_counter()

        # 1. Compute move distribution from move engine
        dist: MoveDistribution = self.move_engine.get_distribution(state)
        if not dist.moves:
            raise RuntimeError(f"No legal moves available in position: {state.board.fen()}")

        # 2. Sample move according to temperature and top-p settings
        chosen_move = dist.sample(
            rng=self.rng,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        compute_elapsed = time.perf_counter() - t_start

        # 3. Predict think time distribution (if timing model present, else default)
        delay_s = 0.0
        timing_debug: dict[str, Any] = {}

        if self.timing_model is not None and self.scheduler is not None:
            think_dist = self.timing_model.predict(state, dist)
            delay_s = self.scheduler.plan(
                state=state,
                think_dist=think_dist,
                compute_elapsed_s=compute_elapsed,
                rng=self.rng,
            )
            timing_debug["think_mean"] = think_dist.mean()

        debug_info: dict[str, Any] = {
            "compute_elapsed_s": compute_elapsed,
            "policy_entropy": dist.entropy,
            "top_move": dist.top_move.uci() if dist.top_move else None,
            "top_probability": dist.top_probability,
            "legal_move_count": dist.legal_move_count,
            "position_wdl": dist.wdl,
            **timing_debug,
        }

        return Decision(
            move=chosen_move,
            delay_s=delay_s,
            debug=debug_info,
        )
