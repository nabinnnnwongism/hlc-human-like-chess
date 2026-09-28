"""Move timing scheduler implementing safety constraints and clock budgeting."""

from __future__ import annotations

import random
from dataclasses import dataclass

from hlc.timing.base import ThinkTimeDistribution
from hlc.types import GameState


@dataclass
class SchedulerConfig:
    """Configuration tokens for Scheduler."""

    move_overhead: float = 0.10  # Estimated network/adapter lag (seconds)
    safety_margin: float = 0.50  # Hard floor buffer to prevent flagging (seconds)
    max_clock_fraction: float = 0.12  # Maximum fraction of own remaining clock per move
    min_delay: float = 0.05  # Absolute minimum wait delay (seconds)
    single_move_delay: float = 0.15  # Near-instant delay for forced/single legal moves (seconds)
    time_trouble_threshold: float = 10.0  # Seconds remaining where time trouble engagement starts
    extreme_time_trouble_threshold: float = (
        3.0  # Seconds remaining where delay collapses to minimum
    )


class Scheduler:
    """Translates think-time distributions into concrete, clock-safe delays in seconds."""

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self.cfg = config if config is not None else SchedulerConfig()

    def plan(
        self,
        state: GameState,
        think_dist: ThinkTimeDistribution,
        compute_elapsed_s: float = 0.0,
        rng: random.Random | None = None,
    ) -> float:
        """Plan the exact delay in seconds for the chosen move.

        Rules applied:
        1. Single legal move (forced recapture/check evasion) -> near-instant response.
        2. Sample think time from distribution, subtract bot compute time: max(0, sample - compute).
        3. Never exceed max_clock_fraction of own clock.
        4. When clock is under time_trouble_threshold, progressively collapse toward min_delay.
        5. Absolute hard bound: leave at least move_overhead + safety_margin on the clock.
        """
        clock = max(0.0, state.clock_self)
        hard_limit = max(0.0, clock - (self.cfg.move_overhead + self.cfg.safety_margin))

        # Critical clock emergency: virtually zero budget left
        if hard_limit <= 0.0:
            return 0.0

        # Rule 1: Single legal move -> near-instant response
        if state.board.legal_moves.count() <= 1:
            forced_delay = max(0.0, self.cfg.single_move_delay - compute_elapsed_s)
            return min(forced_delay, hard_limit)

        # Rule 2: Sample think time from distribution, deduct compute elapsed
        raw_sample = think_dist.sample(rng)
        target_delay = max(0.0, raw_sample - compute_elapsed_s)

        # Rule 3: Cap at max clock fraction
        fraction_cap = self.cfg.max_clock_fraction * clock
        capped_delay = min(target_delay, fraction_cap)

        # Rule 4: Progressive collapse during time trouble
        if clock < self.cfg.time_trouble_threshold:
            if clock <= self.cfg.extreme_time_trouble_threshold:
                trouble_delay = self.cfg.min_delay
            else:
                progress = (clock - self.cfg.extreme_time_trouble_threshold) / (
                    self.cfg.time_trouble_threshold - self.cfg.extreme_time_trouble_threshold
                )
                trouble_delay = self.cfg.min_delay + progress * (capped_delay - self.cfg.min_delay)
            capped_delay = min(capped_delay, max(self.cfg.min_delay, trouble_delay))

        # Rule 5: Hard safety boundary (never flag!)
        final_delay = min(capped_delay, hard_limit)

        # Apply minimum delay floor if within hard limit
        if final_delay < self.cfg.min_delay and hard_limit >= self.cfg.min_delay:
            final_delay = self.cfg.min_delay

        return float(max(0.0, final_delay))
