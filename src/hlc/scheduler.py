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
    time_trouble_threshold: float = 90.0  # Seconds remaining where progressive scaling engages
    extreme_time_trouble_threshold: float = 12.0  # Seconds remaining where delay collapses near minimum
    moves_buffer: float = 25.0  # Mathematical budget: reserve clock across at least 25 moves


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

        Option C Dynamic Adaptive Pacing:
        1. Single legal move (forced recapture/check evasion) -> near-instant response.
        2. Dynamic speed tiers based on clock:
           - Comfort Zone (> 5m): Full human variance, think times up to 12s on complex moves.
           - Brisk Zone (3m - 5m): 20% faster, maximum pause capped at 6.0s.
           - Fast Zone (1.5m - 3m): 40% faster, maximum pause capped at 3.5s.
           - Time Trouble (30s - 90s): 65% faster, moves in 0.8s - 2.0s, no long pauses.
           - Scramble (10s - 30s): Rapid fire 0.3s - 0.9s.
           - Panic (< 10s): High-speed bullet 0.05s - 0.35s.
        3. Mathematical Anti-Flagging Cap:
           - delay <= (clock - safety_margin) / moves_buffer (guarantees clock buffer for 25 moves)
        4. Absolute hard bound: leave at least move_overhead + safety_margin on the clock.
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

        # Rule 2: Dynamic Speed Multiplier and Pause Cap based on remaining clock
        if clock >= 300.0:    # > 5 min: Comfort zone
            speed_mult = 1.00
            pause_cap = 12.0
        elif clock >= 180.0:  # 3 - 5 min: Brisk zone
            speed_mult = 0.80
            pause_cap = 6.0
        elif clock >= 90.0:   # 1.5 - 3 min: Fast zone
            speed_mult = 0.60
            pause_cap = 3.5
        elif clock >= 30.0:   # 30s - 90s: Time trouble zone
            speed_mult = 0.35
            pause_cap = 2.0
        elif clock >= 10.0:   # 10s - 30s: Scramble zone
            speed_mult = 0.18
            pause_cap = 0.9
        else:                 # < 10s: Panic bullet zone
            speed_mult = 0.08
            pause_cap = 0.35

        # Sample think time and scale by clock speed tier, capped by pause limit
        raw_sample = think_dist.sample(rng)
        scaled_think = min(raw_sample * speed_mult, pause_cap)
        target_delay = max(0.0, scaled_think - compute_elapsed_s)

        # Rule 4: Mathematical Anti-Flagging Budget Cap
        # Reserve clock over at least moves_buffer (default 25 moves)
        budget_cap = max(self.cfg.min_delay, (clock - self.cfg.safety_margin) / self.cfg.moves_buffer)
        capped_delay = min(target_delay, budget_cap)

        # Rule 5: Cap at max clock fraction
        fraction_cap = self.cfg.max_clock_fraction * clock
        capped_delay = min(capped_delay, fraction_cap)

        # Rule 6: Progressive collapse during extreme time trouble (< 12s)
        if clock < self.cfg.extreme_time_trouble_threshold:
            capped_delay = min(capped_delay, max(self.cfg.min_delay, clock * 0.05))

        # Rule 7: Hard safety boundary (never flag!)
        final_delay = min(capped_delay, hard_limit)

        # Apply minimum delay floor if within hard limit
        if final_delay < self.cfg.min_delay and hard_limit >= self.cfg.min_delay:
            final_delay = self.cfg.min_delay

        return float(max(0.0, final_delay))
