"""HLCEngine: lichess-bot homemade engine adapter.

Drop this module into the lichess-bot repo's homemade.py (or import from it),
then set `protocol: homemade` and `name: HLCEngine` in config.yml.

The adapter translates chess.engine.Limit (Lichess clocks) into hlc.types.GameState,
calls BotCore.decide(), waits the scheduled delay, then returns PlayResult.

IMPORTANT: Set `fake_think_time: false` in config.yml's engine section so
lichess-bot does not add its own random sleep on top of ours.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

import chess
import chess.engine

from hlc.core import BotCore
from hlc.engines.maia3 import Maia3Engine
from hlc.scheduler import Scheduler, SchedulerConfig
from hlc.timing.baselines import HeuristicBaseline
from hlc.types import GameState

logger = logging.getLogger(__name__)


def _build_bot_core(
    model: str = "maia3-79m",
    elo: int = 1500,
    device: str = "cpu",
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: int = 42,
    move_overhead: float = 0.10,
    safety_margin: float = 0.50,
    max_clock_fraction: float = 0.12,
    min_delay: float = 0.05,
    single_move_delay: float = 0.15,
    timing_backend: str = "heuristic",
    playstyle: str | Any = None,
    time_control: str = "blitz",
    enable_jitter: bool = True,
    enable_tilt: bool = True,
) -> BotCore:
    """Construct a fully configured BotCore instance."""
    move_engine = Maia3Engine(
        model=model,
        elo=elo,
        device=device,
        use_amp=False,
        use_uci_history=True,
        multipv=3,
    )

    timing_model: HeuristicBaseline
    if timing_backend == "heuristic":
        from hlc.timing.baselines import HeuristicBaseline
        # Upgrade: pass elo so sigma adapts to player ELO (nixec research)
        timing_model = HeuristicBaseline(elo=elo)
    else:
        from hlc.timing.baselines import ClockOnlyBaseline
        timing_model = ClockOnlyBaseline(elo=elo)  # type: ignore[assignment]

    scheduler = Scheduler(
        SchedulerConfig(
            move_overhead=move_overhead,
            safety_margin=safety_margin,
            max_clock_fraction=max_clock_fraction,
            min_delay=min_delay,
            single_move_delay=single_move_delay,
        )
    )

    return BotCore(
        move_engine=move_engine,
        timing_model=timing_model,
        scheduler=scheduler,
        rng=random.Random(seed),
        temperature=temperature,
        top_p=top_p,
        playstyle=playstyle,
        playstyle_elo=elo,
        time_control=time_control,
        enable_jitter=enable_jitter,
        enable_tilt=enable_tilt,
        enable_premove=True,   # Upgrade: ~21% near-instant moves for forced/recaptures
        enable_ar1=True,       # Upgrade: AR(1) autocorrelated think times (rho=0.4)
    )


class HLCEngine:
    """Human-Like Chess engine for lichess-bot's homemade engine protocol.

    Subclass MinimalEngine when used inside the lichess-bot repo:

        from lib.engine_wrapper import MinimalEngine
        from hlc.adapters.lichess_bot_engine import HLCEngine as _HLCCore

        class HLCEngine(MinimalEngine):
            # ... delegate to _HLCCore.search(...)

    This module provides the standalone HLCEngine implementation.
    The companion lichess_bot_hook.py (at the project root) provides
    the actual MinimalEngine subclass to drop into lichess-bot.
    """

    def __init__(
        self,
        model: str = "maia3-79m",
        elo: int = 1500,
        device: str = "cpu",
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int = 42,
        move_overhead: float = 0.10,
        safety_margin: float = 0.50,
        max_clock_fraction: float = 0.12,
        min_delay: float = 0.05,
        single_move_delay: float = 0.15,
        timing_backend: str = "heuristic",
        opp_elo: int = 1500,
        playstyle: str | Any = None,
        time_control: str = "blitz",
        enable_jitter: bool = True,
        enable_tilt: bool = True,
    ) -> None:
        self.elo = elo
        self.opp_elo = opp_elo
        self.playstyle = playstyle
        self.time_control = time_control
        self._bot = _build_bot_core(
            model=model,
            elo=elo,
            device=device,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            move_overhead=move_overhead,
            safety_margin=safety_margin,
            max_clock_fraction=max_clock_fraction,
            min_delay=min_delay,
            single_move_delay=single_move_delay,
            timing_backend=timing_backend,
            playstyle=playstyle,
            time_control=time_control,
            enable_jitter=enable_jitter,
            enable_tilt=enable_tilt,
        )
        logger.info(
            "HLCEngine initialised: model=%s elo=%d device=%s timing=%s playstyle=%s tc=%s",
            model,
            elo,
            device,
            timing_backend,
            playstyle or "default",
            time_control,
        )

    def new_game(self) -> None:
        """Notify engine that a new game has started (resets jitter and dynamic tilt)."""
        self._bot.new_game()

    def search(
        self,
        board: chess.Board,
        time_limit: chess.engine.Limit,
        ponder: bool,
        draw_offered: bool,
    ) -> chess.engine.PlayResult:
        """Select a move and wait the human-like delay before returning.

        Parameters match the lichess-bot MinimalEngine.search signature:
            board        — current position
            time_limit   — chess.engine.Limit with white_clock/black_clock/white_inc/black_inc
            ponder       — ignored (we don't ponder)
            draw_offered — ignored (no draw logic yet)

        Returns chess.engine.PlayResult(move, None).
        """
        # --- Extract remaining clocks ----------------------------------------
        if board.turn == chess.WHITE:
            clock_self = float(time_limit.white_clock or 30.0)
            clock_opp = float(time_limit.black_clock or 30.0)
            increment = float(time_limit.white_inc or 0.0)
        else:
            clock_self = float(time_limit.black_clock or 30.0)
            clock_opp = float(time_limit.white_clock or 30.0)
            increment = float(time_limit.black_inc or 0.0)
        # --- Reset per-game jitter when starting a new game ------------------
        if len(board.move_stack) <= 1:
            self._bot.new_game()

        state = GameState(
            board=board.copy(),
            move_history=list(board.move_stack),
            clock_self=max(0.05, clock_self),
            clock_opp=max(0.05, clock_opp),
            increment=increment,
            self_elo=self.elo,
            opp_elo=self.opp_elo,
        )

        # --- Decide (move + delay) -------------------------------------------
        t0 = time.perf_counter()
        decision = self._bot.decide(state)
        compute_elapsed = time.perf_counter() - t0

        # --- Wait the scheduled delay ----------------------------------------
        remaining_sleep = max(0.0, decision.delay_s - compute_elapsed)
        if remaining_sleep > 0.0:
            time.sleep(remaining_sleep)

        actual_think = time.perf_counter() - t0

        logger.debug(
            "HLCEngine: move=%s delay_planned=%.2fs actual=%.2fs "
            "clock_self=%.1fs entropy=%.2f top_p=%.2f",
            decision.move.uci(),
            decision.delay_s,
            actual_think,
            clock_self,
            decision.debug.get("policy_entropy", 0.0),
            decision.debug.get("top_probability", 0.0),
        )

        return chess.engine.PlayResult(decision.move, None)
