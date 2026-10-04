"""BotCore: Coordinates MoveEngine, TimingModel, Scheduler, and Playstyle.

Timing upgrades (v3):
  1. ELO-interpolated log-normal sigma — HeuristicBaseline now adjusts variance
     to match real Lichess timing data at the player's ELO.
  2. Premove system — ~21% of moves are near-instant (<180ms) when the position
     is forced/recapture/opening/momentum-following (nixec port).
  3. AR(1) autocorrelation (rho=0.4) — each move's think time is correlated
     with the previous, producing natural streaks of fast/slow moves.
"""

from __future__ import annotations

import random
import time
from typing import Any

from hlc.engines.guard import TacticalGuard
from hlc.playstyle import (
    StyleVector,
    apply_playstyle,
    apply_game_tilt,
    apply_time_control_offset,
    get_style_vector,
)
from hlc.timing.baselines import ThinkTimeState, should_premove, premove_delay
from hlc.timing.pacing import ClockInfo, MoveFeatures, PacingModel, PacingParams
from hlc.types import Decision, GameState, MoveDistribution, MoveEngine


class BotCore:
    """Core brain of the Human-Like Chess bot.

    Orchestrates move selection, playstyle biasing, timing distribution prediction,
    and delay scheduling.

    Version 3 additions (on top of v2 jitter + tilt + time-control offset):
      - Premove system: near-instant replies for forced/recapture/opening positions.
      - AR(1) autocorrelation: think times correlated across consecutive moves.
      - ELO-aware sigma: timing variance matches player ELO.
    """

    def __init__(
        self,
        move_engine: MoveEngine,
        timing_model: Any = None,
        scheduler: Any = None,
        rng: random.Random | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        playstyle: str | StyleVector | None = None,
        playstyle_elo: int | None = None,
        time_control: str = "blitz",
        enable_jitter: bool = True,
        enable_tilt: bool = True,
        enable_premove: bool = True,
        enable_ar1: bool = True,
        enable_guard: bool = True,
        pacing_model: PacingModel | None = None,
        tactical_guard: TacticalGuard | None = None,
    ) -> None:
        self.move_engine = move_engine
        self.timing_model = timing_model
        self.scheduler = scheduler
        self.rng = rng if rng is not None else random.Random(42)
        self.temperature = temperature
        self.top_p = top_p
        self.playstyle_name: str | None = playstyle if isinstance(playstyle, str) else None
        self.playstyle_elo: int | None = playstyle_elo
        self.time_control = time_control
        self.enable_jitter = enable_jitter
        self.enable_tilt = enable_tilt
        self.enable_premove = enable_premove
        self.enable_ar1 = enable_ar1
        self.enable_guard = enable_guard

        # Base StyleVector (before jitter/tilt)
        self._base_style: StyleVector | None = None
        # Per-game jittered vector (applied once per game at first decide() call)
        self.style_vector: StyleVector | None = None
        self._jitter_applied: bool = False

        # Upgrade 3: AR(1) think-time state (carries correlation across moves)
        self._tt_state = ThinkTimeState()

        # Tactical Guard (converts trivial mate-in-1 and prevents stalemates)
        if tactical_guard is not None:
            self.tactical_guard = tactical_guard
        elif enable_guard:
            self.tactical_guard = TacticalGuard(elo=playstyle_elo or 1500, rng=self.rng)
        else:
            self.tactical_guard = None

        # Pacing Model (soft saturation, realistic floors, smooth phase transitions)
        if pacing_model is not None:
            self.pacing_model = pacing_model
        else:
            self.pacing_model = PacingModel(rng=self.rng)

        self._resolve_style(playstyle, playstyle_elo)

    def _resolve_style(
        self,
        playstyle: str | StyleVector | None,
        elo: int | None,
    ) -> None:
        """Resolve and store the base style vector."""
        if isinstance(playstyle, StyleVector):
            base = playstyle
        elif isinstance(playstyle, str):
            base = get_style_vector(
                playstyle,
                elo=elo if elo is not None else 1500,
                rng=self.rng,
            )
        else:
            base = None

        if base is not None:
            base = apply_time_control_offset(base, self.time_control)

        self._base_style = base
        self.style_vector = base
        self._jitter_applied = False

    def new_game(self) -> None:
        """Call at the start of each new game to reset per-game state."""
        self._jitter_applied = False
        self.style_vector = self._base_style
        # Upgrade 3: reset AR(1) autocorrelation — each game is independent
        self._tt_state.reset()
        if self.pacing_model is not None:
            self.pacing_model.reset()

    def set_playstyle(
        self,
        playstyle: str | StyleVector | None,
        elo: int | None = None,
    ) -> None:
        """Update or reset the active playstyle."""
        self.playstyle_name = playstyle if isinstance(playstyle, str) else None
        self.playstyle_elo = elo if elo is not None else self.playstyle_elo
        self._resolve_style(playstyle, self.playstyle_elo)

    def decide(
        self,
        state: GameState,
        t_received_monotonic: float | None = None,
    ) -> Decision:
        """Process the game state, choose a move, and schedule its think delay."""
        t_start = time.perf_counter()

        # 1. Compute move distribution from move engine
        dist: MoveDistribution = self.move_engine.get_distribution(state)
        if not dist.moves:
            raise RuntimeError(f"No legal moves available in position: {state.board.fen()}")

        # 1b. Resolve live style vector (jitter + tilt)
        live_style = self._get_live_style(state)

        # 1c. Apply playstyle re-weighting if configured
        if live_style is not None:
            reweighted_probs = apply_playstyle(
                board=state.board,
                moves=dist.moves,
                probabilities=dist.probabilities,
                style=live_style,
                our_color=state.turn,
                ply=state.ply_count,
            )
            dist = MoveDistribution(
                moves=dist.moves,
                probabilities=reweighted_probs,
                candidate_details=dist.candidate_details,
                wdl=dist.wdl,
            )

        # 2. Sample move according to temperature and top-p settings
        chosen_move = dist.sample(
            rng=self.rng,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        # 2b. Tactical guard: check mate-in-1 and avoid catastrophic stalemates
        guard_reason = "ok"
        if self.tactical_guard is not None:
            ranked = list(zip(dist.moves, dist.probabilities))
            guard_res = self.tactical_guard.apply(
                board=state.board,
                chosen=chosen_move,
                ranked=ranked,
                my_clock_s=state.clock_self,
            )
            chosen_move = guard_res.move
            guard_reason = guard_res.reason

        compute_elapsed = time.perf_counter() - t_start

        # 3. Timing: PacingModel (with soft saturation & visual floor) or legacy scheduler
        delay_s = 0.0
        is_premove = False
        timing_debug: dict[str, Any] = {}
        if guard_reason != "ok":
            timing_debug["guard_reason"] = guard_reason

        if self.pacing_model is not None:
            # Check recapture: opponent captured last turn and we take back on same square
            last_move = state.board.peek() if state.board.move_stack else None
            prev_board = state.board.copy()
            is_recapture = False
            if last_move:
                try:
                    prev_board.pop()
                    is_recapture = bool(
                        prev_board.is_capture(last_move)
                        and state.board.is_capture(chosen_move)
                        and chosen_move.to_square == last_move.to_square
                    )
                except Exception:
                    pass

            is_timed = 0.0 < state.clock_self < 3600.0
            clk_info = ClockInfo(
                timed=is_timed,
                initial_s=max(state.clock_self, 180.0),
                inc_s=state.increment,
                my_s=state.clock_self,
                opp_s=state.clock_opp,
            )

            feats = MoveFeatures(
                n=state.board.fullmove_number,
                entropy=dist.entropy,
                top_p=dist.top_probability,
                clock=clk_info,
                n_legal=dist.legal_move_count,
                is_recapture=is_recapture,
                is_promotion=bool(chosen_move.promotion),
                in_check=state.board.is_check(),
            )
            pace = self.pacing_model.sample(feats)
            is_premove = pace.premove_like

            if t_received_monotonic is not None:
                elapsed_since_rcv = time.monotonic() - t_received_monotonic
                delay_s = max(self.pacing_model.p.min_s, pace.seconds - elapsed_since_rcv)
            else:
                delay_s = max(self.pacing_model.p.min_s, pace.seconds - compute_elapsed)

            timing_debug["p_instant"] = pace.p_instant
            timing_debug["target_pacing_s"] = pace.seconds
            timing_debug["think_mean"] = pace.seconds
            timing_debug["premove"] = is_premove

        elif self.timing_model is not None and self.scheduler is not None:
            # Legacy scheduler fallback
            if self.enable_premove:
                is_premove = should_premove(
                    state=state,
                    move_dist=dist,
                    last_delay_s=self._tt_state.last_delay_s,
                    rng=self.rng,
                )

            if is_premove:
                raw_delay = premove_delay(rng=self.rng)
                delay_s = max(0.0, raw_delay - compute_elapsed)
                self._tt_state.last_delay_s = raw_delay
                self._tt_state._prev_normalized = -1.5
                timing_debug["premove"] = True
            else:
                think_dist = self.timing_model.predict(state, dist)
                if self.enable_ar1:
                    mean_s = think_dist.mean()
                    import math as _math
                    sigma_approx = max(0.30, _math.sqrt(_math.log(
                        1 + ((mean_s * 0.7) / max(mean_s, 0.001)) ** 2
                    )))
                    mu = _math.log(max(mean_s, 0.001)) - 0.5 * sigma_approx ** 2
                    raw_think = self._tt_state.sample_lognormal(
                        mu=mu, sigma=sigma_approx, rng=self.rng
                    )
                    raw_think = min(raw_think, max(0.0, state.clock_self - 0.6))
                    hard_limit = max(0.0, state.clock_self - 0.60)
                    speed_cap = _get_speed_cap(state.clock_self)
                    capped = min(raw_think, speed_cap, hard_limit)
                    delay_s = max(0.0, capped - compute_elapsed)
                else:
                    delay_s = self.scheduler.plan(
                        state=state,
                        think_dist=think_dist,
                        compute_elapsed_s=compute_elapsed,
                        rng=self.rng,
                    )
                    self._tt_state.last_delay_s = delay_s

                timing_debug["think_mean"] = think_dist.mean()
                timing_debug["premove"] = False

        debug_info: dict[str, Any] = {
            "compute_elapsed_s": compute_elapsed,
            "policy_entropy": dist.entropy,
            "top_move": dist.top_move.uci() if dist.top_move else None,
            "top_probability": dist.top_probability,
            "legal_move_count": dist.legal_move_count,
            "position_wdl": dist.wdl,
            "playstyle": self.playstyle_name or ("custom" if self.style_vector else None),
            "time_control": self.time_control,
            "is_premove": is_premove,
            **timing_debug,
        }

        return Decision(
            move=chosen_move,
            delay_s=delay_s,
            debug=debug_info,
        )


    def _get_live_style(self, state: GameState) -> StyleVector | None:
        """Return the style vector for this specific move, with jitter and tilt."""
        if self._base_style is None:
            return None

        if self.enable_jitter and not self._jitter_applied:
            self.style_vector = self._base_style.jitter(self.rng, sigma=0.06)
            self._jitter_applied = True

        sv = self.style_vector or self._base_style

        if self.enable_tilt:
            mat_balance = _estimate_material_balance(state)
            clock_frac = _estimate_clock_fraction(state)
            sv = apply_game_tilt(sv, mat_balance, clock_frac)

        return sv


# ── Speed cap helper (mirrors Scheduler speed tiers for AR(1) path) ──────────

def _get_speed_cap(clock_s: float) -> float:
    """Return the pause cap for the current clock zone (mirrors scheduler.py tiers)."""
    if clock_s >= 300.0:
        return 12.0
    if clock_s >= 180.0:
        return 6.0
    if clock_s >= 90.0:
        return 3.5
    if clock_s >= 30.0:
        return 2.0
    if clock_s >= 10.0:
        return 0.9
    return 0.35


# ── Helpers ───────────────────────────────────────────────────────────────────

import chess as _chess


def _estimate_material_balance(state: GameState) -> int:
    """Estimate material balance (our_color perspective) in pawn units."""
    _VALUES = {
        _chess.PAWN: 1, _chess.KNIGHT: 3, _chess.BISHOP: 3,
        _chess.ROOK: 5, _chess.QUEEN: 9,
    }
    our = 0
    theirs = 0
    our_color = state.turn
    for sq in _chess.SQUARES:
        p = state.board.piece_at(sq)
        if p is None or p.piece_type == _chess.KING:
            continue
        val = _VALUES.get(p.piece_type, 0)
        if p.color == our_color:
            our += val
        else:
            theirs += val
    return our - theirs


def _estimate_clock_fraction(state: GameState) -> float:
    """Estimate clock fraction remaining (our_color's clock), in [0, 1]."""
    try:
        remaining = state.clock_self
        total = state.clock_self + state.clock_opp
        if total <= 0:
            return 0.5
        return max(0.0, min(1.0, remaining / (total / 2.0)))
    except Exception:
        return 0.5
