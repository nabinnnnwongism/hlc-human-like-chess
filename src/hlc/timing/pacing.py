"""pacing.py — Human pacing model with soft saturation, AR(1) tempo, and visual reaction floor.

Adapted from human pacing research:
- PacingModel: Computes move thinking time using log-normal distributions modulated by:
    - Phase shape (smooth sigmoids: book opening, deep middlegame, rapid ending)
    - Position difficulty (Maia policy entropy and top-move probability)
    - AR(1) autocorrelated tempo streaks (rho=0.40)
    - Soft saturation via hyperbolic tangent (cap * tanh(t / cap)) preventing hard clamping bins
    - Calibrated instant/premove probability with a visual perception floor (min_s=0.12s)
- VirtualClock: Accurately simulates clock decay for untimed / casual games.
- sleep_remaining: Real wall-clock countdown deducting inference and platform latency.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class ClockInfo:
    """Represents current game clock state in seconds."""
    timed: bool          # False for unlimited / casual / computer games
    initial_s: float     # Initial time bank
    inc_s: float         # Increment per move
    my_s: float          # Our remaining time
    opp_s: float         # Opponent's remaining time


class VirtualClock:
    """For untimed games: simulates a realistic clock countdown so pacing feels natural."""

    def __init__(self, initial_s: float = 600.0, inc_s: float = 0.0) -> None:
        self.initial_s = float(initial_s)
        self.inc_s = float(inc_s)
        self.my_s = float(initial_s)
        self.opp_s = float(initial_s)

    def after_our_move(self, spent_s: float) -> None:
        self.my_s = max(1.0, self.my_s - spent_s + self.inc_s)

    def after_opp_move(self, spent_s: float) -> None:
        self.opp_s = max(1.0, self.opp_s - spent_s + self.inc_s)

    def info(self) -> ClockInfo:
        return ClockInfo(
            timed=False,
            initial_s=self.initial_s,
            inc_s=self.inc_s,
            my_s=self.my_s,
            opp_s=self.opp_s,
        )


@dataclass
class MoveFeatures:
    """Position and context features passed to the pacing model."""
    n: int                                # Our move number (1-based fullmove or turn index)
    entropy: float                        # Maia policy entropy in nats
    top_p: float                          # Maia top-move probability
    clock: ClockInfo                      # Current clock state
    n_legal: int = 30                     # Number of legal moves
    is_recapture: bool = False            # Recapturing a just-captured piece
    is_promotion: bool = False            # Move is a pawn promotion
    in_check: bool = False                # King is in check
    opp_last_s: Optional[float] = None    # Opponent's last measured think time in seconds
    surprise: float = 0.0                 # Opponent surprise metric in [0, 1]


@dataclass
class PacingParams:
    """Calibrated parameters controlling human move pacing."""
    # Tempo & Budget
    usage: float = 0.36                   # Share of per-move budget a typical move uses
    exp_total_moves: int = 45             # Expected moves in a game for budgeting
    reserve_s: float = 5.0                # Safety clock reserve

    # Phase shape (smooth sigmoids: fast opening, slow middlegame, faster ending)
    open_n: int = 8
    open_mult: float = 0.50
    mid_mult: float = 1.25
    end_n: int = 32
    end_mult: float = 0.90

    # Difficulty coupling
    ent_mu: float = 1.2
    ent_sd: float = 0.7
    ent_beta: float = 0.32
    surprise_gain: float = 0.35

    # Autocorrelation & noise structure
    rho: float = 0.40                     # AR(1) persistence coefficient
    sigma_ar: float = 0.45                # AR(1) innovation variance
    sigma_idio: float = 0.62              # Idiosyncratic noise
    sigma_game: float = 0.20              # Game-level variance ("form")
    opp_coupling: float = 0.12            # Opponent tempo mirroring

    # Premove / instant mixture
    p_instant_base: float = 0.04
    p_instant_forced: float = 0.55
    p_instant_pressure: float = 0.25
    instant_median_s: float = 0.32
    instant_sigma: float = 0.55

    # Safety bounds & visual reaction floor
    pressure_s: float = 30.0              # Clock threshold for engaging time-trouble scaling
    cap_frac: float = 0.18                # Maximum fraction of clock for deliberate moves
    min_s: float = 0.12                   # Hard visual reaction floor (never 0.00s)

    @classmethod
    def from_json(cls, path: str) -> PacingParams:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


@dataclass
class Pacing:
    """Result of pacing calculation for a turn."""
    seconds: float          # Target total wall-clock duration from move receipt to click
    premove_like: bool      # Whether this was sampled from the instant/premove branch
    p_instant: float        # Evaluated probability of instant response


def _sig(x: float) -> float:
    """Logistic sigmoid function."""
    return 1.0 / (1.0 + math.exp(-x))


class PacingModel:
    """Maintains AR(1) state, opponent tempo coupling, and samples human delays."""

    def __init__(
        self,
        params: Optional[PacingParams] = None,
        rng: Optional[random.Random] = None,
        persona_tempo: float = 0.0,
    ) -> None:
        self.p = params or PacingParams()
        self.rng = rng or random.Random()
        # Per-game form offset + persona tempo habit
        self.game_effect = self.rng.gauss(0.0, self.p.sigma_game) + persona_tempo
        self.ar = 0.0
        self._opp_mean: Optional[float] = None

    def reset(self) -> None:
        """Reset AR(1) autocorrelation state for a new game."""
        self.ar = 0.0
        self._opp_mean = None
        self.game_effect = self.rng.gauss(0.0, self.p.sigma_game)

    def _forced_score(self, f: MoveFeatures) -> float:
        """Score how obvious/forced the reply is (0.0 to 1.0)."""
        s = 0.0
        if f.n_legal <= 1:
            s = 1.0
        if f.is_recapture or f.is_promotion:
            s = max(s, 0.85)
        if f.in_check and f.n_legal <= 3:
            s = max(s, 0.70)
        # Obviousness ramps smoothly with top-move probability (0 at p<=0.5, 0.8 at p>=0.9)
        s = max(s, 0.80 * min(1.0, max(0.0, (f.top_p - 0.5) / 0.4)))
        if f.n <= 3:
            s = max(s, 0.60)
        return s

    def _lognorm(self, median: float, sigma: float) -> float:
        return median * math.exp(self.rng.gauss(0.0, sigma))

    def _update_opp(self, f: MoveFeatures) -> None:
        if f.opp_last_s is not None:
            if self._opp_mean is None:
                self._opp_mean = f.opp_last_s
            else:
                self._opp_mean = 0.8 * self._opp_mean + 0.2 * f.opp_last_s

    def sample(self, f: MoveFeatures) -> Pacing:
        """Sample a target wall-clock think time in seconds."""
        p, rng, clk = self.p, self.rng, f.clock
        my = clk.my_s if math.isfinite(clk.my_s) else 600.0
        self._update_opp(f)

        # ── (a) Premove / Instant branch: obvious moves, not high-entropy ones ──
        forced = self._forced_score(f)
        p_inst = p.p_instant_base + p.p_instant_forced * forced
        # Damp instant moves when entropy is high (complex positions demand thinking)
        p_inst *= 1.0 - 0.30 * min(1.0, max(0.0, (f.entropy - p.ent_mu) / 1.2))
        if my < p.pressure_s:
            p_inst += p.p_instant_pressure * (1.0 - my / p.pressure_s)
        p_inst = min(0.85, max(0.0, p_inst))

        if rng.random() < p_inst:
            self.ar *= p.rho
            t = self._lognorm(p.instant_median_s, p.instant_sigma)
            # Enforce visual perception floor (min_s) so moves never look instant (0.00s)
            target = min(max(t, p.min_s), 1.5, 0.4 * my)
            return Pacing(target, True, p_inst)

        # ── (b) Deliberate branch: budget -> phase -> difficulty -> noise ────────
        moves_left = max(10, p.exp_total_moves - f.n + 5)
        per_move = max(0.0, my - p.reserve_s) / moves_left + 0.8 * clk.inc_s
        phase = (
            p.open_mult
            + (p.mid_mult - p.open_mult) * _sig((f.n - p.open_n) / 2.0)
            + (p.end_mult - p.mid_mult) * _sig((f.n - p.end_n) / 3.0)
        )
        z_ent = (f.entropy - p.ent_mu) / p.ent_sd
        logt = math.log(max(per_move * p.usage, 0.3)) + math.log(phase)
        logt += p.ent_beta * z_ent
        logt += math.log(1.0 + p.surprise_gain * max(0.0, min(1.0, f.surprise)))

        if f.opp_last_s is not None and self._opp_mean is not None:
            logt += p.opp_coupling * math.log((f.opp_last_s + 0.5) / (self._opp_mean + 0.5))

        self.ar = p.rho * self.ar + math.sqrt(1 - p.rho ** 2) * rng.gauss(0.0, 1.0)
        logt += self.game_effect + p.sigma_ar * self.ar + p.sigma_idio * rng.gauss(0.0, 1.0)
        t = math.exp(logt)

        if my < p.pressure_s:
            # Genuine time trouble
            t *= max(0.05, my / p.pressure_s) ** 0.8

        # Soft saturation via hyperbolic tangent (never hard clamps or creates bins)
        cap = max(3.0, p.cap_frac * my + 0.5 * clk.inc_s)
        t = cap * math.tanh(t / cap)
        t = min(t, 0.5 * my)  # Hard anti-flagging safety

        final_s = max(t, min(p.min_s, 0.25 * my))
        return Pacing(final_s, False, p_inst)


def sleep_remaining(t_received_monotonic: float, target_s: float) -> None:
    """Sleep for the remainder of the target duration.

    Engine inference + network lag count towards thinking time.
    Ensures the overall delay equals target_s without double-counting.
    """
    elapsed = time.monotonic() - t_received_monotonic
    left = target_s - elapsed
    if left > 0.0:
        time.sleep(left)


async def async_sleep_remaining(t_received_monotonic: float, target_s: float) -> None:
    """Asynchronous variant of sleep_remaining."""
    elapsed = time.monotonic() - t_received_monotonic
    left = target_s - elapsed
    if left > 0.0:
        await asyncio.sleep(left)
