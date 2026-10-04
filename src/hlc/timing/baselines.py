"""Baseline move timing models with three human-behavior upgrades:

1. ELO-interpolated log-normal sigma (from nixec/Lichess 12M-game research).
2. Premove detection: ~21% of moves are near-instant (<200ms) recaptures/forced.
3. AR(1) autocorrelation (rho=0.4): each move's think time correlates with previous.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from hlc.timing.base import DEFAULT_BUCKET_EDGES, ThinkTimeDistribution, TimingModel
from hlc.types import GameState, MoveDistribution


def _normal_cdf(x: float) -> float:
    """Standard normal CDF Phi(x)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _lognormal_bucket_probs(
    mu: float,
    sigma: float,
    bucket_edges: Sequence[float],
    min_time: float = 0.05,
) -> list[float]:
    """Compute discretized bucket probabilities for LogNormal(mu, sigma^2)."""
    probs: list[float] = []
    num_buckets = len(bucket_edges) - 1

    for i in range(num_buckets):
        low = max(bucket_edges[i], min_time)
        high = max(bucket_edges[i + 1], min_time)
        z_high = (math.log(high) - mu) / sigma
        z_low = (math.log(low) - mu) / sigma
        p = _normal_cdf(z_high) - _normal_cdf(z_low)
        probs.append(max(0.0, p))

    # Add tail probability to last bucket
    tail_z = (math.log(max(bucket_edges[-1], min_time)) - mu) / sigma
    tail_prob = max(0.0, 1.0 - _normal_cdf(tail_z))
    if probs:
        probs[-1] += tail_prob

    total = sum(probs)
    if total > 0.0:
        return [p / total for p in probs]
    return [1.0 / num_buckets] * num_buckets


# ── Upgrade 1: ELO-interpolated sigma ────────────────────────────────────────
# Derived from nixec's Lichess 12M-game dataset analysis.
# At lower ELOs, SD is tighter (more consistent but slower).
# At higher ELOs, SD is wider with heavier tails (more variance, faster moves).
#
# Linear interpolation: SD = a + b * Mean
# where (a, b) is interpolated between:
#   ELO ≤ 1200 → (a=0.08, b=0.72)   low-rated: narrower distribution
#   ELO ~1500  → (a=0.20, b=0.91)   mid-rated: standard blitz timing
#   ELO ≥ 2000 → (a=0.50, b=1.28)   high-rated: wide variance, frequent premoves

_ELO_KNOTS = [1200, 1500, 1800, 2000]
_A_KNOTS    = [0.08, 0.20, 0.38, 0.50]   # intercept coefficients
_B_KNOTS    = [0.72, 0.91, 1.12, 1.28]   # slope coefficients (SD per second of mean)


def _elo_interpolate(elo: int, knots: list[float], values: list[float]) -> float:
    """Piecewise linear interpolation of a value over ELO knots."""
    elo = float(elo)
    if elo <= knots[0]:
        return values[0]
    if elo >= knots[-1]:
        return values[-1]
    for i in range(len(knots) - 1):
        if knots[i] <= elo <= knots[i + 1]:
            t = (elo - knots[i]) / (knots[i + 1] - knots[i])
            return values[i] + t * (values[i + 1] - values[i])
    return values[-1]


def sigma_for_elo(elo: int, mean_s: float) -> float:
    """Compute log-normal sigma from ELO and expected mean think time.

    Uses nixec's empirically-derived SD = a + b*mean formula,
    then converts to lognormal sigma:
        For LogNormal with mean m and std s:
        sigma_ln = sqrt(log(1 + (s/m)^2))
    """
    a = _elo_interpolate(elo, _ELO_KNOTS, _A_KNOTS)
    b = _elo_interpolate(elo, _ELO_KNOTS, _B_KNOTS)
    sd = max(0.05, a + b * mean_s)

    # Convert linear (mean, sd) → lognormal sigma parameter
    # sigma_ln = sqrt(log(1 + (sd/mean)^2))
    ratio = sd / max(mean_s, 0.001)
    sigma_ln = math.sqrt(math.log(1.0 + ratio ** 2))
    # Clamp to a reasonable range
    return max(0.20, min(2.0, sigma_ln))


# ── Upgrade 2: Premove probability model ─────────────────────────────────────
# From nixec's Lichess analysis: ~21.26% of moves are "premoves" (near-instant).
# Modifiers increase/decrease probability based on position characteristics.

_PREMOVE_BASE_RATE = 0.2126          # 21.26% base probability
_PREMOVE_MAX_DELAY = 0.18            # Premoves execute in 80-180ms

# Factors that increase premove probability
_PREMOVE_RECAPTURE_BONUS  = 0.18   # Recapturing a just-captured piece → obvious
_PREMOVE_FORCED_BONUS     = 0.30   # Only 1-2 legal moves → forced response
_PREMOVE_OPENING_BONUS    = 0.10   # Opening moves ≤ ply 6 → book memorized
_PREMOVE_MOMENTUM_BONUS   = 0.08   # Previous move was also fast (<0.4s)

# Factors that decrease premove probability
_PREMOVE_COMPLEX_PENALTY  = -0.12  # High entropy (many good options) → must think
_PREMOVE_ENDGAME_PENALTY  = -0.06  # Endgame: precision required, no fast premoves


def should_premove(
    state: GameState,
    move_dist: MoveDistribution,
    last_delay_s: float = 1.0,
    rng: random.Random | None = None,
) -> bool:
    """Return True if this move should be a premove (near-instant response).

    Probability is adjusted based on:
    - Position type (recapture, forced, opening)
    - Previous move speed (momentum)
    - Position complexity (entropy)
    """
    r = rng or random.Random()
    prob = _PREMOVE_BASE_RATE

    # Forced / near-forced move
    legal = move_dist.legal_move_count
    if legal <= 1:
        prob += _PREMOVE_FORCED_BONUS + 0.15   # near certain for forced moves
    elif legal <= 3:
        prob += _PREMOVE_FORCED_BONUS

    # Opening phase: first 6 plies are often book moves
    if state.ply_count <= 6:
        prob += _PREMOVE_OPENING_BONUS

    # Recapture: if opponent just took a piece on a square we have a piece pointing at
    try:
        if state.board.is_en_passant(state.board.move_stack[-1] if state.board.move_stack else None):
            prob += _PREMOVE_RECAPTURE_BONUS
        elif state.board.move_stack:
            last_move = state.board.peek()
            if state.board.is_capture(last_move):
                prob += _PREMOVE_RECAPTURE_BONUS
    except Exception:
        pass

    # Momentum: if last move was fast, the next is likely fast too
    if last_delay_s < 0.4:
        prob += _PREMOVE_MOMENTUM_BONUS

    # Complexity penalty: high entropy means many options, must think
    if move_dist.entropy > 1.8:
        prob += _PREMOVE_COMPLEX_PENALTY

    # Endgame penalty
    piece_count = len(state.board.piece_map())
    if piece_count <= 12:
        prob += _PREMOVE_ENDGAME_PENALTY

    prob = max(0.0, min(0.95, prob))
    return r.random() < prob


def premove_delay(rng: random.Random | None = None) -> float:
    """Sample a near-instant premove delay.

    From nixec's data: exponential distribution with mean ~100ms,
    capped at 180ms. Models browser input lag + human pre-registration.
    """
    r = rng or random.Random()
    # Exponential with rate=10 (mean=0.10s), floor=0.05s, cap=0.18s
    raw = 0.05 + r.expovariate(10.0)
    return min(_PREMOVE_MAX_DELAY, raw)


# ── Upgrade 3: AR(1) autocorrelation state ───────────────────────────────────
# Human think times are NOT i.i.d. — each move correlates with the previous.
# rho=0.4 means: new_raw = 0.4 * prev_raw + 0.6 * independent_sample
# This produces streaks of fast moves followed by streaks of slower moves,
# which is exactly what human game logs show.

_AR1_RHO = 0.40  # Autocorrelation coefficient (from Lichess data)


class ThinkTimeState:
    """Carries the AR(1) residual between moves for autocorrelated timing.

    Usage:
        state = ThinkTimeState()
        delay = state.sample(mean=2.0, sigma=0.8, rng=rng)  # move 1
        delay = state.sample(mean=1.5, sigma=0.7, rng=rng)  # move 2, correlated with 1
    """

    def __init__(self) -> None:
        self._prev_normalized: float = 0.0  # Previous move's normalized z-score
        self.last_delay_s: float = 1.0       # Last raw delay, used for premove momentum

    def reset(self) -> None:
        """Call at game start to clear autocorrelation state."""
        self._prev_normalized = 0.0
        self.last_delay_s = 1.0

    def sample_lognormal(
        self,
        mu: float,
        sigma: float,
        rng: random.Random | None = None,
    ) -> float:
        """Sample a log-normal value with AR(1) autocorrelation applied.

        AR(1) in normalized space:
            z_t = rho * z_{t-1} + sqrt(1 - rho^2) * eps_t
        where eps_t ~ N(0,1)
        """
        r = rng or random.Random()

        # Independent standard normal sample
        eps = r.gauss(0.0, 1.0)

        # AR(1) blend: correlated z-score
        z = _AR1_RHO * self._prev_normalized + math.sqrt(1.0 - _AR1_RHO ** 2) * eps

        # Store for next move
        self._prev_normalized = z

        # Convert z back to log-normal value: X = exp(mu + sigma * z)
        raw = math.exp(mu + sigma * z)
        self.last_delay_s = max(0.0, raw)
        return self.last_delay_s


# ── Full upgraded models ──────────────────────────────────────────────────────

class ClockOnlyBaseline(TimingModel):
    """Predicts log-normal think time distribution based on clock, increment, and game phase.

    Upgraded: ELO-interpolated sigma instead of fixed 0.65.
    """

    def __init__(
        self,
        base_fraction: float = 0.035,
        inc_fraction: float = 0.60,
        sigma: float = 0.65,           # Fallback if elo not given
        min_time: float = 0.10,
        elo: int = 1500,
        bucket_edges: Sequence[float] = DEFAULT_BUCKET_EDGES,
    ) -> None:
        self.base_fraction = base_fraction
        self.inc_fraction = inc_fraction
        self._fixed_sigma = sigma
        self.min_time = min_time
        self.elo = elo
        self.bucket_edges = tuple(bucket_edges)

    def _get_phase_multiplier(self, ply: int) -> float:
        if ply <= 6:
            return 0.55   # Opening: book moves, fast
        if ply <= 12:
            return 0.80   # Exiting opening
        if ply <= 30:
            return 1.20   # Peak middlegame planning
        if ply <= 50:
            return 1.00   # Late middlegame
        return 0.85        # Endgame

    def compute_target_mean(self, state: GameState) -> float:
        clock_self = max(0.1, state.clock_self)
        increment = max(0.0, state.increment)
        phase_mult = self._get_phase_multiplier(state.ply_count)
        base_time = self.base_fraction * clock_self + self.inc_fraction * increment
        return max(self.min_time, base_time * phase_mult)

    def predict(self, state: GameState, move_dist: MoveDistribution) -> ThinkTimeDistribution:
        target_mean = self.compute_target_mean(state)

        # Upgrade 1: ELO-interpolated sigma
        sigma = sigma_for_elo(self.elo, target_mean)

        mu = math.log(target_mean) - 0.5 * (sigma ** 2)
        probs = _lognormal_bucket_probs(
            mu=mu, sigma=sigma,
            bucket_edges=self.bucket_edges,
            min_time=self.min_time,
        )
        dist = ThinkTimeDistribution(self.bucket_edges, probs)
        return dist.mask_by_remaining_clock(state.clock_self)


class HeuristicBaseline(TimingModel):
    """Timing model extending ClockOnlyBaseline with policy entropy and top-move probability.

    Upgrades:
    1. ELO-interpolated sigma (per move).
    2. Premove detection check (caller uses should_premove() + ThinkTimeState).
    3. Legal-move complexity brackets (ported from nixec's 5-bracket system).
    """

    def __init__(
        self,
        base_fraction: float = 0.035,
        inc_fraction: float = 0.60,
        sigma: float = 0.65,
        min_time: float = 0.10,
        entropy_weight: float = 0.30,
        top_move_weight: float = -0.35,
        elo: int = 1500,
        bucket_edges: Sequence[float] = DEFAULT_BUCKET_EDGES,
    ) -> None:
        self.clock_baseline = ClockOnlyBaseline(
            base_fraction=base_fraction,
            inc_fraction=inc_fraction,
            sigma=sigma,
            min_time=min_time,
            elo=elo,
            bucket_edges=bucket_edges,
        )
        self.elo = elo
        self.min_time = min_time
        self.entropy_weight = entropy_weight
        self.top_move_weight = top_move_weight
        self.bucket_edges = tuple(bucket_edges)

    def _complexity_multiplier(self, legal_count: int) -> float:
        """5-bracket complexity multiplier from nixec's Lichess analysis.

        Legal move count is the best single proxy for position complexity
        (derived from 12M game analysis on Lichess).
        """
        if legal_count <= 5:
            return 0.70    # Very forced / narrow: fast
        if legal_count <= 15:
            return 0.88    # Restricted choice
        if legal_count <= 30:
            return 1.00    # Normal complexity: baseline
        if legal_count <= 45:
            return 1.15    # Complex: more options, more thinking
        return 1.30        # Very open position: many candidate moves

    def predict(self, state: GameState, move_dist: MoveDistribution) -> ThinkTimeDistribution:
        base_mean = self.clock_baseline.compute_target_mean(state)

        # Entropy signal: high entropy → more options → longer think
        entropy_delta = move_dist.entropy - 1.3
        entropy_mult = math.exp(self.entropy_weight * entropy_delta)

        # Top-move probability: high probability (obvious move) → faster
        top_prob = move_dist.top_probability if move_dist.moves else 1.0
        top_prob_delta = top_prob - 0.4
        top_prob_mult = math.exp(self.top_move_weight * top_prob_delta)

        # Legal move complexity bracket (nixec port)
        complexity_mult = self._complexity_multiplier(move_dist.legal_move_count)

        # Single legal move override (forced move)
        if move_dist.legal_move_count <= 1:
            adjusted_mean = self.min_time
        else:
            adjusted_mean = max(
                self.min_time,
                base_mean * entropy_mult * top_prob_mult * complexity_mult,
            )

        # ELO-interpolated sigma
        sigma = sigma_for_elo(self.elo, adjusted_mean)

        mu = math.log(adjusted_mean) - 0.5 * (sigma ** 2)
        probs = _lognormal_bucket_probs(
            mu=mu, sigma=sigma,
            bucket_edges=self.bucket_edges,
            min_time=self.min_time,
        )
        dist = ThinkTimeDistribution(self.bucket_edges, probs)
        return dist.mask_by_remaining_clock(state.clock_self)
