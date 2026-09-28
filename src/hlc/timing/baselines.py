"""Baseline move timing models: ClockOnlyBaseline and HeuristicBaseline."""

from __future__ import annotations

import math
from collections.abc import Sequence

from hlc.timing.base import DEFAULT_BUCKET_EDGES, ThinkTimeDistribution, TimingModel
from hlc.types import GameState, MoveDistribution


def _normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function Phi(x)."""
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

    # Normalize
    total = sum(probs)
    if total > 0.0:
        return [p / total for p in probs]
    return [1.0 / num_buckets] * num_buckets


class ClockOnlyBaseline(TimingModel):
    """Predicts log-normal think time distribution based solely on clock, increment, and game phase.

    Humans in blitz pace their think times primarily by remaining clock and increment,
    with an opening book acceleration and middlegame expansion.
    """

    def __init__(
        self,
        base_fraction: float = 0.035,
        inc_fraction: float = 0.60,
        sigma: float = 0.65,
        min_time: float = 0.10,
        bucket_edges: Sequence[float] = DEFAULT_BUCKET_EDGES,
    ) -> None:
        self.base_fraction = base_fraction
        self.inc_fraction = inc_fraction
        self.sigma = sigma
        self.min_time = min_time
        self.bucket_edges = tuple(bucket_edges)

    def _get_phase_multiplier(self, ply: int) -> float:
        """Game phase adjustment factor."""
        if ply <= 6:
            return 0.55  # Early opening (well-known initial moves)
        if ply <= 12:
            return 0.80  # Transition out of opening
        if ply <= 30:
            return 1.20  # Peak middlegame planning
        if ply <= 50:
            return 1.00  # Late middlegame / early endgame
        return 0.85  # Deep endgame time scramble

    def compute_target_mean(self, state: GameState) -> float:
        """Compute the expected think time in seconds before distribution shaping."""
        clock_self = max(0.1, state.clock_self)
        increment = max(0.0, state.increment)
        phase_mult = self._get_phase_multiplier(state.ply_count)

        base_time = self.base_fraction * clock_self + self.inc_fraction * increment
        target = base_time * phase_mult
        return max(self.min_time, target)

    def predict(self, state: GameState, move_dist: MoveDistribution) -> ThinkTimeDistribution:
        """Predict think-time distribution given remaining clock and increment."""
        target_mean = self.compute_target_mean(state)

        # For LogNormal(mu, sigma^2), E[X] = exp(mu + sigma^2 / 2) -> mu = ln(E[X]) - sigma^2 / 2
        mu = math.log(target_mean) - 0.5 * (self.sigma**2)

        probs = _lognormal_bucket_probs(
            mu=mu,
            sigma=self.sigma,
            bucket_edges=self.bucket_edges,
            min_time=self.min_time,
        )

        dist = ThinkTimeDistribution(self.bucket_edges, probs)
        return dist.mask_by_remaining_clock(state.clock_self)


class HeuristicBaseline(TimingModel):
    """Timing model extending ClockOnlyBaseline with position difficulty features from Maia-3.

    HYPOTHESIS TO TEST IN PHASE 3:
    Humans think longer when position complexity and policy entropy are high (no single clear move),
    and think faster when top-move probability is high (forced move / obvious recapture).
    """

    def __init__(
        self,
        base_fraction: float = 0.035,
        inc_fraction: float = 0.60,
        sigma: float = 0.65,
        min_time: float = 0.10,
        entropy_weight: float = 0.30,
        top_move_weight: float = -0.35,
        bucket_edges: Sequence[float] = DEFAULT_BUCKET_EDGES,
    ) -> None:
        self.clock_baseline = ClockOnlyBaseline(
            base_fraction=base_fraction,
            inc_fraction=inc_fraction,
            sigma=sigma,
            min_time=min_time,
            bucket_edges=bucket_edges,
        )
        self.sigma = sigma
        self.min_time = min_time
        self.entropy_weight = entropy_weight
        self.top_move_weight = top_move_weight
        self.bucket_edges = tuple(bucket_edges)

    def predict(self, state: GameState, move_dist: MoveDistribution) -> ThinkTimeDistribution:
        """Predict think time incorporating Maia-3 policy entropy and top move probability."""
        base_mean = self.clock_baseline.compute_target_mean(state)

        # Baseline entropy in typical positions is ~1.3 nats
        entropy_delta = move_dist.entropy - 1.3
        entropy_mult = math.exp(self.entropy_weight * entropy_delta)

        # Top move probability: high probability (e.g. 0.8) reduces think time
        top_prob = move_dist.top_probability if move_dist.moves else 1.0
        top_prob_delta = top_prob - 0.4
        top_prob_mult = math.exp(self.top_move_weight * top_prob_delta)

        # Single legal move override (forced move)
        if move_dist.legal_move_count <= 1:
            adjusted_mean = self.min_time
        else:
            adjusted_mean = max(self.min_time, base_mean * entropy_mult * top_prob_mult)

        mu = math.log(adjusted_mean) - 0.5 * (self.sigma**2)
        probs = _lognormal_bucket_probs(
            mu=mu,
            sigma=self.sigma,
            bucket_edges=self.bucket_edges,
            min_time=self.min_time,
        )

        dist = ThinkTimeDistribution(self.bucket_edges, probs)
        return dist.mask_by_remaining_clock(state.clock_self)
