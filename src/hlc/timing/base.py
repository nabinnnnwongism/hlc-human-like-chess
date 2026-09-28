"""Base interfaces and data structures for move timing models."""

from __future__ import annotations

import bisect
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from hlc.types import GameState, MoveDistribution

# Default 30 bucket edges (0-27 in 1s increments, then wider, matching blitz think-time ranges)
DEFAULT_BUCKET_EDGES: tuple[float, ...] = (
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
    11.0,
    12.0,
    13.0,
    14.0,
    15.0,
    16.0,
    17.0,
    18.0,
    19.0,
    20.0,
    21.0,
    22.0,
    23.0,
    24.0,
    25.0,
    26.0,
    27.0,
    30.0,
    35.0,
    45.0,
    60.0,
)


@dataclass
class ThinkTimeDistribution:
    """Discretized probability distribution over human think time in seconds."""

    bucket_edges: Sequence[float]
    probabilities: Sequence[float]
    _cum_probs: list[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.bucket_edges) < 2:
            raise ValueError("Bucket edges must have at least 2 entries (start and end).")
        expected_buckets = len(self.bucket_edges) - 1
        if len(self.probabilities) != expected_buckets:
            raise ValueError(
                f"Probabilities count ({len(self.probabilities)}) must match "
                f"bucket count ({expected_buckets})"
            )

        # Normalize probabilities so they strictly sum to 1.0
        total_p = sum(self.probabilities)
        if total_p <= 0.0 or math.isnan(total_p):
            # Uniform fallback
            n = len(self.probabilities)
            self.probabilities = [1.0 / n] * n
        else:
            self.probabilities = [max(0.0, p) / total_p for p in self.probabilities]

        # Precompute cumulative probabilities for fast inverse-CDF sampling
        cum = 0.0
        self._cum_probs = []
        for p in self.probabilities:
            cum += p
            self._cum_probs.append(cum)
        # Ensure the last element is exactly 1.0 to avoid numerical drift
        if self._cum_probs:
            self._cum_probs[-1] = 1.0

    @property
    def num_buckets(self) -> int:
        return len(self.probabilities)

    def mean(self) -> float:
        """Expected think time in seconds."""
        expected = 0.0
        for i, p in enumerate(self.probabilities):
            mid = 0.5 * (self.bucket_edges[i] + self.bucket_edges[i + 1])
            expected += p * mid
        return expected

    def quantile(self, q: float) -> float:
        """Inverse cumulative distribution function (linear interpolation within bucket)."""
        if not (0.0 <= q <= 1.0):
            raise ValueError(f"Quantile q must be in [0, 1], got {q}")
        if q <= 0.0:
            return float(self.bucket_edges[0])
        if q >= 1.0:
            return float(self.bucket_edges[-1])

        idx = bisect.bisect_left(self._cum_probs, q)
        if idx >= len(self.probabilities):
            return float(self.bucket_edges[-1])

        prev_cum = self._cum_probs[idx - 1] if idx > 0 else 0.0
        p_bucket = self.probabilities[idx]
        low = self.bucket_edges[idx]
        high = self.bucket_edges[idx + 1]

        if p_bucket <= 1e-12:
            return float(low)

        frac = (q - prev_cum) / p_bucket
        frac = max(0.0, min(1.0, frac))
        return float(low + frac * (high - low))

    def sample(self, rng: random.Random | None = None) -> float:
        """Sample a continuous think time in seconds from the distribution."""
        r = rng if rng is not None else random.Random()
        u = r.random()
        return self.quantile(u)

    def mask_by_remaining_clock(self, max_seconds: float) -> ThinkTimeDistribution:
        """Zero out and renormalize any probability mass exceeding remaining clock."""
        if max_seconds <= self.bucket_edges[0]:
            # Degenerate: collapse to first bucket
            probs = [1.0] + [0.0] * (self.num_buckets - 1)
            return ThinkTimeDistribution(self.bucket_edges, probs)

        new_probs = list(self.probabilities)
        for i in range(self.num_buckets):
            low = self.bucket_edges[i]
            if low >= max_seconds:
                new_probs[i] = 0.0

        return ThinkTimeDistribution(self.bucket_edges, new_probs)


@runtime_checkable
class TimingModel(Protocol):
    """Protocol for models predicting human think-time distributions."""

    def predict(self, state: GameState, move_dist: MoveDistribution) -> ThinkTimeDistribution:
        """Predict think-time distribution given board state and move policy."""
        ...
