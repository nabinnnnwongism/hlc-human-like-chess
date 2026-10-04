"""Property tests for ThinkTimeDistribution, baselines, and Scheduler clock safety."""

import random

import chess
import pytest

from hlc.scheduler import Scheduler, SchedulerConfig
from hlc.timing.base import DEFAULT_BUCKET_EDGES, ThinkTimeDistribution
from hlc.timing.baselines import ClockOnlyBaseline, HeuristicBaseline
from hlc.types import GameState, MoveDistribution


class TestThinkTimeDistribution:
    """Property tests for ThinkTimeDistribution."""

    def test_probabilities_sum_to_one(self) -> None:
        edges = [0.0, 1.0, 2.0, 5.0, 10.0]
        raw_probs = [0.1, 0.4, 0.3, 0.2]
        dist = ThinkTimeDistribution(edges, raw_probs)

        assert pytest.approx(sum(dist.probabilities), abs=1e-6) == 1.0
        assert len(dist.probabilities) == 4

    def test_unnormalized_probabilities_are_normalized(self) -> None:
        edges = [0.0, 1.0, 2.0, 3.0]
        raw_probs = [10.0, 20.0, 10.0]
        dist = ThinkTimeDistribution(edges, raw_probs)

        assert pytest.approx(sum(dist.probabilities), abs=1e-6) == 1.0
        assert pytest.approx(dist.probabilities[0], abs=1e-6) == 0.25
        assert pytest.approx(dist.probabilities[1], abs=1e-6) == 0.50
        assert pytest.approx(dist.probabilities[2], abs=1e-6) == 0.25

    def test_mean_bounded_within_bucket_edges(self) -> None:
        edges = [1.0, 2.0, 3.0, 4.0]
        dist = ThinkTimeDistribution(edges, [0.2, 0.5, 0.3])
        mean_val = dist.mean()
        assert edges[0] <= mean_val <= edges[-1]

    def test_quantile_monotonicity(self) -> None:
        edges = [0.0, 1.0, 3.0, 6.0, 10.0]
        dist = ThinkTimeDistribution(edges, [0.1, 0.3, 0.4, 0.2])

        quantiles = [dist.quantile(q / 100.0) for q in range(101)]
        for i in range(len(quantiles) - 1):
            assert quantiles[i] <= quantiles[i + 1]

        assert dist.quantile(0.0) == edges[0]
        assert dist.quantile(1.0) == edges[-1]

    def test_seeded_sampling_reproducibility(self) -> None:
        edges = DEFAULT_BUCKET_EDGES
        probs = [1.0 / (len(edges) - 1)] * (len(edges) - 1)
        dist = ThinkTimeDistribution(edges, probs)

        rng1 = random.Random(999)
        samples1 = [dist.sample(rng=rng1) for _ in range(50)]

        rng2 = random.Random(999)
        samples2 = [dist.sample(rng=rng2) for _ in range(50)]

        assert samples1 == samples2

    def test_mask_by_remaining_clock(self) -> None:
        edges = [0.0, 2.0, 5.0, 10.0, 20.0]
        probs = [0.25, 0.25, 0.25, 0.25]
        dist = ThinkTimeDistribution(edges, probs)

        masked = dist.mask_by_remaining_clock(max_seconds=4.0)
        assert pytest.approx(sum(masked.probabilities), abs=1e-6) == 1.0
        # Buckets with lower edge >= 4.0 (i.e. [5.0, 10.0) and [10.0, 20.0)) must be 0.0
        assert masked.probabilities[2] == 0.0
        assert masked.probabilities[3] == 0.0
        assert masked.probabilities[0] > 0.0
        assert masked.probabilities[1] > 0.0


class TestTimingBaselines:
    """Property tests for ClockOnlyBaseline and HeuristicBaseline."""

    def test_clock_baseline_sums_to_one_across_various_clocks(self) -> None:
        model = ClockOnlyBaseline()
        board = chess.Board()

        for clock in [0.5, 3.0, 10.0, 30.0, 180.0, 300.0]:
            state = GameState(board=board, clock_self=clock, increment=1.0)
            dummy_dist = MoveDistribution(moves=[chess.Move.from_uci("e2e4")], probabilities=[1.0])
            dist = model.predict(state, dummy_dist)

            assert pytest.approx(sum(dist.probabilities), abs=1e-5) == 1.0
            # Expected mean must be lower for low clock
            assert dist.mean() <= clock

    def test_clock_pacing_scales_monotonically_with_clock(self) -> None:
        model = ClockOnlyBaseline()
        board = chess.Board()
        dummy_dist = MoveDistribution(moves=[chess.Move.from_uci("e2e4")], probabilities=[1.0])

        state_long = GameState(board=board, clock_self=180.0, increment=2.0)
        state_med = GameState(board=board, clock_self=40.0, increment=2.0)
        state_short = GameState(board=board, clock_self=5.0, increment=2.0)

        dist_long = model.predict(state_long, dummy_dist)
        dist_med = model.predict(state_med, dummy_dist)
        dist_short = model.predict(state_short, dummy_dist)

        assert dist_long.mean() > dist_med.mean() > dist_short.mean()

    def test_heuristic_baseline_responds_to_entropy(self) -> None:
        model = HeuristicBaseline()
        board = chess.Board()
        state = GameState(board=board, clock_self=120.0, increment=1.0)

        # High entropy: 5 equally likely moves (H = ln(5) ~ 1.61 nats)
        m_high = [
            chess.Move.from_uci("e2e4"),
            chess.Move.from_uci("d2d4"),
            chess.Move.from_uci("c2c4"),
            chess.Move.from_uci("g1f3"),
            chess.Move.from_uci("b1c3"),
        ]
        dist_high = MoveDistribution(moves=m_high, probabilities=[0.2] * 5)

        # Low entropy: 1 dominating move (p=0.96)
        m_low = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
        dist_low = MoveDistribution(moves=m_low, probabilities=[0.96, 0.04])

        t_high = model.predict(state, dist_high).mean()
        t_low = model.predict(state, dist_low).mean()

        # High entropy position should yield longer expected think time
        assert t_high > t_low

    def test_heuristic_single_legal_move_collapses_to_min_time(self) -> None:
        model = HeuristicBaseline()
        board = chess.Board()
        state = GameState(board=board, clock_self=180.0)

        single_move_dist = MoveDistribution(
            moves=[chess.Move.from_uci("e2e4")], probabilities=[1.0]
        )
        dist = model.predict(state, single_move_dist)

        # Forced move mean should be around min_time
        assert dist.mean() < 1.0


class TestSchedulerProperties:
    """Rigorous property tests proving Scheduler clock safety across simulated games."""

    def test_single_legal_move_near_instant(self) -> None:
        scheduler = Scheduler()
        # Position with only 1 legal escape/capture (White King captures b2 Rook)
        fen = "k7/8/8/8/8/8/1r6/K7 w - - 0 1"
        board = chess.Board(fen)
        assert board.legal_moves.count() == 1

        state = GameState(board=board, clock_self=60.0)
        edges = DEFAULT_BUCKET_EDGES
        dist = ThinkTimeDistribution(edges, [1.0 / (len(edges) - 1)] * (len(edges) - 1))

        delay = scheduler.plan(state=state, think_dist=dist, compute_elapsed_s=0.02)
        assert delay <= scheduler.cfg.single_move_delay

    def test_delay_never_flags_across_1000_random_clock_states(self) -> None:
        """Property: for any remaining clock in (0.01, 300) seconds,

        the planned delay + move_overhead never exceeds clock_self.
        """
        cfg = SchedulerConfig(move_overhead=0.10, safety_margin=0.50, max_clock_fraction=0.12)
        scheduler = Scheduler(cfg)
        board = chess.Board()  # 20 legal moves

        rng = random.Random(42)
        edges = DEFAULT_BUCKET_EDGES
        probs = [1.0 / (len(edges) - 1)] * (len(edges) - 1)
        dist = ThinkTimeDistribution(edges, probs)

        for _ in range(1000):
            # Sample random remaining clocks from 0.05s to 300.0s
            clock = rng.uniform(0.05, 300.0)
            compute_elapsed = rng.uniform(0.01, 0.30)
            state = GameState(board=board, clock_self=clock)

            delay = scheduler.plan(
                state=state,
                think_dist=dist,
                compute_elapsed_s=compute_elapsed,
                rng=rng,
            )

            # Verification 1: Delay is non-negative
            assert delay >= 0.0

            # Verification 2: Clock Safety Guarantee
            # If clock is enough to cover overhead + safety margin, delay leaves at least safety margin
            if clock > cfg.move_overhead + cfg.safety_margin:
                assert delay + cfg.move_overhead <= clock - cfg.safety_margin + 1e-6
            else:
                # In critical emergency, delay must be 0.0 to prevent flagging
                assert delay == 0.0

            # Verification 3: Delay never exceeds max_clock_fraction
            assert delay <= (cfg.max_clock_fraction * clock) + 1e-6

    def test_scheduler_deducts_compute_time(self) -> None:
        scheduler = Scheduler()
        board = chess.Board()
        state = GameState(board=board, clock_self=180.0)

        edges = [0.0, 5.0, 10.0]
        # Degenerate distribution concentrated on 5-10s (sample ~7.5s)
        dist = ThinkTimeDistribution(edges, [0.0, 1.0])

        rng1 = random.Random(101)
        delay_fast = scheduler.plan(state=state, think_dist=dist, compute_elapsed_s=0.05, rng=rng1)

        rng2 = random.Random(101)
        delay_slow = scheduler.plan(state=state, think_dist=dist, compute_elapsed_s=1.50, rng=rng2)

        # Longer compute must produce shorter planned delay
        assert delay_slow < delay_fast
        assert pytest.approx(delay_fast - delay_slow, abs=0.01) == (1.50 - 0.05)

    def test_scheduler_seeded_reproducibility(self) -> None:
        scheduler = Scheduler()
        board = chess.Board()
        state = GameState(board=board, clock_self=120.0)

        edges = DEFAULT_BUCKET_EDGES
        dist = ThinkTimeDistribution(edges, [1.0 / (len(edges) - 1)] * (len(edges) - 1))

        rng_a = random.Random(777)
        delays_a = [
            scheduler.plan(state=state, think_dist=dist, compute_elapsed_s=0.1, rng=rng_a)
            for _ in range(25)
        ]

        rng_b = random.Random(777)
        delays_b = [
            scheduler.plan(state=state, think_dist=dist, compute_elapsed_s=0.1, rng=rng_b)
            for _ in range(25)
        ]

        assert delays_a == delays_b

    def test_adaptive_clock_pacing_tiers(self) -> None:
        """Verify Option C adaptive clock speed tiers and anti-flagging caps."""
        scheduler = Scheduler()
        board = chess.Board()
        edges = [0.0, 10.0, 20.0]
        # Distribution with large think time sample (~15s)
        dist = ThinkTimeDistribution(edges, [0.0, 1.0])

        # Tier 1: Clock = 75s (1:15 remaining) -> should be capped at 2.0s
        state_75s = GameState(board=board, clock_self=75.0)
        delay_75s = scheduler.plan(state=state_75s, think_dist=dist, compute_elapsed_s=0.01)
        assert delay_75s <= 2.0

        # Tier 2: Clock = 25s (Scramble) -> capped at 0.9s
        state_25s = GameState(board=board, clock_self=25.0)
        delay_25s = scheduler.plan(state=state_25s, think_dist=dist, compute_elapsed_s=0.01)
        assert delay_25s <= 0.9

        # Tier 3: Clock = 8s (Emergency Panic) -> capped at 0.35s
        state_8s = GameState(board=board, clock_self=8.0)
        delay_8s = scheduler.plan(state=state_8s, think_dist=dist, compute_elapsed_s=0.01)
        assert delay_8s <= 0.35

