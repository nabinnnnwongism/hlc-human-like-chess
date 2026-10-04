"""test_pacing_guard.py — Automated verification for PacingModel and TacticalGuard."""

import math
import random
import chess
import pytest

from hlc.engines.guard import TacticalGuard, mates_in_one, MATE_MISS_PRIOR
from hlc.timing.pacing import ClockInfo, MoveFeatures, PacingModel, PacingParams, VirtualClock
from hlc.adapters.chesscom.digital_clock import DigitalClock


def test_pacing_no_repeated_bins():
    """Verify soft saturation prevents repeated delay bins (no clustering)."""
    model = PacingModel(rng=random.Random(42))
    clk = ClockInfo(timed=True, initial_s=180.0, inc_s=0.0, my_s=150.0, opp_s=150.0)

    delays = []
    for move_num in range(1, 50):
        feats = MoveFeatures(
            n=move_num,
            entropy=1.5,
            top_p=0.6,
            clock=clk,
            n_legal=30,
        )
        p = model.sample(feats)
        delays.append(round(p.seconds, 4))

    # All delays must be positive and respect visual floor
    assert all(d >= 0.12 for d in delays)

    # In 49 moves, no single value should repeat more than 3 times
    from collections import Counter
    counts = Counter(delays)
    most_common_count = counts.most_common(1)[0][1]
    assert most_common_count <= 2, f"Found repeating delays in soft saturation: {counts.most_common(5)}"


def test_pacing_anti_flagging_safety():
    """Verify delays never flag even with < 3 seconds on the clock."""
    model = PacingModel(rng=random.Random(123))

    for clock_s in [15.0, 8.0, 4.0, 2.0, 1.0]:
        clk = ClockInfo(timed=True, initial_s=180.0, inc_s=0.0, my_s=clock_s, opp_s=10.0)
        feats = MoveFeatures(n=35, entropy=1.8, top_p=0.5, clock=clk, n_legal=20)
        p = model.sample(feats)
        # Delay must never exceed 50% of remaining clock
        assert p.seconds <= 0.50 * clock_s, f"Flag risk at {clock_s}s: got delay {p.seconds}s"


def test_tactical_guard_converts_mate_in_one():
    """Verify TacticalGuard converts an obvious back-rank mate-in-1."""
    # White to move: Rd1 delivers back-rank mate Rd1-d8#
    fen = "6k1/5ppp/8/8/8/8/8/3R2K1 w - - 0 1"
    board = chess.Board(fen)

    mates = mates_in_one(board)
    assert len(mates) == 1
    mate_move = mates[0]  # Rd1d8

    # Bot originally chose a quiet King move
    blunder_move = chess.Move.from_uci("g1f1")

    # Use low miss roll (high random value means human notices the mate)
    guard = TacticalGuard(elo=1500, rng=random.Random(999))
    res = guard.apply(board, chosen=blunder_move, ranked=[(blunder_move, 0.9)], my_clock_s=60.0)

    assert res.move == mate_move
    assert res.reason == "mate_in_1"


def test_tactical_guard_avoids_stalemate():
    """Verify TacticalGuard redirects candidate move if it would cause accidental stalemate."""
    # King on a8, White King on c7, Pawn on b5.
    # No mate-in-1 exists. Playing b5-b6 stalemates!
    fen = "k7/2K5/8/1P6/8/8/8/8 w - - 0 1"
    board = chess.Board(fen)

    trap = chess.Move.from_uci("b5b6")
    board.push(trap)
    assert board.is_stalemate()
    board.pop()

    alt = chess.Move.from_uci("c7c6")
    guard = TacticalGuard(elo=1500, rng=random.Random(999))
    res = guard.apply(board, chosen=trap, ranked=[(trap, 0.9), (alt, 0.1)], my_clock_s=60.0)

    # Move must be redirected away from stalemate
    assert res.move == alt
    assert res.reason == "stalemate_avoid"


def test_virtual_clock_untimed():
    """Verify VirtualClock decays naturally in untimed games."""
    vclk = VirtualClock(initial_s=600.0, inc_s=0.0)
    info0 = vclk.info()
    assert info0.timed is False
    assert info0.my_s == 600.0

    vclk.after_our_move(spent_s=5.0)
    assert vclk.info().my_s == 595.0

    vclk.after_opp_move(spent_s=3.0)
    assert vclk.info().opp_s == 597.0


def test_digital_clock_stopwatch_rejection():
    """Verify DigitalClock rejects non-monotonic clock updates (Chess.com stopwatch)."""
    clock = DigitalClock(initial_seconds=180.0, increment_s=0.0, time_control_name="Blitz (3 min)")
    clock.has_dom_clocks = True
    clock.white_s = 180.0
    clock.black_s = 180.0

    # Normal tick down: White 175s, Black 178s -> Accepted
    clock.sync_from_dom(175.0, 178.0)
    assert clock.white_s == 175.0
    assert clock.black_s == 178.0

    # Stopwatch update counting UP: White jumps to 210s -> Rejected!
    clock.sync_from_dom(210.0, 175.0)
    assert clock.white_s == 175.0  # Kept previous value!
