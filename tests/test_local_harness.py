"""Tests for LocalHarness: legal moves, PGN output, clock safety."""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

import chess
import chess.pgn
import pytest

from hlc.adapters.local_harness import GameRecord, HarnessOpponent, LocalHarness
from hlc.core import BotCore
from hlc.engines.maia3 import Maia3Engine
from hlc.scheduler import Scheduler, SchedulerConfig
from hlc.timing.baselines import ClockOnlyBaseline


class _RandomOpponent(HarnessOpponent):
    """Fast random-move opponent for testing (no engine subprocess needed)."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def select_move(
        self, board: chess.Board, time_remaining: float, increment: float
    ) -> chess.Move:
        moves = list(board.legal_moves)
        return self._rng.choice(moves)


@pytest.fixture(scope="module")
def engine() -> Maia3Engine:
    return Maia3Engine(
        model="maia3-79m",
        elo=1500,
        device="cpu",
        use_amp=False,
        use_uci_history=False,  # Faster for tests
        multipv=1,
    )


@pytest.fixture(scope="module")
def bot_core(engine: Maia3Engine) -> BotCore:
    return BotCore(
        move_engine=engine,
        timing_model=ClockOnlyBaseline(),
        scheduler=Scheduler(SchedulerConfig(min_delay=0.0, safety_margin=0.05)),
        rng=random.Random(42),
    )


class TestGameRecord:
    """Test PGN serialization with [%clk] tags."""

    def test_pgn_has_clk_tags(self) -> None:
        moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("e7e5")]
        record = GameRecord(
            bot_color=chess.WHITE,
            bot_elo=1500,
            opponent_elo=1500,
            time_control="180+2",
            moves=moves,
            clk_before_white=[180.0],
            clk_before_black=[178.0],
            actual_delays=[1.2, 0.5],
            planned_delays=[1.2, 0.5],
            result="*",
        )
        game = record.to_pgn()
        pgn_str = str(game)
        assert "[%clk" in pgn_str

    def test_pgn_has_headers(self) -> None:
        record = GameRecord(
            bot_color=chess.WHITE,
            bot_elo=1500,
            opponent_elo=1500,
            time_control="180+2",
            result="1-0",
            termination="Checkmate",
        )
        game = record.to_pgn()
        assert game.headers["TimeControl"] == "180+2"
        assert game.headers["White"] == "HLC Bot"


class TestLocalHarnessVirtual:
    """Integration tests: virtual clock mode, no real sleeps."""

    def test_5_games_no_illegal_moves_or_flags(self, bot_core: BotCore) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            opponent = _RandomOpponent(seed=99)
            harness = LocalHarness(
                bot_core=bot_core,
                opponent=opponent,
                runs_dir=tmpdir,
                time_seconds=60.0,
                increment=1.0,
                bot_color=chess.WHITE,
                bot_elo=1500,
                opponent_elo=1500,
                clock_mode="virtual",
                rng=random.Random(77),
                verbose=False,
            )
            records = harness.run_games(5, prefix="test")

        assert len(records) == 5
        for r in records:
            # No flagging allowed
            assert "time" not in r.termination.lower()
            # All moves must be plies > 0
            assert len(r.moves) > 0

    def test_pgn_files_written_with_clk(self, bot_core: BotCore) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            opponent = _RandomOpponent(seed=7)
            harness = LocalHarness(
                bot_core=bot_core,
                opponent=opponent,
                runs_dir=tmpdir,
                time_seconds=60.0,
                increment=1.0,
                bot_color=chess.WHITE,
                bot_elo=1500,
                opponent_elo=1500,
                clock_mode="virtual",
                rng=random.Random(42),
            )
            _records = harness.run_games(2, prefix="pgn_test")
            pgn_files = list(Path(tmpdir).glob("*.pgn"))
            # Read contents while tmpdir is still alive
            pgn_contents = [p.read_text(encoding="utf-8") for p in pgn_files]

        assert len(pgn_contents) == 2
        for content in pgn_contents:
            assert "[%clk" in content

    def test_bot_as_black(self, bot_core: BotCore) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            opponent = _RandomOpponent(seed=5)
            harness = LocalHarness(
                bot_core=bot_core,
                opponent=opponent,
                runs_dir=tmpdir,
                time_seconds=60.0,
                increment=1.0,
                bot_color=chess.BLACK,
                bot_elo=1500,
                opponent_elo=1500,
                clock_mode="virtual",
                rng=random.Random(11),
            )
            records = harness.run_games(3, prefix="black_test")

        assert len(records) == 3
        for r in records:
            assert r.bot_color == chess.BLACK
