"""Tests for Phase 5 Manual Relay CLI.

Verifies:
- Startup banner contains required fair-play notice
- Move input parsing (SAN, UCI, typo handling)
- Inline and explicit clock updates
- Undo, FEN resync, PGN resync, status, legal commands
- Countdown handling
- PGN export with [%clk] tags and full game replay
- Bot as White (immediate first move)
- Integration with real Maia3Engine
"""

from __future__ import annotations

import random
from pathlib import Path

import chess
import chess.pgn
import pytest

from hlc.cli.relay import (
    BANNER,
    RelayConfig,
    RelaySession,
    build_bot_core,
    format_clock_str,
    format_pgn_clk,
    parse_clock_str,
    parse_move_input,
)
from hlc.core import BotCore
from hlc.engines.maia3 import Maia3Engine
from hlc.scheduler import Scheduler, SchedulerConfig
from hlc.timing.baselines import ClockOnlyBaseline
from hlc.types import Decision, GameState

# ---------------------------------------------------------------------------
# Test Fixtures & Mock Helpers
# ---------------------------------------------------------------------------


class FixedDecisionCore:
    """Mock BotCore that returns deterministic moves with zero delay."""

    def __init__(self, move_map: dict[str, str] | None = None) -> None:
        self.move_map = move_map or {}

    def decide(self, state: GameState) -> Decision:
        fen_key = state.board.fen()
        if fen_key in self.move_map:
            move = chess.Move.from_uci(self.move_map[fen_key])
        else:
            # Deterministic first legal move
            move = next(iter(state.board.legal_moves))

        return Decision(
            move=move,
            delay_s=0.0,
            debug={"compute_elapsed_s": 0.01},
        )


def make_scripted_session(
    inputs: list[str],
    core: BotCore | None = None,
    config: RelayConfig | None = None,
    output_sink: list[str] | None = None,
) -> tuple[RelaySession, list[str]]:
    """Helper to construct a RelaySession driven by scripted inputs."""
    input_iter = iter(inputs)
    outputs = output_sink if output_sink is not None else []

    def mock_input(prompt: str = "") -> str:
        try:
            val = next(input_iter)
            outputs.append(f"{prompt}{val}")
            return val
        except StopIteration:
            raise EOFError("End of scripted inputs")

    def mock_print(*args, **kwargs) -> None:
        sep = kwargs.get("sep", " ")
        msg = sep.join(str(a) for a in args)
        outputs.append(msg)

    if core is None:
        cfg = config or RelayConfig()
        core = build_bot_core(cfg, use_dummy=True)

    session = RelaySession(
        core=core,
        config=config or RelayConfig(countdown=False),
        input_fn=mock_input,
        print_fn=mock_print,
        sleep_fn=lambda _: None,
        time_fn=lambda: 100.0,
    )
    return session, outputs


# ---------------------------------------------------------------------------
# Tests: Banner & Disclaimers
# ---------------------------------------------------------------------------


def test_banner_disclaimer_notice() -> None:
    """Verify mandatory fair play disclaimer is present verbatim."""
    required_text = (
        "Intended for games against built-in computer Bots only, never against human opponents."
    )
    assert required_text in BANNER

    session, outputs = make_scripted_session(inputs=["quit"])
    session.run()
    joined = "\n".join(outputs)
    assert required_text in joined
    assert "FAIR PLAY NOTICE" in joined


# ---------------------------------------------------------------------------
# Tests: Clock Parsing and Formatting
# ---------------------------------------------------------------------------


def test_parse_clock_str() -> None:
    assert parse_clock_str("180") == 180.0
    assert parse_clock_str("175.5") == 175.5
    assert parse_clock_str("2:58") == 178.0
    assert parse_clock_str("02:58") == 178.0
    assert parse_clock_str("0:02:58") == 178.0
    assert parse_clock_str("1:15:30") == 4530.0

    with pytest.raises(ValueError):
        parse_clock_str("abc")
    with pytest.raises(ValueError):
        parse_clock_str("1:2:3:4")


def test_format_clock_str() -> None:
    assert format_clock_str(178.0) == "2:58"
    assert format_clock_str(59.0) == "0:59"
    assert format_clock_str(3665.0) == "1:01:05"
    assert format_clock_str(-5.0) == "0:00"


def test_format_pgn_clk() -> None:
    assert format_pgn_clk(178.0) == "[%clk 0:02:58]"
    assert format_pgn_clk(3665.0) == "[%clk 1:01:05]"
    assert format_pgn_clk(0.0) == "[%clk 0:00:00]"


# ---------------------------------------------------------------------------
# Tests: Move Input & Typo Handling
# ---------------------------------------------------------------------------


def test_parse_move_input() -> None:
    board = chess.Board()
    # SAN
    m1 = parse_move_input(board, "e4")
    assert m1 == chess.Move.from_uci("e2e4")

    # UCI
    m2 = parse_move_input(board, "e2e4")
    assert m2 == chess.Move.from_uci("e2e4")

    # Knight move
    m3 = parse_move_input(board, "Nf3")
    assert m3 == chess.Move.from_uci("g1f3")

    # Invalid moves
    with pytest.raises(ValueError):
        parse_move_input(board, "e5")  # Illegal for white on move 1
    with pytest.raises(ValueError):
        parse_move_input(board, "gibberish")
    with pytest.raises(ValueError):
        parse_move_input(board, "")


def test_typo_correction_does_not_crash() -> None:
    """Typo move inputs should prompt error and remain on current turn."""
    # User types 'asdf' (typo), then 'e9' (typo), then legal 'e4', then 'quit'
    session, outputs = make_scripted_session(inputs=["asdf", "e9", "e4", "quit"])
    session.run()

    joined = "\n".join(outputs)
    assert "not a valid legal move" in joined or "not a valid move" in joined
    assert "Opponent played: e4" in joined
    # Board should have registered e4 and bot reply
    assert session.board.fullmove_number >= 2


# ---------------------------------------------------------------------------
# Tests: Clock Commands & In-line Clock Updates
# ---------------------------------------------------------------------------


def test_inline_clock_parsing() -> None:
    """Opponent move with clock tokens updates both clocks."""
    session, _ = make_scripted_session(inputs=["e4 172.5 178.0", "quit"])
    session.run()

    assert session.opp_clock == 172.5
    # Bot clock started at 180, set to 178, then bot moved
    assert session.board.piece_at(chess.E4) is not None


def test_cmd_clock() -> None:
    session, _ = make_scripted_session(inputs=["clock 150 160", "quit"])
    session.run()

    assert session.bot_clock == 150.0
    assert session.opp_clock == 160.0


# ---------------------------------------------------------------------------
# Tests: Status, Legal, Help Commands
# ---------------------------------------------------------------------------


def test_status_and_legal_commands() -> None:
    session, outputs = make_scripted_session(inputs=["status", "legal", "help", "quit"])
    session.run()

    joined = "\n".join(outputs)
    assert "+-----------------+" in joined
    assert "a b c d e f g h" in joined
    assert "Legal moves (20):" in joined
    assert "Available Commands:" in joined


# ---------------------------------------------------------------------------
# Tests: Undo Command
# ---------------------------------------------------------------------------


def test_undo_restores_position_and_clocks() -> None:
    """Undoing a turn rolls back both bot reply and opponent move."""
    # 1. Opponent plays e4 -> bot plays reply
    # 2. Undo
    # 3. Quit
    session, _ = make_scripted_session(inputs=["e4", "undo", "quit"])
    session.run()

    # After undo, board must be back to starting position!
    assert session.board.fen() == chess.STARTING_FEN
    assert session.clk_white == session.config.time_control_s
    assert session.clk_black == session.config.time_control_s
    assert len(session.history) == 0


# ---------------------------------------------------------------------------
# Tests: FEN Resync
# ---------------------------------------------------------------------------


def test_fen_resync() -> None:
    # FEN where Black is to move (bot is Black)
    fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    session, _ = make_scripted_session(inputs=[f"fen {fen}", "quit"])
    session.run()

    # Bot should have made Black's move immediately from that position!
    assert session.board.fen() != fen
    assert session.board.turn == chess.WHITE


# ---------------------------------------------------------------------------
# Tests: PGN Resync
# ---------------------------------------------------------------------------


def test_pgn_resync(tmp_path: Path) -> None:
    sample_pgn = """[Event "Sample"]
[Site "Local"]
[Date "2026.09.28"]
[Round "1"]
[White "Opponent"]
[Black "HLC Bot"]
[Result "*"]
[TimeControl "180+0"]

1. e4 { [%clk 0:02:58] } 1... e5 { [%clk 0:02:57] } *
"""
    pgn_file = tmp_path / "test_resync.pgn"
    pgn_file.write_text(sample_pgn, encoding="utf-8")

    session, _ = make_scripted_session(inputs=[f"pgn {pgn_file}", "quit"])
    session.run()

    assert session.board.piece_at(chess.E4) is not None
    assert session.board.piece_at(chess.E5) is not None
    assert session.clk_white == 178.0
    assert session.clk_black == 177.0


# ---------------------------------------------------------------------------
# Tests: Bot as White (immediate first move)
# ---------------------------------------------------------------------------


def test_bot_as_white_plays_immediately() -> None:
    config = RelayConfig(bot_color=chess.WHITE, countdown=False)
    session, _ = make_scripted_session(inputs=["quit"], config=config)
    session.run()

    # Move 1 by White was made without needing user move input!
    assert session.board.fullmove_number >= 1
    assert session.board.turn == chess.BLACK
    assert len(session.history) == 1
    assert session.history[0].is_bot is True
    assert session.history[0].color == chess.WHITE


# ---------------------------------------------------------------------------
# Tests: Scripted Full Game & PGN Replay Acceptance
# ---------------------------------------------------------------------------


def test_scripted_full_game_and_pgn_replay(tmp_path: Path) -> None:
    """Acceptance test: Full game played to checkmate and replayed from PGN log."""
    # Play Fool's mate:
    # 1. f3 e5 2. g4 Qh4# (0-1)
    # White = Opponent, Black = Bot
    fool_map = {
        # Board after 1. f3 -> Bot plays e7e5
        "rnbqkbnr/pppppppp/8/8/8/5P2/PPPPP1PP/RNBQKBNR b KQkq - 0 1": "e7e5",
        # Board after 1. f3 e5 2. g4 -> Bot plays d8h4#
        "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2": "d8h4",
    }
    core = FixedDecisionCore(move_map=fool_map)
    config = RelayConfig(
        bot_color=chess.BLACK,
        log_dir=tmp_path,
        countdown=False,
        time_control_s=180.0,
    )

    # Scripted opponent moves: f3, then g4
    session, _ = make_scripted_session(
        inputs=["f3", "g4"],
        core=core,
        config=config,
    )
    session.run()

    assert session.board.is_checkmate()
    assert session.board.result() == "0-1"
    assert session.last_pgn_path is not None
    assert session.last_pgn_path.is_file()

    # Replay game from PGN log with python-chess
    with open(session.last_pgn_path, "r", encoding="utf-8") as f:
        replayed_game = chess.pgn.read_game(f)

    assert replayed_game is not None
    assert replayed_game.headers["Result"] == "0-1"
    assert replayed_game.headers["White"] == "Opponent Bot"
    assert replayed_game.headers["Black"] == "HLC Bot"

    # Verify all plies can be replayed and contain [%clk] comments
    replay_board = replayed_game.board()
    node = replayed_game
    ply_count = 0
    while node.variations:
        next_node = node.variation(0)
        assert next_node.comment.startswith("[%clk ")
        replay_board.push(next_node.move)
        ply_count += 1
        node = next_node

    assert ply_count == 4
    assert replay_board.is_checkmate()
    assert replay_board.fen() == session.board.fen()


# ---------------------------------------------------------------------------
# Tests: Integration with Maia-3 PyTorch Model
# ---------------------------------------------------------------------------


def test_maia3_integration_relay(tmp_path: Path) -> None:
    """Verify relay session executes a move with the real Maia-3 79M PyTorch engine."""
    engine = Maia3Engine(
        model="maia3-79m",
        elo=1500,
        device="cpu",
        use_amp=False,
        use_uci_history=False,
        multipv=1,
    )
    core = BotCore(
        move_engine=engine,
        timing_model=ClockOnlyBaseline(),
        scheduler=Scheduler(SchedulerConfig(min_delay=0.0, safety_margin=0.05)),
        rng=random.Random(42),
    )
    config = RelayConfig(
        bot_color=chess.BLACK,
        log_dir=tmp_path,
        countdown=False,
        time_control_s=180.0,
    )

    # Opponent plays e4, bot calculates real Maia-3 response, then quit
    session, _ = make_scripted_session(
        inputs=["e4", "quit"],
        core=core,
        config=config,
    )
    session.run()

    # Move 1 by opponent and response by bot must be legal
    assert len(session.history) == 2
    assert session.history[0].move == chess.Move.from_uci("e2e4")
    assert session.history[1].is_bot is True
    assert session.board.turn == chess.WHITE
