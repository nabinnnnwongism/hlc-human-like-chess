"""Manual Relay CLI: interactive terminal app for private testing vs chess computer bots.

Intended for games against built-in computer Bots only, never against human opponents.
"""

from __future__ import annotations

import argparse
import datetime
import io
import os
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.pgn

from hlc.core import BotCore
from hlc.engines.maia3 import Maia3Engine
from hlc.scheduler import Scheduler, SchedulerConfig
from hlc.timing.baselines import ClockOnlyBaseline, HeuristicBaseline
from hlc.types import Decision, GameState, MoveDistribution, MoveEngine

BANNER = """
================================================================================
⚠️  FAIR PLAY NOTICE
Intended for games against built-in computer Bots only, never against human opponents.
================================================================================
"""

_CLK_PATTERN = re.compile(r"\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]")


def parse_clock_str(val: str) -> float:
    """Parse clock string (seconds, mm:ss, or h:mm:ss) into float seconds."""
    val = val.strip()
    if ":" in val:
        parts = val.split(":")
        if len(parts) == 2:
            mins, secs = parts
            return int(mins) * 60.0 + float(secs)
        elif len(parts) == 3:
            hours, mins, secs = parts
            return int(hours) * 3600.0 + int(mins) * 60.0 + float(secs)
        else:
            raise ValueError(f"Invalid time format: '{val}'. Expected seconds, mm:ss, or h:mm:ss.")
    return float(val)


def format_clock_str(seconds: float) -> str:
    """Format seconds into standard H:MM:SS or M:SS string for display."""
    s = max(0.0, seconds)
    total_secs = int(s)
    hours = total_secs // 3600
    mins = (total_secs % 3600) // 60
    secs = total_secs % 60
    if hours > 0:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"


def format_pgn_clk(seconds: float) -> str:
    """Format seconds into [%clk H:MM:SS] comment for PGN logs."""
    s = max(0.0, seconds)
    total_secs = int(s)
    hours = total_secs // 3600
    mins = (total_secs % 3600) // 60
    secs = total_secs % 60
    return f"[%clk {hours}:{mins:02d}:{secs:02d}]"


def parse_move_input(board: chess.Board, move_str: str) -> chess.Move:
    """Parse a move string as either UCI or SAN against current legal moves."""
    clean = move_str.strip()
    if not clean:
        raise ValueError("Empty move string.")

    # 1. Try UCI (e.g. e2e4, g1f3, e7e8q)
    try:
        candidate = chess.Move.from_uci(clean.lower())
        if candidate in board.legal_moves:
            return candidate
    except ValueError:
        pass

    # 2. Try SAN (e.g. e4, Nf3, O-O, exd5)
    try:
        return board.parse_san(clean)
    except (
        chess.InvalidMoveError,
        chess.IllegalMoveError,
        chess.AmbiguousMoveError,
        ValueError,
    ) as err:
        raise ValueError(f"'{clean}' is not a valid legal move in this position.") from err


def render_board(board: chess.Board) -> str:
    """Render a clear ASCII board diagram with coordinate labels."""
    lines = []
    lines.append("  +-----------------+")
    ranks = str(board).split("\n")
    for r_idx, rank in enumerate(ranks):
        rank_num = 8 - r_idx
        lines.append(f"{rank_num} | {rank} |")
    lines.append("  +-----------------+")
    lines.append("    a b c d e f g h")
    return "\n".join(lines)


@dataclass
class MoveRecord:
    """Information preserved for each move played in the session."""

    move: chess.Move
    san: str
    is_bot: bool
    color: chess.Color
    clk_white_before: float
    clk_black_before: float
    clk_white_after: float
    clk_black_after: float
    delay_s: float = 0.0


@dataclass
class RelayConfig:
    """Configuration for a Manual Relay CLI session."""

    bot_color: chess.Color = chess.BLACK
    time_control_s: float = 180.0
    increment_s: float = 0.0
    self_elo: int = 1500
    opp_elo: int = 1500
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 42
    timing_backend: str = "heuristic"  # "heuristic" | "clock_only"
    model: str = "maia3-79m"
    device: str = "cpu"
    log_dir: Path = field(default_factory=lambda: Path("runs"))
    countdown: bool = True
    countdown_tick: float = 0.2
    auto_clock: bool = True
    move_overhead: float = 0.05
    safety_margin: float = 0.10
    single_move_delay: float = 0.15


class DummyMoveEngine(MoveEngine):
    """Uniform random move engine for fast testing without model weights."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def get_distribution(self, state: GameState) -> MoveDistribution:
        moves = list(state.board.legal_moves)
        if not moves:
            return MoveDistribution(moves=[], probabilities=[])
        p = 1.0 / len(moves)
        return MoveDistribution(moves=moves, probabilities=[p] * len(moves))


class RelaySession:
    """Interactive Relay CLI game session."""

    def __init__(
        self,
        core: BotCore,
        config: RelayConfig | None = None,
        input_fn: Callable[[str], str] = input,
        print_fn: Callable[..., None] = print,
        sleep_fn: Callable[[float], None] = time.sleep,
        time_fn: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.core = core
        self.config = config or RelayConfig()
        self.input_fn = input_fn
        self.print_fn = print_fn
        self.sleep_fn = sleep_fn
        self.time_fn = time_fn

        self.board = chess.Board()
        self.starting_fen = chess.STARTING_FEN
        self.clk_white = self.config.time_control_s
        self.clk_black = self.config.time_control_s
        self.history: list[MoveRecord] = []
        self.game_over = False
        self.last_pgn_path: Path | None = None

    def _safe_print(self, msg: str = "", end: str = "\n", flush: bool = False) -> None:
        try:
            self.print_fn(msg, end=end, flush=flush)
        except TypeError:
            self.print_fn(msg)

    @property
    def bot_clock(self) -> float:
        return self.clk_white if self.config.bot_color == chess.WHITE else self.clk_black

    @bot_clock.setter
    def bot_clock(self, val: float) -> None:
        if self.config.bot_color == chess.WHITE:
            self.clk_white = max(0.0, val)
        else:
            self.clk_black = max(0.0, val)

    @property
    def opp_clock(self) -> float:
        return self.clk_black if self.config.bot_color == chess.WHITE else self.clk_white

    @opp_clock.setter
    def opp_clock(self, val: float) -> None:
        if self.config.bot_color == chess.WHITE:
            self.clk_black = max(0.0, val)
        else:
            self.clk_white = max(0.0, val)

    def print_banner(self) -> None:
        """Display startup disclaimer banner."""
        self._safe_print(BANNER.strip())
        bot_col_name = "White" if self.config.bot_color == chess.WHITE else "Black"
        self._safe_print(
            f"\nSession Config: Bot plays {bot_col_name} | TC: {int(self.config.time_control_s)}+{int(self.config.increment_s)} | Elo: Bot {self.config.self_elo} vs Opp {self.config.opp_elo}"
        )
        self._safe_print("Type 'help' for commands, 'board' for status, 'quit' to exit.\n")

    def print_status(self) -> None:
        """Display board and clock status."""
        self._safe_print("\n" + render_board(self.board))
        turn_str = "White" if self.board.turn == chess.WHITE else "Black"
        w_clk = format_clock_str(self.clk_white)
        b_clk = format_clock_str(self.clk_black)
        bot_label = " [Bot]" if self.config.bot_color == chess.WHITE else ""
        opp_label = " [Bot]" if self.config.bot_color == chess.BLACK else ""
        self._safe_print(f"Turn: {turn_str} | Move: {self.board.fullmove_number}")
        self._safe_print(f"Clocks -> White{bot_label}: {w_clk} | Black{opp_label}: {b_clk}\n")

    def print_help(self) -> None:
        """Display interactive help."""
        help_text = """
Available Commands:
  <move> [opp_clk] [bot_clk] : Play opponent move (SAN or UCI, e.g. 'e4', 'Nf3', 'e2e4', 'e4 175 180')
  undo                       : Take back the last turn (both bot reply and opponent move)
  fen <FEN_STRING>           : Resync position from a FEN string
  pgn <FILE_PATH>            : Resync position from a saved PGN file
  clock <bot_clk> [opp_clk]  : Update clocks (seconds or mm:ss, e.g. 'clock 2:45 2:50')
  board / status             : Show current board, clocks, and turn
  legal                      : Show all legal moves in current position
  help                       : Show this help message
  quit / exit                : Save game PGN and exit
"""
        self._safe_print(help_text.strip())

    def print_legal(self) -> None:
        """Display legal moves in SAN."""
        sans = [self.board.san(m) for m in self.board.legal_moves]
        self._safe_print(f"Legal moves ({len(sans)}): {', '.join(sans)}")

    def cmd_undo(self) -> None:
        """Undo last turn (both bot move and opponent move, restoring state)."""
        if not self.history:
            self._safe_print("⚠️ Nothing to undo: already at starting position.")
            return

        # Determine plies to undo:
        # If it's opponent's turn, bot moved last, and opponent before that -> undo 2 plies.
        # If it's bot's turn, opponent moved last -> undo 1 ply.
        is_opp_turn = self.board.turn != self.config.bot_color
        plies_to_undo = (
            2 if (is_opp_turn and len(self.history) >= 2 and self.history[-1].is_bot) else 1
        )

        undone_records: list[MoveRecord] = []
        for _ in range(plies_to_undo):
            if self.history:
                rec = self.history.pop()
                self.board.pop()
                undone_records.append(rec)

        if undone_records:
            earliest = undone_records[-1]
            self.clk_white = earliest.clk_white_before
            self.clk_black = earliest.clk_black_before
            self._safe_print(
                f"↩️ Undone {len(undone_records)} move(s). Board restored to move {self.board.fullmove_number}."
            )
            self.print_status()

    def cmd_fen(self, fen_str: str) -> None:
        """Resync position from FEN."""
        try:
            new_board = chess.Board(fen_str.strip())
        except ValueError as err:
            self._safe_print(f"⚠️ Invalid FEN string: {err}")
            return

        self.board = new_board
        self.starting_fen = new_board.fen()
        self.history.clear()
        self._safe_print(f"✅ Board resynced from FEN: {self.starting_fen}")
        self.print_status()
        self._check_game_over()

    def cmd_pgn(self, pgn_arg: str) -> None:
        """Resync position from PGN file or string."""
        path = Path(pgn_arg.strip())
        game: chess.pgn.Game | None = None
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    game = chess.pgn.read_game(f)
            except (OSError, ValueError) as err:
                self._safe_print(f"⚠️ Error reading PGN file: {err}")
                return
        else:
            game = chess.pgn.read_game(io.StringIO(pgn_arg))

        if game is None:
            self._safe_print(f"⚠️ Could not parse PGN from '{pgn_arg}'.")
            return

        # Replay game from start
        self.board = game.board()
        self.starting_fen = self.board.fen()
        self.history.clear()

        # Parse initial time control if present
        tc = game.headers.get("TimeControl", "")
        if "+" in tc:
            try:
                base_s, inc_s = tc.split("+")
                self.config.time_control_s = float(base_s)
                self.config.increment_s = float(inc_s)
            except ValueError:
                pass

        self.clk_white = self.config.time_control_s
        self.clk_black = self.config.time_control_s

        moves_replayed = 0
        node = game
        while node.variations:
            next_node = node.variation(0)
            move = next_node.move
            san = self.board.san(move)
            is_white = self.board.turn == chess.WHITE
            is_bot = self.board.turn == self.config.bot_color

            clk_w_before = self.clk_white
            clk_b_before = self.clk_black

            # Check for [%clk] comment
            parsed_clk = None
            if next_node.comment:
                m = _CLK_PATTERN.search(next_node.comment)
                if m:
                    parsed_clk = (
                        int(m.group(1)) * 3600.0 + int(m.group(2)) * 60.0 + float(m.group(3))
                    )

            self.board.push(move)
            moves_replayed += 1

            if parsed_clk is not None:
                if is_white:
                    self.clk_white = parsed_clk
                else:
                    self.clk_black = parsed_clk

            self.history.append(
                MoveRecord(
                    move=move,
                    san=san,
                    is_bot=is_bot,
                    color=chess.WHITE if is_white else chess.BLACK,
                    clk_white_before=clk_w_before,
                    clk_black_before=clk_b_before,
                    clk_white_after=self.clk_white,
                    clk_black_after=self.clk_black,
                )
            )
            node = next_node

        self._safe_print(f"✅ Replayed {moves_replayed} plies from PGN.")
        self.print_status()
        self._check_game_over()

    def cmd_clock(self, args: list[str]) -> None:
        """Update clocks explicitly: clock <bot_clk> [opp_clk]."""
        if not args:
            self._safe_print(
                f"Current clocks: Bot: {format_clock_str(self.bot_clock)}, Opponent: {format_clock_str(self.opp_clock)}"
            )
            return

        try:
            bot_val = parse_clock_str(args[0])
            self.bot_clock = bot_val
            if len(args) >= 2:
                opp_val = parse_clock_str(args[1])
                self.opp_clock = opp_val
            self._safe_print(
                f"⏱️ Clocks updated: Bot: {format_clock_str(self.bot_clock)} | Opponent: {format_clock_str(self.opp_clock)}"
            )
        except ValueError as err:
            self._safe_print(f"⚠️ {err}")

    def execute_bot_turn(self) -> None:
        """Perform bot move selection, display countdown, and apply move."""
        if self._check_game_over():
            return

        state = GameState(
            board=self.board,
            move_history=[rec.move for rec in self.history],
            clock_self=self.bot_clock,
            clock_opp=self.opp_clock,
            increment=self.config.increment_s,
            self_elo=self.config.self_elo,
            opp_elo=self.config.opp_elo,
        )

        decision: Decision = self.core.decide(state)
        move = decision.move
        delay_s = max(0.0, decision.delay_s)
        san = self.board.san(move)
        uci = move.uci()
        compute_elapsed = decision.debug.get("compute_elapsed_s", 0.0)

        self._safe_print(f"\n🤖 Bot proposes: {san} (UCI: {uci}) in {delay_s:.1f}s")
        self._safe_print(f"👉 Play '{san}' on your board when countdown finishes.")

        # Countdown display
        if self.config.countdown and delay_s > 0.05:
            remaining = delay_s
            tick_start = self.time_fn()
            while remaining > 0.01:
                self._safe_print(
                    f"\r  ⏳ Think time remaining: {remaining:4.1f}s / {delay_s:.1f}s ... ",
                    end="",
                    flush=True,
                )
                step = min(self.config.countdown_tick, remaining)
                self.sleep_fn(step)
                elapsed = self.time_fn() - tick_start
                remaining = max(0.0, delay_s - elapsed)
            self._safe_print(
                f"\r  ✅ Think time elapsed ({delay_s:.1f}s).                       \n"
            )
        else:
            self._safe_print(f"  ⚡ Immediate move (delay {delay_s:.2f}s)\n")

        self._safe_print(f"\n{'=' * 42}")
        self._safe_print(f"  👉 PLAY NOW: {san} (UCI: {uci})")
        self._safe_print(f"{'=' * 42}\n")

        clk_w_before = self.clk_white
        clk_b_before = self.clk_black

        # Update bot clock
        total_time_spent = delay_s + compute_elapsed
        self.bot_clock = max(0.0, self.bot_clock - total_time_spent + self.config.increment_s)

        self.board.push(move)
        self.history.append(
            MoveRecord(
                move=move,
                san=san,
                is_bot=True,
                color=self.config.bot_color,
                clk_white_before=clk_w_before,
                clk_black_before=clk_b_before,
                clk_white_after=self.clk_white,
                clk_black_after=self.clk_black,
                delay_s=delay_s,
            )
        )

        self._check_game_over()

    def process_opponent_move(self, move_token: str, clocks: list[str], elapsed_s: float) -> bool:
        """Validate and apply opponent's move, updating clocks."""
        try:
            move = parse_move_input(self.board, move_token)
        except ValueError as err:
            self._safe_print(f"⚠️ {err}")
            self._safe_print("  (Type 'legal' to list all legal moves, or 'help' for commands)")
            return False

        san = self.board.san(move)
        clk_w_before = self.clk_white
        clk_b_before = self.clk_black

        # Handle clocks: manual entry overrides auto-clock
        if len(clocks) >= 2:
            try:
                opp_c = parse_clock_str(clocks[0])
                bot_c = parse_clock_str(clocks[1])
                self.opp_clock = opp_c
                self.bot_clock = bot_c
            except ValueError as err:
                self._safe_print(f"⚠️ Clock parse warning: {err}. Using auto-clock.")
                self.opp_clock = max(0.0, self.opp_clock - elapsed_s + self.config.increment_s)
        elif len(clocks) == 1:
            try:
                opp_c = parse_clock_str(clocks[0])
                self.opp_clock = opp_c
            except ValueError as err:
                self._safe_print(f"⚠️ Clock parse warning: {err}. Using auto-clock.")
                self.opp_clock = max(0.0, self.opp_clock - elapsed_s + self.config.increment_s)
        else:
            # Auto clock
            self.opp_clock = max(0.0, self.opp_clock - elapsed_s + self.config.increment_s)

        opp_color = chess.BLACK if self.config.bot_color == chess.WHITE else chess.WHITE
        self.board.push(move)
        self.history.append(
            MoveRecord(
                move=move,
                san=san,
                is_bot=False,
                color=opp_color,
                clk_white_before=clk_w_before,
                clk_black_before=clk_b_before,
                clk_white_after=self.clk_white,
                clk_black_after=self.clk_black,
                delay_s=elapsed_s,
            )
        )

        self._safe_print(
            f"✔️ Opponent played: {san} (Clocks -> White: {format_clock_str(self.clk_white)}, Black: {format_clock_str(self.clk_black)})"
        )
        self._check_game_over()
        return True

    def _check_game_over(self) -> bool:
        """Check terminal game conditions and output result."""
        if not self.board.is_game_over():
            return False

        self.game_over = True
        result = self.board.result()
        term = ""
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            term = f"Checkmate - {winner} wins"
        elif self.board.is_stalemate():
            term = "Stalemate"
        elif self.board.is_insufficient_material():
            term = "Draw by insufficient material"
        elif self.board.is_seventyfive_moves():
            term = "Draw by 75-move rule"
        elif self.board.is_fivefold_repetition():
            term = "Draw by 5-fold repetition"
        else:
            term = "Game over"

        self._safe_print(f"\n🏁 GAME OVER: {result} ({term})")
        self.save_pgn()
        return True

    def build_pgn(self, result: str | None = None) -> chess.pgn.Game:
        """Construct python-chess Game object with [%clk] annotations."""
        game = chess.pgn.Game()
        game.headers["Event"] = "HLC Manual Relay Bot vs Opponent"
        game.headers["Site"] = "Local (Private Bot Testing)"
        game.headers["Date"] = datetime.datetime.now(datetime.UTC).strftime("%Y.%m.%d")
        game.headers["Round"] = "1"
        game.headers["White"] = (
            "HLC Bot" if self.config.bot_color == chess.WHITE else "Opponent Bot"
        )
        game.headers["Black"] = (
            "HLC Bot" if self.config.bot_color == chess.BLACK else "Opponent Bot"
        )
        game.headers["WhiteElo"] = str(
            self.config.self_elo if self.config.bot_color == chess.WHITE else self.config.opp_elo
        )
        game.headers["BlackElo"] = str(
            self.config.self_elo if self.config.bot_color == chess.BLACK else self.config.opp_elo
        )
        game.headers["TimeControl"] = (
            f"{int(self.config.time_control_s)}+{int(self.config.increment_s)}"
        )

        if result is not None:
            game.headers["Result"] = result
        elif self.board.is_game_over():
            game.headers["Result"] = self.board.result()
        else:
            game.headers["Result"] = "*"

        if self.starting_fen != chess.STARTING_FEN:
            game.headers["SetUp"] = "1"
            game.headers["FEN"] = self.starting_fen

        node = game
        for rec in self.history:
            node = node.add_variation(rec.move)
            clk_after = rec.clk_white_after if rec.color == chess.WHITE else rec.clk_black_after
            node.comment = format_pgn_clk(clk_after)

        return game

    def save_pgn(self, filepath: Path | None = None) -> Path:
        """Save the session game log to PGN file."""
        if filepath is None:
            os.makedirs(self.config.log_dir, exist_ok=True)
            timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
            filepath = self.config.log_dir / f"relay_{timestamp}.pgn"

        game = self.build_pgn()
        with open(filepath, "w", encoding="utf-8") as f:
            exporter = chess.pgn.FileExporter(f)
            game.accept(exporter)

        self.last_pgn_path = filepath
        self._safe_print(f"📁 Game log saved to: {filepath.resolve()}")
        return filepath

    def run(self) -> None:
        """Start the interactive manual relay CLI loop."""
        self.print_banner()
        self.print_status()

        # If bot plays White, bot makes move 1 immediately!
        if self.board.turn == self.config.bot_color and not self.game_over:
            self.execute_bot_turn()
            if not self.game_over:
                self.print_status()

        while not self.game_over:
            t_prompt = self.time_fn()
            try:
                raw_input = self.input_fn("Enter move (or command): ").strip()
            except (EOFError, KeyboardInterrupt):
                self._safe_print("\nSession interrupted. Saving game...")
                self.save_pgn()
                break

            if not raw_input:
                continue

            tokens = raw_input.split()
            cmd = tokens[0].lower()

            if cmd in ("quit", "exit"):
                self._safe_print("Exiting session. Saving game...")
                self.save_pgn()
                break
            elif cmd == "help":
                self.print_help()
                continue
            elif cmd in ("board", "status"):
                self.print_status()
                continue
            elif cmd == "legal":
                self.print_legal()
                continue
            elif cmd == "undo":
                self.cmd_undo()
                continue
            elif cmd == "fen":
                fen_arg = " ".join(tokens[1:])
                self.cmd_fen(fen_arg)
                if self.board.turn == self.config.bot_color and not self.game_over:
                    self.execute_bot_turn()
                    if not self.game_over:
                        self.print_status()
                continue
            elif cmd == "pgn":
                pgn_arg = " ".join(tokens[1:])
                self.cmd_pgn(pgn_arg)
                if self.board.turn == self.config.bot_color and not self.game_over:
                    self.execute_bot_turn()
                    if not self.game_over:
                        self.print_status()
                continue
            elif cmd in ("clock", "clocks"):
                self.cmd_clock(tokens[1:])
                continue

            # Otherwise, attempt to parse as opponent move
            elapsed = self.time_fn() - t_prompt
            move_token = tokens[0]
            clock_tokens = tokens[1:]

            success = self.process_opponent_move(move_token, clock_tokens, elapsed)
            if not success:
                continue

            if not self.game_over and self.board.turn == self.config.bot_color:
                self.execute_bot_turn()
                if not self.game_over:
                    self.print_status()


def build_bot_core(config: RelayConfig, use_dummy: bool = False) -> BotCore:
    """Factory helper to construct BotCore for relay CLI."""
    rng = random.Random(config.seed)
    if use_dummy:
        move_engine: MoveEngine = DummyMoveEngine(seed=config.seed)
    else:
        move_engine = Maia3Engine(
            model=config.model,
            elo=config.self_elo,
            device=config.device,
            use_amp=False,
            use_uci_history=True,
            multipv=1,
        )

    if config.timing_backend == "clock_only":
        timing_model = ClockOnlyBaseline(elo=config.self_elo)
    else:
        timing_model = HeuristicBaseline(elo=config.self_elo)

    scheduler = Scheduler(
        SchedulerConfig(
            move_overhead=config.move_overhead,
            safety_margin=config.safety_margin,
            single_move_delay=config.single_move_delay,
        )
    )

    return BotCore(
        move_engine=move_engine,
        timing_model=timing_model,
        scheduler=scheduler,
        rng=rng,
        temperature=config.temperature,
        top_p=config.top_p,
        enable_premove=True,
        enable_ar1=True,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for Manual Relay."""
    parser = argparse.ArgumentParser(
        description="HLC Manual Relay CLI: Private testing against computer bots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--color", choices=["white", "black"], default="black", help="Color played by the bot."
    )
    parser.add_argument("--time", type=float, default=180.0, help="Initial clock in seconds.")
    parser.add_argument(
        "--inc", type=float, default=0.0, help="Clock increment per move in seconds."
    )
    parser.add_argument("--self-elo", type=int, default=1500, help="Bot Elo rating target.")
    parser.add_argument("--opp-elo", type=int, default=1500, help="Opponent Elo rating.")
    parser.add_argument(
        "--timing",
        choices=["heuristic", "clock_only"],
        default="heuristic",
        help="Timing model backend.",
    )
    parser.add_argument("--model", type=str, default="maia3-79m", help="Maia-3 model name or path.")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (cpu or cuda).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument(
        "--no-countdown", action="store_true", help="Disable visual countdown timer delay."
    )
    parser.add_argument(
        "--dummy-engine",
        action="store_true",
        help="Use random move engine (for quick offline testing).",
    )
    parser.add_argument(
        "--log-dir", type=str, default="runs", help="Directory where PGN game logs are saved."
    )

    args = parser.parse_args(argv)

    bot_color = chess.WHITE if args.color.lower() == "white" else chess.BLACK
    config = RelayConfig(
        bot_color=bot_color,
        time_control_s=args.time,
        increment_s=args.inc,
        self_elo=args.self_elo,
        opp_elo=args.opp_elo,
        seed=args.seed,
        timing_backend=args.timing,
        model=args.model,
        device=args.device,
        log_dir=Path(args.log_dir),
        countdown=not args.no_countdown,
    )

    core = build_bot_core(config, use_dummy=args.dummy_engine)
    session = RelaySession(core=core, config=config)
    session.run()


if __name__ == "__main__":
    main()
