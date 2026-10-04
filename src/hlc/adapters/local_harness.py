"""LocalHarness: runs HLC bot vs Stockfish or Maia-3 5M on a virtual/real-time clock.

Writes PGNs with [%clk] tags to the runs/ directory.
"""

from __future__ import annotations

import datetime
import random
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.engine
import chess.pgn

from hlc.core import BotCore
from hlc.types import GameState

# ---------------------------------------------------------------------------
# Opponent protocol
# ---------------------------------------------------------------------------


class HarnessOpponent(ABC):
    """Abstract opponent in LocalHarness games."""

    @abstractmethod
    def select_move(
        self, board: chess.Board, time_remaining: float, increment: float
    ) -> chess.Move:
        """Return the chosen move and optionally simulate think time."""
        ...

    def close(self) -> None:
        """Clean up any resources."""


class StockfishOpponent(HarnessOpponent):
    """Stockfish opponent running at a given UCI_Elo level.

    Verified options: Stockfish 19 supports UCI_LimitStrength (check) and
    UCI_Elo (int, min=1320 max=3190 default=1320).
    """

    def __init__(
        self,
        stockfish_path: str | Path = "bin/stockfish/stockfish.exe",
        elo: int = 1500,
        movetime_ms: int = 100,
    ) -> None:
        self.elo = max(1320, min(3190, elo))
        self.movetime_ms = movetime_ms
        self._engine = chess.engine.SimpleEngine.popen_uci(str(stockfish_path))
        self._engine.configure({"UCI_LimitStrength": True, "UCI_Elo": self.elo})

    def select_move(
        self, board: chess.Board, time_remaining: float, increment: float
    ) -> chess.Move:
        result = self._engine.play(
            board,
            limit=chess.engine.Limit(time=self.movetime_ms / 1000.0),
        )
        if result.move is None or result.move not in board.legal_moves:
            # Fallback: first legal move
            return next(iter(board.legal_moves))
        return result.move

    def close(self) -> None:
        try:
            self._engine.quit()
        except Exception:
            pass


class Maia3Opponent(HarnessOpponent):
    """Maia-3 5M (or 79M) opponent as the opposing side."""

    def __init__(
        self,
        model: str = "maia3-5m",
        elo: int = 1500,
    ) -> None:
        cmd = [
            sys.executable,
            "-m",
            "maia3.uci",
            "--model",
            model,
            "--use-uci-history",
            "--elo",
            str(elo),
            "--device",
            "cpu",
            "--no-use-amp",
        ]
        self._engine = chess.engine.SimpleEngine.popen_uci(cmd)

    def select_move(
        self, board: chess.Board, time_remaining: float, increment: float
    ) -> chess.Move:
        result = self._engine.play(board, limit=chess.engine.Limit(nodes=1))
        if result.move is None or result.move not in board.legal_moves:
            return next(iter(board.legal_moves))
        return result.move

    def close(self) -> None:
        try:
            self._engine.quit()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Game record
# ---------------------------------------------------------------------------


@dataclass
class GameRecord:
    """Captures the full record of a LocalHarness game."""

    bot_color: chess.Color
    bot_elo: int
    opponent_elo: int
    time_control: str
    moves: list[chess.Move] = field(default_factory=list)
    clk_before_white: list[float] = field(default_factory=list)  # clock before each white move
    clk_before_black: list[float] = field(default_factory=list)  # clock before each black move
    actual_delays: list[float] = field(default_factory=list)  # actual think times per ply
    planned_delays: list[float] = field(default_factory=list)  # scheduler planned delays
    result: str = "*"  # "1-0", "0-1", "1/2-1/2", "*"
    termination: str = ""

    def to_pgn(self) -> chess.pgn.Game:
        """Render to python-chess Game with [%clk] annotations."""
        game = chess.pgn.Game()
        game.headers["Event"] = "HLC LocalHarness"
        game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
        game.headers["White"] = "HLC Bot" if self.bot_color == chess.WHITE else "Opponent"
        game.headers["Black"] = "HLC Bot" if self.bot_color == chess.BLACK else "Opponent"
        game.headers["WhiteElo"] = str(
            self.bot_elo if self.bot_color == chess.WHITE else self.opponent_elo
        )
        game.headers["BlackElo"] = str(
            self.bot_elo if self.bot_color == chess.BLACK else self.opponent_elo
        )
        game.headers["TimeControl"] = self.time_control
        game.headers["Result"] = self.result

        node = game
        white_clk_list = list(self.clk_before_white)
        black_clk_list = list(self.clk_before_black)

        for i, move in enumerate(self.moves):
            is_white_move = i % 2 == 0
            if is_white_move and white_clk_list:
                clk = white_clk_list.pop(0)
            elif not is_white_move and black_clk_list:
                clk = black_clk_list.pop(0)
            else:
                clk = 0.0

            node = node.add_variation(move)
            mins = int(clk) // 60
            secs = clk % 60
            node.comment = f"[%clk {mins}:{secs:05.2f}]"

        game.headers["Termination"] = self.termination
        return game


# ---------------------------------------------------------------------------
# LocalHarness
# ---------------------------------------------------------------------------


class LocalHarness:
    """Runs HLC bot games against an opponent.

    Supports two clock modes:
    - virtual: no wall-clock waiting; clocks are decremented by scheduled delays only
    - realtime: sleeps the planned delay and measures actual think time

    Outputs PGNs with [%clk] tags to runs/.
    """

    def __init__(
        self,
        bot_core: BotCore,
        opponent: HarnessOpponent,
        runs_dir: str | Path = "runs",
        time_seconds: float = 180.0,
        increment: float = 0.0,
        bot_color: chess.Color = chess.WHITE,
        bot_elo: int = 1500,
        opponent_elo: int = 1500,
        clock_mode: str = "virtual",  # "virtual" | "realtime"
        rng: random.Random | None = None,
        verbose: bool = False,
    ) -> None:
        self.bot = bot_core
        self.opponent = opponent
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(exist_ok=True)
        self.time_seconds = time_seconds
        self.increment = increment
        self.bot_color = bot_color
        self.bot_elo = bot_elo
        self.opponent_elo = opponent_elo
        self.clock_mode = clock_mode
        self.rng = rng if rng is not None else random.Random(42)
        self.verbose = verbose

    def _build_state(
        self,
        board: chess.Board,
        clock_bot: float,
        clock_opp: float,
    ) -> GameState:
        return GameState(
            board=board.copy(),
            move_history=list(board.move_stack),
            clock_self=clock_bot,
            clock_opp=clock_opp,
            increment=self.increment,
            self_elo=self.bot_elo,
            opp_elo=self.opponent_elo,
        )

    def play_game(self) -> GameRecord:
        """Play one full game; return a GameRecord."""
        board = chess.Board()
        clock_white = self.time_seconds
        clock_black = self.time_seconds

        record = GameRecord(
            bot_color=self.bot_color,
            bot_elo=self.bot_elo,
            opponent_elo=self.opponent_elo,
            time_control=f"{int(self.time_seconds)}+{int(self.increment)}",
        )

        move_num = 0
        while not board.is_game_over(claim_draw=True):
            is_bot_turn = board.turn == self.bot_color

            # Record clocks before move
            if board.turn == chess.WHITE:
                record.clk_before_white.append(clock_white)
            else:
                record.clk_before_black.append(clock_black)

            clock_self = clock_white if board.turn == chess.WHITE else clock_black
            clock_opp = clock_black if board.turn == chess.WHITE else clock_white

            if is_bot_turn:
                state = self._build_state(board, clock_self, clock_opp)
                t_start = time.perf_counter()
                decision = self.bot.decide(state)
                compute_elapsed = time.perf_counter() - t_start

                planned_delay = decision.delay_s
                record.planned_delays.append(planned_delay)

                if self.clock_mode == "realtime":
                    time.sleep(planned_delay)
                    actual_think = planned_delay
                else:
                    actual_think = planned_delay  # virtual: trust the scheduler

                move = decision.move
                if self.verbose:
                    san = board.san(move)
                    print(
                        f"  [{move_num + 1}] Bot ({'W' if self.bot_color == chess.WHITE else 'B'}) "
                        f"{san:<8} | delay={planned_delay:.1f}s "
                        f"| entropy={decision.debug.get('policy_entropy', 0):.2f} "
                        f"| clk_before={clock_self:.1f}s"
                    )
            else:
                t_opp_start = time.perf_counter()
                move = self.opponent.select_move(board, clock_self, self.increment)
                t_opp_end = time.perf_counter()
                actual_think = t_opp_end - t_opp_start
                planned_delay = actual_think
                record.planned_delays.append(planned_delay)

                if self.verbose:
                    san = board.san(move)
                    color_ch = "B" if self.bot_color == chess.WHITE else "W"
                    print(
                        f"  [{move_num + 1}] Opp ({color_ch}) {san:<8} | think={actual_think:.1f}s"
                    )

            record.actual_delays.append(actual_think)

            # Validate move
            if move not in board.legal_moves:
                raise RuntimeError(
                    f"Illegal move at ply {move_num}: {move} in position {board.fen()}"
                )
            board.push(move)
            record.moves.append(move)

            # Update clocks
            if board.turn != chess.WHITE:  # White just moved
                clock_white = max(0.0, clock_white - actual_think + self.increment)
            else:  # Black just moved
                clock_black = max(0.0, clock_black - actual_think + self.increment)

            # Check flagging
            if clock_white <= 0.0:
                record.result = "0-1"
                record.termination = "White ran out of time"
                break
            if clock_black <= 0.0:
                record.result = "1-0"
                record.termination = "Black ran out of time"
                break

            move_num += 1

        if record.result == "*":
            outcome = board.outcome(claim_draw=True)
            if outcome is not None:
                record.result = outcome.result()
                record.termination = outcome.termination.name.replace("_", " ").title()
            else:
                record.result = "*"

        return record

    def save_pgn(self, record: GameRecord, filename: str | None = None) -> Path:
        """Save a GameRecord as a PGN file with [%clk] annotations."""
        if filename is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"game_{ts}.pgn"
        path = self.runs_dir / filename
        game = record.to_pgn()
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(game))
            f.write("\n\n")
        return path

    def run_games(
        self,
        num_games: int,
        prefix: str = "harness",
    ) -> list[GameRecord]:
        """Run multiple games and save each PGN to runs/."""
        records = []
        for g in range(1, num_games + 1):
            if self.verbose:
                print(f"\n=== Game {g}/{num_games} ===")
            try:
                record = self.play_game()
                fname = f"{prefix}_game{g:04d}.pgn"
                path = self.save_pgn(record, fname)
                records.append(record)
                if self.verbose:
                    print(
                        f"  -> {record.result} | {record.termination} | {len(record.moves)} plies | PGN: {path}"
                    )
            except Exception as exc:
                print(f"  ERROR in game {g}: {exc}")
        return records
