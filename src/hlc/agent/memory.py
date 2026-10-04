"""memory.py — Persistent game and move memory for the HLC Native Agent.

Stores match history, move telemetry (entropy, think times, blunder contexts),
and performance metrics in a local SQLite database (data/hlc_agent.db).
"""

from __future__ import annotations

import datetime
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class GameRecord:
    id: int
    session_id: str
    timestamp: str
    platform: str
    color: str
    self_elo: int
    opp_elo: int
    result: str
    moves_count: int
    accuracy_percent: float
    blunders_count: int
    avg_think_time_s: float
    style_used: str
    fatigue_level: float
    notes: str = ""


class AgentMemory:
    """Thread-safe SQLite storage for the Native Agent.

    Uses a single persistent connection to avoid Windows WinError 32
    (file-in-use) errors that occur when multiple short-lived connections
    are opened and garbage collection hasn't released their handles yet.
    """

    def __init__(self, db_path: str | Path = "data/hlc_agent.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection = sqlite3.connect(
            str(self.db_path), check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        # Disable WAL so no -wal/-shm files are created (simpler cleanup).
        self._conn.execute("PRAGMA journal_mode=DELETE")
        self._init_db()

    def close(self) -> None:
        """Explicitly close the SQLite connection (important on Windows)."""
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    def __enter__(self) -> "AgentMemory":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Return the shared persistent connection."""
        return self._conn

    def _init_db(self) -> None:
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS games (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    platform TEXT DEFAULT 'chess.com',
                    color TEXT NOT NULL,
                    self_elo INTEGER NOT NULL,
                    opp_elo INTEGER NOT NULL,
                    result TEXT DEFAULT 'ongoing',
                    moves_count INTEGER DEFAULT 0,
                    accuracy_percent REAL DEFAULT 0.0,
                    blunders_count INTEGER DEFAULT 0,
                    avg_think_time_s REAL DEFAULT 0.0,
                    style_used TEXT NOT NULL,
                    fatigue_level REAL DEFAULT 0.0,
                    notes TEXT DEFAULT ''
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS moves (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id INTEGER NOT NULL,
                    ply INTEGER NOT NULL,
                    fen_before TEXT NOT NULL,
                    move_uci TEXT NOT NULL,
                    move_san TEXT NOT NULL,
                    think_time_s REAL NOT NULL,
                    policy_entropy REAL DEFAULT 0.0,
                    top_move_prob REAL DEFAULT 0.0,
                    was_top_move INTEGER DEFAULT 0,
                    is_blunder INTEGER DEFAULT 0,
                    blunder_plausibility REAL DEFAULT 0.0,
                    clock_remaining_s REAL DEFAULT 0.0,
                    FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE CASCADE
                )
            """)
            conn.commit()

    def start_game(
        self,
        session_id: str,
        color: str,
        self_elo: int,
        opp_elo: int,
        style_used: str,
        fatigue_level: float = 0.0,
        platform: str = "chess.com",
    ) -> int:
        """Create a new game entry in the database and return its game_id."""
        now = datetime.datetime.now().isoformat()
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO games (
                    session_id, timestamp, platform, color, self_elo, opp_elo,
                    result, style_used, fatigue_level
                ) VALUES (?, ?, ?, ?, ?, ?, 'ongoing', ?, ?)
            """, (session_id, now, platform, color, self_elo, opp_elo, style_used, fatigue_level))
            conn.commit()
            return cursor.lastrowid

    def record_move(
        self,
        game_id: int,
        ply: int,
        fen_before: str,
        move_uci: str,
        move_san: str,
        think_time_s: float,
        policy_entropy: float = 0.0,
        top_move_prob: float = 0.0,
        was_top_move: bool = False,
        is_blunder: bool = False,
        blunder_plausibility: float = 0.0,
        clock_remaining_s: float = 0.0,
    ) -> None:
        """Insert a move telemetry record."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO moves (
                    game_id, ply, fen_before, move_uci, move_san,
                    think_time_s, policy_entropy, top_move_prob,
                    was_top_move, is_blunder, blunder_plausibility,
                    clock_remaining_s
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                game_id, ply, fen_before, move_uci, move_san,
                think_time_s, policy_entropy, top_move_prob,
                1 if was_top_move else 0,
                1 if is_blunder else 0,
                blunder_plausibility, clock_remaining_s
            ))
            conn.commit()

    def finish_game(
        self,
        game_id: int,
        result: str,
        accuracy_percent: float = 0.0,
        notes: str = "",
    ) -> None:
        """Finalize game statistics from recorded moves."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            # Calculate summary stats from moves
            cursor.execute("""
                SELECT COUNT(*), AVG(think_time_s), SUM(is_blunder)
                FROM moves WHERE game_id = ?
            """, (game_id,))
            row = cursor.fetchone()
            moves_count = row[0] or 0
            avg_think = row[1] or 0.0
            blunders = row[2] or 0

            cursor.execute("""
                UPDATE games SET
                    result = ?,
                    moves_count = ?,
                    accuracy_percent = ?,
                    blunders_count = ?,
                    avg_think_time_s = ?,
                    notes = ?
                WHERE id = ?
            """, (result, moves_count, accuracy_percent, blunders, avg_think, notes, game_id))
            conn.commit()

    def get_recent_games(self, limit: int = 20) -> list[GameRecord]:
        """Fetch the most recent games."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, session_id, timestamp, platform, color, self_elo, opp_elo,
                       result, moves_count, accuracy_percent, blunders_count,
                       avg_think_time_s, style_used, fatigue_level, notes
                FROM games
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [
                GameRecord(
                    id=r["id"],
                    session_id=r["session_id"],
                    timestamp=r["timestamp"],
                    platform=r["platform"],
                    color=r["color"],
                    self_elo=r["self_elo"],
                    opp_elo=r["opp_elo"],
                    result=r["result"],
                    moves_count=r["moves_count"],
                    accuracy_percent=r["accuracy_percent"],
                    blunders_count=r["blunders_count"],
                    avg_think_time_s=r["avg_think_time_s"],
                    style_used=r["style_used"],
                    fatigue_level=r["fatigue_level"],
                    notes=r["notes"] or "",
                )
                for r in rows
            ]

    def get_stats_summary(self) -> dict[str, Any]:
        """Compute aggregate statistics across all recorded games."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM games WHERE result != 'ongoing'")
            total_games = cursor.fetchone()[0] or 0

            if total_games == 0:
                return {
                    "total_games": 0,
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "winrate_pct": 0.0,
                    "avg_accuracy": 0.0,
                    "avg_think_time": 0.0,
                    "total_blunders": 0,
                }

            cursor.execute("SELECT COUNT(*) FROM games WHERE result = 'win'")
            wins = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(*) FROM games WHERE result = 'loss'")
            losses = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(*) FROM games WHERE result = 'draw'")
            draws = cursor.fetchone()[0] or 0

            cursor.execute("SELECT AVG(accuracy_percent), AVG(avg_think_time_s), SUM(blunders_count) FROM games WHERE result != 'ongoing'")
            row = cursor.fetchone()
            avg_acc = row[0] or 0.0
            avg_tt = row[1] or 0.0
            tot_blunders = row[2] or 0

            winrate = (wins / total_games) * 100.0 if total_games > 0 else 0.0

            return {
                "total_games": total_games,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "winrate_pct": round(winrate, 1),
                "avg_accuracy": round(avg_acc, 1),
                "avg_think_time": round(avg_tt, 2),
                "total_blunders": tot_blunders,
            }
