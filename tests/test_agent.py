"""test_agent.py — Unit tests for AgentMemory and MetaController."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
import chess
import pytest

from hlc.agent.memory import AgentMemory
from hlc.agent.meta_controller import MetaController


@pytest.fixture
def temp_memory():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_agent.db"
        memory = AgentMemory(db_path=db_path)
        yield memory
        # Explicitly close all SQLite connections before Windows tries to
        # delete the temp directory — otherwise WinError 32 (file in use).
        memory.close()


class TestAgentMemory:
    def test_start_and_finish_game(self, temp_memory: AgentMemory) -> None:
        game_id = temp_memory.start_game(
            session_id="test_sess_1",
            color="white",
            self_elo=1800,
            opp_elo=1780,
            style_used="native_pragmatic",
            fatigue_level=0.05,
        )
        assert game_id > 0

        # Record moves
        temp_memory.record_move(
            game_id=game_id,
            ply=1,
            fen_before=chess.STARTING_FEN,
            move_uci="e2e4",
            move_san="e4",
            think_time_s=1.85,
            policy_entropy=1.2,
            top_move_prob=0.45,
            was_top_move=True,
            clock_remaining_s=178.0,
        )

        temp_memory.finish_game(
            game_id=game_id,
            result="win",
            accuracy_percent=84.5,
            notes="Solid opening win",
        )

        recent = temp_memory.get_recent_games(limit=5)
        assert len(recent) == 1
        assert recent[0].result == "win"
        assert recent[0].accuracy_percent == 84.5
        assert recent[0].moves_count == 1
        assert recent[0].self_elo == 1800

    def test_summary_stats(self, temp_memory: AgentMemory) -> None:
        g1 = temp_memory.start_game("s1", "white", 1800, 1800, "native")
        temp_memory.finish_game(g1, "win", accuracy_percent=80.0)

        g2 = temp_memory.start_game("s1", "black", 1800, 1820, "native")
        temp_memory.finish_game(g2, "loss", accuracy_percent=70.0)

        stats = temp_memory.get_stats_summary()
        assert stats["total_games"] == 2
        assert stats["wins"] == 1
        assert stats["losses"] == 1
        assert stats["winrate_pct"] == 50.0
        assert stats["avg_accuracy"] == 75.0


class TestMetaController:
    def test_native_style_generation(self, temp_memory: AgentMemory) -> None:
        controller = MetaController(memory=temp_memory, base_elo=1800)
        style = controller.get_native_style_vector(elo=1800, fatigue=0.0)
        assert 0.0 <= style.aggression <= 1.0
        assert 0.0 <= style.initiative <= 1.0
        assert 0.0 <= style.prophylaxis <= 1.0
        assert style.bias_strength > 0.5

    def test_fatigue_progression(self, temp_memory: AgentMemory) -> None:
        controller = MetaController(memory=temp_memory, base_elo=1800)
        assert controller.session.fatigue == 0.0

        f1 = controller.update_session_fatigue()
        assert f1 > 0.0

        f2 = controller.update_session_fatigue()
        assert f2 > f1

    def test_target_elo_calibration(self, temp_memory: AgentMemory) -> None:
        controller = MetaController(memory=temp_memory, base_elo=1800)
        target = controller.calibrate_target_elo(detected_self_elo=1820, opp_elo=1800)
        assert 1700 <= target <= 1900

    def test_blunder_plausibility(self, temp_memory: AgentMemory) -> None:
        controller = MetaController(memory=temp_memory, base_elo=1800)
        board = chess.Board()

        # Plausible blunder: time scramble (<10s) and high entropy
        p_high = controller.evaluate_blunder_plausibility(board, entropy=2.2, clock_s=8.0, is_check=True)
        # Implausible blunder: abundant clock (120s) and low entropy
        p_low = controller.evaluate_blunder_plausibility(board, entropy=0.5, clock_s=120.0, is_check=False)

        assert p_high > p_low
        assert p_high >= 0.7
        assert p_low <= 0.2

    def test_unrated_account_calibration(self, temp_memory: AgentMemory) -> None:
        controller = MetaController(memory=temp_memory, base_elo=1500)
        # 1. Brand new unrated account vs unknown/unrated opponent -> defaults around starter baseline
        starter_target = controller.calibrate_target_elo(detected_self_elo=0, opp_elo=0)
        assert 1100 <= starter_target <= 1200

        # 2. Brand new unrated account vs 800 player/bot -> adapts down to opponent tier
        bot_target = controller.calibrate_target_elo(detected_self_elo=0, opp_elo=800)
        assert 750 <= bot_target <= 850

        # 3. Brand new unrated account vs 1400 player -> adapts up to 1400 tier
        higher_target = controller.calibrate_target_elo(detected_self_elo=0, opp_elo=1400)
        assert 1350 <= higher_target <= 1450
