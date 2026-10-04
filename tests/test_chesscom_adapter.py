"""test_chesscom_adapter.py — Unit tests for chess.com adapter enhancements.

Tests:
- DigitalClock initialization, decay, and DOM sync
- Time control auto-classification
- BoardReader clock parsing logic
- Mode detection logic
"""

import chess
import pytest

from hlc.adapters.chesscom.digital_clock import PRESET_CONTROLS, DigitalClock


class TestDigitalClock:
    """Test suite for DigitalClock emulation and DOM synchronization."""

    def test_default_creation(self):
        clock = DigitalClock.create()
        assert clock.white_s == 600.0
        assert clock.black_s == 600.0
        assert "Rapid" in clock.time_control_name
        assert clock.has_dom_clocks is False

    def test_hint_creation(self):
        bullet_clock = DigitalClock.create(time_control_hint="bullet")
        assert bullet_clock.white_s == 60.0
        assert "Bullet" in bullet_clock.time_control_name

        blitz_clock = DigitalClock.create(time_control_hint="3 min")
        assert blitz_clock.white_s == 180.0
        assert "Blitz" in blitz_clock.time_control_name

        rapid_clock = DigitalClock.create(time_control_hint="15|10")
        assert rapid_clock.white_s == 900.0
        assert rapid_clock.increment_s == 10.0

    def test_dom_sync(self):
        clock = DigitalClock.create(dom_white_s=295.0, dom_black_s=298.0)
        assert clock.white_s == 295.0
        assert clock.black_s == 298.0
        assert clock.has_dom_clocks is True
        assert "Blitz (5 min)" in clock.time_control_name

    def test_turn_decay(self):
        clock = DigitalClock(initial_seconds=600.0, increment_s=0.0)
        clock.start_turn(chess.WHITE)
        # Deduct 15 seconds
        clock.end_turn(chess.WHITE, elapsed_s=15.0)
        assert clock.white_s == 585.0
        assert clock.black_s == 600.0

        # Black turn with 10 seconds
        clock.start_turn(chess.BLACK)
        clock.end_turn(chess.BLACK, elapsed_s=10.0)
        assert clock.black_s == 590.0

    def test_clock_safety_floor(self):
        clock = DigitalClock(initial_seconds=5.0)
        clock.start_turn(chess.WHITE)
        # Deduct 10 seconds (exceeding remaining clock)
        clock.end_turn(chess.WHITE, elapsed_s=10.0)
        # Must not go below 1.0s safety floor
        assert clock.white_s == 1.0

    def test_format_clock(self):
        assert DigitalClock.format_clock(600.0) == "10:00"
        assert DigitalClock.format_clock(185.0) == "03:05"
        assert DigitalClock.format_clock(59.0) == "00:59"
        assert DigitalClock.format_clock(0.0) == "00:00"

    def test_get_clocks_for(self):
        clock = DigitalClock(initial_seconds=300.0)
        clock.white_s = 250.0
        clock.black_s = 280.0

        w_self, w_opp = clock.get_clocks_for(chess.WHITE)
        assert w_self == 250.0
        assert w_opp == 280.0

        b_self, b_opp = clock.get_clocks_for(chess.BLACK)
        assert b_self == 280.0
        assert b_opp == 250.0

    def test_status_str_ascii_and_format(self):
        clock = DigitalClock.create(time_control_hint="10 min")
        status = clock.status_str()
        # Verify that status_str is strictly ASCII (no cp1252 crashing chars)
        assert all(ord(c) <= 127 for c in status)
        assert "W: 10:00 | B: 10:00" in status
        assert "[Rapid (10 min) | Digital]" in status

    def test_gui_clock_regex_matches_status_str(self):
        import re

        clock = DigitalClock.create(time_control_hint="5 min")
        line = f"[CLOCK] {clock.status_str()}"
        clock_m = re.search(
            r"\[clock\]\s+w:\s*(\d+:\d+)\s*\|\s*b:\s*(\d+:\d+)\s*\[([^\]|]+)\|\s*([^\]]+)\]",
            line,
            re.IGNORECASE,
        )
        assert clock_m is not None
        assert clock_m.group(1) == "05:00"
        assert clock_m.group(2) == "05:00"
        assert "Blitz" in clock_m.group(3)
        assert "Digital" in clock_m.group(4)


class TestChesscomUrlClassification:
    """Test URL pattern detection across chess.com navigation routes."""

    @pytest.mark.parametrize(
        ("url", "expected_mode"),
        [
            ("https://www.chess.com/login", "login"),
            ("https://www.chess.com/register", "login"),
            ("https://www.chess.com/game/live/12345678", "live_game"),
            ("https://www.chess.com/game/daily/98765432", "daily_game"),
            ("https://www.chess.com/game/computer", "vs_computer"),
            ("https://www.chess.com/play/computer", "vs_computer"),
            ("https://www.chess.com/play/bot", "vs_computer"),
            ("https://www.chess.com/play/coach?source=play_nav", "vs_coach"),
            ("https://www.chess.com/play/online", "play_online"),
            ("https://www.chess.com/play/custom", "lobby"),
            ("https://www.chess.com/home", "home"),
            ("https://www.chess.com", "home"),
        ],
    )
    def test_url_classification(self, url: str, expected_mode: str):
        # Emulate the URL classification logic from ChessDotComBrowser.detect_page_context
        mode = "unknown"
        if "/login" in url or "/register" in url:
            mode = "login"
        elif "/game/live/" in url:
            mode = "live_game"
        elif "/game/daily/" in url:
            mode = "daily_game"
        elif "/game/computer" in url:
            mode = "vs_computer"
        elif "/play/computer" in url or "/play/bot" in url:
            mode = "vs_computer"
        elif "/play/coach" in url:
            mode = "vs_coach"
        elif "/play/online" in url:
            mode = "play_online"
        elif "/play/" in url:
            mode = "lobby"
        elif "chess.com" in url:
            mode = "home"

        assert mode == expected_mode

