"""digital_clock.py — Accurate internal chess clock simulator and DOM synchronizer.

Maintains real-time digital clocks for both sides.
Supports:
- Auto-detection of time control from DOM/URL/clocks (Bullet, Blitz, Rapid, Classical)
- Autonomous clock countdown in matches without on-screen clocks (e.g. vs Computer bots, Coach)
- Real-time DOM clock synchronization for live online games
- Formatting for GUI telemetry and terminal logs
"""

from __future__ import annotations

import logging
import re
import time
from typing import Literal

import chess

logger = logging.getLogger(__name__)

# Standard presets in seconds: (initial_seconds, increment_seconds, display_name)
PRESET_CONTROLS: dict[str, tuple[float, float, str]] = {
    "bullet_1m": (60.0, 0.0, "Bullet (1 min)"),
    "bullet_2m": (120.0, 1.0, "Bullet (2|1)"),
    "blitz_3m": (180.0, 0.0, "Blitz (3 min)"),
    "blitz_3_2": (180.0, 2.0, "Blitz (3|2)"),
    "blitz_5m": (300.0, 0.0, "Blitz (5 min)"),
    "blitz_5_3": (300.0, 3.0, "Blitz (5|3)"),
    "rapid_10m": (600.0, 0.0, "Rapid (10 min)"),
    "rapid_15_10": (900.0, 10.0, "Rapid (15|10)"),
    "classical_30m": (1800.0, 0.0, "Classical (30 min)"),
}


class DigitalClock:
    """Manages digital clocks for both White and Black.

    Ensures that Maia-3 and the Heuristic Timing Engine always receive accurate,
    dynamically decaying clocks even when playing against chess.com computer bots
    that do not display on-screen timers.
    """

    def __init__(
        self,
        initial_seconds: float = 600.0,
        increment_s: float = 0.0,
        time_control_name: str = "Rapid (10 min)",
    ) -> None:
        self.white_s: float = max(1.0, initial_seconds)
        self.black_s: float = max(1.0, initial_seconds)
        self.initial_seconds: float = initial_seconds
        self.increment_s: float = increment_s
        self.time_control_name: str = time_control_name
        self.has_dom_clocks: bool = False
        self._last_tick_time: float = time.perf_counter()
        self._active_color: chess.Color | None = None

    @classmethod
    def create(
        cls,
        time_control_hint: str | None = None,
        dom_white_s: float = 0.0,
        dom_black_s: float = 0.0,
        default_seconds: float = 600.0,
    ) -> "DigitalClock":
        """Factory: create a DigitalClock from DOM readings or string hints.

        Priority order:
        1. Hint string (keyword or time format)
        2. DOM clock values (both must be valid)
        3. Default fallback
        """
        # 1. String hint (highest priority -- explicit user/DOM label)
        if time_control_hint:
            hint_lower = time_control_hint.lower().strip()

            # Check for exact preset key match first
            for key, (init_s, inc_s, name) in PRESET_CONTROLS.items():
                if key in hint_lower or name.lower() in hint_lower:
                    return cls(initial_seconds=init_s, increment_s=inc_s, time_control_name=name)

            # Untimed / Casual matches (vs Computer bots without clock)
            if "no timer" in hint_lower or "untimed" in hint_lower:
                return cls(initial_seconds=600.0, increment_s=0.0, time_control_name="Casual (Untimed)")

            # Keyword-based classification
            if "bullet" in hint_lower:
                # Check for 2+1 or "2 min" variant
                if "2" in hint_lower:
                    return cls(120.0, 1.0, "Bullet (2|1)")
                return cls(60.0, 0.0, "Bullet (1 min)")
            if "blitz" in hint_lower:
                if "5" in hint_lower:
                    return cls(300.0, 0.0, "Blitz (5 min)")
                return cls(180.0, 0.0, "Blitz (3 min)")
            if "rapid" in hint_lower:
                if "15" in hint_lower:
                    return cls(900.0, 10.0, "Rapid (15|10)")
                return cls(600.0, 0.0, "Rapid (10 min)")
            if "classical" in hint_lower or "daily" in hint_lower:
                return cls(1800.0, 0.0, "Classical (30 min)")

            # Time-format hints: "1m", "2m", "3m", "5m", "10m", "15m", "30m",
            # or "1 min", "1+0", "3|2", etc.
            import re as _re
            # Match patterns like "3+2", "3|2", "10+0", "15|10"
            m = _re.search(r'(\d+)\s*[+|]\s*(\d+)', hint_lower)
            if m:
                mins = int(m.group(1))
                inc  = int(m.group(2))
                secs = mins * 60
                tc_name = cls.classify_time_control(secs)
                return cls(float(secs), float(inc), tc_name)
            # Match patterns like "3m", "10m", "1 min", "10 min"
            m = _re.search(r'(\d+)\s*(?:m(?:in)?|minutes?)', hint_lower)
            if m:
                mins = int(m.group(1))
                secs = mins * 60
                tc_name = cls.classify_time_control(secs)
                return cls(float(secs), 0.0, tc_name)
            # Plain numbers like "60", "180", "600" (seconds)
            m = _re.match(r'^(\d+)$', hint_lower.strip())
            if m:
                secs = int(m.group(1))
                # If < 60, assume it's minutes
                if secs < 60:
                    secs *= 60
                tc_name = cls.classify_time_control(float(secs))
                return cls(float(secs), 0.0, tc_name)

        # 2. If DOM clocks have valid positive values (> 5s for at least one)
        dom_max = max(dom_white_s, dom_black_s)
        if dom_max > 5.0:
            # If both clocks are non-zero use the max, else use whichever is non-zero
            init_s = dom_max
            tc_name = cls.classify_time_control(init_s)
            instance = cls(initial_seconds=init_s, time_control_name=tc_name)
            if dom_white_s > 0.0 and dom_black_s > 0.0:
                instance.sync_from_dom(dom_white_s, dom_black_s)
            else:
                # Only one side has a clock value yet -- use it for both
                valid_s = dom_white_s if dom_white_s > 0.0 else dom_black_s
                instance.sync_from_dom(valid_s, valid_s)
            return instance

        # 3. Default fallback
        tc_name = cls.classify_time_control(default_seconds)
        return cls(initial_seconds=default_seconds, time_control_name=tc_name)

    @staticmethod
    def classify_time_control(seconds: float) -> str:
        """Classify duration into a human-readable time control string.

        Boundaries (inclusive upper bound):
          <= 75s   -> Bullet 1 min  (covers exactly 60s and up to 1:15)
          <= 150s  -> Bullet 2 min  (covers 1:16 to 2:30)
          <= 210s  -> Blitz  3 min  (covers 2:31 to 3:30)
          <= 360s  -> Blitz  5 min  (covers 3:31 to 6:00)
          <= 660s  -> Rapid  10 min (covers 6:01 to 11:00)
          <= 960s  -> Rapid  15 min (covers 11:01 to 16:00)
          else     -> Classical
        """
        if seconds <= 75.0:
            return "Bullet (1 min)"
        elif seconds <= 150.0:
            return "Bullet (2 min)"
        elif seconds <= 210.0:
            return "Blitz (3 min)"
        elif seconds <= 360.0:
            return "Blitz (5 min)"
        elif seconds <= 660.0:
            return "Rapid (10 min)"
        elif seconds <= 960.0:
            return "Rapid (15 min)"
        else:
            return f"Classical ({int(seconds // 60)} min)"

    def start_turn(self, turn: chess.Color) -> None:
        """Record the start of a player's turn."""
        self._active_color = turn
        self._last_tick_time = time.perf_counter()

    def end_turn(self, turn: chess.Color, elapsed_s: float | None = None) -> None:
        """Deduct elapsed think time from the player who just moved."""
        now = time.perf_counter()
        if elapsed_s is None:
            elapsed_s = max(0.0, now - self._last_tick_time)

        if turn == chess.WHITE:
            self.white_s = max(1.0, self.white_s - elapsed_s + self.increment_s)
        else:
            self.black_s = max(1.0, self.black_s - elapsed_s + self.increment_s)

        self._last_tick_time = now

    def sync_from_dom(self, dom_white_s: float, dom_black_s: float) -> None:
        """Synchronize with DOM clocks when valid numbers are read."""
        # Never sync DOM clocks in untimed/casual matches (Chess.com displays an elapsed stopwatch)
        tc_lower = self.time_control_name.lower()
        if "untimed" in tc_lower or "no timer" in tc_lower:
            return

        if dom_white_s > 0.0 and dom_black_s > 0.0:
            # Monotonicity check: clocks cannot count upwards by more than increment buffer
            if self.has_dom_clocks:
                buf = self.increment_s + 3.0
                if (dom_white_s > self.white_s + buf) or (dom_black_s > self.black_s + buf):
                    logger.debug(
                        "Rejected non-monotonic DOM clock update (stopwatch detected): W=%.1f B=%.1f",
                        dom_white_s,
                        dom_black_s,
                    )
                    return

            self.white_s = max(1.0, dom_white_s)
            self.black_s = max(1.0, dom_black_s)
            self.has_dom_clocks = True

    def get_clocks_for(self, player_color: chess.Color) -> tuple[float, float]:
        """Return (clock_self, clock_opp) in seconds."""
        if player_color == chess.WHITE:
            return self.white_s, self.black_s
        else:
            return self.black_s, self.white_s

    @staticmethod
    def format_clock(seconds: float) -> str:
        """Format seconds into M:SS or MM:SS."""
        s = max(0, int(seconds))
        mins = s // 60
        secs = s % 60
        return f"{mins:02d}:{secs:02d}"

    def status_str(self) -> str:
        """Return a formatted status string for console and GUI logging."""
        w_str = self.format_clock(self.white_s)
        b_str = self.format_clock(self.black_s)
        mode_tag = "DOM" if self.has_dom_clocks else "Digital"
        return f"W: {w_str} | B: {b_str} [{self.time_control_name} | {mode_tag}]"
