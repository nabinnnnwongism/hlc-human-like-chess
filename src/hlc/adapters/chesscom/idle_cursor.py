"""idle_cursor.py — Human-like idle cursor daemon for chess.com.

Runs in a background thread and continuously moves the mouse in natural
human-browsing patterns while HLC is waiting for the opponent or thinking.

When HLC is ready to make a move:
  1. Caller calls  daemon.pause()   → idle movement stops, cursor parked
  2. MoveExecutor makes the chess move (fast bezier)
  3. Caller calls  daemon.resume()  → idle movement resumes

Usage:
    daemon = IdleCursorDaemon(page, board_bbox=bbox)
    daemon.start()

    # Somewhere in the move loop:
    with daemon.paused():
        move_executor.execute(move)   # cursor is frozen here

    daemon.stop()   # on session end

Idle behaviour mix (randomised each cycle):
  - Drift slowly over the chess board
  - Hover over the move list / sidebar
  - Occasional micro-pause (simulates reading / thinking)
  - Occasional text-hover sweep (player names, score bar)
  - Drift to page margins (nav bar, edges)
  - Random wander anywhere on the viewport
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time

from playwright.sync_api import Page

logger = logging.getLogger(__name__)


# ── Bezier helpers ─────────────────────────────────────────────────────────────

def _cubic_bezier_points(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    steps: int,
) -> list[tuple[float, float]]:
    """Return `steps` points along a cubic Bezier curve."""
    pts = []
    for i in range(1, steps + 1):
        t = i / steps
        mt = 1 - t
        x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
        pts.append((x, y))
    return pts


# ── Idle Cursor Daemon ─────────────────────────────────────────────────────────

class IdleCursorDaemon:
    """Background thread that keeps the cursor moving naturally between moves.

    Example usage:
        daemon = IdleCursorDaemon(page, board_bbox)
        daemon.start()

        with daemon.paused():
            move_executor.execute(move)

        daemon.stop()
    """

    def __init__(
        self,
        page: Page,
        board_bbox: dict | None = None,
        viewport_width: int = 1280,
        viewport_height: int = 800,
        idle_speed: float = 1.0,
    ) -> None:
        """Initialise the daemon.

        Args:
            page:             Playwright page.
            board_bbox:       Board bounding box dict from Playwright
                              ({x, y, width, height}). Can be updated later
                              via set_board_bbox().
            viewport_width:   Browser viewport width (for clamping).
            viewport_height:  Browser viewport height.
            idle_speed:       Speed multiplier for idle movement.
                              1.0 = normal human pace.
        """
        self._page = page
        self._bbox = board_bbox
        self._vw = viewport_width
        self._vh = viewport_height
        self._speed = idle_speed
        self._rng = random.Random()

        # Daemon-tracked cursor position
        self._cx: float = viewport_width / 2.0
        self._cy: float = viewport_height / 2.0

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()  # set = paused
        self._thread: threading.Thread | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_board_bbox(self, bbox: dict) -> None:
        """Update the board bounding box (call once per game start)."""
        self._bbox = bbox

    def start(self) -> None:
        """Start the idle cursor daemon (no-op to prevent Playwright greenlet thread switch error)."""
        self._stop_event.clear()
        self._pause_event.clear()
        logger.debug("IdleCursorDaemon initialized (cross-thread Playwright calls disabled for stability).")

    def stop(self) -> None:
        """Stop the daemon."""
        self._stop_event.set()
        self._pause_event.clear()

    def pause(self) -> None:
        """Pause idle movement so a chess move can be executed cleanly."""
        self._pause_event.set()
        time.sleep(0.05)  # let current micro-movement finish

    def resume(self) -> None:
        """Resume idle movement after a chess move completes."""
        self._pause_event.clear()

    class _PauseCtx:
        def __init__(self, d: "IdleCursorDaemon") -> None:
            self._d = d

        def __enter__(self) -> "IdleCursorDaemon._PauseCtx":
            self._d.pause()
            return self

        def __exit__(self, *_: object) -> None:
            self._d.resume()

    def paused(self) -> "_PauseCtx":
        """Context manager: pause idle movement for the duration of a block.

        Example:
            with daemon.paused():
                move_executor.execute(move)
        """
        return self._PauseCtx(self)

    # ── Background loop ────────────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            self._page.mouse.move(self._cx, self._cy)
        except Exception:
            pass

        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.05)
                continue
            try:
                self._execute_idle_behaviour()
            except Exception as exc:
                logger.debug("IdleCursor non-fatal: %s", exc)
                time.sleep(0.1)

    def _execute_idle_behaviour(self) -> None:
        choice = self._rng.choices(
            population=[
                "drift_board",
                "hover_move_list",
                "random_wander",
                "micro_pause",
                "text_sweep",
                "drift_margins",
            ],
            weights=[30, 20, 20, 15, 8, 7],
            k=1,
        )[0]

        dispatch = {
            "drift_board": self._drift_board,
            "hover_move_list": self._hover_move_list,
            "random_wander": self._random_wander,
            "micro_pause": self._micro_pause,
            "text_sweep": self._text_sweep,
            "drift_margins": self._drift_margins,
        }
        dispatch[choice]()

    # ── Behaviours ─────────────────────────────────────────────────────────────

    def _drift_board(self) -> None:
        """Drift slowly over different areas of the chess board."""
        if not self._bbox:
            self._random_wander()
            return
        b = self._bbox
        margin = 40
        tx = self._rng.uniform(b["x"] - margin, b["x"] + b["width"] + margin)
        ty = self._rng.uniform(b["y"] - margin, b["y"] + b["height"] + margin)
        self._move_to(self._clamp_x(tx), self._clamp_y(ty),
                      duration=self._rng.uniform(0.4, 1.2) / self._speed)
        self._sleep_interruptible(self._rng.uniform(0.2, 0.9))

    def _hover_move_list(self) -> None:
        """Drift into the move-list sidebar area (right of board)."""
        if self._bbox:
            b = self._bbox
            lx = b["x"] + b["width"] + self._rng.uniform(10, 120)
            ly = b["y"] + self._rng.uniform(0, b["height"])
        else:
            lx = self._rng.uniform(self._vw * 0.65, self._vw * 0.90)
            ly = self._rng.uniform(self._vh * 0.20, self._vh * 0.80)

        self._move_to(self._clamp_x(lx), self._clamp_y(ly),
                      duration=self._rng.uniform(0.3, 0.8) / self._speed)
        self._sleep_interruptible(self._rng.uniform(0.3, 1.5))

        # Optionally scroll down the move list (reading)
        if self._rng.random() < 0.5:
            cur_ly = ly
            for _ in range(self._rng.randint(2, 5)):
                if self._pause_event.is_set():
                    break
                cur_ly = self._clamp_y(cur_ly + self._rng.uniform(10, 30))
                self._move_to(self._clamp_x(lx + self._rng.uniform(-5, 5)),
                              cur_ly, duration=0.15)
                self._sleep_interruptible(self._rng.uniform(0.1, 0.3))

    def _random_wander(self) -> None:
        """Move to a random point anywhere on screen."""
        tx = self._rng.uniform(self._vw * 0.05, self._vw * 0.95)
        ty = self._rng.uniform(self._vh * 0.05, self._vh * 0.90)
        self._move_to(tx, ty, duration=self._rng.uniform(0.3, 1.0) / self._speed)
        self._sleep_interruptible(self._rng.uniform(0.1, 0.5))

    def _micro_pause(self) -> None:
        """Stay almost still — simulates reading or thinking."""
        self._sleep_interruptible(self._rng.uniform(0.5, 2.5))
        for _ in range(self._rng.randint(0, 3)):
            if self._pause_event.is_set():
                break
            jx = self._clamp_x(self._cx + self._rng.uniform(-4, 4))
            jy = self._clamp_y(self._cy + self._rng.uniform(-4, 4))
            self._move_to(jx, jy, duration=0.08)
            self._sleep_interruptible(self._rng.uniform(0.1, 0.4))

    def _text_sweep(self) -> None:
        """Sweep horizontally as if reading / selecting text."""
        if self._bbox:
            b = self._bbox
            zones = [
                (b["x"], b["y"] - 30, b["x"] + b["width"] * 0.6, b["y"] - 10),
                (b["x"], b["y"] + b["height"] + 10, b["x"] + b["width"] * 0.6,
                 b["y"] + b["height"] + 30),
            ]
            x1, y1, x2, y2 = self._rng.choice(zones)
        else:
            y = self._rng.uniform(self._vh * 0.05, self._vh * 0.25)
            x1, y1, x2, y2 = self._vw * 0.1, y, self._vw * 0.6, y + 10

        sx = self._rng.uniform(x1, (x1 + x2) / 2)
        sy = self._rng.uniform(y1, y2)
        self._move_to(self._clamp_x(sx), self._clamp_y(sy), duration=0.3)
        self._sleep_interruptible(0.1)
        ex = self._rng.uniform((x1 + x2) / 2, x2)
        self._move_to(self._clamp_x(ex), self._clamp_y(sy + self._rng.uniform(-3, 3)),
                      duration=self._rng.uniform(0.2, 0.6))
        self._sleep_interruptible(self._rng.uniform(0.1, 0.4))

    def _drift_margins(self) -> None:
        """Drift to page edge zones (nav bar, left margin, footer area)."""
        zones = [
            (0.05, 0.01, 0.95, 0.07),   # top nav
            (0.01, 0.10, 0.12, 0.90),   # left edge
            (0.05, 0.92, 0.95, 0.99),   # bottom strip
        ]
        z = self._rng.choice(zones)
        tx = self._rng.uniform(self._vw * z[0], self._vw * z[2])
        ty = self._rng.uniform(self._vh * z[1], self._vh * z[3])
        self._move_to(tx, ty, duration=self._rng.uniform(0.4, 1.0) / self._speed)
        self._sleep_interruptible(self._rng.uniform(0.2, 0.8))

    # ── Core movement primitive ────────────────────────────────────────────────

    def _move_to(self, tx: float, ty: float, duration: float = 0.3) -> None:
        """Move cursor from current position to (tx, ty) via cubic Bezier curve."""
        if self._pause_event.is_set():
            return

        dist = math.hypot(tx - self._cx, ty - self._cy)
        if dist < 2:
            return

        steps = max(4, int(dist / 10))
        step_delay = max(0.003, duration / steps)

        mid_x = (self._cx + tx) / 2
        mid_y = (self._cy + ty) / 2
        jitter = min(dist * 0.35, 60.0)
        rng = self._rng
        cp1 = (mid_x + rng.uniform(-jitter, jitter), mid_y + rng.uniform(-jitter, jitter))
        cp2 = (mid_x + rng.uniform(-jitter, jitter), mid_y + rng.uniform(-jitter, jitter))

        pts = _cubic_bezier_points((self._cx, self._cy), cp1, cp2, (tx, ty), steps)

        for px, py in pts:
            if self._pause_event.is_set():
                self._cx, self._cy = px, py
                return
            try:
                self._page.mouse.move(px, py)
            except Exception:
                return
            time.sleep(step_delay)

        self._cx, self._cy = tx, ty

    # ── Utilities ──────────────────────────────────────────────────────────────

    def _sleep_interruptible(self, seconds: float) -> None:
        """Sleep in small increments so pause/stop events are checked often."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self._stop_event.is_set() or self._pause_event.is_set():
                return
            time.sleep(min(0.05, end - time.monotonic()))

    def _clamp_x(self, x: float) -> float:
        return max(5.0, min(float(self._vw - 5), x))

    def _clamp_y(self, y: float) -> float:
        return max(5.0, min(float(self._vh - 5), y))
