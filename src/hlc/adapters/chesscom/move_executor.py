"""move_executor.py — Translate UCI moves → human-like mouse clicks on chess.com.

Strategy:
  1. Find the board container element and get its bounding box.
  2. Map each square (a1–h8) to a pixel coordinate using the bounding box.
  3. Move the mouse along a Bezier curve from the source square to the destination.
  4. Click source, then click destination (click-click style, not drag).

Chess.com supports both click-click and click-drag. Click-click is more reliable.

The board can be flipped (when HLC plays Black). We account for this.
"""

from __future__ import annotations

import logging
import math
import random
import time
from typing import Literal

import chess
from playwright.sync_api import Page

logger = logging.getLogger(__name__)

# Board CSS selectors
_BOARD_SEL = "wc-chess-board, chess-board"


class MoveExecutor:
    """Executes a chess move on chess.com by simulating human-like mouse clicks.

    Usage:
        executor = MoveExecutor(page, player_color=chess.WHITE)
        executor.execute(chess.Move.from_uci("e2e4"))
    """

    def __init__(
        self,
        page: Page,
        player_color: chess.Color = chess.WHITE,
        jitter_px: int = 6,
    ) -> None:
        """Initialise the MoveExecutor.

        Args:
            page:         The active Playwright Page.
            player_color: The color HLC is playing. Determines board flip detection.
            jitter_px:    Max random pixel offset added to click coordinates (humanises clicks).
        """
        self._page = page
        self._player_color = player_color
        self._jitter_px = jitter_px
        self._rng = random.Random()

    # ── Public API ─────────────────────────────────────────────────────────────
    def execute(self, move: chess.Move) -> None:
        """Click the source square then the destination square for `move`.

        Args:
            move: A python-chess Move (e.g. chess.Move.from_uci("e2e4")).
        """
        board_el = self._page.locator(_BOARD_SEL).first
        bbox = board_el.bounding_box()
        if bbox is None:
            raise RuntimeError("Could not locate chess board bounding box on page.")

        classes = board_el.get_attribute("class") or ""
        is_flipped = ("flipped" in classes) or (self._player_color == chess.BLACK)

        src_x, src_y = self._square_to_xy(move.from_square, bbox, is_flipped)
        dst_x, dst_y = self._square_to_xy(move.to_square,   bbox, is_flipped)

        logger.debug(
            "MoveExecutor: %s  src=(%.0f, %.0f)  dst=(%.0f, %.0f)",
            move.uci(), src_x, src_y, dst_x, dst_y,
        )

        # Click source square
        self._human_click(src_x, src_y)
        time.sleep(self._rng.uniform(0.04, 0.09))

        # Click destination square
        self._human_click(dst_x, dst_y)

        # Handle promotion — default to Queen
        if move.promotion:
            self._handle_promotion(move.promotion, dst_x, dst_y, bbox, is_flipped)

    # ── Private ─────────────────────────────────────────────────────────────────
    def _get_board_bbox(self) -> dict:
        """Return the bounding box of the chess board element."""
        board_el = self._page.locator(_BOARD_SEL).first
        bbox = board_el.bounding_box()
        if bbox is None:
            raise RuntimeError("Could not locate chess board bounding box on page.")
        return bbox

    def _square_to_xy(
        self,
        square: chess.Square,
        bbox: dict,
        is_flipped: bool,
    ) -> tuple[float, float]:
        """Convert a python-chess square to screen pixel coordinates.

        The board is divided into an 8×8 grid inside the bbox.
        When flipped (Black plays at bottom), file and rank axes are inverted.

        Args:
            square:    chess.A1 … chess.H8 (python-chess square index).
            bbox:      Board bounding box {"x", "y", "width", "height"}.
            is_flipped: True when the board is flipped (HLC plays Black).

        Returns:
            (x, y) pixel coordinates including random jitter.
        """
        file_idx = chess.square_file(square)  # 0=a … 7=h
        rank_idx = chess.square_rank(square)  # 0=1 … 7=8

        if is_flipped:
            # Board flipped (Black perspective):
            # File a (idx 0) is on right side -> col = 7 - file_idx
            # Rank 8 (idx 7) is at bottom, Rank 1 (idx 0) is at top -> row = rank_idx
            col = 7 - file_idx
            row = rank_idx
        else:
            # Board normal (White perspective):
            # File a (idx 0) is on left side -> col = file_idx
            # Rank 8 (idx 7) is at top -> row = 7 - rank_idx
            col = file_idx
            row = 7 - rank_idx

        sq_w = bbox["width"]  / 8.0
        sq_h = bbox["height"] / 8.0

        # Centre of the square
        cx = bbox["x"] + col * sq_w + sq_w / 2.0
        cy = bbox["y"] + row * sq_h + sq_h / 2.0

        # Add small random jitter so clicks don't land perfectly centre every time
        jitter = self._jitter_px
        cx += self._rng.uniform(-jitter, jitter)
        cy += self._rng.uniform(-jitter, jitter)

        return cx, cy

    def _human_click(self, x: float, y: float) -> None:
        """Move the mouse smoothly then click at (x, y)."""
        page = self._page
        mouse = page.mouse

        cur_x, cur_y = x + self._rng.uniform(-30, 30), y + self._rng.uniform(-30, 30)
        cp_x = (cur_x + x) / 2 + self._rng.uniform(-15, 15)
        cp_y = (cur_y + y) / 2 + self._rng.uniform(-15, 15)

        steps = self._rng.randint(3, 5)
        for i in range(1, steps + 1):
            t = i / steps
            bx = (1 - t) ** 2 * cur_x + 2 * (1 - t) * t * cp_x + t ** 2 * x
            by = (1 - t) ** 2 * cur_y + 2 * (1 - t) * t * cp_y + t ** 2 * y
            mouse.move(bx, by)
            time.sleep(0.003)

        mouse.click(x, y)

    def _handle_promotion(
        self,
        piece_type: int,
        dst_x: float,
        dst_y: float,
        bbox: dict,
        is_flipped: bool,
    ) -> None:
        """Click the promotion popup. Chess.com shows a vertical list of pieces.

        The promotion dialog appears above/below the destination square.
        Order (top to bottom for White promoting):
          Queen, Knight, Rook, Bishop.
        For Black: order is reversed (bottom to top).
        """
        time.sleep(0.5)  # wait for dialog to appear

        # Vertical offset per piece option (~half square size)
        sq_h = bbox["height"] / 8.0
        piece_order = [chess.QUEEN, chess.KNIGHT, chess.ROOK, chess.BISHOP]

        if piece_type not in piece_order:
            piece_type = chess.QUEEN  # default to Queen

        idx = piece_order.index(piece_type)

        if not is_flipped:
            # White promoting: dialog drops down from top
            click_y = dst_y + idx * sq_h
        else:
            # Black promoting: dialog drops up from bottom
            click_y = dst_y - idx * sq_h

        self._human_click(dst_x, click_y)
        logger.info("Promotion: clicked piece idx=%d at y=%.0f", idx, click_y)
