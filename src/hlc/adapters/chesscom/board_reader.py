"""board_reader.py — Read chess.com board DOM → python-chess Board / FEN.

Chess.com board DOM structure (stable as of 2024/2025):
  <wc-chess-board>
    <div class="board">
      <!-- Squares (backgrounds) -->
      <div class="square square-11"></div>  ← a1
      ...
      <!-- Pieces -->
      <div class="piece wp square-12"></div>   ← white pawn on a2
      <div class="piece bn square-57"></div>   ← black knight on e7
    </div>
  </wc-chess-board>

Square encoding: square-XY where X = file (1=a … 8=h), Y = rank (1…8).
Piece encoding: two-letter class, e.g.:
  wp=white pawn, wn=white knight, wb=white bishop,
  wr=white rook, wq=white queen, wk=white king,
  bp, bn, bb, br, bq, bk  (same for black)
"""

from __future__ import annotations

import logging
import re
from typing import Literal

import chess
from playwright.sync_api import Page

logger = logging.getLogger(__name__)

# Regex that captures the piece class (e.g. "wp", "bk") and square class (e.g. "square-37")
_PIECE_RE = re.compile(r"\b([wb][pnbrqk])\b")
_SQ_RE    = re.compile(r"\bsquare-(\d{2})\b")

# Maps chess.com piece codes → python-chess piece types
_PIECE_TYPE: dict[str, int] = {
    "p": chess.PAWN,
    "n": chess.KNIGHT,
    "b": chess.BISHOP,
    "r": chess.ROOK,
    "q": chess.QUEEN,
    "k": chess.KING,
}


def _chesscom_sq_to_chess(sq_code: str) -> chess.Square | None:
    """Convert chess.com square code "XY" → python-chess Square integer.

    X = file (1=a … 8=h), Y = rank (1…8).
    Returns None if parsing fails.
    """
    if len(sq_code) != 2:
        return None
    try:
        file_idx = int(sq_code[0]) - 1   # 0-based: a=0 … h=7
        rank_idx = int(sq_code[1]) - 1   # 0-based: 1=0 … 8=7
        if not (0 <= file_idx <= 7 and 0 <= rank_idx <= 7):
            return None
        return chess.square(file_idx, rank_idx)
    except ValueError:
        return None


class BoardReader:
    """Reads the chess.com live board DOM and converts it to a python-chess Board.

    Usage:
        reader = BoardReader(page)
        board = reader.read_board(turn=chess.WHITE)
    """

    def __init__(self, page: Page) -> None:
        self._page = page

    # ── Public API ─────────────────────────────────────────────────────────────
    def read_board(self, turn: chess.Color = chess.WHITE) -> chess.Board:
        """Parse all piece divs from the DOM and return a chess.Board.

        Args:
            turn: Whose turn it is (chess.WHITE or chess.BLACK).
                  BoardReader cannot infer turn from the DOM alone; the caller
                  must track this based on move number.

        Returns:
            A chess.Board with pieces placed correctly. Castling rights and
            en-passant are not recoverable from DOM — both are cleared.
        """
        pieces = self._extract_pieces()
        board = chess.Board(fen=None)  # empty board, no castling, no EP
        board.turn = turn
        board.castling_rights = chess.BB_EMPTY

        for square, piece in pieces.items():
            board.set_piece_at(square, piece)

        logger.debug("BoardReader: read %d pieces, turn=%s", len(pieces), "W" if turn else "B")
        return board

    def read_fen(self, turn: chess.Color = chess.WHITE) -> str:
        """Return a FEN string for the current board position."""
        return self.read_board(turn=turn).fen()

    def get_clocks(self) -> tuple[float, float]:
        """Try to read the on-screen clocks.

        Returns (white_seconds, black_seconds).
        Returns (0.0, 0.0) if no clocks are visible on screen.
        """
        try:
            return self._parse_clocks()
        except Exception as e:
            logger.debug("Clock parse failed: %s", e)
            return 0.0, 0.0

    def extract_pieces(self) -> dict[chess.Square, chess.Piece]:
        """Query the DOM for all piece divs, return {square: Piece} mapping."""
        return self._extract_pieces()

    def get_highlight_squares(self) -> list[chess.Square]:
        """Return python-chess squares currently highlighted on the board (last move indicators)."""
        page = self._page
        try:
            class_list = page.evaluate("""
                () => {
                    const board = document.querySelector('#board-single, .board-layout-main, wc-chess-board, chess-board');
                    const els = board ? board.querySelectorAll('.highlight') : document.querySelectorAll('.highlight');
                    const classes = [];
                    for (let i = 0; i < els.length; i++) {
                        classes.push(els[i].className || '');
                    }
                    return classes;
                }
            """)
            squares: list[chess.Square] = []
            for class_str in class_list:
                sq_match = _SQ_RE.search(class_str)
                if sq_match:
                    square = _chesscom_sq_to_chess(sq_match.group(1))
                    if square is not None and square not in squares:
                        squares.append(square)
            return squares
        except Exception:
            return []

    def detect_time_control_from_dom(self) -> str | None:
        """Inspect headers, labels, clocks, URL, and page title to detect time control.

        Uses a wide-net approach including chess.com Play vs Computer specific selectors.
        """
        page = self._page
        try:
            res = page.evaluate(r"""
                () => {
                    // ── Pass 1: dedicated time-control label elements ─────────────────
                    // Includes vs-Computer specific selectors.
                    const tcSelectors = [
                        '.time-control-component',
                        '[data-game-type]',
                        '.game-type-label',
                        '.header-title-component',
                        '.game-meta-component',
                        '.sidebar-game-rules',
                        '.custom-game-options-component',
                        '.game-info-timecondition',
                        '[class*="time-control"]',
                        '[class*="game-type"]',
                        '[class*="clock-label"]',
                        '.challenge-link-component',
                        '.board-layout-sidebar [class*="time"]',
                        '.sidebar-component [class*="time"]',
                        '.game-rules-panel',
                        '.board-controls-timer',
                        '[data-timetype]',
                        '[data-time-control]',
                        '[class*="game-rules"]',
                        '[class*="time-selector"]',
                        '.bot-game-time-component',
                        '.game-settings-time',
                    ];
                    for (const sel of tcSelectors) {
                        try {
                            const el = document.querySelector(sel);
                            if (el) {
                                const t = (el.innerText || el.textContent || '').trim();
                                if (t.length > 0) return t;
                            }
                        } catch(e) {}
                    }

                    // ── Pass 1b: page <title> — often "Play 3 | 0 Blitz | Chess.com" ─
                    try {
                        const title = document.title || '';
                        if (title) return '__title__:' + title;
                    } catch(e) {}

                    // ── Pass 2: clock elements showing MM:SS ──────────────────────────
                    const clockSelectors = [
                        '.clock-time-monospace',
                        '.clock-component',
                        '[class*="clock-time"]',
                        '[class*="clock"][class*="time"]',
                    ];
                    const clockTexts = [];
                    for (const sel of clockSelectors) {
                        const els = document.querySelectorAll(sel);
                        for (const el of els) {
                            const t = (el.innerText || el.textContent || '').trim();
                            if (t && /\d/.test(t)) clockTexts.push(t);
                        }
                        if (clockTexts.length >= 2) break;
                    }
                    if (clockTexts.length > 0) return '__clocks__:' + clockTexts.join('|');

                    return null;
                }
            """)
            if res:
                res = str(res)
                # Strip __title__ prefix — treat it exactly like a label string
                if res.startswith('__title__:'):
                    res = res[len('__title__:'):]

                # If we got explicit label text, scan it for known keywords / time formats
                if not res.startswith('__clocks__:'):
                    lower = res.lower()
                    if 'no timer' in lower or 'untimed' in lower:
                        return 'untimed'
                    if 'bullet' in lower or '1 min' in lower or '1+0' in lower or '2+1' in lower:
                        return 'bullet'
                    if 'blitz' in lower or '3 min' in lower or '5 min' in lower or '3+0' in lower or '5+0' in lower or '3+2' in lower or '5+3' in lower:
                        return 'blitz'
                    if 'rapid' in lower or '10 min' in lower or '15 min' in lower or '10+0' in lower or '15+10' in lower:
                        return 'rapid'
                    if 'classical' in lower or '30 min' in lower or '30+0' in lower:
                        return 'classical'
                    # Try to extract digits like "3", "5", "10", "15" from format "X|Y" or "X+Y" or "X min"
                    import re as _re
                    m = _re.search(r'(\d+)\s*(?:[|+:]|\s*min)', lower)
                    if m:
                        mins = int(m.group(1))
                        if mins <= 2:
                            return 'bullet'
                        elif mins <= 5:
                            return 'blitz'
                        elif mins <= 15:
                            return 'rapid'
                        else:
                            return 'classical'
                    return res  # return the raw label — DigitalClock.create() will parse it

                else:
                    # Infer from clock text values
                    clock_parts = res[len('__clocks__:'):].split('|')
                    import re as _re
                    for ct in clock_parts:
                        m = _re.search(r'(\d+):(\d+)(?::(\d+))?', ct)
                        if m:
                            if m.group(3) is not None:
                                secs = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
                            else:
                                secs = int(m.group(1)) * 60 + int(m.group(2))
                            # Classify
                            if secs <= 75:
                                return 'bullet'
                            elif secs <= 135:
                                return 'bullet'
                            elif secs <= 200:
                                return 'blitz'
                            elif secs <= 330:
                                return 'blitz'
                            elif secs <= 650:
                                return 'rapid'
                            elif secs <= 950:
                                return 'rapid'
                            else:
                                return 'classical'
        except Exception:
            pass

        # Final fallback: check the URL
        try:
            url = page.url or ""
            url_lower = url.lower()
            for tc in ("bullet", "blitz", "rapid", "classical", "daily"):
                if tc in url_lower:
                    return tc
            # Check for time hints in URL query params or path, e.g. "?tc=600" or "/1" or "/3"
            import re as _re
            m = _re.search(r'[?&/](?:tc|time|minutes?)=?(\d+)', url_lower)
            if m:
                mins = int(m.group(1))
                if mins <= 2:
                    return 'bullet'
                elif mins <= 5:
                    return 'blitz'
                elif mins <= 15:
                    return 'rapid'
                else:
                    return 'classical'
        except Exception:
            pass
        return None

    def get_player_ratings(self) -> tuple[int, int]:
        """Try to read (self_elo, opp_elo) from player taglines on screen.

        Returns (self_elo, opp_elo).
        Returns 0 for a side when account is unrated / rating not detectable
        so that callers (MetaController) can handle new accounts appropriately.
        """
        page = self._page
        try:
            ratings = page.evaluate("""
                () => {
                    // Cast a wide net — chess.com changes class names frequently.
                    const BOTTOM_SEL = [
                        '#board-layout-player-bottom .user-tagline-rating',
                        '#board-layout-player-bottom .user-tagline-component',
                        '.player-tagline-bottom .user-tagline-rating',
                        '.player-tagline-bottom .user-rating',
                        '.player-tagline .user-tagline-rating',
                        '.user-tagline-rating',
                        '[data-cy="player-tagline-rating"]',
                        '.clock-bottom .user-tagline-rating',
                    ];
                    const TOP_SEL = [
                        '#board-layout-player-top .user-tagline-rating',
                        '#board-layout-player-top .user-tagline-component',
                        '.player-tagline-top .user-tagline-rating',
                        '.player-tagline-top .user-rating',
                        '.bot-player-rating',
                        '.bot-tagline-rating',
                        '.bot-rating',
                        '.clock-top .user-tagline-rating',
                    ];

                    const findText = (selectors) => {
                        for (const sel of selectors) {
                            const el = document.querySelector(sel);
                            if (el && el.innerText.trim()) return el.innerText.trim();
                        }
                        return null;
                    };

                    // Get all rating elements by vertical position (bottom = self when not flipped)
                    const allRatingEls = Array.from(document.querySelectorAll(
                        '.user-tagline-rating, [class*="tagline"][class*="rating"]'
                    ));
                    if (allRatingEls.length >= 2) {
                        const sorted = allRatingEls.slice().sort(
                            (a, b) => b.getBoundingClientRect().top - a.getBoundingClientRect().top
                        );
                        return [
                            sorted[0].innerText.trim(),   // bottom-most = self
                            sorted[sorted.length - 1].innerText.trim()  // top-most = opponent
                        ];
                    }

                    return [
                        findText(BOTTOM_SEL),
                        findText(TOP_SEL),
                    ];
                }
            """)

            def _parse_rating(val: str | None) -> int:
                """Parse a rating string. Returns 0 if unrated / not found."""
                if not val:
                    return 0
                val_clean = val.strip().lower()
                # Chess.com shows "Unrated" or "?" for new accounts
                if "unrated" in val_clean or "provisional" in val_clean or val_clean == "?":
                    return 0
                m = re.search(r"(\d{2,4})", val)
                if m:
                    return int(m.group(1))
                return 0  # Unknown — let MetaController pick a sensible default

            self_elo = _parse_rating(ratings[0] if ratings else None)
            opp_elo = _parse_rating(ratings[1] if len(ratings) > 1 else None)
            logger.debug("Detected ratings: self=%d opp=%d (raw=%s)", self_elo, opp_elo, ratings)
            return self_elo, opp_elo
        except Exception as e:
            logger.debug("Failed to read player ratings from DOM: %s", e)
            return 0, 0

    # ── Private ─────────────────────────────────────────────────────────────────
    def _extract_pieces(self) -> dict[chess.Square, chess.Piece]:
        """Query the DOM for all piece divs in ONE fast JavaScript call."""
        page = self._page
        try:
            class_list = page.evaluate("""
                () => {
                    const board = document.querySelector('#board-single, .board-layout-main, wc-chess-board, chess-board');
                    const els = board ? board.querySelectorAll('.piece') : document.querySelectorAll('wc-chess-board .piece, chess-board .piece, .board .piece');
                    const classes = [];
                    for (let i = 0; i < els.length; i++) {
                        classes.push(els[i].className || '');
                    }
                    return classes;
                }
            """)
        except Exception:
            return {}

        pieces: dict[chess.Square, chess.Piece] = {}
        for class_str in class_list:
            piece_match = _PIECE_RE.search(class_str)
            sq_match = _SQ_RE.search(class_str)
            if not piece_match or not sq_match:
                continue

            piece_code = piece_match.group(1)  # e.g. "wp"
            sq_code = sq_match.group(1)     # e.g. "12"

            color = chess.WHITE if piece_code[0] == "w" else chess.BLACK
            piece_type = _PIECE_TYPE.get(piece_code[1])
            square = _chesscom_sq_to_chess(sq_code)

            if piece_type is None or square is None:
                continue

            pieces[square] = chess.Piece(piece_type, color)

        return pieces

    def _parse_clocks(self) -> tuple[float, float]:
        """Parse the on-screen clock elements and return (white_s, black_s).

        Uses a very broad selector set so that it works across chess.com's
        frequently-changing DOM structure. Returns (0.0, 0.0) if no clocks found.
        """
        page = self._page

        def _parse_one(text: str) -> float:
            """Parse 'M:SS' or 'H:MM:SS' or bare seconds into float seconds."""
            if not text:
                return 0.0
            m = re.search(r"(\d+):(\d+)(?::(\d+))?", text)
            if m:
                if m.group(3) is not None:
                    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
                return int(m.group(1)) * 60 + float(m.group(2))
            try:
                parts = text.split(":")
                if len(parts) == 1:
                    val = float(parts[0])
                    return val if val > 0 else 0.0
                elif len(parts) == 2:
                    return int(parts[0]) * 60 + float(parts[1])
                else:
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            except ValueError:
                return 0.0

        try:
            clock_data = page.evaluate(r"""
                () => {
                    // ── Strategy 1: known positional wrappers (top / bottom) ─────────
                    const BOTTOM_SELS = [
                        '#board-layout-player-bottom .clock-time-monospace',
                        '#board-layout-player-bottom [class*="clock-time"]',
                        '#board-layout-player-bottom .clock-component',
                        '#board-layout-player-bottom [class*="clock"]',
                        '.player-tagline-bottom .clock-time-monospace',
                        '.player-tagline-bottom .clock-component',
                        '.clock-bottom .clock-time-monospace',
                        '.clock-bottom [class*="clock-time"]',
                        '.clock-bottom',
                        '[data-player-side="bottom"] [class*="clock"]',
                        '[class*="player-bottom"] [class*="clock"]',
                    ];
                    const TOP_SELS = [
                        '#board-layout-player-top .clock-time-monospace',
                        '#board-layout-player-top [class*="clock-time"]',
                        '#board-layout-player-top .clock-component',
                        '#board-layout-player-top [class*="clock"]',
                        '.player-tagline-top .clock-time-monospace',
                        '.player-tagline-top .clock-component',
                        '.clock-top .clock-time-monospace',
                        '.clock-top [class*="clock-time"]',
                        '.clock-top',
                        '[data-player-side="top"] [class*="clock"]',
                        '[class*="player-top"] [class*="clock"]',
                    ];

                    const getText = (sels) => {
                        for (const sel of sels) {
                            try {
                                const el = document.querySelector(sel);
                                if (el) {
                                    const t = (el.innerText || el.textContent || '').trim();
                                    if (t && /\d/.test(t)) return t;
                                }
                            } catch(e) {}
                        }
                        return null;
                    };

                    const bottomText = getText(BOTTOM_SELS);
                    const topText = getText(TOP_SELS);

                    if (bottomText && topText) {
                        return { bottom: bottomText, top: topText, found: true };
                    }

                    // ── Strategy 2: find ALL clock elements, sort by Y position ──────
                    const CLOCK_SELS = [
                        '.clock-time-monospace',
                        '[class*="clock-time"]',
                        '.clock-component',
                        '[class*="clock"][class*="component"]',
                        '[class*="clock"][class*="time"]',
                    ];
                    let allClocks = [];
                    for (const sel of CLOCK_SELS) {
                        const els = Array.from(document.querySelectorAll(sel));
                        for (const el of els) {
                            const t = (el.innerText || el.textContent || '').trim();
                            if (t && /\d:\d/.test(t)) {
                                const rect = el.getBoundingClientRect();
                                if (rect.width > 0 && rect.height > 0) {
                                    allClocks.push({ text: t, y: rect.top });
                                }
                            }
                        }
                        if (allClocks.length >= 2) break;
                    }

                    if (allClocks.length >= 2) {
                        // Sort by Y ascending (top clock has lower Y value)
                        allClocks.sort((a, b) => a.y - b.y);
                        return {
                            top: allClocks[0].text,
                            bottom: allClocks[allClocks.length - 1].text,
                            found: true
                        };
                    }

                    // ── Strategy 3: any element with a time pattern ────────────────
                    const allEls = Array.from(document.querySelectorAll('*'));
                    const timeElems = [];
                    for (const el of allEls) {
                        if (el.children.length > 0) continue;  // leaf nodes only
                        const t = (el.innerText || el.textContent || '').trim();
                        if (/^\d{1,2}:\d{2}$/.test(t)) {
                            const rect = el.getBoundingClientRect();
                            if (rect.width > 0 && rect.height > 0) {
                                timeElems.push({ text: t, y: rect.top });
                            }
                        }
                    }
                    if (timeElems.length >= 2) {
                        timeElems.sort((a, b) => a.y - b.y);
                        return {
                            top: timeElems[0].text,
                            bottom: timeElems[timeElems.length - 1].text,
                            found: true
                        };
                    }

                    return { bottom: '', top: '', found: false };
                }
            """)
        except Exception:
            return 0.0, 0.0

        if not clock_data.get("found"):
            return 0.0, 0.0

        bottom_s = _parse_one(clock_data.get("bottom", ""))
        top_s = _parse_one(clock_data.get("top", ""))

        if bottom_s <= 0.0 and top_s <= 0.0:
            return 0.0, 0.0

        # Determine which clock belongs to which color based on board orientation
        try:
            is_flipped = page.evaluate("""
                () => {
                    const b = document.querySelector('wc-chess-board, chess-board');
                    if (b) {
                        const cls = b.getAttribute('class') || '';
                        return cls.includes('flipped') || b.hasAttribute('flipped');
                    }
                    return false;
                }
            """)
        except Exception:
            is_flipped = False

        if is_flipped:
            # Player at bottom is Black, opponent at top is White
            return top_s, bottom_s
        else:
            # Player at bottom is White, opponent at top is Black
            return bottom_s, top_s
