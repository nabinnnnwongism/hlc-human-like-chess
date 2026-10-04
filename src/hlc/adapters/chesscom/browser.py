"""browser.py — Playwright browser controller for chess.com.

Handles:
  - Launching a stealthy Chromium instance (normal mode)
  - Attaching to an existing Opera / Chrome via CDP (--cdp-port mode)
  - Logging in to chess.com
  - Navigating to Play vs Computer
  - Selecting a bot opponent
  - Starting the game and detecting game-over
  - Session cookie persistence (so the HLC dedicated profile stays logged-in)
"""

from __future__ import annotations

import logging
import queue
import re
import time
from typing import Callable, Literal

from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright

logger = logging.getLogger(__name__)

# Selectors for login-state detection (chess.com 2025/2026)
_SEL_LOGGED_IN = (
    ".user-username-component, "
    ".user-tagline-username, "
    "[data-cy='user-nav-username'], "
    ".nav-action-user .icon-font-chess, "
    ".board-layout-player .user-username-component"
)

# ── Selectors ──────────────────────────────────────────────────────────────────
# Updated for chess.com layout (2025/2026).
_SEL_USERNAME   = "#login-username, input#username:visible, input[name='_username']:visible"
_SEL_PASSWORD   = "#login-password, input#password:visible, input[name='_password']:visible"
_SEL_LOGIN_BTN  = "button#login, button[type='submit']"
_SEL_PLAY_NAV   = "a[href='/play/computer']"                   # Top-nav "Play" → Computer
_SEL_CHOOSE_BOT = ".bot-selection-tile, .select-bot-tile"     # Bot selection tile
_SEL_PLAY_BTN   = "button.bot-selection-cta-button-button, button.ui_v5-button-component, button:has-text('Play')"  # "Play" confirm button
_SEL_SETUP_CTA   = "button.bot-selection-cta-button-button, .selection-sidebar-play-button"
_SEL_BOARD      = "wc-chess-board, chess-board"               # Main board element
_SEL_GAME_OVER  = ".game-over-modal-content, .modal-game-over, .game-over-dialog-content, .game-over-header-component, div[data-cy='game-over-modal'], .board-modal-container"
_SEL_RESIGN_BTN = "button[aria-label='Resign'], button[data-cy='resign-button'], button.resign-button-component, .game-controls-resign, button:has-text('Resign')"
_SEL_IN_GAME_PANEL = ".game-controls-component, .move-list-component, .game-notation-wrap, .moves-component"



class ChessDotComBrowser:
    """Manages the Playwright browser lifecycle for chess.com.

    Normal mode (spawns a new browser window):
        with ChessDotComBrowser(username="u", password="p") as browser:
            browser.start_game_vs_computer(color="white")
            page = browser.page

    CDP mode (attaches to your existing Opera / Chrome):
        with ChessDotComBrowser.connect_cdp(cdp_port=9222) as browser:
            # No login needed — uses your existing session
            page = browser.page
    """

    def __init__(
        self,
        username: str = "",
        password: str = "",
        headless: bool = False,
        slow_mo_ms: int = 80,
        _cdp_mode: bool = False,
        on_event: Callable[[dict], None] | None = None,
    ) -> None:
        """Initialise the browser controller.

        Args:
            username:    Chess.com account username (not needed in CDP mode).
            password:    Chess.com account password (not needed in CDP mode).
            headless:    Run browser invisibly. Keep False so you can watch it.
            slow_mo_ms:  Milliseconds delay between Playwright actions.
            _cdp_mode:   Internal flag — set by connect_cdp(), do not use directly.
            on_event:    Optional callback invoked on browser events (clicks, navigations, tab focus).
        """
        self.username = username
        self.password = password
        self.headless = headless
        self.slow_mo_ms = slow_mo_ms
        self._cdp_mode = _cdp_mode
        self._cdp_port: int = 9222  # overridden by connect_cdp()
        self._on_event = on_event
        self.event_queue: queue.Queue[dict] = queue.Queue()

        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    # ── Context manager ────────────────────────────────────────────────────────
    def __enter__(self) -> "ChessDotComBrowser":
        if self._cdp_mode:
            self._connect_to_cdp()
        else:
            self._start()
        return self

    def __exit__(self, *_) -> None:
        if self._cdp_mode:
            self._disconnect_cdp()
        else:
            self._stop()

    @classmethod
    def connect_cdp(
        cls,
        cdp_port: int = 9222,
        on_event: Callable[[dict], None] | None = None,
    ) -> "ChessDotComBrowser":
        """Create a browser instance that attaches to an existing running browser.

        The browser (Opera, Chrome, Edge, etc.) must be launched with:
            --remote-debugging-port=<cdp_port>

        No login is needed — the existing browser session is reused.

        Args:
            cdp_port: The remote debugging port the browser is listening on.
            on_event: Callback invoked on real-time browser actions (click, navigation, focus).

        Returns:
            A ChessDotComBrowser instance ready to be used as a context manager.
        """
        instance = cls(_cdp_mode=True, on_event=on_event)
        instance._cdp_port = cdp_port
        return instance

    # ── Public helpers ─────────────────────────────────────────────────────────
    @property
    def page(self) -> Page:
        if self._cdp_mode and self._browser:
            return self.get_active_chess_page()
        if self._page is None:
            raise RuntimeError("Browser not started. Use as context manager.")
        return self._page

    def get_active_chess_page(self) -> Page:
        """Find the active chess.com tab, dynamically following where the user is.

        Always performs a full re-scan of ALL open tabs every call.
        Also checks the cached page's LIVE URL (it may have navigated since last scan).
        """
        if self._browser and self._browser.contexts:
            all_pages: list[tuple["Page", object, str]] = []
            for ctx in self._browser.contexts:
                for pg in ctx.pages:
                    try:
                        url = pg.url or ""
                    except Exception:
                        url = ""
                    all_pages.append((pg, ctx, url))

            # Throttle: only log the scan once every 5 s to avoid terminal spam
            _now = time.time()
            if not hasattr(self, "_last_scan_log") or _now - self._last_scan_log >= 5.0:
                logger.debug(
                    "CDP page scan: %d pages found: %s",
                    len(all_pages),
                    [u for _, _, u in all_pages],
                )
                self._last_scan_log = _now


            # Priority 1: chess.com game or play URL
            for pg, ctx, url in all_pages:
                if "chess.com" in url and ("/game/" in url or "/play/" in url):
                    self._page = pg
                    self._context = ctx
                    return pg

            # Priority 2: any chess.com tab (home, puzzles, etc.)
            for pg, ctx, url in all_pages:
                if "chess.com" in url:
                    self._page = pg
                    self._context = ctx
                    return pg

            # Priority 3: re-check cached page's LIVE url (may have navigated)
            if self._page is not None:
                try:
                    live_url = self._page.url or ""
                    if "chess.com" in live_url:
                        return self._page
                except Exception:
                    pass

            # Priority 4: return first non-internal page (caller keeps polling)
            for pg, ctx, url in all_pages:
                if not url.startswith("chrome://") and not url.startswith("about:"):
                    return pg

            # Priority 5: return any page
            if all_pages:
                return all_pages[0][0]

        # Absolute last resort: cached page
        if self._page is not None:
            return self._page
        if self._browser and self._browser.contexts and self._browser.contexts[0].pages:
            self._page = self._browser.contexts[0].pages[0]
            return self._page
        raise RuntimeError("Browser not started or no open pages.")

    def navigate_to_chess(self) -> None:
        """Navigate whatever page is available directly to chess.com.

        Called when no chess.com tab can be found via CDP page enumeration.
        This handles the Opera quirk where user tabs aren't visible to Playwright
        until they are navigated via Playwright itself.
        """
        try:
            # Get whatever page we have (may be an internal Opera page)
            if self._browser and self._browser.contexts:
                for ctx in self._browser.contexts:
                    for pg in ctx.pages:
                        try:
                            url = pg.url or ""
                            # Skip truly internal pages that can't navigate
                            if url.startswith("devtools://"):
                                continue
                            logger.info(
                                "Navigating page '%s' -> chess.com/play/computer", url
                            )
                            pg.goto(
                                "https://www.chess.com/play/computer",
                                wait_until="domcontentloaded",
                                timeout=15_000,
                            )
                            self._page = pg
                            return
                        except Exception:
                            continue
        except Exception as exc:
            logger.warning("navigate_to_chess failed: %s", exc)



    def start_game_vs_computer(
        self,
        color: Literal["white", "black", "random"] = "random",
        bot_name: str | None = None,
    ) -> None:
        """Log in, navigate to Play vs Computer, and start a game.

        Args:
            color:    Which side the human plays. The bot (HLC) plays the other side.
            bot_name: Optional bot name to select (e.g. 'Martin'). If None, picks
                      the first available bot on the page.
        """
        self._login()
        self._navigate_to_computer()
        self._select_bot_and_start(color=color, bot_name=bot_name)
        self._wait_for_board()
        logger.info("Game started. Board is ready.")

    def prepare_user_session(self) -> None:
        """Log in automatically and navigate to /play/computer, ready for user selection."""
        self._login()
        self._navigate_to_computer()
        # Dismiss introductory modal if present so it doesn't block the user
        try:
            start_modal_btn = self.page.locator("button:has-text('Start'), .modal-first-time-button").first
            if start_modal_btn.is_visible(timeout=3000):
                start_modal_btn.click()
                time.sleep(1.0)
        except Exception:
            pass

    # ── Session helpers (CDP mode) ───────────────────────────────────────────

    def is_logged_in(self) -> bool:
        """Return True if the user is logged in to chess.com.

        Uses multiple signals — URL first (fast), then DOM, then JS globals.
        """
        try:
            url = self._page.url or ""

            # Definitely NOT logged in if on the login/register page
            if "/login" in url or "/register" in url:
                return False

            # Almost certainly logged in if on any chess.com page that isn't login
            # (play, game, puzzles, home, etc.)  — skip expensive DOM scan.
            if "chess.com" in url and url not in ("https://www.chess.com/", "https://chess.com/"):
                # Quick JS check to confirm (fast, single expression)
                try:
                    quick = self._page.evaluate(
                        "() => !!(window.SITE_USER && window.SITE_USER.username)"
                    )
                    if quick:
                        return True
                except Exception:
                    pass

            # Full DOM scan for login evidence
            result = self._page.evaluate("""
                () => {
                    // Signal 1: chess.com JS global set after auth
                    if (window.SITE_USER && window.SITE_USER.username) return true;
                    if (window.__chesscom && window.__chesscom.user) return true;

                    // Signal 2: any username text element
                    const sels = [
                        '.user-username-component',
                        '.user-tagline-username',
                        '[data-cy="user-nav-username"]',
                        '[data-user-id]',
                        '.profile-card-component',
                        '.board-layout-player .user-username-component',
                        '.clock-player-name',
                        '.player-tagline',
                        '.board-player-default-component',
                        '.user-miniprofile-component',
                        '[class*="user-username"]',
                        '[class*="player-name"]',
                    ];
                    for (const sel of sels) {
                        const el = document.querySelector(sel);
                        if (el && el.textContent.trim().length > 1) return true;
                    }

                    // Signal 3: logout link (only exists when logged in)
                    if (document.querySelector('a[href*="/logout"]')) return true;

                    // Signal 4: avatar image in nav or board
                    if (document.querySelector(
                        '.user-avatar-component, .nav-action-user img, '
                        + '.header-user-image, img[class*="avatar"]'
                    )) return true;

                    // Signal 5: any element with a data-user attribute
                    if (document.querySelector('[data-user], [data-username], [data-user-id]')) return true;

                    return false;
                }
            """)
            return bool(result)
        except Exception:
            return False

    def get_logged_in_username(self) -> str | None:
        """Return the logged-in username, or None if not logged in."""
        try:
            result = self._page.evaluate("""
                () => {
                    if (window.SITE_USER && window.SITE_USER.username)
                        return window.SITE_USER.username;
                    const el = document.querySelector(
                        '.user-username-component, .user-tagline-username, [data-cy="user-nav-username"]'
                    );
                    return el ? el.textContent.trim() : null;
                }
            """)
            return str(result) if result else None
        except Exception:
            return None

    def save_current_session(self) -> None:
        """Persist the current context's chess.com cookies to disk.

        Call this after confirming the user is logged in so the next
        HLC launch reuses the session automatically.
        """
        try:
            from hlc.adapters.chesscom.session_manager import save_session
            if self._context:
                cookies = self._context.cookies()
                save_session(cookies)
        except Exception as exc:
            logger.warning("Could not save session: %s", exc)

    def detect_page_context(self) -> dict:
        """Deep-scan the current chess.com page and return a comprehensive state dict."""
        try:
            p = self.page
            url = p.url or ""
        except Exception:
            p = self._page
            url = p.url if p else ""

        # ── URL-based mode detection ───────────────────────────────────────────
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

        # ── Single-pass JS scan of the entire page state ───────────────────────
        try:
            js_state = self._page.evaluate(r"""
                () => {
                    const out = {
                        logged_in: false,
                        username: null,
                        game_type: null,
                        board_side: null,
                        in_game: false,
                        game_over: false,
                        move_count: 0,
                        opp_username: null,
                    };

                    // ── Login ─────────────────────────────────────────────
                    if (window.SITE_USER && window.SITE_USER.username) {
                        out.logged_in = true;
                        out.username = window.SITE_USER.username;
                    } else {
                        const uEls = [
                            '.user-username-component',
                            '.user-tagline-username',
                            '[data-cy="user-nav-username"]',
                        ];
                        for (const s of uEls) {
                            const el = document.querySelector(s);
                            if (el && el.textContent.trim()) {
                                out.logged_in = true;
                                out.username = el.textContent.trim();
                                break;
                            }
                        }
                        // fallback: logout link only exists when logged in
                        if (!out.logged_in && document.querySelector('a[href*="/logout"]'))
                            out.logged_in = true;
                        // fallback: avatar image
                        if (!out.logged_in && document.querySelector('.user-avatar-component, .nav-action-user img'))
                            out.logged_in = true;
                    }

                    // ── Board presence ────────────────────────────────────
                    const board = document.querySelector('wc-chess-board, chess-board');
                    out.in_game = !!board;

                    // ── Board side (White or Black) ───────────────────────
                    if (board) {
                        const cls = board.getAttribute('class') || '';
                        out.board_side = cls.includes('flipped') ? 'black' : 'white';
                    }

                    // ── Game over ─────────────────────────────────────────
                    const goModal = document.querySelector(
                        '.game-over-modal-content, .modal-game-over, '
                        + '.game-over-header-component, [data-cy="game-over-modal"]'
                    );
                    // Also check for post-game buttons
                    const rematch = document.querySelector('button.game-over-button-component');
                    out.game_over = !!(goModal || rematch);
                    if (out.game_over) out.in_game = false;

                    // ── Resign button = game is live ──────────────────────
                    const resignBtn = document.querySelector(
                        'button[aria-label="Resign"], button[data-cy="resign-button"]'
                    );
                    if (resignBtn) out.in_game = true;

                    // ── Move count (entries in the notation panel) ────────
                    const moves = document.querySelectorAll(
                        '.move-list-row, .notation-row, .moves-component .move'
                    );
                    out.move_count = moves.length;

                    // ── Game type from DOM / headers ──────────────────────
                    const gtEl = document.querySelector(
                        '.time-control-component, [data-game-type], .game-type-label'
                    );
                    if (gtEl) {
                        const txt = gtEl.textContent.toLowerCase();
                        for (const t of ['bullet','blitz','rapid','classical','daily']) {
                            if (txt.includes(t)) { out.game_type = t; break; }
                        }
                        if (!out.game_type) {
                            const m = txt.match(/(\d+)\s*[\+:]/);
                            if (m) {
                                const mins = parseInt(m[1]);
                                out.game_type = mins <= 2 ? 'bullet'
                                              : mins <= 5 ? 'blitz'
                                              : mins <= 15 ? 'rapid' : 'classical';
                            }
                        }
                    }

                    // ── Opponent username ─────────────────────────────────
                    const playerEls = document.querySelectorAll(
                        '.board-layout-player .user-username-component, '
                        + '.players-component .user-username-component'
                    );
                    for (const el of playerEls) {
                        const uname = el.textContent.trim();
                        if (uname && uname !== out.username) {
                            out.opp_username = uname;
                            break;
                        }
                    }

                    return out;
                }
            """)
        except Exception as exc:
            logger.debug("JS scan failed: %s", exc)
            js_state = {}

        # Merge URL-detected game_type with JS-detected one (JS wins if found)
        url_game_type = self._detect_game_type_from_url(url)
        resolved_game_type = js_state.get("game_type") or url_game_type

        return {
            "logged_in":    bool(js_state.get("logged_in", False)),
            "username":     js_state.get("username"),
            "url":          url,
            "mode":         mode,
            "game_type":    resolved_game_type,
            "board_side":   js_state.get("board_side"),
            "in_game":      self.is_match_active(),
            "game_over":    bool(js_state.get("game_over", False)),
            "move_count":   int(js_state.get("move_count", 0)),
            "opp_username": js_state.get("opp_username"),
        }

    def _detect_game_type_from_url(self, url: str) -> str | None:
        """Extract game type from URL query params or path segments."""
        url_lower = url.lower()
        for t in ("bullet", "blitz", "rapid", "classical", "daily"):
            if t in url_lower:
                return t
        return None

    def is_match_active(self) -> bool:
        """Return True when a chess game is actively live and playable.

        Uses Playwright locator API (not page.evaluate) for reliability in CDP/Opera mode.
        Signals confirmed by debug_dom.py against a live Opera+chess.com session:
          - Resign button visible (aria-label=Resign)  resign_visible=True
          - Undo button visible                         visible_buttons includes 'Undo'
          - Show Hint button visible                    visible_buttons includes 'Show Hint'
        """
        page = self.page
        try:
            # 1. Game-over check
            if self.is_game_over():
                return False

            url = page.url or ""

            # 2. Definite non-match pages
            if "/login" in url or "/register" in url:
                return False
            if url.rstrip("/") in ("https://www.chess.com", "https://chess.com"):
                return False

            # 3. Board element must be visible
            try:
                board_loc = page.locator(_SEL_BOARD).first
                if not board_loc.is_visible(timeout=150):
                    logger.debug("is_match_active: board not visible")
                    return False
            except Exception as _be:
                logger.debug("is_match_active: board check failed: %s", _be)
                return False

            # 4. Signal A: Resign button — CONFIRMED WORKING by debug_dom.py
            for _sel in [
                'button[aria-label="Resign"]',
                'button[data-cy="resign-button"]',
                '.resign-button-component',
            ]:
                try:
                    if page.locator(_sel).first.is_visible(timeout=150):
                        logger.debug("is_match_active: TRUE via resign (%s)", _sel)
                        return True
                except Exception:
                    pass

            # 5. Signal B: Undo button — CONFIRMED in debug visible_buttons list
            for _sel in ['button[aria-label="Undo"]', 'button:has-text("Undo")']:
                try:
                    if page.locator(_sel).first.is_visible(timeout=150):
                        logger.debug("is_match_active: TRUE via Undo button")
                        return True
                except Exception:
                    pass

            # 6. Signal C: Show Hint / Hint button — CONFIRMED in debug visible_buttons
            for _sel in [
                'button:has-text("Show Hint")',
                'button[aria-label="Hint"]',
                'button:has-text("Hint")',
            ]:
                try:
                    if page.locator(_sel).first.is_visible(timeout=150):
                        logger.debug("is_match_active: TRUE via Hint button")
                        return True
                except Exception:
                    pass

            # 7. Signal D: Abort button (some vs-computer modes)
            for _sel in [
                'button[aria-label="Abort"]',
                'button[data-cy="abort-button"]',
                'button:has-text("Abort")',
            ]:
                try:
                    if page.locator(_sel).first.is_visible(timeout=150):
                        logger.debug("is_match_active: TRUE via Abort button")
                        return True
                except Exception:
                    pass

            # 8. Signal E: Takeback / offer draw / live game panel
            for _sel in [
                'button[aria-label="Takeback"]',
                '.game-controls-component',
                '.live-game-buttons-component',
                'button[aria-label="Offer Draw"]',
            ]:
                try:
                    if page.locator(_sel).first.is_visible(timeout=150):
                        logger.debug("is_match_active: TRUE via controls (%s)", _sel)
                        return True
                except Exception:
                    pass

            # 9. JS play-click flag (set by injected listener on Play button click)
            try:
                if page.evaluate("() => window.__hlc_play_clicked === true"):
                    logger.debug("is_match_active: TRUE via __hlc_play_clicked flag")
                    return True
            except Exception:
                pass

            # 10. Setup screen check — if setup Play CTA button visible, game NOT started
            for _sel in [
                'button.selection-sidebar-play-button',
                'button.bot-selection-cta-button-button',
                'button.custom-game-options-play-button',
            ]:
                try:
                    if page.locator(_sel).first.is_visible(timeout=100):
                        logger.debug("is_match_active: FALSE — setup button still visible")
                        return False
                except Exception:
                    pass

            # 11. Fallback: live game URL patterns
            if "/game/live/" in url or "/game/daily/" in url or "/game/computer" in url:
                logger.debug("is_match_active: TRUE via game URL")
                return True

            # 12. Ultimate fallback: >=20 pieces on /play/computer URL (no setup btn)
            if "/play/computer" in url:
                try:
                    _cnt = page.evaluate(
                        "() => document.querySelectorAll('wc-chess-board .piece, chess-board .piece').length"
                    )
                    if isinstance(_cnt, (int, float)) and _cnt >= 20:
                        logger.debug("is_match_active: TRUE via piece-count (%d)", _cnt)
                        return True
                except Exception:
                    pass

        except Exception as _outer:
            logger.debug("is_match_active: outer exception: %s", _outer)
        return False

    def is_game_over(self) -> bool:
        """Returns True if the game-over modal or post-game UI is visible."""
        page = self.page
        try:
            # 1. Check known modal containers
            modal = page.locator(_SEL_GAME_OVER)
            if modal.count() > 0 and modal.first.is_visible(timeout=40):
                return True

            # 2. Check post-game buttons that appear exclusively after game over
            for btn_text in [
                "Game Review",
                "Rematch",
                "New Game",
                "Play Again",
                "New 1 min",
                "New 3 min",
                "New 5 min",
                "New 10 min",
                "New 15|10",
                "New 3|2",
            ]:
                btn = page.locator(f"button:has-text('{btn_text}')")
                if btn.count() > 0 and btn.first.is_visible(timeout=25):
                    return True
        except Exception:
            pass
        return False

    def get_player_color(self) -> Literal["white", "black"]:
        """Detect which color the human player is assigned.

        Uses Multi-Signal Analysis with physical screen coordinates as ground truth:
        1. Screen Y-coordinate of White King (wk) vs Black King (bk):
           - In screen space, Y increases downwards.
           - Whichever king is physically lower on screen (higher Y) is the player's side!
           - If Y(wk) > Y(bk) + 40 -> White King is at bottom -> You are WHITE!
           - If Y(bk) > Y(wk) + 40 -> Black King is at bottom -> You are BLACK!
        2. Board CSS class / attribute:
           - Presence of 'flipped' on <wc-chess-board> or <chess-board> -> BLACK.
        3. Bottom player clock class:
           - .clock-black -> BLACK, .clock-white -> WHITE.
        4. Retries up to 8 times (~3 seconds) to allow pieces and orientation to mount.
        """
        page = self.page
        for attempt in range(8):
            try:
                # Signal 1: Physical screen Y coordinates of White King vs Black King
                result = page.evaluate("""
                    () => {
                        const wk = document.querySelector(
                            '#board-single .piece.wk, .board-layout-main .piece.wk, wc-chess-board .piece.wk, chess-board .piece.wk, .board .piece.wk'
                        );
                        const bk = document.querySelector(
                            '#board-single .piece.bk, .board-layout-main .piece.bk, wc-chess-board .piece.bk, chess-board .piece.bk, .board .piece.bk'
                        );
                        if (wk && bk) {
                            const wkY = wk.getBoundingClientRect().top;
                            const bkY = bk.getBoundingClientRect().top;
                            if (wkY > bkY + 40) return 'white';
                            if (bkY > wkY + 40) return 'black';
                        }

                        // Signal 2: Board class or attribute 'flipped'
                        const board = document.querySelector(
                            '#board-single wc-chess-board, .board-layout-main wc-chess-board, wc-chess-board, chess-board'
                        );
                        if (board) {
                            const cls = board.getAttribute('class') || '';
                            if (cls.includes('flipped') || board.hasAttribute('flipped')) {
                                return 'black';
                            }
                        }

                        // Signal 3: Bottom player clock / tagline color
                        const bottomClock = document.querySelector(
                            '#board-layout-player-bottom .clock-component, .player-tagline-bottom .clock-component'
                        );
                        if (bottomClock) {
                            const cls = bottomClock.getAttribute('class') || '';
                            if (cls.includes('clock-black')) return 'black';
                            if (cls.includes('clock-white')) return 'white';
                        }

                        return null;
                    }
                """)
                if result in ("white", "black"):
                    logger.info("Player color detected: %s (attempt %d)", result.upper(), attempt + 1)
                    return result
            except Exception:
                pass
            time.sleep(0.35)

        logger.warning("Could not definitively detect color after retries -- defaulting to WHITE.")
        return "white"

    # ── Private helpers ────────────────────────────────────────────────────────
    _CLICK_LISTENER_JS = """
        if (!window.__hlc_listener_attached) {
            window.__hlc_listener_attached = true;
            window.__hlc_play_clicked = false;
            window.addEventListener('click', (e) => {
                try {
                    const btn = e.target.closest('button, [role="button"], a');
                    if (btn) {
                        const txt = (btn.innerText || btn.getAttribute('aria-label') || '').trim().toLowerCase();
                        if (txt.includes('play') || txt.includes('start') || txt.includes('choose')) {
                            window.__hlc_play_clicked = true;
                        }
                    }
                } catch (err) {}
            }, true);
        }
    """

    def _start(self) -> None:
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo_ms,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        self._context = self._browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="en-US",
        )
        self._context.add_init_script(
            f"""
            Object.defineProperty(navigator, 'webdriver', {{get: () => undefined}});
            {self._CLICK_LISTENER_JS}
            """
        )
        self._page = self._context.new_page()
        logger.info("Browser started (headless=%s).", self.headless)

    def _stop(self) -> None:
        try:
            if self._page:
                self._page.close()
        except Exception:
            pass
        try:
            if self._context:
                self._context.close()
        except Exception:
            pass
        try:
            if self._browser:
                self._browser.close()
        except Exception:
            pass
        try:
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass
        logger.info("Browser stopped.")

    def _connect_to_cdp(self) -> None:
        """Attach Playwright to an already-running Chromium-based browser via CDP.

        Steps:
          1. Pre-validate that the CDP HTTP endpoint responds (up to 15s).
          2. Attempt Playwright connect_over_cdp() with up to 10 retries.

        This handles the race condition where Opera's TCP port is open but the
        CDP JSON endpoint isn't ready yet (common on Windows with large profiles).
        """
        import urllib.request
        import urllib.error

        cdp_url = f"http://localhost:{self._cdp_port}"
        json_url = f"{cdp_url}/json/version"

        logger.info("Connecting to existing browser via CDP at %s ...", cdp_url)

        # ── Step 1: Wait for the CDP HTTP endpoint to be ready ────────────────
        logger.info("Waiting for CDP HTTP endpoint at %s ...", json_url)
        _deadline = time.time() + 20.0
        _http_ok = False
        while time.time() < _deadline:
            try:
                with urllib.request.urlopen(json_url, timeout=2) as _resp:
                    if _resp.status == 200:
                        _http_ok = True
                        logger.info("CDP HTTP endpoint is ready (HTTP 200).")
                        break
            except (urllib.error.URLError, OSError, ConnectionRefusedError):
                pass
            except Exception:
                pass
            logger.debug("CDP endpoint not ready yet, retrying in 1.5s ...")
            time.sleep(1.5)

        if not _http_ok:
            logger.warning(
                "CDP endpoint did not respond after 20s. "
                "Attempting Playwright connect anyway — may fail."
            )

        # ── Step 2: Connect Playwright with retries ───────────────────────────
        self._playwright = sync_playwright().start()
        _max_attempts = 10
        _last_exc: Exception | None = None
        for _attempt in range(1, _max_attempts + 1):
            try:
                self._browser = self._playwright.chromium.connect_over_cdp(cdp_url)
                logger.info(
                    "Playwright CDP connected on attempt %d/%d.", _attempt, _max_attempts
                )
                _last_exc = None
                break
            except Exception as exc:
                _last_exc = exc
                logger.warning(
                    "CDP connect attempt %d/%d failed: %s — retrying in 2s ...",
                    _attempt,
                    _max_attempts,
                    exc,
                )
                time.sleep(2.0)

        if _last_exc is not None:
            raise RuntimeError(
                f"Could not connect to browser on port {self._cdp_port} "
                f"after {_max_attempts} attempts.\n"
                "Ensure Opera is running with:\n"
                f"  \"C:\\Users\\Admin\\AppData\\Local\\Programs\\Opera\\opera.exe\" "
                f"--remote-debugging-port={self._cdp_port} --remote-allow-origins=*\n"
                f"Original error: {_last_exc}"
            ) from _last_exc

        self._page = self.get_active_chess_page()
        logger.info("CDP mode attached. Tab='%s'  URL=%s", self._page.title(), self._page.url)

        # Inject the play-click listener into ALL open chess.com tabs.
        # In CDP mode add_init_script doesn't run, so we do it imperatively.
        for ctx in self._browser.contexts:
            for pg in ctx.pages:
                if "chess.com" in (pg.url or ""):
                    try:
                        pg.evaluate(self._CLICK_LISTENER_JS)
                    except Exception:
                        pass

    def _disconnect_cdp(self) -> None:
        """Detach from the CDP-connected browser without closing it."""
        # In CDP mode we must NOT close the browser/context/page — they belong to the user.
        try:
            if self._browser:
                self._browser.close()  # only closes the Playwright connection, not the real browser
        except Exception:
            pass
        try:
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass
        logger.info("CDP session detached (browser left running).")

    def _login(self) -> None:
        page = self.page
        logger.info("Logging in as '%s'...", self.username)
        page.goto("https://www.chess.com/login", wait_until="domcontentloaded")
        user_field = page.locator(_SEL_USERNAME).first
        pass_field = page.locator(_SEL_PASSWORD).first
        user_field.wait_for(state="visible", timeout=15_000)
        user_field.fill(self.username)
        pass_field.fill(self.password)
        login_btn = page.locator(_SEL_LOGIN_BTN).first
        login_btn.click()

        # Poll for redirect away from /login or logged-in indicators (up to 30s)
        t_start = time.time()
        logged_in = False
        while time.time() - t_start < 30.0:
            cur_url = page.url or ""
            # Redirected away from the login page
            if "/login" not in cur_url:
                logged_in = True
                break
            # Or user avatar / profile menu has appeared
            try:
                if page.locator(".nav-menu-profile, .user-avatar, .home-username, a[href*='/member/'], .avatar-component").first.is_visible(timeout=300):
                    logged_in = True
                    break
            except Exception:
                pass
            time.sleep(1.0)

        if logged_in:
            logger.info("Login successful. Current URL: %s", page.url)
        else:
            logger.warning("Login redirect wait finished. Proceeding with current URL: %s", page.url)

    def _navigate_to_computer(self) -> None:
        page = self.page
        logger.info("Navigating to Play vs Computer...")
        page.goto("https://www.chess.com/play/computer", wait_until="domcontentloaded")
        time.sleep(2.0)  # Let dynamic content load

    def _select_bot_and_start(
        self,
        color: Literal["white", "black", "random"],
        bot_name: str | None,
    ) -> None:
        page = self.page
        
        # 1. Dismiss introductory modal if present (e.g. "Play the bots" -> "Start")
        try:
            start_modal_btn = page.locator("button:has-text('Start'), .modal-first-time-button").first
            if start_modal_btn.is_visible(timeout=3000):
                logger.info("Dismissing introductory 'Start' modal...")
                start_modal_btn.click()
                time.sleep(1.0)
        except Exception:
            pass

        # 2. Try to find and click the desired bot tile (if specified)
        if bot_name:
            logger.info("Selecting bot: %s", bot_name)
            tile = page.locator(f"[data-bot-name='{bot_name}'], [title='{bot_name}']").first
            if tile.is_visible(timeout=3000):
                tile.scroll_into_view_if_needed()
                tile.click()
                time.sleep(1.0)
            else:
                logger.warning("Bot '%s' not found, playing currently selected bot.", bot_name)

        # 3. Set color if there's a color picker
        if color != "random":
            color_btn_sel = f"button[aria-label*='{color.capitalize()}'], .selection-button-{color}, [data-color='{color}']"
            try:
                btn = page.locator(color_btn_sel).first
                if btn.is_visible(timeout=2000):
                    btn.click()
                    time.sleep(0.5)
            except Exception:
                logger.debug("Color picker not found, continuing with default color.")

        # 4. Click the Play button
        clicked = False
        cta_btn = page.locator("button.bot-selection-cta-button-button").first
        if cta_btn.is_visible(timeout=2500):
            cta_btn.scroll_into_view_if_needed()
            cta_btn.click()
            logger.info("Clicked Play CTA button.")
            clicked = True
        else:
            # Iterate through all buttons to find the visible "Play" button (avoids hidden navbar links)
            for btn in page.locator("button:has-text('Play')").all():
                try:
                    if btn.is_visible():
                        btn.scroll_into_view_if_needed()
                        btn.click()
                        logger.info("Clicked visible Play button: %s", btn.get_attribute("class"))
                        clicked = True
                        break
                except Exception:
                    continue

        if not clicked:
            # Check if there is a "Choose" button first
            choose_btn = page.locator("button:has-text('Choose')").first
            if choose_btn.is_visible(timeout=2000):
                choose_btn.click()
                time.sleep(1.0)
                cta2 = page.locator("button.bot-selection-cta-button-button, button:has-text('Play')").first
                if cta2.is_visible(timeout=3000):
                    cta2.click()
                    logger.info("Clicked Play button after Choose.")
                    clicked = True

        if not clicked:
            logger.warning("No visible Play button clicked -- game may have auto-started.")

        time.sleep(2.5)

    def _wait_for_board(self) -> None:
        self.page.wait_for_selector(_SEL_BOARD, timeout=15_000)
