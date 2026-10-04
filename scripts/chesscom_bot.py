r"""chesscom_bot.py — CLI entry point for the chess.com HLC automation.

Usage (CDP mode — attach to your existing Opera / Chrome):
    python scripts/chesscom_bot.py --cdp-port 9222 --elo 1800

Usage (normal mode — opens a new browser window):
    python scripts/chesscom_bot.py --username YOUR_USER --password YOUR_PASS

CDP mode setup (one-time):
    Launch Opera (Standard) with the remote debug flag:
        "C:\Users\Admin\AppData\Local\Programs\Opera\opera.exe" --remote-debugging-port=9222 "https://www.chess.com"
    Then run the bot with --cdp-port 9222. No password needed.

Options:
    --cdp-port    Attach to existing browser on this debug port     [e.g. 9222]
    --username    chess.com account username (normal mode only)
    --password    chess.com account password (normal mode only)
    --color       Which color HLC plays: white/black/random          [default: random]
    --elo         HLC simulated ELO                                  [default: 1500]
    --opp-elo     Opponent ELO (for timing model)                   [default: 1500]
    --playstyle   Playstyle personality archetype or evolving hybrid [default: rising_fire]
    --model       Maia-3 model name                                  [default: maia3-79m]
    --bot-name    chess.com bot name to challenge                    [default: auto-pick]
    --headless    Run browser without a window (normal mode only)    [default: False]
    --verbose     Enable debug logging                               [default: False]

Multi-game session:
    After each game ends, HLC stays alive and waits for you to start a new
    game — via Rematch, New Game, Play Online, Play vs Computer, or any other
    navigation. Press Ctrl+C in this terminal to end the session.

Examples:
    # CDP mode (recommended — uses your real Opera session):
    python scripts/chesscom_bot.py --cdp-port 9222 --elo 1800 --playstyle rising_fire

    # Normal mode (spawns new browser):
    python scripts/chesscom_bot.py --username testuser --password testpass123 --elo 1800
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import threading
import time

from typing import TYPE_CHECKING

import chess
from hlc.playstyle import PLAYSTYLE_NAMES

if TYPE_CHECKING:
    from hlc.adapters.chesscom.browser import ChessDotComBrowser


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HLC chess.com bot — plays games using Maia-3 (multi-game session).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── CDP mode (recommended) ────────────────────────────────────────────
    parser.add_argument(
        "--cdp-port", type=int, default=None, metavar="PORT",
        help="Attach to your existing Opera/Chrome running with --remote-debugging-port=PORT. "
             "No login needed.",
    )

    # ── Normal mode (spawns new browser) ──────────────────────────────────
    parser.add_argument("--username",  default=None,  help="chess.com username (normal mode)")
    parser.add_argument("--password",  default=None,  help="chess.com password (normal mode)")
    parser.add_argument("--headless",  action="store_true",      help="Run browser headlessly (normal mode)")
    parser.add_argument("--auto",      action="store_true",      help="Fully automated: auto-pick bot and auto-click Play")
    parser.add_argument("--bot-name",  default=None,             help="chess.com bot to challenge")

    # ── Shared options ────────────────────────────────────────────────────
    parser.add_argument("--color",     default="random", choices=["white", "black", "random"],
                        help="Color for HLC to play (default: random)")
    parser.add_argument("--elo",       type=int,   default=1500, help="HLC simulated ELO")
    parser.add_argument("--opp-elo",   type=int,   default=1500, help="Opponent ELO")
    parser.add_argument("--playstyle", default="rising_fire", choices=PLAYSTYLE_NAMES,
                        help=f"Playstyle personality ({', '.join(PLAYSTYLE_NAMES)}) [default: rising_fire]")
    parser.add_argument("--model",     default="maia3-79m",      help="Maia-3 model name")
    parser.add_argument("--no-pause",  action="store_true",      help="Do not pause at game end")
    parser.add_argument("--autonomous", action="store_true",     help="Enable autonomous self-calibrating MetaController")
    parser.add_argument("--time-control", default="auto",        help="Time control hint: auto, rapid, blitz, bullet, 10m, 5m, 3m, 1m [default: auto]")
    parser.add_argument("--no-idle-cursor", action="store_true", help="Disable idle cursor human movement")
    parser.add_argument("--verbose",   action="store_true",      help="Enable debug logging")

    args = parser.parse_args()

    # Validate: either --cdp-port OR --username+--password must be provided
    if args.cdp_port is None and (not args.username or not args.password):
        parser.error(
            "You must provide either:\n"
            "  --cdp-port PORT          (attach to existing browser)"
            "  --username U --password P (spawn new browser)"
        )
    return args


def _pick_color(color_arg: str) -> chess.Color:
    """Resolve color argument to chess.WHITE or chess.BLACK."""
    if color_arg == "white":
        return chess.WHITE
    elif color_arg == "black":
        return chess.BLACK
    else:
        return random.choice([chess.WHITE, chess.BLACK])


def _is_fresh_board(pieces: dict[chess.Square, chess.Piece]) -> bool:
    """Return True if pieces represent a fresh game (initial or near-initial board)."""
    if len(pieces) < 30:
        return False
    white_count = sum(1 for p in pieces.values() if p.color == chess.WHITE)
    black_count = sum(1 for p in pieces.values() if p.color == chess.BLACK)
    return white_count >= 15 and black_count >= 15


def _wait_for_next_game(
    browser: "ChessDotComBrowser",  # type: ignore[name-defined]
    last_url: str,
    logger: logging.Logger,
) -> str:
    """Block until a NEW game is detected on the browser.

    Ensures that:
    1. The previous game's post-game UI has been dismissed.
    2. We don't falsely re-trigger on the game that just finished.
    3. A new match is genuinely underway (new URL OR board reset to starting position).

    Returns the new game's URL.
    """
    from hlc.adapters.chesscom.board_reader import BoardReader

    # Reset the JS play-click flag so next game detection is clean
    try:
        browser.page.evaluate("() => { window.__hlc_play_clicked = false; }")
    except Exception:
        pass

    print("\n" + "=" * 68)
    print(" [HLC] WAITING FOR NEXT GAME...")
    print(" You can now:")
    print("   - Click Rematch to play another game immediately")
    print("   - Click New Game to go back to matchmaking")
    print("   - Navigate to Play Online or Play vs Computer")
    print("   - Play ANY way you like -- HLC will auto-detect and join!")
    print(" Press Ctrl+C in this terminal to end the session.")
    print("=" * 68 + "\n")

    reader = BoardReader(browser.page)

    # Give browser a moment to render the game-over screen if it just finished
    time.sleep(1.5)

    # Wait for the post-game screen to be dismissed first
    # (the user must click Rematch / New Game / navigate away)
    game_over_cleared = False
    t_wait = time.time()
    while time.time() - t_wait < 120:
        try:
            if not browser.is_game_over():
                game_over_cleared = True
                break
        except Exception:
            pass
        time.sleep(0.5)

    if not game_over_cleared:
        logger.warning("Game-over screen still showing after 120s -- continuing anyway.")

    # Now wait for a genuinely new game to start
    while True:
        try:
            curr_url = browser.page.url or ""
            curr_url_base = curr_url.split("?")[0]

            # Skip if game-over modal reappeared
            if browser.is_game_over():
                time.sleep(0.5)
                continue

            # Must have a board with pieces
            pieces = reader.extract_pieces()
            if len(pieces) < 20:
                time.sleep(0.5)
                continue

            # Must look like a fresh starting position
            if not _is_fresh_board(pieces):
                time.sleep(0.5)
                continue

            # Must have active game signals
            if not browser.is_match_active():
                time.sleep(0.5)
                continue

            # For same-URL situations (vs Computer rematch), the board must have reset
            # We accept same URL only if pieces truly show a starting position
            if curr_url_base == last_url.split("?")[0]:
                logger.info("Same URL detected with fresh board -- accepting as new game.")
                return curr_url_base

            # New URL — accept immediately
            return curr_url_base

        except Exception:
            pass

        time.sleep(0.5)



def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("chesscom_bot")

    # Import here so logging is configured first
    from hlc.adapters.chesscom.browser import ChessDotComBrowser
    from hlc.adapters.chesscom.bot import ChessDotComBot

    logger.info("=== HLC chess.com Bot (Multi-Game Session) ===")
    if args.cdp_port:
        logger.info("Mode    : CDP (attaching to existing browser on port %d)", args.cdp_port)
    else:
        logger.info("Mode    : Normal (spawning new browser)")
        logger.info("Account : %s", args.username)
    logger.info("Color   : %s", args.color.upper())
    logger.info("ELO     : %d  |  Opp ELO: %d", args.elo, args.opp_elo)
    try:
        from hlc.playstyle import describe_style, get_style_vector
        _sample_sv = get_style_vector(args.playstyle, elo=args.elo)
        logger.info("Style   : %s", describe_style(args.playstyle, _sample_sv))
    except Exception:
        logger.info("Style   : %s", args.playstyle)
    logger.info("Model   : %s", args.model)
    logger.info("Pacing  : %s", args.time_control)
    if args.bot_name:
        logger.info("Bot     : %s", args.bot_name)

    meta_controller = None
    if args.autonomous:
        from hlc.agent.meta_controller import MetaController
        meta_controller = MetaController()
        logger.info("Autonomous mode: MetaController active (self-calibrating ELO & playstyle)")

    game_count = 0

    try:
        # ── Choose browser mode ─────────────────────────────────────────────────
        if args.cdp_port:
            browser_ctx = ChessDotComBrowser.connect_cdp(cdp_port=args.cdp_port)
        else:
            browser_ctx = ChessDotComBrowser(
                username=args.username,
                password=args.password,
                headless=args.headless,
                slow_mo_ms=0,
            )

        with browser_ctx as browser:

            # ── First-time setup: login and navigate ───────────────────────────
            if args.cdp_port:
                # CDP mode: already in existing browser — instruct user cleanly
                print("\n" + "=" * 68)
                print(" [HLC] ATTACHED TO YOUR OPERA / CHROME BROWSER!")
                print(" [*] The bot is watching chess.com in your browser.")
                print("")
                print(" What to do:")
                print("   * If you're already in a game    -> HLC picks it up automatically")
                print("   * If you're on any chess.com page -> navigate and start a game")
                print("   * Play vs Computer, Coach, Play Online -> all detected live")
                print("")
                print(" [!] HLC will auto-detect the live match and start playing!")
                print("     Press [ENTER] in this terminal at any time to force HLC to take over immediately!")
                print("     After each game, just start another -- HLC stays alive!")
                print("     Press Ctrl+C in this terminal to end the session.")
                print("=" * 68 + "\n")

                user_pressed_enter = threading.Event()

                def _listen_for_enter() -> None:
                    try:
                        input()
                        user_pressed_enter.set()
                    except Exception:
                        pass

                t = threading.Thread(target=_listen_for_enter, daemon=True)
                t.start()

                logger.info("Watching chess.com navigation in your browser...")
                last_mode = None
                last_url = None
                _waiting_logged = False
                _no_chess_since = time.time()
                while not user_pressed_enter.is_set():
                    try:
                        active_pg = browser.get_active_chess_page()
                        raw_url = active_pg.url or ""
                        curr_url = raw_url.split("?")[0]

                        # Skip all processing if not on chess.com yet
                        if "chess.com" not in raw_url:
                            if not _waiting_logged or curr_url != last_url:
                                logger.info(
                                    "Waiting for chess.com tab... (current tab: %s)",
                                    curr_url if curr_url else "blank",
                                )
                                _waiting_logged = True
                                last_url = curr_url

                            # After 6 seconds with no chess.com tab found,
                            # navigate directly — handles Opera's CDP tab visibility quirk
                            if time.time() - _no_chess_since > 6.0:
                                logger.info(
                                    "No chess.com tab found via CDP scan. "
                                    "Navigating available page to chess.com automatically..."
                                )
                                browser.navigate_to_chess()
                                _no_chess_since = time.time()  # reset timer

                            time.sleep(0.8)
                            continue

                        # chess.com found — reset the no-chess timer
                        _no_chess_since = time.time()
                        _waiting_logged = False
                        ctx = browser.detect_page_context()
                        mode = ctx.get("mode", "unknown")

                        if mode != last_mode or curr_url != last_url:
                            last_mode = mode
                            last_url = curr_url
                            if mode == "home":
                                logger.info("chess.com Home. Navigate to Play vs Computer or Play Online.")
                            elif mode == "play_online":
                                logger.info("Play Online. Choose time control and click 'Play'.")
                            elif mode == "vs_computer":
                                logger.info("Play vs Computer. Pick bot, choose side, and click 'Play'.")
                            elif mode == "vs_coach":
                                logger.info("Play Coach. Select session and click 'Play'.")
                            elif mode in ("live_game", "online_game"):
                                logger.info("Live Game detected (%s). Joining...", curr_url)
                            else:
                                logger.info("chess.com: %s [mode: %s]", curr_url, mode)

                        if browser.is_match_active() or user_pressed_enter.is_set():
                            logger.info("Live match active! Taking over gameplay now.")
                            break
                    except Exception:
                        pass

                    time.sleep(0.4)



            elif args.auto:
                logger.info("Starting automated game vs computer...")
                browser.start_game_vs_computer(
                    color=args.color if args.color != "random" else "random",
                    bot_name=args.bot_name,
                )
            else:
                logger.info("Logging in and handing control to you...")
                browser.prepare_user_session()

                print("\n" + "=" * 68)
                print(" [HLC] BROWSER READY -- YOU ARE IN CONTROL!")
                print(" 1. In Chrome/Opera, choose vs Computer OR navigate to Play Online.")
                print(" 2. Pick your bot or time control, color, and game settings.")
                print(" 3. When you are ready, click 'Play' in chess.com!")
                print("")
                print(" [!] HLC will automatically detect the live match and start playing!")
                print("     After each game, just start another -- HLC stays alive!")
                print("     Press Ctrl+C in this terminal to end the session.")
                print("=" * 68 + "\n")

                # Wait for first game using Enter-key fallback too
                user_pressed_enter = threading.Event()

                def _listen_for_enter() -> None:
                    try:
                        input()
                        user_pressed_enter.set()
                    except Exception:
                        pass

                t = threading.Thread(target=_listen_for_enter, daemon=True)
                t.start()

                last_mode = None
                last_url = None
                while not user_pressed_enter.is_set():
                    try:
                        active_pg = browser.get_active_chess_page()
                        raw_url = active_pg.url or ""
                        curr_url = raw_url.split("?")[0]
                        ctx = browser.detect_page_context()
                        mode = ctx.get("mode", "unknown")

                        if mode != last_mode or curr_url != last_url:
                            last_mode = mode
                            last_url = curr_url
                            if mode == "home":
                                logger.info("Page detected: Chess.com Home. Navigate to Play Online or Play vs Computer.")
                            elif mode == "play_online":
                                logger.info("Page detected: Play Online. Choose time control and click 'Play'.")
                            elif mode == "vs_computer":
                                logger.info("Page detected: Play vs Computer. Pick bot, choose side, and click 'Play'.")
                            elif mode == "vs_coach":
                                logger.info("Page detected: Play Coach. Select coach and click 'Play'.")
                            elif mode in ("live_game", "online_game"):
                                logger.info("Page detected: Live Game (%s). Joining active match...", curr_url)

                        if browser.is_match_active():
                            logger.info("Match detected! Taking over gameplay now.")
                            break
                    except Exception:
                        pass
                    time.sleep(0.4)

            # ── Persistent multi-game session loop ─────────────────────────────
            while True:
                game_count += 1
                logger.info("─── Game %d starting ───", game_count)

                # Wait for the board to fully settle (pieces, clocks, orientation)
                # Give it more time to avoid misdetection
                logger.info("Waiting for board to fully load...")
                time.sleep(1.5)

                # Detect player color with retries
                detected_color = browser.get_player_color()
                player_color = chess.WHITE if detected_color == "white" else chess.BLACK
                logger.info("Game %d: Playing as %s (Elo: %d)", game_count, detected_color.upper(), args.elo)

                # Detect time control from DOM with retries
                from hlc.adapters.chesscom.board_reader import BoardReader as _BR
                _reader = _BR(browser.page)
                detected_tc = None
                for _tc_attempt in range(4):
                    detected_tc = _reader.detect_time_control_from_dom()
                    if detected_tc:
                        logger.info("Time control detected from DOM: '%s'", detected_tc)
                        break
                    logger.debug("TC detection attempt %d: no result yet, retrying...", _tc_attempt + 1)
                    time.sleep(0.5)

                if not detected_tc:
                    # Last resort: read initial clock values directly
                    w_s, b_s = _reader.get_clocks()
                    if w_s > 0 or b_s > 0:
                        init_s = max(w_s, b_s)
                        from hlc.adapters.chesscom.digital_clock import DigitalClock as _DC
                        detected_tc = _DC.classify_time_control(init_s)
                        logger.info(
                            "TC detected from clock values (W=%.0fs B=%.0fs): '%s'",
                            w_s, b_s, detected_tc,
                        )
                    else:
                        logger.warning(
                            "Could not detect time control -- defaulting to user arg '%s'. "
                            "If wrong, pass --time-control bullet/blitz/rapid explicitly.",
                            args.time_control,
                        )

                # Resolve final time_control to pass to bot
                # Priority: explicit CLI arg > DOM detection > fallback
                if args.time_control and args.time_control.lower() != "auto":
                    final_tc = args.time_control
                    logger.info("Using CLI --time-control override: '%s'", final_tc)
                else:
                    final_tc = detected_tc  # may be None, bot will then use clock values

                # Create a fresh bot instance per game (clean board/history state)
                bot = ChessDotComBot(
                    page=browser.page,
                    player_color=player_color,
                    elo=args.elo,
                    opp_elo=args.opp_elo,
                    model=args.model,
                    playstyle=args.playstyle,
                    enable_idle_cursor=not args.no_idle_cursor,
                    meta_controller=meta_controller,
                    time_control=final_tc,
                )

                logger.info("Handing off to game loop...")
                bot.run()

                logger.info("─── Game %d complete! ───", game_count)

                # ── Wait for next game ─────────────────────────────────────────
                last_url = browser.page.url or ""
                _wait_for_next_game(browser, last_url, logger)
                logger.info("New game detected! Starting game %d...", game_count + 1)

    except KeyboardInterrupt:
        logger.info("Session ended by user after %d game(s). Goodbye!", game_count)
        sys.exit(0)
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
