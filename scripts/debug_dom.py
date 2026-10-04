"""
debug_dom.py - Connects to Opera CDP, dumps the DOM state, and tests
is_match_active() directly. Run while a game is open in Opera.

Usage:
    .venv\\Scripts\\python.exe scripts\\debug_dom.py
"""
import sys

# Fix Windows terminal encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main():
    from playwright.sync_api import sync_playwright

    cdp_url = "http://localhost:9222"
    print(f"Connecting to {cdp_url}...")

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(cdp_url)

        # Find chess.com page
        page = None
        all_urls = []
        for ctx in browser.contexts:
            for pg in ctx.pages:
                url = pg.url or ""
                all_urls.append(url)
                if "chess.com" in url:
                    page = pg

        if not page:
            print(f"ERROR: No chess.com tab found. Open tabs: {all_urls}")
            return

        print(f"\nFound page : {page.title()}")
        print(f"URL        : {page.url}")
        print("\n" + "=" * 60)

        # ── Full DOM snapshot ─────────────────────────────────────────────────
        result = page.evaluate("""
        () => {
            const out = {};

            // Board elements (by ID — nixec-style)
            out.board_wc              = !!document.querySelector('wc-chess-board');
            out.board_single          = !!document.querySelector('#board-single');
            out.board_computer        = !!document.querySelector('#board-play-computer');
            out.board_vs_personality  = !!document.querySelector('#board-vs-personality');

            // Piece count
            const pieces = document.querySelectorAll(
                'wc-chess-board .piece, chess-board .piece, #board-play-computer .piece'
            );
            out.piece_count = pieces.length;

            // Game-active signals
            const resign = document.querySelector(
                'button[aria-label="Resign"], button[data-cy="resign-button"], .resign-button-component'
            );
            out.resign_visible = resign ? resign.getBoundingClientRect().width > 0 : false;

            const undo = document.querySelector('button[aria-label="Undo"]');
            out.undo_visible = undo ? undo.getBoundingClientRect().width > 0 : false;

            // Look for hint by text content
            const allBtns = [...document.querySelectorAll('button')];
            const hintBtn = allBtns.find(b => (b.innerText||'').trim() === 'Show Hint');
            out.hint_visible = hintBtn ? hintBtn.getBoundingClientRect().width > 0 : false;

            const abort = document.querySelector(
                'button[aria-label="Abort"], button[data-cy="abort-button"]'
            );
            out.abort_visible = abort ? abort.getBoundingClientRect().width > 0 : false;

            // Setup signals (should be false during a game)
            const playBtn = document.querySelector(
                'button.selection-sidebar-play-button, button.bot-selection-cta-button-button'
            );
            out.setup_play_button = playBtn ? playBtn.getBoundingClientRect().width > 0 : false;
            if (playBtn) out.setup_play_text = (playBtn.innerText || '').trim();

            // Post-game signals
            const rematch = document.querySelector('button:contains, .game-over-buttons-component');
            const newGameBtns = allBtns.filter(b => {
                const t = (b.innerText||'').toLowerCase();
                return b.getBoundingClientRect().width > 0 && (t === 'new game' || t === 'rematch');
            });
            out.game_over_buttons = newGameBtns.map(b => b.innerText.trim());

            // Move list
            const moveNodes = document.querySelectorAll('div.node[data-node]');
            out.move_nodes = moveNodes.length;

            const scrollable = document.querySelector('.play-controller-scrollable');
            out.scrollable_visible = scrollable ? scrollable.getBoundingClientRect().width > 0 : false;

            // HLC play-click flag
            out.hlc_play_clicked = window.__hlc_play_clicked || false;

            // All visible button texts
            out.visible_buttons = allBtns
                .filter(b => b.getBoundingClientRect().width > 0)
                .slice(0, 20)
                .map(b => (b.innerText.trim() || b.getAttribute('aria-label') || '?'));

            return out;
        }
        """)

        print("\nDOM SNAPSHOT:")
        for key, val in result.items():
            print(f"  {key:30s}: {val}")

        print("\n" + "=" * 60)
        print("DIAGNOSIS:")

        def ok(msg):  print(f"  [OK] {msg}")
        def no(msg):  print(f"  [--] {msg}")
        def bad(msg): print(f"  [!!] {msg}")

        pc = result.get("piece_count", 0)
        if pc >= 20:
            ok(f"{pc} pieces on board  -> game has started")
        elif pc >= 2:
            no(f"Only {pc} pieces  -> board might be loading")
        else:
            bad("No pieces visible  -> game not started")

        if result.get("resign_visible"):
            ok("Resign button visible  -> GAME IS ACTIVE")
        else:
            no("No Resign button")

        if result.get("undo_visible"):
            ok("Undo button visible  -> GAME IS ACTIVE")
        else:
            no("No Undo button")

        if result.get("hint_visible"):
            ok("Show Hint button visible  -> GAME IS ACTIVE")
        else:
            no("No Show Hint button")

        if result.get("abort_visible"):
            ok("Abort button visible  -> COMPUTER GAME ACTIVE")
        else:
            no("No Abort button")

        if result.get("setup_play_button"):
            bad(f"SETUP SCREEN: Play button visible ('{result.get('setup_play_text')}')  -> game NOT started")
        else:
            ok("No setup Play button  -> not in setup screen")

        go_btns = result.get("game_over_buttons", [])
        if go_btns:
            bad(f"POST-GAME SCREEN: {go_btns}  -> game already OVER")
        else:
            ok("No post-game buttons  -> not in game-over screen")

        print(f"\n  Move nodes: {result.get('move_nodes', 0)}")
        print(f"  Scrollable: {result.get('scrollable_visible')}")
        print(f"  HLC flag  : {result.get('hlc_play_clicked')}")

        # ── Now test is_match_active() directly ──────────────────────────────
        print("\n" + "=" * 60)
        print("TESTING is_match_active() via our browser module...")
        try:
            import logging
            logging.basicConfig(level=logging.DEBUG, format="  %(levelname)s %(name)s: %(message)s")
            sys.path.insert(0, "src")
            from hlc.adapters.chesscom.browser import ChessDotComBrowser

            # Re-use same CDP session
            b2 = ChessDotComBrowser.connect_cdp(cdp_port=9222)
            # Manually connect to existing playwright session
            b2._playwright = p
            b2._browser = browser
            b2._page = page
            b2._cdp_mode = True

            active = b2.is_match_active()
            print(f"\n>>> is_match_active() returned: {active} <<<\n")
            if active:
                print("[OK] DETECTION WORKING! Bot will pick up the game.")
            else:
                print("[!!] Detection still failing - check DEBUG lines above for which signal failed.")
        except Exception as e:
            print(f"[ERR] Could not test is_match_active(): {e}")

        browser.close()


if __name__ == "__main__":
    main()
