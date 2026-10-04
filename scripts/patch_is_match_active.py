"""patch_is_match_active.py - Replaces is_match_active() in browser.py with a robust Playwright-locator version."""

NEW_METHOD = r'''    def is_match_active(self) -> bool:
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

'''

import re

fpath = r"src\hlc\adapters\chesscom\browser.py"
with open(fpath, "r", encoding="utf-8") as f:
    content = f.read()

start = content.find("    def is_match_active(self) -> bool:")
end   = content.find("\n    def is_game_over(self)")

if start == -1 or end == -1:
    print(f"ERROR: markers not found. start={start}, end={end}")
else:
    new_content = content[:start] + NEW_METHOD + content[end + 1:]
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"OK: patched browser.py ({len(content)} -> {len(new_content)} bytes)")
    print(f"   is_match_active starts at byte {start}")
    print(f"   is_game_over starts at original byte {end}")
