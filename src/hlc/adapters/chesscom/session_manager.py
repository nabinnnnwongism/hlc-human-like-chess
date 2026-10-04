"""session_manager.py — Chess.com session cookie persistence for HLC CDP profile.

Because HLC uses a dedicated browser profile (to avoid Chromium's SingletonLock
on the user's real profile), we need to persist the user's chess.com login
cookies across restarts.  This module:

  1. Saves cookies from a connected Playwright BrowserContext to disk.
  2. Loads them back and injects them before the page is loaded, so the user
     appears already logged-in without doing anything manually every time.

Session file: ~/.hlc/session.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_SESSION_PATH = Path.home() / ".hlc" / "chesscom_session.json"

# Domains whose cookies we care about
_CHESS_DOMAINS = {"chess.com", ".chess.com", "www.chess.com"}


# ── Public API ─────────────────────────────────────────────────────────────────

def save_session(cookies: list[dict]) -> None:
    """Persist chess.com cookies from a Playwright cookie list to disk.

    Call this after the user is confirmed to be logged in so the next
    launch re-uses their session automatically.

    Args:
        cookies: The list of dicts returned by ``BrowserContext.cookies()``.
    """
    _SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    chess_cookies = [
        c for c in cookies
        if any(d in c.get("domain", "") for d in _CHESS_DOMAINS)
    ]
    with open(_SESSION_PATH, "w", encoding="utf-8") as fh:
        json.dump({"cookies": chess_cookies}, fh, indent=2)
    logger.info("Session saved — %d chess.com cookies stored at %s",
                len(chess_cookies), _SESSION_PATH)


def load_session() -> list[dict] | None:
    """Load previously saved chess.com cookies from disk.

    Returns:
        A list of cookie dicts ready for ``BrowserContext.add_cookies()``,
        or ``None`` if no valid session file exists.
    """
    if not _SESSION_PATH.exists():
        logger.debug("No saved session found at %s", _SESSION_PATH)
        return None
    try:
        with open(_SESSION_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        cookies: list[dict] = data.get("cookies", [])
        if not cookies:
            return None
        logger.info("Session loaded — %d chess.com cookies from %s",
                    len(cookies), _SESSION_PATH)
        return cookies
    except Exception as exc:
        logger.warning("Could not load session file: %s", exc)
        return None


def clear_session() -> None:
    """Delete the saved session (forces re-login on next launch)."""
    if _SESSION_PATH.exists():
        _SESSION_PATH.unlink()
        logger.info("Session cleared (deleted %s)", _SESSION_PATH)


def has_session() -> bool:
    """Return True if a session file exists on disk."""
    return _SESSION_PATH.exists()
