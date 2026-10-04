"""lichess_bot_hook.py — HLC homemade engine for lichess-bot.

INSTALLATION:
1. Clone lichess-bot:
       git clone https://github.com/lichess-bot-devs/lichess-bot.git lichess-bot
2. Install HLC into the lichess-bot venv:
       cd lichess-bot
       pip install -e <path-to-hlc-project>
3. Copy (or symlink) this file to lichess-bot/homemade.py
   (overwrite or append the HLCEngine class to the existing homemade.py).
4. Edit lichess-bot/config.yml:
       engine:
         protocol: homemade
         name: HLCEngine
         fake_think_time: false  # ← MUST be false — HLC controls its own timing
         homemade_options:
           model: maia3-79m
           elo: 1500
           device: cpu
           temperature: "1.0"
           top_p: "1.0"
           seed: "42"
           timing_backend: heuristic
           opp_elo: "1500"
5. Run: python lichess-bot.py -v
"""

from __future__ import annotations

import logging

import chess
import chess.engine

logger = logging.getLogger(__name__)

# Guard: Only import MinimalEngine when running inside lichess-bot
try:
    from lib.engine_wrapper import MinimalEngine  # type: ignore[import]
    from lib.lichess_types import HOMEMADE_ARGS_TYPE, MOVE  # type: ignore[import]

    _LICHESS_BOT_AVAILABLE = True
except ImportError:
    # Running in standalone / test mode outside lichess-bot
    _LICHESS_BOT_AVAILABLE = False
    MinimalEngine = object  # type: ignore[assignment, misc]

from hlc.adapters.lichess_bot_engine import HLCEngine as _HLCCore


class HLCEngine(MinimalEngine):  # type: ignore[misc]
    """Human-Like Chess homemade engine for lichess-bot.

    Subclasses MinimalEngine (lichess-bot) and delegates all move logic
    to HLCEngine (hlc.adapters.lichess_bot_engine).

    Config keys accepted in `homemade_options` (all optional, with defaults):
        model          : str  = "maia3-79m"
        elo            : int  = 1500
        device         : str  = "cpu"
        temperature    : float = 1.0
        top_p          : float = 1.0
        seed           : int  = 42
        move_overhead  : float = 0.10
        safety_margin  : float = 0.50
        max_clock_frac : float = 0.12
        min_delay      : float = 0.05
        single_move_delay: float = 0.15
        timing_backend : str  = "heuristic"   # "heuristic" | "clock_only"
        opp_elo        : int  = 1500
    """

    def __init__(self, commands, options, stderr, draw_or_resign, game=None, debug=False, **kwargs):
        super().__init__(commands, options, stderr, draw_or_resign, game, debug, **kwargs)

        # Parse homemade_options from the lichess-bot config
        ho = options  # options dict passed by lichess-bot's create_engine()

        def _int(key: str, default: int) -> int:
            return int(ho.get(key, default))

        def _float(key: str, default: float) -> float:
            return float(ho.get(key, default))

        def _str(key: str, default: str) -> str:
            return str(ho.get(key, default))

        self._hlc = _HLCCore(
            model=_str("model", "maia3-79m"),
            elo=_int("elo", 1500),
            device=_str("device", "cpu"),
            temperature=_float("temperature", 1.0),
            top_p=_float("top_p", 1.0),
            seed=_int("seed", 42),
            move_overhead=_float("move_overhead", 0.10),
            safety_margin=_float("safety_margin", 0.50),
            max_clock_fraction=_float("max_clock_frac", 0.12),
            min_delay=_float("min_delay", 0.05),
            single_move_delay=_float("single_move_delay", 0.15),
            timing_backend=_str("timing_backend", "heuristic"),
            opp_elo=_int("opp_elo", 1500),
        )
        logger.info("HLCEngine (lichess-bot hook) ready.")

    def search(
        self,
        board: chess.Board,
        time_limit: chess.engine.Limit,
        ponder: bool,
        draw_offered: bool,
        root_moves: MOVE,
    ) -> chess.engine.PlayResult:
        """Delegate to HLCCore.search(); root_moves is ignored (Maia-3 owns move selection)."""
        return self._hlc.search(board, time_limit, ponder, draw_offered)
