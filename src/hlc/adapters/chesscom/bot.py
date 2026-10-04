"""bot.py — Main game-loop orchestrator for chess.com automation.

Connects:
  BoardReader      (DOM → FEN)
  HLC Engine       (BotCore: Maia-3 move + human timing delay)
  MoveExecutor     (UCI move → mouse clicks)
  IdleCursorDaemon (background human-like cursor movement between moves)

Game loop:
  1. Wait until it's HLC's turn (detect from DOM whose turn it is).
  2. Read the board from DOM.
  3. Call HLC engine → Decision(move, delay_s).
  4. Wait delay_s — idle cursor drifts naturally around the board.
  5. Idle cursor pauses → execute move with mouse clicks.
  6. Idle cursor resumes → repeat until game over.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Literal

import chess
from playwright.sync_api import Page

from hlc.adapters.chesscom.board_reader import BoardReader
from hlc.adapters.chesscom.digital_clock import DigitalClock
from hlc.adapters.chesscom.idle_cursor import IdleCursorDaemon
from hlc.adapters.chesscom.move_executor import MoveExecutor
from hlc.adapters.lichess_bot_engine import _build_bot_core
from hlc.agent.meta_controller import MetaController
from hlc.playstyle import describe_style
from hlc.types import GameState

logger = logging.getLogger(__name__)

# ── DOM selectors for turn / game-over detection ──────────────────────────────
_SEL_CLOCK_BOTTOM_ACTIVE = ".clock-bottom.clock-player-turn, .clock-bottom.active"
_SEL_CLOCK_TOP_ACTIVE    = ".clock-top.clock-player-turn, .clock-top.active"
_SEL_GAME_OVER           = ".game-over-modal-content, .modal-game-over, .game-over-dialog-content, .board-modal-container"


class ChessDotComBot:
    """Full game loop: reads the board, picks moves with HLC, executes them.

    Usage:
        bot = ChessDotComBot(
            page=browser.page,
            player_color=chess.WHITE,
            elo=1500,
            playstyle="rising_fire",
        )
        bot.run()
    """

    def __init__(
        self,
        page: Page,
        player_color: chess.Color = chess.WHITE,
        elo: int = 1500,
        opp_elo: int = 1500,
        model: str = "maia3-79m",
        device: str = "cpu",
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int = 42,
        move_overhead: float = 0.10,
        safety_margin: float = 0.50,
        max_clock_fraction: float = 0.12,
        min_delay: float = 0.5,
        single_move_delay: float = 0.8,
        timing_backend: str = "heuristic",
        poll_interval_s: float = 0.3,
        enable_idle_cursor: bool = True,
        playstyle: str | None = "rising_fire",
        meta_controller: MetaController | None = None,
        time_control: str | None = None,
        digital_clock: DigitalClock | None = None,
    ) -> None:
        """Initialise the bot.

        Args:
            page:               Active Playwright Page (game must already be started).
            player_color:       chess.WHITE or chess.BLACK — which side HLC plays.
            elo:                HLC's simulated ELO (affects Maia-3 move sampling).
            opp_elo:            Opponent's ELO (affects timing model).
            model:              Maia-3 model name ("maia3-79m").
            device:             Torch device ("cpu" or "cuda").
            temperature:        Sampling temperature for move selection.
            top_p:              Nucleus sampling threshold.
            seed:               RNG seed for reproducibility.
            move_overhead:      Fraction of time to reserve for network/click overhead.
            safety_margin:      Hard safety margin on the clock (seconds).
            max_clock_fraction: Maximum fraction of remaining clock to spend thinking.
            min_delay:          Minimum think delay in seconds.
            single_move_delay:  Think delay when only one legal move exists.
            timing_backend:     "heuristic" or "clock_only".
            poll_interval_s:    How often to poll the DOM to detect our turn (seconds).
            enable_idle_cursor: Run background idle cursor movement between moves.
            playstyle:          Playstyle name (e.g. 'rising_fire', 'tal', 'wildcard').
            meta_controller:    Native Agent meta-controller for autonomous learning.
            time_control:       Hint string for time control ("rapid", "blitz", "bullet", "10m").
            digital_clock:      Optional external DigitalClock instance to reuse.
        """
        self._page = page
        self._player_color = player_color
        self._poll_s = poll_interval_s
        self._enable_idle_cursor = enable_idle_cursor
        self._playstyle = playstyle
        self._meta_controller = meta_controller
        self._time_control = time_control

        self._board_reader = BoardReader(page)
        self._move_executor = MoveExecutor(page, player_color=player_color)

        # ── Digital Clock System ──────────────────────────────────────────────
        if digital_clock is not None:
            self._digital_clock = digital_clock
        else:
            w_dom, b_dom = self._board_reader.get_clocks()
            tc_hint = time_control or self._board_reader.detect_time_control_from_dom()
            self._digital_clock = DigitalClock.create(
                time_control_hint=tc_hint,
                dom_white_s=w_dom,
                dom_black_s=b_dom,
                default_seconds=600.0,
            )

        # ── Autonomous Native Agent Handshake ─────────────────────────────────
        detected_self: int = 0
        detected_opp: int = 0
        if self._meta_controller:
            detected_self, detected_opp = self._board_reader.get_player_ratings()

            if detected_self > 0:
                elo = detected_self
                logger.info("Detected self ELO from DOM: %d", elo)
            else:
                logger.info(
                    "New/unrated account detected — MetaController will assign a starting ELO."
                )

            if detected_opp > 0:
                opp_elo = detected_opp

            color_name = "white" if player_color == chess.WHITE else "black"
            target_elo, style_vec, style_name = self._meta_controller.on_game_start(
                color=color_name,
                detected_self_elo=detected_self,   # pass 0 for unrated so MetaController handles it
                opp_elo=detected_opp,
                override_style=playstyle,
            )
            elo = target_elo
            playstyle = style_name


        # Determine time control category for playstyle tuning
        resolved_tc = time_control or getattr(self._digital_clock, "time_control_name", "blitz")

        self._bot_core = _build_bot_core(
            model=model,
            elo=elo,
            device=device,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            move_overhead=move_overhead,
            safety_margin=safety_margin,
            max_clock_fraction=max_clock_fraction,
            min_delay=min_delay,
            single_move_delay=single_move_delay,
            timing_backend=timing_backend,
            playstyle=playstyle,
            time_control=resolved_tc,
        )
        self._elo = elo
        self._opp_elo = opp_elo
        self.detected_self_elo: int = detected_self
        self.detected_opp_elo: int = detected_opp
        self.target_elo: int = elo
        self.active_style_name: str = playstyle or "default"
        self._board = chess.Board()           # We maintain an internal board for accuracy
        self._move_history: list[chess.Move] = []
        self._idle_cursor: IdleCursorDaemon | None = None

        style_desc = ""
        if self._bot_core.style_vector and playstyle:
            style_desc = f" | Style: {describe_style(playstyle, self._bot_core.style_vector)}"

        logger.info(
            "ChessDotComBot ready: color=%s elo=%d model=%s timing=%s clock=[%s]%s",
            "WHITE" if player_color == chess.WHITE else "BLACK",
            elo, model, timing_backend, self._digital_clock.status_str(), style_desc,
        )

    def new_game(self) -> None:
        """Reset internal board state, move history, and BotCore per-game jitter."""
        self._board.reset()
        self._move_history.clear()
        self._bot_core.new_game()

    # ── Public API ─────────────────────────────────────────────────────────────
    def run(self, max_moves: int = 200) -> None:
        """Main game loop. Runs until the game is over or max_moves reached.

        Args:
            max_moves: Safety limit to prevent infinite loops.
        """
        logger.info("Game loop started.")
        self._bot_core.new_game()
        move_num = 0

        opp_color = chess.BLACK if self._player_color == chess.WHITE else chess.WHITE
        if self._player_color == chess.WHITE:
            logger.info("Playing as WHITE. We move first! Initial clock: %s", self._digital_clock.status_str())
            self._digital_clock.start_turn(chess.WHITE)
        else:
            logger.info("Playing as BLACK. Awaiting opponent's (White) opening move... Initial clock: %s", self._digital_clock.status_str())
            self._digital_clock.start_turn(chess.WHITE)

        # ── Start idle cursor daemon ───────────────────────────────────────────
        if self._enable_idle_cursor:
            try:
                board_el = self._page.locator("wc-chess-board, chess-board").first
                bbox = board_el.bounding_box()
            except Exception:
                bbox = None
            self._idle_cursor = IdleCursorDaemon(
                page=self._page,
                board_bbox=bbox,
                viewport_width=1280,
                viewport_height=800,
            )
            self._idle_cursor.start()
            logger.info("Idle cursor daemon active.")

        try:
            while move_num < max_moves:
                # Check game over first
                if self._is_game_over():
                    logger.info("Game over detected. Stopping.")
                    break

                # Wait for our turn (idle cursor drifts during this wait)
                if not self._is_our_turn():
                    time.sleep(self._poll_s)
                    continue

                # Double-check: python-chess board must confirm it's our color's turn
                if self._board.turn != self._player_color:
                    time.sleep(self._poll_s)
                    continue

                logger.info("Our turn (move %d).", move_num + 1)
                self._digital_clock.start_turn(self._player_color)

                # ── DOM board reconciliation (CRITICAL: prevents desynced moves) ──
                # Always verify internal board against real DOM pieces before deciding.
                # This catches any misread opponent move (castling, en passant, promo)
                # that could cause Maia-3 to play from the wrong position.
                try:
                    dom_pieces = self._board_reader.extract_pieces()
                    if dom_pieces and len(dom_pieces) >= 2:
                        internal_map = self._board.piece_map()
                        if internal_map != dom_pieces:
                            diff_count = len(
                                set(internal_map.items()) ^ set(dom_pieces.items())
                            ) // 2
                            logger.warning(
                                "Board desync detected (%d piece(s) differ) — correcting from DOM.",
                                diff_count,
                            )
                            # Build a fresh board from DOM pieces preserving turn
                            fresh = chess.Board()
                            fresh.clear()
                            for sq, pc in dom_pieces.items():
                                fresh.set_piece_at(sq, pc)
                            # Preserve whose turn it is
                            fresh.turn = self._player_color
                            self._board = fresh
                except Exception as _sync_err:
                    logger.debug("Board reconciliation skipped: %s", _sync_err)

                # Sync with DOM clocks if visible
                w_dom, b_dom = self._board_reader.get_clocks()
                if w_dom > 0.0 and b_dom > 0.0:
                    self._digital_clock.sync_from_dom(w_dom, b_dom)

                clock_self, clock_opp = self._digital_clock.get_clocks_for(self._player_color)

                # Build GameState from reconciled board
                state = GameState(
                    board=self._board.copy(),
                    move_history=list(self._move_history),
                    clock_self=max(1.0, clock_self),
                    clock_opp=max(1.0, clock_opp),
                    increment=self._digital_clock.increment_s,
                    self_elo=self._elo,
                    opp_elo=self._opp_elo,
                )

                # Ask HLC engine for a move + timing delay
                t0 = time.perf_counter()
                try:
                    decision = self._bot_core.decide(state)
                except Exception as decide_err:
                    # "No legal moves" or any engine error — treat as game over
                    err_msg = str(decide_err)
                    if "no legal moves" in err_msg.lower() or self._board.is_game_over():
                        logger.info("Game over (no legal moves / engine signal). Stopping.")
                    else:
                        logger.warning("Engine error on move %d: %s -- stopping game.", move_num + 1, decide_err)
                    break
                compute_elapsed = time.perf_counter() - t0


                # Safety: verify the chosen move is legal and moves OUR piece
                if decision.move not in self._board.legal_moves:
                    logger.warning("Illegal move %s -- skipping turn.", decision.move.uci())
                    time.sleep(self._poll_s)
                    continue
                from_sq = decision.move.from_square
                piece_at_src = self._board.piece_at(from_sq)
                if piece_at_src is None or piece_at_src.color != self._player_color:
                    logger.warning(
                        "Move %s would touch a %s piece (we are %s) — skipping.",
                        decision.move.uci(),
                        "WHITE" if (piece_at_src and piece_at_src.color) else "EMPTY",
                        "WHITE" if self._player_color == chess.WHITE else "BLACK",
                    )
                    time.sleep(self._poll_s)
                    continue

                style_tag = f"  style={decision.debug.get('playstyle')}" if decision.debug.get("playstyle") else ""
                guard_tag = f"  guard={decision.debug.get('guard_reason')}" if decision.debug.get("guard_reason") else ""
                logger.info(
                    "HLC decision: %s  delay=%.2fs  [%s]%s%s  entropy=%.2f  top_p=%.2f",
                    decision.move.uci(),
                    decision.delay_s,
                    self._digital_clock.status_str(),
                    style_tag,
                    guard_tag,
                    decision.debug.get("policy_entropy", 0.0),
                    decision.debug.get("top_probability", 0.0),
                )
                # Emit a machine-parseable clock line for the GUI dashboard
                print(f"[CLOCK] {self._digital_clock.status_str()}", flush=True)

                # Wait the scheduled delay — idle cursor drifts during this
                if decision.delay_s > 0.0:
                    time.sleep(decision.delay_s)

                # ── Pause idle cursor → execute move → resume idle cursor ───────
                if self._idle_cursor:
                    with self._idle_cursor.paused():
                        self._execute_move(decision.move)
                else:
                    self._execute_move(decision.move)

                # Deduct clock time for our turn
                self._digital_clock.end_turn(self._player_color, elapsed_s=decision.delay_s)

                # Update internal board
                san_move = self._board.san(decision.move)
                fen_before = self._board.fen()
                self._board.push(decision.move)
                self._move_history.append(decision.move)

                # Start opponent clock for their upcoming turn
                self._digital_clock.start_turn(opp_color)

                # ── Record move telemetry in Native Agent Memory ───────────────
                if self._meta_controller and self._meta_controller.session.current_game_id:
                    self._meta_controller.memory.record_move(
                        game_id=self._meta_controller.session.current_game_id,
                        ply=move_num,
                        fen_before=fen_before,
                        move_uci=decision.move.uci(),
                        move_san=san_move,
                        think_time_s=decision.delay_s,
                        policy_entropy=decision.debug.get("policy_entropy", 0.0),
                        top_move_prob=decision.debug.get("top_probability", 0.0),
                        was_top_move=(decision.move.uci() == decision.debug.get("top_move")),
                        clock_remaining_s=clock_self,
                    )

                move_num += 1
                time.sleep(self._poll_s)  # Brief pause before next poll

        finally:
            # Finalize game with Native Agent
            if self._meta_controller and self._meta_controller.session.current_game_id:
                self._meta_controller.on_game_end(result="completed")

            # Always stop the daemon when the game loop exits
            if self._idle_cursor:
                self._idle_cursor.stop()
                self._idle_cursor = None

        logger.info("Game loop ended after %d moves.", move_num)

    def _execute_move(self, move: chess.Move) -> None:
        """Execute a chess move and verify it registered on the board."""
        self._move_executor.execute(move)

        # Verification: fast check that move registered (src vacated or dst reached)
        verified = False
        for _ in range(4):
            time.sleep(0.06)
            try:
                dom_pieces = self._board_reader.extract_pieces()
                src_vacated = (
                    move.from_square not in dom_pieces
                    or dom_pieces[move.from_square].color != self._player_color
                )
                dst_arrived = (
                    move.to_square in dom_pieces
                    and dom_pieces[move.to_square].color == self._player_color
                )
                if src_vacated or dst_arrived:
                    verified = True
                    break
            except Exception:
                pass

        if not verified:
            logger.warning("Move %s not registered yet -- retrying click.", move.uci())
            self._move_executor.execute(move)
            time.sleep(0.10)

    # ── Private ─────────────────────────────────────────────────────────────────
    def _detect_opponent_move(self) -> chess.Move | None:
        """Detect what move the opponent made by matching highlights and DOM pieces.

        Only called when self._board.turn == OPPONENT color.
        We iterate over the opponent's legal moves, not ours.
        """
        if self._board.turn == self._player_color:
            return None

        try:
            dom_pieces = self._board_reader.extract_pieces()
        except Exception:
            return None

        if not dom_pieces or len(dom_pieces) < 2:
            return None

        # 1. Fast path: check DOM highlights (chess.com highlights the move's src and dst)
        try:
            highlights = self._board_reader.get_highlight_squares()
            if len(highlights) == 2:
                sq_a, sq_b = highlights[0], highlights[1]
                candidates = [
                    m for m in self._board.legal_moves
                    if (m.from_square == sq_a and m.to_square == sq_b)
                    or (m.from_square == sq_b and m.to_square == sq_a)
                ]
                for move in candidates:
                    self._board.push(move)
                    matched = (self._board.piece_map() == dom_pieces)
                    self._board.pop()
                    if matched:
                        return move
        except Exception:
            pass

        # 2. Comprehensive fallback: test all legal moves
        for move in self._board.legal_moves:
            self._board.push(move)
            matched = (self._board.piece_map() == dom_pieces)
            self._board.pop()
            if matched:
                return move

        return None

    def _is_our_turn(self) -> bool:
        """Return True when it is HLC's turn to move."""
        # 1. When HLC is White on move 1, ensure the DOM board is actually loaded and ready
        if len(self._move_history) == 0 and self._player_color == chess.WHITE:
            try:
                dom_pieces = self._board_reader.extract_pieces()
                if len(dom_pieces) < 30:
                    return False
            except Exception:
                return False
            return True

        # 2. Internal board already says it's our turn — go immediately
        if self._board.turn == self._player_color:
            return True

        # 3. It's the opponent's turn on internal board — check if they played yet
        opp_move = self._detect_opponent_move()
        if opp_move is not None:
            logger.info("Opponent played: %s", opp_move.uci())
            opp_color = chess.BLACK if self._player_color == chess.WHITE else chess.WHITE
            self._digital_clock.end_turn(opp_color)
            self._board.push(opp_move)
            self._move_history.append(opp_move)
            # After pushing opponent move, it's now our turn
            return True

        return False

    def _is_game_over(self) -> bool:
        """Return True if the game has ended."""
        # 1. Internal python-chess board detects checkmate/stalemate/etc.
        if self._board.is_game_over():
            return True

        # 2. Check the DOM for post-game dialog or result buttons
        page = self._page
        try:
            for sel in [
                "button:has-text('Game Review')",
                "button:has-text('Rematch')",
                "button:has-text('New Game')",
                "button:has-text('Play Again')",
                ".game-over-modal-content",
                ".modal-game-over",
                ".game-over-dialog-content",
                "div[data-cy='game-over-modal']",
            ]:
                loc = page.locator(sel)
                if loc.count() > 0 and loc.first.is_visible(timeout=30):
                    return True
        except Exception:
            pass

        return False

    def _reconcile_board(self, dom_board: chess.Board) -> bool:
        """Try to reconcile the DOM board with our internal board.

        We use the internal board as ground truth for move legality,
        but sync piece positions from DOM if a significant discrepancy is found.

        Returns True if reconciliation succeeded (positions match).
        """
        # Compare piece maps
        internal_pieces = self._board.piece_map()
        dom_pieces = dom_board.piece_map()

        if internal_pieces == dom_pieces:
            return True

        # Compute symmetric difference
        diff = set(internal_pieces.items()) ^ set(dom_pieces.items())
        if len(diff) <= 4:
            # Small discrepancy (≤2 pieces) — DOM is probably right, sync it
            logger.debug("Reconciling board: %d piece positions differ.", len(diff) // 2)
            for sq, piece in dom_pieces.items():
                self._board.set_piece_at(sq, piece)
            # Clear squares in internal but not DOM
            for sq in set(internal_pieces) - set(dom_pieces):
                self._board.remove_piece_at(sq)
            return True

        # Large discrepancy — completely reset to DOM state
        logger.warning(
            "Large board discrepancy (%d pieces differ). Resetting to DOM state.",
            len(diff) // 2,
        )
        self._board = dom_board
        return False
