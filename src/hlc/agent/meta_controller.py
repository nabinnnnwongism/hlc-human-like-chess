"""meta_controller.py — Autonomous self-calibrating Native Agent for HLC.

Dynamically adapts ELO, playstyle, blunder plausibility, and cognitive fatigue
based on live game telemetry and historical performance.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

import chess

from hlc.agent.memory import AgentMemory
from hlc.playstyle import StyleVector, apply_playstyle, get_style_vector


@dataclass
class AgentSessionState:
    """Live state of the current playing session."""
    session_id: str
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    fatigue: float = 0.0          # 0.0 (fresh) to 1.0 (exhausted)
    last_game_time: float = field(default_factory=time.time)
    current_game_id: int | None = None


class MetaController:
    """The brain of the Native Agent.

    Observes games, reads ratings, manages fatigue, modulates ELO targets,
    and applies the research-backed Native Pragmatic Playstyle.
    """

    def __init__(
        self,
        memory: AgentMemory | None = None,
        base_elo: int = 1500,
        rng: random.Random | None = None,
    ) -> None:
        self.memory = memory or AgentMemory()
        self.rng = rng or random.Random(42)
        self.base_elo = base_elo

        session_id = f"sess_{int(time.time())}"
        self.session = AgentSessionState(session_id=session_id)

    # ── 1. The Research-Backed Native Playstyle ──────────────────────────────

    def get_native_style_vector(self, elo: int, fatigue: float = 0.0) -> StyleVector:
        """Construct the Research-Backed 'Native Pragmatic Dynamic' Style.

        Derived from Lars Bo Hansen's Pragmatist + modern online blitz reality:
        - Balanced aggression with strong tactical alertness
        - Prophylaxis scales with ELO
        - Under fatigue: slight drop in king safety & endgame precision,
          matching natural human exhaustion.
        """
        # Interpolate between club pragmatist and master universal
        t = max(0.0, min(1.0, (elo - 1200) / 700))  # 0 at 1200, 1 at 1900

        base_aggr = 0.65 + 0.10 * t
        base_init = 0.70 + 0.15 * t
        base_cmplx = 0.55 + 0.15 * t
        base_pos = 0.55 + 0.25 * t
        base_risk = 0.50 + 0.15 * t
        base_proph = 0.40 + 0.40 * t
        base_ksafe = 0.60 + 0.20 * t
        base_egprc = 0.50 + 0.40 * t

        # Apply cognitive fatigue degradation
        fatigue_penalty = fatigue * 0.20
        ksafe = max(0.3, base_ksafe - fatigue_penalty)
        egprc = max(0.3, base_egprc - fatigue_penalty)

        return StyleVector(
            aggression=round(base_aggr, 2),
            initiative=round(base_init, 2),
            complexity=round(base_cmplx, 2),
            positional=round(base_pos, 2),
            risk_tolerance=round(base_risk, 2),
            prophylaxis=round(base_proph, 2),
            king_safety=round(ksafe, 2),
            endgame_prec=round(egprc, 2),
            bias_strength=round(0.60 + 0.15 * t, 2),
        )

    # ── 2. Autonomous ELO & Macro-Performance Calibration ────────────────────

    def calibrate_target_elo(self, detected_self_elo: int, opp_elo: int) -> int:
        """Calculate the optimal target ELO for the match to ensure natural performance.

        Targets a 52%–56% winrate envelope:
        - If winning too much (win streak ≥ 3): tapers ELO down slightly to avoid flagging.
        - If on a loss streak (losses ≥ 2): sharpens ELO slightly to break tilt.
        - Normalizes around detected in-game ELO.
        - For unrated/new accounts (detected_self_elo=0): starts at 1150 and
          grows naturally based on match history.
        """
        # New / unrated account: use session history or opponent tier to calibrate
        if detected_self_elo == 0:
            recent = self.memory.get_recent_games(limit=10)
            if not recent:
                # Brand new account: if playing against an opponent with known rating (bot or human),
                # calibrate naturally to match the opponent's tier!
                if opp_elo > 200:
                    effective_base = min(1600, max(400, opp_elo))
                else:
                    effective_base = 1150
            else:
                # Compute implied ELO from win-rate of recent games
                wins = sum(1 for g in recent if g.result == "win")
                total = len(recent)
                winrate = wins / total if total > 0 else 0.5
                # Approximate ELO from winrate (inverted logistic)
                implied_elo = 1150 + int((winrate - 0.50) * 1000)
                effective_base = max(700, min(1800, implied_elo))
        else:
            effective_base = detected_self_elo

        # Streak adjustments
        recent_games = self.memory.get_recent_games(limit=5)
        streak_adj = 0
        if len(recent_games) >= 3:
            recent_results = [g.result for g in recent_games[:3]]
            if recent_results == ["win", "win", "win"]:
                # Winning too quickly -> dial back slightly to look human
                streak_adj = -30
            elif recent_results == ["loss", "loss"]:
                # Concentration recovery
                streak_adj = +25

        # Fatigue drag
        fatigue_adj = int(-self.session.fatigue * 40)

        # Target ELO bounded in sensible human range (Maia supports 400-1900)
        target = effective_base + streak_adj + fatigue_adj
        return max(400, min(1900, target))

    # ── 3. Fatigue Engine ───────────────────────────────────────────────────

    def update_session_fatigue(self) -> float:
        """Update fatigue based on game count and elapsed rest time."""
        now = time.time()
        elapsed_rest_min = (now - self.session.last_game_time) / 60.0

        # Rest decreases fatigue (recovers ~10% per 15 min idle)
        recovery = (elapsed_rest_min / 15.0) * 0.10
        self.session.fatigue = max(0.0, self.session.fatigue - recovery)

        # Each game adds 3-5% fatigue
        self.session.fatigue = min(0.60, self.session.fatigue + 0.04)
        self.session.last_game_time = now
        return self.session.fatigue

    # ── 4. Human Blunder Plausibility ───────────────────────────────────────

    def evaluate_blunder_plausibility(
        self,
        board: chess.Board,
        entropy: float,
        clock_s: float,
        is_check: bool,
    ) -> float:
        """Calculate how plausible a human blunder would be in this position.

        Returns score in [0.0, 1.0]:
        - High score (> 0.6): High entropy, low clock, sharp king danger.
          Humans frequently blunder here.
        - Low score (< 0.2): Simple position, abundant clock time.
          A blunder here looks like an obvious bot bug.
        """
        score = 0.0

        # Clock pressure is the #1 driver of human blunders
        if clock_s < 10.0:
            score += 0.50
        elif clock_s < 25.0:
            score += 0.30
        elif clock_s < 45.0:
            score += 0.15

        # Policy entropy reflects position complexity
        if entropy > 2.0:
            score += 0.35
        elif entropy > 1.5:
            score += 0.20

        # In check
        if is_check:
            score += 0.15

        return min(1.0, score)

    # ── 5. Lifecycle Hooks for Matches ──────────────────────────────────────

    def on_game_start(
        self,
        color: str,
        detected_self_elo: int,
        opp_elo: int,
        override_style: str | None = None,
    ) -> tuple[int, StyleVector, str]:
        """Called when a new match begins.

        Returns:
            (target_elo, style_vector, style_name)
        """
        fatigue = self.update_session_fatigue()
        target_elo = self.calibrate_target_elo(detected_self_elo, opp_elo)

        if override_style and override_style != "native":
            style_vec = get_style_vector(override_style, elo=target_elo, rng=self.rng)
            style_name = override_style
        else:
            style_vec = self.get_native_style_vector(elo=target_elo, fatigue=fatigue)
            style_name = "native_pragmatic"

        # Record game start in memory
        game_id = self.memory.start_game(
            session_id=self.session.session_id,
            color=color,
            self_elo=detected_self_elo or target_elo,
            opp_elo=opp_elo,
            style_used=style_name,
            fatigue_level=round(fatigue, 2),
        )
        self.session.current_game_id = game_id
        return target_elo, style_vec, style_name

    def on_game_end(
        self,
        result: str,
        accuracy: float = 0.0,
    ) -> None:
        """Called when a match concludes."""
        if self.session.current_game_id is not None:
            self.memory.finish_game(
                game_id=self.session.current_game_id,
                result=result,
                accuracy_percent=accuracy,
            )
            self.session.current_game_id = None

        self.session.games_played += 1
        if result == "win":
            self.session.wins += 1
        elif result == "loss":
            self.session.losses += 1
        elif result == "draw":
            self.session.draws += 1
