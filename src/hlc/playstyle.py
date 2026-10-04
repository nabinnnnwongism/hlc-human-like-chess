"""playstyle.py — Research-backed chess personality system for HLC.

Implements 16 distinct playstyles (10 grandmaster archetypes + 5 hybrids + 1
human-like club style) grounded in Version 2 research (October 2026).

Each playstyle is defined as an 8-dimensional style vector and biases Maia-3's
probability distribution BEFORE sampling — zero model changes required.

The 8 dimensions (all in [0.0, 1.0]):
  1. aggression      – desire to attack, create threats, deliver checks
  2. initiative      – forcing tempo; keeping opponent reactive
  3. complexity      – preference for complicated, tactically rich positions
  4. positional      – long-term structural play, outposts, file control
  5. risk_tolerance  – willingness to sacrifice material
  6. prophylaxis     – preventive play; cutting off opponent plans
  7. king_safety     – how much our own king safety is valued
  8. endgame_prec    – shift toward technical precision in endgames

Dynamic Features (Version 2):
  - Per-game jitter: each game's vector is slightly perturbed so no two games
    are identical.
  - In-game tilt: vector shifts based on material balance and clock pressure.
  - Time-control offsets: bullet/blitz/rapid automatically tune the vector.
  - ELO-band variants for the_club_regular to match human play at that rating.

ELO Evolution:
  Evolving hybrids (rising_fire, iron_throne, balanced_evolution, wildcard) shift
  their personality smoothly at ELO breakpoints, simulating a player's growth.

Phase Awareness:
  Biases are gated by game phase (opening / middlegame / endgame) to reflect
  how human players naturally adjust focus across the three phases.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

import chess

# ── Constants ──────────────────────────────────────────────────────────────────

_KING_ZONE_RADIUS = 2
_ENDGAME_PIECE_THRESHOLD = 6   # ≤6 minor/major pieces → endgame

_CENTRE = {chess.D4, chess.D5, chess.E4, chess.E5,
           chess.C3, chess.C4, chess.C5, chess.C6,
           chess.D3, chess.D6, chess.E3, chess.E6,
           chess.F3, chess.F4, chess.F5, chess.F6}

_EXTENDED_CENTRE = {chess.C3, chess.C4, chess.C5, chess.C6,
                    chess.D3, chess.D4, chess.D5, chess.D6,
                    chess.E3, chess.E4, chess.E5, chess.E6,
                    chess.F3, chess.F4, chess.F5, chess.F6}


# ── Style Vector ───────────────────────────────────────────────────────────────

@dataclass
class StyleVector:
    """8-dimensional personality descriptor for a playstyle.

    All values are in [0.0, 1.0]. Higher = stronger pull toward that axis.
    """
    aggression:     float = 0.5
    initiative:     float = 0.5
    complexity:     float = 0.5
    positional:     float = 0.5
    risk_tolerance: float = 0.5
    prophylaxis:    float = 0.5
    king_safety:    float = 0.5
    endgame_prec:   float = 0.5

    # How strongly the style biases the distribution (0 = no bias, 1 = max bias)
    bias_strength: float = 0.6

    def lerp(self, other: "StyleVector", t: float) -> "StyleVector":
        """Linear interpolate toward `other` by factor t ∈ [0, 1]."""
        def _l(a: float, b: float) -> float:
            return a + (b - a) * t
        return StyleVector(
            aggression=_l(self.aggression, other.aggression),
            initiative=_l(self.initiative, other.initiative),
            complexity=_l(self.complexity, other.complexity),
            positional=_l(self.positional, other.positional),
            risk_tolerance=_l(self.risk_tolerance, other.risk_tolerance),
            prophylaxis=_l(self.prophylaxis, other.prophylaxis),
            king_safety=_l(self.king_safety, other.king_safety),
            endgame_prec=_l(self.endgame_prec, other.endgame_prec),
            bias_strength=_l(self.bias_strength, other.bias_strength),
        )

    def clamp(self) -> "StyleVector":
        """Return a copy with all values clamped to [0.0, 1.0]."""
        def _c(v: float) -> float:
            return max(0.0, min(1.0, v))
        return StyleVector(
            aggression=_c(self.aggression),
            initiative=_c(self.initiative),
            complexity=_c(self.complexity),
            positional=_c(self.positional),
            risk_tolerance=_c(self.risk_tolerance),
            prophylaxis=_c(self.prophylaxis),
            king_safety=_c(self.king_safety),
            endgame_prec=_c(self.endgame_prec),
            bias_strength=_c(self.bias_strength),
        )

    def jitter(self, rng: random.Random, sigma: float = 0.06) -> "StyleVector":
        """Return a copy with Gaussian noise added to each dimension.

        Models the natural game-to-game variation of a real human player.
        sigma=0.06 from Version 2 research recommendation.
        """
        def _j(v: float) -> float:
            return max(0.0, min(1.0, v + rng.gauss(0.0, sigma)))
        return StyleVector(
            aggression=_j(self.aggression),
            initiative=_j(self.initiative),
            complexity=_j(self.complexity),
            positional=_j(self.positional),
            risk_tolerance=_j(self.risk_tolerance),
            prophylaxis=_j(self.prophylaxis),
            king_safety=_j(self.king_safety),
            endgame_prec=_j(self.endgame_prec),
            bias_strength=self.bias_strength,  # don't jitter this
        )


# ── Time-Control Offsets ───────────────────────────────────────────────────────
# From Version 2 research: how human behaviour changes across time controls.
# Applied as additive deltas on top of the base style vector, then clamped.

_TC_OFFSETS: dict[str, StyleVector] = {
    "bullet": StyleVector(
        aggression=+0.08, initiative=+0.08, complexity=-0.04,
        positional=-0.08, risk_tolerance=+0.05, prophylaxis=-0.07,
        king_safety=-0.08, endgame_prec=-0.10,
        bias_strength=+0.08,
    ),
    "blitz": StyleVector(
        aggression=0.0, initiative=0.0, complexity=0.0,
        positional=0.0, risk_tolerance=0.0, prophylaxis=0.0,
        king_safety=0.0, endgame_prec=0.0,
        bias_strength=0.0,
    ),
    "rapid": StyleVector(
        aggression=-0.05, initiative=-0.03, complexity=+0.03,
        positional=+0.07, risk_tolerance=-0.05, prophylaxis=+0.07,
        king_safety=+0.05, endgame_prec=+0.10,
        bias_strength=-0.04,
    ),
}


def apply_time_control_offset(style: StyleVector, time_control: str) -> StyleVector:
    """Adjust a style vector for the given time control.

    Args:
        style:        Base style vector.
        time_control: One of 'bullet', 'blitz', 'rapid'. Defaults to 'blitz'.

    Returns:
        A new StyleVector with time-control offsets applied and clamped to [0,1].
    """
    tc = time_control.lower().strip() if time_control else "blitz"
    if tc not in _TC_OFFSETS:
        # Try to infer from common names / presets
        if any(x in tc for x in ["bullet", "1min", "1 min", "1m", "1+0", "2+1", "2min", "2 min", "2m"]):
            tc = "bullet"
        elif any(x in tc for x in ["rapid", "classical", "daily", "10min", "10 min", "10m", "15min", "15 min", "15m", "30min", "30 min", "30m", "10+0", "15+10"]):
            tc = "rapid"
        else:
            tc = "blitz"
    delta = _TC_OFFSETS[tc]

    def _add(a: float, b: float) -> float:
        return max(0.0, min(1.0, a + b))

    return StyleVector(
        aggression=_add(style.aggression, delta.aggression),
        initiative=_add(style.initiative, delta.initiative),
        complexity=_add(style.complexity, delta.complexity),
        positional=_add(style.positional, delta.positional),
        risk_tolerance=_add(style.risk_tolerance, delta.risk_tolerance),
        prophylaxis=_add(style.prophylaxis, delta.prophylaxis),
        king_safety=_add(style.king_safety, delta.king_safety),
        endgame_prec=_add(style.endgame_prec, delta.endgame_prec),
        bias_strength=_add(style.bias_strength, delta.bias_strength),
    )


# ── In-Game Dynamic Tilt ───────────────────────────────────────────────────────

def apply_game_tilt(
    style: StyleVector,
    material_balance: int,      # positive = we're ahead in pawns
    clock_fraction: float,      # remaining_time / initial_time ∈ [0, 1]
) -> StyleVector:
    """Dynamically adjust the style vector based on game state.

    Models how real humans change their play when losing material or under
    time pressure. From Version 2 research recommendations.

    Args:
        style:            Current resolved style vector.
        material_balance: Our material minus opponent's, in pawn units.
                          Positive = we're ahead.
        clock_fraction:   How much clock we have left as a fraction (0–1).

    Returns:
        Adjusted StyleVector (clamped to [0, 1]).
    """
    def _add(a: float, b: float) -> float:
        return max(0.0, min(1.0, a + b))

    aggr   = style.aggression
    init   = style.initiative
    cmplx  = style.complexity
    pos    = style.positional
    risk   = style.risk_tolerance
    proph  = style.prophylaxis
    ksafe  = style.king_safety
    egprc  = style.endgame_prec
    bias   = style.bias_strength

    # ── Material tilt ─────────────────────────────────────────────────────────
    if material_balance <= -3:
        # We're losing significantly → play riskier, more aggressive (desperation)
        aggr   = _add(aggr, +0.10)
        risk   = _add(risk,  +0.10)
        ksafe  = _add(ksafe, -0.05)
    elif material_balance >= 3:
        # We're winning → play safer, simplify, convert
        ksafe  = _add(ksafe, +0.10)
        egprc  = _add(egprc, +0.10)
        pos    = _add(pos,   +0.05)
        risk   = _add(risk,  -0.08)
        aggr   = _add(aggr,  -0.05)

    # ── Clock pressure tilt ────────────────────────────────────────────────────
    if clock_fraction < 0.10:
        # Very low on time → drop precision-heavy dimensions
        proph  = _add(proph, -0.10)
        egprc  = _add(egprc, -0.10)
        aggr   = _add(aggr,  +0.05)  # play more forcing moves

    return StyleVector(
        aggression=aggr, initiative=init, complexity=cmplx,
        positional=pos, risk_tolerance=risk, prophylaxis=proph,
        king_safety=ksafe, endgame_prec=egprc,
        bias_strength=bias,
    )


# ── Grandmaster Base Archetypes (Version 2 corrected) ─────────────────────────

# ── Mikhail Tal (corrected: risk 0.95→0.90, egprc 0.35→0.45, ksafe→0.20)
_TAL = StyleVector(
    aggression=0.92, initiative=0.92, complexity=0.90,
    positional=0.30, risk_tolerance=0.90, prophylaxis=0.15,
    king_safety=0.20, endgame_prec=0.45,
    bias_strength=0.75,
)

# ── Tigran Petrosian (corrected: risk 0.10→0.33, init→0.28, cmplx→0.28)
_PETROSIAN = StyleVector(
    aggression=0.15, initiative=0.28, complexity=0.28,
    positional=0.95, risk_tolerance=0.33, prophylaxis=0.98,
    king_safety=0.95, endgame_prec=0.90,
    bias_strength=0.70,
)

# ── Anatoly Karpov (corrected: risk 0.30→0.20, init 0.50→0.45, egprc 1.00→0.93)
_KARPOV = StyleVector(
    aggression=0.30, initiative=0.45, complexity=0.25,
    positional=0.93, risk_tolerance=0.20, prophylaxis=0.85,
    king_safety=0.88, endgame_prec=0.93,
    bias_strength=0.65,
)

# ── Garry Kasparov (corrected: init 1.00→0.97, cmplx 1.00→0.92, pos 0.55→0.62,
#                               risk 0.75→0.70, proph 0.50→0.55, egprc 0.80→0.78)
_KASPAROV = StyleVector(
    aggression=0.85, initiative=0.97, complexity=0.92,
    positional=0.62, risk_tolerance=0.70, prophylaxis=0.55,
    king_safety=0.45, endgame_prec=0.78,
    bias_strength=0.78,
)

# ── Magnus Carlsen (corrected: risk 0.50→0.38, cmplx 0.50→0.45, bias 0.60→0.55)
_CARLSEN = StyleVector(
    aggression=0.50, initiative=0.58, complexity=0.45,
    positional=0.82, risk_tolerance=0.38, prophylaxis=0.70,
    king_safety=0.78, endgame_prec=1.00,
    bias_strength=0.55,
)

# ── Tal Reformed (kept from original — still valid hybrid)
_TAL_REFORMED = StyleVector(
    aggression=0.72, initiative=0.78, complexity=0.72,
    positional=0.60, risk_tolerance=0.62, prophylaxis=0.50,
    king_safety=0.55, endgame_prec=0.70,
    bias_strength=0.68,
)

# ── NEW: Bobby Fischer (classical precision + forcing play + elite endgame)
_FISCHER = StyleVector(
    aggression=0.65, initiative=0.78, complexity=0.55,
    positional=0.70, risk_tolerance=0.35, prophylaxis=0.55,
    king_safety=0.68, endgame_prec=0.92,
    bias_strength=0.62,
)

# ── NEW: Paul Morphy (development-first, open-game attacker, era caveat)
_MORPHY = StyleVector(
    aggression=0.85, initiative=0.95, complexity=0.45,
    positional=0.55, risk_tolerance=0.62, prophylaxis=0.30,
    king_safety=0.62, endgame_prec=0.60,
    bias_strength=0.72,
)

# ── NEW: Viswanathan Anand (universal tactician, speed-leaning)
_ANAND = StyleVector(
    aggression=0.70, initiative=0.75, complexity=0.70,
    positional=0.62, risk_tolerance=0.45, prophylaxis=0.55,
    king_safety=0.60, endgame_prec=0.68,
    bias_strength=0.58,
)

# ── NEW: Hikaru Nakamura (pragmatic universalist, speed specialist)
_NAKAMURA = StyleVector(
    aggression=0.68, initiative=0.75, complexity=0.75,
    positional=0.50, risk_tolerance=0.55, prophylaxis=0.40,
    king_safety=0.45, endgame_prec=0.72,
    bias_strength=0.65,
)

# ── NEW: Ding Liren (modern universal, positional with hidden depth)
_DING = StyleVector(
    aggression=0.42, initiative=0.50, complexity=0.50,
    positional=0.72, risk_tolerance=0.30, prophylaxis=0.68,
    king_safety=0.72, endgame_prec=0.80,
    bias_strength=0.55,
)

# ── NEW Hybrids (Version 2 — each ≥0.30 from nearest existing vector) ─────────

# The Hustler: Nakamura × Tal × Anand — scramble/swindle specialist
_THE_HUSTLER = StyleVector(
    aggression=0.80, initiative=0.86, complexity=0.88,
    positional=0.35, risk_tolerance=0.68, prophylaxis=0.25,
    king_safety=0.32, endgame_prec=0.55,
    bias_strength=0.70,
)

# The Romantic: Morphy × Tal — open-game attacker, development-first, sound sacs
_THE_ROMANTIC = StyleVector(
    aggression=0.90, initiative=0.94, complexity=0.62,
    positional=0.45, risk_tolerance=0.78, prophylaxis=0.20,
    king_safety=0.48, endgame_prec=0.50,
    bias_strength=0.74,
)

# The Hedgehog: Petrosian × Kasparov — prophylactic fortress with a hidden sting
_THE_HEDGEHOG = StyleVector(
    aggression=0.50, initiative=0.42, complexity=0.50,
    positional=0.72, risk_tolerance=0.48, prophylaxis=0.88,
    king_safety=0.80, endgame_prec=0.78,
    bias_strength=0.68,
)

# The Vise: Fischer × Karpov × Carlsen — low-risk relentless pressure, clinical endings
_THE_VISE = StyleVector(
    aggression=0.45, initiative=0.78, complexity=0.30,
    positional=0.82, risk_tolerance=0.25, prophylaxis=0.72,
    king_safety=0.80, endgame_prec=0.92,
    bias_strength=0.62,
)

# The Spider: Carlsen × Petrosian × Ding — patience, tension without initiative
_THE_SPIDER = StyleVector(
    aggression=0.40, initiative=0.35, complexity=0.75,
    positional=0.60, risk_tolerance=0.30, prophylaxis=0.75,
    king_safety=0.70, endgame_prec=0.85,
    bias_strength=0.60,
)

# The Club Regular: ELO-band baseline (see _club_regular() for variants)
# This is the 1500 blitz base. The function produces per-ELO variants.
_CLUB_REGULAR_1500 = StyleVector(
    aggression=0.58, initiative=0.55, complexity=0.50,
    positional=0.38, risk_tolerance=0.45, prophylaxis=0.22,
    king_safety=0.60, endgame_prec=0.35,
    bias_strength=0.40,
)

_CLUB_REGULAR_1200 = StyleVector(
    aggression=0.62, initiative=0.55, complexity=0.50,
    positional=0.30, risk_tolerance=0.50, prophylaxis=0.15,
    king_safety=0.50, endgame_prec=0.25,
    bias_strength=0.45,
)

_CLUB_REGULAR_1800 = StyleVector(
    aggression=0.54, initiative=0.56, complexity=0.52,
    positional=0.46, risk_tolerance=0.40, prophylaxis=0.30,
    king_safety=0.66, endgame_prec=0.46,
    bias_strength=0.38,
)


# ── ELO-Evolving Hybrids ───────────────────────────────────────────────────────

def _interpolate_by_elo(
    elo: int,
    breakpoints: list[tuple[int, StyleVector]],
) -> StyleVector:
    """Smoothly interpolate between style vectors at given ELO breakpoints."""
    if elo <= breakpoints[0][0]:
        return breakpoints[0][1]
    if elo >= breakpoints[-1][0]:
        return breakpoints[-1][1]
    for i in range(len(breakpoints) - 1):
        lo_elo, lo_style = breakpoints[i]
        hi_elo, hi_style = breakpoints[i + 1]
        if lo_elo <= elo <= hi_elo:
            t = (elo - lo_elo) / (hi_elo - lo_elo)
            return lo_style.lerp(hi_style, t)
    return breakpoints[-1][1]


def _rising_fire(elo: int) -> StyleVector:
    """Chaos → Calculated Aggression. Simulates a reckless player who refined."""
    return _interpolate_by_elo(elo, [
        (1000, _TAL),
        (1300, StyleVector(aggression=0.82, initiative=0.85, complexity=0.80,
                           positional=0.38, risk_tolerance=0.80, prophylaxis=0.28,
                           king_safety=0.30, endgame_prec=0.50, bias_strength=0.70)),
        (1600, _KASPAROV),
        (1900, _TAL_REFORMED),
    ])


def _iron_throne(elo: int) -> StyleVector:
    """Defensive → Strategic Dominance. Solid player who learns to convert."""
    return _interpolate_by_elo(elo, [
        (1000, _PETROSIAN),
        (1300, StyleVector(aggression=0.25, initiative=0.38, complexity=0.28,
                           positional=0.92, risk_tolerance=0.22, prophylaxis=0.90,
                           king_safety=0.90, endgame_prec=0.92, bias_strength=0.65)),
        (1600, _KARPOV),
        (1900, _CARLSEN),
    ])


def _balanced_evolution(elo: int) -> StyleVector:
    """Well-rounded growth: beginner → positional → universal."""
    beginner = StyleVector(
        aggression=0.50, initiative=0.50, complexity=0.45,
        positional=0.50, risk_tolerance=0.50, prophylaxis=0.45,
        king_safety=0.55, endgame_prec=0.45,
        bias_strength=0.40,
    )
    karpov_lite = StyleVector(
        aggression=0.42, initiative=0.52, complexity=0.38,
        positional=0.80, risk_tolerance=0.38, prophylaxis=0.70,
        king_safety=0.75, endgame_prec=0.82,
        bias_strength=0.55,
    )
    return _interpolate_by_elo(elo, [
        (1000, beginner),
        (1400, karpov_lite),
        (1700, _KARPOV),
        (1900, _CARLSEN),
    ])


def _club_regular(elo: int) -> StyleVector:
    """Human-like club player style, calibrated to target ELO band.

    This is the most 'undetectable' style because it models how real humans
    at 1200-1800 actually play — imprecise endgames, missing prophylaxis,
    occasionally unsound risks — NOT how GMs play.
    """
    return _interpolate_by_elo(elo, [
        (1000, _CLUB_REGULAR_1200),
        (1500, _CLUB_REGULAR_1500),
        (2000, _CLUB_REGULAR_1800),
    ])


def _wildcard(elo: int, rng: random.Random) -> StyleVector:
    """Random personality per game — weighted by ELO toward higher-quality styles."""
    chaos_weight = max(0.0, 1.0 - (elo - 1000) / 900)
    precision_weight = max(0.0, (elo - 1200) / 700)

    # Version 2: expanded pool includes all new styles
    styles = [
        _TAL, _PETROSIAN, _KARPOV, _KASPAROV, _CARLSEN, _TAL_REFORMED,
        _FISCHER, _MORPHY, _ANAND, _NAKAMURA, _DING,
        _THE_HUSTLER, _THE_ROMANTIC, _THE_HEDGEHOG, _THE_VISE, _THE_SPIDER,
    ]
    weights = [
        chaos_weight * 2.0 + 0.5,   # tal
        0.5 + precision_weight,      # petrosian
        0.5 + precision_weight,      # karpov
        0.8,                          # kasparov
        0.5 + precision_weight * 1.2, # carlsen
        0.7,                          # tal_reformed
        0.6 + precision_weight,       # fischer
        0.5 + chaos_weight,           # morphy
        0.7,                          # anand
        0.6 + chaos_weight,           # nakamura
        0.5 + precision_weight,       # ding
        0.6 + chaos_weight,           # the_hustler
        0.5 + chaos_weight,           # the_romantic
        0.5 + precision_weight,       # the_hedgehog
        0.5 + precision_weight,       # the_vise
        0.5 + precision_weight,       # the_spider
    ]
    total = sum(weights)
    roll = rng.random() * total
    cum = 0.0
    for s, w in zip(styles, weights):
        cum += w
        if roll <= cum:
            return s
    return _CARLSEN


# ── Style Registry ─────────────────────────────────────────────────────────────

#: All available playstyle names.
PLAYSTYLE_NAMES: list[str] = [
    # Grandmaster archetypes
    "tal", "petrosian", "karpov", "kasparov", "carlsen", "tal_reformed",
    "fischer", "morphy", "anand", "nakamura", "ding",
    # New hybrids (Version 2)
    "the_hustler", "the_romantic", "the_hedgehog", "the_vise", "the_spider",
    # Club/human-like
    "the_club_regular",
    # ELO-evolving
    "rising_fire", "iron_throne", "balanced_evolution", "wildcard",
]

#: One-line descriptions for each playstyle.
PLAYSTYLE_DESCRIPTIONS: dict[str, str] = {
    "tal":                "Magician from Riga. Sacrifices on intuition, creates chaos, wins on nerves.",
    "petrosian":          "Iron Tigran. Prevents everything, surrenders the exchange for fortress.",
    "karpov":             "Boa Constrictor. Restricts, squeezes, converts in the endgame.",
    "kasparov":           "Beast of Baku. Deep preparation, maximal initiative, positional attacks.",
    "carlsen":            "The Grinder. Keeps tension, outplays in equal-looking endgames.",
    "tal_reformed":       "Tal with brakes. Calculated aggression, still loves complexity.",
    "fischer":            "Classical precision. Builds from superiority, converts flawlessly.",
    "morphy":             "Development fanatic. Opens lines fast, finishes before you consolidate.",
    "anand":              "Lightning Kid. Tactical speed, universal preparation, fast patterns.",
    "nakamura":           "Speed King. Pragmatic, off-book, swindles from worse positions.",
    "ding":               "Quiet Storm. Positional depth with hidden complexity at critical moments.",
    "the_hustler":        "Scramble specialist. Muddies water early, burns clock, never resigns.",
    "the_romantic":       "Open-game attacker. Leads in development, opens lines, sacs to finish.",
    "the_hedgehog":       "Compact prophylactic defender. Absorbs pressure, then counterpunches once.",
    "the_vise":           "Relentless low-risk pressure. Simple structures, clinical conversion.",
    "the_spider":         "Patient tension-keeper. Keeps pieces on, waits for overreach.",
    "the_club_regular":   "Strong club player. Active, attack-minded, imprecise endgames (most human-like).",
    "rising_fire":        "ELO-evolving: chaotic → calculated aggression.",
    "iron_throne":        "ELO-evolving: defensive → strategic dominance.",
    "balanced_evolution": "ELO-evolving: well-rounded growth across all phases.",
    "wildcard":           "Random personality per game, weighted by ELO.",
}


def get_style_vector(
    name: str,
    elo: int = 1500,
    rng: random.Random | None = None,
) -> StyleVector:
    """Resolve a playstyle name to a StyleVector for the given ELO.

    Args:
        name:  One of PLAYSTYLE_NAMES.
        elo:   The bot's current ELO — used by evolving hybrids.
        rng:   RNG for the wildcard style. If None, a fresh Random() is used.

    Returns:
        A StyleVector configured for this ELO.
    """
    name = name.lower().strip()
    _rng = rng or random.Random()

    _base_map: dict[str, StyleVector] = {
        "tal":             _TAL,
        "petrosian":       _PETROSIAN,
        "karpov":          _KARPOV,
        "kasparov":        _KASPAROV,
        "carlsen":         _CARLSEN,
        "tal_reformed":    _TAL_REFORMED,
        "fischer":         _FISCHER,
        "morphy":          _MORPHY,
        "anand":           _ANAND,
        "nakamura":        _NAKAMURA,
        "ding":            _DING,
        "the_hustler":     _THE_HUSTLER,
        "the_romantic":    _THE_ROMANTIC,
        "the_hedgehog":    _THE_HEDGEHOG,
        "the_vise":        _THE_VISE,
        "the_spider":      _THE_SPIDER,
    }
    if name in _base_map:
        return _base_map[name]

    _evolving_map: dict[str, Callable] = {
        "rising_fire":        lambda: _rising_fire(elo),
        "iron_throne":        lambda: _iron_throne(elo),
        "balanced_evolution": lambda: _balanced_evolution(elo),
        "the_club_regular":   lambda: _club_regular(elo),
        "wildcard":           lambda: _wildcard(elo, _rng),
    }
    if name in _evolving_map:
        return _evolving_map[name]()

    raise ValueError(
        f"Unknown playstyle '{name}'. Choose from: {', '.join(PLAYSTYLE_NAMES)}"
    )


# ── Game Phase Detection ───────────────────────────────────────────────────────

def _get_game_phase(board: chess.Board, ply: int) -> str:
    """Detect the current game phase.

    Returns:
        "opening"    (first 10 plies)
        "middlegame" (default)
        "endgame"    (few pieces remain)
    """
    if ply < 10:
        return "opening"
    piece_count = sum(
        1 for sq in chess.SQUARES
        if board.piece_at(sq) is not None
        and board.piece_at(sq).piece_type not in (chess.PAWN, chess.KING)
    )
    if piece_count <= _ENDGAME_PIECE_THRESHOLD:
        return "endgame"
    return "middlegame"


# ── Per-Move Feature Extraction ────────────────────────────────────────────────

def _attacks_king_zone(board: chess.Board, move: chess.Move, opp_color: chess.Color) -> bool:
    """Check if the move's destination attacks near the opponent king."""
    opp_king_sq = board.king(opp_color)
    if opp_king_sq is None:
        return False
    kr = chess.square_rank(opp_king_sq)
    kf = chess.square_file(opp_king_sq)
    mr = chess.square_rank(move.to_square)
    mf = chess.square_file(move.to_square)
    return abs(mr - kr) <= _KING_ZONE_RADIUS and abs(mf - kf) <= _KING_ZONE_RADIUS


def _is_piece_advance(board: chess.Board, move: chess.Move, our_color: chess.Color) -> bool:
    """True if piece moves toward opponent's side of board."""
    from_rank = chess.square_rank(move.from_square)
    to_rank = chess.square_rank(move.to_square)
    if our_color == chess.WHITE:
        return to_rank > from_rank
    else:
        return to_rank < from_rank


def _exposes_our_king(board: chess.Board, move: chess.Move, our_color: chess.Color) -> bool:
    """Rough heuristic: does the move open lines toward our king?"""
    our_king_sq = board.king(our_color)
    if our_king_sq is None:
        return False
    piece = board.piece_at(move.from_square)
    if piece and piece.piece_type == chess.PAWN:
        kr = chess.square_rank(our_king_sq)
        kf = chess.square_file(our_king_sq)
        pf = chess.square_file(move.from_square)
        if abs(pf - kf) <= 1:
            return True
    return False


def _is_outpost_move(board: chess.Board, move: chess.Move, our_color: chess.Color) -> bool:
    """True if piece lands on an outpost (can't be attacked by opp pawns)."""
    piece = board.piece_at(move.from_square)
    if not piece or piece.piece_type not in (chess.KNIGHT, chess.BISHOP):
        return False
    sq = move.to_square
    opp_color = not our_color
    opp_pawn_attacks = board.attacks_mask(sq) & board.pieces_mask(chess.PAWN, opp_color)
    return opp_pawn_attacks == 0 and sq in _EXTENDED_CENTRE


def _is_open_file_rook(board: chess.Board, move: chess.Move, our_color: chess.Color) -> bool:
    """True if a rook moves to an open or semi-open file."""
    piece = board.piece_at(move.from_square)
    if not piece or piece.piece_type != chess.ROOK:
        return False
    file_idx = chess.square_file(move.to_square)
    for rank in range(8):
        sq = chess.square(file_idx, rank)
        p = board.piece_at(sq)
        if p and p.piece_type == chess.PAWN and p.color == our_color:
            return False
    return True


def _material_sacrifice_ratio(board: chess.Board, move: chess.Move, our_color: chess.Color) -> float:
    """Return > 1.0 if we're sacrificing more material than we gain.

    Returns:
        0.0 if not a capture.
        Ratio of material given / gained (> 1.0 = sacrifice).
    """
    if not board.is_capture(move):
        return 0.0

    _PIECE_VALUES = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
    }

    our_piece = board.piece_at(move.from_square)
    their_piece = board.piece_at(move.to_square)

    if not our_piece:
        return 0.0
    our_val = _PIECE_VALUES.get(our_piece.piece_type, 0)
    their_val = _PIECE_VALUES.get(their_piece.piece_type, 1) if their_piece else 1

    if their_val == 0:
        return 0.0
    return our_val / max(their_val, 0.5)


def _is_forcing(board: chess.Board, move: chess.Move) -> bool:
    """True if the move is forcing (check or capture)."""
    return board.gives_check(move) or board.is_capture(move)


def _creates_pawn_tension(board: chess.Board, move: chess.Move, our_color: chess.Color) -> bool:
    """True if the move creates a pawn-vs-pawn tension (adds complexity)."""
    piece = board.piece_at(move.from_square)
    if not piece or piece.piece_type != chess.PAWN:
        return False
    opp_color = not our_color
    dest_attacks = board.attacks(move.to_square)
    opp_pawns = board.pieces(chess.PAWN, opp_color)
    return bool(dest_attacks & opp_pawns)


# ── Score a Move Against a Style Vector ───────────────────────────────────────

def _score_move(
    board: chess.Board,
    move: chess.Move,
    style: StyleVector,
    our_color: chess.Color,
    phase: str,
) -> float:
    """Compute how well `move` aligns with the given `style`.

    Returns a multiplier in roughly [0.5, 3.0]:
      > 1.0 → this style favours the move
      < 1.0 → this style discourages the move
      = 1.0 → neutral
    """
    score = 0.0

    is_check   = board.gives_check(move)
    is_capture = board.is_capture(move)
    is_forcing = is_check or is_capture
    atk_king   = _attacks_king_zone(board, move, not our_color)
    advances   = _is_piece_advance(board, move, our_color)
    sacrifice  = _material_sacrifice_ratio(board, move, our_color)
    exposes    = _exposes_our_king(board, move, our_color)
    outpost    = _is_outpost_move(board, move, our_color)
    open_rook  = _is_open_file_rook(board, move, our_color)
    pawn_tens  = _creates_pawn_tension(board, move, our_color)
    to_centre  = move.to_square in _EXTENDED_CENTRE

    piece = board.piece_at(move.from_square)

    # ── 1. Aggression axis ────────────────────────────────────────────────────
    if is_check:   score += style.aggression * 1.2
    if is_capture: score += style.aggression * 0.7
    if atk_king:   score += style.aggression * 0.8
    if advances:   score += style.aggression * 0.3

    # ── 2. Initiative axis ────────────────────────────────────────────────────
    if is_forcing:        score += style.initiative * 0.9
    if atk_king:          score += style.initiative * 0.5
    if pawn_tens:         score += style.initiative * 0.4
    if advances and is_forcing: score += style.initiative * 0.3

    # ── 3. Complexity axis ────────────────────────────────────────────────────
    if pawn_tens: score += style.complexity * 0.5
    if sacrifice > 0.5: score += style.complexity * sacrifice * 0.6
    if is_capture and not is_check: score += style.complexity * 0.3

    # ── 4. Positional axis ────────────────────────────────────────────────────
    if outpost:   score += style.positional * 1.0
    if open_rook: score += style.positional * 0.8
    if to_centre: score += style.positional * 0.4
    if not is_forcing: score += style.positional * 0.2  # quiet moves are positional

    # ── 5. Risk Tolerance axis ────────────────────────────────────────────────
    if sacrifice > 1.0:  score += style.risk_tolerance * sacrifice * 0.8
    if sacrifice > 1.5:  score += style.risk_tolerance * 0.5       # extra bonus for big sac
    if not sacrifice and is_forcing: score += style.risk_tolerance * 0.2

    # Penalise sacrifice for low-risk-tolerance styles
    if sacrifice > 1.0:
        score -= (1.0 - style.risk_tolerance) * sacrifice * 0.6

    # ── 6. Prophylaxis axis ───────────────────────────────────────────────────
    if not is_forcing and not advances:
        score += style.prophylaxis * 0.5
    if piece and not advances and not is_forcing:
        score += style.prophylaxis * 0.3

    # ── 7. King Safety axis ───────────────────────────────────────────────────
    if exposes:
        score -= style.king_safety * 1.2
    if not exposes and not advances:
        score += style.king_safety * 0.15

    # ── 8. Endgame Precision axis (only active in endgame) ───────────────────
    if phase == "endgame":
        if not is_forcing and not is_capture:
            score += style.endgame_prec * 0.6
        if piece and piece.piece_type == chess.KING:
            score += style.endgame_prec * 0.8

    # Phase-specific tuning ───────────────────────────────────────────────────
    if phase == "opening":
        if to_centre and piece and piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            score += style.positional * 0.5
        if piece and piece.piece_type == chess.PAWN and advances:
            score += style.aggression * 0.4
        score *= 0.5  # Clamp bias — let Maia's opening knowledge dominate

    return score


# ── Main Interface: Reweight a Distribution ────────────────────────────────────

def apply_playstyle(
    board: chess.Board,
    moves: list[chess.Move],
    probabilities: list[float],
    style: StyleVector,
    our_color: chess.Color,
    ply: int = 0,
) -> list[float]:
    """Re-weight Maia's probability distribution to match the chosen playstyle.

    The style biases are applied multiplicatively on top of Maia's probabilities,
    then renormalised. Maia's strong priors are preserved; we tilt them rather
    than override them.

    Args:
        board:         Current position (before the move is played).
        moves:         List of candidate moves from Maia's distribution.
        probabilities: Corresponding probabilities from Maia (sum ≈ 1).
        style:         Style vector describing the personality.
        our_color:     The color HLC is playing.
        ply:           Current ply count (for phase detection).

    Returns:
        Re-weighted, renormalised list of probabilities (same length as input).
    """
    if not moves or style.bias_strength < 1e-4:
        return probabilities

    phase = _get_game_phase(board, ply)

    raw_scores = [
        _score_move(board, m, style, our_color, phase)
        for m in moves
    ]

    weights = [
        max(0.05, 1.0 + style.bias_strength * s)
        for s in raw_scores
    ]

    biased = [p * w for p, w in zip(probabilities, weights)]

    total = sum(biased)
    if total <= 0:
        return probabilities
    return [b / total for b in biased]


# ── Describe a StyleVector ─────────────────────────────────────────────────────

def describe_style(name: str, style: StyleVector) -> str:
    """Return a human-readable one-line description of the resolved style."""
    axes = [
        ("Aggr",  style.aggression),
        ("Init",  style.initiative),
        ("Cmplx", style.complexity),
        ("Pos",   style.positional),
        ("Risk",  style.risk_tolerance),
        ("Proph", style.prophylaxis),
        ("KSafe", style.king_safety),
        ("EGPrc", style.endgame_prec),
    ]
    bar = "  ".join(f"{label}={v:.2f}" for label, v in axes)
    desc = PLAYSTYLE_DESCRIPTIONS.get(name, "")
    return f"[{name}]  {bar}  |  bias={style.bias_strength:.2f}  —  {desc}"
