"""test_playstyle.py — Unit tests for the multi-dimensional playstyle personality system."""

from __future__ import annotations

import math
import random
import pytest
import chess

from hlc.playstyle import (
    PLAYSTYLE_NAMES,
    StyleVector,
    apply_playstyle,
    describe_style,
    get_style_vector,
    _get_game_phase,
)
from hlc.core import BotCore
from hlc.types import CandidateMove, GameState, MoveDistribution, MoveEngine


class MockMoveEngine:
    """Mock engine that returns deterministic legal moves with probabilities."""

    def __init__(self, moves: list[chess.Move], probs: list[float]) -> None:
        self.moves = moves
        self.probs = probs

    def get_distribution(self, state: GameState) -> MoveDistribution:
        return MoveDistribution(
            moves=self.moves,
            probabilities=self.probs,
            candidate_details=[
                CandidateMove(m, p) for m, p in zip(self.moves, self.probs)
            ],
        )


class TestPlaystyleRegistry:
    def test_all_21_playstyles_registered(self) -> None:
        assert len(PLAYSTYLE_NAMES) == 21
        expected = {
            # Grandmaster archetypes
            "tal", "petrosian", "karpov", "kasparov", "carlsen", "tal_reformed",
            "fischer", "morphy", "anand", "nakamura", "ding",
            # Hybrids
            "the_hustler", "the_romantic", "the_hedgehog", "the_vise", "the_spider",
            # Club player
            "the_club_regular",
            # Evolving hybrids
            "rising_fire", "iron_throne", "balanced_evolution", "wildcard",
        }
        assert set(PLAYSTYLE_NAMES) == expected

    def test_all_styles_resolve_to_valid_vector(self) -> None:
        for name in PLAYSTYLE_NAMES:
            sv = get_style_vector(name, elo=1500)
            assert isinstance(sv, StyleVector)
            # Check dimensions in [0, 1]
            for dim in [
                sv.aggression, sv.initiative, sv.complexity, sv.positional,
                sv.risk_tolerance, sv.prophylaxis, sv.king_safety, sv.endgame_prec,
                sv.bias_strength,
            ]:
                assert 0.0 <= dim <= 1.0

    def test_invalid_style_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown playstyle 'non_existent'"):
            get_style_vector("non_existent", elo=1500)

    def test_describe_style_format(self) -> None:
        v = get_style_vector("tal", elo=1500)
        desc = describe_style("tal", v)
        assert "[tal]" in desc
        assert "Aggr=" in desc
        assert "bias=" in desc


class TestELOEtEvolution:
    def test_rising_fire_evolution(self) -> None:
        low = get_style_vector("rising_fire", elo=1000)
        mid = get_style_vector("rising_fire", elo=1500)
        high = get_style_vector("rising_fire", elo=1900)

        # At low ELO, rising_fire is wild/chaotic (Tal-like)
        assert low.risk_tolerance > 0.85
        # As ELO grows toward 1900, it refines to Tal Reformed (more positional & endgame precision)
        assert high.endgame_prec > low.endgame_prec
        assert high.positional > low.positional

    def test_iron_throne_evolution(self) -> None:
        low = get_style_vector("iron_throne", elo=1000)
        high = get_style_vector("iron_throne", elo=1900)

        # Starts as Petrosian (super high prophylaxis, low aggression)
        assert low.aggression < 0.25
        assert low.prophylaxis > 0.90
        # Evolves toward Carlsen (universal grinder)
        assert high.aggression > low.aggression
        assert high.initiative > low.initiative

    def test_balanced_evolution(self) -> None:
        low = get_style_vector("balanced_evolution", elo=1000)
        high = get_style_vector("balanced_evolution", elo=1900)
        assert high.endgame_prec > low.endgame_prec
        assert high.positional > low.positional

    def test_wildcard_reproducibility(self) -> None:
        rng1 = random.Random(12345)
        rng2 = random.Random(12345)
        w1 = get_style_vector("wildcard", elo=1500, rng=rng1)
        w2 = get_style_vector("wildcard", elo=1500, rng=rng2)
        assert w1.aggression == w2.aggression
        assert w1.initiative == w2.initiative


class TestPhaseDetection:
    def test_opening_phase(self) -> None:
        board = chess.Board()
        assert _get_game_phase(board, ply=0) == "opening"
        assert _get_game_phase(board, ply=9) == "opening"

    def test_middlegame_phase(self) -> None:
        board = chess.Board()
        assert _get_game_phase(board, ply=15) == "middlegame"

    def test_endgame_phase(self) -> None:
        # Position with only kings and pawns
        board = chess.Board("8/4k3/8/8/8/8/4K3/8 w - - 0 1")
        assert _get_game_phase(board, ply=40) == "endgame"


class TestReweighting:
    def test_zero_bias_returns_original(self) -> None:
        board = chess.Board()
        moves = list(board.legal_moves)[:3]
        probs = [0.5, 0.3, 0.2]
        style = StyleVector(bias_strength=0.0)

        reweighted = apply_playstyle(board, moves, probs, style, chess.WHITE, ply=12)
        assert len(reweighted) == len(probs)
        for orig, rew in zip(probs, reweighted):
            assert math.isclose(orig, rew, rel_tol=1e-5)

    def test_reweighted_sums_to_one(self) -> None:
        board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
        moves = list(board.legal_moves)[:5]
        probs = [0.35, 0.25, 0.20, 0.10, 0.10]
        style = get_style_vector("tal", elo=1800)

        reweighted = apply_playstyle(board, moves, probs, style, chess.WHITE, ply=12)
        assert math.isclose(sum(reweighted), 1.0, rel_tol=1e-5)

    def test_tal_favors_attacking_vs_petrosian(self) -> None:
        # Tactical position: check/capture available
        board = chess.Board("r1bqk2r/pppp1ppp/2n5/2b1p3/2B1P1n1/3P1N2/PPP2PPP/RNBQ1RK1 w kq - 3 6")
        
        # Candidate moves:
        # Bxf7+ (sharp sacrifice check), d4 (central break), h3 (solid quiet)
        bxf7 = chess.Move.from_uci("c4f7")
        d4 = chess.Move.from_uci("d3d4")
        h3 = chess.Move.from_uci("h2h3")
        
        moves = [bxf7, d4, h3]
        base_probs = [0.333, 0.333, 0.334]
        
        tal = get_style_vector("tal", elo=1800)
        pet = get_style_vector("petrosian", elo=1800)
        
        p_tal = apply_playstyle(board, moves, base_probs, tal, chess.WHITE, ply=12)
        p_pet = apply_playstyle(board, moves, base_probs, pet, chess.WHITE, ply=12)
        
        # Tal should give significantly higher probability to Bxf7+ than Petrosian
        idx_bxf7 = moves.index(bxf7)
        assert p_tal[idx_bxf7] > p_pet[idx_bxf7]


class TestBotCoreIntegration:
    def test_botcore_with_playstyle(self) -> None:
        board = chess.Board()
        m1 = chess.Move.from_uci("e2e4")
        m2 = chess.Move.from_uci("d2d4")
        engine = MockMoveEngine([m1, m2], [0.6, 0.4])

        core = BotCore(
            move_engine=engine,
            rng=random.Random(42),
            playstyle="rising_fire",
            playstyle_elo=1800,
        )

        state = GameState(board=board)
        decision = core.decide(state)

        assert decision.move in [m1, m2]
        assert "playstyle" in decision.debug
        assert decision.debug["playstyle"] == "rising_fire"

    def test_botcore_set_playstyle_dynamically(self) -> None:
        board = chess.Board()
        m1 = chess.Move.from_uci("e2e4")
        engine = MockMoveEngine([m1], [1.0])

        core = BotCore(move_engine=engine, rng=random.Random(42))
        assert core.style_vector is None

        core.set_playstyle("kasparov", elo=1900)
        assert core.style_vector is not None
        assert core.playstyle_name == "kasparov"
        assert math.isclose(core.style_vector.complexity, 0.92, abs_tol=1e-3)

        decision = core.decide(GameState(board=board))
        assert decision.debug["playstyle"] == "kasparov"

    def test_apply_time_control_offsets(self) -> None:
        from hlc.playstyle import apply_time_control_offset
        base = get_style_vector("tal", elo=1500)

        bullet = apply_time_control_offset(base, "bullet")
        blitz = apply_time_control_offset(base, "blitz")
        rapid = apply_time_control_offset(base, "rapid")

        # Bullet increases aggression and bias, drops positional
        assert bullet.aggression >= blitz.aggression
        assert bullet.bias_strength >= blitz.bias_strength
        assert bullet.positional <= blitz.positional

        # Rapid increases positional & endgame precision, drops aggression
        assert rapid.positional >= blitz.positional
        assert rapid.endgame_prec >= blitz.endgame_prec

    def test_apply_game_tilt(self) -> None:
        from hlc.playstyle import apply_game_tilt
        base = StyleVector(
            aggression=0.5, initiative=0.5, complexity=0.5,
            positional=0.5, risk_tolerance=0.5, prophylaxis=0.5,
            king_safety=0.5, endgame_prec=0.5, bias_strength=0.5,
        )

        # Behind material: more aggressive and willing to take risks
        behind = apply_game_tilt(base, material_balance=-3, clock_fraction=0.8)
        assert behind.aggression > base.aggression
        assert behind.risk_tolerance > base.risk_tolerance

        # Ahead material: more positional, risk averse
        ahead = apply_game_tilt(base, material_balance=+4, clock_fraction=0.8)
        assert ahead.positional > base.positional
        assert ahead.risk_tolerance < base.risk_tolerance

        # Severe time trouble: precision drops, forcing moves prioritized
        flagging = apply_game_tilt(base, material_balance=0, clock_fraction=0.05)
        assert flagging.prophylaxis < base.prophylaxis
        assert flagging.endgame_prec < base.endgame_prec
        assert flagging.aggression > base.aggression

    def test_per_game_jitter_and_new_game(self) -> None:
        board = chess.Board()
        m1 = chess.Move.from_uci("e2e4")
        engine = MockMoveEngine([m1], [1.0])

        core = BotCore(
            move_engine=engine,
            rng=random.Random(100),
            playstyle="carlsen",
            enable_jitter=True,
            enable_tilt=False,
        )

        # Before any move is decided, jitter is not applied
        base_aggr = core.style_vector.aggression

        # First move of game 1 triggers jitter
        state1 = GameState(board=board)
        core.decide(state1)
        g1_style = core.style_vector
        assert core._jitter_applied is True

        # Second move in game 1 preserves identical jittered vector
        core.decide(state1)
        assert core.style_vector == g1_style

        # new_game() resets jitter state
        core.new_game()
        assert core._jitter_applied is False

        # First move of game 2 applies fresh jitter
        core.decide(state1)
        g2_style = core.style_vector
        assert core._jitter_applied is True
        # Different games should have slight variations
        assert (g1_style.aggression != g2_style.aggression or
                g1_style.initiative != g2_style.initiative)
