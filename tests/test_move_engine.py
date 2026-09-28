"""Unit tests for MoveEngine legality, move generation, and sampling."""

import random

import chess
import pytest

from hlc.engines.maia3 import Maia3Engine
from hlc.types import GameState, MoveDistribution


@pytest.fixture(scope="module")
def maia3_engine() -> Maia3Engine:
    """Fixture providing a loaded Maia3Engine instance."""
    return Maia3Engine(
        model="maia3-79m",
        elo=1500,
        device="cpu",
        use_amp=False,
        use_uci_history=True,
        multipv=3,
    )


class TestMoveDistribution:
    """Tests for MoveDistribution properties and sampling."""

    def test_distribution_properties(self) -> None:
        moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
        probs = [0.7, 0.3]
        dist = MoveDistribution(moves=moves, probabilities=probs)

        assert dist.top_move == chess.Move.from_uci("e2e4")
        assert dist.top_probability == 0.7
        assert dist.legal_move_count == 2
        assert dist.entropy > 0.0

    def test_deterministic_argmax_with_zero_temperature(self) -> None:
        moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
        probs = [0.8, 0.2]
        dist = MoveDistribution(moves=moves, probabilities=probs)

        sampled = dist.sample(temperature=0.0)
        assert sampled == chess.Move.from_uci("e2e4")

    def test_reproducible_sampling_with_seeded_rng(self) -> None:
        moves = [
            chess.Move.from_uci("e2e4"),
            chess.Move.from_uci("d2d4"),
            chess.Move.from_uci("c2c4"),
        ]
        probs = [0.5, 0.3, 0.2]
        dist = MoveDistribution(moves=moves, probabilities=probs)

        rng1 = random.Random(12345)
        samples1 = [dist.sample(rng=rng1, temperature=1.0).uci() for _ in range(20)]

        rng2 = random.Random(12345)
        samples2 = [dist.sample(rng=rng2, temperature=1.0).uci() for _ in range(20)]

        assert samples1 == samples2


class TestMaia3Legality:
    """Rigorous legality tests for Maia3Engine across special chess rules."""

    def test_standard_starting_position_legality(self, maia3_engine: Maia3Engine) -> None:
        board = chess.Board()
        state = GameState(board=board)
        dist = maia3_engine.get_distribution(state)

        legal_set = set(board.legal_moves)
        assert len(dist.moves) == len(legal_set)
        assert set(dist.moves) == legal_set
        assert pytest.approx(sum(dist.probabilities), abs=1e-5) == 1.0

        for move in dist.moves:
            assert move in legal_set

    def test_pawn_promotion_legality(self, maia3_engine: Maia3Engine) -> None:
        # Position with immediate pawn promotion: White pawn on e7, Black King on a8
        fen = "k7/4P3/8/8/8/8/8/4K3 w - - 0 1"
        board = chess.Board(fen)
        state = GameState(board=board)
        dist = maia3_engine.get_distribution(state)

        legal_set = set(board.legal_moves)
        assert len(dist.moves) > 0
        for move in dist.moves:
            assert move in legal_set, f"Illegal move generated: {move}"

        # Ensure promotion moves exist in the distribution
        promotions = [m for m in dist.moves if m.promotion is not None]
        assert len(promotions) == 4  # e7e8q, e7e8r, e7e8b, e7e8n
        assert any(m.uci() == "e7e8q" for m in promotions)

    def test_black_pawn_promotion_legality(self, maia3_engine: Maia3Engine) -> None:
        # Black pawn on e2 about to promote on e1
        fen = "4k3/8/8/8/8/8/4p3/K7 b - - 0 1"
        board = chess.Board(fen)
        state = GameState(board=board)
        dist = maia3_engine.get_distribution(state)

        legal_set = set(board.legal_moves)
        for move in dist.moves:
            assert move in legal_set, f"Illegal move generated: {move}"

        promotions = [m for m in dist.moves if m.promotion is not None]
        assert len(promotions) == 4
        assert any(m.uci() == "e2e1q" for m in promotions)

    def test_castling_legality(self, maia3_engine: Maia3Engine) -> None:
        # Position with both O-O and O-O-O legal for White
        fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
        board = chess.Board(fen)
        state = GameState(board=board)
        dist = maia3_engine.get_distribution(state)

        legal_set = set(board.legal_moves)
        assert chess.Move.from_uci("e1g1") in legal_set
        assert chess.Move.from_uci("e1c1") in legal_set

        for move in dist.moves:
            assert move in legal_set

        # Castling moves must be in the returned distribution
        assert any(m.uci() == "e1g1" for m in dist.moves)
        assert any(m.uci() == "e1c1" for m in dist.moves)

    def test_castling_prevented_by_attacked_square(self, maia3_engine: Maia3Engine) -> None:
        # Black rook on f8 attacks f1, preventing White kingside castling
        fen = "r3k2r/8/8/8/8/5r2/8/R3K2R w KQkq - 0 1"
        board = chess.Board(fen)
        assert chess.Move.from_uci("e1g1") not in board.legal_moves

        state = GameState(board=board)
        dist = maia3_engine.get_distribution(state)

        assert not any(m.uci() == "e1g1" for m in dist.moves)
        for move in dist.moves:
            assert move in board.legal_moves

    def test_en_passant_legality(self, maia3_engine: Maia3Engine) -> None:
        # White pawn on e5, Black just played d7-d5, en passant target d6
        fen = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
        board = chess.Board(fen)
        ep_move = chess.Move.from_uci("e5d6")
        assert ep_move in board.legal_moves

        state = GameState(board=board)
        dist = maia3_engine.get_distribution(state)

        assert ep_move in dist.moves
        for move in dist.moves:
            assert move in board.legal_moves

    def test_evading_check_legality(self, maia3_engine: Maia3Engine) -> None:
        # White King in double check: must move King
        fen = "rnb1k1nr/pp3ppp/8/1B1pp3/1b2P3/2N5/PPP2PPP/R1BQK1NR b KQkq - 1 6"
        board = chess.Board(fen)
        assert board.is_check()

        state = GameState(board=board)
        dist = maia3_engine.get_distribution(state)

        legal_set = set(board.legal_moves)
        assert len(dist.moves) == len(legal_set)
        for move in dist.moves:
            assert move in legal_set
            # Pushing the move must leave king not in check
            b = board.copy()
            b.push(move)
            assert not b.is_check()

    def test_absolute_pin_legality(self, maia3_engine: Maia3Engine) -> None:
        # White Bishop on e2 pinned to King on e1 by Black Rook on e8
        fen = "4r1k1/8/8/8/8/8/4B3/4K3 w - - 0 1"
        board = chess.Board(fen)
        assert board.is_pinned(chess.WHITE, chess.E2)

        state = GameState(board=board)
        dist = maia3_engine.get_distribution(state)

        for move in dist.moves:
            assert move in board.legal_moves
            assert move.from_square != chess.E2 or move.to_square == chess.E8
