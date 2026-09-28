"""Maia-3 MoveEngine implementation for Human-Like Chess."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import chess
import torch
from maia3.dataset import get_historical_tokens, get_legal_moves_mask, tokenize_board
from maia3.model_registry import (
    resolve_checkpoint_path,
)
from maia3.uci import (
    clamp_multipv,
    invert_wdl,
    load_model,
    parse_args,
    wdl_from_value_logits,
)
from maia3.utils import get_all_possible_moves, mirror_move

from hlc.types import CandidateMove, GameState, MoveDistribution, MoveEngine


@dataclass
class Maia3Config:
    """Configuration for Maia3Engine."""

    model: str = "maia3-79m"
    checkpoint_path: str | None = None
    elo: int = 1500
    self_elo: int = 1500
    oppo_elo: int = 1500
    temperature: float = 1.0  # 0.0 = argmax, 1.0 = standard human distribution
    top_p: float = 1.0  # 1.0 = disabled, < 1.0 = nucleus sampling
    device: str = "cpu"
    use_amp: bool = False
    use_uci_history: bool = True
    multipv: int = 5  # Candidate moves for WDL estimation (0 disables candidate WDL)
    history_len: int = 8


class Maia3Engine(MoveEngine):
    """Maia-3 Move Engine implementing the MoveEngine Protocol.

    Provides exact, full move distributions over all legal moves using in-process
    inference with Maia3 weights. Full legal-move policy entropy, top-p / temperature
    sampling, and candidate WDL human outcome predictions are exposed.
    """

    def __init__(
        self,
        model: str = "maia3-79m",
        elo: int = 1500,
        self_elo: int | None = None,
        oppo_elo: int | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        device: str = "cpu",
        use_amp: bool = False,
        use_uci_history: bool = True,
        multipv: int = 5,
        checkpoint_path: str | None = None,
    ) -> None:
        self.model_name = model
        self.elo = elo
        self.self_elo = self_elo if self_elo is not None else elo
        self.oppo_elo = oppo_elo if oppo_elo is not None else elo
        self.temperature = temperature
        self.top_p = top_p
        self.device = device
        self.use_amp = use_amp
        self.use_uci_history = use_uci_history
        self.multipv = clamp_multipv(multipv) if multipv > 0 else 0
        self.checkpoint_path = checkpoint_path

        # Move vocabulary
        self.all_moves = get_all_possible_moves()
        self.all_moves_dict: dict[str, int] = {m: i for i, m in enumerate(self.all_moves)}
        self.idx_to_move: dict[int, str] = {i: m for m, i in self.all_moves_dict.items()}

        # Build CLI-compatible config namespace for Maia3 loader
        argv = [
            "--model",
            self.model_name,
            "--device",
            self.device,
        ]
        if not self.use_amp:
            argv.append("--no-use-amp")
        if self.use_uci_history:
            argv.append("--use-uci-history")
        if self.checkpoint_path:
            argv.extend(["--checkpoint-path", self.checkpoint_path])

        self.cfg = parse_args(argv)
        if self.cfg.checkpoint_path is None:
            self.cfg.checkpoint_path = resolve_checkpoint_path(self.cfg.model_spec)

        # Load PyTorch model into memory
        self.model = load_model(self.cfg)
        self.model.eval()

    def set_elo(self, elo: int) -> None:
        """Set both self and opponent Elo."""
        self.elo = elo
        self.self_elo = elo
        self.oppo_elo = elo

    def set_self_elo(self, elo: int) -> None:
        """Set side-to-move Elo."""
        self.self_elo = elo

    def set_oppo_elo(self, elo: int) -> None:
        """Set opponent Elo."""
        self.oppo_elo = elo

    def _reconstruct_history(self, state: GameState) -> deque[torch.Tensor]:
        """Reconstruct sequence of board tokenizations up to history_len."""
        history: deque[torch.Tensor] = deque(maxlen=self.cfg.history)

        if not self.use_uci_history:
            # When uci history is disabled, replicate current position tokenization
            history.append(tokenize_board(state.board))
            return history

        # Attempt to reconstruct history from board.move_stack or state.move_history
        moves = list(state.board.move_stack)
        if not moves and state.move_history:
            moves = list(state.move_history)

        if moves:
            # Replay moves from starting position or root
            try:
                replay_board = chess.Board()
                history.append(tokenize_board(replay_board))
                for move in moves:
                    if move in replay_board.legal_moves:
                        replay_board.push(move)
                        history.append(tokenize_board(replay_board))
                    else:
                        break
                # Verify that final replay board matches state board
                if replay_board.fen() == state.board.fen():
                    return history
            except (ValueError, chess.IllegalMoveError):
                pass

        # Fallback: seed history with current board position
        history.clear()
        history.append(tokenize_board(state.board))
        return history

    def _tokens_from_history(self, history: deque[torch.Tensor]) -> torch.Tensor:
        """Build historical tokens tensor."""
        return get_historical_tokens(
            history,
            self.cfg,
            base=0.0,
            inc=0.0,
            clk_left_before=0.0,
            clk_ponder=0.0,
        )

    def _history_after_move(
        self,
        board: chess.Board,
        history: deque[torch.Tensor],
        move: chess.Move,
    ) -> deque[torch.Tensor]:
        """Produce the token history after a candidate move is played."""
        next_board = board.copy(stack=False)
        next_board.push(move)
        if self.use_uci_history:
            next_history = deque(history, maxlen=self.cfg.history)
            next_history.append(tokenize_board(next_board))
        else:
            next_history = deque([tokenize_board(next_board)], maxlen=self.cfg.history)
        return next_history

    @torch.no_grad()
    def get_distribution(self, state: GameState) -> MoveDistribution:
        """Compute the full move probability distribution and WDL for the position."""
        legal_moves = list(state.board.legal_moves)
        if not legal_moves or state.board.is_game_over():
            return MoveDistribution(moves=[], probabilities=[])

        # Resolve Elos (state overrides defaults if provided)
        s_elo = state.self_elo if state.self_elo else self.self_elo
        o_elo = state.opp_elo if state.opp_elo else self.oppo_elo

        # 1. Reconstruct board history
        history = self._reconstruct_history(state)
        tokens = self._tokens_from_history(history).unsqueeze(0).to(self.device)

        self_elos = torch.tensor([s_elo], dtype=torch.long, device=self.device)
        oppo_elos = torch.tensor([o_elo], dtype=torch.long, device=self.device)

        # 2. Forward pass for move policy and position value
        logits_move, logits_value, _ = self.model(tokens, self_elos, oppo_elos)

        # Extract position WDL (win, draw, loss)
        pos_wdl = wdl_from_value_logits(logits_value[0])

        # 3. Mask to legal moves
        legal_mask = get_legal_moves_mask(state.board, self.all_moves_dict).to(self.device)
        logits = logits_move[0].float().masked_fill(~legal_mask, float("-inf"))
        all_probs = torch.softmax(logits, dim=-1)

        # 4. Extract probability for each legal move
        is_black = state.board.turn == chess.BLACK
        move_prob_pairs: list[tuple[chess.Move, float]] = []

        for move in legal_moves:
            uci_str = mirror_move(move.uci()) if is_black else move.uci()
            idx = self.all_moves_dict.get(uci_str)
            if idx is not None:
                prob = float(all_probs[idx].item())
            else:
                prob = 0.0
            move_prob_pairs.append((move, prob))

        # Sort moves descending by probability
        move_prob_pairs.sort(key=lambda item: item[1], reverse=True)

        sorted_moves = [item[0] for item in move_prob_pairs]
        raw_probs = [item[1] for item in move_prob_pairs]

        # Re-normalize to ensure strictly valid probability sum of 1.0
        total_p = sum(raw_probs)
        if total_p > 0:
            sorted_probs = [p / total_p for p in raw_probs]
        else:
            sorted_probs = [1.0 / len(sorted_moves)] * len(sorted_moves)

        # 5. Candidate WDL estimation for top K moves
        candidates: list[CandidateMove] = []
        k_candidates = min(self.multipv, len(sorted_moves))

        if k_candidates > 0:
            top_k_moves = sorted_moves[:k_candidates]
            cand_tokens_list = [
                self._tokens_from_history(self._history_after_move(state.board, history, m))
                for m in top_k_moves
            ]
            cand_tokens = torch.stack(cand_tokens_list).to(self.device)

            # Invert perspective for the opponent who moves next
            cand_self_elos = torch.full(
                (k_candidates,), o_elo, dtype=torch.long, device=self.device
            )
            cand_oppo_elos = torch.full(
                (k_candidates,), s_elo, dtype=torch.long, device=self.device
            )

            _, cand_value_logits, _ = self.model(cand_tokens, cand_self_elos, cand_oppo_elos)

            for idx, m in enumerate(top_k_moves):
                cand_wdl = invert_wdl(wdl_from_value_logits(cand_value_logits[idx]))
                candidates.append(
                    CandidateMove(
                        move=m,
                        probability=sorted_probs[idx],
                        wdl=cand_wdl,
                    )
                )

        return MoveDistribution(
            moves=sorted_moves,
            probabilities=sorted_probs,
            candidate_details=candidates,
            wdl=pos_wdl,
        )

    def sample_move(
        self,
        state: GameState,
        rng: random.Random | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> chess.Move:
        """Convenience method to compute distribution and sample a move."""
        dist = self.get_distribution(state)
        temp = temperature if temperature is not None else self.temperature
        tp = top_p if top_p is not None else self.top_p
        return dist.sample(rng=rng, temperature=temp, top_p=tp)
