"""Phase 1 Sanity Script: Print top-5 moves with probabilities and WDL for 5 known positions."""

import chess

from hlc.engines.maia3 import Maia3Engine
from hlc.types import GameState

KNOWN_POSITIONS = [
    {
        "name": "1. Starting Position (White to move)",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "description": "Initial board: opening moves distribution across e4, d4, c4, Nf3.",
    },
    {
        "name": "2. After 1. e4 (Black to move)",
        "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "description": "Human responses to King's Pawn: e5, c5 (Sicilian), e6 (French), c6 (Caro-Kann).",
    },
    {
        "name": "3. Italian Game - Giuoco Piano (White to move)",
        "fen": "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "description": "Standard classical opening tabiya: c3, d3, O-O, d4 candidates.",
    },
    {
        "name": "4. Sharp Tactical Open Position (White to move)",
        "fen": "r1b1kb1r/pppp1ppp/8/4q3/4n3/2N5/PPP1BPPP/R1BQK2R w KQkq - 0 8",
        "description": "Knight on e4 attacked; White can play Nxe4, O-O, etc.",
    },
    {
        "name": "5. King and Pawn Endgame (White to move)",
        "fen": "8/8/5k2/8/4K3/8/4P3/8 w - - 0 1",
        "description": "White pawn on e2 with Kings in opposition; standard endgame technique.",
    },
]


def run_phase1_sanity() -> None:
    print("=== Phase 1 Sanity Check: Maia-3 79M Top-5 Moves for 5 Known Positions ===\n")

    engine = Maia3Engine(
        model="maia3-79m",
        elo=1500,
        device="cpu",
        use_amp=False,
        use_uci_history=True,
        multipv=5,
    )

    for i, pos_info in enumerate(KNOWN_POSITIONS, start=1):
        board = chess.Board(pos_info["fen"])
        state = GameState(board=board, self_elo=1500, opp_elo=1500)

        dist = engine.get_distribution(state)

        print("=" * 80)
        print(f"Position {i}: {pos_info['name']}")
        print(f"FEN: {pos_info['fen']}")
        print(f"Context: {pos_info['description']}")
        print(f"Legal moves count: {dist.legal_move_count}")
        print(f"Policy Shannon Entropy: {dist.entropy:.3f} nats")
        if dist.wdl:
            w, d, l = dist.wdl
            print(
                f"Position Outcome WDL: Win {w / 10:.1f}%, Draw {d / 10:.1f}%, Loss {l / 10:.1f}%"
            )
        print("-" * 80)
        print(
            f"{'Rank':<5} {'SAN':<8} {'UCI':<8} {'Probability':<14} {'Candidate WDL (W/D/L)':<24}"
        )
        print("-" * 80)

        # Print top 5 candidates
        top_k = min(5, len(dist.moves))
        for rank in range(top_k):
            move = dist.moves[rank]
            prob = dist.probabilities[rank]
            san = board.san(move)
            uci = move.uci()

            # Find matching candidate details if available
            cand_detail = next((c for c in dist.candidate_details if c.move == move), None)
            if cand_detail and cand_detail.wdl:
                cw, cd, cl = cand_detail.wdl
                wdl_str = f"{cw / 10:.1f}% / {cd / 10:.1f}% / {cl / 10:.1f}%"
            else:
                wdl_str = "N/A"

            print(f"{rank + 1:<5} {san:<8} {uci:<8} {prob * 100:6.2f}%        {wdl_str:<24}")

        print("\n")


if __name__ == "__main__":
    run_phase1_sanity()
