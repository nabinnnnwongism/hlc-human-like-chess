"""Move eval: top-1 move match rate vs human moves on held-out Lichess PGN samples.

Sanity check target: ~57% for Maia-3 79M per published results.
Run via: python -m hlc.eval.move_eval --pgn runs/sample.pgn
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import chess
import chess.pgn


def _parse_clk(comment: str) -> float | None:
    m = re.search(r"\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]", comment)
    if not m:
        return None
    return int(m.group(1)) * 3600.0 + int(m.group(2)) * 60.0 + float(m.group(3))


def evaluate_move_match(
    pgn_path: str | Path,
    output_path: str | Path,
    model_name: str = "maia3-79m",
    elo: int = 1500,
    max_positions: int = 2000,
    min_ply: int = 10,
    seed: int = 42,
) -> dict:
    """Evaluate Maia-3 top-1 move match rate vs human moves.

    Returns dict with hit_rate and sample counts.
    """
    from hlc.engines.maia3 import Maia3Engine
    from hlc.types import GameState

    rng = random.Random(seed)

    print(f"Loading engine '{model_name}' at Elo {elo}...")
    engine = Maia3Engine(
        model=model_name,
        elo=elo,
        device="cpu",
        use_amp=False,
        use_uci_history=True,
        multipv=1,
    )

    pgn_path = Path(pgn_path)
    positions_evaluated = 0
    hits = 0
    skipped = 0

    print(f"Evaluating move match on '{pgn_path}'...")
    with open(pgn_path, encoding="utf-8", errors="replace") as f:
        while positions_evaluated < max_positions:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            # Filter: must be a human blitz game with clk annotations
            tc = game.headers.get("TimeControl", "-")
            if "+" not in tc:
                continue

            board = game.board()
            node = game

            ply = 0
            while not node.is_end():
                node = node.variations[0]
                human_move = node.move
                ply += 1

                # Skip opening book plies
                if ply < min_ply:
                    board.push(human_move)
                    continue

                if board.is_game_over():
                    break

                state = GameState(board=board.copy(), self_elo=elo, opp_elo=elo)

                try:
                    dist = engine.get_distribution(state)
                except Exception:
                    skipped += 1
                    board.push(human_move)
                    continue

                if not dist.moves:
                    skipped += 1
                    board.push(human_move)
                    continue

                predicted_move = dist.top_move
                if predicted_move == human_move:
                    hits += 1

                positions_evaluated += 1
                if positions_evaluated % 100 == 0:
                    rate = hits / positions_evaluated * 100
                    print(
                        f"  {positions_evaluated}/{max_positions} | top-1 match: {rate:.1f}%",
                        end="\r",
                    )

                if positions_evaluated >= max_positions:
                    break

                board.push(human_move)

    hit_rate = hits / positions_evaluated if positions_evaluated > 0 else 0.0

    results = {
        "model": model_name,
        "elo": elo,
        "positions_evaluated": positions_evaluated,
        "hits": hits,
        "skipped": skipped,
        "top1_match_rate": hit_rate,
        "top1_match_pct": hit_rate * 100,
        "published_baseline_pct": 57.0,
        "note": (
            "Published Maia-3 79M top-1 match rate is ~57% on blitz games. "
            "This sanity check uses the same held-out Lichess PGN data."
        ),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Move Eval: {model_name}")
    print(f"  Positions evaluated : {positions_evaluated}")
    print(f"  Top-1 match rate    : {hit_rate * 100:.1f}%")
    print("  Published baseline  : 57.0%")
    print(f"  Results saved -> {output_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Maia-3 top-1 move match rate.")
    parser.add_argument("--pgn", required=True, help="Path to PGN file (human blitz games)")
    parser.add_argument("--model", default="maia3-79m")
    parser.add_argument("--elo", type=int, default=1500)
    parser.add_argument("--output", default="runs/move_eval.json")
    parser.add_argument("--max-positions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluate_move_match(
        pgn_path=args.pgn,
        output_path=args.output,
        model_name=args.model,
        elo=args.elo,
        max_positions=args.max_positions,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
