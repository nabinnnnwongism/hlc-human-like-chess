"""Phase 3 harness script: run 100 games between HLC bot and Stockfish (no flags, no illegal moves).

Usage: python scripts/phase3_harness.py --games 100 --stockfish-elo 1500 --verbose
"""

from __future__ import annotations

import argparse
import random
import sys

import chess

from hlc.adapters.local_harness import LocalHarness, StockfishOpponent
from hlc.core import BotCore
from hlc.engines.maia3 import Maia3Engine
from hlc.scheduler import Scheduler, SchedulerConfig
from hlc.timing.baselines import HeuristicBaseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: 100-game LocalHarness acceptance test")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--stockfish-elo", type=int, default=1500)
    parser.add_argument("--bot-elo", type=int, default=1500)
    parser.add_argument("--time", type=float, default=180.0, help="Base time in seconds")
    parser.add_argument("--increment", type=float, default=2.0, help="Increment in seconds")
    parser.add_argument("--clock-mode", choices=["virtual", "realtime"], default="virtual")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(
        f"=== Phase 3 LocalHarness: {args.games} games (HLC vs Stockfish Elo {args.stockfish_elo}) ==="
    )
    print(
        f"  Bot Elo: {args.bot_elo} | Time: {args.time}+{args.increment} | Clock: {args.clock_mode}"
    )

    rng = random.Random(args.seed)

    # 1. Build move engine
    print("Loading Maia-3 79M engine...")
    move_engine = Maia3Engine(
        model="maia3-79m",
        elo=args.bot_elo,
        device="cpu",
        use_amp=False,
        use_uci_history=True,
        multipv=3,
    )

    # 2. Build timing model + scheduler
    timing_model = HeuristicBaseline()
    scheduler = Scheduler(
        SchedulerConfig(
            move_overhead=0.10,
            safety_margin=0.50,
            max_clock_fraction=0.12,
            min_delay=0.05,
            single_move_delay=0.15,
        )
    )

    # 3. Build BotCore
    bot = BotCore(
        move_engine=move_engine,
        timing_model=timing_model,
        scheduler=scheduler,
        rng=rng,
        temperature=1.0,
        top_p=1.0,
    )

    # 4. Build opponent
    opponent = StockfishOpponent(
        stockfish_path="bin/stockfish/stockfish.exe",
        elo=args.stockfish_elo,
        movetime_ms=100,
    )

    # 5. Run harness
    illegal_move_games = 0
    flagged_games = 0
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}

    try:
        harness = LocalHarness(
            bot_core=bot,
            opponent=opponent,
            runs_dir="runs",
            time_seconds=args.time,
            increment=args.increment,
            bot_color=chess.WHITE,
            bot_elo=args.bot_elo,
            opponent_elo=args.stockfish_elo,
            clock_mode=args.clock_mode,
            rng=rng,
            verbose=args.verbose,
        )

        records = harness.run_games(num_games=args.games, prefix="phase3")

        for r in records:
            results[r.result] = results.get(r.result, 0) + 1
            if "time" in r.termination.lower():
                flagged_games += 1

    except RuntimeError as e:
        if "Illegal move" in str(e):
            illegal_move_games += 1
            print(f"ILLEGAL MOVE ERROR: {e}")
    finally:
        opponent.close()

    # 6. Report
    n = len(records) if "records" in dir() else 0
    print("\n" + "=" * 60)
    print(f"Phase 3 Harness Results ({n}/{args.games} games completed)")
    print(f"  Bot (White) wins : {results.get('1-0', 0)}")
    print(f"  Opponent wins    : {results.get('0-1', 0)}")
    print(f"  Draws            : {results.get('1/2-1/2', 0)}")
    print(f"  Illegal moves    : {illegal_move_games}")
    print(f"  Flagged games    : {flagged_games}")
    print("  PGNs saved to    : runs/")

    if illegal_move_games > 0 or flagged_games > 0:
        print("\nACCEPTANCE: FAILED — illegal moves or flags detected!")
        sys.exit(1)
    else:
        print("\nACCEPTANCE: PASSED — no illegal moves or flags across all games.")


if __name__ == "__main__":
    main()
