"""run_selfplay_audit.py — Local self-play test for PacingModel verification.

Runs BotCore (with the new PacingModel) as White against Stockfish as Black.
No Maia-3 / torch required: a StockfishMoveEngine stub provides the move
probability distribution using Stockfish multi-PV scores.

Usage:
    python scripts/run_selfplay_audit.py
    python scripts/run_selfplay_audit.py --games 3 --moves 40 --elo 1500
"""

from __future__ import annotations

import argparse
import datetime
import math
import random
import sys
import time
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import chess
import chess.engine

from hlc.core import BotCore
from hlc.types import GameState, MoveDistribution
from hlc.timing.pacing import PacingModel

STOCKFISH_PATH = ROOT / "bin" / "stockfish" / "stockfish.exe"

# ─────────────────────────────────────────────────────────────────────────────
# StockfishMoveEngine: satisfies the MoveEngine Protocol using multi-PV scores
# ─────────────────────────────────────────────────────────────────────────────


class StockfishMoveEngine:
    """MoveEngine stub backed by Stockfish multi-PV analysis.

    Converts centipawn scores to probabilities via a softmax, giving BotCore a
    realistic distribution (varying entropy) to pass to PacingModel.
    """

    MULTIPV = 10
    MOVETIME_MS = 80
    TEMP = 0.35  # softmax temperature

    def __init__(self, path: Path = STOCKFISH_PATH, elo: int = 1500) -> None:
        self._engine = chess.engine.SimpleEngine.popen_uci(str(path))
        self._engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})

    def get_distribution(self, state: GameState) -> MoveDistribution:
        board = state.board
        legal = list(board.legal_moves)
        n = len(legal)

        if n == 0:
            return MoveDistribution(moves=[], probabilities=[])

        try:
            infos = self._engine.analyse(
                board,
                limit=chess.engine.Limit(time=self.MOVETIME_MS / 1000.0),
                multipv=min(self.MULTIPV, n),
                info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
            )
        except Exception:
            probs = [1.0 / n] * n
            return MoveDistribution(moves=legal, probabilities=probs)

        move_scores: dict[chess.Move, float] = {}
        for info in infos:
            pv = info.get("pv")
            score = info.get("score")
            if not pv or score is None:
                continue
            move = pv[0]
            cp = score.white().score(mate_score=10000)
            if cp is None:
                cp = 0.0
            move_scores[move] = float(cp) / 100.0

        min_score = (min(move_scores.values()) - 3.0) if move_scores else -3.0
        scores = [move_scores.get(m, min_score) for m in legal]

        max_s = max(scores)
        exps = [math.exp((s - max_s) / self.TEMP) for s in scores]
        total = sum(exps)
        probs = [e / total for e in exps]

        return MoveDistribution(moves=legal, probabilities=probs)

    def close(self) -> None:
        try:
            self._engine.quit()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Self-play loop
# ─────────────────────────────────────────────────────────────────────────────


def play_game(
    bot: BotCore,
    sf_opponent: chess.engine.SimpleEngine,
    time_s: float,
    inc_s: float,
    bot_color: chess.Color,
    bot_elo: int,
    opp_elo: int,
    max_plies: int,
    verbose: bool,
) -> list[dict]:
    board = chess.Board()
    clock_bot = time_s
    clock_opp = time_s
    records: list[dict] = []
    bot.new_game()

    for ply in range(max_plies * 2):
        if board.is_game_over(claim_draw=True):
            break

        clk_self = clock_bot if board.turn == bot_color else clock_opp
        clk_opp  = clock_opp if board.turn == bot_color else clock_bot

        if board.turn == bot_color:
            state = GameState(
                board=board.copy(),
                move_history=list(board.move_stack),
                clock_self=clk_self,
                clock_opp=clk_opp,
                increment=inc_s,
                self_elo=bot_elo,
                opp_elo=opp_elo,
            )
            t0 = time.perf_counter()
            decision = bot.decide(state)
            compute_elapsed = time.perf_counter() - t0

            delay   = decision.delay_s
            move    = decision.move
            entropy = decision.debug.get("policy_entropy", 0.0)
            top_p   = decision.debug.get("top_probability", 1.0)

            records.append({
                "ply": ply,
                "move": move.uci(),
                "delay": delay,
                "entropy": entropy,
                "top_p": top_p,
                "clock_before": clk_self,
            })

            if verbose:
                san = board.san(move)
                print(
                    f"  ply={ply+1:3d} Bot  {san:<8} "
                    f"delay={delay:5.2f}s  ent={entropy:.2f}  clk={clk_self:.0f}s"
                )

            board.push(move)
            clock_bot = max(0.1, clk_self - delay + inc_s)

        else:
            try:
                result = sf_opponent.play(board, limit=chess.engine.Limit(time=0.05))
                opp_move = result.move
            except Exception:
                opp_move = None

            if opp_move is None or opp_move not in board.legal_moves:
                opp_move = next(iter(board.legal_moves), None)
            if opp_move is None:
                break

            if verbose:
                san = board.san(opp_move)
                print(f"  ply={ply+1:3d} Opp  {san}")

            board.push(opp_move)
            clock_opp = max(0.1, clk_opp - 0.05 + inc_s)

        if clock_bot <= 0 or clock_opp <= 0:
            break

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Inline audit
# ─────────────────────────────────────────────────────────────────────────────


def audit(records: list[dict]) -> None:
    delays    = [r["delay"]   for r in records]
    entropies = [r["entropy"] for r in records]
    n = len(delays)
    if n < 5:
        print("Not enough moves to audit.")
        return

    INSTANT = 0.6
    BIN_W   = 0.1

    instant_share = sum(1 for d in delays if d < INSTANT) / n

    from collections import Counter
    bins = [round(d // BIN_W * BIN_W, 1) for d in delays]
    bin_counts = Counter(bins)
    busiest = bin_counts.most_common(1)[0][1] / n
    top2    = sum(c for _, c in bin_counts.most_common(2)) / n

    def corr(xs, ys):
        if len(xs) < 3:
            return 0.0
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx  = math.sqrt(sum((x - mx)**2 for x in xs) + 1e-12)
        dy  = math.sqrt(sum((y - my)**2 for y in ys) + 1e-12)
        return num / (dx * dy)

    log_d = [math.log(max(d, 0.01)) for d in delays]
    corr_all   = corr(log_d, entropies)
    delib      = [(math.log(max(r["delay"],0.01)), r["entropy"])
                  for r in records if r["delay"] >= INSTANT]
    corr_delib = corr([x for x,_ in delib], [y for _,y in delib]) if delib else 0.0

    hard         = [r for r in records if r["entropy"] >= 1.8]
    p_inst_hard  = sum(1 for r in hard if r["delay"] < INSTANT) / max(len(hard), 1)
    easy         = [r for r in records if r["entropy"] < 0.8]
    p_inst_easy  = sum(1 for r in easy if r["delay"] < INSTANT) / max(len(easy), 1)

    lag1 = corr(log_d[:-1], log_d[1:]) if n > 2 else 0.0

    mean_d  = sum(delays) / n
    sd      = math.sqrt(sum((d - mean_d)**2 for d in delays) / n)
    cv      = sd / max(mean_d, 0.001)

    sorted_d       = sorted(delays)
    median_d       = sorted_d[n // 2]
    max_d          = max(delays)
    max_over_med   = max_d / max(median_d, 0.001)
    share_gt8      = sum(1 for d in delays if d > 8.0) / n

    opening = [r["delay"] for r in records if r["ply"] // 2 < 12]
    middle  = [r["delay"] for r in records if 12 <= r["ply"] // 2 < 35]
    phase_ratio = (
        (sum(middle) / max(len(middle), 1))
        / max(sum(opening) / max(len(opening), 1), 0.001)
    )
    p75 = sorted_d[int(0.75 * n)]

    rows = [
        ("instant_share (<0.6s)",       instant_share,  0.05, 0.25),
        ("busiest 0.1s-bin share",      busiest,        0.00, 0.20),
        ("top-2 bins share",            top2,           0.00, 0.30),
        ("corr(log t, ent) all moves",  corr_all,       0.15, 0.60),
        ("corr(log t, ent) deliberate", corr_delib,     0.15, 0.60),
        ("P(instant | easy, ent<0.8)",  p_inst_easy,    0.20, 0.70),
        ("P(instant | hard, ent>=1.8)", p_inst_hard,    0.00, 0.15),
        ("lag-1 autocorr(log t)",       lag1,           0.05, 0.50),
        ("SD / mean",                   cv,             0.80, 1.50),
        ("max / median",                max_over_med,   6.00, 60.0),
        ("share of moves > 8s",         share_gt8,      0.01, 0.15),
        ("mid / opening mean",          phase_ratio,    1.20, 3.50),
        ("75th percentile (s)",         p75,            3.00, 7.00),
    ]

    print(f"\nTIMING AUDIT (Self-play: {n} bot moves)\n")
    hdr = f"  {'metric':<33} {'value':>8}   {'target':^15}  verdict"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    passes = 0
    for label, val, lo, hi in rows:
        ok = lo <= val <= hi
        if ok:
            passes += 1
        mark = "PASS" if ok else "FAIL"
        icon = "[OK]" if ok else "[!!]"
        print(f"  {label:<33} {val:8.2f}   [{lo:.2f} - {hi:.2f}]    {icon} {mark}")

    print(f"\n  Result: {passes}/{len(rows)} metrics PASS\n")

    # Delay histogram (visual check)
    print("  Delay histogram (0.5s buckets):")
    from collections import Counter as _C
    hbins = _C(round(d // 0.5 * 0.5, 1) for d in delays)
    for b in sorted(hbins):
        bar = "#" * hbins[b]
        print(f"    {b:5.1f}s | {bar} ({hbins[b]})")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-play PacingModel audit")
    parser.add_argument("--games",   type=int,   default=2,     help="Number of games")
    parser.add_argument("--moves",   type=int,   default=40,    help="Max moves per game")
    parser.add_argument("--time",    type=float, default=180.0, help="Clock per side (s)")
    parser.add_argument("--inc",     type=float, default=0.0,   help="Increment (s)")
    parser.add_argument("--elo",     type=int,   default=1500,  help="Bot ELO")
    parser.add_argument("--opp-elo", type=int,   default=1500,  help="Opponent ELO")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"Self-play audit — {args.games} game(s) × {args.moves} moves | "
          f"Clock {args.time:.0f}+{args.inc:.0f} | Elo {args.elo}")
    print(f"Stockfish: {STOCKFISH_PATH}\n")

    rng = random.Random(42)
    sf_engine = StockfishMoveEngine(STOCKFISH_PATH, elo=args.elo)
    bot = BotCore(
        move_engine=sf_engine,
        rng=rng,
        temperature=1.1,
        top_p=0.95,
        enable_guard=True,
        pacing_model=PacingModel(rng=rng),
    )

    sf_opp = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_PATH))
    sf_opp.configure({"UCI_LimitStrength": True, "UCI_Elo": args.opp_elo})

    all_records: list[dict] = []
    try:
        for g in range(1, args.games + 1):
            bot_color = chess.WHITE if g % 2 == 1 else chess.BLACK
            side = "White" if bot_color == chess.WHITE else "Black"
            print(f"=== Game {g}/{args.games}  (Bot plays {side}) ===")
            records = play_game(
                bot=bot,
                sf_opponent=sf_opp,
                time_s=args.time,
                inc_s=args.inc,
                bot_color=bot_color,
                bot_elo=args.elo,
                opp_elo=args.opp_elo,
                max_plies=args.moves,
                verbose=args.verbose,
            )
            all_records.extend(records)
            print(f"  -> {len(records)} bot moves recorded\n")
    finally:
        sf_engine.close()
        sf_opp.quit()

    # Save raw log
    runs_dir = ROOT / "runs"
    runs_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = runs_dir / f"selfplay_{ts}.log"
    with open(log_path, "w") as f:
        for r in all_records:
            f.write(
                f"ply={r['ply']}  move={r['move']}  delay={r['delay']:.4f}  "
                f"entropy={r['entropy']:.4f}  top_p={r['top_p']:.4f}  "
                f"clock={r['clock_before']:.1f}\n"
            )
    print(f"Raw log saved: {log_path}\n")

    audit(all_records)


if __name__ == "__main__":
    main()
