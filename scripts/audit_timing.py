#!/usr/bin/env python3
"""audit_timing.py — Measure how human the move TIMING of an HLC log looks, and
compare it against the new PacingModel on the same game.

Usage:
  python scripts/audit_timing.py HLC.txt game.pgn [--games 400] [--overhead 0.15]
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import re
import sys

import numpy as np

from hlc.timing.pacing import ClockInfo, MoveFeatures, PacingModel, PacingParams, VirtualClock

LINE = re.compile(
    r"HLC decision: (\S+)\s+delay=([\d.]+)s\s+\[W: (\d+):(\d+) \| B: (\d+):(\d+).*?"
    r"entropy=([\d.]+)\s+top_p=([\d.]+)"
)


def parse_hlc_log(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().replace("\r", "")
    for ln in content.split("\n"):
        m = LINE.search(ln)
        if m:
            rows.append(
                dict(
                    uci=m.group(1),
                    delay=float(m.group(2)),
                    W=int(m.group(3)) * 60 + int(m.group(4)),
                    B=int(m.group(5)) * 60 + int(m.group(6)),
                    ent=float(m.group(7)),
                    top_p=float(m.group(8)),
                )
            )
    return rows


def clock_check(rows: list[dict]) -> None:
    """Does the logged clock agree with the 'Blitz 3+0' label?"""
    if len(rows) < 2:
        return
    x = np.array([r["delay"] for r in rows[:-1]])
    y = np.array([b["B"] - a["B"] for a, b in zip(rows[:-1], rows[1:])])
    slope, icpt = np.polyfit(x, y, 1)
    dW = rows[-1]["W"] - rows[0]["W"]
    print("CLOCK CHECK")
    print(f"  Black clock change per move = {slope:+.2f} * delay {icpt:+.2f}")
    print(f"  White clock {rows[0]['W']}s -> {rows[-1]['W']}s ({dW:+d}s over {len(rows)} moves)")
    if dW > 5 or icpt > 0.8:
        print(
            f"  => clocks GAIN time every move: an increment of ~{icpt + 0.5:.0f}s exists, "
            "or an elapsed stopwatch was read. Read actual clocks from API/clocks.\n"
        )


def features_from_pgn(pgn_text: str, rows: list[dict]) -> list[dict]:
    import chess
    import chess.pgn

    g = chess.pgn.read_game(io.StringIO(pgn_text))
    if g is None:
        return []
    b, out, n, last = g.board(), [], 0, None
    last_was_capture = False
    for mv in g.mainline_moves():
        if b.turn == chess.BLACK:
            if n < len(rows):
                r = rows[n]
                recap = (
                    last is not None
                    and b.is_capture(mv)
                    and mv.to_square == last.to_square
                    and last_was_capture
                )
                out.append(
                    dict(
                        n=n + 1,
                        entropy=r["ent"],
                        top_p=r["top_p"],
                        n_legal=b.legal_moves.count(),
                        in_check=b.is_check(),
                        is_recapture=recap,
                        is_promotion=bool(mv.promotion),
                    )
                )
                n += 1
        last_was_capture = b.is_capture(mv)
        last = mv
        b.push(mv)
        if n >= len(rows):
            break
    return out


def simulate_game(feats: list[dict], params: PacingParams, seed: int, inc: float = 2.0, init: float = 180.0) -> np.ndarray:
    rng = random.Random(seed)
    model = PacingModel(params, rng)
    clk = VirtualClock(init, inc)
    t_out = []
    opp_last = None
    for f in feats:
        mf = MoveFeatures(
            n=f["n"],
            entropy=f["entropy"],
            top_p=f["top_p"],
            clock=clk.info(),
            n_legal=f["n_legal"],
            is_recapture=f["is_recapture"],
            is_promotion=f["is_promotion"],
            in_check=f["in_check"],
            opp_last_s=opp_last,
        )
        t = model.sample(mf).seconds
        t_out.append(t)
        clk.after_our_move(t)
        opp_last = 1.0 * math.exp(rng.gauss(0, 0.6))
        clk.after_opp_move(opp_last)
    return np.array(t_out)


def metrics(t: Sequence[float], e: Sequence[float]) -> dict[str, float]:
    t, e = np.asarray(t, float), np.asarray(e, float)
    inst = t < 0.6
    ni = ~inst
    tn, en = t[ni], e[ni]
    if len(tn) == 0:
        tn = t
        en = e
    _, cnt = np.unique(np.floor(tn / 0.1).astype(int), return_counts=True)
    cnt = np.sort(cnt)[::-1]
    lt = np.log(np.maximum(t, 1e-4))
    lo, hi = e < 0.8, e >= 1.8
    return {
        "instant_share (<0.6s)": float(inst.mean()),
        "instant_share (<1s)": float((t < 1.0).mean()),
        "mean per move (s)": float(t.mean()),
        "busiest 0.1s-bin share": float(cnt[0] / len(tn)),
        "top-2 bins share": float(cnt[:2].sum() / len(tn)),
        "corr(log t, ent) all moves": float(np.corrcoef(lt, e)[0, 1]),
        "corr(log t, ent) deliberate": float(np.corrcoef(np.log(np.maximum(tn, 1e-4)), en)[0, 1]),
        "P(instant | easy, ent<0.8)": float(inst[lo].mean()) if lo.any() else np.nan,
        "P(instant | hard, ent>=1.8)": float(inst[hi].mean()) if hi.any() else np.nan,
        "lag-1 autocorr(log t)": float(np.corrcoef(lt[:-1], lt[1:])[0, 1]) if len(lt) > 1 else 0.0,
        "SD / mean": float(t.std() / max(t.mean(), 1e-4)),
        "max / median": float(t.max() / max(np.median(t), 1e-4)),
        "share of moves > 8s": float((t > 8).mean()),
        "mid / opening mean": float(t[15:40].mean() / max(t[:15].mean(), 1e-4)) if len(t) >= 40 else 1.5,
        "75th percentile (s)": float(np.percentile(t, 75)),
        "median (s)": float(np.median(t)),
    }


TARGETS = {
    "instant_share (<0.6s)": (0.05, 0.25),
    "busiest 0.1s-bin share": (0.0, 0.20),
    "top-2 bins share": (0.0, 0.30),
    "corr(log t, ent) all moves": (0.15, 0.60),
    "corr(log t, ent) deliberate": (0.15, 0.60),
    "P(instant | easy, ent<0.8)": (0.20, 0.70),
    "P(instant | hard, ent>=1.8)": (0.0, 0.15),
    "lag-1 autocorr(log t)": (0.05, 0.50),
    "SD / mean": (0.8, 1.5),
    "max / median": (6.0, 60.0),
    "share of moves > 8s": (0.01, 0.15),
    "mid / opening mean": (1.2, 3.5),
    "75th percentile (s)": (3.0, 7.0),
}

REF_KEYS = {
    "instant_share (<1s)": ("abs", 0.04),
    "SD / mean": ("rel", 0.25),
    "share of moves > 8s": ("abs", 0.04),
    "75th percentile (s)": ("abs", 1.0),
    "median (s)": ("abs", 1.0),
    "mean per move (s)": ("rel", 0.25),
    "mid / opening mean": ("rel", 0.30),
}


def compare_ref(new: dict[str, float], ref_path: str) -> int:
    with open(ref_path, "r", encoding="utf-8") as f:
        ref = json.load(f)
    print(f"\nVS HUMAN REFERENCE  ({ref.get('n_games', '?')} games, tc {ref.get('tc', '?')}, elo {ref.get('elo', '?')})")
    print(f"{'metric':28s}{'human':>10s}{'model':>10s}   verdict")
    bad = 0
    for k, (kind, tol) in REF_KEYS.items():
        if k not in ref or k not in new:
            continue
        h, m = ref[k], new[k]
        ok = abs(m - h) <= tol if kind == "abs" else abs(m / h - 1) <= tol
        bad += not ok
        print(f"{k:28s}{h:10.2f}{m:10.2f}   {'ok' if ok else 'OFF'}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit HLC move timing distributions against human sanity targets.")
    ap.add_argument("hlc_log", help="Path to HLC terminal/log file")
    ap.add_argument("pgn", nargs="?", default=None, help="Optional path to game PGN")
    ap.add_argument("--games", type=int, default=200, help="Number of simulated games")
    ap.add_argument("--overhead", type=float, default=0.15, help="Estimated compute latency in seconds")
    ap.add_argument("--params", help="PacingParams json path")
    ap.add_argument("--init", type=float, default=180.0, help="Initial clock (s)")
    ap.add_argument("--inc", type=float, default=2.0, help="Increment (s)")
    ap.add_argument("--ref", help="reference_stats.json path")
    args = ap.parse_args()

    rows = parse_hlc_log(args.hlc_log)
    if not rows:
        sys.exit(f"No 'HLC decision' lines found in {args.hlc_log}")
    clock_check(rows)

    ent = [r["ent"] for r in rows]
    delays = [r["delay"] + args.overhead for r in rows]
    old = metrics(delays, ent)

    print(f"\nTIMING AUDIT (Tested log: {len(rows)} moves from {args.hlc_log})")
    print(f"{'metric':28s}{'old HLC':>10s}   target        verdict")
    for k, (lo, hi) in TARGETS.items():
        val = old.get(k, 0.0)
        ok = "PASS" if (lo <= val <= hi) else "FAIL"
        print(f"{k:28s}{val:10.2f}   {lo:5.2f}-{hi:<6.2f}  {ok}")

    if args.pgn:
        params = PacingParams.from_json(args.params) if args.params else PacingParams()
        with open(args.pgn, "r", encoding="utf-8") as f:
            pgn_content = f.read()
        feats = features_from_pgn(pgn_content, rows)
        if feats:
            per = [metrics(simulate_game(feats, params, s, inc=args.inc, init=args.init), ent) for s in range(args.games)]
            new = {k: float(np.nanmean([m[k] for m in per])) for k in old}
            print(f"\nSIMULATED PACING MODEL ({args.games} games with PacingModel):")
            print(f"{'metric':28s}{'old HLC':>10s}{'PacingModel':>12s}   target        old  new")
            for k, (lo, hi) in TARGETS.items():
                ok_old = "PASS" if lo <= old[k] <= hi else "FAIL"
                ok_new = "PASS" if lo <= new[k] <= hi else "FAIL"
                print(f"{k:28s}{old[k]:10.2f}{new[k]:12.2f}   {lo:5.2f}-{hi:<6.2f}  {ok_old} {ok_new}")
            if args.ref:
                compare_ref(new, args.ref)


if __name__ == "__main__":
    main()
