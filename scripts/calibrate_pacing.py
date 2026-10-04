#!/usr/bin/env python3
"""
calibrate_from_lichess_db.py - replace the PRIOR numbers in human_pacing.py with
measurements from real games of your rating band and time control.

Data: https://database.lichess.org  (CC0, monthly .pgn.zst; %clk present since 2017)

  python calibrate_from_lichess_db.py lichess_db_standard_rated_2026-06.pgn.zst \
         --tc 180+2 --elo 1400 1600 --max-games 20000 --mate-miss \
         --out params.json --ref reference_stats.json

Then:  python audit_timing.py HLC.txt game.pgn --params params.json

Caveats (read them):
  * %clk has WHOLE-SECOND resolution, so 'instant' means "< 1 s" and 0.1 s atom
    tests cannot be run on this data. Premoves are folded into 'instant'.
  * Entropy coupling (ent_beta) needs Maia's entropy per position; this script
    leaves it at the prior. Run Maia over ~5k sampled positions and regress
    log(time) on entropy within deliberate moves to fit it.
  * 'usage' is left alone: tune it until audit_timing's 75th percentile matches
    reference_stats.json (the clock feedback makes it a weak, self-correcting knob).
"""
import argparse
import io
import json
import math
import random
from collections import defaultdict

import numpy as np

from hlc.timing.pacing import PacingParams


def open_pgn(path):
    if path.endswith(".zst"):
        import zstandard
        return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(path, "rb")),
                                encoding="utf-8")
    return open(path, encoding="utf-8")


def parse_tc(tc):
    a, b = tc.split("+")
    return float(a), float(b)


def games(fh, tc, lo, hi, limit):
    import chess.pgn
    n = 0
    while n < limit:
        g = chess.pgn.read_game(fh)
        if g is None:
            return
        h = g.headers
        if h.get("TimeControl") != tc:
            continue
        try:
            we, be = int(h["WhiteElo"]), int(h["BlackElo"])
        except (KeyError, ValueError):
            continue
        if not (lo <= we <= hi and lo <= be <= hi):
            continue
        n += 1
        yield g


def extract(g, init, inc, want_mates):
    """Per-move records for both sides + mate-in-1 opportunities."""
    b, last_clk, recs, mates = g.board(), {True: init, False: init}, [], []
    own_n, prev_cap_sq = {True: 0, False: 0}, None
    for node in g.mainline():
        mv, side = node.move, b.turn
        clk = node.clock()
        own_n[side] += 1
        n_legal = b.legal_moves.count()
        in_check = b.is_check()
        recap = prev_cap_sq is not None and b.is_capture(mv) and mv.to_square == prev_cap_sq
        if want_mates and clk is not None:
            avail = []
            for m in b.legal_moves:
                if b.gives_check(m):
                    b.push(m)
                    if b.is_checkmate():
                        avail.append(m)
                    b.pop()
            if avail:
                mates.append((last_clk[side], mv in avail))
        if clk is not None:
            spent = last_clk[side] + (inc if own_n[side] > 1 else 0.0) - clk
            recs.append(dict(side=side, n=own_n[side], spent=max(0.0, spent), clock_before=last_clk[side],
                             n_legal=n_legal, in_check=in_check, recap=recap))
            last_clk[side] = clk
        prev_cap_sq = mv.to_square if b.is_capture(mv) else None
        b.push(mv)
    return recs, mates



# ---------------------------------------------------------------------------
# Simulation-based tuning (method of simulated moments)
#
# Why: regressing log(time) on phase/clock is biased here. %clk has whole-second
# resolution (every move < 1 s is censored) and the model already derives its
# budget from the clock, so a clock covariate double-counts it. Tuning against
# moments computed the SAME way on data and on simulated, equally-quantised
# games avoids both problems. No python-chess needed.
# ---------------------------------------------------------------------------
# Maia entropy / top-move-prob mix of the bot's own 66-move log. Replace with a
# list of (entropy, top_p) pairs from your logs via ent_pool for a better match.
ENT_MIX = ((0.0, 0.5, 11), (0.5, 1.2, 19), (1.2, 1.8, 25), (1.8, 2.6, 11))


def _draw_ent(rng, ent_pool=None):
    if ent_pool:
        return rng.choice(ent_pool)
    r, acc, e = rng.random() * sum(w for _, _, w in ENT_MIX), 0.0, 1.2
    for lo, hi, w in ENT_MIX:
        acc += w
        if r <= acc:
            e = rng.uniform(lo, hi)
            break
    return e, min(0.99, max(0.08, math.exp(-1.1 * e) + rng.gauss(0, 0.07)))   # ASSUMED top_p~entropy link


def synth_records(P, init, inc, lengths, pool, rng, ent_pool=None):
    """Simulate games with PacingModel; return per-game records shaped like
    extract()'s output, with clocks quantised to whole seconds like Lichess %clk."""
    from human_pacing import ClockInfo, MoveFeatures, PacingModel
    games_out = []
    for L in lengths:
        models = {s: PacingModel(P, random.Random(rng.random())) for s in (True, False)}
        clk = {True: float(init), False: float(init)}
        shown = {True: init, False: init}
        own, last, recs, side = {True: 0, False: 0}, {True: None, False: None}, [], True
        for _ in range(L):
            own[side] += 1
            n = own[side]
            n_legal, in_check, recap = rng.choice(pool)          # real positions' features
            e, tp = _draw_ent(rng, ent_pool)
            mf = MoveFeatures(n=n, entropy=e, top_p=tp, n_legal=n_legal, in_check=in_check,
                              is_recapture=recap, opp_last_s=last[not side],
                              clock=ClockInfo(False, init, inc, clk[side], clk[not side]))
            t = models[side].sample(mf).seconds
            last[side] = t
            add = inc if n > 1 else 0.0
            clk[side] = max(1.0, clk[side] - t + add)
            sh = round(clk[side])
            recs.append(dict(side=side, n=n, spent=max(0.0, float(shown[side] + add - sh)),
                             clock_before=shown[side], n_legal=n_legal, in_check=in_check, recap=recap))
            shown[side] = sh
            side = not side
        games_out.append(recs)
    return games_out


def moments(R):
    t = np.array([r["spent"] for r in R])
    n = np.array([r["n"] for r in R])
    forced = np.array([(r["n_legal"] == 1) or r["recap"] for r in R])
    return {"inst": float((t < 1.0).mean()),
            "inst_forced": float((t[forced] < 1.0).mean()) if forced.any() else float("nan"),
            "m_open": float(t[n <= 15].mean()),
            "m_mid": float(t[(n > 15) & (n <= 40)].mean()),
            "m_end": float(t[n > 40].mean()) if (n > 40).any() else float("nan"),
            "sd_mean": float(t.std() / t.mean())}


# (PacingParams field, moment it is paired with, gain, bounds)
KNOBS = (("p_instant_base", "inst", 1.0, (0.005, 0.60)),
         ("p_instant_forced", "inst_forced", 1.0, (0.02, 0.90)),
         ("open_mult", "m_open", 0.8, (0.15, 3.0)),
         ("usage", "m_mid", 0.8, (0.08, 1.5)),
         ("end_mult", "m_end", 0.8, (0.25, 3.0)),
         ("sigma_idio", "sd_mean", 0.9, (0.25, 1.30)))


def tune(P, R, per_game, init, inc, rounds=20, games=500, seed=0, ent_pool=None, verbose=True):
    """Fixed-point iteration: scale each knob by (target/simulated)^gain of its moment.
    Common random numbers each round make the map deterministic, so it converges."""
    target = moments(R)
    lengths = [len(g) for g in per_game]
    pool = [(r["n_legal"], r["in_check"], r["recap"]) for r in R]
    sim = {}
    for it in range(rounds):
        rng = random.Random(seed)
        L = [rng.choice(lengths) for _ in range(games)]
        sim = moments([r for g in synth_records(P, init, inc, L, pool, rng, ent_pool) for r in g])
        err = max(abs(sim[k] / target[k] - 1) for _, k, _, _ in KNOBS if target[k] > 0 and sim[k] > 0)
        if verbose:
            print(f"  tune round {it:2d}: worst moment error {err * 100:5.1f}%")
        if err < 0.02:
            break
        for name, key, gain, (lo, hi) in KNOBS:
            if target[key] > 0 and sim[key] > 0 and not math.isnan(target[key]):
                setattr(P, name, min(hi, max(lo, getattr(P, name) * (target[key] / sim[key]) ** gain)))
    return P, {"target": target, "simulated": sim}


def fit_params(R, per_game, init, inc=2.0, P=None, tune_rounds=20, ent_pool=None, verbose=True):
    """Pure statistics (no python-chess): records -> (PacingParams, reference stats)."""
    t = np.array([r["spent"] for r in R])
    n = np.array([r["n"] for r in R])
    inst = t < 1.0
    delib = ~inst
    P = P or PacingParams()

    # --- instant (premove-like) mixture ------------------------------------
    forced = np.array([(r["n_legal"] == 1) or r["recap"] for r in R])
    base = inst[~forced & (n > 3)].mean()
    P.p_instant_base = float(base)
    P.p_instant_forced = float(max(0.0, inst[forced].mean() - base)) if forced.any() else P.p_instant_forced

    # --- deliberate-time regression -----------------------------------------
    X = np.column_stack([
        np.ones(delib.sum()), (n[delib] <= P.open_n), (n[delib] > P.end_n),
        np.log(np.array([max(r["clock_before"], 1) for r in R])[delib] / init),
        np.log(np.array([r["n_legal"] for r in R])[delib]),
        np.array([r["in_check"] for r in R])[delib], np.array([r["recap"] for r in R])[delib]]).astype(float)
    y = np.log(np.maximum(t[delib], 0.5))
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    sigma = float(resid.std())
    # NB: phase multipliers are NOT taken from `coef` (biased: clock double-count + <1 s censoring);
    # tune() below sets them by simulation.

    # --- residual AR(1) across consecutive deliberate own moves --------------
    pairs = []
    idx = np.flatnonzero(delib)
    pos = {k: i for i, k in enumerate(idx)}
    flat = 0
    for recs in per_game:
        last = {True: None, False: None}
        for r in recs:
            if r["spent"] >= 1.0:
                cur = resid[pos[flat]]
                if last[r["side"]] is not None:
                    pairs.append((last[r["side"]], cur))
                last[r["side"]] = cur
            else:
                last[r["side"]] = None
            flat += 1
    r1 = float(np.corrcoef(np.array(pairs).T)[0, 1]) if len(pairs) > 100 else P.rho * 0.4
    r1 = max(0.0, r1)
    s_ar2 = min(0.9, r1 / P.rho) * sigma ** 2
    P.sigma_ar = float(math.sqrt(s_ar2))
    # residual variance also contains the entropy-driven part the regression cannot see (ent_beta^2 * Var(z)=ent_beta^2)
    P.sigma_idio = float(math.sqrt(max(sigma ** 2 - s_ar2 - P.sigma_game ** 2 - P.ent_beta ** 2, 0.04)))

    fit_report = None
    if tune_rounds:
        P, fit_report = tune(P, R, per_game, init, inc, rounds=tune_rounds, ent_pool=ent_pool, verbose=verbose)

    # --- reference statistics for audit_timing -------------------------------
    mid = (n > 15) & (n <= 40)
    opn = n <= 15
    ref = {
        "instant_share (<1s)": float(inst.mean()),
        "SD / mean": float(t.std() / t.mean()),
        "share of moves > 8s": float((t > 8).mean()),
        "75th percentile (s)": float(np.percentile(t, 75)),
        "median (s)": float(np.median(t)),
        "mid / opening mean": float(t[mid].mean() / t[opn].mean()),
        "mean per move (s)": float(t.mean()),
        "residual lag-1 autocorr": r1,
    }
    if fit_report:
        ref["tuning"] = fit_report
    return P, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--tc", default="180+2")
    ap.add_argument("--elo", nargs=2, type=int, default=[1400, 1600])
    ap.add_argument("--max-games", type=int, default=20000)
    ap.add_argument("--mate-miss", action="store_true")
    ap.add_argument("--out", default="params.json")
    ap.add_argument("--ref", default="reference_stats.json")
    ap.add_argument("--hlc-log", help="bot log: use its (entropy, top_p) pairs for the simulation step")
    ap.add_argument("--no-tune", action="store_true", help="skip the simulation tuning step")
    a = ap.parse_args()
    init, inc = parse_tc(a.tc)

    R, M, per_game = [], [], []
    for g in games(open_pgn(a.pgn), a.tc, a.elo[0], a.elo[1], a.max_games):
        recs, mates = extract(g, init, inc, a.mate_miss)
        if len(recs) < 30:
            continue
        R += recs
        M += mates
        per_game.append(recs)
    if not R:
        raise SystemExit("no usable games - check --tc / --elo / file")
    print(f"{len(per_game)} games, {len(R)} timed moves")

    ent_pool = None
    if a.hlc_log:
        from audit_timing import parse_hlc_log
        ent_pool = [(r["ent"], r["top_p"]) for r in parse_hlc_log(a.hlc_log)] or None
    P, ref = fit_params(R, per_game, init, inc, ent_pool=ent_pool, tune_rounds=0 if a.no_tune else 20)
    ref.update(n_games=len(per_game), tc=a.tc, elo=a.elo)
    if a.mate_miss and M:
        bucket = defaultdict(lambda: [0, 0])
        for clock, played in M:
            k = "<10s" if clock < 10 else "10-30s" if clock < 30 else ">=30s"
            bucket[k][0] += 1
            bucket[k][1] += (not played)
        ref["mate_in_1_miss_rate"] = {k: {"opportunities": v[0], "miss_rate": v[1] / v[0]} for k, v in bucket.items()}
        tot = sum(v[0] for v in bucket.values())
        ref["mate_in_1_miss_rate"]["overall"] = sum(v[1] for v in bucket.values()) / tot
        print("PUT THIS INTO MATE_MISS_PRIOR for this rating band:", ref["mate_in_1_miss_rate"]["overall"])

    P.to_json(a.out)
    json.dump(ref, open(a.ref, "w"), indent=2)
    print(json.dumps(ref, indent=2))
    print(f"\nwrote {a.out} (fitted PacingParams) and {a.ref} (human reference stats)")


if __name__ == "__main__":
    main()
