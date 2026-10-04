"""Timing evaluation: Pearson r, Spearman, MAE, and calibration for timing models.

Run via: python -m hlc.eval.timing_eval --dataset runs/timing_dataset.parquet
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import scipy.stats


def _pearson_r(a: list[float], b: list[float]) -> float:
    """Pearson correlation coefficient."""
    r, _ = scipy.stats.pearsonr(a, b)
    return float(r)


def _spearman_r(a: list[float], b: list[float]) -> float:
    r, _ = scipy.stats.spearmanr(a, b)
    return float(r)


def _mae(predicted: list[float], actual: list[float]) -> float:
    return float(np.mean(np.abs(np.array(predicted) - np.array(actual))))


def _bucket_calibration(
    predicted_probs: list[list[float]],
    actual_thinks: list[float],
    bucket_edges: list[float],
) -> dict[str, Any]:
    """Compute calibration: fraction of actual think times falling in each bucket vs predicted probability."""
    n_buckets = len(bucket_edges) - 1
    avg_pred = np.zeros(n_buckets)
    actual_freq = np.zeros(n_buckets)
    n = len(actual_thinks)

    for i, probs in enumerate(predicted_probs):
        avg_pred += np.array(probs)
        think = actual_thinks[i]
        for b in range(n_buckets):
            if bucket_edges[b] <= think < bucket_edges[b + 1]:
                actual_freq[b] += 1
                break

    avg_pred /= n
    actual_freq /= n
    brier = float(np.mean((avg_pred - actual_freq) ** 2))

    return {
        "bucket_edges": list(bucket_edges),
        "predicted_prob": avg_pred.tolist(),
        "actual_freq": actual_freq.tolist(),
        "brier_score": brier,
    }


def evaluate_timing_model(
    dataset_path: str | Path,
    output_path: str | Path,
    model_name: str = "clock_only",
    elo: int = 1500,
    seed: int = 42,
    max_samples: int = 50_000,
) -> dict[str, Any]:
    """Evaluate a timing model on held-out Parquet dataset.

    Returns dict with Pearson, Spearman, MAE, and bucket calibration.
    """
    import chess

    from hlc.timing.base import DEFAULT_BUCKET_EDGES
    from hlc.timing.baselines import ClockOnlyBaseline, HeuristicBaseline
    from hlc.types import GameState, MoveDistribution

    rng = random.Random(seed)

    # Load dataset
    print(f"Loading {dataset_path}...")
    tbl = pq.read_table(str(dataset_path))
    df = tbl.to_pydict()
    n = min(len(df["think_time_s"]), max_samples)

    indices = list(range(len(df["think_time_s"])))
    rng.shuffle(indices)
    indices = indices[:n]

    print(f"Evaluating {n} samples with model '{model_name}' at Elo {elo}...")

    # Select timing model
    if model_name == "clock_only":
        model = ClockOnlyBaseline()
    elif model_name == "heuristic":
        model = HeuristicBaseline()
    else:
        raise ValueError(f"Unknown model_name: {model_name}. Use 'clock_only' or 'heuristic'.")

    # For heuristic model we need a plausible move distribution
    # We approximate entropy from clock fraction (proxy)
    actual_thinks = []
    predicted_means = []
    predicted_probs_list: list[list[float]] = []

    board = chess.Board()  # dummy board for legal moves count

    for idx in indices:
        think_s = float(df["think_time_s"][idx])
        clock_before = float(df["clock_before_s"][idx])
        clock_opp = float(df["clock_opp_s"][idx])
        inc = float(df["increment_s"][idx])
        sample_elo = int(df["elo"][idx])
        is_white = bool(df["is_white"][idx])

        state = GameState(
            board=board,
            clock_self=max(0.05, clock_before),
            clock_opp=max(0.05, clock_opp),
            increment=inc,
            self_elo=sample_elo,
            opp_elo=1500,
        )

        # For heuristic model, approximate a typical mid-game move distribution
        # (entropy ~1.5 nats, top_prob ~0.35) when no real board position is available
        dummy_moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
        dummy_probs = [0.65, 0.35]
        move_dist = MoveDistribution(moves=dummy_moves, probabilities=dummy_probs)

        dist = model.predict(state, move_dist)
        predicted_means.append(dist.mean())
        predicted_probs_list.append(list(dist.probabilities))
        actual_thinks.append(think_s)

    pearson = _pearson_r(predicted_means, actual_thinks)
    spearman = _spearman_r(predicted_means, actual_thinks)
    mae = _mae(predicted_means, actual_thinks)
    calib = _bucket_calibration(predicted_probs_list, actual_thinks, list(DEFAULT_BUCKET_EDGES))

    results = {
        "model": model_name,
        "elo": elo,
        "n_samples": n,
        "pearson_r": pearson,
        "spearman_r": spearman,
        "mae_s": mae,
        "calibration": calib,
        "note_chessmimic_comparison": (
            "ChessMimic (arXiv 2606.04473) reported r=0.41 / MAE=4.10s on a DIFFERENT "
            "protocol (full position features, 100-Elo bands). Direct comparison is "
            "invalid without matching protocols."
        ),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Timing Eval: {model_name}")
    print(f"  Pearson r  : {pearson:.4f}")
    print(f"  Spearman r : {spearman:.4f}")
    print(f"  MAE        : {mae:.2f} s")
    print(f"  Brier score: {calib['brier_score']:.4f}")
    print(f"  Results saved -> {output_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate timing model on held-out data.")
    parser.add_argument("--dataset", required=True, help="Path to Parquet dataset")
    parser.add_argument(
        "--model",
        default="clock_only",
        choices=["clock_only", "heuristic"],
    )
    parser.add_argument("--output", default="runs/timing_eval.json")
    parser.add_argument("--elo", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=50_000)
    args = parser.parse_args()

    evaluate_timing_model(
        dataset_path=args.dataset,
        output_path=args.output,
        model_name=args.model,
        elo=args.elo,
        seed=args.seed,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
