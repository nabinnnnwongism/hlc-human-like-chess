"""Phase 0 Spike: Run Maia-3 79M via python-chess on CPU and benchmark move latency."""

import statistics
import sys
import time

import chess
import chess.engine


def run_phase0_spike(num_plies: int = 20, model_name: str = "maia3-79m") -> dict[str, float]:
    print(f"=== Starting Phase 0 Spike: {model_name} on CPU ({num_plies} plies) ===")
    cmd = [
        sys.executable,
        "-m",
        "maia3.uci",
        "--model",
        model_name,
        "--use-uci-history",
        "--elo",
        "1500",
        "--device",
        "cpu",
        "--no-use-amp",
    ]

    print(f"Launching engine command: {' '.join(cmd)}")
    start_launch = time.perf_counter()
    engine = chess.engine.SimpleEngine.popen_uci(cmd)
    launch_elapsed = time.perf_counter() - start_launch
    print(f"Engine launched in {launch_elapsed:.2f}s")

    board = chess.Board()
    latencies: list[float] = []
    moves_played: list[str] = []

    try:
        for ply in range(1, num_plies + 1):
            if board.is_game_over():
                print(f"Game ended early at ply {ply}: {board.result()}")
                break

            t0 = time.perf_counter()
            # nodes=1 as per Maia3 specification
            result = engine.play(board, chess.engine.Limit(nodes=1))
            t1 = time.perf_counter()

            move = result.move
            if move is None or move not in board.legal_moves:
                raise RuntimeError(f"Illegal or null move produced at ply {ply}: {move}")

            elapsed_ms = (t1 - t0) * 1000.0
            latencies.append(elapsed_ms)
            san_move = board.san(move)
            moves_played.append(san_move)
            board.push(move)

            print(
                f"Ply {ply:02d}: {san_move:<7} (UCI: {move.uci():<5}) | "
                f"latency: {elapsed_ms:6.1f} ms | Turn: {'White' if board.turn == chess.BLACK else 'Black'}"
            )

    finally:
        engine.quit()

    # Calculate statistics
    p50 = statistics.median(latencies)
    sorted_latencies = sorted(latencies)
    p95_idx = round(0.95 * (len(sorted_latencies) - 1))
    p95 = sorted_latencies[p95_idx]
    mean_lat = statistics.mean(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)

    print("\n=== Phase 0 Latency Summary (ms) ===")
    print(f"Plies played: {len(latencies)}")
    print(f"Min:  {min_lat:.1f} ms")
    print(f"p50:  {p50:.1f} ms")
    print(f"Mean: {mean_lat:.1f} ms")
    print(f"p95:  {p95:.1f} ms")
    print(f"Max:  {max_lat:.1f} ms")
    print(f"Final FEN: {board.fen()}")
    print(f"PGN Moves: {' '.join(moves_played)}")

    return {
        "p50_ms": p50,
        "p95_ms": p95,
        "mean_ms": mean_lat,
        "min_ms": min_lat,
        "max_ms": max_lat,
        "plies": float(len(latencies)),
    }


if __name__ == "__main__":
    run_phase0_spike(20)
