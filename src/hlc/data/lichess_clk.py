"""Lichess blitz PGN data pipeline: stream, parse think times, and store as Parquet."""

from __future__ import annotations

import gzip
import io
import re
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Regex patterns for PGN parsing
_CLK_PATTERN = re.compile(r"\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]")
_HEADER_PATTERN = re.compile(r'^\[(\w+)\s+"([^"]*)"\]')
_MOVE_PATTERN = re.compile(
    r"\d+\.{1,3}\s*([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|O-O(?:-O)?|0-0(?:-0)?)"
    r"[+#?!]* \{[^}]*\}"
)


def _parse_clk(comment: str) -> float | None:
    """Parse [%clk h:mm:ss.cc] tag from move comment, return float seconds."""
    m = _CLK_PATTERN.search(comment)
    if not m:
        return None
    hours, mins, secs = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return hours * 3600.0 + mins * 60.0 + secs


def _parse_pgn_stream(text_io: io.TextIOWrapper) -> Iterator[dict]:
    """Parse a stream of PGN games into per-ply think-time dicts.

    Yields records with:
        game_id, ply, think_time_s, remaining_clock_s, clock_opp_s,
        increment_s, white_elo, black_elo, is_white
    """
    headers: dict[str, str] = {}
    move_lines: list[str] = []
    in_game = False
    buffer: list[str] = []

    def process_game(hdrs: dict, moves_text: str) -> Iterator[dict]:
        # Extract time control
        tc = hdrs.get("TimeControl", "-")
        if tc == "-" or "+" not in tc:
            return
        try:
            base_str, inc_str = tc.split("+")
            base_s = float(base_str)
            inc_s = float(inc_str)
        except ValueError:
            return

        # Filter: blitz only (60–360s base time)
        if not (60.0 <= base_s <= 360.0):
            return

        # Filter: skip bot accounts (Lichess marks bots)
        if hdrs.get("WhiteTitle") == "BOT" or hdrs.get("BlackTitle") == "BOT":
            return

        try:
            w_elo = int(hdrs.get("WhiteElo", "0"))
            b_elo = int(hdrs.get("BlackElo", "0"))
        except ValueError:
            return

        if w_elo < 800 or b_elo < 800:
            return

        game_id = hdrs.get("Site", "").split("/")[-1]

        # Parse ply-level clock data
        # Format: move_san { [%clk H:MM:SS] }
        ply_pattern = re.compile(
            r"(?:([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|O-O(?:-O)?|0-0(?:-0)?)([+#?!=]*)\s*\{([^}]*)\})"
        )

        plies = ply_pattern.findall(moves_text)
        if not plies:
            return

        clocks: list[float] = []
        for _move_san, _ann, comment in plies:
            clk = _parse_clk(comment)
            if clk is None:
                return  # All plies must have clock info
            clocks.append(clk)

        # Reconstruct think times: think = clock_before - clock_after + increment
        # clock_before ply i = clock of same side at ply i-2 (or base at start)
        w_clock_prev = base_s
        b_clock_prev = base_s

        for i, clock_after in enumerate(clocks):
            is_white = i % 2 == 0
            if is_white:
                clock_before = w_clock_prev
                clock_opp = b_clock_prev
                think = clock_before - clock_after + inc_s
                w_clock_prev = clock_after
            else:
                clock_before = b_clock_prev
                clock_opp = w_clock_prev
                think = clock_before - clock_after + inc_s
                b_clock_prev = clock_after

            # Filter: skip opening plies (first 5 plies per side = first 10 half-moves)
            if i < 10:
                continue

            # Filter: negative think times are corrupt
            if think < 0.0 or think > 120.0:
                continue

            elo = w_elo if is_white else b_elo

            yield {
                "game_id": game_id,
                "ply": i,
                "think_time_s": think,
                "clock_before_s": clock_before,
                "clock_after_s": clock_after,
                "clock_opp_s": clock_opp,
                "increment_s": inc_s,
                "base_s": base_s,
                "elo": elo,
                "is_white": is_white,
            }

    for raw_line in text_io:
        line = raw_line.rstrip("\n")

        if line.startswith("["):
            if in_game and buffer:
                moves_text = " ".join(buffer)
                yield from process_game(headers, moves_text)
                headers = {}
                buffer = []
                in_game = False

            m = _HEADER_PATTERN.match(line)
            if m:
                headers[m.group(1)] = m.group(2)

        elif line.strip():
            in_game = True
            buffer.append(line)

    # Flush last game
    if in_game and buffer:
        moves_text = " ".join(buffer)
        yield from process_game(headers, moves_text)


def build_parquet_from_pgn(
    pgn_path: str | Path,
    output_path: str | Path,
    max_records: int | None = None,
    chunk_size: int = 100_000,
) -> int:
    """Stream a PGN file (optionally .gz or .zst compressed) and write Parquet.

    Returns total number of ply records written.
    """
    pgn_path = Path(pgn_path)
    output_path = Path(output_path)

    schema = pa.schema(
        [
            ("game_id", pa.string()),
            ("ply", pa.int32()),
            ("think_time_s", pa.float32()),
            ("clock_before_s", pa.float32()),
            ("clock_after_s", pa.float32()),
            ("clock_opp_s", pa.float32()),
            ("increment_s", pa.float32()),
            ("base_s", pa.float32()),
            ("elo", pa.int32()),
            ("is_white", pa.bool_()),
        ]
    )

    writer = pq.ParquetWriter(str(output_path), schema, compression="snappy")
    total = 0
    batch: list[dict] = []

    def flush_batch(b: list[dict]) -> int:
        arrays = {k: [r[k] for r in b] for k in schema.names}
        tbl = pa.table({k: pa.array(v, type=schema.field(k).type) for k, v in arrays.items()})
        writer.write_table(tbl)
        return len(b)

    def open_stream(p: Path):
        if p.suffix == ".zst":
            import zstandard as zstd

            ctx = zstd.ZstdDecompressor()
            raw = open(p, "rb")
            return io.TextIOWrapper(ctx.stream_reader(raw), encoding="utf-8", errors="replace")
        elif p.suffix in (".gz",):
            return io.TextIOWrapper(gzip.open(p, "rb"), encoding="utf-8", errors="replace")
        else:
            return open(p, encoding="utf-8", errors="replace")

    try:
        with open_stream(pgn_path) as f:
            for record in _parse_pgn_stream(f):
                batch.append(record)
                if len(batch) >= chunk_size:
                    total += flush_batch(batch)
                    batch = []
                    print(f"  Written {total:,} ply records...", end="\r", flush=True)
                if max_records and total >= max_records:
                    break

        if batch:
            total += flush_batch(batch)
    finally:
        writer.close()

    print(f"\nDone: {total:,} ply records -> {output_path}")
    return total
