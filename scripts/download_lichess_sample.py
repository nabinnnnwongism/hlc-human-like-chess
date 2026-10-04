"""Download blitz games with clock annotations from the Lichess API.

Streams games from a list of popular blitz players (public API, no token needed).
Saves combined PGN to data/lichess_blitz_sample.pgn.

Usage:
    python scripts/download_lichess_sample.py
    python scripts/download_lichess_sample.py --games-per-user 200 --output data/blitz.pgn

API reference: https://lichess.org/api#tag/Games/operation/apiGamesUser
Rate limit: 20 req/s authenticated, ~2 req/s unauthenticated — we add a small sleep.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Popular Lichess blitz players (all public accounts, high game count, various Elo bands)
# Mix of Elo bands: 1000-1400, 1400-1800, 1800-2200, 2200+ so calibration covers the range
DEFAULT_USERS = [
    # 2200+ (strong players)
    "DrNykterstein",  # Magnus Carlsen
    "nihalsarin2002",  # Nihal Sarin
    "FabianoCaruana",  # Caruana
    "lachesisq",  # GM
    "reveco__cl",  # GM
    # 1800-2200
    "Lance5500",
    "Joeri666",
    "UrbanDestroyer666",
    # 1400-1800
    "the_pirate_king",
    "AtomicOtter",
    "Haraka",
    # 1000-1400 (intermediate)
    "BlindfoldChampion",
    "Penguingim1",
    "ZugAddict",
    "TacticalTiger42",
]


def _download_user_games(
    username: str,
    max_games: int = 300,
    perf_type: str = "blitz",
    token: str | None = None,
) -> bytes:
    """Download up to `max_games` blitz PGN games for `username` from Lichess API."""
    url = (
        f"https://lichess.org/api/games/user/{username}"
        f"?max={max_games}&perfType={perf_type}&clocks=true&opening=false&evals=false"
    )
    headers = {
        "Accept": "application/x-chess-pgn",
        "User-Agent": "HLC-Research-Bot/1.0 (github.com/personal-project; non-commercial)",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=60) as resp:
            return resp.read()
    except HTTPError as e:
        print(f"  HTTP {e.code} for {username}: {e.reason}")
        return b""
    except URLError as e:
        print(f"  Connection error for {username}: {e.reason}")
        return b""


def _count_games(pgn_bytes: bytes) -> int:
    """Count number of [Event ...] headers in PGN bytes."""
    return pgn_bytes.count(b"[Event ")


def download_sample(
    users: list[str],
    games_per_user: int,
    output_path: Path,
    sleep_between: float = 1.5,
    token: str | None = None,
) -> int:
    """Download games from all users and write combined PGN. Returns total game count."""
    output_path.parent.mkdir(exist_ok=True)
    total_games = 0

    with open(output_path, "wb") as out_f:
        for i, user in enumerate(users, 1):
            print(
                f"[{i}/{len(users)}] Downloading {games_per_user} blitz games for '{user}'...",
                end=" ",
            )
            pgn_data = _download_user_games(user, max_games=games_per_user, token=token)
            n = _count_games(pgn_data)
            print(f"{n} games")
            if pgn_data:
                out_f.write(pgn_data)
                if not pgn_data.endswith(b"\n"):
                    out_f.write(b"\n")
            total_games += n
            if i < len(users):
                time.sleep(sleep_between)

    return total_games


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Lichess blitz sample for HLC timing eval."
    )
    parser.add_argument(
        "--games-per-user",
        type=int,
        default=300,
        help="Max games per user (Lichess API cap: 300 without auth).",
    )
    parser.add_argument(
        "--output", default="data/lichess_blitz_sample.pgn", help="Output PGN file path."
    )
    parser.add_argument(
        "--token", default=None, help="Optional Lichess API token (increases rate limits)."
    )
    parser.add_argument(
        "--users", nargs="+", default=None, help="Override list of Lichess usernames."
    )
    parser.add_argument(
        "--sleep", type=float, default=1.5, help="Seconds to sleep between requests (default 1.5s)."
    )
    args = parser.parse_args()

    users = args.users or DEFAULT_USERS
    output = Path(args.output)

    print(f"Downloading from {len(users)} users x {args.games_per_user} games each...")
    print(f"Output: {output}\n")

    total = download_sample(
        users=users,
        games_per_user=args.games_per_user,
        output_path=output,
        sleep_between=args.sleep,
        token=args.token,
    )

    size_mb = output.stat().st_size / 1024 / 1024 if output.exists() else 0
    print(f"\nDone: {total} games total | {size_mb:.1f} MB -> {output}")
    print("\nNext step: build the Parquet timing dataset:")
    print(
        f'  python -c "from hlc.data.lichess_clk import build_parquet_from_pgn; '
        f"build_parquet_from_pgn('{output}', 'runs/timing_dataset.parquet')\""
    )


if __name__ == "__main__":
    main()
