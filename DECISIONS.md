# Architecture & Engineering Decisions Log (DECISIONS.md)

## ADR-001: Python Runtime & Tooling
- **Decision**: Python 3.11 (`3.11.9`) managed via venv (`.venv`).
- **Rationale**: Python 3.11 provides optimal PyTorch performance and broad prebuilt wheel compatibility on Windows x64.
- **Tooling**: `pytest` for testing, `ruff` for linting and formatting, strict type hints (`typing` / `dataclasses`).

## ADR-002: Upstream Verifications & Fact Checking
- **Maia-3 UCI CLI & Options**:
  - Confirmed CLI entrypoints: `maia3-uci`, `maia3-cache`, `maia3-79m`, `maia3-5m`.
  - Confirmed options: `Elo`, `SelfElo`, `OppoElo`, `Temperature` (0 = argmax), `TopP` (1.0 = disabled), `MultiPV` (clamped between 1 and 20).
  - Confirmed clock awareness: Maia-3 has **NO clock option**; `include_time_info` defaults to `False` in architecture configs. It is purely trained on blitz positions without remaining clock conditioning.
  - Confirmed flags: `--use-uci-history`, `--device cpu`, `--no-use-amp`.
  - Checkpoint loading: Lazy loading on `isready` or `go`.
  - Hugging Face License: Verified `UofTCSSLab/Maia3-79M` is licensed under AGPLv3.
- **Lichess Open Database**:
  - Verified license: Creative Commons CC0 (Public Domain Dedication).
- **Lichess-Bot Integration**:
  - Homemade engine protocol uses `MinimalEngine` where `search(self, board, time_limit, ponder, draw_offered, root_moves)` receives `time_limit` (with remaining clocks and increments). Built-in `fake_think_time` will remain OFF to allow HLC scheduler to control move timing.

## ADR-003: Full Move Distribution Strategy (UCI MultiPV vs Python In-Process)
- **Problem**: The project requires `MoveEngine.get_distribution(state) -> MoveDistribution` providing legal moves and their probabilities, policy entropy, top-move probability, legal-move count, and WDL values.
- **Analysis**:
  - *UCI Protocol*: Maia-3's UCI implementation clamps `MultiPV` to a maximum of 20 (`clamp_multipv` line 214 in `uci.py`). In complex middle-games with 30-50 legal moves, UCI truncates the tail distribution, preventing exact calculation of policy entropy. Additionally, UCI info lines report integer permille probabilities and centipawn approximations.
  - *Python API*: In-process inference directly accesses the model's raw logits over all 1968 possible UCI moves, applies `legal_mask` to all legal moves on `board`, and computes the exact softmax probability distribution, entropy, top-p/temperature sampling, and value head WDL without IPC serialization overhead or truncation.
- **Decision**: Implement `Maia3Engine` using the direct Python API as the primary engine for high fidelity, performance, and complete entropy calculation, while also maintaining UCI command compatibility for standalone engine testing.

## ADR-004: Reproducibility & RNG Contract
- **Decision**: All stochastic operations (move sampling with temperature/top-p, think-time distribution sampling) must accept an explicit `random.Random` or `numpy.random.Generator` seeded instance. Default seeds are configurable in `config/default.yaml`.

## ADR-005: Boundary & Site-Agnostic Core
- **Decision**: The core `BotCore`, `MoveEngine`, `TimingModel`, and `Scheduler` know nothing about specific websites (Lichess, chess.com).
- **Enforcement**: Strictly no browser automation, OCR, or chess.com scraping. Offline and Bot-account testing only.

## ADR-006: Think-Time Discretization (30 Buckets) & Continuous Inversion
- **Decision**: Discretize human think time across 30 non-uniform intervals matching empirical blitz distributions: 1-second buckets from 0 to 27s, followed by [27, 30), [30, 35), [35, 45), [45, 60+).
- **Sampling**: Continuous sampling via piecewise-linear quantile inversion over precomputed cumulative probabilities ensures smooth think times rather than step-function quantization.

## ADR-007: Log-Normal Timing Formulations & The Complexity Hypothesis
- **Decision**: Parametric baselines utilize a log-normal distribution with dispersion $\sigma \approx 0.65$, matching empirical right-skewed human think times.
- **Hypothesis**: `HeuristicBaseline` modulates target mean think time with Maia-3 policy entropy ($H$) and top-move probability ($p_1$). High entropy and low top-move probability expand think time; forced moves collapse think time.
- **Status**: Maintained explicitly as a testable hypothesis to be validated empirically against held-out Lichess PGNs in Phase 3.

## ADR-008: Scheduler Flag-Safety Contract
- **Decision**: The Scheduler guarantees that the bot will never flag on time across any clock state.
- **Invariant**:
  $$\text{delay} \le \max\left(0, \text{clock\_self} - (\text{move\_overhead} + \text{safety\_margin})\right)$$
  If remaining clock is below the overhead and safety margin, planned delay collapses immediately to 0.0s. Forced moves (single legal escape/recapture) bypass normal sampling and respond near-instantly within `single_move_delay` (0.15s).

## ADR-009: Lichess-Bot Homemade Engine Adapter
- **Decision**: Implement the Lichess interface as a lichess-bot "homemade" engine (class `HLCEngine` in `lichess_bot_hook.py`) that subclasses `MinimalEngine`.
- **Rationale**:
  - The homemade engine protocol is the official, sanctioned way to run a Python bot on Lichess without reimplementing the UCI/XBoard protocols or running a subprocess.
  - `MinimalEngine.search(board, time_limit, ponder, draw_offered, root_moves)` delivers `chess.engine.Limit` with `white_clock`, `black_clock`, `white_inc`, `black_inc` directly — no custom parsing needed.
  - ADR-005 boundary is maintained: the `MinimalEngine` subclass (`lichess_bot_hook.py`) is a thin 20-line shim; all logic lives in `hlc.adapters.lichess_bot_engine.HLCEngine`.
- **`fake_think_time: false`**: MUST be set in `config.yml` — lichess-bot's built-in random sleep would stack on top of the HLC scheduler's delay, causing double-think-time.
- **Clock extraction**: `time_limit.white_clock`/`black_clock` give remaining seconds for each side; `None` fallback defaults to 30.0s (defensive).
- **`ponder`**: HLC does not ponder (no background inference between opponent's moves). Always returns `PlayResult(move, None)`.
- **`root_moves`**: Passed through lichess-bot but ignored — Maia-3 selects moves from the full distribution; restricting to `root_moves` would break human-likeness.

## ADR-010: Manual Relay CLI Architecture & Fair Play Boundaries
- **Decision**: Implement `hlc.cli.relay` as an interactive terminal relay for private testing against computer bots.
- **Fair Play Enforcement**:
  - The CLI outputs a mandatory banner at startup and documentation in `README.md`:
    `⚠️ Intended for games against built-in computer Bots only, never against human opponents.`
  - The user manually enters moves and makes the moves on their own screen/board. Strictly no browser automation, screen scraping, or chess.com API integration.
- **Dual Clock Tracking**:
  - *Automatic mode*: Uses high-precision `time.perf_counter()` to deduct elapsed time between prompts from opponent's clock, and deducts planned think delay + compute elapsed from bot's clock, adding increment per move.
  - *Manual override*: Supports explicit clock inputs inline with moves (e.g. `e4 175 180` or `e4 2:55 3:00`) or via standalone commands (`clock 175 180`).
- **Resync & Typo Resilience**:
  - Typo correction: Invalid SAN/UCI or illegal moves are trapped and reported with error messages without altering board or clock state.
  - Undo: Reverts the last turn (both bot reply and opponent move, restoring exact prior clocks and board).
  - FEN / PGN import: Allows on-the-fly resynchronization from any position or game file, parsing `[%clk]` tags when present.
- **Standard PGN Export**:
  - All games are written to `runs/relay_YYYYMMDD_HHMMSS.pgn` with standard FIDE/Lichess `[%clk H:MM:SS]` annotations, fully parseable and verifiable by `chess.pgn`.

## ADR-011: Multi-Dimensional Playstyle Personality System
- **Decision**: Implement an 8-dimensional playstyle personality system with 6 grandmaster archetypes (`tal`, `petrosian`, `karpov`, `kasparov`, `carlsen`, `tal_reformed`) and 4 evolving hybrids (`rising_fire`, `iron_throne`, `balanced_evolution`, `wildcard`).
- **Mathematical Formulation**:
  - Moves are evaluated against an 8-axis vector $\mathbf{v} \in [0, 1]^8$: aggression, initiative, complexity, positional, risk tolerance, prophylaxis, king safety, and endgame precision.
  - Candidate moves from Maia-3 receive alignment scores based on python-chess bitboard feature extraction (checks, king zone attacks, forward lunges, outposts, open files, sacrifices, and pawn tension).
  - Biasing operates multiplicatively on top of Maia-3's prior distribution:
    $$p_i' \propto p_i \cdot \max(0.05, 1.0 + \beta \cdot s_i)$$
    where $\beta$ is the style bias strength and $s_i$ is the alignment score.
- **Game Phase Awareness**:
  - Dynamically detects `opening` ($\le 10$ plies), `middlegame`, and `endgame` ($\le 6$ non-pawn/non-king pieces), adjusting weights accordingly (e.g. damping in opening to respect opening theory, activating endgame precision in endgames).
- **Smooth ELO Evolution**:
  - Evolving hybrids interpolate their style vectors smoothly across ELO thresholds, reflecting natural human development trajectories.
- **Zero Retraining Requirement**:
  - Works entirely in-process before sampling, preserving Maia-3's human move priors without modifying neural network weights.
