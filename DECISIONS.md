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
