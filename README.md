# Human-Like Chess (HLC)

A research chess bot that models both **human move selection** and **human-like move timing** at chosen rating levels in blitz.

```
GameAdapter (local harness | lichess-bot | manual CLI)
   -> GameState {board, move_history, clocks(self,opp), increment, self_elo, opp_elo}
BotCore.decide(state) ->
   MoveEngine (Maia3Engine)      -> MoveDistribution (legal moves + probs, WDL) -> sample move
   TimingModel (pluggable)       -> Distribution over think seconds
   Scheduler                     -> target_delay seconds
   -> Decision {move, delay_s, debug}   (adapter waits, then sends the move)
```

---

## 1. Disclaimers, Boundaries & Fair Play
- **Purpose**: Realistic human opponent simulation and ML research for blitz chess.
- **Fair Play & Automated Access Policy**:
  - Strictly **NO** browser automation, screen capture, OCR, mouse/keyboard emulation, or DOM scraping touching chess.com or any external chess platform.
  - The core bot architecture is completely site-agnostic.
  - Offline private testing only or through official dedicated Lichess Bot accounts via the official Bot API.
- **Data & Training**:
  - Evaluation and training datasets are derived strictly from the **Lichess Open Database** ([database.lichess.org](https://database.lichess.org)), released under the Creative Commons CC0 1.0 Universal public domain dedication.
  - No data is scraped from chess.com.
- **Licenses**:
  - Maia-3 code and weights: AGPL-3.0 ([github.com/CSSLab/maia3](https://github.com/CSSLab/maia3)).
  - lichess-bot: AGPLv3.
  - python-chess: GPL-3.0-or-later.
  - This project is licensed under AGPL-3.0 and kept strictly non-commercial. See [LICENSES.md](LICENSES.md).

---

## 2. Installation & Quick Start

### Prerequisites
- Python 3.11 (64-bit)
- Windows / Linux / macOS

### Setup Virtual Environment
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate

# Install dependencies and HLC package in editable mode
pip install --upgrade pip
pip install -e _maia3_repo
pip install -e .
```

### Download / Cache Weights
```powershell
maia3-cache --model maia3-79m
```

---

## 3. Verification & Acceptance Scripts

### Phase 0: Environment Spike & CPU Latency Benchmark
Runs 20 legal plies with Maia-3 79M on CPU via python-chess and reports p50/p95 latency:
```powershell
python scripts/phase0_spike.py
```

### Phase 1: Move Engine Top-5 Sanity Verification
Computes exact move probabilities, Shannon policy entropy, and candidate outcome WDL across 5 diverse standard chess positions:
```powershell
python scripts/phase1_top5_sanity.py
```

### Run Tests and Linting
```powershell
pytest -v
ruff check .
ruff format --check .
```

---

## 4. Key Architectural Decisions
See [DECISIONS.md](DECISIONS.md) for full Architecture Decision Records (ADRs):
- **ADR-001**: Python 3.11.9 runtime with strict typing, pytest, and ruff.
- **ADR-002**: Upstream fact verification for Maia-3 UCI options, clock independence, and AGPLv3 licensing.
- **ADR-003**: Full move distribution via in-process PyTorch forward pass (overcoming UCI MultiPV $\le 20$ truncation to compute exact policy entropy).
- **ADR-004**: Seeded reproducibility across all move sampling and timing distributions.
- **ADR-005**: Strict boundary isolation keeping adapters decoupled from chess engine logic.
