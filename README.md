<div align="center">

# 🧠 HLC — Human-Like Chess

**A neural chess agent engineered to play like a human, not an engine.**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: Personal Use Only](https://img.shields.io/badge/license-Personal%20Use%20Only-red.svg)](LICENSE)
[![Engine: Maia-3 79M](https://img.shields.io/badge/engine-Maia--3%2079M-green.svg)](https://github.com/CSSLab/maia3)
[![Platform: Chess.com](https://img.shields.io/badge/platform-Chess.com-brightgreen.svg)](https://chess.com)
[![Tests: 87 Passed](https://img.shields.io/badge/tests-87%20passed-success.svg)](#6-testing)

*Powered by Maia-3 · Adaptive Timing · 10 Playstyle Personalities · Multi-Game Session Management*

</div>

---

## What is HLC?

HLC (Human-Like Chess) is an autonomous chess agent designed for **behavioral fidelity** — it plays the way real humans play at a given rating level, not like a classical chess engine computing optimal moves.

Where a conventional engine (Stockfish, Lc0) plays the mathematically strongest move as fast as possible, HLC:

- **Thinks at the right pace** — fast on obvious moves, slow on complex positions
- **Makes human-plausible mistakes** — not random blunders, but the kind of oversights real players at your Elo make
- **Has a consistent personality** — attacks, grinds, sacrifices, or plays solid depending on its configured playstyle
- **Adapts across a full session** — memory, fatigue simulation, and Elo self-calibration across multiple games

---

## Architecture Overview

```
GameAdapter (Chess.com CDP | Lichess Bot | Terminal Relay | Local Harness)
   |
   v
GameState { board, move_history, clocks, self_elo, opp_elo }
   |
   |---> Maia3Engine (79M params) ------> Move Distribution (all legal moves + probabilities)
   |                                           |
   |---> PlaystyleVector / Archetype ----------+---> Re-weighted Distribution & Top-p Sampling
   |                                           |
   |---> Shannon Entropy H(P) ----------------+---> Position Complexity Score
   |                                           |
   |---> PacingModel (clock-aware) -----------+---> Sampled Think Time (seconds)
   |
   v
Scheduler -------------------------------------------------> Final Delay (with clock safety floor)
   |
   v
Decision { move, delay_s, entropy, top_p }
   |
   v
MoveExecutor (DOM Click / UCI Command / Relay Output)
```

---

## Features

### 1. Maia-3 Neural Move Engine

HLC uses **Maia-3-79M**, the largest model in the Maia-3 family (Chessformer, ICLR 2026), trained on tens of millions of real human games from Lichess and Chess.com.

- **Skill-conditioned**: A single model covers all Elo levels. Pass your target rating and it adjusts move quality accordingly.
- **Full move distribution**: Returns probabilities across all legal moves, not just the top pick. HLC samples from this distribution like a real human.
- **Reliable range**: 1000–2400 Elo. Best human fidelity in the 1200–2200 band.

---

### 2. Human-Like Timing System

HLC does not use fixed or uniformly random delays. Every move delay is calculated from three live signals:

| Signal | Effect |
|--------|--------|
| **Shannon Entropy H(P)** | High entropy (many plausible moves) = longer think time |
| **Top move probability** | Obvious best move = shorter reaction |
| **Live clock reading** | As time runs low, pacing accelerates automatically |

Move delays span a wide, natural distribution — from **0.1s** instant recaptures to **15+ seconds** on critical tactical decisions.

---

### 3. Ten Playstyle Personalities

Each style is a vector of 8 independent axes that bias move sampling. Configurable at launch.

| Playstyle | Philosophy | Signature |
|-----------|------------|-----------|
| `rising_fire` | Dynamic Hybrid | Attack-first, seeks complications, modern aggressive balance |
| `tal` | Romantic Sacrifice | Max volatility, speculative piece attacks, pure chaos |
| `tal_reformed` | Mature Tactical | Calculated aggression with grounded technique |
| `petrosian` | Iron Defense | Prophylaxis, fortresses, opponent error capitalization |
| `karpov` | Positional Boa | Squeeze play, restriction, grinding small advantages |
| `kasparov` | Explosive Dynamism | Sharp kingside initiatives, piece activity over material |
| `carlsen` | Pragmatic Grind | Ruthless endgame precision, relentless pressure |
| `iron_throne` | Fortified Grinder | Low blunder rate, maximum solidity, patience |
| `balanced_evolution` | Adaptive | Shifts posture based on match score and game phase |
| `wildcard` | Creative Entropy | Unconventional openings, unpredictable candidate selection |

---

### 4. Multi-Game Session Management

HLC runs an entire session autonomously:

- **Auto-detects game start/end** via DOM and board state monitoring
- **Survives rematch, new game, and any navigation** — reconnects automatically
- **MetaController**: In autonomous mode, self-calibrates Elo and adapts playstyle based on results
- **Session memory**: Tracks fatigue, streaks, and opponent patterns in SQLite

---

### 5. Desktop GUI Dashboard

- Single-click launch via `run.bat`
- Live telemetry: White/Black clocks, active playstyle, Shannon entropy, move delay
- One-click CDP browser launch
- Autonomous mode toggle — hands-free multi-game sessions
- Session analytics: win rate, avg think time, accuracy trends

---

### 6. Multi-Platform Support

| Platform | Mode |
|----------|------|
| **Chess.com** | CDP browser attachment (Opera) |
| **Lichess** | Official Bot API adapter |
| **Offline** | Terminal Relay CLI + local Stockfish harness |

---

## Installation

### Prerequisites

- **Python 3.11** (64-bit) — [python.org](https://www.python.org/)
- **Opera Browser** — [opera.com](https://www.opera.com/)
- Windows 10/11 (Linux/macOS supported for CLI/Relay modes)

### 1. Clone the repository

```powershell
git clone https://github.com/nabinnnnwongism/hlc-human-like-chess.git
cd hlc-human-like-chess
```

### 2. Create virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install --upgrade pip
pip install -e _maia3_repo
pip install -e .
```

### 4. Pre-cache model weights

```powershell
maia3-cache --model maia3-79m
```

Downloads the Maia-3-79M weights from Hugging Face (~300MB). Only needed once.

---

## Usage

### Option A — Desktop GUI (Recommended)

```powershell
python scripts/launch_gui.py
# or simply double-click run.bat
```

1. Click **"Launch Browser with CDP"** in the GUI
2. Log into [chess.com](https://chess.com) in the opened Opera window
3. Navigate to **Play vs Computer** or **Play Online**
4. Select Elo, playstyle, and mode
5. Click **START AGENT**

---

### Option B — Command Line (Chess.com)

**Step 1:** Launch Opera with remote debugging.

Double-click `start_opera_cdp.bat`, or manually:

```powershell
"C:\Users\[YOU]\AppData\Local\Programs\Opera\opera.exe" --remote-debugging-port=9222 --user-data-dir="C:\hlc_opera_profile"
```

**Step 2:** Log into chess.com in that Opera window and start or join a game.

**Step 3:** Run HLC:

```powershell
# Default (1500 Elo, rising_fire)
$env:PYTHONPATH = "src"; .venv\Scripts\python.exe scripts\chesscom_bot.py --cdp-port 9222

# Custom Elo and playstyle
$env:PYTHONPATH = "src"; .venv\Scripts\python.exe scripts\chesscom_bot.py --cdp-port 9222 --elo 1850 --playstyle kasparov

# Autonomous self-calibrating mode
$env:PYTHONPATH = "src"; .venv\Scripts\python.exe scripts\chesscom_bot.py --cdp-port 9222 --autonomous
```

**Full argument reference:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--cdp-port` | `9222` | Port Opera is running on |
| `--elo` | `1500` | Maia-3 skill level (800–2400) |
| `--playstyle` | `rising_fire` | One of the 10 archetypes listed above |
| `--autonomous` | off | Enables MetaController self-calibration |
| `--color` | `random` | Force `white` or `black` |

---

### Option C — Terminal Relay (Offline Testing)

```powershell
$env:PYTHONPATH = "src"; .venv\Scripts\python.exe scripts\relay_cli.py --color white --time 180 --inc 2 --self-elo 1600
```

**In-session commands:**

| Command | Description |
|---------|-------------|
| `e4` or `e2e4` | Play opponent move (SAN or UCI) |
| `e4 2:55 2:58` | Play move with explicit clock times |
| `undo` | Take back the last move |
| `fen <FEN>` | Sync board to a specific position |
| `board` | Print current board and clock state |
| `quit` | End the session |

---

## Configuration

Edit [`config/default.yaml`](config/default.yaml) to change system defaults:

```yaml
move_engine:
  name: "maia3"
  model: "maia3-79m"       # maia3-5m | maia3-23m | maia3-79m
  elo: 1500
  temperature: 1.0
  top_p: 1.0

timing_model:
  backend: "heuristic"
  heuristic:
    base_fraction: 0.035
    sigma: 0.65
    entropy_weight: 0.3
    top_move_weight: -0.2

scheduler:
  move_overhead: 0.1
  safety_margin: 0.5
  time_trouble_threshold: 10.0
```

---

## Building a Standalone Executable

```powershell
python scripts/build_exe.py
```

Output: `dist/HLC_Native_Agent.exe`

---

## Testing

```powershell
pytest -v
```

All 87 tests pass across:

| Test File | What It Covers |
|-----------|---------------|
| `test_agent.py` | Memory, streak tracking, fatigue |
| `test_chesscom_adapter.py` | Clock sync, color detection, URL routing |
| `test_lichess_bot_engine.py` | Lichess Bot API adapter |
| `test_move_engine.py` | Maia-3 forward pass, sampling |
| `test_playstyle.py` | Archetype vectors, cosine similarity |
| `test_relay_cli.py` | UCI commands, PGN logging |
| `test_timing_and_scheduler.py` | Think times, clock safety |

---

## License

**HLC Personal Use License v1.0** — See [`LICENSE`](LICENSE)

> ✅ You may download and run this software for personal, non-commercial use.
>
> ❌ You may NOT modify, redistribute, sell, or use this software in any commercial product.

---

## Third-Party Credits

| Component | License | Link |
|-----------|---------|------|
| Maia-3 (UofTCSSLab) | AGPL-3.0 | [github.com/CSSLab/maia3](https://github.com/CSSLab/maia3) |
| python-chess (Niklas Fiekas) | GPL-3.0 | [github.com/niklasf/python-chess](https://github.com/niklasf/python-chess) |
| lichess-bot | AGPL-3.0 | [github.com/lichess-bot-devs/lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) |
| Lichess Open Database | CC0 1.0 | [database.lichess.org](https://database.lichess.org) |

---

## Disclaimer

> ⚠️ This software is intended for **research into human behavioral modeling** and testing against computer bots only.
>
> Never use this tool against human opponents without their knowledge. Automated play against unsuspecting humans violates the Terms of Service of chess platforms.
