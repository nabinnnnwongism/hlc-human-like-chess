# Project Human Chess (HLC) — System Overview & Evolution

## 1. Executive Summary & Vision

**Project Human Chess (HLC)** is an advanced autonomous chess agent designed to play chess in a manner that is indistinguishable from human players. Unlike conventional chess engines (such as Stockfish or Leela Chess Zero) whose objective is to compute the objective mathematically optimal move and maximize centipawn advantage, HLC's objective is **behavioral fidelity**:

- It thinks like a human (dynamic cognitive delays modulated by position complexity, time pressure, and board tension).
- It selects moves like a human (leveraging the Maia-3 deep neural network trained on millions of real online games across specific Elo brackets).
- It possesses human-like personalities (distinct playstyle archetypes ranging from hyper-aggressive romantics to stubborn positional grinders).
- It experiences human-like conditions (cognitive fatigue, momentum, tilt, and contextual blunder plausibility).
- It interfaces with platforms naturally (DOM screen reading, human mouse micro-delays, and idle cursor drifts via Chrome DevTools Protocol).

---

## 2. Project Evolution & Major Milestones

The system evolved through six distinct architectural phases:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            HLC EVOLUTION ROADMAP                            │
├─────────────────┬───────────────────┬───────────────────┬───────────────────┤
│ Phase 1 & 2     │ Phase 3           │ Phase 4           │ Phase 5 & 6       │
│ Core Engine &   │ Playstyle System  │ Autonomous Agent  │ Live Automation & │
│ Timing Model    │ & Archetypes      │ & MetaController  │ Desktop Dashboard │
└─────────────────┴───────────────────┴───────────────────┴───────────────────┘
```

### Phase 1: Core Engine & Timing Foundation
- **Maia-3 Neural Network Integration (`Maia3Engine`)**: Replaced traditional alpha-beta search with a neural policy network trained on real human games. Evaluates legal moves and outputs a probability distribution mirroring what a human of a specific Elo rating (from 1100 to 1900+) would choose.
- **Probabilistic Move Sampling**: Built-in support for temperature scaling, top-p (nucleus) sampling, and fallback verification against python-chess legal move structures.
- **Policy Entropy Metric**: Measures position ambiguity. When one move is obvious, entropy is low; when many plausible options exist, entropy spikes.
- **Human Timing Engine (`HeuristicBaseline`)**: Instead of static or random delays, think times are calculated dynamically based on:
  - Phase of game (opening book moves are fast; complex middlegame positions require thought; endgames depend on piece count).
  - Clock pressure (rapid decay as time runs out).
  - Tactical tension and move entropy.
- **Fail-Safe Scheduler (`Scheduler`)**: Enforces clock fraction limits, blitz safety thresholds (never flagging on sub-second clocks), and move overhead compensation.

### Phase 2: Lichess Bot Adapter & MultiPV
- **`lichess-bot` Homemade Protocol Hook (`HLCEngine`)**: Created an adapter interfacing directly with the standard `lichess-bot` client.
- **MultiPV Candidate Generation**: Extracts top candidate moves from Maia-3 to provide nuanced candidate pools for style evaluation.
- **Time Limit & Clock Synchronization**: Synchronizes external clock limits (`chess.engine.Limit`) with HLC internal `GameState`.

### Phase 3: Playstyle Personalities & Behavioral Modulation
- **Theoretical Grounding**: Implemented behavioral archetypes inspired by Grandmaster Lars Bo Hansen’s four human player classifications (Pragmatist, Classical, Romantic, Activist).
- **Core Archetypes Implemented**:
  - `rising_fire`: Aggressive, tactical, attacking pressure with king attacks.
  - `tal`: Romantic, high tactical alertness, sacrifice-tolerant, sharp lines.
  - `karpov`: Solid, positional, prophylactic, low-risk, error-free.
  - `petrosian`: Defensive, fortress-building, counter-punching.
  - `capablanca`: Harmonious, piece-centralizing, rapid simplified endgames.
  - `wildcard`: Dynamic hybrid varying tendencies across phases.
- **Mathematical Style Vectors (`StyleVector`)**: Moves from Maia-3 are re-scored using weighted heuristics:
  ```
  score(m) = log(P_maia(m)) + w_tactical * S_tactical(m) + w_positional * S_positional(m) + ...
  ```

### Phase 4: Autonomous Native Agent & Meta-Controller
- **Persistent SQLite Memory (`AgentMemory`)**: Local database tracking complete session analytics:
  - Game outcomes (win, loss, draw).
  - Ply-by-ply think times and clock states.
  - Move policy entropy and top-move agreement.
  - Historical Elo trajectories and opponents faced.
- **Autonomous Meta-Controller (`MetaController`)**:
  - **Self-Calibrating Elo**: Detects whether the account is new or unrated; automatically adapts target Elo to opponent strength; scales down slightly on win streaks (anti-cheat evasion) and recalibrates on loss streaks.
  - **Cognitive Fatigue Engine**: Tracks continuous session duration and games played; gradually increases blunder probability and slightly shifts think times as fatigue accumulates, recovering during idle rest intervals.
  - **Human Blunder Plausibility**: Evaluates whether a blunder is realistic given clock pressure, check status, and position entropy (avoiding unnatural blunders in trivial positions).

### Phase 5: Chess.com Live Adapter (CDP Automation)
- **Zero-Extension CDP Attachment (`ChessDotComBrowser`)**: Uses the Chrome DevTools Protocol to attach directly to an active browser window (Opera GX, Google Chrome, or Brave) on port 9222.
- **DOM Board Reader (`BoardReader`)**: Extracts board state from Chess.com's dynamic DOM elements (custom elements, coordinate systems, clocks, turn indicators, and player headers) and converts them into standard FEN strings.
- **Human-Like Move Execution (`MoveExecutor`)**: Converts UCI moves to coordinate clicks with humanized micro-delays between mouse down, mouse up, and square selections.
- **Idle Cursor Daemon (`IdleCursorDaemon`)**: Background daemon simulating natural mouse fidgeting and hovering across the board during the opponent's turn.
- **Multi-Game Session Orchestrator (`chesscom_bot.py`)**: Persistent game loop detecting match starts, player color changes on rematch, game over modals, and navigation across lobby pages.

### Phase 6: Desktop Dashboard GUI & Standalone Packaging
- **CustomTkinter GUI (`app.py` & `launch_gui.py`)**:
  - Modern dark-themed dashboard.
  - Autonomous vs. Manual control toggle.
  - Real-time telemetry: Live Game State, Detected Elo, Target Calibrated Elo, Active Playstyle, and Session Fatigue bar.
  - Embedded real-time console log stream.
  - 1-Click Browser CDP Launcher.
  - Database clear and match analytics refresh.
- **Packaging Pipeline (`build_exe.py` & `HLC_Native_Agent.spec`)**: Configured PyInstaller bundle specification to compile the entire suite into a standalone Windows executable.

---

## 3. Architecture & Codebase Map

```
Project Human chess/
├── config/
│   ├── default.yaml               # Core default parameters (scheduler, engine, paths)
│   └── lichess_bot_config.yml     # Configuration for lichess-bot integration
├── data/
│   └── hlc_memory.db              # SQLite persistence for games, plies, and ratings
├── scripts/
│   ├── launch_gui.py              # Entry point for Desktop CustomTkinter App
│   ├── chesscom_bot.py            # Battle-tested CLI & subprocess bot worker
│   ├── build_exe.py               # PyInstaller standalone packager
│   ├── phase3_harness.py          # Benchmark test harness for playstyles
│   └── relay_cli.py               # Standalone stdin/stdout UCI relay
├── src/hlc/
│   ├── core.py                    # BotCore orchestrator (combines move engine + timing)
│   ├── scheduler.py               # Clock safety, think-time bounding & scheduling
│   ├── playstyle.py               # Style vectors, Lars Bo Hansen archetypes & scoring
│   ├── types.py                   # Dataclasses (GameState, Decision, Limit, etc.)
│   ├── agent/
│   │   ├── memory.py              # SQLite storage for sessions, games, and moves
│   │   └── meta_controller.py     # Elo auto-calibration, fatigue & blunder plausibility
│   ├── engines/
│   │   └── maia3.py               # Maia-3 deep neural network move prediction
│   ├── timing/
│   │   ├── baselines.py           # HeuristicBaseline human delay modeling
│   │   └── base.py                # Abstract timing interface
│   ├── adapters/
│   │   ├── lichess_bot_engine.py  # Lichess homemade engine adapter
│   │   └── chesscom/
│   │       ├── browser.py         # Playwright CDP session manager
│   │       ├── board_reader.py    # DOM to FEN / clock / player parser
│   │       ├── bot.py             # Chess.com game-loop orchestrator
│   │       ├── move_executor.py   # DOM coordinate clicker with micro-delays
│   │       ├── idle_cursor.py     # Background cursor wander daemon
│   │       └── session_manager.py # Cookie & profile session bridge
│   └── gui/
│       └── app.py                 # CustomTkinter Desktop Application
└── tests/                         # 80 automated unit & integration test cases
```

---

## 4. Verification & Test Coverage

The test suite covers 80 automated tests spanning all core mathematical and operational modules:
- `test_agent.py`: Memory recording, streak adaptation, fatigue accumulation.
- `test_lichess_bot_engine.py`: Clock conversions, move generation, adapter hooks.
- `test_local_harness.py`: Local simulated game loops and time safety.
- `test_move_engine.py`: Maia-3 inference, temperature scaling, top-p sampling.
- `test_playstyle.py`: Vector math, archetype application, entropy weighting.
- `test_relay_cli.py`: CLI UCI commands and argument handling.
- `test_timing_and_scheduler.py`: Heuristic think times, blitz clock protection, overhead offsets.

All 80 tests currently pass.
