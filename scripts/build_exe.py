"""build_exe.py — Compile HLC Native Agent into a standalone Windows .exe.

Usage:
    python scripts/build_exe.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def build():
    root = Path(__file__).resolve().parent.parent
    dist_dir = root / "dist"
    build_dir = root / "build"
    src_dir = root / "src"

    print("=== Building HLC Native Agent Standalone Executable ===")
    print(f"Project root: {root}")

    maia3_dir = root / "_maia3_repo"

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name=HLC_Native_Agent",
        "--onefile",
        "--windowed",
        "--noconfirm",
        f"--paths={src_dir}",
        f"--paths={maia3_dir}",
        f"--paths={root}",
        f"--paths={root / 'scripts'}",
        "--clean",
        "--collect-all=customtkinter",
        "--collect-all=playwright",
        "--collect-all=maia3",
        "--collect-all=hlc",
        "--collect-all=scripts",
        "--hidden-import=maia3",
        "--hidden-import=maia3.dataset",
        "--hidden-import=maia3.model_registry",
        "--hidden-import=maia3.models",
        "--hidden-import=maia3.presets",
        "--hidden-import=maia3.uci",
        "--hidden-import=maia3.utils",
        "--hidden-import=chess",
        "--hidden-import=chess.pgn",
        "--hidden-import=sqlite3",
        "--hidden-import=scripts",
        "--hidden-import=scripts.chesscom_bot",
        "--hidden-import=chesscom_bot",
        str(root / "scripts" / "launch_gui.py"),
    ]

    print("Running PyInstaller command...")
    print(" ".join(cmd))
    res = subprocess.run(cmd, cwd=str(root))

    if res.returncode == 0:
        exe_path = dist_dir / "HLC_Native_Agent.exe"
        print("\n" + "=" * 65)
        print(" [SUCCESS] Standalone Executable built successfully!")
        print(f" Location: {exe_path}")
        print(" You can double-click this .exe directly to launch HLC.")
        print("=" * 65 + "\n")
    else:
        print(f"\n[ERROR] PyInstaller failed with exit code {res.returncode}")


if __name__ == "__main__":
    build()
