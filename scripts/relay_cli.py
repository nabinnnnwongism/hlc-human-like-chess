#!/usr/bin/env python3
"""Launcher script for HLC Manual Relay CLI.

Usage:
  python scripts/relay_cli.py [--color white|black] [--time 180] [--inc 0] [--self-elo 1500]

⚠️  FAIR PLAY NOTICE:
Intended for games against built-in computer Bots only, never against human opponents.
"""

from hlc.cli.relay import main

if __name__ == "__main__":
    main()
