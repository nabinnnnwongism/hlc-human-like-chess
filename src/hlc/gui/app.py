"""app.py — Desktop GUI Dashboard for HLC Autonomous Native Agent.

Built with CustomTkinter for a modern dark-theme user experience.
Allows zero-touch autonomous mode, manual playstyle override, live telemetry,
and match performance analytics.
"""

from __future__ import annotations

import datetime
import logging
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import customtkinter as ctk

from hlc.agent.memory import AgentMemory
from hlc.agent.meta_controller import MetaController
from hlc.playstyle import PLAYSTYLE_NAMES, describe_style, get_style_vector

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class HLCDesktopApp(ctk.CTk):
    """Main Desktop Window for the Human-Like Chess Native Agent."""

    def __init__(self) -> None:
        super().__init__()

        self.title("HLC - Human-Like Chess [Autonomous Native Agent]")
        self.geometry("900x700")
        self.minsize(850, 600)

        # Core Agent references
        self.memory = AgentMemory()
        self.meta_controller = MetaController(memory=self.memory)

        # Background worker state
        self._bot_thread: threading.Thread | None = None
        self._bot_proc: subprocess.Popen | None = None
        self._stop_event = threading.Event()
        self._is_running = False

        self._build_ui()
        self._start_periodic_refresh()

    def _build_ui(self) -> None:
        # Top Header Bar
        header = ctk.CTkFrame(self, height=60, corner_radius=0)
        header.pack(fill="x", side="top", padx=0, pady=0)

        title_lbl = ctk.CTkLabel(
            header,
            text="HLC NATIVE AGENT",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        title_lbl.pack(side="left", padx=20, pady=12)

        self.status_badge = ctk.CTkLabel(
            header,
            text="STANDBY",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#888888",
        )
        self.status_badge.pack(side="right", padx=20, pady=12)

        # Tabview
        self.tabview = ctk.CTkTabview(self, corner_radius=8)
        self.tabview.pack(fill="both", expand=True, padx=15, pady=10)

        self.tab_dashboard = self.tabview.add("  Live Dashboard  ")
        self.tab_analytics = self.tabview.add("  Analytics & Memory  ")
        self.tab_settings = self.tabview.add("  Settings  ")

        self._build_dashboard_tab()
        self._build_analytics_tab()
        self._build_settings_tab()

    # ── Dashboard Tab ──────────────────────────────────────────────────────────

    def _build_dashboard_tab(self) -> None:
        dash = self.tab_dashboard

        # Left column: Mode & Controls
        left_col = ctk.CTkFrame(dash, width=320)
        left_col.pack(side="left", fill="y", padx=10, pady=10)

        mode_lbl = ctk.CTkLabel(
            left_col,
            text="OPERATING MODE",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#3B82F6",
        )
        mode_lbl.pack(anchor="w", padx=15, pady=(15, 5))

        self.mode_var = ctk.StringVar(value="native")
        self.radio_native = ctk.CTkRadioButton(
            left_col,
            text="Autonomous Native Agent",
            variable=self.mode_var,
            value="native",
            command=self._on_mode_change,
        )
        self.radio_native.pack(anchor="w", padx=15, pady=6)

        native_hint = ctk.CTkLabel(
            left_col,
            text="- Learns session performance\n- Self-calibrates target ELO\n- Pragmatic Dynamic playstyle\n- Zero manual tweaking",
            font=ctk.CTkFont(size=11),
            text_color="#9CA3AF",
            justify="left",
        )
        native_hint.pack(anchor="w", padx=35, pady=(0, 10))

        self.radio_manual = ctk.CTkRadioButton(
            left_col,
            text="Manual Archetype Override",
            variable=self.mode_var,
            value="manual",
            command=self._on_mode_change,
        )
        self.radio_manual.pack(anchor="w", padx=15, pady=6)

        self.style_dropdown = ctk.CTkOptionMenu(
            left_col,
            values=PLAYSTYLE_NAMES,
            state="disabled",
            command=self._on_style_selected,
        )
        self.style_dropdown.set("rising_fire")
        self.style_dropdown.pack(fill="x", padx=15, pady=(5, 15))

        # Manual ELO Slider (active only in manual mode)
        self.elo_lbl = ctk.CTkLabel(left_col, text="Manual Target ELO: 1800", font=ctk.CTkFont(size=12))
        self.elo_lbl.pack(anchor="w", padx=15, pady=(5, 0))

        self.elo_slider = ctk.CTkSlider(
            left_col,
            from_=1100,
            to=1900,
            number_of_steps=16,
            state="disabled",
            command=self._on_elo_slider_change,
        )
        self.elo_slider.set(1800)
        self.elo_slider.pack(fill="x", padx=15, pady=(5, 20))

        # Action Buttons
        self.btn_toggle = ctk.CTkButton(
            left_col,
            text="START AGENT",
            height=40,
            fg_color="#10B981",
            hover_color="#059669",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._toggle_agent,
        )
        self.btn_toggle.pack(fill="x", padx=15, pady=(10, 5))

        self.btn_launch_browser = ctk.CTkButton(
            left_col,
            text="Launch Browser with CDP",
            height=36,
            fg_color="#1D4ED8",
            hover_color="#1E40AF",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._launch_browser_cdp,
        )
        self.btn_launch_browser.pack(fill="x", padx=15, pady=(0, 5))

        launch_hint = ctk.CTkLabel(
            left_col,
            text="Auto-opens your browser with CDP\nenabled so HLC can attach to it.",
            font=ctk.CTkFont(size=10),
            text_color="#6B7280",
            justify="left",
        )
        launch_hint.pack(anchor="w", padx=15, pady=(0, 10))

        # Right column: Telemetry Card & Live Log
        right_col = ctk.CTkFrame(dash)
        right_col.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        telemetry_lbl = ctk.CTkLabel(
            right_col,
            text="LIVE AGENT TELEMETRY",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#3B82F6",
        )
        telemetry_lbl.pack(anchor="w", padx=15, pady=(15, 8))

        # Info Grid
        info_frame = ctk.CTkFrame(right_col, fg_color="#1E293B")
        info_frame.pack(fill="x", padx=15, pady=5)

        self.lbl_game_status = ctk.CTkLabel(
            info_frame,
            text="Game State:  Idle (Waiting for chess.com match)",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.lbl_game_status.pack(anchor="w", padx=12, pady=4)

        self.lbl_detected_ratings = ctk.CTkLabel(
            info_frame,
            text="Detected ELO:  Self: --  |  Opponent: --",
            font=ctk.CTkFont(size=12),
        )
        self.lbl_detected_ratings.pack(anchor="w", padx=12, pady=4)

        self.lbl_target_elo = ctk.CTkLabel(
            info_frame,
            text="Calibrated Target ELO:  1800 (Auto)",
            font=ctk.CTkFont(size=12),
        )
        self.lbl_target_elo.pack(anchor="w", padx=12, pady=4)

        self.lbl_active_style = ctk.CTkLabel(
            info_frame,
            text="Active Style:  Native Pragmatic Dynamic",
            font=ctk.CTkFont(size=12),
        )
        self.lbl_active_style.pack(anchor="w", padx=12, pady=4)

        self.lbl_clock = ctk.CTkLabel(
            info_frame,
            text="Digital Clock:  --:--  vs  --:--",
            font=ctk.CTkFont(size=12),
        )
        self.lbl_clock.pack(anchor="w", padx=12, pady=4)

        self.lbl_time_control = ctk.CTkLabel(
            info_frame,
            text="Time Control:  Auto (Detecting)",
            font=ctk.CTkFont(size=12),
        )
        self.lbl_time_control.pack(anchor="w", padx=12, pady=4)

        # Fatigue Bar
        fatigue_box = ctk.CTkFrame(info_frame, fg_color="transparent")
        fatigue_box.pack(fill="x", padx=12, pady=6)
        ctk.CTkLabel(fatigue_box, text="Session Fatigue:", font=ctk.CTkFont(size=11)).pack(side="left")
        self.fatigue_bar = ctk.CTkProgressBar(fatigue_box, width=150)
        self.fatigue_bar.set(0.0)
        self.fatigue_bar.pack(side="left", padx=10)
        self.lbl_fatigue_val = ctk.CTkLabel(fatigue_box, text="0%", font=ctk.CTkFont(size=11))
        self.lbl_fatigue_val.pack(side="left")

        # Live Console Output Box
        log_lbl = ctk.CTkLabel(right_col, text="ACTIVITY LOG", font=ctk.CTkFont(size=12, weight="bold"))
        log_lbl.pack(anchor="w", padx=15, pady=(15, 4))

        self.log_textbox = ctk.CTkTextbox(right_col, height=180, font=ctk.CTkFont(family="Consolas", size=11))
        self.log_textbox.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self._log("HLC Native Agent initialized. Ready to attach to Opera / Chrome.")

    # ── Analytics Tab ──────────────────────────────────────────────────────────

    def _build_analytics_tab(self) -> None:
        an = self.tab_analytics

        # Summary Metrics Header Cards
        cards_frame = ctk.CTkFrame(an, fg_color="transparent")
        cards_frame.pack(fill="x", padx=15, pady=15)

        self.card_games = self._create_metric_card(cards_frame, "TOTAL GAMES", "0")
        self.card_winrate = self._create_metric_card(cards_frame, "WIN RATE", "0.0%")
        self.card_accuracy = self._create_metric_card(cards_frame, "AVG ACCURACY", "0.0%")
        self.card_think = self._create_metric_card(cards_frame, "AVG THINK TIME", "0.0s")

        # Recent games table
        history_lbl = ctk.CTkLabel(an, text="RECENT MATCH HISTORY", font=ctk.CTkFont(size=13, weight="bold"))
        history_lbl.pack(anchor="w", padx=15, pady=(10, 5))

        self.history_scroll = ctk.CTkScrollableFrame(an, height=320)
        self.history_scroll.pack(fill="both", expand=True, padx=15, pady=(0, 15))

    def _create_metric_card(self, parent: ctk.CTkFrame, title: str, value: str) -> ctk.CTkLabel:
        box = ctk.CTkFrame(parent, fg_color="#1E293B", corner_radius=8, width=180, height=80)
        box.pack(side="left", expand=True, fill="both", padx=6)
        ctk.CTkLabel(box, text=title, font=ctk.CTkFont(size=11), text_color="#9CA3AF").pack(pady=(12, 2))
        val_lbl = ctk.CTkLabel(box, text=value, font=ctk.CTkFont(size=20, weight="bold"), text_color="#3B82F6")
        val_lbl.pack(pady=(0, 12))
        return val_lbl

    # ── Settings Tab ───────────────────────────────────────────────────────────

    def _build_settings_tab(self) -> None:
        st = self.tab_settings

        container = ctk.CTkScrollableFrame(st)
        container.pack(fill="both", expand=True, padx=20, pady=20)

        # CDP Port
        ctk.CTkLabel(container, text="Browser Remote Debugging Port (CDP):", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", pady=(10, 4))
        self.cdp_entry = ctk.CTkEntry(container, width=200)
        self.cdp_entry.insert(0, "9222")
        self.cdp_entry.pack(anchor="w", pady=(0, 4))
        ctk.CTkLabel(
            container,
            text="Click \"Launch Browser with CDP\" on the Dashboard to open Opera (Standard) / Chrome\nautomatically with this port enabled.",
            font=ctk.CTkFont(size=10),
            text_color="#6B7280",
            justify="left",
        ).pack(anchor="w", pady=(0, 12))

        # Preferred Browser Choice
        ctk.CTkLabel(container, text="Preferred Browser:", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", pady=(5, 4))
        self.browser_choice_dropdown = ctk.CTkOptionMenu(
            container,
            values=["Opera (Standard)", "Opera GX", "Google Chrome", "Brave", "Microsoft Edge", "Auto (Detect)"],
            width=240,
        )
        self.browser_choice_dropdown.set("Opera (Standard)")
        self.browser_choice_dropdown.pack(anchor="w", pady=(0, 15))

        # Custom browser path (optional)
        ctk.CTkLabel(container, text="Custom Browser Executable Path (optional):", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", pady=(5, 4))
        self.browser_path_entry = ctk.CTkEntry(container, width=420, placeholder_text="Leave blank to use preferred browser above")
        self.browser_path_entry.pack(anchor="w", pady=(0, 15))

        # Model Choice
        ctk.CTkLabel(container, text="Maia Engine Model:", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", pady=(5, 4))
        self.model_dropdown = ctk.CTkOptionMenu(container, values=["maia3-79m", "maia3-23m", "maia3-5m"])
        self.model_dropdown.set("maia3-79m")
        self.model_dropdown.pack(anchor="w", pady=(0, 15))

        # Time Control Pacing
        ctk.CTkLabel(container, text="Bot Game Time Control (when unclocked):", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", pady=(5, 4))
        self.time_control_dropdown = ctk.CTkOptionMenu(
            container,
            values=["Auto (Detect from Page)", "Rapid (10 min)", "Rapid (15|10)", "Blitz (3 min)", "Blitz (5 min)", "Bullet (1 min)", "Classical (30 min)"],
            width=240,
        )
        self.time_control_dropdown.set("Auto (Detect from Page)")
        self.time_control_dropdown.pack(anchor="w", pady=(0, 15))

        # Idle Cursor Switch
        self.idle_cursor_switch = ctk.CTkSwitch(container, text="Enable Idle Human Cursor Movement (Anti-Cheat Stealth)")
        self.idle_cursor_switch.select()
        self.idle_cursor_switch.pack(anchor="w", pady=10)

        # Clear Database Button
        ctk.CTkLabel(container, text="Data Management:", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", pady=(20, 4))
        btn_reset_db = ctk.CTkButton(
            container,
            text="Clear Match History Database",
            fg_color="#EF4444",
            hover_color="#DC2626",
            width=220,
            command=self._clear_database,
        )
        btn_reset_db.pack(anchor="w", pady=5)

    # ── UI Event Handlers ──────────────────────────────────────────────────────

    def _on_mode_change(self) -> None:
        if self.mode_var.get() == "native":
            self.style_dropdown.configure(state="disabled")
            self.elo_slider.configure(state="disabled")
            self.lbl_active_style.configure(text="Active Style:  Native Pragmatic Dynamic (Autonomous)")
        else:
            self.style_dropdown.configure(state="normal")
            self.elo_slider.configure(state="normal")
            self.lbl_active_style.configure(text=f"Active Style:  Manual ({self.style_dropdown.get()})")

    def _on_style_selected(self, val: str) -> None:
        self.lbl_active_style.configure(text=f"Active Style:  Manual ({val})")

    def _on_elo_slider_change(self, val: float) -> None:
        elo_int = int(val)
        self.elo_lbl.configure(text=f"Manual Target ELO: {elo_int}")
        self.lbl_target_elo.configure(text=f"Calibrated Target ELO:  {elo_int} (Manual)")

    def _log(self, msg: str) -> None:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        def _do_log():
            try:
                if self.winfo_exists():
                    self.log_textbox.insert("end", f"[{timestamp}] {msg}\n")
                    self.log_textbox.see("end")
            except Exception:
                pass
        try:
            self.after(0, _do_log)
        except Exception:
            pass

    def _set_status(self, text: str) -> None:
        def _do():
            try:
                if self.winfo_exists():
                    self.lbl_game_status.configure(text=text)
            except Exception:
                pass
        try:
            self.after(0, _do)
        except Exception:
            pass

    def _set_badge(self, text: str, color: str) -> None:
        def _do():
            try:
                if self.winfo_exists():
                    self.status_badge.configure(text=text, text_color=color)
            except Exception:
                pass
        try:
            self.after(0, _do)
        except Exception:
            pass

    def _toggle_agent(self) -> None:
        if not self._is_running:
            self._start_agent()
        else:
            self._stop_agent()

    def _is_port_open(self, port: int, host: str = "127.0.0.1") -> bool:
        """Check if a local TCP port is accepting connections."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                return s.connect_ex((host, port)) == 0
        except Exception:
            return False

    def _find_browser_exe(self) -> str | None:
        """Locate Opera, Opera GX, Brave, Chrome, or Edge on the system."""
        custom_path = ""
        try:
            custom_path = self.browser_path_entry.get().strip()
        except Exception:
            pass
        if custom_path and Path(custom_path).is_file():
            return custom_path

        preferred = "Opera (Standard)"
        try:
            preferred = self.browser_choice_dropdown.get()
        except Exception:
            pass

        loc = os.environ.get("LOCALAPPDATA", "")
        pfiles = os.environ.get("PROGRAMFILES", "")
        pfiles86 = os.environ.get("PROGRAMFILES(X86)", "")

        browser_map: dict[str, list[str]] = {
            "Opera (Standard)": [
                os.path.join(loc, "Programs", "Opera", "opera.exe"),
                os.path.join(loc, "Programs", "Opera", "launcher.exe"),
                os.path.join(pfiles, "Opera", "opera.exe"),
                os.path.join(pfiles86, "Opera", "opera.exe"),
            ],
            "Opera GX": [
                os.path.join(loc, "Programs", "Opera GX", "opera.exe"),
                os.path.join(loc, "Programs", "Opera GX", "launcher.exe"),
                os.path.join(pfiles, "Programs", "Opera GX", "opera.exe"),
            ],
            "Brave": [
                os.path.join(pfiles, "BraveSoftware", "Brave-Browser", "Application", "brave.exe"),
                os.path.join(loc, "BraveSoftware", "Brave-Browser", "Application", "brave.exe"),
            ],
            "Google Chrome": [
                os.path.join(pfiles, "Google", "Chrome", "Application", "chrome.exe"),
                os.path.join(pfiles86, "Google", "Chrome", "Application", "chrome.exe"),
                os.path.join(loc, "Google", "Chrome", "Application", "chrome.exe"),
            ],
            "Microsoft Edge": [
                os.path.join(pfiles86, "Microsoft", "Edge", "Application", "msedge.exe"),
                os.path.join(pfiles, "Microsoft", "Edge", "Application", "msedge.exe"),
            ],
        }

        # If user picked a specific browser, try it first
        if preferred in browser_map:
            for p in browser_map[preferred]:
                if p and Path(p).is_file():
                    return p

        # Otherwise search in order: Standard Opera first, then Opera GX, Brave, Chrome, Edge
        search_order = ["Opera (Standard)", "Opera GX", "Brave", "Google Chrome", "Microsoft Edge"]
        candidates: list[str] = []
        for b_name in search_order:
            candidates.extend(browser_map[b_name])

        # Windows Registry App Paths fallback
        try:
            import winreg
            for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                try:
                    with winreg.OpenKey(hive, r"Software\Microsoft\Windows\CurrentVersion\App Paths") as key:
                        for i in range(winreg.QueryInfoKey(key)[0]):
                            subkey_name = winreg.EnumKey(key, i)
                            if any(b in subkey_name.lower() for b in ("opera", "chrome", "brave", "edge")):
                                try:
                                    with winreg.OpenKey(key, subkey_name) as subkey:
                                        val = winreg.QueryValue(subkey, None)
                                        if val:
                                            cleaned = val.strip('\"')
                                            if cleaned not in candidates:
                                                candidates.append(cleaned)
                                except Exception:
                                    pass
                except Exception:
                    pass
        except Exception:
            pass

        return next((p for p in candidates if p and Path(p).is_file()), None)

    def _get_user_profile_dir(self, browser_exe: str) -> str | None:
        """Find the user's actual browser profile directory (so we reuse their logged-in session)."""
        appdata = os.environ.get("APPDATA", "")
        loc = os.environ.get("LOCALAPPDATA", "")
        exe_lower = browser_exe.lower()

        profile_candidates = []
        if "opera gx" in exe_lower or "opera gx" in Path(browser_exe).parent.name.lower():
            profile_candidates = [
                os.path.join(appdata, "Opera Software", "Opera GX Stable"),
            ]
        elif "opera" in exe_lower:
            profile_candidates = [
                os.path.join(appdata, "Opera Software", "Opera Stable"),
                os.path.join(loc, "Opera Software", "Opera Stable"),
            ]
        elif "brave" in exe_lower:
            profile_candidates = [
                os.path.join(loc, "BraveSoftware", "Brave-Browser", "User Data"),
            ]
        elif "chrome" in exe_lower:
            profile_candidates = [
                os.path.join(loc, "Google", "Chrome", "User Data"),
            ]
        elif "msedge" in exe_lower or "edge" in exe_lower:
            profile_candidates = [
                os.path.join(loc, "Microsoft", "Edge", "User Data"),
            ]

        return next((p for p in profile_candidates if p and Path(p).is_dir()), None)

    def _is_browser_process_running(self, browser_exe: str) -> bool:
        """Check if the given browser is already running as a process."""
        try:
            browser_name = Path(browser_exe).stem.lower()
            result = subprocess.run(
                ["tasklist", "/FI", f"IMAGENAME eq {Path(browser_exe).name}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=3
            )
            return Path(browser_exe).name.lower() in result.stdout.lower()
        except Exception:
            return False

    def _open_tab_in_running_browser(self, browser_exe: str, url: str) -> bool:
        """Open a new tab in an already-running browser instance (single-instance mode)."""
        try:
            # Chromium single-instance: passing a URL when browser is running opens a new tab
            subprocess.Popen(
                [browser_exe, url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            return False

    def _launch_browser_cdp(self, silent: bool = False) -> bool:
        """Smart browser launcher:
        1. If CDP port already open → connect directly (best case, no action needed).
        2. If browser already running WITHOUT CDP → open a chess.com tab in it and
           instruct user to restart with CDP, OR attempt CDP re-launch with real profile.
        3. If browser not running → launch with real user profile + CDP port.
        """
        try:
            port = int(self.cdp_entry.get().strip() or "9222")
        except ValueError:
            port = 9222

        # 1. CDP port already open? Just connect, nothing to do.
        if self._is_port_open(port):
            if not silent:
                self._log(f"Browser already running with CDP on port {port}. Ready!")
            return True

        browser_exe = self._find_browser_exe()
        if not browser_exe:
            self._log(
                "[!] Could not find Opera, Chrome, or Brave automatically.\n"
                "    Please paste the full .exe path in Settings > Custom Browser Path."
            )
            return False

        browser_name = Path(browser_exe).stem

        # IMPORTANT: Always use a DEDICATED HLC profile directory, NOT the user's
        # real profile. This avoids the Chromium SingletonLock — the OS prevents
        # two instances from sharing the same --user-data-dir, so launching with
        # the real profile just opens a tab in the existing window (no CDP port).
        # Login persistence is handled by session_manager (cookie bridge).
        profile_dir = str(Path.home() / ".hlc" / "browser_profile")
        Path(profile_dir).mkdir(parents=True, exist_ok=True)

        # Check if we have a saved session to show a friendlier message
        from hlc.adapters.chesscom.session_manager import has_session
        has_saved = has_session()

        if not silent:
            self._log(
                f"Launching {browser_name} with CDP port {port}..."
            )
            if has_saved:
                self._log("Restoring your saved chess.com session...")
            else:
                self._log(
                    "First run: please log in to chess.com in the browser window.\n"
                    "Your session will be saved automatically for future launches."
                )

        try:
            subprocess.Popen(
                [
                    browser_exe,
                    f"--remote-debugging-port={port}",
                    f"--user-data-dir={profile_dir}",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-features=ChromeWhatsNewUI",
                    "https://www.chess.com",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception as e:
            self._log(f"[!] Failed to launch browser: {e}")
            return False

    def _start_agent(self) -> None:
        self._is_running = True
        self._stop_event.clear()
        self.btn_toggle.configure(text="STOP AGENT", fg_color="#EF4444", hover_color="#DC2626")
        self.status_badge.configure(text="RUNNING", text_color="#10B981")
        self._log("Agent started. Watching browser CDP session...")

        # Spawn bot orchestrator thread
        self._bot_thread = threading.Thread(target=self._run_bot_worker, daemon=True)
        self._bot_thread.start()

    def _stop_agent(self, reason: str = "user") -> None:
        if not self._is_running and reason == "user":
            return
        self._is_running = False
        self._stop_event.set()

        # Terminate any running bot subprocess
        try:
            if self._bot_proc is not None and self._bot_proc.poll() is None:
                self._bot_proc.terminate()
        except Exception:
            pass

        def _do_stop():
            try:
                self.btn_toggle.configure(text="START AGENT", fg_color="#10B981", hover_color="#059669")
                self.status_badge.configure(text="STANDBY", text_color="#888888")
                if reason == "user":
                    self.lbl_game_status.configure(text="Game State:  Stopped by user")
                elif reason == "cdp_fail":
                    self.lbl_game_status.configure(text="Game State:  [!] Browser connection failed")
                elif reason == "error":
                    self.lbl_game_status.configure(text="Game State:  Stopped (worker error)")
                else:
                    self.lbl_game_status.configure(text="Game State:  Standby")
            except Exception:
                pass
        self.after(0, _do_stop)

        if reason == "user":
            self._log("Agent stopped by user.")
        elif reason == "cdp_fail":
            self._log("Agent stopped: Could not connect to browser.")
        elif reason == "error":
            self._log("Agent stopped due to an error.")

    def _clear_database(self) -> None:
        try:
            with self.memory._get_connection() as conn:
                conn.execute("DELETE FROM moves")
                conn.execute("DELETE FROM games")
                conn.commit()
            self._log("Database cleared successfully.")
            self._refresh_analytics()
        except Exception as e:
            self._log(f"Error clearing database: {e}")

    # ── Background Worker Loop ─────────────────────────────────────────────────

    def _run_bot_worker(self) -> None:
        """Background thread: launch browser, then run chesscom_bot.py as a subprocess.

        Playwright sync_api cannot run inside a secondary threading.Thread
        due to Python greenlet thread-affinity constraints, which caused UI freezes
        and event-loop lockups. By running the bot as an isolated subprocess, Playwright
        executes on its own main thread with full stability, while stdout streams
        seamlessly into the GUI dashboard.
        """
        import re
        import sys
        from pathlib import Path as _Path

        try:
            port = int(self.cdp_entry.get().strip() or "9222")
        except ValueError:
            port = 9222

        # Step 1: auto-launch browser if CDP port not yet open
        if not self._is_port_open(port):
            self._log(f"Browser not detected on port {port} -- auto-launching...")
            ok = self._launch_browser_cdp(silent=True)
            if ok:
                self._set_status("Game State:  Browser launching...")
                self._log(f"Waiting for browser on port {port}...")
                t0 = time.time()
                while time.time() - t0 < 15.0 and not self._stop_event.is_set():
                    if self._is_port_open(port):
                        break
                    time.sleep(0.4)

        if self._stop_event.is_set():
            return

        if not self._is_port_open(port):
            self._log(
                f"[!] Port {port} never opened.\n"
                f"   Try: Dashboard -> Launch Browser with CDP, then START AGENT again."
            )
            self.after(0, lambda: self._stop_agent(reason="cdp_fail"))
            return

        # Step 2: collect UI settings
        mode = self.mode_var.get()
        selected_style = self.style_dropdown.get()
        manual_elo = int(self.elo_slider.get())
        model_name = self.model_dropdown.get()
        idle_cursor = bool(self.idle_cursor_switch.get())
        time_control_val = self.time_control_dropdown.get()

        tc_arg = None
        if "10 min" in time_control_val: tc_arg = "10m"
        elif "15" in time_control_val: tc_arg = "15m"
        elif "3 min" in time_control_val: tc_arg = "3m"
        elif "5 min" in time_control_val: tc_arg = "5m"
        elif "1 min" in time_control_val: tc_arg = "1m"
        elif "Classical" in time_control_val: tc_arg = "30m"

        # Step 3: build the CLI command
        is_frozen = getattr(sys, "frozen", False)
        project_root = str(_Path(__file__).resolve().parent.parent.parent.parent)
        bot_script  = str(_Path(project_root) / "scripts" / "chesscom_bot.py")

        if is_frozen:
            cmd = [
                sys.executable,
                "--bot-worker",
                "--cdp-port", str(port),
                "--elo",       str(manual_elo),
                "--model",     model_name,
                "--no-pause",
                "--verbose",
            ]
        else:
            cmd = [
                sys.executable,
                "-u",
                bot_script,
                "--cdp-port", str(port),
                "--elo",       str(manual_elo),
                "--model",     model_name,
                "--no-pause",
                "--verbose",
            ]

        if mode in ("autonomous", "native"):
            cmd.append("--autonomous")
        else:
            cmd.extend(["--playstyle", selected_style])

        if tc_arg:
            cmd.extend(["--time-control", tc_arg])

        if not idle_cursor:
            cmd.append("--no-idle-cursor")

        style_info = "Autonomous (Self-Calibrating)" if mode in ("autonomous", "native") else selected_style
        self._log(f"[>] Starting HLC Bot engine: ELO={manual_elo}  Style={style_info}")
        self._log("  (Playwright isolated in dedicated subprocess -- full UI responsiveness)")
        self._set_status("Game State:  Connected -- waiting for match...")

        try:
            import subprocess as _sp
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            proc = _sp.Popen(
                cmd,
                stdout=_sp.PIPE,
                stderr=_sp.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=project_root if not is_frozen else None,
                env=env,
            )
            self._bot_proc = proc

            # Stream stdout line-by-line into the GUI log
            for raw_line in proc.stdout:
                line = raw_line.rstrip()
                if not line:
                    continue

                # Strip logger timestamp prefix if present (HH:MM:SS  LEVEL  name: msg)
                display = line
                m = re.match(r"^\d{2}:\d{2}:\d{2}\s+\S+\s+\S+:\s*(.*)", line)
                if m:
                    display = m.group(1)

                self._log(display)

                # Update badge/status based on keywords in the output
                ll = line.lower()
                if "[clock]" in ll:
                    # Format: [CLOCK] W: MM:SS | B: MM:SS [TimeControl | DOM/Digital]
                    clock_m = re.search(
                        r"\[clock\]\s+w:\s*(\d+:\d+)\s*\|\s*b:\s*(\d+:\d+)\s*\[(.+?)\s*\|\s*(?:DOM|Digital)",
                        line, re.IGNORECASE
                    )
                    if clock_m:
                        w_str = clock_m.group(1)
                        b_str = clock_m.group(2)
                        tc_str = clock_m.group(3).strip()
                        self.after(0, lambda w=w_str, b=b_str: self.lbl_clock.configure(text=f"Digital Clock:  {w} (W)  vs  {b} (B)"))
                        self.after(0, lambda tc=tc_str: self.lbl_time_control.configure(text=f"Time Control:  {tc}"))
                elif "playing as white" in ll:
                    self._set_badge("PLAYING (WHITE)", "#10B981")
                    self._set_status("Game State:  Playing as White (HLC moves first)")
                elif "playing as black" in ll:
                    self._set_badge("PLAYING (BLACK)", "#F59E0B")
                    self._set_status("Game State:  Playing as Black (Waiting for White move)")
                elif "opponent played:" in ll:
                    opp_m = re.search(r"opponent played:\s*(\S+)", line, re.IGNORECASE)
                    if opp_m:
                        self._set_status(f"Game State:  Opponent played {opp_m.group(1)} -- HLC thinking...")
                elif "page detected:" in ll:
                    page_m = re.search(r"page detected:\s*(.*)", line, re.IGNORECASE)
                    if page_m:
                        self._set_status(f"Page:  {page_m.group(1)}")
                elif "starting game" in ll or ("game " in ll and "starting" in ll):
                    self._set_badge("PLAYING", "#F59E0B")
                    self._set_status(f"Game State:  {display}")
                elif "waiting for" in ll or "watching chess.com" in ll:
                    self._set_badge("RUNNING", "#10B981")
                    self._set_status("Game State:  Watching for game...")
                elif "complete" in ll or "finished" in ll:
                    self._set_badge("RUNNING", "#10B981")
                    self.after(0, self._refresh_analytics)

                if self._stop_event.is_set():
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    break

            proc.wait()
            self._bot_proc = None

        except Exception as e:
            self._log(f"Bot subprocess error: {e}")
            self._bot_proc = None

        if self._is_running:
            self.after(0, lambda: self._stop_agent(reason="worker_exit"))

    # ── Periodic UI Refresh ────────────────────────────────────────────────────

    def _start_periodic_refresh(self) -> None:
        try:
            if not self.winfo_exists():
                return
            self._update_fatigue_display()
            self.after(5000, self._start_periodic_refresh)
        except Exception:
            pass

    def _update_fatigue_display(self) -> None:
        try:
            if not self.winfo_exists():
                return
            fatigue = getattr(self.meta_controller.session, "fatigue", 0.0)
            try:
                self.fatigue_bar.set(fatigue)
            except Exception:
                pass
            try:
                self.lbl_fatigue_val.configure(text=f"{int(fatigue * 100)}%")
            except Exception:
                pass
        except Exception:
            pass

    def _refresh_analytics(self) -> None:
        stats = self.memory.get_stats_summary()
        self.card_games.configure(text=str(stats["total_games"]))
        self.card_winrate.configure(text=f"{stats['winrate_pct']}%")
        self.card_accuracy.configure(text=f"{stats['avg_accuracy']}%")
        self.card_think.configure(text=f"{stats['avg_think_time']}s")

        # Refresh recent games table
        for widget in self.history_scroll.winfo_children():
            widget.destroy()

        games = self.memory.get_recent_games(limit=10)
        if not games:
            ctk.CTkLabel(self.history_scroll, text="No matches recorded yet.", text_color="#9CA3AF").pack(pady=20)
            return

        # Table header
        header_row = ctk.CTkFrame(self.history_scroll, fg_color="#1E293B")
        header_row.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(header_row, text="ID", width=40, font=ctk.CTkFont(weight="bold")).pack(side="left", padx=4)
        ctk.CTkLabel(header_row, text="Color", width=60, font=ctk.CTkFont(weight="bold")).pack(side="left", padx=4)
        ctk.CTkLabel(header_row, text="Opp ELO", width=70, font=ctk.CTkFont(weight="bold")).pack(side="left", padx=4)
        ctk.CTkLabel(header_row, text="Result", width=70, font=ctk.CTkFont(weight="bold")).pack(side="left", padx=4)
        ctk.CTkLabel(header_row, text="Moves", width=60, font=ctk.CTkFont(weight="bold")).pack(side="left", padx=4)
        ctk.CTkLabel(header_row, text="Style", width=120, font=ctk.CTkFont(weight="bold")).pack(side="left", padx=4)
        ctk.CTkLabel(header_row, text="Avg Think", width=80, font=ctk.CTkFont(weight="bold")).pack(side="left", padx=4)

        for g in games:
            row = ctk.CTkFrame(self.history_scroll, fg_color="#0F172A")
            row.pack(fill="x", padx=4, pady=2)

            res_color = "#10B981" if g.result == "win" else ("#EF4444" if g.result == "loss" else "#9CA3AF")
            ctk.CTkLabel(row, text=f"#{g.id}", width=40).pack(side="left", padx=4)
            ctk.CTkLabel(row, text=g.color.capitalize(), width=60).pack(side="left", padx=4)
            ctk.CTkLabel(row, text=str(g.opp_elo), width=70).pack(side="left", padx=4)
            ctk.CTkLabel(row, text=g.result.upper(), width=70, text_color=res_color, font=ctk.CTkFont(weight="bold")).pack(side="left", padx=4)
            ctk.CTkLabel(row, text=str(g.moves_count), width=60).pack(side="left", padx=4)
            ctk.CTkLabel(row, text=g.style_used, width=120).pack(side="left", padx=4)
            ctk.CTkLabel(row, text=f"{g.avg_think_time_s:.1f}s", width=80).pack(side="left", padx=4)


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    app = HLCDesktopApp()
    app.mainloop()


if __name__ == "__main__":
    main()
