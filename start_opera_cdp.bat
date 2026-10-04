@echo off
:: ============================================================
:: start_opera_cdp.bat
:: Launches Standard Opera with remote debugging on port 9222.
:: Opens chess.com automatically.
::
:: Run THIS first, THEN run start_bot.bat in a second terminal.
:: ============================================================

set OPERA_EXE=C:\Users\Admin\AppData\Local\Programs\Opera\opera.exe
set CDP_PORT=9222
set CHESS_URL=https://www.chess.com/play/computer

echo.
echo ============================================================
echo  HLC - Launching Standard Opera with CDP Debug Port %CDP_PORT%
echo ============================================================
echo.

:: Check if Opera is already running on this port
netstat -an 2>nul | findstr ":%CDP_PORT% " >nul 2>&1
if %errorlevel%==0 (
    echo [OK] Port %CDP_PORT% already active -- Opera may already be running.
    echo      If the bot fails to connect, close Opera and rerun this script.
    echo.
    pause
    exit /b 0
)

:: Check Opera executable exists
if not exist "%OPERA_EXE%" (
    echo [ERROR] Standard Opera not found at:
    echo         %OPERA_EXE%
    echo.
    echo Please install Standard Opera from https://www.opera.com/
    pause
    exit /b 1
)

echo [*] Starting Opera with --remote-debugging-port=%CDP_PORT% ...
echo [*] Opening: %CHESS_URL%
echo.

:: Launch Opera with CDP. Use a dedicated user-data-dir to avoid profile conflicts.
:: --no-first-run and --disable-default-apps suppress the setup wizard.
start "" "%OPERA_EXE%" ^
    --remote-debugging-port=%CDP_PORT% ^
    --remote-allow-origins=* ^
    --no-first-run ^
    --disable-default-apps ^
    --disable-extensions-except= ^
    --user-data-dir="%USERPROFILE%\AppData\Local\HLC_Opera_CDP" ^
    "%CHESS_URL%"

echo [*] Opera is starting...
echo [*] Wait for the browser window to fully open and chess.com to load.
echo.
echo [!] Once chess.com has loaded, open a NEW terminal and run:
echo         start_bot.bat
echo.
echo     OR run the bot manually:
echo         .venv\Scripts\python.exe scripts\chesscom_bot.py --cdp-port %CDP_PORT% --elo 1500
echo.
pause
