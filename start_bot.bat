@echo off
:: ============================================================
:: start_bot.bat
:: Waits for Opera CDP to be ready, then starts the HLC bot.
::
:: Run start_opera_cdp.bat FIRST, then run this script.
:: ============================================================

set CDP_PORT=9222
set PYTHON=.venv\Scripts\python.exe
set SCRIPT=scripts\chesscom_bot.py

:: ─── Parse arguments (forward all extra args to the bot) ───────────────
set EXTRA_ARGS=%*

echo.
echo ============================================================
echo  HLC Bot Launcher
echo  Waiting for Opera CDP on port %CDP_PORT%...
echo ============================================================
echo.

:: Poll until port 9222 is active (Opera fully loaded)
set /a ATTEMPTS=0
:WAIT_LOOP
netstat -an 2>nul | findstr ":%CDP_PORT% " >nul 2>&1
if %errorlevel%==0 goto PORT_READY

set /a ATTEMPTS+=1
if %ATTEMPTS% GEQ 30 (
    echo [ERROR] Timed out waiting for Opera on port %CDP_PORT% after 30 seconds.
    echo         Make sure you ran start_opera_cdp.bat first!
    pause
    exit /b 1
)

echo [.] Waiting for port %CDP_PORT%... (attempt %ATTEMPTS%/30)
timeout /t 1 /nobreak >nul
goto WAIT_LOOP

:PORT_READY
echo [OK] Opera CDP port %CDP_PORT% is active!
echo.

:: Small extra delay to ensure CDP HTTP endpoint is ready
timeout /t 2 /nobreak >nul

:: Verify the CDP endpoint responds
for /f %%i in ('powershell -NoProfile -Command "try { $r=(Invoke-WebRequest -Uri http://localhost:%CDP_PORT%/json/version -TimeoutSec 3 -UseBasicParsing).StatusCode; Write-Output $r } catch { Write-Output 0 }"') do set CDP_STATUS=%%i

if "%CDP_STATUS%"=="200" (
    echo [OK] CDP endpoint verified (HTTP 200).
) else (
    echo [WARN] CDP endpoint check returned: %CDP_STATUS%
    echo        Proceeding anyway -- bot will retry connection.
)

echo.
echo [*] Starting HLC bot...
echo     Script : %SCRIPT%
echo     Port   : %CDP_PORT%
echo.

%PYTHON% %SCRIPT% --cdp-port %CDP_PORT% %EXTRA_ARGS%

echo.
echo [HLC] Bot session ended.
pause
