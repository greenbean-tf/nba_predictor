@echo off
set PROJECT=C:\Users\User\Desktop\code\nba_predictor
set LOG=%PROJECT%\logs\restart.log
mkdir %PROJECT%\logs 2>nul
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0

:loop
echo [%date% %time%] Starting... >> %LOG%
cd /d %PROJECT%
python -u src/live/live_trader.py --mode both --capital 100 --kelly 0.8 --min-ev 0.05 --min-edge 0.05 --pregame-std 14.6 --halftime-std 10.7 --live --yes

:: exit code 0 = normal exit (all games done), restart only on crash
if %errorlevel% == 0 (
    echo [%date% %time%] All games done. Exiting. >> %LOG%
    goto end
)

echo [%date% %time%] Crashed (code=%errorlevel%). Restarting in 30s... >> %LOG%
timeout /t 30
goto loop

:end
echo [%date% %time%] Simulator finished for today. >> %LOG%
