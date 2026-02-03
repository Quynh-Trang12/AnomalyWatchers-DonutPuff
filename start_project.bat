@echo off
echo Starting Fraud Simulator...

:: Start Backend
start cmd /k "echo Starting Backend... & uvicorn backend.app.main:app --reload --port 8000"

:: Start Frontend
start cmd /k "echo Starting Frontend... & npm run dev"

echo Services passed to background windows.
pause
