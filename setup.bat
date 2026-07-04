@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 -m venv .venv
) else (
    python -m venv .venv
)
if errorlevel 1 goto :error

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 goto :error

if not exist .env copy .env.example .env >nul
if not exist documents mkdir documents

echo.
echo Setup complete.
echo 1. Edit .env and set CHAT_MODEL and EMBED_MODEL.
echo 2. Put files inside the documents folder.
echo 3. Run check_lmstudio.bat, then run_ingest.bat, then run_chat.bat.
pause
exit /b 0

:error
echo.
echo Setup failed. Install a 64-bit Python 3.11 or 3.12 and ensure it is on PATH.
pause
exit /b 1
