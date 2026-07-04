@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Virtual environment not found. Run setup.bat first.
  pause
  exit /b 1
)
set /p QUESTION=Question to inspect: 
".venv\Scripts\python.exe" ask.py --show-context "%QUESTION%"
pause
