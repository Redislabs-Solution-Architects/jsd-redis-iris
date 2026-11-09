@echo off
setlocal

cd /d "%~dp0"

"%~dp0venv\Scripts\python.exe" -m uvicorn web_app:app --reload

endlocal
