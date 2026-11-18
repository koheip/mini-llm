@echo off
chcp 65001 > nul
"%~dp0.venv\Scripts\python.exe" "%~dp0web_ui.py"
pause
