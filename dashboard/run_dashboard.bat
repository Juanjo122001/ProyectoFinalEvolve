@echo off
REM Lanza el dashboard VOLT usando el entorno virtual del proyecto
cd /d "%~dp0\.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
python -c "import dash" 2>nul || pip install -r dashboard\requirements.txt
start "" http://127.0.0.1:8050
python dashboard\app.py
