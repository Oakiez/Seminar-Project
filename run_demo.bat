@echo off
rem Start the Dynamic GIS Recommender demo and open it in the default browser.
cd /d "%~dp0"
".venv\Scripts\streamlit.exe" run app.py --server.port 8501 --browser.gatherUsageStats false
