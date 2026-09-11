@echo off
title Ollama Cloudflare Tunnel for Streamlit Cloud
echo ====================================================================
echo Starting Secure Cloudflare Tunnel for Local Ollama (Port 11434)...
echo ====================================================================
echo.
echo NOTE: Keep this terminal window open while using Streamlit Cloud!
echo.
set OLLAMA_ORIGINS=*
set OLLAMA_HOST=0.0.0.0:11434
"C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://127.0.0.1:11434 --http-host-header="localhost:11434"
pause
