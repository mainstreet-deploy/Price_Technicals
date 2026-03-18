@echo off
setlocal

:: ============================================================
:: run_fetch.bat
:: Price Technicals Routine — Step 1: Data Fetch
::
:: Calls fetch_price_technicals.py to pull historical EOD
:: prices from FMP for every ticker in watchlist.txt and
:: save raw JSON to cache_.
::
:: Log written to: logs\fetch_log.txt
:: ============================================================

cd /d "%~dp0"

if not exist logs mkdir logs
if not exist cache_ mkdir cache_

echo [%date% %time%] ============================================ >> logs\fetch_log.txt
echo [%date% %time%] run_fetch.bat  START                        >> logs\fetch_log.txt
echo [%date% %time%] ============================================ >> logs\fetch_log.txt

python fetch_price_technicals.py >> logs\fetch_log.txt 2>&1

if %errorlevel% neq 0 (
    echo [%date% %time%] ERROR: fetch_price_technicals.py exited with code %errorlevel% >> logs\fetch_log.txt
    exit /b 1
)

echo [%date% %time%] run_fetch.bat  COMPLETE >> logs\fetch_log.txt
exit /b 0
