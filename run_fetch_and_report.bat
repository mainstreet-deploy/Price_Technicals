@echo off
:: Price Technicals — Fetch + Report (combined launcher)

cd /d "%~dp0"
if not exist logs mkdir logs

echo.
echo ====================================
echo  Step 1: Fetch Price Data
echo ====================================
echo.
python fetch_price_technicals.py
if errorlevel 1 (
    echo.
    echo [ERROR] Fetch step failed. See logs\fetch_log.txt for details.
    echo ====================================
    exit /b 1
)

echo.
echo ====================================
echo  Step 2: Generate Technicals Report
echo ====================================
echo.
python generate_technicals_report.py
if errorlevel 1 (
    echo.
    echo [ERROR] Report step failed. See logs\report_log.txt for details.
    echo ====================================
    exit /b 1
)

echo.
echo ====================================
echo  All done!
echo ====================================
