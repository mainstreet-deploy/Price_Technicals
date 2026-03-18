@echo off
setlocal

:: ============================================================
:: run_report.bat
:: Price Technicals Routine — Step 2: Report Generation
::
:: Calls generate_technicals_report.py to compute moving
:: averages, return periods, and cross signals, then renders
:: PDF reports to output_.
::
:: Log written to: logs\report_log.txt
:: ============================================================

cd /d "%~dp0"

if not exist logs     mkdir logs
if not exist output_  mkdir output_

echo [%date% %time%] ============================================ >> logs\report_log.txt
echo [%date% %time%] run_report.bat  START                       >> logs\report_log.txt
echo [%date% %time%] ============================================ >> logs\report_log.txt

python generate_technicals_report.py >> logs\report_log.txt 2>&1

if %errorlevel% neq 0 (
    echo [%date% %time%] ERROR: generate_technicals_report.py exited with code %errorlevel% >> logs\report_log.txt
    exit /b 1
)

echo [%date% %time%] run_report.bat  COMPLETE >> logs\report_log.txt
exit /b 0
