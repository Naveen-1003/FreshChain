@echo off
echo ===== Freshness Detection API Server =====
echo.

set PYTHON="C:/Users/K R ARAVIND/OneDrive/Desktop/freshness/.venv/Scripts/python.exe"

REM Check if option was provided
if "%1"=="check" goto :check_model
if "%1"=="debug" goto :debug_mode

:normal_start
echo Starting server in normal mode...
echo.
echo Web interface: http://localhost:8000
echo API documentation: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.
%PYTHON% -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
goto :end

:debug_mode
echo Starting server in DEBUG mode...
echo.
echo Web interface: http://localhost:8000
echo API documentation: http://localhost:8000/docs
echo.
echo Extra logging enabled
echo.
set TF_CPP_MIN_LOG_LEVEL=0
%PYTHON% -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload --log-level debug
goto :end

:check_model
echo Running model diagnostics...
echo.
%PYTHON% check_model.py
echo.
echo Would you like to start the server now? (Y/N)
set /p choice="> "
if /i "%choice%"=="Y" goto :normal_start
if /i "%choice%"=="y" goto :normal_start
goto :end

:end
echo.
echo Server stopped.
pause