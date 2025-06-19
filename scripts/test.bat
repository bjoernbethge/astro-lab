@echo off
REM Test runner for astro-lab that suppresses fake-bpy-module memory leak warnings

echo Running astro-lab tests with clean output...

REM Set environment variables for deterministic tests
set PYTHONHASHSEED=0
set MALLOC_CHECK_=0

REM Run pytest and filter output
uv run pytest %* 2>&1 | findstr /v "Error: Not freed memory blocks" | findstr /v "total unfreed memory"

REM Preserve exit code
exit /b %ERRORLEVEL% 