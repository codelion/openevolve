@echo off
REM Script to run OpenEvolve on Huawei 2017 CDN optimization problem (Windows)

setlocal enabledelayedexpansion

echo Starting OpenEvolve evolution for Huawei CodeCraft 2017 CDN Optimization
echo ========================================================================
echo.

REM Check if running from correct directory
if not exist "initial_program.py" (
    echo Error: Must run from examples\huawei_2017 directory
    exit /b 1
)

REM Check if case examples exist
if not exist "case_example" (
    echo Error: case_example directory not found
    exit /b 1
)

REM Default values
set ITERATIONS=300
set CONFIG=config.qwen.yaml
set CHECKPOINT=

REM Parse arguments
:parse_args
if "%~1"=="" goto run_evolution
if "%~1"=="--iterations" (
    set ITERATIONS=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--config" (
    set CONFIG=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--checkpoint" (
    set CHECKPOINT=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--help" (
    echo Usage: %0 [OPTIONS]
    echo.
    echo Options:
    echo   --iterations N      Number of evolution iterations (default: 300^)
    echo   --config FILE       Config file to use (default: config.qwen.yaml^)
    echo   --checkpoint DIR    Resume from checkpoint directory
    echo   --help              Show this help message
    exit /b 0
)
echo Unknown option: %~1
echo Use --help for usage information
exit /b 1

:run_evolution
REM Build command
set CMD=python ..\..\openevolve-run.py initial_program.py evaluator.py --config %CONFIG% --iterations %ITERATIONS%

if not "%CHECKPOINT%"=="" (
    set CMD=%CMD% --checkpoint %CHECKPOINT%
)

echo Configuration:
echo   Iterations: %ITERATIONS%
echo   Config: %CONFIG%
if not "%CHECKPOINT%"=="" (
    echo   Checkpoint: %CHECKPOINT%
)
echo.
echo Running: %CMD%
echo.

REM Run evolution
%CMD%

echo.
echo Evolution complete! Check openevolve_output\ for results.
echo Best solution: openevolve_output\best\best_program.py
