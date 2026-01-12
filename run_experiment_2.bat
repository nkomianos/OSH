@echo off
REM OSH Experiment 2: The Epsilon Ablation
REM This script runs the Epsilon Ablation experiment

echo ============================================================
echo OSH EXPERIMENT 2: THE EPSILON ABLATION
echo ============================================================
echo.
echo This experiment proves that FP32 precision is necessary
echo to prevent quantization error accumulation.
echo.
echo Prerequisites:
echo   - Experiment 1 must be complete (LoRA antidote trained)
echo   - ./osh_antidote_v1/ directory must exist
echo.
echo Estimated time: 30-60 minutes
echo.
set /p continue="Continue? (y/n) "
if /i not "%continue%"=="y" (
    echo Aborted.
    exit /b 1
)

REM Check if antidote exists
if not exist "osh_antidote_v1" (
    echo.
    echo ============================================================
    echo ERROR: LoRA Antidote Not Found
    echo ============================================================
    echo.
    echo The file ./osh_antidote_v1/ does not exist.
    echo You must run Experiment 1 first to train the antidote.
    echo.
    echo Run: python osh_exp_1.py
    echo      (or run_experiment_1.bat for full workflow)
    echo.
    exit /b 1
)

echo.
echo ============================================================
echo RUNNING EPSILON ABLATION
echo ============================================================
echo Starting at: %date% %time%
echo.

python epsilon_ablation.py

if errorlevel 1 (
    echo.
    echo ============================================================
    echo EXPERIMENT FAILED
    echo ============================================================
    echo Check errors above.
    exit /b 1
)

echo.
echo ============================================================
echo EXPERIMENT 2 COMPLETE!
echo ============================================================
echo.
echo Generated files:
echo   epsilon_ablation.png    - Precision comparison plot
echo   epsilon_ablation.pdf    - High-res version (for paper)
echo.
echo Review the plot to verify:
echo   - Red line (BF16) drifts upward
echo   - Green line (FP32) stays near zero
echo   - This proves the TEE protocol is necessary
echo.
echo Next: Experiment 3 (Blind Surgeon Attack)
echo   Run: python blind_surgeon.py
echo.
echo ============================================================
pause
