@echo off
REM OSH Experiment 1: The Lazarus Plot
REM This script runs the complete Experiment 1 workflow

echo ============================================================
echo OSH EXPERIMENT 1: THE LAZARUS PLOT
echo ============================================================
echo.
echo This will:
echo   1. Validate your setup
echo   2. Train the LoRA antidote (~2-4 hours)
echo   3. Run the Lazarus plot sweep (~1-2 hours)
echo.
echo Total estimated time: 3-6 hours (GPU dependent)
echo.
set /p continue="Continue? (y/n) "
if /i not "%continue%"=="y" (
    echo Aborted.
    exit /b 1
)

REM Step 1: Validate setup
echo.
echo ============================================================
echo STEP 1: VALIDATING SETUP
echo ============================================================
python validate_setup.py

if errorlevel 1 (
    echo.
    echo Failed validation. Please fix errors above.
    exit /b 1
)

set /p continue="Validation complete. Proceed to training? (y/n) "
if /i not "%continue%"=="y" (
    echo Aborted.
    exit /b 1
)

REM Step 2: Train antidote
echo.
echo ============================================================
echo STEP 2: TRAINING ANTIDOTE (This will take 2-4 hours^)
echo ============================================================
echo Starting at: %date% %time%
echo.

python osh_exp_1.py

if errorlevel 1 (
    echo.
    echo Training failed. Check errors above.
    exit /b 1
)

echo.
echo Training complete at: %date% %time%
echo Antidote saved to: ./osh_antidote_v1/
echo Convergence plot: antidote_convergence.png

REM Check if antidote exists
if not exist "osh_antidote_v1" (
    echo ERROR: Antidote directory not found
    exit /b 1
)

set /p continue="Training complete. Proceed to Lazarus plot? (y/n) "
if /i not "%continue%"=="y" (
    echo Aborted. You can run the Lazarus plot later with:
    echo   python lazarus_plot.py
    exit /b 0
)

REM Step 3: Run Lazarus plot
echo.
echo ============================================================
echo STEP 3: LAZARUS PLOT (This will take 1-2 hours^)
echo ============================================================
echo Starting at: %date% %time%
echo.

python lazarus_plot.py

if errorlevel 1 (
    echo.
    echo Lazarus plot failed. Check errors above.
    exit /b 1
)

echo.
echo Lazarus plot complete at: %date% %time%
echo.
echo ============================================================
echo EXPERIMENT 1 COMPLETE!
echo ============================================================
echo.
echo Generated files:
echo   ./osh_antidote_v1/          - Trained LoRA weights
echo   antidote_convergence.png    - Training curve
echo   lazarus_plot.png            - Main result (for paper)
echo   lazarus_plot.pdf            - High-res version
echo.
echo Next steps:
echo   1. Review lazarus_plot.png
echo   2. Check if locked line explodes (^>5x baseline)
echo   3. Check if unlocked line stays flat (^<1.5x baseline)
echo.
echo If results are good, proceed to Experiments 2-5
echo See: experimentation_plan
echo.
echo ============================================================
pause
