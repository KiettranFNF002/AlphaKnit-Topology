@echo off
setlocal enabledelayedexpansion
echo ==================================================
echo  AlphaKnit-Topology v6.6-F "NASA-Style" Launcher
echo ==================================================

:: 1. Environment Check/Setup
if not exist venv (
    echo [SETUP] Creating Virtual Environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    python -m pip install -r requirements_pc.txt
) else (
    call venv\Scripts\activate.bat
)

set PYTHONPATH=src
set CUDA_LAUNCH_BLOCKING=0

:: 2. Dataset Check/Generation
echo.
echo [DATA] Checking dataset...
if not exist "data\processed\dataset\*.tar" (
    echo Dataset shards not found. Starting local generation...
    python scripts\gen_shards_direct.py --output_dir data\processed\dataset --n_samples 50000 --shard_size 1000
) else (
    echo Dataset shards found. Skipping generation.
)

:: 3. Automated Phase Detection
echo.
echo [CONFIG] Detecting latest training state...
set LAST_EPOCH=-1
if not exist scripts\get_last_epoch.py (
    echo [ERROR] scripts\get_last_epoch.py missing!
    pause
    exit /b 1
)

for /f %%i in ('python scripts/get_last_epoch.py') do set LAST_EPOCH=%%i

echo.
echo ==============================================
echo  LATEST STATE: Epoch %LAST_EPOCH%
echo ==============================================

if %LAST_EPOCH% LSS 0 (
    echo [LAUNCH] No previous state. Starting PHASE 1: Grammar Warmup.
    goto PHASE_1
)
if %LAST_EPOCH% LSS 11 (
    echo [RESUME] Phase 1 incomplete. Resuming Grammar Warmup.
    goto PHASE_1
)
if %LAST_EPOCH% LSS 22 (
    echo [TRANSITION] Grammar master found. Entering/Resuming PHASE 2: Airlock.
    goto PHASE_2
)
echo [MASTERY] Physics ignition confirmed. Resuming PHASE 3: Topology Discovery.
goto PHASE_3


:PHASE_1
echo --- PHASE 1: Grammar Warmup (Epoch 1-11) ---
echo Target: Entropy Stability | BS 128 (64x2)
python src/alphaknit/train.py --batch_size 64 --grad_accum_steps 2 --num_workers 4 --epochs 11 --lr 3e-4 --resume_auto --run_name "v6.6F_Grammar"
if %ERRORLEVEL% NEQ 0 goto ERROR
goto RESTART_SEQUENCE

:PHASE_2
echo.
echo --- PHASE 2: Airlock Transition (Epoch 12-22) ---
echo Target: Physics Emergence | Selective Reset | SHOCK LR
python src/alphaknit/train.py --batch_size 48 --grad_accum_steps 2 --num_workers 4 --epochs 22 --lr 1.5e-4 --reset_optimizer --resume_auto --run_name "v6.6F_Emergence"
if %ERRORLEVEL% NEQ 0 goto ERROR
goto RESTART_SEQUENCE

:PHASE_3
echo.
echo --- PHASE 3: Topology Mastery (Epoch 23-100) ---
echo Target: Structural Falsification | BS 32 (Acc 4)
python src/alphaknit/train.py --batch_size 32 --grad_accum_steps 4 --num_workers 4 --epochs 100 --lr 1e-4 --resume_auto --run_name "v6.6F_Mastery"
if %ERRORLEVEL% NEQ 0 goto ERROR
goto END


:RESTART_SEQUENCE
echo.
echo Phase completed. Re-detecting next sequence...
timeout /t 5
%0
goto :EOF

:ERROR
echo.
echo [FATAL] Training process crashed or interrupted.
pause
exit /b 1

echo.
echo ==============================================
echo  ALPHA-KNIT v6.6-F DISCOVERY COMPLETE! 🚀
echo  Status: Phase 11 Verified
echo ==============================================
pause
