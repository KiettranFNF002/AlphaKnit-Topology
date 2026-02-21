@echo off
setlocal enabledelayedexpansion
echo ==============================================
echo  AlphaKnit-Topology v5.0 NASA-Style Launcher
echo ==============================================

:: Environment Setup
if not exist venv (
    echo [SETUP] Creating Virtual Environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    python -m pip install tqdm networkx matplotlib webdataset
) else (
    call venv\Scripts\activate.bat
)

set PYTHONPATH=src
set CUDA_LAUNCH_BLOCKING=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

:: v5.0 NASA-Style Launch Sequence
:: Automatically detects the current state and picks the right phase.

echo Detecting latest training state...
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
    echo [LAUNCH] No previous state found. Starting PHASE 1.
    goto PHASE_1
)
if %LAST_EPOCH% LSS 11 (
    echo [RESUME] Phase 1 Grammar Warmup incomplete. Resuming Phase 1.
    goto PHASE_1
)
if %LAST_EPOCH% LSS 22 (
    echo [TRANSITION] Grammar master found. Entering/Resuming PHASE 2 AIRLOCK.
    goto PHASE_2
)
echo [MASTERY] Physics ignition confirmed. Resuming PHASE 3 TOPOLOGY.
goto PHASE_3


:PHASE_1
echo --- PHASE 1: Grammar Warmup (Epoch 1-11) ---
echo BS 64, Acc 2 (Eff BS 128) | Goal: Entropy Stability
python -m alphaknit.train --batch_size 64 --grad_accum_steps 2 --num_workers 4 --epochs 11 --lr 3e-4 --resume_auto --dataset_dir "data/processed/dataset/shard-{0000..0049}.tar"
if %ERRORLEVEL% NEQ 0 goto ERROR
goto RESTART_SEQUENCE

:PHASE_2
echo.
echo --- PHASE 2: Airlock Transition (Epoch 12-22) ---
echo BS 36, Acc 1 | Selective Reset | Goal: Physics Emergence
python -m alphaknit.train --batch_size 36 --grad_accum_steps 1 --num_workers 4 --epochs 22 --lr 1.5e-4 --reset_optimizer --resume_auto --dataset_dir "data/processed/dataset/shard-{0000..0049}.tar"
if %ERRORLEVEL% NEQ 0 goto ERROR
goto RESTART_SEQUENCE

:PHASE_3
echo.
echo --- PHASE 3: Topology Mastery (Epoch 23-100) ---
echo BS 30, Acc 1 | Goal: Structural Complexity
python -m alphaknit.train --batch_size 30 --grad_accum_steps 1 --num_workers 4 --epochs 100 --lr 2e-4 --resume_auto --dataset_dir "data/processed/dataset/shard-{0000..0049}.tar"
if %ERRORLEVEL% NEQ 0 goto ERROR
goto END

:RESTART_SEQUENCE
echo.
echo Phase finished. Restarting sequence to detect next phase...
timeout /t 5
:: For Windows batch, we use call or just rerun the script
%0
goto :EOF

:ERROR
echo.
echo [FATAL] Training process interrupted or crashed.
pause
exit /b 1

:END
echo.
echo ==============================================
echo  ALPHA-KNIT TRAINING COMPLETE! 🚀
echo ==============================================
pause
