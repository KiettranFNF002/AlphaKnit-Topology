@echo off
setlocal
echo ==============================================
echo  AlphaKnit-Topology Local PC Training Launcher
echo ==============================================

if not exist venv (
    echo [1/4] Creating virtual environment (venv)...
    python -m venv venv
) else (
    echo [1/4] Virtual environment already exists.
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Installing/Updating required dependencies...
pip install -r requirements_pc.txt

echo.
echo [4/4] Checking dataset...
if not exist "data\processed\dataset\*.tar" (
    echo Dataset shards not found in data\processed\dataset\
    echo Starting local dataset generation ^(50,000 samples^)...
    echo This might take a few minutes depending on CPU speed.
    python scripts\gen_shards_direct.py --output_dir data\processed\dataset --n_samples 50000 --shard_size 1000
) else (
    echo Dataset shards found. Skipping generation.
)

echo.
echo ==============================================
echo  Starting AlphaKnit Training! 🚀
echo  (Press Ctrl+C to stop training manually)
echo ==============================================
echo.

:: You can increase batch_size to 256 or 512 if the PC has a high-end GPU (12GB+ VRAM)
python src/alphaknit/train.py --batch_size 128 --grad_accum_steps 4 --num_workers 4

echo.
echo Training process finished or interrupted.
pause
