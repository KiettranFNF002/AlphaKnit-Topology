@echo off
setlocal
echo ==============================================
echo  AlphaKnit-Topology Local PC Training Launcher
echo ==============================================

if not exist venv (
    echo [1/3] Creating virtual environment (venv)...
    python -m venv venv
) else (
    echo [1/3] Virtual environment already exists.
)

echo [2/3] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/3] Installing/Updating required dependencies...
pip install -r requirements_pc.txt

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
