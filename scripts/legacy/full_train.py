"""
LEGACY - DO NOT USE
Full training run: trains the best config for 20 epochs with cosine LR.
Edit BEST_CONFIG below to match the winner from hparam_search.py.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
from src.alphaknit.train import train

DATASET_DIR    = "data/processed/dataset_5k"
CHECKPOINT_DIR = "checkpoints"

# ── Edit this to match hparam_search.py winner ──────────────────────
BEST_CONFIG = {
    "d_model":  128,
    "n_heads":  4,
    "n_layers": 3,
    "ffn_dim":  256,
    "lr":       1e-3,
}
# ────────────────────────────────────────────────────────────────────

print("="*60)
print("FULL TRAINING RUN — 20 epochs")
print(f"Config: {BEST_CONFIG}")
print("="*60)

history = train(
    dataset_dir=DATASET_DIR,
    checkpoint_dir=CHECKPOINT_DIR,
    epochs=20,
    batch_size=32,
    lr=BEST_CONFIG["lr"],
    d_model=BEST_CONFIG["d_model"],
    n_heads=BEST_CONFIG["n_heads"],
    n_layers=BEST_CONFIG["n_layers"],
    ffn_dim=BEST_CONFIG["ffn_dim"],
    scheduler_type="cosine",
    run_name="full",
    device_str="cpu",
    val_split=0.1,
)

# Print final table
print("\n" + "="*60)
print(f"{'Epoch':<8} {'Train Loss':<14} {'Val Loss':<14}")
print("-"*36)
for h in history:
    print(f"{h['epoch']:<8} {h['train_loss']:<14.4f} {h['val_loss']:<14.4f}")

best_val = min(h["val_loss"] for h in history)
best_ep  = min(history, key=lambda h: h["val_loss"])["epoch"]
print(f"\nBest val loss: {best_val:.4f} at epoch {best_ep}")
print("Checkpoint: checkpoints/best_model_full.pt")
