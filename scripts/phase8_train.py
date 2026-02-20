"""
Phase 8 Training Script — Resume + 50 epochs

Changes vs full_train.py:
  - Resumes from best_model_full.pt (keeps prior knowledge)
  - 50 epochs total (was 20)
  - Label smoothing 0.1 (handled in train.py)
  - Logs compile_success_rate every 5 epochs
  - Early stopping patience=10
  - Saves as best_model_phase8.pt
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
from src.alphaknit.train import train

DATASET_DIR    = "data/processed/dataset_5k"
CHECKPOINT_DIR = "checkpoints"
RESUME_FROM    = "checkpoints/best_model_full.pt"

CONFIG = {
    "d_model":  128,
    "n_heads":  4,
    "n_layers": 3,
    "ffn_dim":  256,
    "lr":       1e-3,   # full LR since training from scratch
}

# NOTE: Resuming from the best model saved by this script.
RESUME_FROM = "checkpoints/best_model_phase8.pt"

print("=" * 65)
print("PHASE 8 TRAINING — 50 epochs, label_smoothing=0.1")
print(f"Config: {CONFIG}")
print(f"Resume: {RESUME_FROM}")
print("=" * 65)

history = train(
    dataset_dir=DATASET_DIR,
    checkpoint_dir=CHECKPOINT_DIR,
    epochs=50,
    batch_size=32,
    lr=CONFIG["lr"],
    d_model=CONFIG["d_model"],
    n_heads=CONFIG["n_heads"],
    n_layers=CONFIG["n_layers"],
    ffn_dim=CONFIG["ffn_dim"],
    scheduler_type="cosine",
    run_name="phase8",
    device_str="auto",
    val_split=0.1,
    label_smoothing=0.1,
    early_stop_patience=10,
    log_compile_every=5,
    resume_checkpoint=RESUME_FROM,
)

# Print final summary table
print("\n" + "=" * 75)
print(f"{'Epoch':<8} {'Train':<12} {'Val':<12} {'Compile%':<12}")
print("-" * 44)
for h in history:
    cr = h.get("compile_success_rate")
    cr_str = f"{cr*100:.1f}%" if cr is not None else ""
    print(f"{h['epoch']:<8} {h['train_loss']:<12.4f} {h['val_loss']:<12.4f} {cr_str:<12}")

best_val = min(h["val_loss"] for h in history)
best_ep  = min(history, key=lambda h: h["val_loss"])["epoch"]

# Find latest compile rate
compile_rates = [(h["epoch"], h["compile_success_rate"]) for h in history if "compile_success_rate" in h]
if compile_rates:
    last_cr = compile_rates[-1]
    print(f"\nCompile success (epoch {last_cr[0]}): {last_cr[1]*100:.1f}%")

print(f"Best val loss: {best_val:.4f} at epoch {best_ep}")
print("Checkpoint: checkpoints/best_model_phase8.pt")

# Print top confusions from last logged epoch
for h in reversed(history):
    if "top_confusions" in h and h["top_confusions"]:
        print(f"\nTop confusions (epoch {h['epoch']}):")
        for c in h["top_confusions"]:
            print(f"  pred={c['pred']:<6} true={c['true']:<6} count={c['count']}")
        break
