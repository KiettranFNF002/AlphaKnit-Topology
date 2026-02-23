"""
Hyperparameter grid search: trains 3 configs for 5 epochs each,
saves results to checkpoints/hparam_results.json.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
from src.alphaknit.train import train

DATASET_DIR  = "data/debug/dataset_5k"
CHECKPOINT_DIR = "checkpoints"
EPOCHS = 5

CONFIGS = [
    {"name": "A", "d_model": 128, "n_heads": 4, "n_layers": 3, "ffn_dim": 256, "lr": 1e-3},
    {"name": "B", "d_model": 256, "n_heads": 4, "n_layers": 4, "ffn_dim": 512, "lr": 5e-4},
    {"name": "C", "d_model": 128, "n_heads": 4, "n_layers": 4, "ffn_dim": 256, "lr": 5e-4},
]

results = []

for cfg in CONFIGS:
    print(f"\n{'='*50}")
    print(f"Config {cfg['name']}: d_model={cfg['d_model']}, n_layers={cfg['n_layers']}, lr={cfg['lr']}")
    print('='*50)

    history = train(
        dataset_dir=DATASET_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
        epochs=EPOCHS,
        batch_size=32,
        lr=cfg["lr"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        ffn_dim=cfg["ffn_dim"],
        scheduler_type="cosine",
        run_name=f"hparam_{cfg['name']}",
        device_str="cpu",
        val_split=0.1,
    )

    best_val = min(h["val_loss"] for h in history)
    results.append({
        "config": cfg["name"],
        "d_model": cfg["d_model"],
        "n_layers": cfg["n_layers"],
        "lr": cfg["lr"],
        "best_val_loss": round(best_val, 4),
        "history": history,
    })
    print(f"Config {cfg['name']} best val loss: {best_val:.4f}")

# Save results
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
results_path = os.path.join(CHECKPOINT_DIR, "hparam_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

# Print summary
print("\n" + "="*50)
print("HYPERPARAMETER SEARCH RESULTS")
print("="*50)
print(f"{'Config':<8} {'d_model':<10} {'n_layers':<10} {'lr':<10} {'Val Loss':<10}")
print("-"*50)
for r in sorted(results, key=lambda x: x["best_val_loss"]):
    print(f"{r['config']:<8} {r['d_model']:<10} {r['n_layers']:<10} {r['lr']:<10} {r['best_val_loss']:<10.4f}")

best = min(results, key=lambda x: x["best_val_loss"])
print(f"\nBest config: {best['config']} (val_loss={best['best_val_loss']:.4f})")
print(f"Results saved to: {results_path}")
