"""
ASCII training curve plotter.
Reads checkpoints/training_history_<run_name>.json and prints a text chart.

Usage:
    python scripts/plot_curves.py full
    python scripts/plot_curves.py hparam_A
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

run_name = sys.argv[1] if len(sys.argv) > 1 else "full"
history_path = os.path.join("checkpoints", f"training_history_{run_name}.json")

if not os.path.exists(history_path):
    print(f"No history file found: {history_path}")
    sys.exit(1)

with open(history_path) as f:
    history = json.load(f)

epochs     = [h["epoch"] for h in history]
train_loss = [h["train_loss"] for h in history]
val_loss   = [h["val_loss"] for h in history]

# ── ASCII chart ──────────────────────────────────────────────────────
WIDTH  = 60
HEIGHT = 20

all_vals = train_loss + val_loss
y_min = min(all_vals) * 0.95
y_max = max(all_vals) * 1.05
y_range = y_max - y_min or 1.0

def to_row(val):
    return int((1 - (val - y_min) / y_range) * (HEIGHT - 1))

# Build grid
grid = [[" "] * WIDTH for _ in range(HEIGHT)]

for i, (tl, vl) in enumerate(zip(train_loss, val_loss)):
    x = int(i / max(len(epochs) - 1, 1) * (WIDTH - 1))
    grid[to_row(tl)][x] = "T"
    grid[to_row(vl)][x] = "V"

print(f"\n=== Training Curves: {run_name} ===")
print(f"  T=train_loss  V=val_loss")
print(f"  y: [{y_min:.4f} .. {y_max:.4f}]")
print()

for r, row in enumerate(grid):
    y_label = y_max - (r / (HEIGHT - 1)) * y_range
    print(f"  {y_label:6.3f} |{''.join(row)}|")

print("         " + "-" * WIDTH)
print(f"         Epoch 1{' ' * (WIDTH - 14)}Epoch {len(epochs)}")

# ── Summary table ────────────────────────────────────────────────────
print(f"\n{'Epoch':<8} {'Train':<12} {'Val':<12}")
print("-"*32)
for h in history:
    marker = " ←best" if h["val_loss"] == min(val_loss) else ""
    print(f"{h['epoch']:<8} {h['train_loss']:<12.4f} {h['val_loss']:<12.4f}{marker}")

print(f"\nBest val loss: {min(val_loss):.4f} at epoch {val_loss.index(min(val_loss)) + 1}")
