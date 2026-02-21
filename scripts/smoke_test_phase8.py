"""Quick 2-epoch smoke test for Phase 8 training."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.alphaknit.train import train

print('Running quick 2-epoch smoke test...')
history = train(
    dataset_dir='data/processed/dataset_5k',
    checkpoint_dir='checkpoints',
    epochs=2,
    batch_size=32,
    lr=1e-3,
    d_model=128,
    n_heads=4,
    n_layers=3,
    ffn_dim=256,
    scheduler_type='cosine',
    run_name='phase8_test',
    device_str='cpu',
    val_split=0.1,
    label_smoothing=0.1,
    early_stop_patience=10,
    log_compile_every=1,
    resume_checkpoint=None,
)

print()
print('=== SMOKE TEST PASSED ===')
for h in history:
    cr = h.get('compile_success_rate', 'N/A')
    print(f"  Epoch {h['epoch']}: train={h['train_loss']:.4f}  val={h['val_loss']:.4f}  compile={cr}")
    if 'top_confusions' in h:
        for c in h['top_confusions']:
            print(f"    confusion: pred={c['pred']} true={c['true']} count={c['count']}")
