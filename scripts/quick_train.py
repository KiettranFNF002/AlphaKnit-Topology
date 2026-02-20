import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.alphaknit.train import train

history = train(
    dataset_dir='data/processed/dataset',
    checkpoint_dir='checkpoints',
    epochs=5,
    batch_size=32,
    lr=1e-3,
    device_str='cpu',
    val_split=0.1,
)

print('\nFinal losses:')
for h in history:
    print('  Epoch %d: train=%.4f val=%.4f' % (h['epoch'], h['train_loss'], h['val_loss']))

first = history[0]['train_loss']
last  = history[-1]['train_loss']
print('\nLoss decreased:', last < first)
