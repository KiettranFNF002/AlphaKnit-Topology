# AlphaKnit 🧶

AlphaKnit is an AI system that translates 3D shapes into knitting / amigurumi crochet patterns. You upload a point cloud or mesh, and the model generates a compilable stitch sequence.

## Architecture

```
3D Point Cloud (N×3)
        │
   PointNetEncoder          ← multi-scale: max-pool + avg-pool concat, BatchNorm
        │
  KnittingTransformer       ← Transformer decoder (d_model=128, 3 layers, 4 heads)
        │
  Token Sequence            ← e.g. "mr_6 sc sc inc sc inc sc eos"
        │
  KnittingCompiler          ← validates topology, builds stitch graph
        │
  ForwardSimulator          ← reconstructs 3D surface for comparison
```

**Phase 8 results (50-epoch training):**
| Metric               | Before | After   |
|----------------------|--------|---------|
| Val Loss             | 0.788  |**0.501**|
| Compile Success Rate | 33.6%  |**92.2%**|

## Project Structure

```
src/alphaknit/
├── model.py          # PointNetEncoder + KnittingTransformer (greedy & beam decode)
├── train.py          # Training loop (label smoothing, early stopping, compile logging)
├── inference.py      # AlphaKnitPredictor — wraps model + compiler
├── compiler.py       # KnittingCompiler — validates stitch sequences
├── simulator.py      # ForwardSimulator — reconstruct mesh from stitch graph
├── tokenizer.py      # Vocabulary & token ↔ ID conversion
├── knitting_dataset.py  # PyTorch Dataset
├── parser.py         # Stack-based pattern parser
└── config.py         # Shared constants (vocab, seq lengths, etc.)

scripts/
├── phase8_train.py   # 50-epoch Phase 8 training script
├── eval_phase8.py    # Evaluate greedy vs beam search on test samples
├── generate_data.py  # Synthetic data generation
└── evaluate.py       # Full evaluation pipeline

checkpoints/
└── best_model_phase8.pt  # Best model (Epoch 49, val_loss=0.499)

app.py                # Streamlit web demo
```

## Installation

```bash
pip install -r requirements.txt
```

## Running the Demo

```bash
streamlit run app.py
```

Upload a `.npy` (point cloud) or `.obj` / `.ply` (mesh) file. Adjust the **Beam Width** slider in the sidebar to switch between greedy decoding (fast) and beam search (higher quality).

## Training

```bash
# Fresh Phase 8 training (from scratch — new architecture required)
python scripts/phase8_train.py

# Evaluate after training
python scripts/eval_phase8.py --samples 200
```

## Key Phase 8 Improvements

- **Multi-scale PointNet encoder** — concatenates max-pool + avg-pool for richer geometry representation
- **Label smoothing (α=0.1)** — prevents the model from fixating on `sc` (most frequent token)
- **Compile-guided beam search** — prunes beams that violate basic stitch topology rules
- **Per-epoch compile success rate** — the real metric: % of decoded sequences that pass the compiler
- **Early stopping** (patience = 10) + full checkpoint resume support

## Tests

```bash
python -m pytest tests/
```
