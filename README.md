<<<<<<< HEAD
# AlphaKnit 🧶 - v6.0 The Watchtower

AlphaKnit is a research-grade AI system that translates 3D point clouds into knitting / amigurumi crochet patterns. This version (v6.0) introduces the **"Watchtower" Research Observatory**, focusing on the physics of topological emergence through deep passive telemetry.
=======
# AlphaKnit 🧶

AlphaKnit is an AI system that translates 3D shapes into knitting / amigurumi crochet patterns. You upload a point cloud or mesh, and the model generates a compilable stitch sequence.
>>>>>>> 6715203057079fa13e1fd3855c710092996e127c

## Architecture

```
3D Point Cloud (N×3)
        │
<<<<<<< HEAD
   PointNetEncoder          ← multi-scale: max-pool + avg-pool + Angular Positional Encoding
        │
   KnittingTransformer       ← Encoder-decoder with Sequential Factorized prediction heads
        │
   Watchtower Observatory    ← Research telemetry (Phase Lag, Latent Portraits, TTF Loss)
        │
   Stitch Tuple Sequence     ← (type, p1_offset, p2_offset)
        │
   KnittingCompiler          ← Validates topology & builds stitch graph
```

## Watchtower Observatory Features

- **Latent Phase Portraits**: Online PCA trajectory visualization of structural embeddings.
- **Phase Lag Monitoring**: Real-time optimizer alignment tracking to detect "Explosions of Choice".
- **Topology Tension Field (TTF)**: Passive bias encouraging structural organization through edge-density penalties.
- **Crystallization Checkpointing**: Automated "Golden Checkpoint" saves at the peak of topological competence.
=======
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
>>>>>>> 6715203057079fa13e1fd3855c710092996e127c

## Project Structure

```
src/alphaknit/
<<<<<<< HEAD
├── model.py            # PointNetEncoder + Factorized KnittingTransformer
├── train.py            # Phase-aware training loop (v6.0 Watchtower integration)
├── research.py         # [NEW] Phase Lag & Latent Phase Portraits
├── metrics.py          # [NEW] Structural Logit Margin & TTF Loss
├── inference.py        # AlphaKnitPredictor — wraps model + compiler
├── compiler.py         # KnittingCompiler — validates stitch sequences
├── simulator.py        # ForwardSimulator — reconstruct mesh from stitch graph
├── tokenizer.py        # Vocabulary & Edge-Action tokenization
├── knitting_dataset.py # WebDataset-optimized loader
├── parser.py           # Stack-based pattern parser
└── config.py           # Shared constants
=======
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
>>>>>>> 6715203057079fa13e1fd3855c710092996e127c
```

## Installation

```bash
<<<<<<< HEAD
pip install -r requirements_pc.txt  # Optimized for local PC (CUDA-ready)
```

## Training

AlphaKnit v6.0 uses a "Self-Aware" launch sequence. The system automatically detects your current epoch, selects the appropriate transition phase (Airlock), and chains checkpoints.

```cmd
.\run_pc.bat
```

### Visualization and Telemetry

To visualize the research data and latent phase portraits after training:

```bash
python scripts/plot_v6_telemetry.py --history checkpoints/training_history_phase9b_dev.json
```

This generates:
- `plots/v6_metrics.png`: Logit Margin, Phase Lag, and Accuracy.
- `plots/phase_portrait.png`: The PCA trajectory showing the "Path to Emergence".

## Phase Strategy Evolution

| Version | Focus | Key Technology |
|---|---|---|
| **v4.0** | Stability | Selective Reset + Shock LR |
| **v5.0** | Automation | State-aware Curriculum (PhaseDetector) |
| **v6.0** | Research | Watchtower Observatory (Passive Telemetry) |
=======
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
>>>>>>> 6715203057079fa13e1fd3855c710092996e127c

## Tests

```bash
python -m pytest tests/
```
