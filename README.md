# AlphaKnit 🧶 - v6.0 The Watchtower

AlphaKnit is a research-grade AI system that translates 3D point clouds into knitting / amigurumi crochet patterns. This version (v6.0) introduces the **"Watchtower" Research Observatory**, focusing on the physics of topological emergence through deep passive telemetry.

## Architecture

```
3D Point Cloud (N×3)
        │
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

## Project Structure

```
src/alphaknit/
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
```

## Installation

```bash
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

## Tests

```bash
python -m pytest tests/
```
