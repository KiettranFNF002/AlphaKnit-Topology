import os
import torch
import numpy as np

from alphaknit import config
from alphaknit.model import KnittingTransformer
from alphaknit.dataset_builder import DatasetBuilder
from alphaknit.compiler import KnittingCompiler

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # 1. Initialize Model
    print("Loading model...")
    model = KnittingTransformer(
        d_model=128,      # MATCHES COLAB TRAIN_CONFIG
        n_heads=4,
        n_layers=3,
        ffn_dim=256,
    ).to(device)

    # 2. Load Checkpoint
    ckpt_path = "checkpoints/best_model_colab_v6.6F.pt"
    if not os.path.exists(ckpt_path):
        print(f"❌ Cannot find {ckpt_path}. Please make sure it's in the current directory.")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"✅ Model loaded successfully (from Epoch {ckpt.get('epoch', '?')}, Val Loss: {ckpt.get('val_loss', 0.0):.4f})")

    # 3. Generate a fresh point cloud for testing
    print("\nGenerating a random Amigurumi target shape...")
    builder = DatasetBuilder(output_dir="/tmp/_alphaknit_test_dummy")
    # Generate 1 random valid sample
    raw_sample = None
    for i in range(10):  # try up to 10 times to get a valid sample
        raw_sample = builder._generate_one(i)
        if raw_sample is not None:
            break
            
    if raw_sample is None:
        print("⚠️ Failed to generate a valid target shape using DatasetBuilder.")
        print("Trying with a random dummy point cloud...")
        pc_np = np.random.randn(config.N_POINTS, 3).astype(np.float32)
    else:
        pc_np = raw_sample["point_cloud"]
        print(f"✅ Generated shape ID: {raw_sample.get('id', 'Unknown')}")

    # Ensure shape is (N_POINTS, 3)
    if pc_np.shape[0] > config.N_POINTS:
        idx = np.random.choice(pc_np.shape[0], config.N_POINTS, replace=False)
        pc_np = pc_np[idx]
    elif pc_np.shape[0] < config.N_POINTS:
        pad = np.zeros((config.N_POINTS - pc_np.shape[0], 3), dtype=np.float32)
        pc_np = np.concatenate([pc_np, pad], axis=0)
        
    pc_tensor = torch.tensor(pc_np, dtype=torch.float32).unsqueeze(0).to(device) # (1, N_POINTS, 3)

    # 4. Inference (Greedy Decode)
    print(f"\n🚀 Running Inference on {pc_np.shape[0]} points...")
    with torch.no_grad():
        pred_tuples = model.greedy_decode(pc_tensor, max_len=config.MAX_SEQ_LEN)
    
    pred = pred_tuples[0]
    print(f"📋 Generated sequence ({len(pred)} tuples):")
    for i, (t, p1, p2) in enumerate(pred[:20]):
        token_name = config.ID_TO_TOKEN.get(t, f"<ID:{t}>")
        print(f"   [{i:3d}] {token_name:8s} (p1={p1}, p2={p2})")
    if len(pred) > 20:
        print(f"   ... ({len(pred) - 20} more tuples)")

    # 5. Compile the sequence
    print("\n⚙️ Compiling generated sequence to Stitch Graph...")
    compiler = KnittingCompiler()
    tokens_str = [f"{config.ID_TO_TOKEN.get(t, '<UNK>')}({p1},{p2})" for t, p1, p2 in pred]
    
    try:
        graph = compiler.compile(tokens_str)
        print(f"✅ Compile SUCCESS!")
        print(f"🧶 Graph Stats: {len(graph.nodes)} stitches generated.")
        
        # Optional: Save compiler output if needed
        import networkx as nx
        from collections import Counter
        types = [n.stitch_type for n in graph.nodes.values()]
        print(f"📊 Stitch distribution: {dict(Counter(types))}")
        
    except Exception as e:
        print(f"❌ Compile FAILED: {e}")

if __name__ == "__main__":
    main()
