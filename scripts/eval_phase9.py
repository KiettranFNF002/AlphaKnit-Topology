import torch
import numpy as np
import os
import sys
sys.path.append(os.getcwd())
import math
from src.alphaknit import config
from src.alphaknit.model import KnittingTransformer
from src.alphaknit.simulator import ForwardSimulator
from src.alphaknit.compiler import KnittingCompiler
from src.alphaknit.inference import AlphaKnitPredictor

# Directories
STRESS_DIR = "data/stress_test"
CHECKPOINT = "src/checkpoints/best_model_phase9b_full.pt" # Phase 9B Full Checkpoint
CONFIG = {"d_model": 128, "n_heads": 4, "n_layers": 3, "ffn_dim": 256}

simulator = ForwardSimulator()
compiler = KnittingCompiler()

def compute_spatial_entropy(tokens):
    """
    Measure how 'clustered' the increases/decreases are.
    Low entropy = Clustered (Good for asymmetric).
    High entropy = Uniform (Bad/Memorization).
    """
    # 1. Identify rows (naive approach or compiler-based).
    # Simple: just look at sliding windows of 6-10 tokens.
    # If we see "inc inc inc", that's low entropy locally.
    # If "sc inc sc inc sc inc", high entropy.
    
    # Phase 9B tuple strings look like "inc(1,0)"
    inc_counts = [1 if t.startswith("inc") else 0 for t in tokens]
    if sum(inc_counts) == 0: return 0.0
    
    # Calculate Gini coefficient or Entropy of intervals between incs?
    # Simple metric: Ratio of (inc followed by inc) / total_inc
    inc_pairs = 0
    for i in range(len(tokens)-1):
        if tokens[i].startswith("inc") and tokens[i+1].startswith("inc"):
            inc_pairs += 1
            
    clustering_score = inc_pairs / max(1, sum(inc_counts))
    return clustering_score # Higher is MORE clustered (Better for bulge)

def eval_rotation_sensitivity(predictor, pc_path):
    """
    Rotate input PC by 90 degrees and check if output *pattern* changes
    in a way that reflects rotation (hard) or just changes at all (sensitivity).
    """
    pc = np.load(pc_path) # (N, 3)
    
    # Rot 90 deg around Y
    theta = np.pi / 2
    rot_matrix = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    pc_rot = pc @ rot_matrix.T
    
    # Predict
    res_orig = predictor.predict(pc, beam_width=1)
    res_rot = predictor.predict(pc_rot, beam_width=1)
    
    tokens_orig = res_orig["tokens"]
    tokens_rot = res_rot["tokens"]
    
    # Compare
    # Exact match = Bad (Rotation Invariant/Blind)
    # Different = Good (Sensitive)
    # Ideally: Shifted.
    
    is_identical = (tokens_orig == tokens_rot)
    return not is_identical, tokens_orig, tokens_rot

def main():
    if not os.path.exists(CHECKPOINT):
        print(f"Checkpoint {CHECKPOINT} not found. Using Phase 8 best for baseline.")
        model_path = "checkpoints/best_model_phase8.pt"
    else:
        model_path = CHECKPOINT

    print(f"Evaluating Model: {model_path}")
    predictor = AlphaKnitPredictor.load(model_path, device_str="cpu")
    
    # Load Stress Cases
    cases = ["case1_bulge", "case2_bend", "case3_ridge", "case4_tilt"]
    
    print("\n--- Lie Detector Metrics ---")
    
    for case in cases:
        path = f"{STRESS_DIR}/{case}_pc.npy"
        if not os.path.exists(path): continue
        
        pc = np.load(path)
        result = predictor.predict(pc, beam_width=3) # Use beam search for best chance
        tokens = result["tokens"]
        
        # 1. Compile Rate
        is_valid = result["valid"]
        
        # 2. Spatial Clustering (Entropy)
        clustering = compute_spatial_entropy(tokens)
        
        # 3. Rotation Test
        is_sensitive, t_orig, t_rot = eval_rotation_sensitivity(predictor, path)
        
        print(f"Case: {case}")
        print(f"  Valid: {is_valid}")
        print(f"  Inc Clustering: {clustering:.2f} (Target: >0.3 for Bulge)")
        print(f"  Rotation Sensitive: {is_sensitive}")
        print(f"  Output len: {len(tokens)}")
        if not is_valid:
            print(f"  Errors: {result.get('errors')}")

if __name__ == "__main__":
    main()
