
import sys, os
import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.alphaknit.inference import AlphaKnitPredictor
from src.alphaknit import config

def compare_decoding():
    checkpoint_path = "checkpoints/best_model_phase8_test.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Propably smoke test checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    predictor = AlphaKnitPredictor.load(checkpoint_path, device_str="cpu")
    
    # Create a random point cloud (or use a real one if available)
    # Using random for now just to test the mechanism, though output will be nonsense
    print("Generating random point cloud...")
    pc = np.random.randn(256, 3).astype(np.float32)
    
    print("\n--- Greedy Decode ---")
    try:
        res_greedy = predictor.predict(pc, beam_width=1)
        print(f"Tokens: {res_greedy['tokens'][:10]}...")
        print(f"Valid: {res_greedy['valid']}")
        if res_greedy['errors']:
            print(f"Errors: {res_greedy['errors'][0]}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Greedy decode failed: {e}")
        return

    print("\n--- Beam Search (width=3) ---")
    try:
        res_beam = predictor.predict(pc, beam_width=3)
        print(f"Tokens: {res_beam['tokens'][:10]}...")
        print(f"Valid: {res_beam['valid']}")
        if res_beam['errors']:
            print(f"Errors: {res_beam['errors'][0]}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Beam decode failed: {e}")
        return
        
    # Compare
    if res_greedy['tokens'] == res_beam['tokens']:
        print("\nResult: Identical output (expected for random noise or strong signal)")
    else:
        print("\nResult: Beam search produced different output!")

if __name__ == "__main__":
    compare_decoding()
