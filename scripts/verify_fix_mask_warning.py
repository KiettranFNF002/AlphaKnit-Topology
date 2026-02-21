
import sys, os
import torch
import warnings

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.alphaknit.model import KnittingTransformer
from src.alphaknit import config

def verify_fix():
    print("Verifying fix in KnittingTransformer...")
    
    # Force warnings to raise so we catch them
    warnings.simplefilter("always")
    
    model = KnittingTransformer()
    model.eval()
    
    B, N, T = 2, 100, 20
    point_cloud = torch.randn(B, N, 3)
    tgt_tokens = torch.randint(0, config.VOCAB_SIZE, (B, T))
    
    # Create padding mask (bool)
    pad_mask = (tgt_tokens == config.PAD_ID)
    
    try:
        with warnings.catch_warnings(record=True) as w:
            _ = model(point_cloud, tgt_tokens, tgt_key_padding_mask=pad_mask)
            
            relevant_warnings = [
                warn for warn in w 
                if "support for mismatched key_padding_mask and attn_mask is deprecated" in str(warn.message).lower()
            ]
            
            if relevant_warnings:
                print(f"FAILED: Caught {len(relevant_warnings)} relevant warnings:")
                for warning in relevant_warnings:
                    print(f"- {warning.message}")
            else:
                print("PASSED: No relevant warnings caught.")
                
    except Exception as e:
        print(f"Error during forward pass: {e}")

if __name__ == "__main__":
    verify_fix()
