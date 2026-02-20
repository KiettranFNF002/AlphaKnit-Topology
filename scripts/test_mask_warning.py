
import torch
import torch.nn as nn
import warnings

# Force warnings to raise so we catch them
warnings.simplefilter("always")

def test_mask_warning():
    print("Testing mixed masks (Float causal + Bool padding)...")
    d_model = 32
    nhead = 4
    decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
    
    B, T = 2, 10
    tgt = torch.randn(B, T, d_model)
    memory = torch.randn(B, 5, d_model)
    
    # Float causal mask (default from generate_square_subsequent_mask)
    causal_mask_float = nn.Transformer.generate_square_subsequent_mask(T)
    
    # Bool padding mask
    tgt_pad_mask_bool = torch.zeros(B, T, dtype=torch.bool)
    tgt_pad_mask_bool[0, 5:] = True # mask last 5 tokens of first sample
    
    try:
        with warnings.catch_warnings(record=True) as w:
            _ = decoder_layer(tgt, memory, tgt_mask=causal_mask_float, tgt_key_padding_mask=tgt_pad_mask_bool)
            if w:
                print(f"Caught {len(w)} warnings:")
                for warning in w:
                    print(f"- {warning.category.__name__}: {warning.message}")
            else:
                print("No warnings caught.")
    except Exception as e:
        print(f"Error: {e}")

    print("\nTesting matched masks (Bool causal + Bool padding)...")
    # Bool causal mask
    # Upper triangle is True (masked), strictly above diagonal
    causal_mask_bool = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    
    try:
        with warnings.catch_warnings(record=True) as w:
            _ = decoder_layer(tgt, memory, tgt_mask=causal_mask_bool, tgt_key_padding_mask=tgt_pad_mask_bool)
            if w:
                print(f"Caught {len(w)} warnings:")
                for warning in w:
                    print(f"- {warning.category.__name__}: {warning.message}")
            else:
                print("No warnings caught (FIX WORKED).")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_mask_warning()
