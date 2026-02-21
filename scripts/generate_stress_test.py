import torch
import numpy as np
import math
import sys
import os
sys.path.append(os.getcwd())
from src.alphaknit.config import VOCAB, MAX_SEQ_LEN
from src.alphaknit.simulator import ForwardSimulator
from src.alphaknit.compiler import KnittingCompiler

# Setup directories
OUTPUT_DIR = "data/stress_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

compiler = KnittingCompiler()
simulator = ForwardSimulator(stitch_width=0.5, stitch_height=0.4)

def save_sample(name, tokens, pc):
    # Pad tokens
    token_ids = [VOCAB.get(t, 0) for t in tokens]
    if len(token_ids) > MAX_SEQ_LEN:
        token_ids = token_ids[:MAX_SEQ_LEN]
    
    # Save
    np.save(f"{OUTPUT_DIR}/{name}_pc.npy", pc)
    # Save tokens as text for inspection
    with open(f"{OUTPUT_DIR}/{name}_tokens.txt", "w") as f:
        f.write(" ".join(tokens))
    
    # Also save as .pt for easy loading
    torch.save({
        "point_cloud": torch.tensor(pc, dtype=torch.float32),
        "tokens": torch.tensor(token_ids, dtype=torch.long),
        "name": name
    }, f"{OUTPUT_DIR}/{name}.pt")
    print(f"Generated {name}: {len(tokens)} tokens, {pc.shape[0]} points")

def generate_base_cylinder(rows=5, expansion=True):
    """Helper to start with mr_6 and expand to 12 or maintain."""
    tokens = ["mr_6"]
    # R1: mr_6 (6)
    
    if expansion:
        # R2: inc x 6 -> 12
        tokens.extend(["inc"] * 6)
        current = 12
    else:
        current = 6
        
    # Standard cylinder rows
    for _ in range(rows):
        tokens.extend(["sc"] * current)
        
    return tokens, current

# -----------------------------------------------------------------------------
# Case 1: Asymmetric Bulge (Bụng bia)
# clustered inc in one sector
# -----------------------------------------------------------------------------
def gen_asymmetric_bulge():
    tokens, current = generate_base_cylinder(rows=2, expansion=True) # Start 12
    
    # Bulge Row: inc concentrated in first 1/3 of stitches
    # 12 stitches -> inc on first 4, sc on rest 8
    # New count: 4*2 + 8 = 16
    for i in range(12):
        if i < 4: 
            tokens.append("inc")
        else:
            tokens.append("sc")
    current = 16
    
    # Maintain bulge for 2 rows
    for _ in range(2):
        tokens.extend(["sc"] * current)
        
    # Decrease bulge (symmetric to inc)
    # 16 stitches. The first 8 stitches correspond to the "inc" region (4 parents -> 8 children)
    # Wait, simple reversing logic: dec first 4 pairs (8 stitches -> 4), sc rest 8
    # 8 + 8 = 16
    for i in range(0, 16):
        if i < 8: # The bugle area
             # We need to act on PAIRS for dec. 
             # But here we are writing tokens. 'dec' consumes 2 parent slots.
             # So we place 'dec' tokens.
             # We want to place 'dec's such that they consume the bulge.
             pass
    
    # Simplification: Just generate the bulge geometry. We don't strictly need to close it perfectly for this test.
    # Just close normally
    return tokens

# -----------------------------------------------------------------------------
# Case 2: Bending Tube (Cùi chỏ)
# inc one side, dec other side in SAME ROW
# -----------------------------------------------------------------------------
def gen_bending_tube():
    tokens, current = generate_base_cylinder(rows=2, expansion=True) # 12
    
    # Bend Row: 
    # Front (0-5): inc x 3 (consumes 3, output 6)
    # Back (6-11): dec x 3 (consumes 6, output 3)
    # Net parent consumption: 3 + 6 = 9 != 12. Invalid row.
    # Need to balance consumption.
    
    # Try: 
    # Side A (0-5): inc x 6 (consumes 6, makes 12) -> Expansion
    # Side B (6-11): dec x 3 (consumes 6, makes 3) -> Contraction
    # Total consumed: 6 + 6 = 12 (Valid row).
    # Next row count: 12 + 3 = 15.
    
    for _ in range(6): tokens.append("inc")
    for _ in range(3): tokens.append("dec")
    
    # Maintain for visual check
    tokens.extend(["sc"] * 15)
    return tokens

# -----------------------------------------------------------------------------
# Case 3: Sharp Ridge (Vây cá)
# Stacked incs at same theta
# -----------------------------------------------------------------------------
def gen_sharp_ridge():
    tokens, current = generate_base_cylinder(rows=1, expansion=True) # 12
    
    # Ridge over 5 rows
    # ALWAYS inc at index 0, dec at index -1 to keep count constant?
    # Or just spiral ridge: inc at 0, sc rest.
    
    for r in range(5):
        # Inc at 0
        tokens.append("inc") 
        # SC rest (current - 1 items)
        tokens.extend(["sc"] * (current - 1))
        current += 1 # 12 -> 13 -> 14...
        
    return tokens

# -----------------------------------------------------------------------------
# Case 4: Phase Shift (Rotation)
# Same shape, rotated tokens
# -----------------------------------------------------------------------------
def gen_phase_shift_pair():
    # Base: Cylinder with one bump
    base = []
    base.append("mr_6")
    base.extend(["inc"] * 6) # 12
    # Bump row: inc at 0
    base.append("inc")
    base.extend(["sc"] * 11) # 13
    base.extend(["sc"] * 13)
    
    # Rotated: Shift the bump row tokens
    # Shift bump by 6 stitches (~180 degrees)
    rotated = []
    rotated.append("mr_6")
    rotated.extend(["inc"] * 6) # 12
    # Bump row shifted: sc x 6, inc, sc x 5
    rotated.extend(["sc"] * 6)
    rotated.append("inc")
    rotated.extend(["sc"] * 5)
    rotated.extend(["sc"] * 13)
    
    return base, rotated

# -----------------------------------------------------------------------------
# Case 4: Tilted Plane (Slanted Top)
# Dec concentrated on one side at the end
# -----------------------------------------------------------------------------
def gen_tilted_plane():
    tokens, current = generate_base_cylinder(rows=3, expansion=True) # 12
    
    # Slant Row 1: dec on first half (0-5), sc on rest
    # 6 stitches for dec -> 3 decs (consumes 6)
    # Rest 6 stitches -> sc x 6 (consumes 6)
    # Total parent: 12. Next: 3 + 6 = 9.
    for _ in range(3): tokens.append("dec")
    tokens.extend(["sc"] * 6)
    current = 9
    
    # Slant Row 2: dec on first part again
    # first 3 stitches (from prev decs) -> sc? 
    # Let's dec again.
    # We have 9 stitches. 
    # Dec x 1 (consumes 2) + sc x 7 = 9 consumed. Output 1+7 = 8.
    tokens.append("dec")
    tokens.extend(["sc"] * 7)
    current = 8
    
    # Just stop here.
    return tokens

# -----------------------------------------------------------------------------
# Generate All
# -----------------------------------------------------------------------------
def main():
    print("Generating Stress Test Dataset...")
    
    try:
        # Case 1
        tokens_1 = gen_asymmetric_bulge()
        print(f"Case 1 Tokens: {len(tokens_1)}")
        graph_1 = compiler.compile(tokens_1)
        pc_1 = simulator.to_point_cloud(graph_1)
        save_sample("case1_bulge", tokens_1, pc_1)
        
        # Case 2
        tokens_2 = gen_bending_tube()
        print(f"Case 2 Tokens: {len(tokens_2)}")
        graph_2 = compiler.compile(tokens_2)
        pc_2 = simulator.to_point_cloud(graph_2)
        save_sample("case2_bend", tokens_2, pc_2)
        
        # Case 3
        tokens_3 = gen_sharp_ridge()
        print(f"Case 3 Tokens: {len(tokens_3)}")
        graph_3 = compiler.compile(tokens_3)
        pc_3 = simulator.to_point_cloud(graph_3)
        save_sample("case3_ridge", tokens_3, pc_3)

        # Case 4
        tokens_4 = gen_tilted_plane()
        print(f"Case 4 Tokens: {len(tokens_4)}")
        graph_4 = compiler.compile(tokens_4)
        pc_4 = simulator.to_point_cloud(graph_4)
        save_sample("case4_tilt", tokens_4, pc_4)
        
        # Case 5 (Phase Shift)
        t_a, t_b = gen_phase_shift_pair()
        g_a = compiler.compile(t_a)
        pc_a = simulator.to_point_cloud(g_a)
        save_sample("case5_phase_A", t_a, pc_a)
        
        g_b = compiler.compile(t_b)
        pc_b = simulator.to_point_cloud(g_b)
        save_sample("case5_phase_B", t_b, pc_b)
    
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()

    print(f"Complete. Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
