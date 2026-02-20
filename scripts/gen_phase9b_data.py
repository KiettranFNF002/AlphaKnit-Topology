import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from alphaknit.dataset_builder import DatasetBuilder

if __name__ == "__main__":
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "dataset_phase9b_full"))
    os.makedirs(out_dir, exist_ok=True)
    builder = DatasetBuilder(output_dir=out_dir)
    builder.build(n_samples=50000, verbose=True)
