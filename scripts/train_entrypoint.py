"""Official training entrypoint (Phase 9+)."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.alphaknit.train import train


if __name__ == "__main__":
    train(
        dataset_dir="data/processed/dataset",
        checkpoint_dir="checkpoints",
        run_name="phase9",
        device_str="auto",
    )
