import os
import sys
import json
import argparse

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from alphaknit.dataset_builder import DatasetBuilder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "dataset_phase9b_full")))
    parser.add_argument("--n_samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    builder = DatasetBuilder(output_dir=args.output_dir, seed=args.seed)
    samples = builder.build(n_samples=args.n_samples, verbose=True, resume=args.resume)

    with open(os.path.join(args.output_dir, "generation_metadata.json"), "w") as f:
        json.dump(
            {
                "seed": args.seed,
                "resume": args.resume,
                "generated_samples": len(samples),
            },
            f,
            indent=2,
        )
