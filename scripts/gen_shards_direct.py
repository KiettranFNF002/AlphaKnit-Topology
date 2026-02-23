"""
Direct-to-Shard Dataset Generator (Phase 9B)

Bypasses Google Drive's 100k file limit by generating samples into memory
and directly packing them into Tensorized WebDataset Tar Shards.
"""
import os
import sys
import tarfile
import argparse
import tempfile
import json
import time
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from alphaknit.dataset_builder import DatasetBuilder
from alphaknit import config
from pack_tensor_dataset import build_tensor_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/processed/shards_phase9b_full")
    parser.add_argument("--n_samples", type=int, default=50000)
    parser.add_argument("--shard_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    builder = DatasetBuilder(output_dir="/tmp/alphaknit_dummy", seed=args.seed) # We won't actually save to this dir

    print(f"Generating {args.n_samples} samples directly to shards in {args.output_dir}...")

    skipped = 0
    samples_generated = 0
    shard_id = 0
    sample_index = 0
    attempts = 0

    if args.resume:
        existing_shards = sorted(
            f for f in os.listdir(args.output_dir)
            if f.startswith("shard-") and f.endswith(".tar")
        )
        if existing_shards:
            shard_id = int(existing_shards[-1].split("-")[1].split(".")[0]) + 1
            metadata_path = os.path.join(args.output_dir, "generation_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    metadata = json.load(f)
                samples_generated = int(metadata.get("generated_samples", 0))
            else:
                for shard_name in existing_shards:
                    with tarfile.open(os.path.join(args.output_dir, shard_name), "r") as tar:
                        count = sum(1 for m in tar.getmembers() if m.isfile() and m.name.endswith(".pt"))
                        samples_generated += count
            sample_index = samples_generated

    # We will use a temporary file to interface with tarfile and torch.save
    temp_dir = tempfile.mkdtemp()
    
    pbar = tqdm(total=args.n_samples, desc="Total Progress", initial=samples_generated)

    while samples_generated < args.n_samples:
        shard_path = os.path.join(args.output_dir, f"shard-{shard_id:04d}.tar")
        samples_in_this_shard = min(args.shard_size, args.n_samples - samples_generated)
        
        with tarfile.open(shard_path, "w") as tar:
            count_in_shard = 0
            while count_in_shard < samples_in_this_shard:
                # Generate raw sample
                attempts += 1
                raw_sample = builder._generate_one(sample_index)
                if raw_sample is None:
                    skipped += 1
                    if skipped > args.n_samples * 5:
                        print(f"Warning: too many skipped samples ({skipped}). Stopping early.")
                        return
                    continue
                if not builder._is_valid_sample(raw_sample):
                    skipped += 1
                    continue
                 
                name = raw_sample['id']
                raw_sample["generation_meta"] = {
                    "seed": args.seed,
                    "generated_at": int(time.time()),
                }
                pc = raw_sample.pop("point_cloud")
                
                # Mock the json and npy files in memory/tmp to run through our packer
                tmp_json = os.path.join(temp_dir, f"{name}.json")
                tmp_npy = os.path.join(temp_dir, f"{name}.npy")
                
                with open(tmp_json, "w") as f:
                    json.dump(raw_sample, f)
                np.save(tmp_npy, pc)

                # Pre-processing to fully padded tensorized form
                tensor_sample = build_tensor_sample(tmp_json, tmp_npy, config.MAX_SEQ_LEN, config.N_POINTS)
                
                # Save to temporary pt file
                tmp_pt = os.path.join(temp_dir, f"{name}.pt")
                torch.save(tensor_sample, tmp_pt)
                
                # Add to tar archive
                tar.add(tmp_pt, arcname=f"{name}.pt")
                
                # Cleanup tmp files
                os.remove(tmp_json)
                os.remove(tmp_npy)
                os.remove(tmp_pt)
                
                count_in_shard += 1
                samples_generated += 1
                sample_index += 1
                pbar.update(1)

        shard_id += 1

    pbar.close()
    
    # Cleanup tmp dir
    os.rmdir(temp_dir)
    invalid_ratio = skipped / max(attempts, 1)
    print(f"\nDone! Created {shard_id} shards containing {samples_generated} samples.")
    print(f"Skipped (invalid generator outputs): {skipped}")
    print(f"Invalid ratio: {invalid_ratio:.3f}")
    with open(os.path.join(args.output_dir, "generation_metadata.json"), "w") as f:
        json.dump(
            {
                "seed": args.seed,
                "resume": args.resume,
                "generated_samples": samples_generated,
                "skipped_samples": skipped,
                "invalid_ratio": invalid_ratio,
            },
            f,
            indent=2,
        )

if __name__ == "__main__":
    main()
