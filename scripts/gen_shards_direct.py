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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    builder = DatasetBuilder(output_dir="/tmp/alphaknit_dummy") # We won't actually save to this dir

    print(f"Generating {args.n_samples} samples directly to shards in {args.output_dir}...")

    skipped = 0
    samples_generated = 0
    shard_id = 0
    
    # We will use a temporary file to interface with tarfile and torch.save
    temp_dir = tempfile.mkdtemp()
    
    pbar = tqdm(total=args.n_samples, desc="Total Progress")

    while samples_generated < args.n_samples:
        shard_path = os.path.join(args.output_dir, f"shard-{shard_id:04d}.tar")
        samples_in_this_shard = min(args.shard_size, args.n_samples - samples_generated)
        
        with tarfile.open(shard_path, "w") as tar:
            count_in_shard = 0
            while count_in_shard < samples_in_this_shard:
                # Generate raw sample
                raw_sample = builder._generate_one(samples_generated + count_in_shard)
                if raw_sample is None:
                    skipped += 1
                    if skipped > args.n_samples * 5:
                        print(f"Warning: too many skipped samples ({skipped}). Stopping early.")
                        return
                    continue
                
                name = raw_sample['id']
                pc = raw_sample.pop("point_cloud")
                
                # Mock the json and npy files in memory/tmp to run through our packer
                tmp_json = os.path.join(temp_dir, f"{name}.json")
                tmp_npy = os.path.join(temp_dir, f"{name}.npy")
                
                import json
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
                pbar.update(1)

        shard_id += 1

    pbar.close()
    
    # Cleanup tmp dir
    os.rmdir(temp_dir)
    print(f"\nDone! Created {shard_id} shards containing {samples_generated} samples.")
    print(f"Skipped (invalid generator outputs): {skipped}")

if __name__ == "__main__":
    main()
