"""
DatasetBuilder: Generates a large canonical synthetic dataset.

Each sample contains:
  - canonical flat token sequence
  - stitch counts per row
  - shape class (sphere / cylinder / cone / other)
  - point cloud saved as .npy

Pipeline per sample:
  generate → canonicalize → compile → validate → simulate → save
"""

import json
import os
import time
import random
import numpy as np

from .generator_v2 import SpatialGeneratorV2
from .validator import GraphValidator
from .simulator import ForwardSimulator


class DatasetBuilder:

    def __init__(
        self,
        output_dir: str = "data/processed/dataset_phase9b",
        min_rows: int = 4,
        max_rows: int = 20,
        stitch_width: float = 0.5,
        stitch_height: float = 0.4,
        seed: int = 42,
    ):
        self.output_dir = output_dir
        self.min_rows = min_rows
        self.max_rows = max_rows
        self.seed = seed

        # Phase 9B: Edge-Action Generator + DAG construction
        self._gen = SpatialGeneratorV2(min_rows=min_rows, max_rows=max_rows)
        self._validator = GraphValidator()
        self._sim = ForwardSimulator(stitch_width=stitch_width, stitch_height=stitch_height)

        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def build(self, n_samples: int, verbose: bool = True, resume: bool = False) -> list:
        """
        Generate n_samples canonical samples and save to disk.

        Returns:
            List of sample metadata dicts (without point cloud arrays).
        """
        samples = []
        skipped = 0
        existing_ids = []
        if resume:
            existing_ids = sorted(
                f[:-5] for f in os.listdir(self.output_dir)
                if f.endswith(".json") and f.startswith("sample_")
            )
        if existing_ids:
            next_index = max(int(sid.split("_")[-1]) for sid in existing_ids) + 1
        else:
            next_index = 0
        attempts = 0

        while len(samples) < n_samples:
            attempts += 1
            sample = self._generate_one(next_index + len(samples))
            if sample is None:
                skipped += 1
                if skipped > n_samples * 5:
                    print(f"Warning: too many skipped samples ({skipped}). Stopping early.")
                    break
                continue
            if not self._is_valid_sample(sample):
                skipped += 1
                continue

            # Save point cloud
            npy_path = os.path.join(self.output_dir, f"{sample['id']}.npy")
            np.save(npy_path, sample.pop("point_cloud"))

            sample["generation_meta"] = {
                "seed": self.seed,
                "generated_at": int(time.time()),
            }
            # Save metadata JSON
            json_path = os.path.join(self.output_dir, f"{sample['id']}.json")
            with open(json_path, "w") as f:
                json.dump(sample, f, indent=2)
            samples.append(sample)

            if verbose and len(samples) % max(1, n_samples // 10) == 0:
                print(f"  Generated {len(samples)}/{n_samples} samples...")

        self._save_stats(samples, skipped=skipped, attempts=attempts)

        if verbose:
            print(f"\nDone: {len(samples)} samples saved to '{self.output_dir}'")
            print(f"Skipped (invalid): {skipped}")
            if attempts:
                print(f"Invalid ratio: {skipped / attempts:.3f}")

        return samples

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _generate_one(self, idx: int) -> dict | None:
        """Generate and validate one sample. Returns None if invalid."""
        py_state = random.getstate()
        np_state = np.random.get_state()
        try:
            random.seed(self.seed + idx)
            np.random.seed(self.seed + idx)
            raw = self._gen.generate_pattern()
            edge_sequence = raw["edge_sequence"]
            row_data = raw["metadata"]
            graph = raw["stitch_graph"]

            # Validate — skip samples with critical errors
            errors = self._validator.validate(graph)
            critical = [e for e in errors if e.severity == "critical"]
            if critical:
                return None

            # Simulate
            pc = self._sim.to_point_cloud(graph)
            if not np.all(np.isfinite(pc)):
                return None

            # Phase 9C: PCA Canonicalization
            pc_aligned = self._canonicalize_point_cloud(pc)
            
            counts = graph.stitch_count_per_row()
            shape_class = classify_shape(row_data)

            sample_id = f"sample_{idx:05d}"
            return {
                "id": sample_id,
                "edge_sequence": edge_sequence,
                "stitch_counts_per_row": {str(k): v for k, v in counts.items()},
                "n_stitches": graph.size,
                "n_rows": len(counts),
                "is_closed": raw["is_closed"],
                "shape_class": shape_class,
                "point_cloud": pc_aligned,          # popped before JSON save
            }

        except Exception:
            return None
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)

    def _is_valid_sample(self, sample: dict) -> bool:
        edge_sequence = sample.get("edge_sequence", [])
        return (
            bool(edge_sequence)
            and sample.get("n_stitches", 0) > 0
            and sample.get("n_rows", 0) > 0
        )

    def _canonicalize_point_cloud(self, pc: np.ndarray) -> np.ndarray:
        """
        Align the point cloud's primary axis to the X-axis using PCA on the XZ plane.
        Includes a density-based fallback for 180-degree ambiguity and symmetric shapes.
        Note: The simulator outputs (x, y, z) where y is the vertical axis.
        """
        # 1. Center on the XZ plane (ignore Y for orientation)
        pc_xz = pc[:, [0, 2]]
        centroid = np.mean(pc_xz, axis=0)
        centered_xz = pc_xz - centroid
        
        # 2. Compute PCA
        cov_matrix = np.cov(centered_xz, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Primary eigenvector (anchor direction)
        anchor = eigenvectors[:, 0]
        
        # 3. Symmetry Fallback & Ambiguity Resolution
        # If eigenvalues are very close (e.g., sphere/cylinder), PCA axis is unstable.
        # We use a density heuristic: project points onto the anchor.
        # Ensure the side with HIGHER point density (e.g., a bulge) faces +X.
        projections = centered_xz @ anchor
        skewness = np.mean(projections**3)  # simple measure of density skew
        
        if skewness < 0:
            anchor = -anchor
        # Fallback if perfectly symmetric: just ensure anchor_x > 0
        elif abs(skewness) < 1e-5 and anchor[0] < 0:
            anchor = -anchor
            
        # 4. Compute rotation angle to align anchor to positive X-axis
        # anchor = (u_x, u_z)
        theta = np.arctan2(anchor[1], anchor[0])
        
        # We need to rotate by -theta to bring anchor to (1, 0)
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        
        # 3D Rotation matrix around Y axis
        rot_y = np.array([
            [cos_t,  0, sin_t],
            [0,      1, 0    ],
            [-sin_t, 0, cos_t]
        ])
        
        # Apply rotation (centroid centering is kept relative to origin for full PC)
        # Note: We don't permanently subtract centroid, just rotate the whole shape.
        pc_aligned = pc @ rot_y.T
        return pc_aligned

    def _save_stats(self, samples: list, skipped: int = 0, attempts: int = 0):
        """Save dataset statistics to dataset_stats.json."""
        if not samples:
            return

        shape_counts = {}
        stitch_counts = []
        row_counts = []
        edge_counts = []

        for s in samples:
            sc = s.get("shape_class", "other")
            shape_counts[sc] = shape_counts.get(sc, 0) + 1
            stitch_counts.append(s["n_stitches"])
            row_counts.append(s["n_rows"])
            edge_counts.append(len(s.get("edge_sequence", [])))

        stats = {
            "total_samples": len(samples),
            "avg_nodes": float(sum(stitch_counts) / len(stitch_counts)),
            "avg_edges": float(sum(edge_counts) / len(edge_counts)),
            "invalid_ratio": float(skipped / attempts) if attempts > 0 else 0.0,
            "shape_distribution": shape_counts,
            "stitch_count": {
                "min": int(min(stitch_counts)),
                "max": int(max(stitch_counts)),
                "mean": float(sum(stitch_counts) / len(stitch_counts)),
            },
            "row_count": {
                "min": int(min(row_counts)),
                "max": int(max(row_counts)),
                "mean": float(sum(row_counts) / len(row_counts)),
            },
        }

        stats_path = os.path.join(self.output_dir, "dataset_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        return stats


# ------------------------------------------------------------------ #
#  Shape classifier (standalone function)                             #
# ------------------------------------------------------------------ #

def classify_shape(row_data: list) -> str:
    """
    Classify the overall shape from row metadata.

    Args:
        row_data: list of {"row": int, "action": str, "stitch_count": int}

    Returns:
        "sphere" | "cylinder" | "cone_expand" | "cone_contract" | "other"
    """
    if not row_data:
        return "other"

    actions = [r["action"] for r in row_data]
    n = len(actions)

    expand_count = actions.count("expand")
    maintain_count = actions.count("maintain")
    contract_count = actions.count("contract")

    # Cylinder: mostly maintain
    if maintain_count >= n * 0.6:
        return "cylinder"

    # Sphere: expand in first half, contract in second half
    mid = n // 2
    first_half = actions[:mid]
    second_half = actions[mid:]
    first_expand = first_half.count("expand")
    second_contract = second_half.count("contract")

    if first_expand >= mid * 0.4 and second_contract >= (n - mid) * 0.4:
        return "sphere"

    # Cone expanding: mostly expand, little contract
    if expand_count >= n * 0.5 and contract_count <= n * 0.2:
        return "cone_expand"

    # Cone contracting: mostly contract, little expand
    if contract_count >= n * 0.5 and expand_count <= n * 0.2:
        return "cone_contract"

    return "other"
