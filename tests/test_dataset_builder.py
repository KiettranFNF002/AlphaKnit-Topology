import os
import json
import numpy as np
import pytest
import tempfile

from alphaknit.dataset_builder import DatasetBuilder, classify_shape


class TestClassifyShape:

    def test_sphere(self):
        # expand first half, contract second half
        row_data = (
            [{"row": i, "action": "expand", "stitch_count": 6 + i*6} for i in range(4)]
            + [{"row": i+4, "action": "contract", "stitch_count": 24 - i*6} for i in range(4)]
        )
        assert classify_shape(row_data) == "sphere"

    def test_cylinder(self):
        row_data = [{"row": i, "action": "maintain", "stitch_count": 12} for i in range(8)]
        assert classify_shape(row_data) == "cylinder"

    def test_cone_expand(self):
        row_data = [{"row": i, "action": "expand", "stitch_count": 6 + i*6} for i in range(6)]
        assert classify_shape(row_data) == "cone_expand"

    def test_empty(self):
        assert classify_shape([]) == "other"


class TestDatasetBuilder:

    def test_build_creates_files(self, tmp_path):
        builder = DatasetBuilder(output_dir=str(tmp_path), min_rows=3, max_rows=6)
        samples = builder.build(n_samples=5, verbose=False)
        assert len(samples) == 5

        # Check files exist
        for s in samples:
            json_path = tmp_path / f"{s['id']}.json"
            npy_path = tmp_path / f"{s['id']}.npy"
            assert json_path.exists(), f"Missing {json_path}"
            assert npy_path.exists(), f"Missing {npy_path}"

    def test_samples_have_required_keys(self, tmp_path):
        builder = DatasetBuilder(output_dir=str(tmp_path), min_rows=3, max_rows=6)
        samples = builder.build(n_samples=3, verbose=False)
        required = {"id", "edge_sequence",
                    "stitch_counts_per_row", "n_stitches", "n_rows",
                    "is_closed", "shape_class"}
        for s in samples:
            assert required.issubset(s.keys()), f"Missing keys: {required - s.keys()}"

    def test_point_clouds_are_finite(self, tmp_path):
        builder = DatasetBuilder(output_dir=str(tmp_path), min_rows=3, max_rows=6)
        samples = builder.build(n_samples=5, verbose=False)
        for s in samples:
            npy_path = tmp_path / f"{s['id']}.npy"
            pc = np.load(str(npy_path))
            assert pc.ndim == 2 and pc.shape[1] == 3, f"Bad shape: {pc.shape}"
            assert np.all(np.isfinite(pc)), "Non-finite values in point cloud"

    def test_stats_file_created(self, tmp_path):
        builder = DatasetBuilder(output_dir=str(tmp_path), min_rows=3, max_rows=6)
        builder.build(n_samples=5, verbose=False)
        stats_path = tmp_path / "dataset_stats.json"
        assert stats_path.exists()
        with open(stats_path) as f:
            stats = json.load(f)
        assert stats["total_samples"] == 5
        assert "shape_distribution" in stats
        assert "stitch_count" in stats

    def test_json_is_valid(self, tmp_path):
        builder = DatasetBuilder(output_dir=str(tmp_path), min_rows=3, max_rows=6)
        samples = builder.build(n_samples=3, verbose=False)
        for s in samples:
            json_path = tmp_path / f"{s['id']}.json"
            with open(json_path) as f:
                data = json.load(f)
            assert data["id"] == s["id"]
            assert isinstance(data["edge_sequence"], list)
            assert isinstance(data["n_stitches"], int)
