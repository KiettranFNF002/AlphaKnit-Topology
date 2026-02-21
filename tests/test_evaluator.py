"""
Tests for Phase 5: Inference & Evaluation
"""

import numpy as np
import pytest
import torch

from src.alphaknit import config
from src.alphaknit.evaluator import Evaluator
from src.alphaknit.inference import AlphaKnitPredictor
from src.alphaknit.model import KnittingTransformer
from src.alphaknit.knitting_dataset import KnittingDataset


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

@pytest.fixture
def evaluator():
    return Evaluator()


@pytest.fixture
def small_dataset(tmp_path):
    from src.alphaknit.dataset_builder import DatasetBuilder
    builder = DatasetBuilder(output_dir=str(tmp_path), min_rows=3, max_rows=5)
    builder.build(n_samples=10, verbose=False)
    return KnittingDataset(str(tmp_path), n_points=64, max_seq_len=50)


@pytest.fixture
def small_model():
    return KnittingTransformer(d_model=32, n_heads=2, n_layers=1, ffn_dim=64, max_seq_len=50)


# ------------------------------------------------------------------ #
#  Evaluator: token_accuracy                                           #
# ------------------------------------------------------------------ #

class TestTokenAccuracy:

    def test_identical(self, evaluator):
        ids = [4, 5, 6, 7]
        assert evaluator.token_accuracy(ids, ids) == 1.0

    def test_all_wrong(self, evaluator):
        pred = [4, 4, 4]
        gt   = [5, 5, 5]
        assert evaluator.token_accuracy(pred, gt) == 0.0

    def test_partial(self, evaluator):
        pred = [4, 5, 6, 7]
        gt   = [4, 5, 0, 0]
        acc = evaluator.token_accuracy(pred, gt)
        assert 0.0 < acc < 1.0

    def test_empty_gt(self, evaluator):
        assert evaluator.token_accuracy([], []) == 1.0

    def test_length_mismatch_penalised(self, evaluator):
        pred = [4, 5, 6, 7, 7, 7]   # longer
        gt   = [4, 5, 6, 7]
        acc = evaluator.token_accuracy(pred, gt)
        assert acc < 1.0


# ------------------------------------------------------------------ #
#  Evaluator: edit_distance                                            #
# ------------------------------------------------------------------ #

class TestEditDistance:

    def test_identical(self, evaluator):
        seq = [4, 5, 6]
        assert evaluator.edit_distance(seq, seq) == 0

    def test_empty(self, evaluator):
        assert evaluator.edit_distance([], []) == 0

    def test_insertion(self, evaluator):
        # "sc" vs "sc sc" — 1 insertion
        assert evaluator.edit_distance([5], [5, 5]) == 1

    def test_substitution(self, evaluator):
        # "sc" vs "inc" — 1 substitution
        assert evaluator.edit_distance([5], [6]) == 1

    def test_works_on_strings(self, evaluator):
        assert evaluator.edit_distance(["sc", "inc"], ["sc", "inc"]) == 0
        assert evaluator.edit_distance(["sc"], ["inc"]) == 1


# ------------------------------------------------------------------ #
#  Evaluator: chamfer_distance                                         #
# ------------------------------------------------------------------ #

class TestChamferDistance:

    def test_identical(self, evaluator):
        pc = np.random.randn(10, 3).astype(np.float32)
        assert evaluator.chamfer_distance(pc, pc) == pytest.approx(0.0, abs=1e-5)

    def test_offset(self, evaluator):
        pc1 = np.zeros((5, 3), dtype=np.float32)
        pc2 = np.ones((5, 3), dtype=np.float32)
        d = evaluator.chamfer_distance(pc1, pc2)
        assert d > 0

    def test_empty_returns_inf(self, evaluator):
        pc = np.zeros((5, 3), dtype=np.float32)
        assert evaluator.chamfer_distance(np.zeros((0, 3)), pc) == float("inf")


# ------------------------------------------------------------------ #
#  Evaluator: stitch_count_mae                                         #
# ------------------------------------------------------------------ #

class TestStitchCountMAE:

    def test_identical(self, evaluator):
        tokens = ["mr_6"] + ["sc"] * 6
        assert evaluator.stitch_count_mae(tokens, tokens) == pytest.approx(0.0)

    def test_different(self, evaluator):
        pred = ["mr_6"] + ["sc"] * 6
        gt   = ["mr_6"] + ["inc"] * 6
        mae = evaluator.stitch_count_mae(pred, gt)
        assert mae > 0


# ------------------------------------------------------------------ #
#  AlphaKnitPredictor                                                  #
# ------------------------------------------------------------------ #

class TestAlphaKnitPredictor:

    def test_predict_returns_dict(self, small_model):
        predictor = AlphaKnitPredictor(small_model, device=torch.device("cpu"))
        pc = np.random.randn(64, 3).astype(np.float32)
        result = predictor.predict(pc, max_len=30, n_points=64)
        assert "tokens" in result
        assert "valid" in result
        assert "graph" in result
        assert "errors" in result
        assert isinstance(result["tokens"], list)
        assert isinstance(result["valid"], bool)

    def test_predict_from_npy(self, small_model, tmp_path):
        pc = np.random.randn(64, 3).astype(np.float32)
        npy_path = str(tmp_path / "test.npy")
        np.save(npy_path, pc)
        predictor = AlphaKnitPredictor(small_model, device=torch.device("cpu"))
        result = predictor.predict_from_file(npy_path)
        assert "tokens" in result

    def test_format_result(self, small_model):
        predictor = AlphaKnitPredictor(small_model, device=torch.device("cpu"))
        pc = np.random.randn(64, 3).astype(np.float32)
        result = predictor.predict(pc, max_len=30, n_points=64)
        text = predictor.format_result(result)
        assert "Tokens" in text
        assert "Valid" in text


# ------------------------------------------------------------------ #
#  Evaluator: evaluate_dataset                                         #
# ------------------------------------------------------------------ #

class TestEvaluateDataset:

    def test_runs_on_small_dataset(self, evaluator, small_model, small_dataset):
        summary = evaluator.evaluate_dataset(
            small_model, small_dataset, n_samples=5,
            device=torch.device("cpu")
        )
        assert summary["n_evaluated"] == 5
        assert "mean_token_acc" in summary
        assert "mean_edit_dist" in summary
        assert "compile_success_rate" in summary
        assert len(summary["per_sample"]) == 5

    def test_summary_values_in_range(self, evaluator, small_model, small_dataset):
        summary = evaluator.evaluate_dataset(
            small_model, small_dataset, n_samples=5,
            device=torch.device("cpu")
        )
        assert 0.0 <= summary["mean_token_acc"] <= 1.0
        assert summary["mean_edit_dist"] >= 0
        assert 0.0 <= summary["compile_success_rate"] <= 1.0
