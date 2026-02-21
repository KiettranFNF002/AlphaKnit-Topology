"""
Tests for Phase 4: AI Model
  - PointNetEncoder forward pass shapes
  - KnittingTransformer forward pass shapes
  - greedy_decode output format
  - KnittingDataset loading
  - Training step runs without error
  - Loss decreases over multiple steps
"""

import os
import json
import numpy as np
import pytest
import torch
import tempfile

from src.alphaknit import config
from src.alphaknit.model import PointNetEncoder, KnittingTransformer
from src.alphaknit.knitting_dataset import KnittingDataset, make_dataloaders
from src.alphaknit.train import train_epoch, evaluate


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

@pytest.fixture
def small_dataset(tmp_path):
    """Create a tiny synthetic dataset (10 samples) for testing."""
    from src.alphaknit.dataset_builder import DatasetBuilder
    builder = DatasetBuilder(output_dir=str(tmp_path), min_rows=3, max_rows=5)
    builder.build(n_samples=10, verbose=False)
    return str(tmp_path)


@pytest.fixture
def model():
    return KnittingTransformer(
        d_model=32, n_heads=2, n_layers=1, ffn_dim=64
    )


@pytest.fixture
def device():
    return torch.device("cpu")


# ------------------------------------------------------------------ #
#  PointNetEncoder                                                     #
# ------------------------------------------------------------------ #

class TestPointNetEncoder:

    def test_output_shape(self):
        enc = PointNetEncoder(d_model=64)
        x = torch.randn(4, 128, 3)   # (B=4, N=128, 3)
        out = enc(x)
        assert out.shape == (4, 64), f"Expected (4, 64), got {out.shape}"

    def test_single_sample(self):
        enc = PointNetEncoder(d_model=32)
        x = torch.randn(1, 256, 3)
        out = enc(x)
        assert out.shape == (1, 32)

    def test_output_finite(self):
        enc = PointNetEncoder(d_model=64)
        x = torch.randn(2, 64, 3)
        out = enc(x)
        assert torch.all(torch.isfinite(out))


# ------------------------------------------------------------------ #
#  KnittingTransformer                                                 #
# ------------------------------------------------------------------ #

class TestKnittingTransformer:

    def test_forward_shape(self, model):
        B, N, T = 2, 64, 10
        pc = torch.randn(B, N, 3)
        tokens = torch.randint(0, config.VOCAB_SIZE, (B, T))
        logits = model(pc, tokens)
        assert logits.shape == (B, T, config.VOCAB_SIZE), \
            f"Expected ({B}, {T}, {config.VOCAB_SIZE}), got {logits.shape}"

    def test_forward_finite(self, model):
        pc = torch.randn(2, 64, 3)
        tokens = torch.randint(0, config.VOCAB_SIZE, (2, 8))
        logits = model(pc, tokens)
        assert torch.all(torch.isfinite(logits))

    def test_greedy_decode_format(self):
        torch.manual_seed(0)
        model = KnittingTransformer(d_model=32, n_heads=2, n_layers=1, ffn_dim=64, max_seq_len=30)
        pc = torch.randn(1, 64, 3)
        results = model.greedy_decode(pc, max_len=20)
        assert len(results) == 1
        assert isinstance(results[0], list)
        # Just check it returns a list of ints
        assert all(isinstance(x, int) for x in results[0])

    def test_ids_to_tokens(self, model):
        ids = [config.VOCAB["mr_6"], config.VOCAB["sc"], config.VOCAB["inc"]]
        tokens = model.ids_to_tokens(ids)
        assert tokens == ["mr_6", "sc", "inc"]


# ------------------------------------------------------------------ #
#  KnittingDataset                                                     #
# ------------------------------------------------------------------ #

class TestKnittingDataset:

    def test_len(self, small_dataset):
        ds = KnittingDataset(small_dataset, n_points=64, max_seq_len=50)
        assert len(ds) == 10

    def test_item_shapes(self, small_dataset):
        ds = KnittingDataset(small_dataset, n_points=64, max_seq_len=50)
        pc, src, tgt = ds[0]
        assert pc.shape == (64, 3), f"pc shape: {pc.shape}"
        assert src.shape == (50,), f"src shape: {src.shape}"
        assert tgt.shape == (50,), f"tgt shape: {tgt.shape}"

    def test_item_dtypes(self, small_dataset):
        ds = KnittingDataset(small_dataset, n_points=64, max_seq_len=50)
        pc, src, tgt = ds[0]
        assert pc.dtype == torch.float32
        assert src.dtype == torch.long
        assert tgt.dtype == torch.long

    def test_starts_with_sos(self, small_dataset):
        ds = KnittingDataset(small_dataset, n_points=64, max_seq_len=50)
        _, src, _ = ds[0]
        assert src[0].item() == config.SOS_ID

    def test_make_dataloaders(self, small_dataset):
        train_loader, val_loader = make_dataloaders(
            small_dataset, val_split=0.2, batch_size=4
        )
        assert len(train_loader) > 0
        assert len(val_loader) > 0


# ------------------------------------------------------------------ #
#  Training                                                            #
# ------------------------------------------------------------------ #

class TestTraining:

    def test_train_step_runs(self, small_dataset, model, device):
        """One training step should not raise."""
        train_loader, _ = make_dataloaders(
            small_dataset, val_split=0.2, batch_size=4
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=config.PAD_ID)
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        assert isinstance(loss, float)
        assert loss > 0

    def test_loss_decreases(self, small_dataset, device):
        """Loss should decrease over 8 epochs on a small dataset."""
        torch.manual_seed(42)
        model = KnittingTransformer(d_model=32, n_heads=2, n_layers=1, ffn_dim=64)
        train_loader, _ = make_dataloaders(
            small_dataset, val_split=0.2, batch_size=4
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=config.PAD_ID)

        losses = []
        for _ in range(8):
            loss = train_epoch(model, train_loader, optimizer, criterion, device)
            losses.append(loss)

        # Min loss should be lower than first epoch (robust to non-monotonic decrease)
        assert min(losses) < losses[0], \
            f"Loss did not decrease at all: {losses}"
