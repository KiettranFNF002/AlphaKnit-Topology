"""
KnittingDataset: loads (point_cloud, token_sequence) pairs from disk.

Each sample on disk:
  sample_XXXXX.json  — metadata including canonical_sequence
  sample_XXXXX.npy   — point cloud (N, 3) float32
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from . import config


class KnittingDataset(Dataset):
    """
    Loads JSON + .npy pairs from a dataset directory.

    Returns per sample:
        point_cloud : FloatTensor (N_POINTS, 3)   — padded/sampled
        src_tokens  : LongTensor  (seq_len,)       — <SOS> + tokens (teacher forcing input)
        tgt_tokens  : LongTensor  (seq_len,)       — tokens + <EOS> (prediction target)
    """

    def __init__(self, dataset_dir: str, n_points: int = config.N_POINTS,
                 max_seq_len: int = config.MAX_SEQ_LEN):
        self.dataset_dir = dataset_dir
        self.n_points = n_points
        self.max_seq_len = max_seq_len

        # Collect all sample IDs
        self.sample_ids = sorted([
            f[:-5] for f in os.listdir(dataset_dir)
            if f.endswith(".json") and f.startswith("sample_")
        ])

        if not self.sample_ids:
            raise ValueError(f"No samples found in {dataset_dir}")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]

        # Load point cloud
        npy_path = os.path.join(self.dataset_dir, f"{sid}.npy")
        pc = np.load(npy_path).astype(np.float32)  # (N, 3)

        # Pad or randomly sample to fixed N_POINTS
        pc = self._fix_point_cloud(pc)

        # Phase 9B: Load edge_sequence (list of [type_id, p1_offset, p2_offset])
        json_path = os.path.join(self.dataset_dir, f"{sid}.json")
        with open(json_path) as f:
            meta = json.load(f)

        edge_sequence = meta.get("edge_sequence", [])
        
        # Teacher forcing: src = <SOS> + tuples, tgt = tuples + <EOS>
        sos_tuple = [config.SOS_ID, 0, 0]
        eos_tuple = [config.EOS_ID, 0, 0]
        
        src = [sos_tuple] + edge_sequence
        tgt = edge_sequence + [eos_tuple]

        # Truncate to max_seq_len
        src = src[:self.max_seq_len]
        tgt = tgt[:self.max_seq_len]

        # Pad to max_seq_len
        pad_tuple = [config.PAD_ID, 0, 0]
        src = src + [pad_tuple] * (self.max_seq_len - len(src))
        tgt = tgt + [pad_tuple] * (self.max_seq_len - len(tgt))

        return (
            torch.tensor(pc, dtype=torch.float32),
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
        )

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _fix_point_cloud(self, pc: np.ndarray) -> np.ndarray:
        """Pad with zeros or randomly subsample to exactly N_POINTS."""
        n = pc.shape[0]
        if n >= self.n_points:
            # Random subsample
            idx = np.random.choice(n, self.n_points, replace=False)
            return pc[idx]
        else:
            # Pad with zeros
            pad = np.zeros((self.n_points - n, 3), dtype=np.float32)
            return np.concatenate([pc, pad], axis=0)

    def _tokenize(self, tokens: list) -> list:
        return [config.VOCAB.get(t, config.UNK_ID) for t in tokens]

    def _pad(self, seq: list, length: int) -> list:
        return seq + [config.PAD_ID] * (length - len(seq))


def make_dataloaders(dataset_dir: str, val_split: float = 0.1,
                     batch_size: int = config.BATCH_SIZE,
                     num_workers: int = 0):
    """Split dataset into train/val DataLoaders."""
    from torch.utils.data import DataLoader, random_split

    dataset = KnittingDataset(dataset_dir)
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val

    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
