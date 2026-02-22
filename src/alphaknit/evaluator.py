"""
Evaluator: Metrics for measuring AlphaKnit model quality.

Metrics:
  - token_accuracy:      % tokens correct (position-by-position)
  - stitch_count_mae:    mean absolute error in stitch count per row
  - edit_distance:       Levenshtein distance between token sequences
  - chamfer_distance:    symmetric nearest-neighbour distance between point clouds
  - evaluate_dataset:    run all metrics over N samples from a dataset
"""

import numpy as np
import torch
from typing import List

from . import config
from .compiler import KnittingCompiler, CompileError
from .simulator import ForwardSimulator


class Evaluator:

    def __init__(self):
        self._compiler = KnittingCompiler()
        self._sim = ForwardSimulator()

    # ------------------------------------------------------------------ #
    #  Individual metrics                                                  #
    # ------------------------------------------------------------------ #

    def token_accuracy(self, pred_ids: List, gt_ids: List) -> float:
        """
        Position-by-position token accuracy.
        Supports both ID lists and tuple/list lists.
        """
        if not gt_ids:
            return 1.0 if not pred_ids else 0.0

        def to_tpl(seq):
            return [tuple(x) if isinstance(x, (list, tuple, np.ndarray)) else x for x in seq]

        p_tpl = to_tpl(pred_ids)
        g_tpl = to_tpl(gt_ids)

        length = min(len(p_tpl), len(g_tpl))
        correct = sum(p == g for p, g in zip(p_tpl[:length], g_tpl[:length]))
        total = max(len(p_tpl), len(g_tpl))
        return correct / total if total > 0 else 1.0

    def stitch_count_mae(
        self, pred_tokens: List[str], gt_tokens: List[str]
    ) -> float:
        """
        Compile both token sequences and compare stitch counts per row.
        Returns MAE over rows (aligned to shorter sequence).
        Returns None if either sequence fails to compile.
        """
        try:
            pred_graph = self._compiler.compile(pred_tokens)
            gt_graph   = self._compiler.compile(gt_tokens)
        except CompileError:
            return None

        pred_counts = pred_graph.stitch_count_per_row()
        gt_counts   = gt_graph.stitch_count_per_row()

        rows = sorted(set(pred_counts) | set(gt_counts))
        if not rows:
            return 0.0

        errors = [
            abs(pred_counts.get(r, 0) - gt_counts.get(r, 0))
            for r in rows
        ]
        return float(np.mean(errors))

    def edit_distance(self, pred: List, gt: List) -> int:
        """
        Levenshtein edit distance between two sequences.
        Supports tokens as tuples, lists, or strings.
        """
        def to_tpl(seq):
            return [tuple(x) if isinstance(x, (list, tuple, np.ndarray)) else x for x in seq]
            
        p = to_tpl(pred)
        g = to_tpl(gt)

        m, n = len(p), len(g)
        # DP table
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[:]
            dp[0] = i
            for j in range(1, n + 1):
                if p[i - 1] == g[j - 1]:
                    dp[j] = prev[j - 1]
                else:
                    dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
        return dp[n]

    def chamfer_distance(self, pc1: np.ndarray, pc2: np.ndarray) -> float:
        """
        Symmetric Chamfer distance between two point clouds.

        Args:
            pc1, pc2: (N, 3) and (M, 3) numpy arrays

        Returns:
            float — lower is better, 0 for identical clouds
        """
        if pc1.shape[0] == 0 or pc2.shape[0] == 0:
            return float("inf")

        # For each point in pc1, find nearest in pc2
        diff1 = pc1[:, None, :] - pc2[None, :, :]   # (N, M, 3)
        dist1 = np.sqrt((diff1 ** 2).sum(axis=-1))   # (N, M)
        min1  = dist1.min(axis=1).mean()              # scalar

        # For each point in pc2, find nearest in pc1
        min2 = dist1.min(axis=0).mean()               # scalar

        return float(min1 + min2)

    # ------------------------------------------------------------------ #
    #  Dataset-level evaluation                                            #
    # ------------------------------------------------------------------ #

    def evaluate_dataset(
        self,
        model,
        dataset,
        n_samples: int = 100,
        device: torch.device = None,
    ) -> dict:
        """
        Run inference on n_samples from dataset and compute all metrics.

        Args:
            model:     KnittingTransformer (already loaded)
            dataset:   KnittingDataset instance
            n_samples: how many samples to evaluate
            device:    torch device

        Returns:
            dict with mean metrics and per-sample results
        """
        if device is None:
            device = torch.device("cpu")

        model.eval()
        n_samples = min(n_samples, len(dataset))

        results = []
        token_accs = []
        edit_dists = []
        sc_maes    = []
        chamfers   = []

        for i in range(n_samples):
            batch = dataset[i]
            pc, src, tgt = batch['point_cloud'], batch['src_tokens'], batch['tgt_tokens']
            pc_batch = pc.unsqueeze(0).to(device)   # (1, N, 3)

            # Greedy decode
            pred_ids = model.greedy_decode(pc_batch, max_len=config.MAX_SEQ_LEN)[0]
            gt_ids   = tgt.tolist()
            # Strip PAD and EOS from ground truth
            gt_ids = [t for t in gt_ids if t not in (config.PAD_ID, config.EOS_ID)]

            # Convert to tokens
            pred_tokens = model.ids_to_tokens(pred_ids)
            gt_tokens   = model.ids_to_tokens(gt_ids)

            # Metrics
            acc  = self.token_accuracy(pred_ids, gt_ids)
            edist = self.edit_distance(pred_ids, gt_ids)
            sc_mae = self.stitch_count_mae(pred_tokens, gt_tokens)

            # Chamfer: compile predicted tokens → point cloud
            try:
                pred_graph = self._compiler.compile(pred_tokens)
                pred_pc    = self._sim.to_point_cloud(pred_graph)
                gt_pc      = pc.numpy()
                chamfer    = self.chamfer_distance(pred_pc, gt_pc)
            except (CompileError, Exception):
                chamfer = None

            token_accs.append(acc)
            edit_dists.append(edist)
            if sc_mae is not None:
                sc_maes.append(sc_mae)
            if chamfer is not None:
                chamfers.append(chamfer)

            results.append({
                "sample_idx":   i,
                "pred_tokens":  pred_tokens[:20],   # truncate for display
                "gt_tokens":    gt_tokens[:20],
                "token_acc":    round(acc, 4),
                "edit_dist":    edist,
                "sc_mae":       round(sc_mae, 4) if sc_mae is not None else None,
                "chamfer":      round(chamfer, 4) if chamfer is not None else None,
            })

        summary = {
            "n_evaluated":       n_samples,
            "mean_token_acc":    round(float(np.mean(token_accs)), 4),
            "mean_edit_dist":    round(float(np.mean(edit_dists)), 4),
            "mean_sc_mae":       round(float(np.mean(sc_maes)), 4) if sc_maes else None,
            "mean_chamfer":      round(float(np.mean(chamfers)), 4) if chamfers else None,
            "compile_success_rate": round(len(sc_maes) / n_samples, 4),
            "per_sample":        results,
        }
        return summary
