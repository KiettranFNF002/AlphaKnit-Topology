"""
AlphaKnitPredictor: End-to-end inference pipeline.

Usage:
    predictor = AlphaKnitPredictor.load("checkpoints/best_model.pt")
    result = predictor.predict(point_cloud_npy)
    print(result["tokens"])   # ["mr_6", "inc", "sc", ...]
    print(result["valid"])    # True if graph compiles and validates
"""

import numpy as np
import torch

from . import config
from .model import KnittingTransformer
from .compiler import KnittingCompiler, CompileError
from .validator import GraphValidator
from .simulator import ForwardSimulator


class AlphaKnitPredictor:

    def __init__(self, model: KnittingTransformer, device: torch.device = None):
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

        self._compiler = KnittingCompiler()
        self._validator = GraphValidator()
        self._sim = ForwardSimulator()

    # ------------------------------------------------------------------ #
    #  Factory                                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, checkpoint_path: str, device_str: str = "auto") -> "AlphaKnitPredictor":
        """
        Load a trained model from a checkpoint file.

        Args:
            checkpoint_path: path to .pt checkpoint saved by train.py
            device_str:      "auto", "cpu", or "cuda"

        Returns:
            AlphaKnitPredictor instance
        """
        if device_str == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_str)

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

        model = KnittingTransformer()
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} "
                  f"(val_loss={checkpoint.get('val_loss', '?'):.4f})")
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # HuggingFace style or Phase 8 resume style
            model.load_state_dict(checkpoint["model_state_dict"])
            print("Loaded checkpoint with 'model_state_dict'")
        else:
            # Assume direct state_dict
            try:
                model.load_state_dict(checkpoint)
                print("Loaded checkpoint as direct state_dict")
            except Exception as e:
                print(f"Warning: Failed to load state dict directly: {e}")
                # Fallback or re-raise
                raise e

        return cls(model, device)

    # ------------------------------------------------------------------ #
    #  Inference                                                           #
    # ------------------------------------------------------------------ #

    def predict(
        self,
        point_cloud: np.ndarray,
        max_len: int = config.MAX_SEQ_LEN,
        n_points: int = config.N_POINTS,
        beam_width: int = 1,
    ) -> dict:
        """
        End-to-end prediction: point cloud → stitch tokens → validated graph.

        Args:
            point_cloud: (N, 3) numpy array (any N)
            max_len:     maximum token sequence length to generate
            n_points:    fixed point cloud size (pad/sample to this)
            beam_width:  1=greedy (fast), >1=compile-guided beam search

        Returns:
            dict with keys:
              tokens   : list of stitch token strings
              token_ids: list of token ids
              graph    : StitchGraph (or None if compile failed)
              valid    : bool — True if graph compiled and validated
              errors   : list of ValidationError (empty if valid)
        """
        # Normalise point cloud to fixed size
        pc = self._fix_point_cloud(point_cloud, n_points)
        # Phase 9B: Apply Learned Orientation Anchor (PCA Alignment)
        pc = self._canonicalize_point_cloud(pc)
        pc_tensor = torch.tensor(pc, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Decode
        with torch.no_grad():
            if beam_width > 1:
                pred_ids = self.model.beam_decode(
                    pc_tensor, beam_width=beam_width, max_len=max_len, compile_guided=True
                )
            else:
                pred_ids = self.model.greedy_decode(pc_tensor, max_len=max_len)[0]

        tokens = self.model.ids_to_tokens(pred_ids)

        # Compile and validate
        graph = None
        errors = []
        valid = False

        try:
            graph = self._compiler.compile(tokens)
            errors = self._validator.validate(graph)
            critical = [e for e in errors if e.severity == "critical"]
            valid = len(critical) == 0
        except CompileError as e:
            errors = [str(e)]

        return {
            "tokens":    tokens,
            "token_ids": pred_ids,
            "graph":     graph,
            "valid":     valid,
            "errors":    errors,
        }

    def predict_from_file(self, npy_path: str) -> dict:
        """Load a .npy point cloud file and predict."""
        pc = np.load(npy_path)
        return self.predict(pc)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _fix_point_cloud(self, pc: np.ndarray, n_points: int) -> np.ndarray:
        """Pad with zeros or randomly subsample to exactly n_points."""
        pc = pc.astype(np.float32)
        n = pc.shape[0]
        if n >= n_points:
            idx = np.random.choice(n, n_points, replace=False)
            return pc[idx]
        else:
            pad = np.zeros((n_points - n, 3), dtype=np.float32)
            return np.concatenate([pc, pad], axis=0)

    def _canonicalize_point_cloud(self, pc: np.ndarray) -> np.ndarray:
        """
        Phase 9B: Align the point cloud's primary axis to the X-axis using PCA on the XZ plane.
        Resolves 180-degree ambiguity using density skewness.
        """
        pc_xz = pc[:, [0, 2]]
        centroid = np.mean(pc_xz, axis=0)
        centered_xz = pc_xz - centroid
        
        cov_matrix = np.cov(centered_xz, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        anchor = eigenvectors[:, 0]
        
        projections = centered_xz @ anchor
        skewness = np.mean(projections**3)
        
        if skewness < 0:
            anchor = -anchor
        elif abs(skewness) < 1e-5 and anchor[0] < 0:
            anchor = -anchor
            
        theta = np.arctan2(anchor[1], anchor[0])
        
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        
        rot_y = np.array([
            [cos_t,  0, sin_t],
            [0,      1, 0    ],
            [-sin_t, 0, cos_t]
        ])
        
        return pc @ rot_y.T

    def format_result(self, result: dict) -> str:
        """Pretty-print a prediction result."""
        lines = [
            f"Tokens ({len(result['tokens'])}): {' '.join(result['tokens'][:30])}{'...' if len(result['tokens']) > 30 else ''}",
            f"Valid:  {result['valid']}",
        ]
        if result["graph"]:
            counts = result["graph"].stitch_count_per_row()
            lines.append(f"Rows:   {len(counts)} | Stitches: {result['graph'].size}")
            lines.append(f"Counts: {dict(list(counts.items())[:8])}")
        if result["errors"]:
            lines.append(f"Errors: {result['errors'][:3]}")
        return "\n".join(lines)
