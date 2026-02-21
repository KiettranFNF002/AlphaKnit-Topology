import torch
import torch.nn.functional as F
import numpy as np
import os

@torch.no_grad()
def compute_phase_lag(model, optimizer):
    """
    Cosine similarity between gradient direction and Adam momentum (exp_avg).
    Only measures FINAL transformer + output head to focus on topological orientation.
    """
    cosines = []
    
    # Target only late layers where orientation and topology are finalized
    # We use common keywords found in model architectures
    target_keywords = ["transformer.layers.-1", "lm_head", "output_proj", "final_norm"]
    
    for name, p in model.named_parameters():
        if not any(k in name for k in target_keywords):
            continue
            
        if p.grad is None:
            continue
            
        state = optimizer.state.get(p, None)
        if not state or "exp_avg" not in state:
            continue
            
        # Memory safety: detach momentum
        grad = p.grad.detach().flatten()
        momentum = state["exp_avg"].detach().flatten()
        
        if grad.numel() == 0:
            continue
            
        # Cosine similarity between gradient direction and momentum direction
        # Note: Adam update is roughly -exp_avg, we look at the alignment
        cos = F.cosine_similarity(
            grad.unsqueeze(0),
            momentum.unsqueeze(0),
            dim=1
        )
        cosines.append(cos.item())

    if not cosines:
        return 1.0 # Default to aligned if no data
        
    return sum(cosines) / len(cosines)


class LatentPhasePortrait:
    """
    VRAM-safe telemetry. Stores exactly ONE pooled structural embedding per epoch.
    Allows visualization of the learning trajectory (Phase Portrait).
    """
    def __init__(self):
        self.history = []

    @torch.no_grad()
    def capture(self, hidden_states, structural_mask):
        """
        hidden_states: [B, T, D] (last layer hidden states)
        structural_mask: [B, T] (bool mask of topology-defining tokens)
        """
        # Ensure mask is broadcastable [B, T, 1]
        mask = structural_mask.unsqueeze(-1).float()
        
        # Mean pooling only over structural tokens across entire batch
        sum_hidden = (hidden_states * mask).sum(dim=(0, 1))
        denom = mask.sum() + 1e-6
        
        pooled = sum_hidden / denom # [D]
        self.history.append(pooled.cpu().numpy())

    def get_history(self):
        if not self.history:
            return None
        return np.stack(self.history)


class EmergenceTracker:
    """
    Detects the "Crystallization Window" for post-peak checkpoint saving.
    Monitors Velocity and Acceleration of competence metrics.
    """
    def __init__(self, window_size=5):
        self.history = []
        self.best_velocity = -1e9
        self.peak_epoch = None
        self.window_size = window_size

    def update(self, score, epoch):
        self.history.append(score)
        if len(self.history) < 2:
            return False

        # Current velocity (dComp/dt)
        vel = self.history[-1] - self.history[-2]
        
        if vel > self.best_velocity:
            self.best_velocity = vel
            self.peak_epoch = epoch
            print(f"📈 NEW EMERGENCE PEAK: Velocity {vel:.4f} at Epoch {epoch}")

        # Crystallization window: 3-5 epochs after peak
        if self.peak_epoch is not None:
            age = epoch - self.peak_epoch
            if 3 <= age <= 5:
                return True
        return False
