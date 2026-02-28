import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os

class ReproducibleContext:
    """
    v6.6-G: Technical Rigor.
    Captures and restores full RNG state across multiple frameworks.
    """
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu_rng_state = None
        self.cuda_rng_state = None
        self.numpy_rng_state = None
        self.random_state = None

    def __enter__(self):
        self.cpu_rng_state = torch.get_rng_state()
        if self.device.type == 'cuda':
            self.cuda_rng_state = torch.cuda.get_rng_state(self.device)
        self.numpy_rng_state = np.random.get_state()
        self.random_state = random.getstate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_rng_state(self.cpu_rng_state)
        if self.device.type == 'cuda' and self.cuda_rng_state is not None:
            torch.cuda.set_rng_state(self.cuda_rng_state, self.device)
        np.random.set_state(self.numpy_rng_state)
        random.setstate(self.random_state)

@torch.no_grad()
def compute_phase_lag(model, optimizer, eps=1e-8):
    """
    v6.6-F: True Adam Update Direction Phase Lag.
    Cosine similarity between gradient direction and actual Adam update vector.
    Grounds energy in proper learning rate scales.
    """
    cosines = []
    update_energies = []
    
    # Identify last layer name dynamically (no more "-1")
    last_layer_prefix = None
    for name, _ in model.named_parameters():
         if "transformer.layers" in name:
             parts = name.split(".")
             idx = int(parts[2])
             if last_layer_prefix is None or idx > int(last_layer_prefix.split(".")[-1]):
                 last_layer_prefix = f"transformer.layers.{idx}"
    
    target_keywords = ["lm_head", "output_proj", "final_norm"]
    if last_layer_prefix:
        target_keywords.append(last_layer_prefix)
    
    beta1, beta2 = 0.9, 0.999
    lr = 1e-3
    for group in optimizer.param_groups:
        beta1, beta2 = group.get('betas', (0.9, 0.999))
        lr = group.get('lr', 1e-3)
        break

    for name, p in model.named_parameters():
        if not any(k in name for k in target_keywords) or p.grad is None:
            continue
            
        state = optimizer.state.get(p, None)
        if not state or "exp_avg" not in state or "exp_avg_sq" not in state:
            continue
            
        step = state.get('step', 0)
        if isinstance(step, torch.Tensor):
            step = step.item()
            
        if step == 0: continue

        grad = p.grad.detach().flatten()
        m = state["exp_avg"].detach().flatten()
        v = state["exp_avg_sq"].detach().flatten()
        
        # Bias correction
        m_hat = m / (1 - beta1 ** step)
        v_hat = v / (1 - beta2 ** step)
        
        # Actual Update Direction (Grounded with LR and normalized by second moment)
        update_dir = -lr * m_hat / (torch.sqrt(v_hat) + eps)
        
        if grad.numel() == 0 or update_dir.numel() == 0:
            continue
            
        cos = F.cosine_similarity(grad.unsqueeze(0), update_dir.unsqueeze(0), dim=1)
        cosines.append(cos.item())
        update_energies.append(torch.norm(update_dir).item())

    if not cosines:
        return 1.0, 0.0
        
    avg_lag = sum(cosines) / len(cosines)
    avg_energy = sum(update_energies) / len(update_energies)
    return avg_lag, avg_energy


class HiddenProbePool:
    """
    v6.6-F: Observer Decoupling.
    Manages multiple orthogonal probe batches and rotates them to prevent instrument internalization.
    """
    def __init__(self, probe_loader, num_pools=3):
        self.pools = []
        self.active_idx = 0
        self.rotation_count = 0
        
        # Partition probe_loader into distinct pools using array_split to handle uneven sizes
        all_batches = list(probe_loader)
        if not all_batches:
            self.pools = [[]]
            return
            
        idx_splits = np.array_split(range(len(all_batches)), num_pools)
        self.pools = [[all_batches[i] for i in s] for s in idx_splits]
        
    def get_batch(self):
        pool = self.pools[self.active_idx]
        return pool[np.random.randint(len(pool))]

    def rotate(self):
        self.active_idx = (self.active_idx + 1) % len(self.pools)
        self.rotation_count += 1
        print(f"🔄 DECOUPLING: Probe Pool rotated to idx {self.active_idx}")

    def compute_pib(self, model, train_grads_dict, criterion, device):
        """
        Probe Interference Bias (PIB): measures how much the probe gradients
        align with the training gradients. High alignment = instrument internalization.
        """
        pool = self.pools[self.active_idx]
        if not pool:
            return 0.0

        raw = pool[np.random.randint(len(pool))]
        # Normalize: WebDataset returns (pc, src, tgt) tuples
        if isinstance(raw, (list, tuple)):
            batch = {'point_cloud': raw[0], 'src_tokens': raw[1], 'tgt_tokens': raw[2]}
        else:
            batch = raw

        inputs = batch['point_cloud'].to(device)
        src = batch['src_tokens'].to(device)

        model.eval()
        with torch.random.fork_rng():
            torch.manual_seed(42)
            model.zero_grad()
            pad_mask = (src[:, :, 0] == 0)
            outputs = model(inputs, src, tgt_key_padding_mask=pad_mask)
            logits_type = outputs[0] if isinstance(outputs, tuple) else outputs
            tgt = batch['tgt_tokens'].to(device)
            B, T, _ = tgt.shape
            V = logits_type.shape[-1]
            loss = criterion(logits_type.reshape(B * T, V), tgt[:, :, 0].reshape(B * T))
            loss.backward()

        cos_sims = []
        last_layer_prefix = None
        for name, _ in model.named_parameters():
            if "transformer.layers" in name:
                parts = name.split(".")
                idx = int(parts[2])
                if last_layer_prefix is None or idx > int(last_layer_prefix.split(".")[-1]):
                    last_layer_prefix = f"transformer.layers.{idx}"

        for name, p in model.named_parameters():
            if last_layer_prefix and name.startswith(last_layer_prefix) and p.grad is not None and name in train_grads_dict:
                g_p = p.grad.detach().flatten()
                g_t = train_grads_dict[name].detach().flatten()
                sim = torch.nn.functional.cosine_similarity(g_p.unsqueeze(0), g_t.unsqueeze(0))
                cos_sims.append(sim.item())

        model.zero_grad()
        model.train()

        if not cos_sims:
            return 0.0
        return sum(cos_sims) / len(cos_sims)


class LatentPhasePortrait:
    """
    VRAM-safe telemetry. Stores exactly ONE pooled structural embedding per epoch.
    Allows visualization of the learning trajectory (Phase Portrait).
    v6.6-F Level 2: Supports Fixed Anchor Curvature to eliminate data noise.
    """
    def __init__(self):
        self.history = []
        self.anchor_history = [] # For fixed anchor curvature
        self.locked_basis = None 

    @torch.no_grad()
    def capture(self, hidden_states, structural_mask, is_anchor=False):
        """
        hidden_states: [B, T, D] (last layer hidden states)
        structural_mask: [B, T] (bool mask of topology-defining tokens)
        """
        if hidden_states is None:
            return

        h = hidden_states.detach()
        mask = structural_mask.unsqueeze(-1).float()
        
        # Grounding: Mean-of-means pooling to remove batch size bias
        per_sample = (h * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6) # [B, D]
        pooled = per_sample.mean(dim=0) # [D]
        
        vector = pooled.cpu().float().numpy()
        if is_anchor:
            self.anchor_history.append(vector)
        else:
            self.history.append(vector)

    def get_curvature(self):
        """Calculates curvature ONLY on fixed anchors to isolate model behavior."""
        if len(self.anchor_history) < 2:
            return 0.0
        v1 = self.anchor_history[-2]
        v2 = self.anchor_history[-1]
        return np.linalg.norm(v2 - v1) / (np.linalg.norm(v1) + 1e-8)

    def get_history(self):
        if not self.history:
            return None
        return np.stack(self.history)


class ModelRealityAnchors:
    """
    v6.6-F: Grounding telemetry in physical invariants.
    Tracks Normalized Weight Curvature and Representational Rank (SVD).
    """
    def __init__(self):
        self.prev_weights = {}
        self.history = {"curvature": [], "rank": [], "mi_leak": [], "stability": []}
        self.prev_latents = None

    @torch.no_grad()
    def update(self, model, current_latents):
        """
        current_latents: [D] (from LatentPhasePortrait)
        """
        # 1. Normalized Weight Curvature
        last_layer_prefix = None
        for name, _ in model.named_parameters():
             if "transformer.layers" in name:
                 parts = name.split(".")
                 idx = int(parts[2])
                 if last_layer_prefix is None or idx > int(last_layer_prefix.split(".")[-1]):
                     last_layer_prefix = f"transformer.layers.{idx}"

        curvatures = []
        for name, p in model.named_parameters():
            if last_layer_prefix and name.startswith(last_layer_prefix): 
                w = p.detach().cpu()
                if name in self.prev_weights:
                    diff = torch.norm(w - self.prev_weights[name])
                    norm = torch.norm(self.prev_weights[name]) + 1e-8
                    curvatures.append((diff / norm).item())
                self.prev_weights[name] = w
        
        if curvatures:
            self.history["curvature"].append(sum(curvatures) / len(curvatures))

        # 2. Representation Stability: cos(L_t, L_{t-1})
        if self.prev_latents is not None and current_latents is not None:
            l1 = torch.from_numpy(self.prev_latents)
            l2 = torch.from_numpy(current_latents)
            stab = F.cosine_similarity(l1.unsqueeze(0), l2.unsqueeze(0)).item()
            self.history["stability"].append(stab)
        
        self.prev_latents = current_latents

    @torch.no_grad()
    def compute_rank(self, latents_batch):
        """
        Spectral Rank: exp(Entropy(SingularValues))
        v6.6-F Level 2: Normalized by sqrt(N) to remove sample size bias.
        """
        if latents_batch.size(0) < 2: return 0.0
        
        centered = latents_batch - latents_batch.mean(dim=0)
        # Normalize by sqrt of samples to stabilize scale
        centered = centered / (np.sqrt(latents_batch.size(0)) + 1e-8)
        
        # v6.6-F Grounding: Use fast and stable linalg.svdvals
        S = torch.linalg.svdvals(centered)
        
        probs = S / (S.sum() + 1e-8)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        rank = torch.exp(entropy).item()
        self.history["rank"].append(rank)
        return rank

    def track_mi_leak(self, latents, probe_labels):
        """
        Feature Mutual Information proxy: Correlation between probe identity and latents.
        High correlation = Instrument Internalization.
        """
        l_norm = torch.norm(latents, dim=-1).cpu().numpy()
        y = probe_labels.cpu().numpy().flatten()[:len(l_norm)]
        
        if len(np.unique(y)) < 2: return 0.0
        
        corr = np.abs(np.corrcoef(l_norm, y)[0, 1])
        self.history["mi_leak"].append(corr)
        return corr


class FeatureFingerprint:
    """
    v6.6-F Level 3: Mechanistic Identity.
    Tracks the principal directions of hidden activations to identify "Structural Invariants".
    If the top K directions (eigenvectors) stabilize, it confirms a "Representational Discovery".
    """
    def __init__(self, top_k=5):
        self.top_k = top_k
        self.history = [] # List of top_k eigenvectors [K, D]
        self.persistence = [] # Cosine similarity of top_k directions over time

    @torch.no_grad()
    def update(self, hidden_states):
        """
        hidden_states: [B, T, D]
        """
        if hidden_states is None: return
        
        # 1. Flatten to [N, D] where N = B*T
        B, T, D = hidden_states.shape
        flat = hidden_states.reshape(-1, D)
        centered = flat - flat.mean(dim=0)
        
        # 2. Compute SVD to get principal directions (V.T)
        # Using linalg.svd for full V matrix
        _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
        top_directions = Vh[:self.top_k].detach().cpu().numpy() # [K, D]
        
        # 3. Track Persistence (Cosine similarity with previous directions)
        if self.history:
            prev = self.history[-1]
            # Average cosine similarity of the top direction
            sim = np.abs(np.dot(top_directions[0], prev[0]))
            self.persistence.append(sim)
        
        self.history.append(top_directions)
        return self.persistence[-1] if self.persistence else 1.0

    def get_stability(self):
        if not self.persistence: return 0.0
        return np.mean(self.persistence[-3:]) # Last 3 checks


class SemanticsEngine:
    """
    v6.6-G Level 5: Mechanistic Instrument.
    Analyzes the "Combinatorial Gauss-Bonnet" invariant of the stitch graph.
    Sum(Angle Deficit) = 2 * PI * Euler_Characteristic(G).
    """
    def __init__(self, vocab):
        self.vocab = vocab
        self.inc_id = vocab.get("inc", -1)
        self.dec_id = vocab.get("dec", -1)
        self.history = [] 

    def compute_flux(self, tokens_batch):
        """
        Calculates the discrete curvature flux: K = Sum(inc) - Sum(dec).
        For an Amigurumi sphere (chi=2), K_target is typically 12 
        (representing the 4*PI deficit shift required to close a discrete MR6 manifold).
        """
        if self.inc_id == -1 or self.dec_id == -1: return 0.0
        
        inc_mask = (tokens_batch == self.inc_id).float()
        dec_mask = (tokens_batch == self.dec_id).float()
        
        # Combinatorial flux: net growth nodes
        net_flux = (inc_mask.sum(dim=1) - dec_mask.sum(dim=1)).mean().item()
        self.history.append(net_flux)
        return net_flux

    def compute_soft_flux(self, logits_batch):
        """
        Differentiable combinatorial flux using probability-weighted angle deficits.
        """
        if self.inc_id == -1 or self.dec_id == -1: return torch.tensor(0.0)
        
        probs = torch.softmax(logits_batch, dim=-1)
        inc_probs = probs[:, :, self.inc_id]
        dec_probs = probs[:, :, self.dec_id]
        
        # Soft Combinatorial Curvature
        soft_flux = (inc_probs.sum(dim=1) - dec_probs.sum(dim=1)).mean()
        return soft_flux

    def get_violation(self, target_flux=12.0):
        """
        Measures violation of the Combinatorial Gauss-Bonnet expectation.
        V = |K - 2 * PI * chi(Sphere)| normalized to token units.
        """
        if not self.history: return 0.0
        return abs(self.history[-1] - target_flux)


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

    def update(self, score, epoch, threshold=0.01):
        self.history.append(score)
        if len(self.history) < 3: # Need at least 3 for acceleration
            return False

        vel = self.history[-1] - self.history[-2]
        prev_vel = self.history[-2] - self.history[-3]
        accel = vel - prev_vel
        
        # v6.6-F Grounding: Emergence requires crossing noise floor AND positive acceleration (crystallization)
        if vel > threshold and accel > 0:
            if self.peak_epoch is None or vel > self.best_velocity:
                self.best_velocity = vel
                self.peak_epoch = epoch
                print(f"📈 NEW EMERGENCE PEAK: Velocity {vel:.4f}, Accel {accel:.4f} at Epoch {epoch}")

        if self.peak_epoch is not None:
            age = epoch - self.peak_epoch
            if 3 <= age <= 5:
                return True
        return False
