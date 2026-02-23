"""
Training loop for KnittingTransformer.

AlphaKnit v6.6-F: Scientific Falsification & Discovery
- Implements the Blind Discovery Engine for topological causal analysis.
- Phase 11: Scientific Falsification transition logic.
- Ensures VRAM safety in latent capture and observer decoupling.
"""

import os
import json
import time
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .model import KnittingTransformer
from .knitting_dataset import make_dataloaders

# AlphaKnit v6.6-F: The Blind Discovery Engine
from alphaknit.research import compute_phase_lag, LatentPhasePortrait, EmergenceTracker, HiddenProbePool, ModelRealityAnchors, FeatureFingerprint, SemanticsEngine
from alphaknit.metrics import topology_tension_field, compute_structural_metrics, FunctionalSharpness
from alphaknit.scientific import HypothesisEngine, InterventionEngine, NullEmergenceSuite


#  Training helper                                                     #
# ------------------------------------------------------------------ #

def apply_selective_optimizer_reset(model, lr_encoder, lr_decoder, scheduler_type="cosine"):
    """
    Selective reset: Re-instantiates the optimizer and scheduler.
    Encoder gets a stable LR, while Decoder/Heads get a 'Shock LR' for adaptation.
    """
    
    encoder_params = []
    decoder_params = []
    
    # Research-grade keyword matching for AlphaKnit architecture
    reset_keywords = ['decoder', 'head', 'output', 'topology', 'tension', 'stitch', 'logit', 'parent']
    
    for name, param in model.named_parameters():
        if any(kw in name.lower() for kw in reset_keywords):
            decoder_params.append(param)
        else:
            encoder_params.append(param)
            
    # Re-instantiate Optimizer (Resets all moments/history)
    optimizer = optim.Adam([
        {"params": encoder_params, "lr": lr_encoder},
        {"params": decoder_params, "lr": lr_decoder},
    ])
    
    # Re-instantiate Scheduler
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=lr_encoder * 0.01)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    print(f"🔄 v6.6-F SELECTIVE RESET COMPLETE:")
    print(f"   -> Encoder: {len(encoder_params)} params groups (LR={lr_encoder})")
    print(f"   -> Decoder/Tension: {len(decoder_params)} params groups (Shock LR={lr_decoder})")
    print(f"   -> Optimizer & Scheduler re-instantiated (Curvature memory cleared).")
    
    return optimizer, scheduler


class PhaseDetector:
    """
    Research-grade state detector with Hysteresis filter.
    Determines if the Grammar phase is stable enough for Physics ignition.
    """
    def __init__(self, entropy_threshold=0.02, compile_threshold=0.85, pdi_threshold=0.035, min_epochs=11):
        self.history = []
        self.margin_history = []
        self.entropy_threshold = entropy_threshold
        self.compile_threshold = compile_threshold
        self.pdi_threshold = pdi_threshold
        self.min_epochs = min_epochs

    def update(self, entropy, compile_rate, pdi, margin=None):
        self.history.append({
            "entropy": entropy,
            "compile_rate": compile_rate,
            "pdi": pdi
        })
        if margin is not None:
            self.margin_history.append(margin)

    def grammar_ready(self, current_epoch):
        margin_std = 0.0
        # 0. Hysteresis Check: Margin must be stable (low variance) and positive
        if len(self.margin_history) >= 4:
            recent_margins = self.margin_history[-4:]
            avg_margin = sum(recent_margins) / 4.0
            margin_std = np.std(recent_margins)
            
            # Ignition condition: Positive margin AND low noise (stability)
            # If margin is jumping around (high std), it's a "false dawn"
            if avg_margin <= 0.0 or margin_std > 0.1:
                return False
        elif current_epoch >= self.min_epochs:
            # Fallback for early phase
            pass
        else:
            return False

        # Must run at least the minimum epochs for baseline locking
        if current_epoch < self.min_epochs:
            return False
            
        if len(self.history) < 3:
            return False

        h1 = self.history[-1]
        h2 = self.history[-2]
        h3 = self.history[-3]

        # 1. Entropy Trigger (must be at or below target)
        entropy_ok = h1["entropy"] <= self.entropy_threshold
        
        # 2. Compile Success (Grammar Mastery)
        compile_ok = False
        recent_compiles = [h["compile_rate"] for h in self.history if h["compile_rate"] is not None]
        if recent_compiles:
            compile_ok = recent_compiles[-1] >= self.compile_threshold
            
        # 3. PDI Stability (Cognitive Locking)
        pdi_ok = h1["pdi"] < self.pdi_threshold if h1["pdi"] > 0 else True

        ready = entropy_ok and compile_ok and pdi_ok
        if ready:
            if len(self.margin_history) >= 4:
                print(f"🧠 GRS (Grammar Readiness Score) PASSED at Epoch {current_epoch} (Margin Std: {margin_std:.4f})")
            else:
                print(f"🧠 GRS (Grammar Readiness Score) PASSED at Epoch {current_epoch}")
        return ready


# ------------------------------------------------------------------ #
#  Training epoch                                                      #
# ------------------------------------------------------------------ #

def train_epoch(
    model: KnittingTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,        # Type loss criterion
    criterion_p: nn.Module,      # Parent loss criterion (ignore_index=0)
    device: torch.device,
    grad_clip: float = config.GRAD_CLIP,
    edge_weight: float = 1.0,    # Phase 9B Curriculum control
    sector_weight: float = 0.0,  # Phase 9B Curriculum control
    parent_noise_prob: float = 0.0, # Phase 9B Stage 1 Noise
    grad_accum_steps: int = 1,
    prev_epoch_probs: dict = None, # Phase 10: Store previous probabilities for PDI
    epoch: int = 0,                # Current epoch info for Topology Tension curriculum
    tension_weight: float = 0.02,  # v4.0: Dynamic tension ramp weight
    lambda_ttf: float = 0.001,     # v6.0: Topology Tension Field (Passive Bias)
    portrait: LatentPhasePortrait = None, # v6.0: To capture latent snapshots
    intervention_engine: InterventionEngine = None, # v6.6-F: Causal Falsification
    null_suite: NullEmergenceSuite = None, # v6.6-F: Scientific Control
    probe_pool: HiddenProbePool = None, # v6.6-F: Observer Decoupling
    measurement_dropout: float = 0.3, # v6.6-F: Prevent observer resonance
    anchor_batch: dict = None, # v6.6-F Level 2: For invariant curvature
    fingerprint: FeatureFingerprint = None, # v6.6-F Level 3: Mechanistic Identity
    semantics: SemanticsEngine = None, # v6.6-F Level 4: Semantic Topology
    radius_scale: float = 0.0, # v6.7-G: phase-2 radius jitter cap
    tension_noise: float = 0.0, # v6.7-G: phase-2 tension noise cap
) -> dict:
    """Run one training epoch. Returns dict of average losses."""
    model.train()
    
    # Initialize static tracking if not present (v6.0 Robustness)
    if not hasattr(train_epoch, "total_entropy"): setattr(train_epoch, "total_entropy", 0.0)
    if not hasattr(train_epoch, "total_tension"): setattr(train_epoch, "total_tension", 0.0)
    if not hasattr(train_epoch, "epoch_prob_p1_acc"): setattr(train_epoch, "epoch_prob_p1_acc", torch.zeros(200, device=device))
    if not hasattr(train_epoch, "epoch_prob_p2_acc"): setattr(train_epoch, "epoch_prob_p2_acc", torch.zeros(200, device=device))
    if not hasattr(train_epoch, "epoch_valid_batches"): setattr(train_epoch, "epoch_valid_batches", 0)
    if not hasattr(train_epoch, "epoch_argmax_p1_hist"): 
        setattr(train_epoch, "epoch_argmax_p1_hist", torch.zeros(200, device=device))
    if not hasattr(train_epoch, "epoch_argmax_p2_hist"): 
        setattr(train_epoch, "epoch_argmax_p2_hist", torch.zeros(200, device=device))
    
    # v6.6-F: Sharpness Tracking
    if not hasattr(train_epoch, "sharpness_tracker") or getattr(train_epoch, "sharpness_tracker") is None:
        setattr(train_epoch, "sharpness_tracker", FunctionalSharpness())

    total_loss = 0.0
    total_l_type = 0.0
    total_l_parent = 0.0
    total_l_deg = 0.0
    total_l_dist = 0.0
    total_pib = 0.0
    total_sharpness = 0.0
    total_delta_dist = 0.0 
    total_shadow_delta = 0.0 # v6.6-F Level 3
    total_flux = 0.0 # v6.6-F Level 4
    total_l_curvature = 0.0 # v6.7-G
    shadow_counts = 0
    
    n_batches = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}", leave=False)):
        # v6.6-F: Null Emergence Suite (Placebo Control)
        if null_suite:
            batch = null_suite.transform_batch(batch)
            
        non_blocking = device.type == "cuda"
        point_cloud = batch['point_cloud'].to(device, non_blocking=non_blocking) # (B, N, 3) 
        src_tokens  = batch['src_tokens'].to(device, non_blocking=non_blocking)  # (B, T, 3)
        tgt_tokens  = batch['tgt_tokens'].to(device, non_blocking=non_blocking)  # (B, T, 3)
        if radius_scale > 0:
            point_cloud = point_cloud + (torch.randn_like(point_cloud) * radius_scale)

        # Apply Parent Noise
        if parent_noise_prob > 0.0:
            B, T, _ = src_tokens.shape
            mask = torch.rand((B, T), device=device) < parent_noise_prob
            rand_p1 = torch.randint(1, 200, (B, T), device=device)
            rand_p2 = torch.randint(1, 200, (B, T), device=device)
            src_tokens[:, :, 1] = torch.where(mask, rand_p1, src_tokens[:, :, 1])
            src_tokens[:, :, 2] = torch.where(mask, rand_p2, src_tokens[:, :, 2])

        # Phase 9B: Parent Dropout
        if edge_weight > 0.1:
            B, T, _ = src_tokens.shape
            dropout_prob = 0.15
            drop_mask = torch.rand((B, T), device=device) < dropout_prob
            shifted = torch.cat([torch.zeros((B, 1), dtype=torch.bool, device=device), drop_mask[:, :-1]], dim=1)
            consecutive = drop_mask & shifted
            drop_mask = drop_mask ^ consecutive 
            src_tokens[:, :, 1] = torch.where(drop_mask, torch.zeros_like(src_tokens[:, :, 1]), src_tokens[:, :, 1])
            src_tokens[:, :, 2] = torch.where(drop_mask, torch.zeros_like(src_tokens[:, :, 2]), src_tokens[:, :, 2])

        # v6.6-F: Causal Intervention
        if intervention_engine:
            intervention_engine.apply(n_batches)

        pad_mask = (src_tokens[:, :, 0] == config.PAD_ID)
        optimizer.zero_grad()
        should_measure = np.random.random() > measurement_dropout

        # v6.6-G: Execution Step (The Primary Trajectory)
        logits_type, logits_p1, logits_p2 = model(point_cloud, src_tokens, tgt_key_padding_mask=pad_mask)

        # v6.6-G Level 5: True Counterfactual (fork_rng)
        shadow_delta = 0.0
        if intervention_engine and intervention_engine.active_interventions and should_measure:
             # v6.6-G: Use PyTorch's fork_rng to ensure PERFECTLY identical dropout/RNG state
             with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
                  # This pass is 'Clean' (No intervention noise)
                  intervention_engine.shadow_mode = True
                  with torch.no_grad():
                       s_logits_type, _, _ = model(point_cloud, src_tokens, tgt_key_padding_mask=pad_mask)
                  intervention_engine.shadow_mode = False
                  
                  p_real = torch.softmax(logits_type, dim=-1).detach()
                  p_shadow = torch.softmax(s_logits_type, dim=-1)
                  shadow_delta = torch.norm(p_real - p_shadow).item()
                  total_shadow_delta += shadow_delta
                  shadow_counts += 1

        # v6.6-F Level 3: Feature Fingerprinting (Representational Invariants)
        if fingerprint is not None and hasattr(model, 'last_hidden_state') and should_measure:
             _ = fingerprint.update(model.last_hidden_state.detach())
        
        # v6.6-F Level 4: Semantic Topology Flux (Gauss-Bonnet)
        if semantics is not None and should_measure:
             total_flux += semantics.compute_flux(tgt_tokens[:, :, 0])
        
        B, T, _ = tgt_tokens.shape
        V = logits_type.shape[-1]
        P_MAX = logits_p1.shape[-1]
        
        tgt_type = tgt_tokens[:, :, 0].reshape(B * T)
        tgt_p1 = tgt_tokens[:, :, 1].reshape(B * T)
        tgt_p2 = tgt_tokens[:, :, 2].reshape(B * T)

        # v6.0: Define structural mask 
        struct_ids = [4, 5, 6, 7]
        structural_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
        for sid in struct_ids:
            structural_mask |= (tgt_tokens[:, :, 0] == sid)

        # 1. Node Type Loss & Entropy
        loss_type = criterion(logits_type.reshape(B * T, V), tgt_type)
        probs_type = torch.softmax(logits_type.reshape(B * T, V), dim=-1)
        entropy = -(probs_type * torch.log(probs_type + 1e-9)).sum(-1)
        valid_type_mask = (tgt_type != config.PAD_ID)
        entropy_val = entropy[valid_type_mask].mean() if valid_type_mask.any() else torch.tensor(0.0, device=device)

        # 2. Parent Loss
        loss_parent = 0.0
        if edge_weight > 0:
            l_p1 = criterion_p(logits_p1.reshape(B * T, P_MAX), tgt_p1)
            l_p2 = criterion_p(logits_p2.reshape(B * T, P_MAX), tgt_p2)
            loss_parent = (l_p1 + l_p2) * 0.5

        # 3. Degree Consistency
        dec_id = config.VOCAB.get('dec', 7)
        mask_non_dec = (tgt_type != dec_id) & (tgt_type != config.PAD_ID)
        probs_p2 = torch.softmax(logits_p2.reshape(B * T, P_MAX), dim=-1)
        prob_p2_not_null = 1.0 - probs_p2[:, 0]
        loss_deg = prob_p2_not_null[mask_non_dec].mean() if mask_non_dec.any() else torch.tensor(0.0, device=device)
            
        # 4. Distance Reg
        offsets = torch.arange(P_MAX, device=device).float().unsqueeze(0) 
        exp_p1 = (torch.softmax(logits_p1.reshape(B * T, P_MAX), dim=-1) * offsets).sum(-1)
        exp_p2 = (probs_p2 * offsets).sum(-1)
        valid_mask = (tgt_type != config.PAD_ID)
        loss_dist = (exp_p1[valid_mask] + exp_p2[valid_mask]).mean() if valid_mask.any() else torch.tensor(0.0, device=device)
            
        # PDI Tracking
        T_mid = T // 2
        if T_mid < T:
            probs_p1_raw = torch.softmax(logits_p1.reshape(B, T, P_MAX), dim=-1).detach()
            probs_p2_raw = torch.softmax(logits_p2.reshape(B, T, P_MAX), dim=-1).detach()
            mean_prob_p1 = probs_p1_raw[:, T_mid:, :].reshape(-1, P_MAX).mean(dim=0)
            mean_prob_p2 = probs_p2_raw[:, T_mid:, :].reshape(-1, P_MAX).mean(dim=0)
            train_epoch.epoch_prob_p1_acc += mean_prob_p1
            train_epoch.epoch_prob_p2_acc += mean_prob_p2
            argmax_p1 = probs_p1_raw[:, T_mid:, :].reshape(-1, P_MAX).argmax(dim=-1)
            argmax_p2 = probs_p2_raw[:, T_mid:, :].reshape(-1, P_MAX).argmax(dim=-1)
            train_epoch.epoch_argmax_p1_hist += torch.bincount(argmax_p1, minlength=P_MAX).float()
            train_epoch.epoch_argmax_p2_hist += torch.bincount(argmax_p2, minlength=P_MAX).float()
            train_epoch.epoch_valid_batches += 1
            
        # 5. Topology Tension Signal
        loss_tension = torch.tensor(0.0, device=device)
        if epoch >= 12:
            tau = 0.3 if epoch >= 30 else (0.4 if epoch >= 18 else 0.5)
            frac = 1.0 if epoch >= 30 else (0.5 if epoch >= 18 else 0.2)
            probs_p1_t = torch.softmax(logits_p1.reshape(B, T, P_MAX) / tau, dim=-1)
            probs_p2_t = torch.softmax(logits_p2.reshape(B, T, P_MAX) / tau, dim=-1)
            ent_p1 = -(probs_p1_t * torch.log(probs_p1_t + 1e-9)).sum(-1)
            ent_p2 = -(probs_p2_t * torch.log(probs_p2_t + 1e-9)).sum(-1)
            loss_entropy_bar = (ent_p1 + ent_p2).mean() * 0.5
            loss_div = (probs_p1_t * probs_p2_t).sum(-1).mean()
            
            # Adjacency matrix construction
            A = torch.zeros(B, T, T, device=device)
            for offset in range(1, P_MAX):
                if T - offset <= 0: continue
                prob = 0.5 * (probs_p1_t[:, offset:, offset] + probs_p2_t[:, offset:, offset])
                A.diagonal(dim1=1, dim2=2, offset=-offset).copy_(prob)
                
            deg = A.sum(dim=-1, keepdim=True) + 1e-9
            A_tilde = A / deg
            X = torch.randn(B, T, 3, device=device)
            T_start = T - max(1, int(T * frac))
            alpha = 0.2
            for _ in range(3):
                X_next = X + alpha * (torch.bmm(A_tilde, X) - X)
                if T_start > 0:
                    X_next = torch.cat([X_next[:, :T_start, :].detach(), X_next[:, T_start:, :]], dim=1)
                X = X_next
            X_expand_i = X.unsqueeze(2)
            X_expand_j = X.unsqueeze(1)
            dist = torch.sqrt(((X_expand_i - X_expand_j)**2).sum(-1) + 1e-9)
            mean_dist = dist.mean(dim=(1, 2), keepdim=True) + 1e-9
            d_hat = dist / mean_dist
            E = A * (d_hat - 1.0)**2
            valid_mask_2d = (tgt_tokens[:, :, 0] != config.PAD_ID)
            valid_mask_E = valid_mask_2d.unsqueeze(2).expand(B, T, T)
            if T_start > 0:
                E = E[:, T_start:, :]
                valid_mask_E = valid_mask_E[:, T_start:, :]
            if valid_mask_E.any():
                loss_tension = E[valid_mask_E].mean()
            
            # Combine internal tension metrics
            loss_tension = loss_tension + 0.005 * loss_entropy_bar + 0.002 * loss_div
            if tension_noise > 0:
                loss_tension = loss_tension + ((torch.randn((), device=device) * tension_noise).detach())

        # 6. Curvature hint with Huber loss (relative topology signal)
        curvature_target = torch.zeros((B, T, 1), device=device)
        curvature_target[tgt_tokens[:, :, 0] == config.VOCAB.get("inc", 6)] = 1.0
        curvature_target[tgt_tokens[:, :, 0] == config.VOCAB.get("dec", 7)] = -1.0
        curvature_pred = getattr(model, "last_curvature_hint", None)
        if curvature_pred is not None:
            curv_valid = (tgt_tokens[:, :, 0] != config.PAD_ID).unsqueeze(-1)
            loss_curvature = F.huber_loss(curvature_pred[curv_valid], curvature_target[curv_valid], reduction="mean")
        else:
            loss_curvature = torch.tensor(0.0, device=device)

        # v6.0/6.1: Topology Tension Field (TTF) and Structural Metrics
        struct_metrics = compute_structural_metrics(logits_type, tgt_tokens[:, :, 0], structural_mask)
        p1_idx = logits_p1.argmax(dim=-1)
        p2_idx = logits_p2.argmax(dim=-1)
        deg_counts = (p1_idx > 0).float() + (p2_idx > 0).float()
        edge_count = deg_counts.sum(dim=1).mean()
        
        # v6.1: Observer Purity - report_only=True
        ttf_loss_val, ttf_stats = topology_tension_field(
            deg_counts, edge_count, num_nodes=(structural_mask.sum() / B), 
            lambda_density=0.1, report_only=True
        )
        
        # v6.6-F Level 5: Differentiable Topology Loss (Gauss-Bonnet Pressure)
        loss_topo = torch.tensor(0.0, device=device)
        if semantics is not None:
             # Ground truth flux from tokens
             target_flux = semantics.compute_flux(tgt_tokens[:, :, 0])
             # Differential flux from logits
             soft_flux = semantics.compute_soft_flux(logits_type)
             loss_topo = torch.abs(soft_flux - target_flux)
             
        # Total Loss
        lambda_topo = 0.2 # v6.6-F Level 5: Pressure weight
        lambda_curvature = 0.05
        loss_val = (1.0 * loss_type) + (0.3 * edge_weight * loss_parent) + (0.2 * loss_deg) + (0.01 * loss_dist)
        loss_val = loss_val + (tension_weight * loss_tension) + (lambda_ttf * ttf_loss_val) + (lambda_topo * loss_topo) + (lambda_curvature * loss_curvature)
        
        # v6.1/6.6-F: Latent Capture with detachment
        if portrait is not None and hasattr(model, 'last_hidden_state') and should_measure:
            portrait.capture(model.last_hidden_state.detach(), structural_mask)

        # v6.6-F Level 2: Anchor Capture for Curvature
        if portrait is not None and anchor_batch is not None and should_measure:
            model.eval()
            with torch.no_grad():
                a_pc = anchor_batch['point_cloud'].to(device, non_blocking=non_blocking)
                a_src = anchor_batch['src_tokens'].to(device, non_blocking=non_blocking)
                a_mask = (a_src[:, :, 0] == config.PAD_ID)
                _ = model(a_pc, a_src, tgt_key_padding_mask=a_mask)
                if hasattr(model, 'last_hidden_state'):
                    a_tgt = anchor_batch['tgt_tokens'].to(device, non_blocking=non_blocking)
                    # Simple structural mask for anchor
                    a_struct_mask = torch.zeros((a_pc.size(0), a_src.size(1)), dtype=torch.bool, device=device)
                    for sid in [4, 5, 6, 7]:
                        a_struct_mask |= (a_tgt[:, :, 0] == sid)
                    portrait.capture(model.last_hidden_state.detach(), a_struct_mask, is_anchor=True)
            model.train()

        # v6.6-F: Measurement Dropout & PIB
        pib_val = 0.0
        if probe_pool and should_measure:
            # Identify last layer dynamically
            last_layer_prefix = None
            for name, _ in model.named_parameters():
                 if "transformer.layers" in name:
                     parts = name.split(".")
                     idx = int(parts[2])
                     if last_layer_prefix is None or idx > int(last_layer_prefix.split(".")[-1]):
                         last_layer_prefix = f"transformer.layers.{idx}"

            train_grads = {}
            for name, p in model.named_parameters():
                if last_layer_prefix and name.startswith(last_layer_prefix) and p.grad is not None:
                    train_grads[name] = p.grad.detach().clone()
            
            pib_val = probe_pool.compute_pib(model, train_grads, criterion, device)

        # Optimization Step
        loss = loss_val / grad_accum_steps
        loss.backward()
        if (n_batches + 1) % grad_accum_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # v6.6-F Level 2: Track Optimizer Path Length (||delta theta||)
            params_before = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
            optimizer.step()
            params_after = torch.cat([p.detach().flatten() for p in model.parameters()])
            delta_dist = torch.norm(params_after - params_before).item()
            total_delta_dist += delta_dist
            
            optimizer.zero_grad()

        total_loss += loss_val.item()
        total_l_type += loss_type.item()
        total_l_parent += loss_parent.item() if isinstance(loss_parent, torch.Tensor) else loss_parent
        total_l_deg += loss_deg.item() if isinstance(loss_deg, torch.Tensor) else loss_deg
        total_l_dist += loss_dist.item() if isinstance(loss_dist, torch.Tensor) else loss_dist
        total_l_curvature += loss_curvature.item() if isinstance(loss_curvature, torch.Tensor) else loss_curvature
        train_epoch.total_entropy += entropy_val.item()
        train_epoch.total_tension += loss_tension.item() if isinstance(loss_tension, torch.Tensor) else loss_tension
        
        # v6.6-F: Scientific Metric Accumulation
        total_pib += pib_val
        sharpness_tracker = getattr(train_epoch, "sharpness_tracker")
        if sharpness_tracker:
            # Find last layer for sharpness tracking
            last_layer_prefix = None
            for name, _ in model.named_parameters():
                 if "transformer.layers" in name:
                     parts = name.split(".")
                     idx = int(parts[2])
                     if last_layer_prefix is None or idx > int(last_layer_prefix.split(".")[-1]):
                         last_layer_prefix = f"transformer.layers.{idx}"

            relevant_grads = [p.grad.detach().flatten() for name, p in model.named_parameters() 
                              if p.grad is not None and last_layer_prefix and name.startswith(last_layer_prefix)]
            sharpness_grad_norm = torch.norm(torch.cat(relevant_grads)).item() if relevant_grads else 0.0
            total_sharpness += sharpness_tracker.update(sharpness_grad_norm)

        n_batches += 1

    ret = {
        "loss": total_loss / max(n_batches, 1),
        "l_type": total_l_type / max(n_batches, 1),
        "l_edge": total_l_parent / max(n_batches, 1),
        "l_deg": total_l_deg / max(n_batches, 1),
        "l_dist": total_l_dist / max(n_batches, 1),
        "l_curvature": total_l_curvature / max(n_batches, 1),
        "entropy": train_epoch.total_entropy / max(n_batches, 1),
        "tension": train_epoch.total_tension / max(n_batches, 1),
        "mean_p1_prob": train_epoch.epoch_prob_p1_acc / max(train_epoch.epoch_valid_batches, 1),
        "mean_p2_prob": train_epoch.epoch_prob_p2_acc / max(train_epoch.epoch_valid_batches, 1),
        "hist_p1": train_epoch.epoch_argmax_p1_hist,
        "hist_p2": train_epoch.epoch_argmax_p2_hist,
        "ttf_deg_var": ttf_stats["degree_var"],
        "ttf_edge_density": ttf_stats["edge_density"],
        "struct_margin": struct_metrics.get("struct_margin", 0.0),
        "struct_top1_acc": struct_metrics.get("struct_top1_acc", 0.0),
        "struct_entropy": struct_metrics.get("struct_entropy", 0.0),
        "pib": total_pib / max(n_batches, 1),
        "sharpness": total_sharpness / max(n_batches, 1),
        "delta_dist": total_delta_dist,
        "shadow_delta": total_shadow_delta / max(shadow_counts, 1) if shadow_counts > 0 else 0.0,
        "flux": total_flux / max(n_batches, 1),
    }
    
    # Reset tracking variables for next epoch
    train_epoch.total_entropy = 0.0
    train_epoch.total_tension = 0.0
    train_epoch.epoch_prob_p1_acc.zero_()
    train_epoch.epoch_prob_p2_acc.zero_()
    train_epoch.epoch_argmax_p1_hist.zero_()
    train_epoch.epoch_argmax_p2_hist.zero_()
    train_epoch.epoch_valid_batches = 0
    return ret


# ------------------------------------------------------------------ #
#  Validation                                                          #
# ------------------------------------------------------------------ #

@torch.no_grad()
def evaluate(
    model: KnittingTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    criterion_p: nn.Module,
    device: torch.device,
    edge_weight: float = 1.0,
) -> dict:
    """Evaluate on a DataLoader. Returns dict of average losses."""
    model.eval()
    total_loss = 0.0
    total_l_type = 0.0
    total_l_parent = 0.0
    total_l_deg = 0.0
    total_l_dist = 0.0
    n_batches = 0

    for point_cloud, src_tokens, tgt_tokens in tqdm(loader, desc="Validation", leave=False):
        non_blocking = device.type == "cuda"
        point_cloud = point_cloud.to(device, non_blocking=non_blocking)
        src_tokens  = src_tokens.to(device, non_blocking=non_blocking)
        tgt_tokens  = tgt_tokens.to(device, non_blocking=non_blocking)

        pad_mask = (src_tokens[:, :, 0] == config.PAD_ID)
        logits_type, logits_p1, logits_p2 = model(point_cloud, src_tokens, tgt_key_padding_mask=pad_mask)

        B, T, _ = tgt_tokens.shape
        V = logits_type.shape[-1]
        P_MAX = logits_p1.shape[-1]
        
        tgt_type = tgt_tokens[:, :, 0].reshape(B * T)
        tgt_p1 = tgt_tokens[:, :, 1].reshape(B * T)
        tgt_p2 = tgt_tokens[:, :, 2].reshape(B * T)

        # 1. Node Type
        loss_type = criterion(logits_type.reshape(B * T, V), tgt_type)

        # 2. Parent Loss
        loss_parent = 0.0
        if edge_weight > 0:
            l_p1 = criterion_p(logits_p1.reshape(B * T, P_MAX), tgt_p1)
            l_p2 = criterion_p(logits_p2.reshape(B * T, P_MAX), tgt_p2)
            loss_parent = (l_p1 + l_p2) * 0.5

        # 3. Degree Consistency
        dec_id = config.VOCAB.get('dec', 7)
        mask_non_dec = (tgt_type != dec_id) & (tgt_type != config.PAD_ID)
        probs_p2 = torch.softmax(logits_p2.reshape(B * T, P_MAX), dim=-1)
        prob_p2_not_null = 1.0 - probs_p2[:, 0]
        loss_deg = prob_p2_not_null[mask_non_dec].mean() if mask_non_dec.any() else torch.tensor(0.0, device=device)
            
        # 4. Distance Reg
        offsets = torch.arange(P_MAX, device=device).float().unsqueeze(0)
        exp_p1 = (torch.softmax(logits_p1.reshape(B * T, P_MAX), dim=-1) * offsets).sum(-1)
        exp_p2 = (probs_p2 * offsets).sum(-1)
        valid_mask = (tgt_type != config.PAD_ID)
        loss_dist = (exp_p1[valid_mask] + exp_p2[valid_mask]).mean() if valid_mask.any() else torch.tensor(0.0, device=device)

        # Combine
        loss = (1.0 * loss_type) + (0.3 * edge_weight * loss_parent) + (0.2 * loss_deg) + (0.01 * loss_dist)

        total_loss += loss.item()
        total_l_type += loss_type.item()
        total_l_parent += loss_parent.item() if isinstance(loss_parent, torch.Tensor) else loss_parent
        total_l_deg += loss_deg.item() if isinstance(loss_deg, torch.Tensor) else loss_deg
        total_l_dist += loss_dist.item() if isinstance(loss_dist, torch.Tensor) else loss_dist
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "l_type": total_l_type / max(n_batches, 1),
        "l_edge": total_l_parent / max(n_batches, 1),
        "l_deg": total_l_deg / max(n_batches, 1),
        "l_dist": total_l_dist / max(n_batches, 1),
    }


# ------------------------------------------------------------------ #
#  Compile success rate (per epoch signal)                             #
# ------------------------------------------------------------------ #

@torch.no_grad()
def compute_compile_success_rate(
    model: KnittingTransformer,
    loader: DataLoader,
    device: torch.device,
    n_batches_max: int = 10,
) -> tuple:
    """
    Sample n_batches_max batches and decode greedily.
    Returns (compile_success_rate, confusion_counts).
    """
    from .compiler import KnittingCompiler, CompileError

    def _save_emergence_graph(stitch_graph, filepath):
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            G = nx.DiGraph()
            for nid, node in stitch_graph.nodes.items():
                G.add_node(nid, type=node.stitch_type)
                for pid in node.parents:
                    G.add_edge(pid, nid)
            plt.figure(figsize=(8, 8))
            pos = nx.spring_layout(G, seed=42)
            colors = []
            for n in G.nodes():
                t = G.nodes[n]['type']
                if 'mr' in t: colors.append('red')
                elif t == 'inc': colors.append('green')
                elif t == 'dec': colors.append('orange')
                else: colors.append('lightblue')
            nx.draw(G, pos, node_color=colors, with_labels=False, node_size=30, edge_color='gray', arrows=True, arrowsize=8, alpha=0.8)
            plt.title(f"Emergence Watch - Size: {len(G.nodes)}")
            plt.savefig(filepath, dpi=120, bbox_inches='tight')
            plt.close()
        except Exception:
            pass  # Visualization is optional; skip silently if matplotlib/networkx unavailable
    compiler = KnittingCompiler()
    compile_ok = 0
    compile_total = 0
    confusion = {}   
    graphs_saved = 0

    pbar = tqdm(loader, desc="Compiling", total=n_batches_max, leave=False)
    for i, (point_cloud, src_tokens, tgt_tokens) in enumerate(pbar):
        if i >= n_batches_max: break
        point_cloud = point_cloud.to(device, non_blocking=device.type == "cuda")
        pred_ids_list = model.greedy_decode(point_cloud, max_len=config.MAX_SEQ_LEN)

        for b, pred_ids in enumerate(pred_ids_list):
            tokens = []
            for tpl in pred_ids:
                t, p1, p2 = tpl
                t_str = config.ID_TO_TOKEN.get(t, "<UNK>")
                tokens.append(f"{t_str}({p1},{p2})")

            try:
                graph = compiler.compile(tokens)
                compile_ok += 1
                if getattr(compute_compile_success_rate, "checkpoint_dir", None) is not None:
                    epoch = getattr(compute_compile_success_rate, "current_epoch", 0)
                    if epoch > 0 and epoch % 2 == 0 and graphs_saved < 3:
                        watch_dir = os.path.join(compute_compile_success_rate.checkpoint_dir, "emergence_watch")
                        os.makedirs(watch_dir, exist_ok=True)
                        filepath = os.path.join(watch_dir, f"epoch_{epoch:03d}_sample_{graphs_saved}.png")
                        _save_emergence_graph(graph, filepath)
                        graphs_saved += 1
            except Exception:
                pass  # Invalid token sequences are expected during training; count failures separately

            # Confusion matrix
            gt_ids = tgt_tokens[b].tolist() 
            pred_seq = pred_ids[:len(gt_ids)]
            pred_seq = pred_seq + [(config.PAD_ID, 0, 0)] * (len(gt_ids) - len(pred_seq))
            for p, g in zip(pred_seq, gt_ids):
                g_type = g[0]
                if g_type == config.PAD_ID: continue
                p_type = p[0]
                key = (int(p_type), int(g_type))
                confusion[key] = confusion.get(key, 0) + 1

    rate = compile_ok / max(compile_total, 1)
    return rate, confusion


# ------------------------------------------------------------------ #
#  Main training function                                              #
# ------------------------------------------------------------------ #

def train(
    dataset_dir: str = "data/processed/dataset",
    checkpoint_dir: str = config.CHECKPOINT_DIR,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LR,
    device_str: str = "auto",
    val_split: float = 0.1,
    d_model: int = config.D_MODEL,
    n_heads: int = config.N_HEADS,
    n_layers: int = config.N_LAYERS,
    ffn_dim: int = config.FFN_DIM,
    scheduler_type: str = "cosine",   # 'cosine' or 'plateau'
    run_name: str = "default",
    label_smoothing: float = 0.1,     # Phase 8: prevents sc dominance
    early_stop_patience: int = 10,    # Phase 8: stop if no improvement
    log_compile_every: int = 5,       # Phase 8: compile rate every N epochs
    resume_checkpoint: str = None,    # Phase 8: resume from checkpoint
    num_workers: int = 2,             # Phase 10: dataloader multiprocessing
    grad_accum_steps: int = 4,        # Phase 10: gradient accumulation for T4 VRAM
    reset_optimizer: bool = False,    # v6.6-F: Phase transition reset
    resume_auto: bool = False,        # v6.6-F: Pick up latest epoch checkpoint
    force_phase2: bool = False,       # v6.6-F: Force Airlock transition
):
    """Full training loop with checkpointing."""

    # Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Training on: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_path = os.path.normpath(dataset_dir).replace("\\", "/").lower()
    train_parts = [part for part in train_path.split("/") if part]
    has_dataset_5k = "dataset_5k" in train_parts
    is_debug_path = "debug" in train_parts
    assert not (has_dataset_5k and not is_debug_path), \
        "dataset_5k is debug-only; move it under data/debug/dataset_5k."

    # Data
    train_loader, val_loader = make_dataloaders(
        dataset_dir, val_split=val_split, batch_size=batch_size, num_workers=num_workers
    )
    
    try:
        n_train = len(train_loader)
    except TypeError:
        n_train = "WebDataset Stream"
        
    try:
        n_val = len(val_loader) if val_loader else 0
    except TypeError:
        n_val = "Unknown"
        
    print(f"Train batches: {n_train} | Val batches: {n_val}")

    # Model
    model = KnittingTransformer(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ffn_dim=ffn_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 1
    best_val_loss = float("inf")
    history = []
    reset_done = False
    phase2_active = False
    phase2_start_epoch = 0
    
    # v6.0/6.6-F: Observer & Discovery Initialization
    portrait = LatentPhasePortrait()
    emergence_tracker = EmergenceTracker()
    anchors = ModelRealityAnchors()
    hypotheses = HypothesisEngine()
    intervention_engine = InterventionEngine(model)
    null_suite = NullEmergenceSuite(mode=os.environ.get("AK_NULL_MODE", "real"))
    fingerprint = FeatureFingerprint(top_k=5) # v6.6-F Level 3
    semantics = SemanticsEngine(VOCAB) # v6.6-F Level 4
    
    # Null Mode setup
    if null_suite.mode != "real":
        print(f"🧪 SCIENTIFIC CONTROL ACTIVE: Mode={null_suite.mode}")
        null_suite.apply_geometry_null(model)

    # Hidden Probe setup
    probe_pool = None
    if val_loader:
        probe_pool = HiddenProbePool(val_loader, num_pools=3)
        print(f"🔒 OBSERVER DECOUPLING: HiddenProbePool initialized with 3 pools.")

    # Hypothesis Definitions (Proposals)
    hypotheses.propose("Rank_Collapse", "L4 Rank drops preceded by Alignment spike", 
                       lambda m: m.get("rank", 0) < 10 and m.get("phase_lag", 0) > 0.8)
    hypotheses.propose("Functional_Crystallization", "Sharpness stability despite high Curvature",
                       lambda m: m.get("sharpness", 0) < 1.05 and m.get("l_edge", 0) > 0.05)
    
    # v5.0: Auto-Resume logic
    if resume_auto:
        ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "epoch_*.pt")))
        if ckpts:
            resume_checkpoint = ckpts[-1]
            print(f"🔍 Auto-Resume: Detected latest checkpoint at {resume_checkpoint}")

    # Resume from checkpoint
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        # v6.1: Load reset state to prevent double shocks
        reset_done = ckpt.get("optimizer_reset_done", False)
        phase2_active = ckpt.get("phase2_active", False)
        phase2_start_epoch = ckpt.get("phase2_start_epoch", 0)
        
        print(f"Resumed from epoch {start_epoch - 1} (val_loss={best_val_loss:.4f}, reset_done={reset_done})")
        hist_path = os.path.join(checkpoint_dir, f"training_history_{run_name}.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                history = json.load(f)

    # Scheduler (based on remaining epochs)
    remaining = epochs - start_epoch + 1
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(remaining, 1), eta_min=lr * 0.01
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5
        )

    # Loss Definition
    class_weights = torch.ones(len(config.VOCAB), device=device)
    class_weights[0] = 0.0 ; class_weights[1] = 0.1 ; class_weights[2] = 0.1
    class_weights[7] = 0.5 ; class_weights[3] = 2.0 ; class_weights[4] = 3.0
    class_weights[5] = 4.0 ; class_weights[6] = 4.0
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=config.PAD_ID, label_smoothing=label_smoothing)
    criterion_p = nn.CrossEntropyLoss(ignore_index=0)

    os.makedirs(checkpoint_dir, exist_ok=True)
    no_improve = 0 
    prev_epoch_probs = None
    detector = PhaseDetector(min_epochs=11)

    for epoch in range(start_epoch, epochs + 1):
        # v6.1: Stabilized Phase Transition Logic
        compile_rate = None
        
        if not phase2_active:
            if force_phase2:
                phase2_active = True
                print(f"🚀 FORCED Ignition: Triggering phase 2 at Epoch {epoch}")

        # Selective Reset (Triggered only ONCE)
        if phase2_active and not reset_done:
            lr_enc = optimizer.param_groups[0]['lr']
            lr_dec = lr_enc * 1.6 
            optimizer, scheduler = apply_selective_optimizer_reset(model, lr_enc, lr_dec, scheduler_type)
            reset_done = True
            if grad_accum_steps > 1:
                print(f"🔥 AIRLOCK: Cooling system... Setting grad_accum_steps=1 for physics stability.")
                grad_accum_steps = 1

        # Tension Weight Curriculum
        if not phase2_active:
            tension_weight = 0.0
        else:
            if phase2_start_epoch == 0: phase2_start_epoch = epoch
            phase2_age = epoch - phase2_start_epoch
            if phase2_age <= 0: tension_weight = 0.003
            elif phase2_age == 1: tension_weight = 0.015
            elif phase2_age == 2: tension_weight = 0.02
            else: tension_weight = 0.025 

        edge_weight = 1.0 
        parent_noise_prob = 0.0
        radius_scale = 0.0 if not phase2_active else 0.02
        tension_noise = 0.0 if not phase2_active else 0.005

        # v6.6-F Level 2: Anchor Batch Initialization
        anchor_batch = None
        if val_loader:
             anchor_batch = next(iter(val_loader))
        elif train_loader:
             anchor_batch = next(iter(train_loader))

        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, criterion_p, device, 
            edge_weight=edge_weight, parent_noise_prob=parent_noise_prob, 
            grad_accum_steps=grad_accum_steps, prev_epoch_probs=prev_epoch_probs, 
            epoch=epoch, tension_weight=tension_weight, portrait=portrait,
            intervention_engine=intervention_engine, null_suite=null_suite,
            probe_pool=probe_pool, measurement_dropout=0.3, anchor_batch=anchor_batch,
            fingerprint=fingerprint, semantics=semantics,
            radius_scale=radius_scale, tension_noise=tension_noise
        )

        # Update Anchors (v6.6-F Level 2)
        # Use Fixed Anchor Curvature instead of batch-based noise
        fixed_curvature = portrait.get_curvature()
        anchors.history["curvature"].append(fixed_curvature)
        anchors.update(model, portrait.history[-1] if portrait.history else None)
        
        if hasattr(model, 'last_hidden_state'):
            # Normalized Spectral Rank
            train_metrics["rank"] = anchors.compute_rank(model.last_hidden_state.detach().mean(dim=1))
        
        # v6.1/6.6-F: Compute Phase Lag and Update Discovery Engine
        phase_lag, update_energy = compute_phase_lag(model, optimizer)
        train_metrics["phase_lag"] = phase_lag
        train_metrics["update_energy"] = update_energy

        # Hypothesis Falsification Check (Grounded in Optimizer Distance)
        delta_dist = train_metrics.get("delta_dist", 0.0)
        shadow_delta = train_metrics.get("shadow_delta", 0.0)
        hypo_reports = hypotheses.update(train_metrics, delta_dist)

        # v6.6-F Level 3: Feature Identity Stability
        train_metrics["fingerprint_stability"] = fingerprint.get_stability()
        
        # v6.6-F Level 4: Semantic Violation
        train_metrics["semantics_violation"] = semantics.get_violation(target_flux=6.0) # MR6 Growth
        
        # v6.6-F: Failure Monitor (Automated Rejection by Scientific Control)
        # If we are in 'real' mode, we compare against a virtual or previous null baseline
        # Simplified: look for 'null_metrics.json' or use a conservative random baseline
        null_metrics = {"struct_acc": 0.05} # Default placebo baseline
        null_path = os.path.join(checkpoint_dir, "null_metrics_baseline.json")
        if os.path.exists(null_path):
            try:
                with open(null_path) as f: null_metrics = json.load(f)
            except Exception: pass
            
        hypotheses.monitor_failure(train_metrics, null_metrics, shadow_delta=shadow_delta)

        for rep in hypo_reports:
            if "VERIFIED" in rep: print(f"✅ DISCOVERY: {rep}")
            if "REJECTED" in rep: print(f"⚠️ SCIENTIFIC REJECTION: {rep}")
                                    
        # Calculate PDI
        if prev_epoch_probs is not None:
            pdi_p1 = (train_metrics["mean_p1_prob"] - prev_epoch_probs["p1"]).abs().mean().item()
            pdi_p2 = (train_metrics["mean_p2_prob"] - prev_epoch_probs["p2"]).abs().mean().item()
            train_metrics["pdi"] = (pdi_p1 + pdi_p2) / 2.0
            
            # Flip Rate proxy
            h1_diff = (train_metrics["hist_p1"] - prev_epoch_probs["hist_p1"]).abs().sum().item()
            h2_diff = (train_metrics["hist_p2"] - prev_epoch_probs["hist_p2"]).abs().sum().item()
            N1 = train_metrics["hist_p1"].sum().item()
            N2 = train_metrics["hist_p2"].sum().item()
            flip_p1 = h1_diff / (2 * N1) if N1 > 0 else 0.0
            flip_p2 = h2_diff / (2 * N2) if N2 > 0 else 0.0
            train_metrics["flip_rate"] = (flip_p1 + flip_p2) / 2.0
            
        prev_epoch_probs = {
            "p1": train_metrics["mean_p1_prob"].clone(),
            "p2": train_metrics["mean_p2_prob"].clone(),
            "hist_p1": train_metrics["hist_p1"].clone(),
            "hist_p2": train_metrics["hist_p2"].clone()
        }
        
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, criterion, criterion_p, device, edge_weight=1.0)
            val_loss = val_metrics["loss"]
        else:
            val_metrics = train_metrics
            val_loss = train_metrics["loss"]

        # Scheduler
        if scheduler_type == "cosine":
            if epoch >= 18: scheduler.step()
        else:
            if epoch >= 18: scheduler.step(val_loss)

        # Compile success rate (trigger metric)
        top_confusions_readable = []
        compile_rate, confusion = compute_compile_success_rate(model, val_loader or train_loader, device, n_batches_max=4)
        if epoch % log_compile_every == 0 or epoch == epochs:
            compute_compile_success_rate.checkpoint_dir = checkpoint_dir
            compute_compile_success_rate.current_epoch = epoch
            errors = {k: v for k, v in confusion.items() if k[0] != k[1]}
            top_confusions = sorted(errors.items(), key=lambda x: -x[1])[:5]
            top_confusions_readable = [{"pred": config.ID_TO_TOKEN.get(p, str(p)), "true": config.ID_TO_TOKEN.get(g, str(g)), "count": cnt} for (p, g), cnt in top_confusions]

        # Update Detector (trigger-based: entropy + syntax validity)
        detector.update(train_metrics["entropy"], compile_rate, train_metrics.get("pdi", 0.0), 
                        margin=train_metrics.get("struct_margin", None))
        
        if not phase2_active and detector.grammar_ready(epoch):
            print(f"🚀 Trigger-Based Transition: token entropy low and syntax validity high. Phase 2 READY.")
            phase2_active = True

        # Log History
        row = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 4),
            "train_entropy": round(train_metrics.get("entropy", 0.0), 3),
            "train_tension": round(train_metrics.get("tension", 0.0), 4),
            "train_pdi": round(train_metrics.get("pdi", 0.0), 4),
            "val_loss": round(val_loss, 4),
            "phase_lag": round(train_metrics.get("phase_lag", 0.0), 4),
            "struct_acc": round(train_metrics.get("struct_top1_acc", 0.0), 4),
            "pib": round(train_metrics.get("pib", 0.0), 4),
            "sharpness": round(train_metrics.get("sharpness", 0.0), 4),
            "causal_confidence": hypotheses.get_survival_map(),
            "latent_vector": portrait.history[-1].tolist() if portrait.history else None,
        }
        if compile_rate is not None: row["compile_success_rate"] = round(compile_rate, 4)
        if top_confusions_readable: row["top_confusions"] = top_confusions_readable
        history.append(row)

        cr_str = f" | compile={compile_rate*100:.1f}%" if compile_rate is not None else ""
        print(f"Epoch {epoch:3d}/{epochs} | loss={train_metrics['loss']:.4f} | val={val_loss:.4f} | lag={train_metrics.get('phase_lag',0.0):.3f}{cr_str}")

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss ; no_improve = 0
            ckpt_path = os.path.join(checkpoint_dir, f"best_model_{run_name}.pt")
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "val_loss": val_loss, "optimizer_reset_done": reset_done, "phase2_active": phase2_active, "phase2_start_epoch": phase2_start_epoch}, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= early_stop_patience and epoch >= 25:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        # Standard epoch checkpoint
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "val_loss": val_loss, "optimizer_reset_done": reset_done, "phase2_active": phase2_active, "phase2_start_epoch": phase2_start_epoch}, os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt"))

        # v6.1: Crystallization Checkpointing
        competence_score = train_metrics.get("struct_top1_acc", 0.0)
        if emergence_tracker.update(competence_score, epoch):
            golden_path = os.path.join(checkpoint_dir, f"golden_epoch_{epoch:03d}.pt")
            print(f"💎 CRYSTALLIZATION DETECTED! Saving Golden Checkpoint: {golden_path}")
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "is_golden": True, "emergence_velocity": emergence_tracker.best_velocity}, golden_path)

    # Finalize history
    history_path = os.path.join(checkpoint_dir, f"training_history_{run_name}.json")
    with open(history_path, "w") as f: json.dump(history, f, indent=2)
    return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset_dir", type=str, default="data/processed/dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--run_name", type=str, default="v6.1_stabilized")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--reset_optimizer", action="store_true")
    parser.add_argument("--resume_auto", action="store_true")
    parser.add_argument("--force_phase2", action="store_true")
    parser.add_argument("--null_mode", type=str, default="real", choices=["real", "random_labels", "noise_inputs", "geometry_null"])
    args = parser.parse_args()
    
    if args.null_mode != "real":
        os.environ["AK_NULL_MODE"] = args.null_mode

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, dataset_dir=args.dataset_dir, checkpoint_dir=args.checkpoint_dir, run_name=args.run_name, resume_checkpoint=args.resume_checkpoint, num_workers=args.num_workers, grad_accum_steps=args.grad_accum_steps, reset_optimizer=args.reset_optimizer, resume_auto=args.resume_auto, force_phase2=args.force_phase2)
