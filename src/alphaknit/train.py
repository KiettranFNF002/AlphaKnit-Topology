"""
Training loop for KnittingTransformer.

Changes (Phase 8):
- Label smoothing 0.1 (prevents overconfidence on dominant `sc` token)
- Per-epoch compile_success_rate logging (true leading indicator)
- Confusion matrix logging every N epochs (detects sc↔inc confusion)
- Early stopping (patience=10) to avoid wasted compute
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config
from .model import KnittingTransformer
from .knitting_dataset import make_dataloaders


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
) -> dict:
    """Run one training epoch. Returns dict of average losses."""
    model.train()
    total_loss = 0.0
    total_l_type = 0.0
    total_l_parent = 0.0
    total_l_deg = 0.0
    total_l_dist = 0.0
    
    n_batches = 0

    for point_cloud, src_tokens, tgt_tokens in loader:
        point_cloud = point_cloud.to(device)   # (B, N, 3)
        src_tokens  = src_tokens.to(device)    # (B, T, 3)
        tgt_tokens  = tgt_tokens.to(device)    # (B, T, 3)

        # Apply Parent Noise to src_tokens (teacher forcing sequence) if requested
        if parent_noise_prob > 0.0:
            B, T, _ = src_tokens.shape
            mask = torch.rand((B, T), device=device) < parent_noise_prob
            # Random offset between 1 and max_parent_offset (assume 200 for now)
            rand_p1 = torch.randint(1, 200, (B, T), device=device)
            rand_p2 = torch.randint(1, 200, (B, T), device=device)
            src_tokens[:, :, 1] = torch.where(mask, rand_p1, src_tokens[:, :, 1])
            src_tokens[:, :, 2] = torch.where(mask, rand_p2, src_tokens[:, :, 2])

        # Phase 9B: Parent Dropout (15% masked to 0, never 2 consecutive)
        # We only apply this after Stage 1 warmup (i.e. when edge_weight is meaningful)
        if edge_weight > 0.1:
            B, T, _ = src_tokens.shape
            dropout_prob = 0.15
            
            # Generate random dropout mask
            drop_mask = torch.rand((B, T), device=device) < dropout_prob
            
            # Remove consecutive dropouts: if mask[t] and mask[t-1] are both True, set mask[t] to False
            # We can do this with a shifted logical AND, then XOR
            shifted = torch.cat([torch.zeros((B, 1), dtype=torch.bool, device=device), drop_mask[:, :-1]], dim=1)
            consecutive = drop_mask & shifted
            drop_mask = drop_mask ^ consecutive # Flips True to False where consecutive is True
            
            # Apply dropout (0 is padding/unknown for parent embedding)
            src_tokens[:, :, 1] = torch.where(drop_mask, torch.zeros_like(src_tokens[:, :, 1]), src_tokens[:, :, 1])
            src_tokens[:, :, 2] = torch.where(drop_mask, torch.zeros_like(src_tokens[:, :, 2]), src_tokens[:, :, 2])

        # Padding mask based on Type token
        pad_mask = (src_tokens[:, :, 0] == config.PAD_ID)  # (B, T)

        optimizer.zero_grad()

        # Forward
        logits_type, logits_p1, logits_p2 = model(point_cloud, src_tokens, tgt_key_padding_mask=pad_mask)
        
        B, T, _ = tgt_tokens.shape
        V = logits_type.shape[-1]
        P_MAX = logits_p1.shape[-1]
        
        tgt_type = tgt_tokens[:, :, 0].reshape(B * T)
        tgt_p1 = tgt_tokens[:, :, 1].reshape(B * T)
        tgt_p2 = tgt_tokens[:, :, 2].reshape(B * T)

        # 1. Node Type Loss & Entropy
        loss_type = criterion(logits_type.reshape(B * T, V), tgt_type)
        
        # Calculate Node Entropy to monitor "intelligence jump" / phase transition
        probs_type = torch.softmax(logits_type.reshape(B * T, V), dim=-1)
        # Add epsilon to prevent log(0)
        entropy = -(probs_type * torch.log(probs_type + 1e-9)).sum(-1)
        valid_type_mask = (tgt_type != config.PAD_ID)
        entropy_val = entropy[valid_type_mask].mean() if valid_type_mask.any() else torch.tensor(0.0, device=device)

        # 2. Parent Loss (only active if edge_weight > 0)
        loss_parent = 0.0
        if edge_weight > 0:
            l_p1 = criterion_p(logits_p1.reshape(B * T, P_MAX), tgt_p1)
            l_p2 = criterion_p(logits_p2.reshape(B * T, P_MAX), tgt_p2)
            loss_parent = (l_p1 + l_p2) * 0.5

        # 3. Degree Consistency Loss
        # Penalize assigning p2>0 when node type is NOT dec.
        # mr_6, sc, inc only take 1 parent total (or 0 for mr_6).
        # We extract the predicted probability of p2 > 0 and penalize it if true type != dec.
        # (Using true type since decoder is teacher-forced).
        dec_id = config.VOCAB.get('dec', 7)
        mask_non_dec = (tgt_type != dec_id) & (tgt_type != config.PAD_ID)
        
        # Prob of p2 being > 0 is 1.0 - prob(p2 == 0)
        probs_p2 = torch.softmax(logits_p2.reshape(B * T, P_MAX), dim=-1)
        prob_p2_not_null = 1.0 - probs_p2[:, 0]
        
        loss_deg = 0.0
        if mask_non_dec.any():
            loss_deg = prob_p2_not_null[mask_non_dec].mean()
            
        # 4. Parent Distance Regularization (only on non-PAD)
        # We penalize the expected value of |p|.
        # offsets = torch.arange(P_MAX, device=device).float()
        # expected_p1 = (torch.softmax(logits_p1, dim=-1) * offsets).sum(-1) ...
        # Faster/Simpler: just L1 on true offset for now, or L1 on argmax?
        # A differentiable way is expected value.
        offsets = torch.arange(P_MAX, device=device).float().unsqueeze(0) # (1, P_MAX)
        exp_p1 = (torch.softmax(logits_p1.reshape(B * T, P_MAX), dim=-1) * offsets).sum(-1)
        exp_p2 = (probs_p2 * offsets).sum(-1)
        
        valid_mask = (tgt_type != config.PAD_ID)
        loss_dist = 0.0
        if valid_mask.any():
            loss_dist = (exp_p1[valid_mask] + exp_p2[valid_mask]).mean()
            
        # 4.5 Parent Decision Instability (PDI) - Emergence Seismograph
        # We compute PDI without temperature sharpening to get the "true" model confidence shift
        probs_p1_raw = torch.softmax(logits_p1.reshape(B, T, P_MAX), dim=-1).detach()
        probs_p2_raw = torch.softmax(logits_p2.reshape(B, T, P_MAX), dim=-1).detach()
        
        # Topology-aligned PDI: Only compute on nodes in the second half of the sequence
        # to isolate structural decisions from initial grammar grammar zone.
        T_mid = T // 2
        
        if not hasattr(train_epoch, "epoch_prob_p1_acc"):
            train_epoch.epoch_prob_p1_acc = torch.zeros(P_MAX, device=device)
            train_epoch.epoch_prob_p2_acc = torch.zeros(P_MAX, device=device)
            # For flip rate, we track the histogram of argmax decisions
            train_epoch.epoch_argmax_p1_hist = torch.zeros(P_MAX, device=device)
            train_epoch.epoch_argmax_p2_hist = torch.zeros(P_MAX, device=device)
            train_epoch.epoch_valid_nodes = 0.0
            
        if T_mid < T:
            # We track the mean probability vector for the second half of the sequence
            mean_prob_p1 = probs_p1_raw[:, T_mid:, :].reshape(-1, P_MAX).mean(dim=0)
            mean_prob_p2 = probs_p2_raw[:, T_mid:, :].reshape(-1, P_MAX).mean(dim=0)
            
            train_epoch.epoch_prob_p1_acc += mean_prob_p1
            train_epoch.epoch_prob_p2_acc += mean_prob_p2
            
            # For flip rate, we add the counts of argmax predictions
            argmax_p1 = probs_p1_raw[:, T_mid:, :].reshape(-1, P_MAX).argmax(dim=-1)
            argmax_p2 = probs_p2_raw[:, T_mid:, :].reshape(-1, P_MAX).argmax(dim=-1)
            train_epoch.epoch_argmax_p1_hist += torch.bincount(argmax_p1, minlength=P_MAX).float()
            train_epoch.epoch_argmax_p2_hist += torch.bincount(argmax_p2, minlength=P_MAX).float()
            
        # 5. Topology Tension Signal (Phase 10.5)
        loss_tension = torch.tensor(0.0, device=device)
        loss_div = torch.tensor(0.0, device=device)
        loss_entropy = torch.tensor(0.0, device=device)
        
        if epoch >= 12:
            if epoch < 18:
                tau = 0.5
                frac = 0.2
            elif epoch < 30:
                tau = 0.4
                frac = 0.5
            else:
                tau = 0.3
                frac = 1.0
                
            # Compute probabilities with temperature
            probs_p1_t = torch.softmax(logits_p1.reshape(B, T, P_MAX) / tau, dim=-1)
            probs_p2_t = torch.softmax(logits_p2.reshape(B, T, P_MAX) / tau, dim=-1)
            
            # Entropy barrier
            ent_p1 = -(probs_p1_t * torch.log(probs_p1_t + 1e-9)).sum(-1)
            ent_p2 = -(probs_p2_t * torch.log(probs_p2_t + 1e-9)).sum(-1)
            loss_entropy = (ent_p1 + ent_p2).mean() * 0.5
            
            # Parent Diversity Bias
            loss_div = (probs_p1_t * probs_p2_t).sum(-1).mean()
            
            # Continuous Adjacency Matrix A: (B, T, T)
            A = torch.zeros(B, T, T, device=device)
            for offset in range(1, P_MAX): # 0 is PAD/no parent
                if T - offset <= 0: continue
                prob = 0.5 * (probs_p1_t[:, offset:, offset] + probs_p2_t[:, offset:, offset])
                A.diagonal(dim1=1, dim2=2, offset=-offset).copy_(prob)
                
            # Degree Normalization
            deg = A.sum(dim=-1, keepdim=True) + 1e-9
            A_tilde = A / deg
            
            # Initialization
            X = torch.randn(B, T, 3, device=device)
            T_active = max(1, int(T * frac))
            T_start = T - T_active
            
            # Residual Laplacian Relaxation
            alpha = 0.2
            for _ in range(3):
                X_next = X + alpha * (torch.bmm(A_tilde, X) - X)
                if T_start > 0:
                    X_early = X_next[:, :T_start, :].detach()
                    X_next = torch.cat([X_early, X_next[:, T_start:, :]], dim=1)
                X = X_next
                
            # Scale-Invariant Tension Energy
            X_expand_i = X.unsqueeze(2) # (B, T, 1, 3)
            X_expand_j = X.unsqueeze(1)    # (B, 1, T, 3)
            dist = torch.sqrt(((X_expand_i - X_expand_j)**2).sum(-1) + 1e-9) # (B, T, T)
            
            mean_dist = dist.mean(dim=(1, 2), keepdim=True) + 1e-9
            d_hat = dist / mean_dist
            
            E = A * (d_hat - 1.0)**2 # (B, T, T)
            
            # Mask inactive nodes and padding
            valid_mask_2d = (tgt_tokens[:, :, 0] != config.PAD_ID) # (B, T)
            valid_mask_E = valid_mask_2d.unsqueeze(2).expand(B, T, T) # (B, T, T)
            
            if T_start > 0:
                E = E[:, T_start:, :]
                valid_mask_E = valid_mask_E[:, T_start:, :]
                
            if valid_mask_E.any():
                loss_tension = E[valid_mask_E].mean()

        # Phase 9B Total Loss Combination
        # Weights: Type: 1.0, Edge: 0.3, Deg: 0.2, Dist: 0.01 (Chamfer will be added separately if integrated)
        W_TYPE = 1.0
        W_PARENT = 0.3 * edge_weight
        W_DEG = 0.2
        W_DIST = 0.01
        
        W_TENSION = 0.02
        W_ENTROPY = 0.005
        W_DIV = 0.002
        
        loss_val = (W_TYPE * loss_type) + (W_PARENT * loss_parent) + (W_DEG * loss_deg) + (W_DIST * loss_dist)
        loss_val = loss_val + (W_TENSION * loss_tension) + (W_ENTROPY * loss_entropy) + (W_DIV * loss_div)
        
        # Scale loss before backward for gradient accumulation
        loss = loss_val / grad_accum_steps
        loss.backward()
        
        if (n_batches + 1) % grad_accum_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss_val.item()
        total_l_type += loss_type.item()
        total_l_parent += loss_parent.item() if isinstance(loss_parent, torch.Tensor) else loss_parent
        total_l_deg += loss_deg.item() if isinstance(loss_deg, torch.Tensor) else loss_deg
        total_l_dist += loss_dist.item() if isinstance(loss_dist, torch.Tensor) else loss_dist
        
        # Initialize total_entropy if not exists, since it's a new metric
        if not hasattr(train_epoch, "total_entropy"):
            train_epoch.total_entropy = 0.0
        train_epoch.total_entropy += entropy_val.item()
        
        if not hasattr(train_epoch, "total_tension"):
            train_epoch.total_tension = 0.0
        if isinstance(loss_tension, torch.Tensor):
            train_epoch.total_tension += loss_tension.item()
        else:
            train_epoch.total_tension += loss_tension
        
        n_batches += 1

    ret = {
        "loss": total_loss / max(n_batches, 1),
        "l_type": total_l_type / max(n_batches, 1),
        "l_edge": total_l_parent / max(n_batches, 1),
        "l_deg": total_l_deg / max(n_batches, 1),
        "l_dist": total_l_dist / max(n_batches, 1),
        "entropy": train_epoch.total_entropy / max(n_batches, 1),
        "tension": train_epoch.total_tension / max(n_batches, 1),
        "mean_p1_prob": train_epoch.epoch_prob_p1_acc / max(n_batches, 1),
        "mean_p2_prob": train_epoch.epoch_prob_p2_acc / max(n_batches, 1),
        "hist_p1": train_epoch.epoch_argmax_p1_hist,
        "hist_p2": train_epoch.epoch_argmax_p2_hist,
    }
    # Reset tracking variable
    train_epoch.total_entropy = 0.0
    train_epoch.total_tension = 0.0
    train_epoch.epoch_prob_p1_acc = torch.zeros(P_MAX, device=device)
    train_epoch.epoch_prob_p2_acc = torch.zeros(P_MAX, device=device)
    train_epoch.epoch_argmax_p1_hist = torch.zeros(P_MAX, device=device)
    train_epoch.epoch_argmax_p2_hist = torch.zeros(P_MAX, device=device)
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

    for point_cloud, src_tokens, tgt_tokens in loader:
        point_cloud = point_cloud.to(device)
        src_tokens  = src_tokens.to(device)
        tgt_tokens  = tgt_tokens.to(device)

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
        
        loss_deg = 0.0
        if mask_non_dec.any():
            loss_deg = prob_p2_not_null[mask_non_dec].mean()
            
        # 4. Distance Reg
        offsets = torch.arange(P_MAX, device=device).float().unsqueeze(0)
        exp_p1 = (torch.softmax(logits_p1.reshape(B * T, P_MAX), dim=-1) * offsets).sum(-1)
        exp_p2 = (probs_p2 * offsets).sum(-1)
        
        valid_mask = (tgt_type != config.PAD_ID)
        loss_dist = 0.0
        if valid_mask.any():
            loss_dist = (exp_p1[valid_mask] + exp_p2[valid_mask]).mean()

        # Combine
        W_TYPE = 1.0
        W_PARENT = 0.3 * edge_weight
        W_DEG = 0.2
        W_DIST = 0.01
        
        loss = (W_TYPE * loss_type) + (W_PARENT * loss_parent) + (W_DEG * loss_deg) + (W_DIST * loss_dist)

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

    confusion_counts: dict mapping (pred_token, true_token) → count
    for non-PAD positions. Useful to detect sc↔inc confusion.
    """
    import os
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
        except Exception as e:
            pass # Silently fail if plotting is unavailable

    model.eval()
    compiler = KnittingCompiler()

    compile_ok = 0
    compile_total = 0
    confusion = {}   # (pred_id, true_id) → count
    graphs_saved = 0

    for i, (point_cloud, src_tokens, tgt_tokens) in enumerate(loader):
        if i >= n_batches_max:
            break

        point_cloud = point_cloud.to(device)
        src_tokens  = src_tokens.to(device)
        tgt_tokens  = tgt_tokens.to(device)

        # Greedy decode (batch)
        pred_ids_list = model.greedy_decode(point_cloud, max_len=config.MAX_SEQ_LEN)

        for b, pred_ids in enumerate(pred_ids_list):
            # Format tuples (type, p1, p2) to full syntax for compiler
            tokens = []
            for tpl in pred_ids:
                if isinstance(tpl, tuple) and len(tpl) == 3:
                    t, p1, p2 = tpl
                    t_str = config.ID_TO_TOKEN.get(t, "<UNK>")
                    tokens.append(f"{t_str}({p1},{p2})")
                else:
                    t_str = config.ID_TO_TOKEN.get(tpl, "<UNK>")
                    tokens.append(f"{t_str}(0,0)") # Fallback

            try:
                graph = compiler.compile(tokens)
                compile_ok += 1
                
                # Visual Emergence Watcher
                if getattr(compute_compile_success_rate, "checkpoint_dir", None) is not None:
                    epoch = getattr(compute_compile_success_rate, "current_epoch", 0)
                    if epoch > 0 and epoch % 2 == 0 and graphs_saved < 3:
                        watch_dir = os.path.join(compute_compile_success_rate.checkpoint_dir, "emergence_watch")
                        os.makedirs(watch_dir, exist_ok=True)
                        filepath = os.path.join(watch_dir, f"epoch_{epoch:03d}_sample_{graphs_saved}.png")
                        _save_emergence_graph(graph, filepath)
                        graphs_saved += 1
                        
            except (CompileError, Exception):
                pass
            compile_total += 1

            # Confusion: compare to ground truth (tgt_tokens)
            gt_ids = tgt_tokens[b].tolist() # list of [type, p1, p2]
            pred_seq = pred_ids[:len(gt_ids)]
            # Pad pred to gt length
            pad_tuple = (config.PAD_ID, 0, 0)
            pred_seq = pred_seq + [pad_tuple] * (len(gt_ids) - len(pred_seq))
            
            for p, g in zip(pred_seq, gt_ids):
                g_type = g[0]
                if g_type == config.PAD_ID:
                    continue
                p_type = p[0] if isinstance(p, tuple) else p
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
):
    """Full training loop with checkpointing."""

    # Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Training on: {device}")

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

    # Resume from checkpoint
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch - 1} (val_loss={best_val_loss:.4f})")
        # Load existing history if available
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

    # Phase 10: Anti-Collapse Class Weights for Node Type Prediction
    # Weights for pad/cls/sep are 0 (ignored), others are balanced to prevent 'dec' (7) dominance
    # config.VOCAB = {'pad': 0, 'cls': 1, 'sep': 2, 'sc': 3, 'inc': 4, 'mr_l': 5, 'mr_r': 6, 'dec': 7}
    class_weights = torch.ones(len(config.VOCAB), device=device)
    class_weights[0] = 0.0 # pad ignore
    class_weights[1] = 0.1 # cls low impact
    class_weights[2] = 0.1 # sep low impact
    class_weights[7] = 0.5 # dec is very common, reduce priority
    class_weights[3] = 2.0 # sc
    class_weights[4] = 3.0 # inc
    class_weights[5] = 4.0 # mr_l
    class_weights[6] = 4.0 # mr_r
    
    # Phase 8: label smoothing — prevents overconfidence on sc
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=config.PAD_ID,
        label_smoothing=label_smoothing,
    )
    
    # Phase 9B: Parent prediction loss (ignore 0 offset / PAD)
    criterion_p = nn.CrossEntropyLoss(ignore_index=0)

    import math
    os.makedirs(checkpoint_dir, exist_ok=True)
    no_improve = 0  # early stopping counter
    
    prev_epoch_probs = None

    for epoch in range(start_epoch, epochs + 1):
        # Phase 10 Training Curriculum
        # Slowed down Sigmoid transition for edge weight (Centered at Epoch 10, width 4)
        edge_weight = 1.0 / (1.0 + math.exp(-(epoch - 10) / 4.0))
        # Sector loss activates late (epoch 20)
        sector_weight = 1.0 / (1.0 + math.exp(-(epoch - 20) / 2.0))
        # 5% parent noise during Stage 1 (Epoch 0-10)
        parent_noise_prob = 0.05 if epoch <= 10 else 0.0

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, criterion_p, device, 
                                    edge_weight=edge_weight, sector_weight=sector_weight, 
                                    parent_noise_prob=parent_noise_prob, grad_accum_steps=grad_accum_steps,
                                    prev_epoch_probs=prev_epoch_probs)
                                    
        # Calculate PDI (Parent Decision Instability)
        pdi = 0.0
        flip_rate = 0.0
        # Since we cannot easily do strict node-to-node matching due to dataloader shuffling, 
        # we track the shift in the global histogram of the aligned probability distributions. 
        # This proxy has been verified to capture the 'cognitive instability' during the rewrite phase.
        if prev_epoch_probs is not None:
            pdi_p1 = (train_metrics["mean_p1_prob"] - prev_epoch_probs["p1"]).abs().mean().item()
            pdi_p2 = (train_metrics["mean_p2_prob"] - prev_epoch_probs["p2"]).abs().mean().item()
            pdi = (pdi_p1 + pdi_p2) / 2.0
            train_metrics["pdi"] = pdi
            
            # Compute Flip Rate (using histogram L1 distance as a proxy for hard argmax flips)
            # sum(|H_t - H_{t-1}|) / (2 * N) gives the proportion of assignments that changed bin
            h1_diff = (train_metrics["hist_p1"] - prev_epoch_probs["hist_p1"]).abs().sum().item()
            h2_diff = (train_metrics["hist_p2"] - prev_epoch_probs["hist_p2"]).abs().sum().item()
            N1 = train_metrics["hist_p1"].sum().item()
            N2 = train_metrics["hist_p2"].sum().item()
            flip_p1 = h1_diff / (2 * N1) if N1 > 0 else 0.0
            flip_p2 = h2_diff / (2 * N2) if N2 > 0 else 0.0
            flip_rate = (flip_p1 + flip_p2) / 2.0
            train_metrics["flip_rate"] = flip_rate
            
        prev_epoch_probs = {
            "p1": train_metrics["mean_p1_prob"].clone(),
            "p2": train_metrics["mean_p2_prob"].clone(),
            "hist_p1": train_metrics["hist_p1"].clone(),
            "hist_p2": train_metrics["hist_p2"].clone()
        }
        
        if val_loader is not None:
            # We must evaluate with edge_weight=1.0 so that the validation loss is a STATIC, absolute metric.
            # If we evaluate with the curriculum edge_weight, the total val_loss will artificially INCREASE 
            # as the curriculum ramps up, which triggers Early Stopping incorrectly because it looks like degradation!
            val_metrics = evaluate(model, val_loader, criterion, criterion_p, device, edge_weight=1.0)
            val_loss = val_metrics["loss"]
        else:
            # If skipping validation (e.g. tar shards without explicit val split)
            val_metrics = train_metrics
            val_loss = train_metrics["loss"]

        # Phase 10: Delay scheduler steps during curriculum representation rewrite
        if scheduler_type == "cosine":
            if epoch >= 18:
                scheduler.step()
        else:
            if epoch >= 18:
                scheduler.step(val_loss)

        # Phase 8: compile success rate (every N epochs or last epoch)
        compile_rate = None
        top_confusions = []
        if epoch % log_compile_every == 0 or epoch == epochs:
            # Inject context for Emergence Watcher
            compute_compile_success_rate.checkpoint_dir = checkpoint_dir
            compute_compile_success_rate.current_epoch = epoch
            
            compile_rate, confusion = compute_compile_success_rate(
                model, val_loader or train_loader, device, n_batches_max=8
            )
            # Top confusions (pred≠true, non-PAD)
            errors = {k: v for k, v in confusion.items() if k[0] != k[1]}
            top_confusions = sorted(errors.items(), key=lambda x: -x[1])[:5]
            top_confusions_readable = [
                {
                    "pred": config.ID_TO_TOKEN.get(p, str(p)),
                    "true": config.ID_TO_TOKEN.get(g, str(g)),
                    "count": cnt,
                }
                for (p, g), cnt in top_confusions
            ]
        else:
            top_confusions_readable = []

        row = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 4),
            "train_l_type": round(train_metrics["l_type"], 4),
            "train_l_edge": round(train_metrics["l_edge"], 4),
            "train_l_deg": round(train_metrics["l_deg"], 4),
            "train_entropy": round(train_metrics.get("entropy", 0.0), 3),
            "train_tension": round(train_metrics.get("tension", 0.0), 4),
            "train_pdi": round(train_metrics.get("pdi", 0.0), 4),
            "train_flip": round(train_metrics.get("flip_rate", 0.0), 4),
            "val_loss": round(val_metrics["loss"], 4),
            "edge_weight": round(edge_weight, 3),
        }
        if compile_rate is not None:
            row["compile_success_rate"] = round(compile_rate, 4)
        if top_confusions_readable:
            row["top_confusions"] = top_confusions_readable

        history.append(row)

        # Console log
        cr_str = f" | compile={compile_rate*100:.1f}%" if compile_rate is not None else ""
        ent_tension_str = f", ent={train_metrics.get('entropy', 0.0):.2f}, ts={train_metrics.get('tension', 0.0):.3f}, pdi={train_metrics.get('pdi', 0.0):.3f}, flip={train_metrics.get('flip_rate', 0.0):.3f}"
        print(f"Epoch {epoch:3d}/{epochs} | train={train_metrics['loss']:.4f} (ty={train_metrics['l_type']:.2f}, ed={train_metrics['l_edge']:.2f}{ent_tension_str}) | val={val_loss:.4f}{cr_str}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            ckpt_path = os.path.join(checkpoint_dir, f"best_model_{run_name}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": {
                    "d_model": d_model, "n_heads": n_heads,
                    "n_layers": n_layers, "ffn_dim": ffn_dim,
                },
            }, ckpt_path)
        else:
            no_improve += 1
            # Phase 10: Early stopping ONLY triggers after Epoch 25 to allow emergence
            if no_improve >= early_stop_patience and epoch >= 25:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
                break

    # Save history
    history_path = os.path.join(checkpoint_dir, f"training_history_{run_name}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {checkpoint_dir}/best_model_{run_name}.pt")
    return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset_dir", type=str, default="data/processed/dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--run_name", type=str, default="phase9b_dev")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    args = parser.parse_args()
    
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dataset_dir=args.dataset_dir,
        checkpoint_dir=args.checkpoint_dir,
        run_name=args.run_name,
        resume_checkpoint=args.resume_checkpoint,
        num_workers=args.num_workers,
        grad_accum_steps=args.grad_accum_steps
    )
