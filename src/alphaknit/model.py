"""
AlphaKnit AI Model

Architecture:
  PointNetEncoder (multi-scale):  (B, N, 3) → (B, d_model)
  KnittingTransformer: encoder memory + token sequence → token logits

Phase 8 changes:
  - Multi-scale encoder: max-pool + avg-pool concatenated → project to d_model
  - BatchNorm after each MLP layer (stable with synthetic data)
  - Compile-guided beam search: prunes beams that are already invalid
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


# ------------------------------------------------------------------ #
#  PointNet Encoder (multi-scale)                                     #
# ------------------------------------------------------------------ #

class PointNetEncoder(nn.Module):
    """
    Multi-scale PointNet: per-point MLP → max-pool AND avg-pool → concat → project.

    max-pool captures dominant features (sharp edges, extremes).
    avg-pool captures global distribution (overall density).
    Together they give a richer surface representation.

    Input:  (B, N, 3)
    Output: (B, d_model)
    """

    def __init__(self, d_model: int = config.D_MODEL):
        super().__init__()
        # Per-point MLP with BatchNorm (stable on clean synthetic data)
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # After concat(max, avg) → dim=512
        # Phase 9C: We append [sin(theta), cos(theta), r] per pool → total +3 to global feature?
        # Actually, theta and r are PER POINT. We must append them BEFORE pooling, OR
        # compute them, append to raw features, and push through MLP. 
        # But ADR says: "Append [sin(θ), cos(θ), r] to input features AFTER normalization (post-BatchNorm)."
        # This implies we can do a secondary MLP or just append it to the 256-dim feature 
        # before pooling. Let's append to the 256-dim feature array before pooling.
        
        # So per-point: 256 (from MLP) + 3 (sin, cos, r) = 259
        # Pooling concat: 259 (max) + 259 (avg) = 518
        self.proj = nn.Linear(518, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, 3) point cloud
        Returns:
            (B, d_model) global feature
        """
        B, N, _ = x.shape

        # Per-point features — BatchNorm1d expects (B*N, C)
        x_flat = x.reshape(B * N, 3)
        feat = self.mlp(x_flat)          # (B*N, 256)
        feat = feat.reshape(B, N, 256)   # (B, N, 256)

        # ------------------------------------------------------------- #
        # Phase 9C: Angular Positional Encoding (relative to layer center)
        # Because we don't have explicit layer indices here, we calculate
        # a local density-based center or just use the batch centroid. 
        # Wait, the point cloud in PointNet doesn't have layer metadata!
        # Alternative: We group points by Y-coordinate (height) to find 
        # per-layer centers dynamically.
        # Since crochet layers grow in Y, we can bucket points by Y.
        # ------------------------------------------------------------- #
        device = x.device
        
        # Sort points by Y to approximate layers (simplistic, but effective)
        # For true per-layer, we'd need layer IDs passed in. 
        # Let's use 10 height buckets as a proxy for layers.
        # Actually, simpler and robust: use Z-X slices. 
        y = x[:, :, 1]
        y_min = y.min(dim=1, keepdim=True).values
        y_max = y.max(dim=1, keepdim=True).values
        # Normalize Y to [0, 10] buckets
        y_norm = (y - y_min) / (y_max - y_min + 1e-6)
        y_buckets = (y_norm * 10).long()
        
        # Calculate centroids per bucket
        # We will just broadcast and compute weighted centers.
        # To avoid slow loops, we compute a global center for now, 
        # but weighting by Y proximity could act as a local center.
        # Let's compute a simple running local center using a Gaussian kernel over Y.
        
        # For simplicity and speed, let's use the slice median as center.
        x_centers = torch.zeros(B, N, device=device)
        z_centers = torch.zeros(B, N, device=device)
        
        for b in range(B):
            for i in range(11):
                mask = (y_buckets[b] == i)
                if mask.any():
                    cx = x[b, mask, 0].mean()
                    cz = x[b, mask, 2].mean()
                    x_centers[b, mask] = cx
                    z_centers[b, mask] = cz
                    
        dx = x[:, :, 0] - x_centers
        dz = x[:, :, 2] - z_centers
        
        theta = torch.atan2(dz, dx)  # (B, N)
        r = torch.sqrt(dx**2 + dz**2) # (B, N)
        
        # Features to append: sin(theta), cos(theta), r
        spatial_feats = torch.stack([torch.sin(theta), torch.cos(theta), r], dim=-1) # (B, N, 3)
        
        # Append AFTER MLP (post-BatchNorm)
        feat = torch.cat([feat, spatial_feats], dim=-1) # (B, N, 256 + 3 = 259)

        # Multi-scale pooling
        feat_max = feat.max(dim=1).values   # (B, 259) — dominant features
        feat_avg = feat.mean(dim=1)         # (B, 259) — global density

        # Concatenate and project
        feat_cat = torch.cat([feat_max, feat_avg], dim=1)  # (B, 518)
        out = self.proj(feat_cat)     # (B, d_model)
        return self.norm(out)


# ------------------------------------------------------------------ #
#  Knitting Transformer                                                #
# ------------------------------------------------------------------ #

class KnittingTransformer(nn.Module):
    """
    Encoder-decoder Transformer for point cloud → stitch token sequence.
    Phase 9B: Edge-Action Sequence Model.
    Generates (node_type, p1_offset, p2_offset) tuples.
    Decoder features a Sequential Factorized prediction head.
    """

    def __init__(
        self,
        vocab_size: int = config.VOCAB_SIZE,
        d_model: int = config.D_MODEL,
        n_heads: int = config.N_HEADS,
        n_layers: int = config.N_LAYERS,
        ffn_dim: int = config.FFN_DIM,
        dropout: float = config.DROPOUT,
        max_seq_len: int = config.MAX_SEQ_LEN,
        max_parent_offset: int = 200, # Max historical distance for parent pointers
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.max_parent_offset = max_parent_offset

        # Encoder (multi-scale)
        self.point_encoder = PointNetEncoder(d_model)

        # Token embedding + positional encoding
        self.type_emb = nn.Embedding(vocab_size, d_model, padding_idx=config.PAD_ID)
        self.p1_emb = nn.Embedding(max_parent_offset, d_model, padding_idx=0) # offset=0 means no parent (e.g. padding/MR)
        self.p2_emb = nn.Embedding(max_parent_offset, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Sequential Factorized Decoder Heads
        # 1. Predict Node Type
        self.type_head = nn.Linear(d_model, vocab_size)
        
        # 2. Predict Parent 1 (conditioned on predicted/true Node Type embedding)
        self.p1_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, max_parent_offset)
        )
        
        # 3. Predict Parent 2 (conditioned on predicted/true Node Type AND Parent 1 embedding)
        self.p2_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, max_parent_offset)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.type_emb.weight)
        nn.init.xavier_uniform_(self.p1_emb.weight)
        nn.init.xavier_uniform_(self.p2_emb.weight)
        nn.init.xavier_uniform_(self.type_head.weight)
        nn.init.zeros_(self.type_head.bias)

    def forward(
        self,
        point_cloud: torch.Tensor,   # (B, N, 3)
        tgt_tokens: torch.Tensor,    # (B, T, 3) — decoder input tuple: (type, p1, p2)
        tgt_key_padding_mask: torch.Tensor = None,  # (B, T) bool, True=ignore
    ) -> tuple:
        """
        Returns:
            logits_type: (B, T, vocab_size)
            logits_p1: (B, T, max_parent_offset)
            logits_p2: (B, T, max_parent_offset)
        """
        B, T, _ = tgt_tokens.shape

        # Encode point cloud → memory (B, 1, d_model)
        memory = self.point_encoder(point_cloud).unsqueeze(1)  # (B, 1, d_model)

        # Token embeddings + positional
        # tgt_tokens is (type, p1, p2)
        types = tgt_tokens[:, :, 0]
        p1s = tgt_tokens[:, :, 1]
        p2s = tgt_tokens[:, :, 2]
        
        # For training (teacher forcing), we use ground truth embeddings.
        t_emb = self.type_emb(types)
        p1_emb = self.p1_emb(p1s)
        p2_emb = self.p2_emb(p2s)
        
        # Combine embeddings (sum is standard for multi-modal, or concat and project)
        combined_emb = t_emb + p1_emb + p2_emb
        
        positions = torch.arange(T, device=tgt_tokens.device).unsqueeze(0)  # (1, T)
        tgt_emb = combined_emb + self.pos_emb(positions)      # (B, T, d_model)

        # Causal mask (prevent attending to future tokens)
        causal_mask = torch.triu(
            torch.ones((T, T), dtype=torch.bool, device=tgt_tokens.device),
            diagonal=1
        )

        # Decode (shared hidden state h)
        h = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # (B, T, d_model)
        
        # v6.0: Expose last hidden state for research telemetry (Latent Phase Portrait)
        self.last_hidden_state = h

        # 1. Type Logits
        logits_type = self.type_head(h) # (B, T, vocab_size)
        
        # 2. P1 Logits
        # Conditioned on h and the embedding of the type (using teacher forced type during training for now, 
        # but to be strictly causal, we use the type embedding of the ground truth at this step).
        # Wait, the target is to predict step t's components.
        # h already contains context up to t-1.
        # To predict type at t: use h.
        # To predict p1 at t: use concat(h, embed(type_t)). In training, type_t is ground truth.
        h_p1 = torch.cat([h, t_emb], dim=-1)
        logits_p1 = self.p1_head(h_p1) # (B, T, max_offset)
        
        # 3. P2 Logits
        # Conditioned on h, embed(type_t), embed(p1_t)
        h_p2 = torch.cat([h, t_emb, p1_emb], dim=-1)
        logits_p2 = self.p2_head(h_p2)

        return logits_type, logits_p1, logits_p2

    # ------------------------------------------------------------------ #
    #  Greedy Decode                                                       #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def greedy_decode(
        self,
        point_cloud: torch.Tensor,   # (1, N, 3) or (B, N, 3)
        max_len: int = 200,
    ) -> list:
        """
        Greedy autoregressive decoding for Edge-Action tuples.

        Returns:
            List of tuple lists (one per batch item), stopping at <EOS>.
        """
        self.eval()
        B = point_cloud.shape[0]
        device = point_cloud.device

        # Start with <SOS, 0, 0>
        # Shape: (B, 1, 3)
        generated = torch.zeros((B, 1, 3), dtype=torch.long, device=device)
        generated[:, 0, 0] = config.SOS_ID
        
        finished = [False] * B

        for step in range(max_len):
            # 1. Forward pass to get Type logits
            # `generated` has shape (B, T, 3). We only need the last step's logits.
            memory = self.point_encoder(point_cloud).unsqueeze(1)
            
            types = generated[:, :, 0]
            p1s = generated[:, :, 1]
            p2s = generated[:, :, 2]
            
            t_emb = self.type_emb(types)
            p1_emb = self.p1_emb(p1s)
            p2_emb = self.p2_emb(p2s)
            
            combined_emb = t_emb + p1_emb + p2_emb
            
            T = generated.shape[1]
            positions = torch.arange(T, device=device).unsqueeze(0)
            tgt_emb = combined_emb + self.pos_emb(positions)
            
            causal_mask = torch.triu(torch.ones((T, T), dtype=torch.bool, device=device), diagonal=1)
            
            h = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=causal_mask)
            
            # Predict Type
            logits_type = self.type_head(h[:, -1:, :]) # (B, 1, vocab_size)
            
            # Mask special tokens
            for tok_id in (config.PAD_ID, config.SOS_ID, config.UNK_ID):
                logits_type[:, 0, tok_id] = float('-inf')
            if T > 1:
                logits_type[:, 0, config.VOCAB.get('mr_6', 4)] = float('-inf')
                
            next_type = logits_type.argmax(dim=-1) # (B, 1)
            
            # 2. Predict P1
            # We need to embed the PREDICTED type to feed into P1 head.
            pred_t_emb = self.type_emb(next_type) # (B, 1, d_model)
            h_p1 = torch.cat([h[:, -1:, :], pred_t_emb], dim=-1)
            logits_p1 = self.p1_head(h_p1) # (B, 1, max_offset)
            next_p1 = logits_p1.argmax(dim=-1) # (B, 1)
            
            # 3. Predict P2
            pred_p1_emb = self.p1_emb(next_p1)
            h_p2 = torch.cat([h[:, -1:, :], pred_t_emb, pred_p1_emb], dim=-1)
            logits_p2 = self.p2_head(h_p2) # (B, 1, max_offset)
            next_p2 = logits_p2.argmax(dim=-1) # (B, 1)
            
            # Combine
            next_tuple = torch.stack([next_type, next_p1, next_p2], dim=-1) # (B, 1, 3)
            generated = torch.cat([generated, next_tuple], dim=1)

            for b in range(B):
                if next_type[b, 0].item() == config.EOS_ID:
                    finished[b] = True

            if all(finished):
                break

        # Convert to lists of tuples, strip <SOS> and <EOS>
        results = []
        for b in range(B):
            seq = generated[b].tolist() # List of [type, p1, p2]
            if seq and seq[0][0] == config.SOS_ID:
                seq = seq[1:]
            
            clean_seq = []
            for tpl in seq:
                if tpl[0] == config.EOS_ID:
                    break
                clean_seq.append(tuple(tpl))
            results.append(clean_seq)

        return results

    # ------------------------------------------------------------------ #
    #  Compile-Guided Beam Search (Phase 8)                               #
    # ------------------------------------------------------------------ #


    # Beam Search left as a TODO for later if needed, since it requires
    # updating the prefix viable checker for graphs.
    @torch.no_grad()
    def beam_decode(
        self,
        point_cloud: torch.Tensor,
        beam_width: int = 5,
        max_len: int = 200,
        compile_guided: bool = False,
    ) -> list:
        # Fallback to greedy for Phase 9B initially
        return self.greedy_decode(point_cloud, max_len=max_len)[0]

    def ids_to_tokens(self, ids: list) -> list:
        """Phase 9B: Convert tuple list to pseudo-string list for debug."""
        res = []
        for tpl in ids:
            t_id, p1, p2 = tpl
            t_str = config.ID_TO_TOKEN.get(t_id, "<UNK>")
            res.append(f"{t_str}({p1},{p2})")
        return res
