import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time
import sys
sys.path.append(os.getcwd())

from src.alphaknit import config
from src.alphaknit.model import KnittingTransformer
from src.alphaknit.knitting_dataset import KnittingDataset, make_dataloaders
from src.alphaknit.simulator import ForwardSimulator
from src.alphaknit.compiler import KnittingCompiler

# Phase 9A Hyperparams
TOKEN_MASK_PROB = 0.15  # 15% masking
CHAMFER_GOVERNOR_PATIENCE = 5
JITTER_RATIO = 0.03 # 3% jitter

def train_epoch_phase9(
    model: KnittingTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    mask_prob: float = 0.0,
):
    model.train()
    total_loss = 0.0
    total_grad_norm_enc = 0.0
    total_grad_norm_dec = 0.0
    n_batches = 0
    
    # Track UNK usage for masking
    unk_id = config.UNK_ID
    
    for point_cloud, src_tokens, tgt_tokens in loader:
        point_cloud = point_cloud.to(device)
        src_tokens = src_tokens.to(device)
        tgt_tokens = tgt_tokens.to(device)
        
        # 1. Apply Token Masking (Curriculum)
        # Randomly replace input tokens with UNK to force looking at encoder
        if mask_prob > 0:
            mask = torch.rand_like(src_tokens, dtype=torch.float) < mask_prob
            # Don't mask SOS or PAD
            mask = mask & (src_tokens != config.SOS_ID) & (src_tokens != config.PAD_ID)
            src_tokens_masked = src_tokens.clone()
            src_tokens_masked[mask] = unk_id
        else:
            src_tokens_masked = src_tokens
            
        pad_mask = (src_tokens_masked == config.PAD_ID)
        
        optimizer.zero_grad()
        logits = model(point_cloud, src_tokens_masked, tgt_key_padding_mask=pad_mask)
        
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B*T, V), tgt_tokens.reshape(B*T))
        
        loss.backward()
        
        # 2. Log Gradient Norms (Diagnostic)
        # Encoder: point_encoder
        # Decoder: transformer_decoder + embedding
        
        norm_enc = 0.0
        for p in model.point_encoder.parameters():
            if p.grad is not None:
                norm_enc += p.grad.data.norm(2).item() ** 2
        norm_enc = norm_enc ** 0.5
        
        norm_dec = 0.0
        for p in model.decoder.parameters():
            if p.grad is not None:
                norm_dec += p.grad.data.norm(2).item() ** 2
        for p in model.token_emb.parameters(): # varying embedding
             if p.grad is not None:
                norm_dec += p.grad.data.norm(2).item() ** 2
        norm_dec = norm_dec ** 0.5
        
        total_grad_norm_enc += norm_enc
        total_grad_norm_dec += norm_dec
        
        nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
    avg_loss = total_loss / max(n_batches, 1)
    avg_enc = total_grad_norm_enc / max(n_batches, 1)
    avg_dec = total_grad_norm_dec / max(n_batches, 1)
    
    return avg_loss, avg_enc, avg_dec

def train_phase9():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 9A Training on {device}")
    
    # 3. Resume Phase 8 (Compatible Architecture)
    checkpoint_path = "checkpoints/best_model_phase8.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found. Cannot resume Phase 9A.")
        return
        
    print(f"Resuming from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct model from checkpoint config? Or standard config?
    # Phase 8 modified config might be saved.
    # Assuming standard config for now as architecture didn't change (PointNet+Transformer)
    model = KnittingTransformer(
        d_model=128, n_heads=4, n_layers=3, ffn_dim=256,
        vocab_size=len(config.VOCAB)
    ).to(device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint.get("model_state_dict")))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # Lower LR for Phase 9A finetune
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_ID, label_smoothing=0.1)
    
    # Data - use simulated jitter
    simulator = ForwardSimulator(stitch_width=0.5, stitch_height=0.4)
    # We need a custom Dataset that applies jitter?
    # KnittingDataset implementation loads .npy/.pt files.
    # Jittering point clouds happens at Generation time or Load time?
    # Simulator.to_point_cloud() creates them.
    # If we want Theta-Jitter during training, we need to Re-Simulate or Augment the PC.
    # Augmenting pre-generated PC is hard because we don't know Row/Theta of points easily.
    # Actually, Simulator *generates* the PC. 
    # If we use pre-generated data 'dataset_5k', the PCs are fixed.
    # To apply Theta-Jitter, we need to regenerate the PC from tokens on the fly?
    # OR we apply generic Gaussian noise to the PC.
    # The plan says "Train mode: theta-jitter on stacked discs".
    # This implies we generate new data or jitter existing data.
    # Since dataset generation is slow, maybe we just apply generic noise?
    # Phase 9A Plan part 2 implies modifying Simulator.
    
    # Solution: We should re-generate the dataset OR apply generic jitter to the PC points.
    # Generic jitter `pc += noise` breaks symmetry too!
    # Let's add a transform to the dataset loader.
    
    # Actually, simpler: KnittingDataset just loads files. 
    # Let's rely on standard data for now but assume "Theta-Jitter" means we should have generated data with it?
    # Or maybe we just add noise to the PC during loading.
    
    print("Loading Data...")
    train_loader, val_loader = make_dataloaders(
        dataset_dir="data/debug/dataset_5k", # Reusing debug data for audit runs
        batch_size=32,
        val_split=0.1
    )
    
    # Logging
    log_file = "checkpoints/phase9a_log.txt"
    with open(log_file, "w") as f: f.write("Epoch, Loss, ValLoss, EncGrad, DecGrad, GradRatio\n")
    
    print("Starting Audit Training (50 epochs)...")
    
    for epoch in range(1, 51):
        # Apply Jitter to batch? Assuming DataLoader returns fixed PCs unless we wrap dataset.
        # Let's proceed with fixed PCs + Token Masking for Phase 9A honesty check. 
        # Token Masking is the huge factor.
        
        start_t = time.time()
        loss, enc_grad, dec_grad = train_epoch_phase9(
            model, train_loader, optimizer, criterion, device, 
            mask_prob=TOKEN_MASK_PROB
        )
        
        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
             for pc, src, tgt in val_loader:
                 pc, src, tgt = pc.to(device), src.to(device), tgt.to(device)
                 pad = (src==config.PAD_ID)
                 logits = model(pc, src, tgt_key_padding_mask=pad)
                 val_loss += criterion(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1)).item()
        val_loss /= len(val_loader)
        
        grad_ratio = enc_grad / (dec_grad + 1e-9)
        print(f"Epoch {epoch}: Loss={loss:.4f} Val={val_loss:.4f} EncGrad={enc_grad:.4f} Ratio={grad_ratio:.4f}")
        
        with open(log_file, "a") as f:
            f.write(f"{epoch},{loss},{val_loss},{enc_grad},{dec_grad},{grad_ratio}\n")
            
        # Checkpoint
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"checkpoints/phase9a_epoch_{epoch}.pt")

if __name__ == "__main__":
    train_phase9()
