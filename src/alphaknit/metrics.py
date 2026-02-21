import torch
import torch.nn.functional as F

def topology_tension_field(node_degrees, edge_count, num_nodes, lambda_density=0.1, report_only=False):
    """
    Structural Regularization Loss.
    Penalizes high degree variance and rewards minimum edge density to avoid "Lazy Collapse".
    """
    # 1. Variance Penalty: Stable topology has consistent local connectivity
    degree_var = node_degrees.var(dim=1).mean()
    
    # 2. Anti-Collapse Density Penalty: Prevent the model from cheating by deleting edges
    # Standard knitting graph usually has effective density ~1.5 to 2.5
    edge_density = edge_count / (num_nodes + 1e-6)
    density_penalty = torch.relu(0.15 - edge_density) # Activate if below 0.15 density
    
    ttf_loss = degree_var + lambda_density * density_penalty
    
    stats = {
        "degree_var": degree_var.detach().item(),
        "edge_density": edge_density.detach().item()
    }
    
    if report_only:
        return torch.tensor(0.0, device=ttf_loss.device, requires_grad=True), stats
        
    return ttf_loss, stats


def compute_structural_metrics(logits, targets, structural_mask, topk=(1, 3)):
    """
    Measures the "Logit Margin" and Top-K accuracy specifically for structural decisions.
    logits: [B, T, V]
    targets: [B, T]
    structural_mask: [B, T] (bool)
    """
    # Extract only structural tokens
    logits_s = logits[structural_mask] # [N, V]
    targets_s = targets[structural_mask] # [N]
    
    if logits_s.numel() == 0:
        return {}
        
    # 1. Logit Margin (Primary Order Parameter)
    # distance between the true token logit and the best 'wrong' token logit
    true_logits = logits_s.gather(1, targets_s.unsqueeze(1)).squeeze(1)
    
    top2 = torch.topk(logits_s, k=2, dim=1).values
    # If the true token is the top-1, compare against top-2. Else compare against top-1.
    best_wrong = torch.where(
        top2[:, 0] == true_logits,
        top2[:, 1],
        top2[:, 0]
    )
    margin = (true_logits - best_wrong).mean()
    
    # 2. Top-K Structural Accuracy
    metrics = {"struct_margin": margin.item()}
    for k in topk:
        preds = torch.topk(logits_s, k=k, dim=1).indices
        correct = (preds == targets_s.unsqueeze(1)).any(dim=1)
        metrics[f"struct_top{k}_acc"] = correct.float().mean().item()
        
    # 3. Structural Entropy (Confidence level)
    probs = F.softmax(logits_s, dim=1)
    entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1).mean()
    metrics["struct_entropy"] = entropy.item()
    
    return metrics
