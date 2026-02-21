import torch
import torch.nn.functional as F
import numpy as np

class InterventionEngine:
    """
    v6.6-F: Causal Intervention Engine.
    Injects noise or clips rank in specific layers to test causal hypotheses.
    Uses PyTorch forward hooks for mechanical integration.
    """
    def __init__(self, model):
        self.model = model
        self.active_interventions = {} # {layer_name: {"type": str, "remaining": int, "handle": handle}}
        self.random_baseline_prob = 0.15

    def register_intervention(self, layer_name, type="noise", duration=5):
        # Clean up existing hook if present
        if layer_name in self.active_interventions:
            self._remove_hook(layer_name)
            
        handle = self._attach_hook(layer_name, type)
        self.active_interventions[layer_name] = {
            "type": type, 
            "remaining": duration,
            "handle": handle
        }
        print(f"🛠️ INTERVENTION: Registered {type} on {layer_name} for {duration} steps.")

    def _attach_hook(self, layer_name, intervention_type):
        """Finds the module and registers a forward hook."""
        target_module = None
        for name, module in self.model.named_modules():
            if name == layer_name:
                target_module = module
                break
        
        if target_module is None:
            print(f"⚠️ INTERVENTION WARNING: Layer {layer_name} not found.")
            return None
            
        return target_module.register_forward_hook(self.hook_fn)

    def _remove_hook(self, layer_name):
        """Removes the PyTorch hook handle."""
        data = self.active_interventions.get(layer_name)
        if data and data["handle"]:
            data["handle"].remove()
            print(f"🧹 INTERVENTION: Removed hook from {layer_name}")

    def apply(self, current_step):
        """
        Manages life cycle of interventions.
        """
        # 1. Random Intervention Baseline (Scientific Control)
        if np.random.random() < self.random_baseline_prob:
            # Gather candidate layers (Transformer layers are primary targets)
            layer_names = [n for n, _ in self.model.named_modules() if "transformer.layers" in n and "." not in n.replace("transformer.layers", "")]
            if layer_names:
                random_layer = np.random.choice(layer_names)
                if random_layer not in self.active_interventions:
                    self.register_intervention(random_layer, "noise", duration=1)

        # 2. Decay durations and clean up expired hooks
        to_remove = []
        for name, data in self.active_interventions.items():
            data["remaining"] -= 1
            if data["remaining"] <= 0:
                to_remove.append(name)
        
        for name in to_remove:
            self._remove_hook(name)
            self.active_interventions.pop(name, None)

    def hook_fn(self, module, input, output):
        """The actual perturbation logic."""
        # We compute noise without grad, but the addition must happen in-graph
        # to allow backprop to flow through the layer's inputs.
        if isinstance(output, tuple):
            h = output[0]
            with torch.no_grad():
                noise = torch.randn_like(h) * (h.std() + 1e-6) * 0.1
            perturbed = h + noise
            return (perturbed, *output[1:])
        else:
            with torch.no_grad():
                noise = torch.randn_like(output) * (output.std() + 1e-6) * 0.1
            return output + noise


class HypothesisEngine:
    """
    v6.6-F: Causal Falsification Engine.
    Automates the "Discovery" process by verifying persistence and invariance.
    """
    def __init__(self, n_seeds_target=5):
        self.hypotheses = []
        self.n_seeds_target = n_seeds_target
        self.persistence_window = 3
        self.causal_confidence = {}

    def propose(self, name, description, condition_fn):
        self.hypotheses.append({
            "name": name,
            "desc": description,
            "condition": condition_fn,
            "streak": 0,
            "status": "PROPOSED",
            "history": []
        })

    def update(self, metrics, epoch):
        report = []
        for h in self.hypotheses:
            # Type-safe streak access
            streak_raw = h.get("streak", 0)
            streak = int(streak_raw) if isinstance(streak_raw, (int, float)) else 0
            
            condition_fn = h.get("condition")
            is_met = False
            if callable(condition_fn):
                try:
                    is_met = condition_fn(metrics)
                except Exception:
                    is_met = False

            if is_met:
                streak += 1
                if streak >= self.persistence_window:
                    h["status"] = "VERIFIED"
            else:
                if h["status"] == "VERIFIED":
                    print(f"🔴 FALSIFIED: Discovery '{h['name']}' failed persistence check at epoch {epoch}.")
                    h["status"] = "FALSIFIED"
                streak = 0
            
            h["streak"] = streak
            h["history"].append(h["status"])
            name = str(h.get("name", "Unknown"))
            report.append(f"{name}: {h['status']} ({streak}/{self.persistence_window})")
        
        return report

    def monitor_failure(self, real_metrics, null_metrics):
        """
        Automated Hypothesis Rejection: If Null Mode (Placebo) shows stronger structural
        emergence than the real run, then the current hypothesis is invalid.
        """
        for h in self.hypotheses:
            if h["status"] == "VERIFIED":
                if null_metrics.get("struct_acc", 0) > real_metrics.get("struct_acc", 0) * 1.5:
                    print(f"⚠️ REJECTED: Discovery '{h['name']}' rejected by Null Control at epoch {real_metrics['epoch']}.")
                    h["status"] = "REJECTED_BY_CONTROL"

    def get_survival_map(self):
        return {h["name"]: h["status"] for h in self.hypotheses}


class NullEmergenceSuite:
    """
    v6.6-F: Scientific Control Suite.
    Manages "Placebo" training seeds (Random Labels, Noise Inputs, Geometry Null).
    """
    def __init__(self, mode="real"):
        self.mode = mode # "real", "random_labels", "noise_inputs", "geometry_null"

    def transform_batch(self, batch):
        if self.mode == "real":
            return batch
        
        if self.mode == "random_labels":
            # Semantic Breakdown
            batch['type_labels'] = batch['type_labels'][torch.randperm(batch['type_labels'].size(0))]
        
        if self.mode == "noise_inputs":
            # Structural Breakdown
            batch['point_cloud'] = batch['point_cloud'] + torch.randn_like(batch['point_cloud']) * 0.5
            
        return batch

    def apply_geometry_null(self, model):
        if self.mode == "geometry_null":
            print("🧱 GEOMETRY NULL: Scrambling layer connectivity...")
            # Simple version: randomize weights to break representational paths
            for p in model.parameters():
                if p.dim() >= 2:
                    torch.nn.init.orthogonal_(p)
