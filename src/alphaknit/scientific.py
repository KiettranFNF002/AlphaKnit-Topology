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
        self.active_interventions = {} 
        self.random_baseline_prob = 0.15
        self.shadow_mode = False # v6.6-F Level 3: Dual-pass counterfactual mode

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
        if self.shadow_mode:
            # In shadow mode, we don't apply noise (Identity pass)
            return output

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
    v6.6-F Level 2: Grounded in Optimizer Path Length (Energy) instead of Epoch count.
    """
    def __init__(self, persistence_threshold=5.0):
        self.hypotheses = []
        self.persistence_threshold = persistence_threshold # Cumulative ||delta theta||
        self.causal_confidence = {}

    def propose(self, name, description, condition_fn):
        self.hypotheses.append({
            "name": name,
            "desc": description,
            "condition": condition_fn,
            "path_traveled": 0.0,
            "status": "PROPOSED",
            "history": []
        })

    def update(self, metrics, delta_dist):
        """
        delta_dist: Optimizer distance traveled in this interval.
        """
        report = []
        for h in self.hypotheses:
            condition_fn = h.get("condition")
            is_met = False
            if callable(condition_fn):
                try:
                    is_met = condition_fn(metrics)
                except Exception:
                    is_met = False

            if is_met:
                h["path_traveled"] += delta_dist
                if h["path_traveled"] >= self.persistence_threshold:
                    if h["status"] != "VERIFIED":
                         print(f"✨ VERIFIED: Discovery '{h['name']}' met persistence threshold ({h['path_traveled']:.2f}/{self.persistence_threshold:.2f})")
                    h["status"] = "VERIFIED"
            else:
                if h["status"] == "VERIFIED":
                    print(f"🔴 FALSIFIED: Discovery '{h['name']}' failed at distance {h['path_traveled']:.4f}.")
                    h["status"] = "FALSIFIED"
                h["path_traveled"] = 0.0
            
            h["history"].append(h["status"])
            name = str(h.get("name", "Unknown"))
            report.append(f"{name}: {h['status']} ({h['path_traveled']:.2f}/{self.persistence_threshold:.2f})")
        
        return report

    def monitor_failure(self, real_metrics, null_metrics, shadow_delta=None):
        """
        v6.6-F Level 3: Adversarial Falsification.
        1. Rejection by Placebo: If Null Suite shows similar emergence.
        2. Rejection by Shadow: If intervention has no significant delta vs non-intervention.
        """
        for h in self.hypotheses:
            if h["status"] == "VERIFIED":
                # 1. Placebo Check
                if null_metrics.get("struct_acc", 0) > real_metrics.get("struct_acc", 0) * 1.5:
                    print(f"⚠️ REJECTED: Discovery '{h['name']}' rejected by Null Control at epoch {real_metrics.get('epoch', '?')}.")
                    h["status"] = "REJECTED_BY_CONTROL"
                
                # 2. Shadow Delta Check (Counterfactual)
                if shadow_delta is not None:
                     # If the delta (perturbation effect) is too small, it's not a causal link
                     if shadow_delta < 0.05: # Threshold for causal relevance
                         print(f"⚠️ REJECTED: Discovery '{h['name']}' rejected by Shadow Path (Delta={shadow_delta:.4f}).")
                         h["status"] = "FALSIFIED_BY_SHADOW"

    def get_survival_map(self):
        return {h["name"]: h.get("status", "Unknown") for h in self.hypotheses}


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
