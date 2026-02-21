import torch
import torch.nn as nn
import numpy as np
import pytest
from alphaknit.scientific import InterventionEngine, HypothesisEngine, NullEmergenceSuite
from alphaknit.research import FeatureFingerprint
from alphaknit.model import KnittingTransformer

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 10)
        self.transformer = nn.Module()
        self.transformer.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])

    def forward(self, x):
        x = self.layer1(x)
        for layer in self.transformer.layers:
            x = layer(x)
        return x

def test_intervention_mechanical_hook():
    model = SimpleModel()
    engine = InterventionEngine(model)
    
    x = torch.randn(1, 10)
    
    # Baseline output
    torch.manual_seed(42)
    out_clean = model(x).detach()
    
    # Register intervention on a specific layer
    engine.register_intervention("transformer.layers.1", type="noise", duration=2)
    
    # Output with intervention
    torch.manual_seed(42)
    out_perturbed = model(x).detach()
    
    # Verify the intervention changed the output
    assert not torch.allclose(out_clean, out_perturbed), "Intervention should change output"
    
    # Step engine once - should still be active (duration was 2, now 1 remaining)
    engine.apply(0)
    torch.manual_seed(42)
    out_perturbed_2 = model(x).detach()
    assert not torch.allclose(out_clean, out_perturbed_2), "Intervention should still be active"
    
    # Step engine again - should be removed (duration reaches 0)
    engine.apply(1)
    torch.manual_seed(42)
    out_cleaned = model(x).detach()
    assert torch.allclose(out_clean, out_cleaned, atol=1e-6), "Output should return to baseline after hook removal"

def test_null_emergence_suite():
    suite = NullEmergenceSuite(mode="noise_inputs")
    points_orig = torch.ones(2, 10, 3)
    batch = {
        'point_cloud': points_orig.clone()
    }
    transformed = suite.transform_batch(batch)
    assert not torch.allclose(points_orig, transformed['point_cloud']), "Noise inputs should perturb points"

def test_hypothesis_engine_persistence():
    # v6.6-F Level 2: Grounded in distance
    engine = HypothesisEngine(persistence_threshold=1.5)
    engine.propose("Test_Hypo", "Description", lambda m: m["acc"] > 0.8)
    
    # Distance 1.0: Met but not threshold (1.5)
    engine.update({"acc": 0.9}, 1.0)
    assert engine.hypotheses[0]["status"] == "PROPOSED"
    assert engine.hypotheses[0]["path_traveled"] == 1.0
    
    # Distance +1.0: Met -> VERIFIED (total 2.0 > 1.5)
    engine.update({"acc": 0.9}, 1.0)
    assert engine.hypotheses[0]["path_traveled"] == 2.0
    assert engine.hypotheses[0]["status"] == "VERIFIED"
    
    # Not met -> FALSIFIED & path reset
    engine.update({"acc": 0.5}, 0.5)
    assert engine.hypotheses[0]["status"] == "FALSIFIED"
    assert engine.hypotheses[0]["path_traveled"] == 0.0

def test_hypothesis_engine_shadow_rejection():
    engine = HypothesisEngine(persistence_threshold=1.0)
    engine.propose("Causal_Hypo", "Desc", lambda m: m["acc"] > 0.8)
    engine.update({"acc": 0.9}, 2.0)
    assert engine.hypotheses[0]["status"] == "VERIFIED"
    
    # Shadow Delta too small -> Rejection
    engine.monitor_failure({"acc": 0.9}, {"struct_acc": 0.0}, shadow_delta=0.01)
    assert engine.hypotheses[0]["status"] == "FALSIFIED_BY_SHADOW"

def test_feature_fingerprint_stability():
    fingerprint = FeatureFingerprint(top_k=2)
    
    # Hidden states: [B, T, D]
    h1 = torch.ones(2, 4, 10)
    h1[0, 0, 0] = 2.0 # Add some variance
    
    s1 = fingerprint.update(h1)
    assert s1 == 1.0 # First check is baseline
    
    # Same state -> High stability
    s2 = fingerprint.update(h1)
    assert s2 > 0.99
    
    # Orthogonal shift -> Low stability
    h2 = torch.randn(2, 4, 10)
    s3 = fingerprint.update(h2)
    assert s3 < 0.5 or s3 < s2
