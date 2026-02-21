import math
import numpy as np
import pytest
from src.alphaknit.compiler import KnittingCompiler
from src.alphaknit.simulator import ForwardSimulator

COMPILER = KnittingCompiler()
SIM = ForwardSimulator(stitch_width=0.5, stitch_height=0.4)


def compile_tokens(tokens):
    return COMPILER.compile(tokens)


# ============================================================
# Position tests
# ============================================================

class TestPositions:

    def test_mr6_positions_count(self):
        """mr_6 → 6 positions."""
        graph = compile_tokens(['mr_6'])
        pos = SIM.compute_positions(graph)
        assert len(pos) == 6

    def test_mr6_positions_on_circle(self):
        """mr_6 root stitches should all be at the same radius and y=0."""
        graph = compile_tokens(['mr_6'])
        pos = SIM.compute_positions(graph)
        for sid, (x, y, z) in pos.items():
            r = math.sqrt(x**2 + z**2)
            expected_r = (6 * 0.5) / (2 * math.pi)
            assert abs(r - expected_r) < 1e-6, f"Stitch {sid}: radius {r} != {expected_r}"
            assert y == pytest.approx(0.0)

    def test_maintain_row_same_radius(self):
        """sc row should have same radius as mr_6 row."""
        graph = compile_tokens(['mr_6'] + ['sc'] * 6)
        pos = SIM.compute_positions(graph)
        # Row 0 and row 1 both have 6 stitches → same radius
        row0_r = math.sqrt(pos[0][0]**2 + pos[0][2]**2)
        row1_r = math.sqrt(pos[6][0]**2 + pos[6][2]**2)
        assert abs(row0_r - row1_r) < 1e-6

    def test_expand_row_larger_radius(self):
        """inc row should have larger radius than mr_6 row."""
        graph = compile_tokens(['mr_6'] + ['inc'] * 6)
        pos = SIM.compute_positions(graph)
        row0_r = math.sqrt(pos[0][0]**2 + pos[0][2]**2)
        # Row 1 has 12 stitches → larger radius
        row1_node = next(n for n in graph.nodes.values() if n.row == 1)
        p = pos[row1_node.id]
        row1_r = math.sqrt(p[0]**2 + p[2]**2)
        assert row1_r > row0_r

    def test_contract_row_smaller_radius(self):
        """dec row should have smaller radius than inc row."""
        graph = compile_tokens(['mr_6'] + ['inc'] * 6 + ['dec'] * 6)
        pos = SIM.compute_positions(graph)
        # Row 1 has 12 stitches, row 2 has 6 → row2 radius < row1 radius
        row1_node = next(n for n in graph.nodes.values() if n.row == 1)
        row2_node = next(n for n in graph.nodes.values() if n.row == 2)
        r1 = math.sqrt(pos[row1_node.id][0]**2 + pos[row1_node.id][2]**2)
        r2 = math.sqrt(pos[row2_node.id][0]**2 + pos[row2_node.id][2]**2)
        assert r2 < r1

    def test_y_increases_with_row(self):
        """y coordinate should increase with row number."""
        graph = compile_tokens(['mr_6'] + ['sc'] * 6 + ['sc'] * 6)
        pos = SIM.compute_positions(graph)
        row0_y = pos[0][1]
        row1_node = next(n for n in graph.nodes.values() if n.row == 1)
        row2_node = next(n for n in graph.nodes.values() if n.row == 2)
        row1_y = pos[row1_node.id][1]
        row2_y = pos[row2_node.id][1]
        assert row0_y < row1_y < row2_y


# ============================================================
# Point cloud tests
# ============================================================

class TestPointCloud:

    def test_shape(self):
        """Point cloud shape should be (N, 3)."""
        graph = compile_tokens(['mr_6'] + ['sc'] * 6)
        pc = SIM.to_point_cloud(graph)
        assert pc.shape == (12, 3)
        assert pc.dtype == np.float32

    def test_all_finite(self):
        """No NaN or Inf in point cloud."""
        graph = compile_tokens(['mr_6'] + ['inc'] * 6 + ['sc', 'inc'] * 6)
        pc = SIM.to_point_cloud(graph)
        assert np.all(np.isfinite(pc))


# ============================================================
# Mesh tests
# ============================================================

class TestMesh:

    def test_mesh_builds(self):
        """Mesh should build without errors for a standard pattern."""
        graph = compile_tokens(['mr_6'] + ['inc'] * 6 + ['sc'] * 12)
        mesh = SIM.build_mesh(graph)
        assert mesh is not None

    def test_mesh_has_vertices(self):
        """Mesh should have the correct number of vertices."""
        graph = compile_tokens(['mr_6'] + ['sc'] * 6)
        mesh = SIM.build_mesh(graph)
        assert len(mesh.vertices) == 12

    def test_mesh_has_faces(self):
        """Mesh should have faces for a multi-row pattern."""
        graph = compile_tokens(['mr_6'] + ['sc'] * 6)
        mesh = SIM.build_mesh(graph)
        assert len(mesh.faces) > 0
