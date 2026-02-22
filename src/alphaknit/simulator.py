"""
ForwardSimulator: Converts a StitchGraph into a 3D mesh.

Strategy: Cylindrical coordinate layout.
  - Each stitch gets a 3D position based on its row (height) and column (angle).
  - radius(row) = stitch_count_at_row × stitch_width / (2π)
  - θ = 2π × column / stitch_count_at_row
  - y = row × stitch_height

This is deterministic and correct enough for training data generation.
90% of amigurumi shape is determined by stitch count distribution, not physics.
"""

import math
import numpy as np
from .stitch_graph import StitchGraph


class ForwardSimulator:

    def __init__(self, stitch_width: float = 0.5, stitch_height: float = 0.4):
        """
        Args:
            stitch_width:  Approximate width of one stitch (cm). Controls radius.
            stitch_height: Approximate height of one row (cm). Controls y-spacing.
        """
        self.stitch_width = stitch_width
        self.stitch_height = stitch_height

    # ------------------------------------------------------------------ #
    #  Core: compute 3D position for every stitch                         #
    # ------------------------------------------------------------------ #

    def compute_positions(self, graph: StitchGraph, jitter_ratio: float = 0.0) -> dict:
        """
        Assign a 3D (x, y, z) coordinate to each stitch node.
        
        Args:
            jitter_ratio: Magnitude of random noise (relative to stitch dimensions).
                          Used during training to break perfect axial symmetry.
                          Range typically 0.02 - 0.05.

        Returns:
            dict[stitch_id -> (x, y, z)]
        """
        counts = graph.stitch_count_per_row()
        positions = {}

        for sid, node in graph.nodes.items():
            row = node.row
            col = node.column
            row_count = counts.get(row, 1)

            # Radius from stitch count (circumference = N × w → r = N×w / 2π)
            radius = (row_count * self.stitch_width) / (2 * math.pi)

            # Angle: evenly distribute columns around the circle
            theta = (2 * math.pi * col) / row_count if row_count > 0 else 0
            
            # Apply Jitter (Phase 9A)
            # Break the perfect ring and row steps
            if jitter_ratio > 0:
                # Jitter radius (make rings imperfect)
                r_noise = np.random.uniform(-jitter_ratio, jitter_ratio)
                radius *= (1.0 + r_noise)
                
                # Jitter Y (make rows uneven)
                y_noise = np.random.uniform(-jitter_ratio, jitter_ratio) * self.stitch_height
                y = (row * self.stitch_height) + y_noise
                
                # Jitter Theta slightly (irregular stitch width)
                t_noise = np.random.uniform(-jitter_ratio, jitter_ratio) * (2 * math.pi / max(1, row_count))
                theta += t_noise
            else:
                y = row * self.stitch_height

            x = radius * math.cos(theta)
            z = radius * math.sin(theta)
            

            positions[sid] = (x, y, z)

        return positions
        
    def apply_laplacian_smoothing(self, mesh, iterations=3, lambda_val=0.5):
        """
        Apply Laplacian smoothing to the mesh for visualization.
        Does not change number of vertices/faces.
        """
        if mesh is None: return None
        try:
             import trimesh
             # Trimesh has built-in smoothing, or we implement simple version
             trimesh.smoothing.filter_laplacian(mesh, iterations=iterations, lamb=lambda_val)
             return mesh
        except Exception:
             return mesh # Fallback if fails


    # ------------------------------------------------------------------ #
    #  Point cloud (for PointNet / AI input)                              #
    # ------------------------------------------------------------------ #

    def to_point_cloud(self, graph: StitchGraph) -> np.ndarray:
        """
        Returns stitch positions as a numpy array of shape (N, 3),
        ordered by stitch id.
        """
        positions = self.compute_positions(graph)
        ids = sorted(positions.keys())
        return np.array([positions[sid] for sid in ids], dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  Mesh construction                                                   #
    # ------------------------------------------------------------------ #

    def build_mesh(self, graph: StitchGraph):
        """
        Build a triangulated surface mesh from the stitch graph.

        Strategy:
          - For each pair of consecutive rows, connect stitches into quads.
          - Handle inc (1 parent → 2 children) and dec (2 parents → 1 child)
            by splitting/merging quads.
          - Each quad is split into 2 triangles.

        Returns:
            trimesh.Trimesh  (or None if trimesh not installed)
        """
        try:
            import trimesh
        except ImportError:
            raise ImportError(
                "trimesh is required for mesh building. "
                "Install it with: pip install trimesh"
            )

        positions = self.compute_positions(graph)
        counts = graph.stitch_count_per_row()
        rows = sorted(counts.keys())

        if len(rows) < 2:
            # Only one row — return a degenerate mesh instead of PointCloud for API consistency
            verts = np.array(list(positions.values()), dtype=np.float64)
            return trimesh.Trimesh(vertices=verts, faces=[], process=False)

        vertices = []
        vert_index = {}  # stitch_id → vertex index

        for sid, pos in positions.items():
            vert_index[sid] = len(vertices)
            vertices.append(pos)

        vertices = np.array(vertices, dtype=np.float64)
        faces = []

        for i in range(len(rows) - 1):
            bottom_row = graph.get_row(rows[i])
            top_row = graph.get_row(rows[i + 1])

            # Build parent → children mapping for this row pair
            parent_to_children = {}
            for top_node in top_row:
                for pid in top_node.parents:
                    if pid not in parent_to_children:
                        parent_to_children[pid] = []
                    parent_to_children[pid].append(top_node.id)

            # Walk along bottom row and connect to top
            bottom_ids = [n.id for n in bottom_row]
            n_bottom = len(bottom_ids)

            for j, bot_id in enumerate(bottom_ids):
                next_bot_id = bottom_ids[(j + 1) % n_bottom]
                children = parent_to_children.get(bot_id, [])

                if len(children) == 0:
                    # Dropped stitch — skip
                    continue
                elif len(children) == 1:
                    # sc or dec-child: simple triangle or quad
                    top_id = children[0]
                    next_children = parent_to_children.get(next_bot_id, [])
                    if next_children:
                        next_top_id = next_children[0]
                        # Quad: bot → next_bot → next_top → top
                        b = vert_index[bot_id]
                        nb = vert_index[next_bot_id]
                        t = vert_index[top_id]
                        nt = vert_index[next_top_id]
                        faces.append([b, nb, nt])
                        faces.append([b, nt, t])
                    else:
                        # Triangle only
                        b = vert_index[bot_id]
                        nb = vert_index[next_bot_id]
                        t = vert_index[top_id]
                        faces.append([b, nb, t])
                elif len(children) == 2:
                    # inc: 1 parent → 2 children
                    top_id1, top_id2 = children[0], children[1]
                    b = vert_index[bot_id]
                    nb = vert_index[next_bot_id]
                    t1 = vert_index[top_id1]
                    t2 = vert_index[top_id2]
                    faces.append([b, t1, t2])
                    faces.append([b, nb, t2])

        if not faces:
            verts = np.array(list(positions.values()), dtype=np.float64)
            return trimesh.Trimesh(vertices=verts, faces=[], process=False)

        faces = np.array(faces, dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        return mesh

    # ------------------------------------------------------------------ #
    #  Export                                                              #
    # ------------------------------------------------------------------ #

    def export_obj(self, graph: StitchGraph, path: str):
        """Export mesh to .obj file."""
        mesh = self.build_mesh(graph)
        mesh.export(path)
        return path
