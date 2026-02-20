"""
StitchGraph: DAG representation of a crocheted fabric.
Each stitch is a node; parent edges represent which stitches
the new stitch was worked into.

Phase 9B/C additions:
- layer_id: explicit round index
- theta_sector: angular grouping (0-7 for 8 sectors)
- ring_size_at_creation: circumference context for this stitch
"""

class StitchNode:
    __slots__ = ('id', 'stitch_type', 'parents', 'children', 'row', 'column', 'layer_id', 'theta_sector', 'ring_size_at_creation')

    def __init__(self, id, stitch_type, parents=None, row=0, column=0, layer_id=0, theta_sector=0, ring_size_at_creation=0):
        self.id = id
        self.stitch_type = stitch_type
        self.parents = parents or []
        self.children = []
        self.row = row
        self.column = column
        
        # Phase 9B/C topological properties
        self.layer_id = layer_id
        self.theta_sector = theta_sector
        self.ring_size_at_creation = ring_size_at_creation

    def to_dict(self):
        return {
            'id': self.id,
            'stitch_type': self.stitch_type,
            'parents': self.parents,
            'children': self.children,
            'row': self.row,
            'column': self.column,
            'layer_id': self.layer_id,
            'theta_sector': self.theta_sector,
            'ring_size_at_creation': self.ring_size_at_creation,
        }

    def __repr__(self):
        return f"Stitch({self.id}, {self.stitch_type}, row={self.row}, parents={self.parents})"


class StitchGraph:
    """Directed Acyclic Graph of stitches."""

    def __init__(self):
        self.nodes = {}       # id -> StitchNode
        self._next_id = 0

    def add_stitch(self, stitch_type, parents=None, row=0, column=0, layer_id=0, theta_sector=0, ring_size_at_creation=0):
        """Create a new stitch node linked to its parents."""
        sid = self._next_id
        self._next_id += 1

        parent_ids = [p if isinstance(p, int) else p.id for p in (parents or [])]
        node = StitchNode(sid, stitch_type, parent_ids, row, column, layer_id, theta_sector, ring_size_at_creation)
        self.nodes[sid] = node

        # Back-link children
        for pid in parent_ids:
            if pid in self.nodes:
                self.nodes[pid].children.append(sid)

        return node

    # ---- Queries ----

    def get_row(self, row_num):
        """Return all stitches in a given row, sorted by column."""
        return sorted(
            [n for n in self.nodes.values() if n.row == row_num],
            key=lambda n: n.column
        )

    def stitch_count_per_row(self):
        """Return dict {row_number: stitch_count}."""
        counts = {}
        for n in self.nodes.values():
            counts[n.row] = counts.get(n.row, 0) + 1
        return dict(sorted(counts.items()))

    def root_nodes(self):
        """Stitches with no parents (cast-on / magic ring)."""
        return [n for n in self.nodes.values() if not n.parents]

    def leaf_nodes(self):
        """Stitches with no children (last row)."""
        return [n for n in self.nodes.values() if not n.children]

    @property
    def size(self):
        return len(self.nodes)

    # ---- Serialization ----

    def to_dict(self):
        return {
            'nodes': {sid: node.to_dict() for sid, node in self.nodes.items()},
            'size': self.size,
        }

    @classmethod
    def from_dict(cls, data):
        graph = cls()
        for sid_str, ndata in data['nodes'].items():
            sid = int(sid_str)
            node = StitchNode(
                sid, ndata['stitch_type'], ndata['parents'],
                ndata.get('row', 0), ndata.get('column', 0),
                ndata.get('layer_id', 0), ndata.get('theta_sector', 0),
                ndata.get('ring_size_at_creation', 0)
            )
            node.children = ndata.get('children', [])
            graph.nodes[sid] = node
            graph._next_id = max(graph._next_id, sid + 1)
        return graph
