"""
GraphValidator: Checks structural invariants of a StitchGraph.
Returns a list of ValidationError objects.
"""

from .stitch_graph import StitchGraph


class ValidationError:
    def __init__(self, error_type, severity, stitch_id, message):
        self.error_type = error_type    # 'cycle', 'orphan', 'balance', 'locality'
        self.severity = severity        # 'critical', 'high', 'medium', 'low'
        self.stitch_id = stitch_id
        self.message = message

    def __repr__(self):
        return f"[{self.severity.upper()}] {self.error_type}: {self.message} (stitch {self.stitch_id})"


class GraphValidator:

    def validate(self, graph: StitchGraph):
        """Run all checks. Returns list of ValidationError."""
        errors = []
        errors.extend(self.check_acyclic(graph))
        errors.extend(self.check_no_orphans(graph))
        errors.extend(self.check_balance(graph))
        errors.extend(self.check_locality(graph))
        return errors

    def check_acyclic(self, graph: StitchGraph):
        """Detect cycles using DFS."""
        errors = []
        visited = set()
        in_stack = set()

        def dfs(sid):
            if sid in in_stack:
                errors.append(ValidationError(
                    'cycle', 'critical', sid,
                    f'Cycle detected involving stitch {sid}'
                ))
                return
            if sid in visited:
                return
            visited.add(sid)
            in_stack.add(sid)
            node = graph.nodes.get(sid)
            if node:
                for child_id in node.children:
                    dfs(child_id)
            in_stack.discard(sid)

        for sid in graph.nodes:
            if sid not in visited:
                dfs(sid)
        return errors

    def check_no_orphans(self, graph: StitchGraph):
        """Non-root stitches must have parents."""
        errors = []
        for sid, node in graph.nodes.items():
            if node.row > 0 and not node.parents:
                errors.append(ValidationError(
                    'orphan', 'critical', sid,
                    f'Stitch {sid} at row {node.row} has no parents'
                ))
        return errors

    def check_balance(self, graph: StitchGraph):
        """Verify that each row consumes exactly all parent slots from the previous row."""
        errors = []
        counts = graph.stitch_count_per_row()
        rows = sorted(counts.keys())

        for i in range(1, len(rows)):
            prev_row = rows[i - 1]
            curr_row = rows[i]
            prev_count = counts[prev_row]

            # Count consumed parent slots by collecting all parent references
            row_nodes = graph.get_row(curr_row)
            consumed_parents = set()
            for n in row_nodes:
                for pid in n.parents:
                    consumed_parents.add(pid)

            total_slots = len(consumed_parents)

            if total_slots != prev_count:
                errors.append(ValidationError(
                    'balance', 'high', -1,
                    f'Row {curr_row}: consumed {total_slots} parent slots '
                    f'but prev row has {prev_count} stitches'
                ))

        return errors

    def check_locality(self, graph: StitchGraph):
        """Decrease parents should be adjacent (column difference <= 1)."""
        errors = []
        for sid, node in graph.nodes.items():
            if node.stitch_type == 'dec' and len(node.parents) == 2:
                p1 = graph.nodes.get(node.parents[0])
                p2 = graph.nodes.get(node.parents[1])
                if p1 and p2 and abs(p1.column - p2.column) > 1:
                    errors.append(ValidationError(
                        'locality', 'low', sid,
                        f'Decrease stitch {sid} has non-adjacent parents '
                        f'(cols {p1.column} and {p2.column})'
                    ))
        return errors
