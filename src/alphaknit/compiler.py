from __future__ import annotations

import re
from .stitch_graph import StitchGraph


class CompileError(Exception):
    pass


class KnittingCompiler:
    # STITCH_OUTPUT removed since Phase 9B has 1:1 token-node mapping.

    TOKEN_PATTERN = re.compile(r'^(?P<type>[a-zA-Z_0-9]+)\((?P<p1>-?\d+),(?P<p2>-?\d+)\)$')

    def _parse_token(self, token: str, index: int):
        """
        Parse token of form 'type(p1,p2)' into (type, p1, p2).
        Raises CompileError on malformed tokens.
        """
        match = self.TOKEN_PATTERN.match(token)
        if not match:
            raise CompileError(f"Invalid token syntax: '{token}' at position {index}")
        node_type = match.group('type')
        try:
            p1 = int(match.group('p1'))
            p2 = int(match.group('p2'))
        except ValueError:
            raise CompileError(f"Invalid parent offsets in token: '{token}' at position {index}")
        return node_type, p1, p2

    def compile(self, tokens):
        """
        Compile a list of stitch tokens with explicit parent offsets into a StitchGraph.

        Args:
            tokens: list of strings, e.g. ['mr_6(0,0)', 'sc(1,0)', 'inc(2,0)']
        Returns:
            StitchGraph
        """
        graph = StitchGraph()
        stitches_history = []

        for i, raw_token in enumerate(tokens):
            node_type, p1, p2 = self._parse_token(raw_token, i)

            # --- Magic Ring ---
            if node_type.startswith('mr_'):
                try:
                    count = int(node_type.split('_')[1])
                except (IndexError, ValueError):
                    raise CompileError(f"Invalid magic ring token: '{raw_token}' at position {i}")

                for _ in range(count):
                    node = graph.add_stitch('mr', parents=[])
                    stitches_history.append(node.id)
                continue

            # --- Regular stitches ---
            parents = []
            if p1 > 0:
                p1 = max(1, min(p1, len(stitches_history)))
                parents.append(stitches_history[-p1])
            if p2 > 0:
                p2 = max(1, min(p2, len(stitches_history)))
                parents.append(stitches_history[-p2])

            # Every non-MR token represents exactly 1 node
            node = graph.add_stitch(node_type, parents=parents)
            stitches_history.append(node.id)

        return graph
