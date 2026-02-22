from __future__ import annotations

import re
from .stitch_graph import StitchGraph


class CompileError(Exception):
    pass


class KnittingCompiler:
    # STITCH_OUTPUT removed since Phase 9B has 1:1 token-node mapping.

    TOKEN_PATTERN = re.compile(r'^(?P<type>[a-zA-Z_0-9]+)(\((?P<p1>-?\d+),(?P<p2>-?\d+)\))?$')

    def _parse_token(self, token: str, index: int):
        """
        Parse token of form 'type(p1,p2)' into (type, p1, p2).
        Supports shorthand 'type' (defaults to p1=0, p2=0).
        Raises CompileError on malformed tokens.
        """
        match = self.TOKEN_PATTERN.match(token)
        if not match:
            raise CompileError(f"Invalid token syntax: '{token}' at position {index}")
        
        node_type = match.group('type')
        p1_val = match.group('p1')
        p2_val = match.group('p2')
        
        try:
            # v6.6-G: Default to (1,0) for connectivity if not MR and not specified
            is_mr = node_type.startswith('mr_')
            default_p1 = 0 if is_mr else 1
            
            p1 = int(p1_val) if p1_val is not None else default_p1
            p2 = int(p2_val) if p2_val is not None else 0
        except ValueError:
             raise CompileError(f"Invalid parent offsets in token: '{token}' at position {index}")
             
        return node_type, p1, p2

    def compile(self, tokens):
        """
        Compile a list of stitch tokens into a StitchGraph.
        Supports both legacy shorthand (sc, inc, dec) and mechanistic type(p1,p2).
        """
        graph = StitchGraph()
        stitches_history = []
        
        parent_pool = []
        child_pool = []
        current_row = 0
        
        for i, raw_token in enumerate(tokens):
            node_type, p1, p2 = self._parse_token(raw_token, i)
            
            # --- 1. Magic Ring (Row 0 Start) ---
            if node_type.startswith('mr_'):
                try:
                    count = int(node_type.split('_')[1])
                except (IndexError, ValueError):
                    raise CompileError(f"Invalid magic ring token: '{raw_token}'")
                
                # MR resets the structure to a new base
                for c in range(count):
                    node = graph.add_stitch('mr', parents=[], row=0, column=c)
                    stitches_history.append(node.id)
                    parent_pool.append(node.id)
                current_row = 1
                child_pool = []
                continue

            # --- 2. Mechanistic (Explicit Parents) ---
            if '(' in raw_token:
                parents = []
                if p1 > 0 and stitches_history:
                    idx1 = max(1, min(p1, len(stitches_history)))
                    parents.append(stitches_history[-idx1])
                if p2 > 0 and stitches_history:
                    idx2 = max(1, min(p2, len(stitches_history)))
                    parents.append(stitches_history[-idx2])
                
                # Use depth logic for row
                row = 0
                if parents:
                    row = max(graph.nodes[pid].row for pid in parents) + 1
                col = len([n for n in graph.nodes.values() if n.row == row])
                
                node = graph.add_stitch(node_type, parents=parents, row=row, column=col)
                stitches_history.append(node.id)
                child_pool.append(node.id)
                continue

            # --- 3. Legacy Shorthand (Implicit Parents) ---
            # Advance to next row if parent pool is exhausted
            if not parent_pool:
                if child_pool:
                    parent_pool = child_pool[:]
                    child_pool = []
                    current_row += 1
            
            if node_type == 'sc':
                pid = parent_pool.pop(0) if parent_pool else None
                node = graph.add_stitch('sc', parents=[pid] if pid is not None else [], 
                                        row=current_row, column=len(child_pool))
                stitches_history.append(node.id)
                child_pool.append(node.id)
            elif node_type == 'inc':
                # inc (increase) worked into one parent, producing 2 stitches
                pid = parent_pool.pop(0) if parent_pool else None
                for _ in range(2):
                    node = graph.add_stitch('sc', parents=[pid] if pid is not None else [], 
                                            row=current_row, column=len(child_pool))
                    stitches_history.append(node.id)
                    child_pool.append(node.id)
            elif node_type == 'dec':
                # dec (decrease) worked into two parents, producing 1 stitch
                pids = []
                if parent_pool: pids.append(parent_pool.pop(0))
                if parent_pool: pids.append(parent_pool.pop(0))
                node = graph.add_stitch('dec', parents=pids, 
                                        row=current_row, column=len(child_pool))
                stitches_history.append(node.id)
                child_pool.append(node.id)
            elif node_type == 'mr' or node_type.startswith('mr_'):
                # Handled above, but if it appears again or without _, we might need caution
                # For now, just allow it if it matched above.
                pass
            else:
                raise CompileError(f"Unknown shorthand token type: '{node_type}' at position {i}")
                
        return graph
