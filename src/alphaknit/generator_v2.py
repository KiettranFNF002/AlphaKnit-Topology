import random
import math
from .stitch_graph import StitchGraph

class SpatialGeneratorV2:
    """
    Phase 9B: Edge-Action Generator
    Constructs a StitchGraph DAG causally without a 1D token intermediary.
    Output: 
      - The StitchGraph DAG
      - sequence of tuples: (node_type_id, p1_offset, p2_offset)
    """

    # Vocabulary assignments for Node Types (match config VOCAB roughly, but distinct for graph)
    TYPES = {
        "mr_6": 4,
        "sc": 5,
        "inc": 6,
        "dec": 7
    }

    def __init__(self, min_rows=5, max_rows=30, max_width=60, num_sectors=8):
        self.min_rows = min_rows
        self.max_rows = max_rows
        self.max_width = max_width
        self.num_sectors = num_sectors

    def generate_pattern(self):
        """
        Build a random amigurumi shape robustly.
        Returns:
            graph: StitchGraph object
            edge_sequence: list of (type_id, p1_offset, p2_offset) tuples
            is_closed: bool
            metadata: row shape metadata
        """
        graph = StitchGraph()
        edge_sequence = []
        row_data = []

        # ------------------------------------------------------------- #
        # Row 0: Magic Ring
        # ------------------------------------------------------------- #
        current_layer_nodes = []
        # mr_6 creates 6 root nodes
        for c in range(6):
            sector = int((c / 6) * self.num_sectors)
            node = graph.add_stitch(
                stitch_type='mr', parents=[], row=0, column=c, 
                layer_id=0, theta_sector=sector, ring_size_at_creation=6
            )
            current_layer_nodes.append(node.id)
        
        # We append a singular MR action (p1=0, p2=0 for no parents)
        edge_sequence.append((self.TYPES["mr_6"], 0, 0))
        
        current_circumference = 6
        total_rows = random.randint(self.min_rows, self.max_rows)
        is_closed = False

        for row in range(1, total_rows + 1):
            action = self._choose_action(current_circumference, row, total_rows)
            
            parent_ids = current_layer_nodes
            parent_count = len(parent_ids)
            
            new_layer_nodes = []
            
            # Phase 9C: Topology Noise
            # Delayed increases: occasionally shift increases from start of sequence
            noise_delay = random.randint(0, 2) if action == "expand" else 0
            
            p_idx = 0 # pointer in parent_ids
            col = 0
            
            while p_idx < parent_count:
                remaining_parents = parent_count - p_idx
                
                # Determine next stitch type
                stitch_type = 'sc'
                parents_to_consume = []
                
                p1_offset = 0
                p2_offset = 0

                # Action specific logic
                if action == "expand":
                    # We want to add 6 incs evenly spaced
                    spacing = parent_count // 6
                    # Use delayed noise if applicable
                    if (p_idx + noise_delay) % max(1, spacing) == 0 and remaining_parents >= 1 and (current_circumference - parent_count < 6):
                        stitch_type = 'inc'
                        parents_to_consume = [parent_ids[p_idx]]
                        # Parent offset: id distance (not index distance, but absolute id)
                        # We use index-based offset relative to the current generation step to keep vocab bounded.
                        # Wait, absolute ID offset is better: current_id - parent_id
                        p1_offset = graph._next_id - parents_to_consume[0]
                        
                        # Inc creates 2 children. We add them sequentially.
                        sector = int(((p_idx + 0.5) / parent_count) * self.num_sectors)
                        # Optional: sector jitter
                        sector = (sector + random.choice([-1, 0, 1])) % self.num_sectors

                        node1 = graph.add_stitch('inc', parents_to_consume, row=row, column=col,
                                                layer_id=row, theta_sector=sector, ring_size_at_creation=current_circumference)
                        node2 = graph.add_stitch('inc', parents_to_consume, row=row, column=col+1,
                                                layer_id=row, theta_sector=sector, ring_size_at_creation=current_circumference)
                        
                        new_layer_nodes.extend([node1.id, node2.id])
                        edge_sequence.append((self.TYPES["inc"], p1_offset, 0)) # First child
                        edge_sequence.append((self.TYPES["inc"], graph._next_id - 1 - parents_to_consume[0], 0)) # Second child
                        col += 2
                        p_idx += 1
                        current_circumference += 1
                        continue

                elif action == "contract":
                    spacing = parent_count // 6
                    if p_idx % max(1, spacing) == 0 and remaining_parents >= 2 and (parent_count - current_circumference < 6):
                        stitch_type = 'dec'
                        parents_to_consume = [parent_ids[p_idx], parent_ids[p_idx+1]]
                        
                        p1_offset = graph._next_id - parents_to_consume[0]
                        p2_offset = graph._next_id - parents_to_consume[1]
                        
                        sector = int(((p_idx + 0.5) / parent_count) * self.num_sectors)
                        
                        node = graph.add_stitch('dec', parents_to_consume, row=row, column=col,
                                                layer_id=row, theta_sector=sector, ring_size_at_creation=current_circumference)
                        
                        new_layer_nodes.append(node.id)
                        edge_sequence.append((self.TYPES["dec"], p1_offset, p2_offset))
                        col += 1
                        p_idx += 2
                        current_circumference -= 1
                        continue

                # Default fallback: SC
                stitch_type = 'sc'
                parents_to_consume = [parent_ids[p_idx]]
                p1_offset = graph._next_id - parents_to_consume[0]
                
                sector = int(((p_idx + 0.5) / parent_count) * self.num_sectors)
                node = graph.add_stitch('sc', parents_to_consume, row=row, column=col,
                                        layer_id=row, theta_sector=sector, ring_size_at_creation=current_circumference)
                
                new_layer_nodes.append(node.id)
                edge_sequence.append((self.TYPES["sc"], p1_offset, 0))
                col += 1
                p_idx += 1

            current_layer_nodes = new_layer_nodes
            
            row_data.append({
                "row": row,
                "action": action,
                "stitch_count": current_circumference,
            })
            
            if current_circumference <= 3: # Effectively closed
                is_closed = True
                break

        return {
            "edge_sequence": edge_sequence, # list of tuples
            "metadata": row_data,
            "is_closed": is_closed,
            "stitch_graph": graph,
            "flat_sequence": [] # deprecate, but keep key for compatibility
        }

    def _choose_action(self, current_circumference, row, total_rows):
        choices = []
        if current_circumference < self.max_width:
            choices.append("expand")
        choices.extend(["maintain"] * 2) 
        if current_circumference > 6:
            choices.append("contract")
            
        action = random.choice(choices)
        
        # Force contract at end to close shape
        if row > total_rows - 3 and current_circumference > 12:
            action = "contract"
        if row == total_rows and current_circumference > 3:
             action = "contract"
             
        return action
