import random
from .compiler import KnittingCompiler

class AmigurumiGenerator:
    def __init__(self, min_rows=5, max_rows=30, max_width=60):
        self.min_rows = min_rows
        self.max_rows = max_rows
        self.max_width = max_width
        
        self.tokens = {
            "START": "<SOS>",
            "END": "<EOS>",
            "MR": "mr_6",
            "SC": "sc",
            "INC": "inc",
            "DEC": "dec"
        }

    def generate_layer_action(self, current_stitches):
        choices = []
        # Expand: only if not too wide
        if current_stitches < self.max_width:
            choices.append("expand")
            
        # Maintain: always posisble
        choices.extend(["maintain"] * 2) 
        
        # Contract: only if > 6
        if current_stitches > 6:
            choices.append("contract")
            
        return random.choice(choices)

    def generate_pattern(self):
        # Start with Magic Ring 6
        pattern_sequence = [self.tokens["MR"]]
        current_stitches = 6 
        row_data = [] 
        
        total_rows = random.randint(self.min_rows, self.max_rows)
        
        # Track simulated loop closure
        is_closed = False

        for row in range(1, total_rows + 1):
            action = self.generate_layer_action(current_stitches)
            
            # Force contract at end to close shape
            if row > total_rows - 3 and current_stitches > 12:
                action = "contract"
            if row == total_rows and current_stitches > 0:
                 action = "contract" # Try to close at last row

            layer_tokens = []
            
            if action == "expand":
                sc_count = (current_stitches // 6) - 1
                segment = []
                if sc_count > 0:
                     segment.extend([self.tokens["SC"]] * sc_count)
                segment.append(self.tokens["INC"])
                
                layer_tokens = segment * 6
                current_stitches += 6
                
            elif action == "maintain":
                layer_tokens = [self.tokens["SC"]] * current_stitches
                
            elif action == "contract":
                sc_count = (current_stitches // 6) - 2
                
                segment = []
                if sc_count < 0:
                    # current_stitches == 6: 3 decs consume all 6 slots → 3 stitches
                    segment = [self.tokens["DEC"]] * 3
                    current_stitches = 3
                else:
                    if sc_count > 0:
                        segment.extend([self.tokens["SC"]] * sc_count)
                    segment.append(self.tokens["DEC"])
                    current_stitches -= 6
                
                layer_tokens = segment * 6 if sc_count >= 0 else segment

            pattern_sequence.extend(layer_tokens)
            
            row_data.append({
                "row": row,
                "action": action,
                "stitch_count": current_stitches,
            })
            
            if current_stitches == 0:
                is_closed = True
                break

        # Compile to StitchGraph
        compiler = KnittingCompiler()
        stitch_graph = compiler.compile(pattern_sequence)

        return {
            "flat_sequence": pattern_sequence,
            "metadata": row_data,
            "is_closed": is_closed,
            "stitch_graph": stitch_graph,
        }

