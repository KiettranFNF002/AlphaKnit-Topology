import re
from .config import ID_TO_TOKEN, VOCAB, SOS_ID, EOS_ID, PAD_ID
from .compiler import KnittingCompiler

class PatternFormatter:
    """
    Converts a flat sequence of tokens into a human-readable crochet pattern.
    
    Format:
    R1: MR 6              (6)
    R2: inc x 6           (12)
    R3: (sc, inc) x 6     (18)
    ...
    """
    def __init__(self):
        self.compiler = KnittingCompiler()
        
    def format_ids(self, token_ids: list) -> str:
        """Entry point for token IDs (e.g. from model output)."""
        # 1. Convert IDs to strings
        tokens = []
        for tid in token_ids:
            if tid in (SOS_ID, EOS_ID, PAD_ID):
                continue
            tokens.append(ID_TO_TOKEN.get(tid, f"unk_{tid}"))
            
        return self.format_tokens(tokens)

    def format_tokens(self, tokens: list) -> str:
        """
        Main formatting logic.
        Uses KnittingCompiler logic to detect row boundaries.
        """
        if not tokens:
            return "Empty pattern."

        # Re-run simulation to get exact row breaks
        # We duplicate logic from KnittingCompiler but track the row number
        
        # Or simpler: Split by logic.
        # But wait, our Compiler doesn't save "Row X starts at index Y". 
        # It calculates parent/child slots.
        # So we MUST run the row simulation to know where lines break.
        
        rows = []
        current_row_tokens = []
        
        # Simulation state
        # Initial row is usually MR 6
        # Compiler logic:
        # parent_slots starts at 0? No, usually infinite for initial chain, but for MR start:
        # MR 6 -> produces 6 loops. End of row.
        
        # Let's use the exact logic from _is_prefix_viable (Simulator) 
        # to track row boundaries.
        
        # Standard Amigurumi logic:
        # START
        # R1: mr_6 -> 6 child loops. 
        # New Row: parent_slots = 6. 
        # ... consume parent_slots ... 
        # parent_slots == 0 -> End Row.
        
        parent_slots = 0
        child_slots = 0
        
        # Check start
        if tokens[0] == "mr_6":
            # R1 is strictly "mr_6"
            rows.append(["mr_6"])
            parent_slots = 6 # R1 produces 6 loops for R2
            tokens = tokens[1:]
        else:
            # Maybe standard chain start? Not supported by our model yet primarily.
            # Assume implicit start? No, model always outputs mr_6.
            # If missing, just treat as continuous bag of stitches
            parent_slots = 999 
        
        current_row = []
        
        for t in tokens:
            current_row.append(t)
            
            consumed = 0
            produced = 0
            
            if t == "sc":
                consumed = 1
                produced = 1
            elif t == "inc":
                consumed = 1
                produced = 2
            elif t == "dec":
                consumed = 2
                produced = 1
            elif t == "mr_6": 
                # Should not happen mid-stream usually
                consumed = 0
                produced = 6
                
            parent_slots -= consumed
            child_slots += produced
            
            if parent_slots <= 0:
                # Row done
                rows.append(current_row)
                current_row = []
                parent_slots = child_slots
                child_slots = 0
                
        # Append leftovers
        if current_row:
            rows.append(current_row)
            
        # Format each row
        lines = []
        total_stitches = 0
        
        for r_idx, row_tokens in enumerate(rows):
            row_num = r_idx + 1
            
            # Calculate stitch count (last parent_slots of this row = child_slots of prev)
            # Actually, child_slots of *this* row is the stitch count.
            # Let's recompute simply
            count = 0
            for t in row_tokens:
                if t == "mr_6": count += 6
                elif t == "sc": count += 1
                elif t == "inc": count += 2
                elif t == "dec": count += 1
            
            # Compress repeats
            instruction = self._compress_row(row_tokens)
            
            lines.append(f"**R{row_num}:** {instruction} ({count})")
            
        return "\n\n".join(lines)

    def _compress_row(self, tokens: list) -> str:
        """
        Compress sequences like 'sc sc sc inc sc sc sc inc' -> '(sc, sc, sc, inc) x 2'?
        Or 'sc x 6'.
        """
        if not tokens: return ""
        if len(tokens) == 1: return tokens[0]
        
        # 1. Simple Run Compression: "sc sc sc" -> "sc x 3"
        # 2. Pattern Compression: "sc inc sc inc" -> "(sc, inc) x 2"
        
        # Simple RLE first
        rle = []
        if not tokens: return ""
        
        current = tokens[0]
        count = 1
        for t in tokens[1:]:
            if t == current:
                count += 1
            else:
                rle.append((current, count))
                current = t
                count = 1
        rle.append((current, count))
        
        # Convert RLE to string parts
        parts = []
        for cmd, cnt in rle:
            if cnt > 1:
                parts.append(f"{cmd} x {cnt}")
            else:
                parts.append(cmd)
                
        # Detect repeating patterns in parts? 
        # E.g. "sc", "inc", "sc", "inc" -> "(sc, inc) x 2"
        # This is a bit complex for a simple formatter, RLE is a good 80% solution.
        # Let's try to detect (sc, inc) x N specifically as it is very common.
        
        final_str = ", ".join(parts)
        
        # Heuristic for standard expansion: (sc x N, inc) x 6
        # If the whole row is just repeats of something
        
        return final_str

