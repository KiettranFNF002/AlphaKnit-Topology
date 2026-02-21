import re

class AmigurumiStackParser:
    def __init__(self):
        # Atomic instructions
        self.instructions = {'sc', 'inc', 'dec', 'blo', 'flo', 'slst', 'hdc', 'dc', 'tr', 'mr'}
        
        # Normalization mapping
        self.replacements = [
            (r'(?i)\b(magic ring|mg ring)\b', 'mr'),
            (r'(?i)\b(single crochet|sc)\b', 'sc'),
            (r'(?i)\b(increase|inc|v|2\s*sc\s*in\s*next(?:\s*st)?)\b', 'inc'),
            (r'(?i)\b(decrease|dec|invdec|a|sc2tog)\b', 'dec'),
            (r'(?i)\b(back loop only|blo)\b', 'blo'),
            (r'(?i)\b(front loop only|flo)\b', 'flo'),
            (r'(?i)\b(slip stitch|slst|sl st|ss)\b', 'slst'),
            (r'(?i)\b(half double crochet|hdc)\b', 'hdc'),
            (r'(?i)\b(double crochet|dc)\b', 'dc'),
            (r'(?i)\b(treble crochet|tr)\b', 'tr'),
            (r'(?i)\b(chain|ch)\b', 'ch'),
        ]

    def normalize(self, text):
        """Standardizes terminology."""
        text = text.lower().strip()
        for pattern, replacement in self.replacements:
            text = re.sub(pattern, replacement, text)
        return text

    def tokenize(self, text):
        """
        Splits text into meaningful tokens for the stack parser.
        Example: "R1: (sc, inc) x 6" -> ['(', 'sc', ',', 'inc', ')', 'x', '6']
        """
        # Remove metadata like "R1:" or totals "(18)"
        text = re.sub(r'^(?:r|row|round|hàng)\s*\d+(?:-\d+)?\s*[:.]?\s*', '', text, flags=re.IGNORECASE)
        # Remove trailing totals inside () or []
        text = re.sub(r'[\(\[]\d+[\)\]]$', '', text) 
        
        # Normalize terms
        text = self.normalize(text)
        
        # Clean up syntax characters
        text = text.replace('*', '').replace('[', '(').replace(']', ')')
        
        # Tokenize: capturing parens, multipliers "x N", numbers, and words
        token_pattern = r'(\(|\)|x\s*\d+|\d+|[a-z_]+)'
        tokens = re.findall(token_pattern, text)
        
        # Filter out noise (commas, empty strings)
        return [t.strip() for t in tokens if t.strip() and t != ',' and t != 'st']

    def parse(self, text):
        """
        Main method: Parses a raw instruction string into a flat list of atomic actions.
        Handles nested loops like "((sc, inc) x 2, sc) x 2".
        """
        tokens = self.tokenize(text)
        
        stack = []          # Storage for incomplete groups
        current_group = []  # Currently active group of instructions
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # CASE 1: Start new group
            if token == '(':
                stack.append(current_group)
                current_group = []
                
            # CASE 2: End group
            elif token == ')':
                # Check for multiplier immediately following ')'
                multiplier = 1
                if i + 1 < len(tokens):
                    next_token = tokens[i+1]
                    # Format: "x 6" or "x6"
                    if next_token.startswith('x'):
                        try:
                            # Extract number from "x 6"
                            multiplier = int(re.search(r'\d+', next_token).group())
                            i += 1 # Skip multiplier token
                        except: pass
                    # Format: "(sc, inc) 6" (implicit multiplier)
                    elif next_token.isdigit(): 
                        multiplier = int(next_token)
                        i += 1

                # Expand the group
                expanded_group = current_group * multiplier
                
                # Merge back to parent group
                if stack:
                    parent_group = stack.pop()
                    parent_group.extend(expanded_group)
                    current_group = parent_group
                else:
                    # If unbalanced brackets (extra closing), just keep going
                    current_group = expanded_group

            # CASE 3: Atomic instructions (sc, inc...)
            elif token in self.instructions:
                current_group.append(token)

            # CASE 4: Numeric counts (e.g., "2 sc")
            elif token.isdigit():
                count = int(token)
                # Look ahead for instruction
                if i + 1 < len(tokens) and tokens[i+1] in self.instructions:
                    instruction = tokens[i+1]
                    current_group.extend([instruction] * count)
                    i += 1 # Skip instruction token
                else:
                    # Number without instruction might be garbage or noise, ignore
                    pass
            
            # CASE 5: "mr" handling (Magic Ring often written as "mr 6")
            elif token == 'mr':
                 # Check if number follows
                count = 6 # Default
                if i + 1 < len(tokens) and tokens[i+1].isdigit():
                     count = int(tokens[i+1])
                     i += 1
                # Treat MR as a special token usually at start
                current_group.append(f"mr_{count}")

            i += 1
            
        return current_group
