import json
import os

class AmigurumiTokenizer:
    def __init__(self, vocab_file=None):
        self.special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        
        # Standard crochet instruction tokens
        self.stitch_types = [
            "mr_4", "mr_5", "mr_6", "mr_8", # Common Magic Rings
            "sc", "inc", "dec", 
            "blo", "flo", "slst", 
            "ch", "hdc", "dc", "tr",
            "inc_hdc", "dec_hdc", 
            "inc_dc", "dec_dc"
        ]
        
        # Build vocabulary
        self.vocab = {token: i for i, token in enumerate(self.special_tokens + self.stitch_types)}
        self.idx_to_token = {i: token for token, i in self.vocab.items()}
        
        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)

    def encode(self, token_list):
        """
        Converts a list of tokens (strings) to a list of IDs (integers).
        Adds <SOS> at the start and <EOS> at the end.
        """
        indices = [self.vocab["<SOS>"]]
        for token in token_list:
             # Handle tokens not in vocab -> UNK
            idx = self.vocab.get(token, self.vocab["<UNK>"])
            indices.append(idx)
        indices.append(self.vocab["<EOS>"])
        return indices

    def decode(self, indices):
        """
        Converts a list of IDs back to tokens.
        Skips <SOS>, <EOS>, <PAD>.
        """
        tokens = []
        for idx in indices:
            token = self.idx_to_token.get(str(idx)) # Handle string/int keys if loaded from json
            if not token:
                 token = self.idx_to_token.get(int(idx), "<UNK>")
            
            if token in ["<SOS>", "<EOS>", "<PAD>"]:
                continue
            tokens.append(token)
        return tokens

    def save_vocab(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.vocab, f, indent=2)

    def load_vocab(self, filepath):
        with open(filepath, 'r') as f:
            self.vocab = json.load(f)
        self.idx_to_token = {i: token for token, i in self.vocab.items()}
