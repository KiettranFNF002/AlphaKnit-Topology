import pytest
import os
from src.alphaknit.tokenizer import AmigurumiTokenizer

def test_tokenizer_encoding():
    tokenizer = AmigurumiTokenizer()
    tokens = ["sc", "inc", "dec"]
    ids = tokenizer.encode(tokens)
    
    # Should start with SOS and end with EOS
    assert ids[0] == tokenizer.vocab["<SOS>"]
    assert ids[-1] == tokenizer.vocab["<EOS>"]
    assert len(ids) == 5 # SOS + 3 tokens + EOS

def test_tokenizer_decoding():
    tokenizer = AmigurumiTokenizer()
    tokens = ["sc", "inc"]
    ids = tokenizer.encode(tokens)
    decoded = tokenizer.decode(ids)
    
    assert decoded == tokens # SOS/EOS are stripped

def test_tokenizer_unk_token():
    tokenizer = AmigurumiTokenizer()
    tokens = ["sc", "random_stitch", "inc"]
    ids = tokenizer.encode(tokens)
    
    unk_id = tokenizer.vocab["<UNK>"]
    assert ids[2] == unk_id # 'random_stitch' should be UNK

def test_vocabulary_save_load(tmp_path):
    tokenizer = AmigurumiTokenizer()
    vocab_path = tmp_path / "vocab.json"
    
    tokenizer.save_vocab(str(vocab_path))
    assert os.path.exists(vocab_path)
    
    new_tokenizer = AmigurumiTokenizer(vocab_file=str(vocab_path))
    assert new_tokenizer.vocab == tokenizer.vocab
