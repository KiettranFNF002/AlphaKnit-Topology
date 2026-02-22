import pytest
from alphaknit.parser import AmigurumiStackParser

def test_parser_normalization():
    parser = AmigurumiStackParser()
    assert parser.normalize("Magic Ring") == "mr"
    assert parser.normalize("2 sc in next st") == "inc"
    assert parser.normalize("Back Loop Only") == "blo"

def test_parser_tokenization():
    parser = AmigurumiStackParser()
    text = "R1: (sc, inc) x 6 (18)"
    tokens = parser.tokenize(text)
    # Expect: ['(', 'sc', 'inc', ')', 'x 6']
    assert tokens == ['(', 'sc', 'inc', ')', 'x 6']

def test_parser_nested_loops():
    parser = AmigurumiStackParser()
    # Complex pattern: ((sc, inc) x 2, sc) x 2
    # Inner: (sc, inc, sc, inc)
    # Outer: (sc, inc, sc, inc, sc) * 2
    # Result: sc, inc, sc, inc, sc, sc, inc, sc, inc, sc
    text = "((sc, inc) x 2, sc) x 2"
    result = parser.parse(text)
    
    expected = ['sc', 'inc', 'sc', 'inc', 'sc'] * 2
    assert result == expected

def test_parser_implicit_multiplier():
    parser = AmigurumiStackParser()
    # (sc, inc) 6 -> should be treated as (sc, inc) x 6
    text = "(sc, inc) 6"
    result = parser.parse(text)
    expected = ['sc', 'inc'] * 6
    assert result == expected

def test_parser_numeric_prefix():
    parser = AmigurumiStackParser()
    # 2 sc -> sc, sc
    text = "2 sc, inc"
    result = parser.parse(text)
    assert result == ['sc', 'sc', 'inc']
