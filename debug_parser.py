from src.alphaknit.parser import AmigurumiStackParser

def run_debug():
    parser = AmigurumiStackParser()
    
    # Case 1: Tokenization
    text1 = "R1: (sc, inc) x 6 (18)"
    tokens1 = parser.tokenize(text1)
    print(f"Case 1 Tokens: {tokens1}")
    
    # Case 2: Nested Loops
    text2 = "((sc, inc) x 2, sc) x 2"
    result2 = parser.parse(text2)
    print(f"Case 2 Result: {result2}")
    
    # Case 3: Numeric Prefix
    text3 = "2 sc, inc"
    tokens3 = parser.tokenize(text3)
    print(f"Case 3 Tokens: {tokens3}")
    result3 = parser.parse(text3)
    print(f"Case 3 Result: {result3}")

if __name__ == "__main__":
    run_debug()
