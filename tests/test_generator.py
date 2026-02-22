import pytest
from alphaknit.generator import AmigurumiGenerator

def test_generator_structure():
    gen = AmigurumiGenerator(min_rows=5, max_rows=10)
    result = gen.generate_pattern()
    
    assert "flat_sequence" in result
    assert "metadata" in result
    assert len(result["flat_sequence"]) > 0
    assert len(result["metadata"]) > 0

def test_generator_closed_loop():
    # Test if it generates valid stitch counts (divisible by 6 usually)
    gen = AmigurumiGenerator()
    result = gen.generate_pattern()
    
    for row in result["metadata"]:
        # Stitches should be multiples of 6 (expand/maintain/contract) or 3 (final close)
        assert row["stitch_count"] % 3 == 0, (
            f"Row {row['row']} has stitch_count={row['stitch_count']} "
            f"which is not a multiple of 3"
        )

def test_generator_start_token():
    gen = AmigurumiGenerator()
    result = gen.generate_pattern()
    # Should start with MR_6
    assert "mr_6" in result["flat_sequence"][0] or "mr_6" in result["flat_sequence"][1]
