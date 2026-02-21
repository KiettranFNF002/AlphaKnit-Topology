"""
Integration tests: full pipeline
  tokens → compile → validate → canonicalize
  generator → compile → validate
"""
import pytest
from src.alphaknit.compiler import KnittingCompiler
from src.alphaknit.validator import GraphValidator
from src.alphaknit.canonicalizer import canonicalize_pattern
from src.alphaknit.generator import AmigurumiGenerator


COMPILER = KnittingCompiler()
VALIDATOR = GraphValidator()


# ============================================================
# Full pipeline tests
# ============================================================

class TestFullPipeline:

    def test_magic_ring_only(self):
        """mr_6 alone compiles to 6 root nodes with no errors."""
        graph = COMPILER.compile(['mr_6'])
        errors = VALIDATOR.validate(graph)
        assert graph.size == 6
        assert len(errors) == 0

    def test_sphere_expansion_row(self):
        """
        A standard amigurumi expansion row:
          mr_6 → inc x6 → (sc, inc) x6
        Should produce rows [6, 12, 18] with no validation errors.
        """
        tokens = ['mr_6'] + ['inc'] * 6 + ['sc', 'inc'] * 6
        graph = COMPILER.compile(tokens)
        errors = VALIDATOR.validate(graph)
        counts = graph.stitch_count_per_row()
        assert counts == {0: 6, 1: 12, 2: 18}
        assert len(errors) == 0

    def test_maintain_row(self):
        """mr_6 + 6 sc = two rows of 6, no errors."""
        tokens = ['mr_6'] + ['sc'] * 6
        graph = COMPILER.compile(tokens)
        errors = VALIDATOR.validate(graph)
        assert graph.stitch_count_per_row() == {0: 6, 1: 6}
        assert len(errors) == 0

    def test_contract_row(self):
        """mr_6 + inc*6 + dec*6 = rows [6, 12, 6], no errors."""
        tokens = ['mr_6'] + ['inc'] * 6 + ['dec'] * 6
        graph = COMPILER.compile(tokens)
        errors = VALIDATOR.validate(graph)
        counts = graph.stitch_count_per_row()
        assert counts == {0: 6, 1: 12, 2: 6}
        assert len(errors) == 0

    def test_canonicalize_equivalent_patterns(self):
        """
        Two patterns with same structure but different inc placement
        should produce the same stitch count after canonicalization.
        """
        # Uneven: all incs first
        p1 = ['mr_6'] + ['inc', 'inc', 'sc', 'sc', 'sc', 'sc']
        # Even: incs spread out
        p2 = ['mr_6'] + ['sc', 'inc', 'sc', 'sc', 'inc', 'sc']

        c1 = canonicalize_pattern(p1)
        c2 = canonicalize_pattern(p2)

        # Both should have same token counts
        assert c1.count('inc') == c2.count('inc')
        assert c1.count('sc') == c2.count('sc')

    def test_graph_parent_links_correct(self):
        """Every non-root stitch should have parents in the previous row."""
        tokens = ['mr_6'] + ['sc'] * 6
        graph = COMPILER.compile(tokens)
        row1_nodes = graph.get_row(1)
        row0_ids = {n.id for n in graph.get_row(0)}
        for node in row1_nodes:
            for pid in node.parents:
                assert pid in row0_ids, f"Parent {pid} not in row 0"


# ============================================================
# Generator integration
# ============================================================

class TestGeneratorIntegration:

    def test_generated_pattern_compiles(self):
        """Generated patterns should compile without errors."""
        gen = AmigurumiGenerator(min_rows=3, max_rows=8)
        for _ in range(10):
            result = gen.generate_pattern()
            flat = result['flat_sequence']
            graph = COMPILER.compile(flat)
            assert graph.size > 0

    def test_generated_pattern_validates(self):
        """Generated patterns should pass validation."""
        gen = AmigurumiGenerator(min_rows=3, max_rows=8)
        for _ in range(10):
            result = gen.generate_pattern()
            flat = result['flat_sequence']
            graph = COMPILER.compile(flat)
            errors = VALIDATOR.validate(graph)
            critical = [e for e in errors if e.severity == 'critical']
            assert len(critical) == 0, f"Critical errors: {critical}"

    def test_generated_stitch_graph_returned(self):
        """generate_pattern should return a stitch_graph key."""
        gen = AmigurumiGenerator(min_rows=3, max_rows=5)
        result = gen.generate_pattern()
        assert 'stitch_graph' in result
        assert result['stitch_graph'].size > 0
