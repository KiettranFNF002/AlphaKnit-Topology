import pytest
from alphaknit.stitch_graph import StitchGraph, StitchNode
from alphaknit.compiler import KnittingCompiler, CompileError
from alphaknit.validator import GraphValidator
from alphaknit.canonicalizer import canonicalize_row, canonicalize_pattern


# ============================================================
# StitchGraph Tests
# ============================================================

class TestStitchGraph:
    def test_add_stitch_and_backlink(self):
        g = StitchGraph()
        root = g.add_stitch('mr', parents=[], row=0, column=0)
        child = g.add_stitch('sc', parents=[root.id], row=1, column=0)
        assert child.parents == [root.id]
        assert child.id in g.nodes[root.id].children

    def test_stitch_count_per_row(self):
        g = StitchGraph()
        for c in range(6):
            g.add_stitch('mr', row=0, column=c)
        for c in range(6):
            g.add_stitch('sc', parents=[c], row=1, column=c)
        counts = g.stitch_count_per_row()
        assert counts == {0: 6, 1: 6}

    def test_serialization_roundtrip(self):
        g = StitchGraph()
        g.add_stitch('mr', row=0, column=0)
        g.add_stitch('sc', parents=[0], row=1, column=0)
        data = g.to_dict()
        g2 = StitchGraph.from_dict(data)
        assert g2.size == 2
        assert g2.nodes[1].parents == [0]


# ============================================================
# Compiler Tests
# ============================================================

class TestCompiler:
    def test_magic_ring(self):
        compiler = KnittingCompiler()
        graph = compiler.compile(['mr_6'])
        assert graph.size == 6
        counts = graph.stitch_count_per_row()
        assert counts == {0: 6}

    def test_mr6_plus_sc_row(self):
        """mr_6 + 6 sc = row 0 (6 stitches) + row 1 (6 stitches)"""
        compiler = KnittingCompiler()
        tokens = ['mr_6'] + ['sc'] * 6
        graph = compiler.compile(tokens)
        assert graph.size == 12
        counts = graph.stitch_count_per_row()
        assert counts == {0: 6, 1: 6}

    def test_mr6_plus_expand(self):
        """mr_6 + (sc, inc) x 3 -> row 0: 6, row 1: 9"""
        compiler = KnittingCompiler()
        tokens = ['mr_6'] + ['sc', 'inc'] * 3
        graph = compiler.compile(tokens)
        counts = graph.stitch_count_per_row()
        assert counts[0] == 6
        assert counts[1] == 9  # 3 sc + 3 inc (each inc=2 output) = 3+6=9

    def test_mr6_full_inc(self):
        """mr_6 + inc x 6 -> row 0: 6, row 1: 12"""
        compiler = KnittingCompiler()
        tokens = ['mr_6'] + ['inc'] * 6
        graph = compiler.compile(tokens)
        counts = graph.stitch_count_per_row()
        assert counts[0] == 6
        assert counts[1] == 12

    def test_unknown_token_raises(self):
        compiler = KnittingCompiler()
        with pytest.raises(CompileError):
            compiler.compile(['mr_6', 'unknown_stitch'])

    def test_multi_row_expansion(self):
        """mr_6 + inc*6 + (sc, inc)*6 -> rows: 6, 12, 18"""
        compiler = KnittingCompiler()
        tokens = ['mr_6'] + ['inc'] * 6 + ['sc', 'inc'] * 6
        graph = compiler.compile(tokens)
        counts = graph.stitch_count_per_row()
        assert counts[0] == 6
        assert counts[1] == 12
        assert counts[2] == 18


# ============================================================
# Validator Tests
# ============================================================

class TestValidator:
    def test_valid_graph_no_errors(self):
        compiler = KnittingCompiler()
        graph = compiler.compile(['mr_6'] + ['sc'] * 6)
        validator = GraphValidator()
        errors = validator.validate(graph)
        assert len(errors) == 0

    def test_detect_orphan(self):
        """Manually create an orphan stitch."""
        g = StitchGraph()
        g.add_stitch('mr', row=0, column=0)
        g.add_stitch('sc', parents=[], row=1, column=0)  # orphan!
        validator = GraphValidator()
        errors = validator.validate(g)
        orphan_errors = [e for e in errors if e.error_type == 'orphan']
        assert len(orphan_errors) == 1

    def test_valid_expansion(self):
        compiler = KnittingCompiler()
        graph = compiler.compile(['mr_6'] + ['inc'] * 6)
        validator = GraphValidator()
        errors = validator.validate(graph)
        assert len(errors) == 0


# ============================================================
# Canonicalizer Tests
# ============================================================

class TestCanonicalizer:
    def test_canonicalize_row_even_spacing(self):
        # 2 incs among 4 sc = should be evenly spaced
        row = ['inc', 'inc', 'sc', 'sc', 'sc', 'sc']
        result = canonicalize_row(row)
        # Should place inc at regular intervals
        assert result.count('inc') == 2
        assert result.count('sc') == 4
        assert len(result) == 6

    def test_all_sc_unchanged(self):
        row = ['sc', 'sc', 'sc']
        result = canonicalize_row(row)
        assert result == ['sc', 'sc', 'sc']

    def test_canonicalize_pattern_preserves_mr(self):
        tokens = ['mr_6'] + ['sc'] * 6
        result = canonicalize_pattern(tokens)
        assert result[0] == 'mr_6'
        assert len(result) == 7
