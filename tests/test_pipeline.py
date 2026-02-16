"""
test_pipeline.py — Unit tests for the RNA 3D structure prediction pipeline.

Tests each module independently and validates the end-to-end pipeline
with mock data. Updated for v2 improvements.
"""

import os
import sys
import tempfile
import numpy as np
import pytest

# Ensure we can import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rna_fold.align import needleman_wunsch, sequence_identity, fast_kmer_score
from rna_fold.transfer import (
    transfer_coordinates, generate_de_novo,
    multi_template_consensus, C1_C1_DISTANCE,
)
from rna_fold.refine import refine_coordinates
from rna_fold.ensemble import generate_ensemble, apply_noise, apply_hinge
from rna_fold.parser import parse_cif, _tokenize_cif_line
from rna_fold.pipeline import predict_structure, _write_target
from rna_fold.nussinov import predict_base_pairs, get_paired_regions, is_in_helix


# ============================================================
# Alignment Tests
# ============================================================
class TestAlignment:
    def test_identical_sequences(self):
        seq = "ACGUACGU"
        aq, at, score, mapping = needleman_wunsch(seq, seq)
        assert aq == seq
        assert at == seq
        assert len(mapping) == len(seq)
        for i, (qi, ti) in enumerate(mapping):
            assert qi == i
            assert ti == i

    def test_single_mismatch(self):
        q = "ACGUACGU"
        t = "ACGUUCGU"
        aq, at, score, mapping = needleman_wunsch(q, t)
        ident = sequence_identity(aq, at)
        assert ident > 0.5

    def test_gap_insertion(self):
        q = "ACGUACGU"
        t = "ACGUXXACGU"
        aq, at, score, mapping = needleman_wunsch(q, t)
        assert len(mapping) > 0

    def test_empty_sequences(self):
        aq, at, score, mapping = needleman_wunsch("", "ACGU")
        assert len(mapping) == 0

    def test_kmer_score_identical(self):
        seq = "ACGUACGUACGU"
        assert fast_kmer_score(seq, seq) == 1.0

    def test_kmer_score_different(self):
        assert fast_kmer_score("AAAA", "CCCC") == 0.0

    def test_kmer_score_partial(self):
        score = fast_kmer_score("ACGUACGU", "ACGUXXXX")
        assert 0 < score < 1


# ============================================================
# Coordinate Transfer Tests
# ============================================================
class TestTransfer:
    def test_full_transfer(self):
        template_coords = np.array([
            [0, 0, 0], [5.9, 0, 0], [11.8, 0, 0], [17.7, 0, 0]
        ], dtype=float)
        mapping = [(0, 0), (1, 1), (2, 2), (3, 3)]
        result = transfer_coordinates("ACGU", template_coords, mapping)
        assert result.shape == (4, 3)
        np.testing.assert_array_almost_equal(result, template_coords)

    def test_gap_filling(self):
        template_coords = np.array([
            [0, 0, 0], [5.9, 0, 0], [11.8, 0, 0], [17.7, 0, 0]
        ], dtype=float)
        mapping = [(0, 0), (3, 3)]
        result = transfer_coordinates("ACGU", template_coords, mapping)
        assert result.shape == (4, 3)
        assert not np.any(np.isnan(result))

    def test_de_novo_compact(self):
        coords = generate_de_novo("ACGUACGUACGUACGU", geometry="compact")
        assert coords.shape == (16, 3)
        assert not np.any(np.isnan(coords))
        # Check that bonds are approximately correct
        for i in range(15):
            dist = np.linalg.norm(coords[i + 1] - coords[i])
            assert dist > 2.0, f"Bond {i}-{i+1} too short: {dist}"

    def test_de_novo_helix(self):
        coords = generate_de_novo("ACGUACGU", geometry="helix")
        assert coords.shape == (8, 3)
        assert not np.any(np.isnan(coords))

    def test_de_novo_linear(self):
        coords = generate_de_novo("ACGU", geometry="linear")
        assert coords.shape == (4, 3)
        for i in range(3):
            dist = np.linalg.norm(coords[i + 1] - coords[i])
            assert abs(dist - C1_C1_DISTANCE) < 0.01


# ============================================================
# Multi-Template Consensus Tests
# ============================================================
class TestMultiTemplateConsensus:
    def test_two_templates(self):
        """Consensus of two templates should be between them."""
        n = 8
        seq = "ACGUACGU"
        coords1 = np.zeros((n, 3))
        coords2 = np.zeros((n, 3))
        for i in range(n):
            coords1[i] = [i * 5.9, 0, 0]
            coords2[i] = [i * 5.9, 10, 0]  # Shifted in Y

        mapping = [(i, i) for i in range(n)]
        results = [
            {"identity": 0.8, "coverage": 1.0, "mapping": mapping},
            {"identity": 0.6, "coverage": 1.0, "mapping": mapping},
        ]
        coords_list = [(coords1, list(range(n))), (coords2, list(range(n)))]

        consensus = multi_template_consensus(seq, results, coords_list)
        assert consensus.shape == (n, 3)
        assert not np.any(np.isnan(consensus))
        # Consensus Y should be between 0 and 10 (weighted)
        for i in range(n):
            assert 0 < consensus[i, 1] < 10

    def test_single_template_consensus(self):
        """With one template, consensus should match that template."""
        seq = "ACGU"
        coords = np.array([[0, 0, 0], [5.9, 0, 0], [11.8, 0, 0], [17.7, 0, 0]])
        mapping = [(0, 0), (1, 1), (2, 2), (3, 3)]
        results = [{"identity": 1.0, "coverage": 1.0, "mapping": mapping}]
        coords_list = [(coords, [1, 2, 3, 4])]

        consensus = multi_template_consensus(seq, results, coords_list)
        np.testing.assert_array_almost_equal(consensus, coords)


# ============================================================
# Nussinov Tests
# ============================================================
class TestNussinov:
    def test_simple_hairpin(self):
        """A sequence like GCAAAGC should form a hairpin."""
        seq = "GCAAAGC"
        pairs = predict_base_pairs(seq)
        assert len(pairs) >= 1
        # G and C at ends should pair
        assert (0, 6) in pairs

    def test_paired_regions(self):
        """Test identification of helical stems."""
        pairs = [(0, 10), (1, 9), (2, 8)]
        regions = get_paired_regions(pairs, 11)
        assert len(regions) >= 1
        assert len(regions[0]["pairs"]) == 3

    def test_is_in_helix(self):
        pairs = [(0, 10), (1, 9)]
        assert is_in_helix(0, pairs) == True
        assert is_in_helix(10, pairs) == True
        assert is_in_helix(5, pairs) == False

    def test_empty_sequence(self):
        pairs = predict_base_pairs("")
        assert pairs == []

    def test_short_sequence(self):
        pairs = predict_base_pairs("AC")
        assert pairs == []


# ============================================================
# Refinement Tests
# ============================================================
class TestRefinement:
    def test_preserves_shape(self):
        coords = np.random.rand(20, 3) * 50
        refined = refine_coordinates(coords)
        assert refined.shape == coords.shape

    def test_no_nans(self):
        coords = generate_de_novo("A" * 30, geometry="helix")
        refined = refine_coordinates(coords)
        assert not np.any(np.isnan(refined))

    def test_short_sequence(self):
        coords = np.array([[0, 0, 0]], dtype=float)
        refined = refine_coordinates(coords)
        assert refined.shape == (1, 3)


# ============================================================
# Ensemble Tests
# ============================================================
class TestEnsemble:
    def test_generates_five_models(self):
        base = generate_de_novo("ACGUACGU", geometry="helix")
        ensemble = generate_ensemble(base, n_models=5)
        assert len(ensemble) == 5
        for model in ensemble:
            assert model.shape == base.shape

    def test_models_are_diverse(self):
        base = generate_de_novo("ACGUACGUACGUACGU", geometry="helix")
        ensemble = generate_ensemble(base, n_models=5)
        for i in range(1, 5):
            rmsd = np.sqrt(np.mean((ensemble[0] - ensemble[i]) ** 2))
            assert rmsd > 0.01, f"Model {i} is identical to model 0"

    def test_noise_changes_coords(self):
        base = generate_de_novo("ACGUACGU", geometry="helix")
        noised = apply_noise(base, scale=1.0)
        assert not np.allclose(base, noised)

    def test_hinge_changes_coords(self):
        base = generate_de_novo("ACGUACGUACGUACGU", geometry="helix")
        hinged = apply_hinge(base, angle=15.0)
        assert not np.allclose(base, hinged)
        mid = len(base) // 2
        np.testing.assert_array_almost_equal(base[:mid], hinged[:mid])


# ============================================================
# CIF Parser Tests
# ============================================================
class TestParser:
    def test_tokenize_simple(self):
        tokens = _tokenize_cif_line("ATOM 1 C1' A G 1 10.0 20.0 30.0")
        assert len(tokens) == 9
        assert tokens[0] == "ATOM"

    def test_tokenize_quoted(self):
        tokens = _tokenize_cif_line("ATOM 1 'C1 prime' A")
        assert "C1 prime" in tokens

    def test_parse_mock_cif(self):
        """Test parsing a minimal mock CIF file."""
        cif_content = """data_test
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.label_alt_id
ATOM 1 C1' A A 1 10.000 20.000 30.000 .
ATOM 2 C1' C A 2 15.900 20.000 30.000 .
ATOM 3 C1' G A 3 21.800 20.000 30.000 .
ATOM 4 C1' U A 4 27.700 20.000 30.000 .
#
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cif',
                                         delete=False) as f:
            f.write(cif_content)
            tmpfile = f.name

        try:
            chains = parse_cif(tmpfile)
            assert "A" in chains
            seq, coords, resids = chains["A"]
            assert seq == "ACGU"
            assert coords.shape == (4, 3)
            np.testing.assert_almost_equal(coords[0], [10.0, 20.0, 30.0])
        finally:
            os.unlink(tmpfile)


# ============================================================
# Submission Format Tests
# ============================================================
class TestSubmissionFormat:
    def test_write_target(self):
        """Verify submission CSV format is correct."""
        import io
        sequence = "ACGU"
        base = generate_de_novo(sequence, geometry="linear")
        models = generate_ensemble(base, n_models=5)

        f = io.StringIO()
        _write_target(f, "R1107", sequence, models)
        output = f.getvalue()

        lines = output.strip().split("\n")
        assert len(lines) == 4

        parts = lines[0].split(",")
        assert len(parts) == 18
        assert parts[0] == "R1107_1"
        assert parts[1] == "A"
        assert parts[2] == "1"

        for line in lines:
            parts = line.split(",")
            for coord_str in parts[3:]:
                val = float(coord_str)
                assert -999.999 <= val <= 9999.999


# ============================================================
# Integration Tests
# ============================================================
class TestIntegration:
    def test_end_to_end_no_templates(self):
        """Full pipeline with no templates — de novo only."""
        sequence = "ACGUACGUACGUACGU"
        models = predict_structure(
            sequence=sequence,
            template_index=[],
            n_models=5,
            verbose=False,
        )
        assert len(models) == 5
        for model in models:
            assert model.shape == (len(sequence), 3)
            assert not np.any(np.isnan(model))

    def test_compact_de_novo_diversity(self):
        """Test that compact de novo produces diverse models."""
        sequence = "ACGUACGUACGUACGUACGUACGU"
        models = predict_structure(
            sequence=sequence,
            template_index=[],
            n_models=5,
            verbose=False,
        )
        # Check diversity
        for i in range(1, 5):
            rmsd = np.sqrt(np.mean((models[0] - models[i]) ** 2))
            assert rmsd > 0.01, f"Model {i} not diverse enough"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
