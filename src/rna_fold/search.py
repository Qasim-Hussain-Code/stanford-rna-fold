"""
search.py — Template search engine for finding homologous RNA structures.

Uses k-mer pre-filtering for speed, followed by full Needleman-Wunsch
alignment on the top candidates.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .parser import parse_cif
from .align import needleman_wunsch, sequence_identity, fast_kmer_score


def search_templates(
    query_sequence: str,
    template_index: List[dict],
    top_n: int = 5,
    kmer_k: int = 4,
    kmer_prefilter_n: int = 50,
    min_identity: float = 0.20,
    min_coverage: float = 0.50,
    max_kmer_candidates: int = 0,
) -> List[dict]:
    """
    Search for the best template matches for a query sequence.

    Strategy:
        1. k-mer pre-filter: score all templates cheaply, keep top candidates
        2. Full N-W alignment on candidates: rank by sequence identity × coverage

    Parameters
    ----------
    query_sequence : str
        The RNA sequence to fold (ACGU).
    template_index : list of dict
        Pre-built index from parser.build_template_index().
    top_n : int
        Number of top templates to return.
    kmer_k : int
        k-mer size for pre-filtering.
    kmer_prefilter_n : int
        Number of candidates to keep after k-mer filtering.
    min_identity : float
        Minimum sequence identity threshold.
    min_coverage : float
        Minimum alignment coverage threshold.

    Returns
    -------
    list of dict
        Top templates with keys: file, chain_id, sequence, identity,
        coverage, alignment_score, aligned_query, aligned_template, mapping.
    """
    query_len = len(query_sequence)
    if query_len == 0:
        return []

    # max_kmer_candidates overrides kmer_prefilter_n when set
    prefilter_n = max_kmer_candidates if max_kmer_candidates > 0 else kmer_prefilter_n

    # Step 1: k-mer pre-filter
    scored = []
    for entry in template_index:
        tseq = entry["sequence"]
        # Length filter: template should be within 2x length ratio
        ratio = len(tseq) / query_len if query_len > 0 else 999
        if ratio < 0.3 or ratio > 3.0:
            continue
        kscore = fast_kmer_score(query_sequence, tseq, k=kmer_k)
        scored.append((kscore, entry))

    # Sort by k-mer score, keep top candidates
    scored.sort(key=lambda x: -x[0])
    candidates = [entry for _, entry in scored[:prefilter_n]]

    # Step 2: Full alignment on candidates
    results = []
    for entry in candidates:
        tseq = entry["sequence"]
        aligned_q, aligned_t, aln_score, mapping = needleman_wunsch(
            query_sequence, tseq
        )

        identity = sequence_identity(aligned_q, aligned_t)
        coverage = len(mapping) / query_len if query_len > 0 else 0

        if identity < min_identity or coverage < min_coverage:
            continue

        results.append({
            "file": entry["file"],
            "chain_id": entry["chain_id"],
            "sequence": tseq,
            "identity": identity,
            "coverage": coverage,
            "alignment_score": aln_score,
            "aligned_query": aligned_q,
            "aligned_template": aligned_t,
            "mapping": mapping,
        })

    # Rank by combined score: identity × coverage
    results.sort(key=lambda x: -(x["identity"] * x["coverage"]))
    return results[:top_n]


def get_template_coords(template_result: dict) -> Tuple[np.ndarray, List[int]]:
    """
    Load the actual 3D coordinates for a template hit.

    Returns
    -------
    coords : np.ndarray of shape (N, 3)
    residue_ids : list of int
    """
    chains = parse_cif(template_result["file"])
    chain_id = template_result["chain_id"]

    if chain_id in chains:
        _, coords, resids = chains[chain_id]
        return coords, resids

    # Fallback: return longest chain
    if chains:
        best = max(chains.values(), key=lambda x: len(x[0]))
        return best[1], best[2]

    return np.empty((0, 3)), []
