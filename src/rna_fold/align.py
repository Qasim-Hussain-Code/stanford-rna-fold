"""
align.py — Needleman-Wunsch global sequence alignment for RNA.

Pure NumPy implementation with affine gap penalties. Features:
  - RIBOSUM-like substitution matrix for structure-aware scoring
  - Affine gap penalties optimized for RNA homology search
  - Fast k-mer pre-filter for template screening
"""

import numpy as np
from typing import List, Tuple

# RIBOSUM85-60 inspired substitution matrix for RNA
# Derived from structural alignments of RNA families.
# Scores transitions/transversions and wobble pairs differently.
# Format: score(a, b) where a, b in {A, C, G, U}
_RIBOSUM = {
    ("A", "A"):  2, ("A", "C"): -2, ("A", "G"):  0, ("A", "U"): -1,
    ("C", "A"): -2, ("C", "C"):  3, ("C", "G"): -2, ("C", "U"): -1,
    ("G", "A"):  0, ("G", "C"): -2, ("G", "G"):  3, ("G", "U"):  0,
    ("U", "A"): -1, ("U", "C"): -1, ("U", "G"):  0, ("U", "U"):  2,
}

# Gap penalties
GAP_OPEN = -5
GAP_EXTEND = -1


def _substitution_score(a: str, b: str) -> int:
    """Get substitution score from RIBOSUM matrix."""
    a = a.upper()
    b = b.upper()
    return _RIBOSUM.get((a, b), -2)  # Default mismatch


def needleman_wunsch(query: str, template: str,
                     match: int = 2,
                     mismatch: int = -1,
                     gap_open: int = GAP_OPEN,
                     gap_extend: int = GAP_EXTEND,
                     use_ribosum: bool = True,
                     ) -> Tuple[str, str, float, List[Tuple[int, int]]]:
    """
    Global alignment with affine gap penalties.

    Parameters
    ----------
    query, template : str
        RNA sequences (ACGU).
    match, mismatch : int
        Fallback scores (used when use_ribosum=False).
    gap_open, gap_extend : int
        Affine gap penalties.
    use_ribosum : bool
        If True, use RIBOSUM substitution matrix instead of match/mismatch.

    Returns
    -------
    aligned_query : str
    aligned_template : str
    score : float
    mapping : list of (query_pos, template_pos)
    """
    n = len(query)
    m = len(template)

    NEG_INF = -1e9
    M = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)
    X = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)
    Y = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)

    M[0, 0] = 0.0
    for i in range(1, n + 1):
        X[i, 0] = gap_open + (i - 1) * gap_extend
    for j in range(1, m + 1):
        Y[0, j] = gap_open + (j - 1) * gap_extend

    TB_M = np.zeros((n + 1, m + 1), dtype=np.int8)
    TB_X = np.zeros((n + 1, m + 1), dtype=np.int8)
    TB_Y = np.zeros((n + 1, m + 1), dtype=np.int8)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if use_ribosum:
                s = _substitution_score(query[i - 1], template[j - 1])
            else:
                s = match if query[i - 1] == template[j - 1] else mismatch

            opts_m = [M[i - 1, j - 1] + s, X[i - 1, j - 1] + s,
                      Y[i - 1, j - 1] + s]
            best_m = int(np.argmax(opts_m))
            M[i, j] = opts_m[best_m]
            TB_M[i, j] = best_m

            opts_x = [M[i - 1, j] + gap_open, X[i - 1, j] + gap_extend]
            best_x = int(np.argmax(opts_x))
            X[i, j] = opts_x[best_x]
            TB_X[i, j] = [0, 1][best_x]

            opts_y = [M[i, j - 1] + gap_open, Y[i, j - 1] + gap_extend]
            best_y = int(np.argmax(opts_y))
            Y[i, j] = opts_y[best_y]
            TB_Y[i, j] = [0, 2][best_y]

    final_scores = [M[n, m], X[n, m], Y[n, m]]
    state = int(np.argmax(final_scores))
    score = final_scores[state]

    # Traceback
    aligned_q = []
    aligned_t = []
    i, j = n, m

    while i > 0 or j > 0:
        if state == 0:
            if i == 0 or j == 0:
                break
            aligned_q.append(query[i - 1])
            aligned_t.append(template[j - 1])
            state = TB_M[i, j]
            i -= 1
            j -= 1
        elif state == 1:
            aligned_q.append(query[i - 1])
            aligned_t.append("-")
            state = TB_X[i, j]
            i -= 1
        else:
            aligned_q.append("-")
            aligned_t.append(template[j - 1])
            state = TB_Y[i, j]
            j -= 1

    while i > 0:
        aligned_q.append(query[i - 1])
        aligned_t.append("-")
        i -= 1
    while j > 0:
        aligned_q.append("-")
        aligned_t.append(template[j - 1])
        j -= 1

    aligned_q = "".join(reversed(aligned_q))
    aligned_t = "".join(reversed(aligned_t))

    mapping = []
    qi, ti = 0, 0
    for aq, at in zip(aligned_q, aligned_t):
        if aq != "-" and at != "-":
            mapping.append((qi, ti))
            qi += 1
            ti += 1
        elif aq != "-":
            qi += 1
        else:
            ti += 1

    return aligned_q, aligned_t, score, mapping


def sequence_identity(aligned_q: str, aligned_t: str) -> float:
    """Compute fraction of identical positions in an alignment."""
    matches = sum(1 for a, b in zip(aligned_q, aligned_t)
                  if a == b and a != "-")
    total = max(
        sum(1 for c in aligned_q if c != "-"),
        sum(1 for c in aligned_t if c != "-"),
        1,
    )
    return matches / total


def fast_kmer_score(query: str, template: str, k: int = 4) -> float:
    """
    Fast k-mer based similarity score for pre-filtering.

    Returns fraction of query k-mers found in template.
    Much faster than full alignment for initial screening.
    """
    if len(query) < k or len(template) < k:
        return 0.0

    query_kmers = set(query[i:i + k] for i in range(len(query) - k + 1))
    template_kmers = set(template[i:i + k] for i in range(len(template) - k + 1))

    if not query_kmers:
        return 0.0

    shared = len(query_kmers & template_kmers)
    return shared / len(query_kmers)
