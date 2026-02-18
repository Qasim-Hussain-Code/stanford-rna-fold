"""
align.py — Needleman-Wunsch global sequence alignment for RNA.

Pure NumPy implementation with affine gap penalties. Features:
  - RIBOSUM-like substitution matrix for structure-aware scoring
  - Affine gap penalties optimized for RNA homology search
  - Banded alignment mode for long sequences (O(n*band) vs O(n*m))
  - Fast k-mer pre-filter for template screening
"""

import numpy as np
from typing import List, Tuple

# RIBOSUM85-60 inspired substitution matrix for RNA
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
    return _RIBOSUM.get((a.upper(), b.upper()), -2)


def needleman_wunsch(query: str, template: str,
                     match: int = 2,
                     mismatch: int = -1,
                     gap_open: int = GAP_OPEN,
                     gap_extend: int = GAP_EXTEND,
                     use_ribosum: bool = True,
                     band_width: int = 0,
                     ) -> Tuple[str, str, float, List[Tuple[int, int]]]:
    """
    Global alignment with affine gap penalties.

    Parameters
    ----------
    query, template : str
        RNA sequences (ACGU).
    band_width : int
        If > 0 and both sequences are long, use banded alignment.
        Only cells within band_width of the diagonal are computed.
        Set to 0 for full DP (default).
    """
    n = len(query)
    m = len(template)

    # Auto-select banded mode for long sequences
    if band_width == 0 and n * m > 500_000:
        band_width = max(50, min(n, m) // 4)

    if band_width > 0:
        return _banded_nw(query, template, gap_open, gap_extend,
                         use_ribosum, match, mismatch, band_width)

    NEG_INF = -1e9
    M_mat = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)
    X_mat = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)
    Y_mat = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)

    M_mat[0, 0] = 0.0
    for i in range(1, n + 1):
        X_mat[i, 0] = gap_open + (i - 1) * gap_extend
    for j in range(1, m + 1):
        Y_mat[0, j] = gap_open + (j - 1) * gap_extend

    TB_M = np.zeros((n + 1, m + 1), dtype=np.int8)
    TB_X = np.zeros((n + 1, m + 1), dtype=np.int8)
    TB_Y = np.zeros((n + 1, m + 1), dtype=np.int8)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if use_ribosum:
                s = _substitution_score(query[i - 1], template[j - 1])
            else:
                s = match if query[i - 1] == template[j - 1] else mismatch

            opts_m = [M_mat[i - 1, j - 1] + s, X_mat[i - 1, j - 1] + s,
                      Y_mat[i - 1, j - 1] + s]
            best_m = int(np.argmax(opts_m))
            M_mat[i, j] = opts_m[best_m]
            TB_M[i, j] = best_m

            opts_x = [M_mat[i - 1, j] + gap_open, X_mat[i - 1, j] + gap_extend]
            best_x = int(np.argmax(opts_x))
            X_mat[i, j] = opts_x[best_x]
            TB_X[i, j] = [0, 1][best_x]

            opts_y = [M_mat[i, j - 1] + gap_open, Y_mat[i, j - 1] + gap_extend]
            best_y = int(np.argmax(opts_y))
            Y_mat[i, j] = opts_y[best_y]
            TB_Y[i, j] = [0, 2][best_y]

    final_scores = [M_mat[n, m], X_mat[n, m], Y_mat[n, m]]
    state = int(np.argmax(final_scores))
    score = final_scores[state]

    # Traceback
    aligned_q, aligned_t = _traceback(
        query, template, TB_M, TB_X, TB_Y, state, n, m
    )

    mapping = _build_mapping(aligned_q, aligned_t)
    return aligned_q, aligned_t, score, mapping


def _banded_nw(query: str, template: str,
               gap_open: int, gap_extend: int,
               use_ribosum: bool, match: int, mismatch: int,
               band: int) -> Tuple[str, str, float, List[Tuple[int, int]]]:
    """
    Banded Needleman-Wunsch alignment.

    Only fills cells within 'band' positions of the diagonal,
    reducing time from O(n*m) to O(n*band).
    """
    n = len(query)
    m = len(template)
    NEG_INF = -1e9

    # Scale factor for diagonal mapping when n != m
    scale = m / n if n > 0 else 1.0

    M_mat = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)
    X_mat = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)
    Y_mat = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)

    M_mat[0, 0] = 0.0

    # Initialize edges within band
    for i in range(1, min(n + 1, band + 1)):
        X_mat[i, 0] = gap_open + (i - 1) * gap_extend
    for j in range(1, min(m + 1, band + 1)):
        Y_mat[0, j] = gap_open + (j - 1) * gap_extend

    TB_M = np.zeros((n + 1, m + 1), dtype=np.int8)
    TB_X = np.zeros((n + 1, m + 1), dtype=np.int8)
    TB_Y = np.zeros((n + 1, m + 1), dtype=np.int8)

    for i in range(1, n + 1):
        # Compute the expected diagonal position
        diag_j = int(i * scale)
        j_lo = max(1, diag_j - band)
        j_hi = min(m, diag_j + band)

        for j in range(j_lo, j_hi + 1):
            if use_ribosum:
                s = _substitution_score(query[i - 1], template[j - 1])
            else:
                s = match if query[i - 1] == template[j - 1] else mismatch

            opts_m = [M_mat[i - 1, j - 1] + s, X_mat[i - 1, j - 1] + s,
                      Y_mat[i - 1, j - 1] + s]
            best_m = int(np.argmax(opts_m))
            M_mat[i, j] = opts_m[best_m]
            TB_M[i, j] = best_m

            opts_x = [M_mat[i - 1, j] + gap_open, X_mat[i - 1, j] + gap_extend]
            best_x = int(np.argmax(opts_x))
            X_mat[i, j] = opts_x[best_x]
            TB_X[i, j] = [0, 1][best_x]

            opts_y = [M_mat[i, j - 1] + gap_open, Y_mat[i, j - 1] + gap_extend]
            best_y = int(np.argmax(opts_y))
            Y_mat[i, j] = opts_y[best_y]
            TB_Y[i, j] = [0, 2][best_y]

    final_scores = [M_mat[n, m], X_mat[n, m], Y_mat[n, m]]
    state = int(np.argmax(final_scores))
    score = final_scores[state]

    aligned_q, aligned_t = _traceback(
        query, template, TB_M, TB_X, TB_Y, state, n, m
    )

    mapping = _build_mapping(aligned_q, aligned_t)
    return aligned_q, aligned_t, score, mapping


def _traceback(query, template, TB_M, TB_X, TB_Y, state, n, m):
    """Shared traceback for full and banded NW."""
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

    return "".join(reversed(aligned_q)), "".join(reversed(aligned_t))


def _build_mapping(aligned_q: str, aligned_t: str) -> List[Tuple[int, int]]:
    """Build (query_pos, template_pos) mapping from aligned sequences."""
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
    return mapping


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
    """
    if len(query) < k or len(template) < k:
        return 0.0

    query_kmers = set(query[i:i + k] for i in range(len(query) - k + 1))
    template_kmers = set(template[i:i + k] for i in range(len(template) - k + 1))

    if not query_kmers:
        return 0.0

    shared = len(query_kmers & template_kmers)
    return shared / len(query_kmers)
