"""
align.py — Needleman-Wunsch global sequence alignment for RNA.

Pure NumPy implementation with affine gap penalties optimized for
RNA template matching. No external dependencies.
"""

import numpy as np
from typing import List, Tuple

# RNA-specific scoring
MATCH_SCORE = 2
MISMATCH_SCORE = -1
GAP_OPEN = -5
GAP_EXTEND = -1


def needleman_wunsch(query: str, template: str,
                     match: int = MATCH_SCORE,
                     mismatch: int = MISMATCH_SCORE,
                     gap_open: int = GAP_OPEN,
                     gap_extend: int = GAP_EXTEND,
                     ) -> Tuple[str, str, float, List[Tuple[int, int]]]:
    """
    Global alignment with affine gap penalties.

    Parameters
    ----------
    query, template : str
        RNA sequences (ACGU).
    match, mismatch : int
        Scores for match/mismatch positions.
    gap_open, gap_extend : int
        Affine gap penalties.

    Returns
    -------
    aligned_query : str
        Aligned query with '-' for gaps.
    aligned_template : str
        Aligned template with '-' for gaps.
    score : float
        Alignment score.
    mapping : list of (query_pos, template_pos)
        Zero-indexed pairs of matched positions (no gaps).
    """
    n = len(query)
    m = len(template)

    # Score matrices: M (match), X (gap in template), Y (gap in query)
    NEG_INF = -1e9
    M = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)
    X = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)
    Y = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)

    M[0, 0] = 0.0
    for i in range(1, n + 1):
        X[i, 0] = gap_open + (i - 1) * gap_extend
    for j in range(1, m + 1):
        Y[0, j] = gap_open + (j - 1) * gap_extend

    # Traceback matrices
    TB_M = np.zeros((n + 1, m + 1), dtype=np.int8)
    TB_X = np.zeros((n + 1, m + 1), dtype=np.int8)
    TB_Y = np.zeros((n + 1, m + 1), dtype=np.int8)
    # Traceback codes: 0=M, 1=X, 2=Y

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match if query[i - 1] == template[j - 1] else mismatch

            # M[i,j]: best alignment ending with a match at (i,j)
            opts_m = [M[i - 1, j - 1] + s, X[i - 1, j - 1] + s,
                      Y[i - 1, j - 1] + s]
            best_m = int(np.argmax(opts_m))
            M[i, j] = opts_m[best_m]
            TB_M[i, j] = best_m

            # X[i,j]: gap in template (consume query residue)
            opts_x = [M[i - 1, j] + gap_open, X[i - 1, j] + gap_extend]
            best_x = int(np.argmax(opts_x))
            X[i, j] = opts_x[best_x]
            TB_X[i, j] = [0, 1][best_x]

            # Y[i,j]: gap in query (consume template residue)
            opts_y = [M[i, j - 1] + gap_open, Y[i, j - 1] + gap_extend]
            best_y = int(np.argmax(opts_y))
            Y[i, j] = opts_y[best_y]
            TB_Y[i, j] = [0, 2][best_y]

    # Determine which matrix has the best score at (n, m)
    final_scores = [M[n, m], X[n, m], Y[n, m]]
    state = int(np.argmax(final_scores))
    score = final_scores[state]

    # Traceback
    aligned_q = []
    aligned_t = []
    i, j = n, m

    while i > 0 or j > 0:
        if state == 0:  # M
            if i == 0 or j == 0:
                break
            aligned_q.append(query[i - 1])
            aligned_t.append(template[j - 1])
            state = TB_M[i, j]
            i -= 1
            j -= 1
        elif state == 1:  # X — gap in template
            aligned_q.append(query[i - 1])
            aligned_t.append("-")
            state = TB_X[i, j]
            i -= 1
        else:  # Y — gap in query
            aligned_q.append("-")
            aligned_t.append(template[j - 1])
            state = TB_Y[i, j]
            j -= 1

    # Handle remaining residues
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

    # Build mapping: (query_pos, template_pos) for matched positions
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
