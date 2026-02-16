"""
nussinov.py — RNA secondary structure prediction via the Nussinov algorithm.

A minimal dynamic programming implementation that predicts RNA base pairs
from sequence. Used to guide gap filling by identifying helical regions.
"""

import numpy as np
from typing import List, Set, Tuple

# Canonical RNA base pairs
_PAIRS = {
    ("A", "U"), ("U", "A"),
    ("G", "C"), ("C", "G"),
    ("G", "U"), ("U", "G"),  # Wobble pair
}

MIN_HAIRPIN_LOOP = 3  # Minimum number of unpaired bases in a hairpin


def predict_base_pairs(sequence: str) -> List[Tuple[int, int]]:
    """
    Predict RNA secondary structure using the Nussinov algorithm.

    Parameters
    ----------
    sequence : str
        RNA sequence (ACGU).

    Returns
    -------
    list of (int, int)
        Base pairs as zero-indexed tuples (i, j) where i < j.
    """
    n = len(sequence)
    if n < MIN_HAIRPIN_LOOP + 2:
        return []

    # Speed guard: for long sequences, Nussinov is O(n³).
    # At n=500, that's ~125M operations (OK). At n=2000, it's 8B (too slow).
    # For long sequences, split into windows and merge.
    MAX_NUSSINOV_LEN = 500
    if n > MAX_NUSSINOV_LEN:
        return _windowed_nussinov(sequence, window_size=MAX_NUSSINOV_LEN, overlap=50)

    # DP table
    dp = np.zeros((n, n), dtype=np.int32)

    # Fill DP table
    for span in range(MIN_HAIRPIN_LOOP + 1, n):
        for i in range(n - span):
            j = i + span

            # Case 1: j is unpaired
            dp[i, j] = dp[i, j - 1]

            # Case 2: j pairs with some k (i <= k < j - MIN_HAIRPIN_LOOP)
            for k in range(i, j - MIN_HAIRPIN_LOOP):
                if _can_pair(sequence[k], sequence[j]):
                    score = 1
                    left = dp[i, k - 1] if k > i else 0
                    right = dp[k + 1, j - 1] if k + 1 <= j - 1 else 0
                    total = left + score + right
                    dp[i, j] = max(dp[i, j], total)

    # Traceback
    pairs = []
    _traceback(dp, sequence, 0, n - 1, pairs)

    return sorted(pairs)


def get_paired_regions(pairs: List[Tuple[int, int]], n: int) -> List[dict]:
    """
    Identify contiguous helical (stem) regions from base pairs.

    Parameters
    ----------
    pairs : list of (int, int)
        Base pairs.
    n : int
        Sequence length.

    Returns
    -------
    list of dict
        Each dict has "pairs" (list of (i,j)), "start", "end" fields.
    """
    if not pairs:
        return []

    # Group consecutive stacking pairs into stems
    stems = []
    current_stem = [pairs[0]]

    for k in range(1, len(pairs)):
        prev_i, prev_j = pairs[k - 1]
        curr_i, curr_j = pairs[k]
        # Consecutive stacking: i increases by 1, j decreases by 1
        if curr_i == prev_i + 1 and curr_j == prev_j - 1:
            current_stem.append(pairs[k])
        else:
            if len(current_stem) >= 2:  # At least 2 stacking pairs
                stems.append(current_stem)
            current_stem = [pairs[k]]

    if len(current_stem) >= 2:
        stems.append(current_stem)

    regions = []
    for stem in stems:
        all_indices = set()
        for i, j in stem:
            all_indices.add(i)
            all_indices.add(j)
        regions.append({
            "pairs": stem,
            "start": min(all_indices),
            "end": max(all_indices),
            "indices": sorted(all_indices),
        })

    return regions


def is_in_helix(position: int, pairs: List[Tuple[int, int]]) -> bool:
    """Check if a residue position is part of a base pair."""
    paired_positions = set()
    for i, j in pairs:
        paired_positions.add(i)
        paired_positions.add(j)
    return position in paired_positions


def get_pair_partner(position: int,
                     pairs: List[Tuple[int, int]]) -> int:
    """Get the pairing partner of a position, or -1 if unpaired."""
    for i, j in pairs:
        if i == position:
            return j
        if j == position:
            return i
    return -1


def _windowed_nussinov(
    sequence: str,
    window_size: int = 500,
    overlap: int = 50,
) -> List[Tuple[int, int]]:
    """
    Windowed Nussinov for long sequences.

    Splits the sequence into overlapping windows, runs the standard
    Nussinov on each window, then merges and deduplicates results.
    Cannot find very long-range pairs (>window_size apart), but those
    are rare in RNA anyway.
    """
    n = len(sequence)
    all_pairs = set()
    step = window_size - overlap

    for start in range(0, n, step):
        end = min(start + window_size, n)
        subseq = sequence[start:end]

        if len(subseq) < MIN_HAIRPIN_LOOP + 2:
            continue

        # Run standard Nussinov on the window
        dp = np.zeros((len(subseq), len(subseq)), dtype=np.int32)
        for span in range(MIN_HAIRPIN_LOOP + 1, len(subseq)):
            for i in range(len(subseq) - span):
                j = i + span
                dp[i, j] = dp[i, j - 1]
                for k in range(i, j - MIN_HAIRPIN_LOOP):
                    if _can_pair(subseq[k], subseq[j]):
                        left = dp[i, k - 1] if k > i else 0
                        right = dp[k + 1, j - 1] if k + 1 <= j - 1 else 0
                        dp[i, j] = max(dp[i, j], left + 1 + right)

        local_pairs = []
        _traceback(dp, subseq, 0, len(subseq) - 1, local_pairs)

        # Offset pairs to global coordinates
        for i, j in local_pairs:
            all_pairs.add((start + i, start + j))

    return sorted(all_pairs)


def _can_pair(a: str, b: str) -> bool:
    """Check if two bases can form a canonical pair."""
    return (a.upper(), b.upper()) in _PAIRS


def _traceback(dp: np.ndarray, seq: str, i: int, j: int,
               pairs: List[Tuple[int, int]]) -> None:
    """Recursive traceback to recover base pairs."""
    if i >= j:
        return

    # Case 1: j is unpaired
    if dp[i, j] == dp[i, j - 1]:
        _traceback(dp, seq, i, j - 1, pairs)
        return

    # Case 2: j pairs with some k
    for k in range(i, j - MIN_HAIRPIN_LOOP):
        if _can_pair(seq[k], seq[j]):
            left = dp[i, k - 1] if k > i else 0
            right = dp[k + 1, j - 1] if k + 1 <= j - 1 else 0
            if dp[i, j] == left + 1 + right:
                pairs.append((k, j))
                if k > i:
                    _traceback(dp, seq, i, k - 1, pairs)
                if k + 1 <= j - 1:
                    _traceback(dp, seq, k + 1, j - 1, pairs)
                return
