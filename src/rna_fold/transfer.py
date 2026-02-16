"""
transfer.py — Coordinate transfer from template to query with gap filling.

Core TBM step: use alignment mapping to copy template C1' coordinates
to matching query positions, then fill gaps geometrically.

Improvements:
  - Nussinov-guided gap filling (helical geometry for paired residues)
  - Multi-template consensus (weighted coordinate averaging)
  - Better de novo backbone (constrained random walk with angle constraints)
"""

import numpy as np
from typing import List, Optional, Tuple

# Canonical C1'-C1' distance in RNA (Angstroms)
C1_C1_DISTANCE = 5.9
# A-form helix parameters
HELIX_RISE = 2.8
HELIX_RADIUS = 9.4
HELIX_TWIST_DEG = 32.7
# Bond angle between consecutive C1' atoms (typical ~120°)
BOND_ANGLE_RAD = np.radians(120.0)


def transfer_coordinates(
    query_sequence: str,
    template_coords: np.ndarray,
    mapping: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Transfer template coordinates to query positions using alignment mapping,
    then fill gaps with geometric interpolation.

    Parameters
    ----------
    query_sequence : str
        Full query RNA sequence.
    template_coords : np.ndarray, shape (M, 3)
        Template C1' coordinates.
    mapping : list of (query_pos, template_pos)
        Alignment mapping (zero-indexed).

    Returns
    -------
    np.ndarray, shape (N, 3)
        Full predicted coordinates for every query residue.
    """
    n = len(query_sequence)
    coords = np.full((n, 3), np.nan)
    known_mask = np.zeros(n, dtype=bool)

    # Step 1: Direct copy at matched positions
    for q_pos, t_pos in mapping:
        if q_pos < n and t_pos < len(template_coords):
            coords[q_pos] = template_coords[t_pos]
            known_mask[q_pos] = True

    # Step 2: Fill gaps (with optional Nussinov guidance)
    try:
        from .nussinov import predict_base_pairs
        pairs = predict_base_pairs(query_sequence)
    except Exception:
        pairs = []

    coords = _fill_gaps(coords, known_mask, pairs)

    return coords


def multi_template_consensus(
    query_sequence: str,
    template_results: List[dict],
    template_coords_list: List[Tuple[np.ndarray, List[int]]],
) -> np.ndarray:
    """
    Blend coordinates from multiple templates using identity-weighted averaging.

    This is a key improvement: instead of trusting a single template, we
    average across top-N templates weighted by sequence identity. Dramatically
    reduces noise from any single template error.

    Parameters
    ----------
    query_sequence : str
        Query RNA sequence.
    template_results : list of dict
        Template search results with 'identity', 'mapping' fields.
    template_coords_list : list of (coords, resids)
        Corresponding template coordinates.

    Returns
    -------
    np.ndarray, shape (N, 3)
    """
    n = len(query_sequence)
    weighted_sum = np.zeros((n, 3), dtype=np.float64)
    weight_sum = np.zeros(n, dtype=np.float64)

    for result, (t_coords, t_resids) in zip(template_results, template_coords_list):
        identity = result.get("identity", 0.5)
        coverage = result.get("coverage", 0.5)
        weight = identity * coverage  # Combined quality weight
        mapping = result["mapping"]

        for q_pos, t_pos in mapping:
            if q_pos < n and t_pos < len(t_coords):
                weighted_sum[q_pos] += weight * t_coords[t_pos]
                weight_sum[q_pos] += weight

    # Compute weighted average where available
    coords = np.full((n, 3), np.nan)
    known_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        if weight_sum[i] > 1e-8:
            coords[i] = weighted_sum[i] / weight_sum[i]
            known_mask[i] = True

    # Fill remaining gaps
    try:
        from .nussinov import predict_base_pairs
        pairs = predict_base_pairs(query_sequence)
    except Exception:
        pairs = []

    coords = _fill_gaps(coords, known_mask, pairs)
    return coords


def generate_de_novo(sequence: str, geometry: str = "compact") -> np.ndarray:
    """
    Generate de novo coordinates when no template is available.

    Parameters
    ----------
    sequence : str
        RNA sequence.
    geometry : str
        "compact" for constrained random walk (best for TM-score),
        "helix" for A-form helix,
        "linear" for straight chain.

    Returns
    -------
    np.ndarray, shape (N, 3)
    """
    n = len(sequence)
    if n == 0:
        return np.empty((0, 3))

    if geometry == "compact":
        return _generate_compact_walk(sequence)
    elif geometry == "helix":
        return _generate_helix(n)
    else:  # linear
        coords = np.zeros((n, 3))
        for i in range(n):
            coords[i] = [i * C1_C1_DISTANCE, 0.0, 0.0]
        return coords


def _generate_helix(n: int) -> np.ndarray:
    """Generate A-form RNA helix coordinates."""
    coords = np.zeros((n, 3))
    twist_rad = np.radians(HELIX_TWIST_DEG)
    for i in range(n):
        angle = i * twist_rad
        coords[i] = [
            HELIX_RADIUS * np.cos(angle),
            HELIX_RADIUS * np.sin(angle),
            i * HELIX_RISE,
        ]
    return coords


def _generate_compact_walk(sequence: str) -> np.ndarray:
    """
    Generate a compact random walk structure with realistic RNA geometry.

    This produces structures that are:
      - Compact (good for TM-score against diverse targets)
      - Respect C1'-C1' bond distance (~5.9 Å)
      - Respect bond angles (~120° between consecutive bonds)
      - Avoid steric clashes (minimum 3.5 Å between non-bonded atoms)

    Much better than a straight helix as a de novo fallback because it explores
    the conformational space more effectively.
    """
    n = len(sequence)
    coords = np.zeros((n, 3))

    if n <= 1:
        return coords

    # First two residues define the initial direction
    coords[1] = [C1_C1_DISTANCE, 0.0, 0.0]

    # Use Nussinov to predict base pairs for structure guidance
    try:
        from .nussinov import predict_base_pairs
        pairs = predict_base_pairs(sequence)
        paired = {}
        for i, j in pairs:
            paired[i] = j
            paired[j] = i
    except Exception:
        paired = {}

    for i in range(2, n):
        prev_dir = coords[i - 1] - coords[i - 2]
        prev_dir_norm = np.linalg.norm(prev_dir)
        if prev_dir_norm < 1e-8:
            prev_dir = np.array([1.0, 0.0, 0.0])
        else:
            prev_dir = prev_dir / prev_dir_norm

        # Generate a random direction with bond angle constraint
        # The new bond should make ~120° angle with the previous
        best_pos = None
        best_score = -1e9

        for attempt in range(10):
            # Random rotation around the previous direction
            rand_axis = np.random.normal(0, 1, 3)
            rand_axis -= np.dot(rand_axis, prev_dir) * prev_dir
            rand_norm = np.linalg.norm(rand_axis)
            if rand_norm < 1e-8:
                rand_axis = np.array([0, 0, 1])
            else:
                rand_axis /= rand_norm

            # Rotate previous direction by ~60° around the random perpendicular
            # (supplementary to 120° bond angle)
            perturb_angle = np.radians(60 + np.random.normal(0, 15))
            new_dir = (np.cos(perturb_angle) * prev_dir +
                       np.sin(perturb_angle) * rand_axis)
            new_dir /= np.linalg.norm(new_dir)

            candidate = coords[i - 1] + new_dir * C1_C1_DISTANCE

            # Score: prefer compact structures without clashes
            score = 0
            min_dist = float('inf')
            for j in range(max(0, i - 30), i - 2):
                d = np.linalg.norm(candidate - coords[j])
                min_dist = min(min_dist, d)
                if d < 3.5:  # Steric clash
                    score -= 100
                elif d < 8.0:  # Prefer moderately close (compact)
                    score += 1.0

            # If this residue is paired, bias toward its partner
            if i in paired and paired[i] < i:
                partner_pos = coords[paired[i]]
                partner_dist = np.linalg.norm(candidate - partner_pos)
                # Ideal base pair distance ~8-12 Å for C1'-C1'
                if 7.0 < partner_dist < 13.0:
                    score += 10.0

            if min_dist >= 3.5 or best_pos is None:
                if score > best_score:
                    best_score = score
                    best_pos = candidate

        coords[i] = best_pos if best_pos is not None else coords[i - 1] + prev_dir * C1_C1_DISTANCE

    return coords


def _fill_gaps(coords: np.ndarray, known_mask: np.ndarray,
               pairs: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
    """
    Fill gaps in coordinate array using geometric interpolation.
    Uses Nussinov base pairs to guide helical geometry where appropriate.
    """
    n = len(coords)
    if n == 0:
        return coords

    if pairs is None:
        pairs = []

    # If nothing is known, generate de novo
    if not np.any(known_mask):
        return generate_de_novo("A" * n, geometry="compact")

    filled = coords.copy()
    known_indices = np.where(known_mask)[0]
    first_known = known_indices[0]
    last_known = known_indices[-1]

    # Fill leading gap
    if first_known > 0:
        direction = _get_extension_direction(filled, first_known, forward=False)
        for i in range(first_known - 1, -1, -1):
            filled[i] = filled[i + 1] + direction * C1_C1_DISTANCE
            filled[i] += np.random.normal(0, 0.3, 3)

    # Fill trailing gap
    if last_known < n - 1:
        direction = _get_extension_direction(filled, last_known, forward=True)
        for i in range(last_known + 1, n):
            filled[i] = filled[i - 1] + direction * C1_C1_DISTANCE
            filled[i] += np.random.normal(0, 0.3, 3)

    # Fill internal gaps
    gap_start = None
    for i in range(n):
        if np.isnan(filled[i, 0]):
            if gap_start is None:
                gap_start = i
        else:
            if gap_start is not None:
                _interpolate_gap(filled, gap_start, i - 1, pairs)
                gap_start = None

    return filled


def _interpolate_gap(coords: np.ndarray, start: int, end: int,
                     pairs: Optional[List[Tuple[int, int]]] = None) -> None:
    """
    Fill a gap [start, end] using interpolation.
    If Nussinov pairs indicate helical regions within the gap,
    use helical geometry for those residues.
    """
    n = len(coords)
    left = start - 1
    right = end + 1

    if left < 0 or right >= n:
        return

    p_left = coords[left]
    p_right = coords[right]
    gap_len = end - start + 1
    total_segments = gap_len + 1

    # Check if any gap residues are in base pairs
    gap_in_helix = False
    if pairs:
        gap_positions = set(range(start, end + 1))
        for i, j in pairs:
            if i in gap_positions or j in gap_positions:
                gap_in_helix = True
                break

    if gap_in_helix and gap_len >= 4:
        # Use helical interpolation: generate a helix segment and
        # superpose its endpoints onto the flanking known coordinates
        helix = _generate_helix(gap_len + 2)
        # Scale and rotate helix to fit between left and right anchors
        helix_start = helix[0]
        helix_end = helix[-1]
        target_vec = p_right - p_left
        helix_vec = helix_end - helix_start
        target_len = np.linalg.norm(target_vec)
        helix_len = np.linalg.norm(helix_vec)

        if helix_len > 1e-8 and target_len > 1e-8:
            scale = target_len / helix_len
            for i in range(gap_len):
                t = (i + 1) / (gap_len + 1)
                # Blend between scaled helix and linear interpolation
                helix_pos = p_left + scale * (helix[i + 1] - helix_start)
                linear_pos = p_left + t * target_vec
                coords[start + i] = 0.6 * helix_pos + 0.4 * linear_pos
        else:
            # Fallback to linear
            for i in range(gap_len):
                t = (i + 1) / total_segments
                coords[start + i] = p_left + t * (p_right - p_left)
    else:
        # Linear interpolation
        for i in range(gap_len):
            t = (i + 1) / total_segments
            coords[start + i] = p_left + t * (p_right - p_left)

    # Enforce C1'-C1' distance constraints
    _enforce_distances(coords, left, right)


def _enforce_distances(coords: np.ndarray, left: int, right: int,
                       iterations: int = 25) -> None:
    """Iteratively adjust positions to enforce C1'-C1' distances."""
    for _ in range(iterations):
        for i in range(left, right):
            p1 = coords[i]
            p2 = coords[i + 1]
            diff = p2 - p1
            dist = np.linalg.norm(diff)
            if dist < 1e-8:
                diff = np.random.normal(0, 1, 3)
                dist = np.linalg.norm(diff)

            correction = (dist - C1_C1_DISTANCE) / (2 * dist) * diff

            if i > left:
                coords[i] += correction
            if i + 1 < right:
                coords[i + 1] -= correction


def _get_extension_direction(coords: np.ndarray, idx: int,
                             forward: bool) -> np.ndarray:
    """Get a unit direction vector for extending beyond known coordinates."""
    if forward and idx > 0:
        direction = coords[idx] - coords[idx - 1]
    elif not forward and idx < len(coords) - 1:
        known = np.where(~np.isnan(coords[idx:, 0]))[0]
        if len(known) >= 2:
            i2 = idx + known[1]
            direction = coords[idx] - coords[i2]
        else:
            direction = np.array([0.0, 0.0, -C1_C1_DISTANCE])
    else:
        direction = np.array([C1_C1_DISTANCE, 0.0, 0.0])

    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction = direction / norm

    return direction
