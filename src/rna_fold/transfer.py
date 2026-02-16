"""
transfer.py — Coordinate transfer from template to query with gap filling.

Core TBM step: use alignment mapping to copy template C1' coordinates
to matching query positions, then fill gaps geometrically.
"""

import numpy as np
from typing import List, Optional, Tuple

# Canonical C1'-C1' distance in RNA (Angstroms)
C1_C1_DISTANCE = 5.9


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

    # Step 2: Fill gaps
    coords = _fill_gaps(coords, known_mask)

    return coords


def generate_de_novo(sequence: str, geometry: str = "helix") -> np.ndarray:
    """
    Generate de novo coordinates when no template is available.

    Parameters
    ----------
    sequence : str
        RNA sequence.
    geometry : str
        "helix" for A-form helix, "linear" for straight chain.

    Returns
    -------
    np.ndarray, shape (N, 3)
    """
    n = len(sequence)
    if n == 0:
        return np.empty((0, 3))

    coords = np.zeros((n, 3))

    if geometry == "helix":
        # A-form RNA helix parameters
        # Rise per residue: ~2.8 Å, radius: ~9.4 Å, twist: ~32.7°/residue
        rise = 2.8
        radius = 9.4
        twist_deg = 32.7
        twist_rad = np.radians(twist_deg)

        for i in range(n):
            angle = i * twist_rad
            coords[i] = [
                radius * np.cos(angle),
                radius * np.sin(angle),
                i * rise,
            ]
    else:  # linear
        for i in range(n):
            coords[i] = [i * C1_C1_DISTANCE, 0.0, 0.0]

    return coords


def _fill_gaps(coords: np.ndarray, known_mask: np.ndarray) -> np.ndarray:
    """
    Fill gaps in coordinate array using geometric interpolation.
    Enforces approximate C1'-C1' distance constraints.
    """
    n = len(coords)
    if n == 0:
        return coords

    # If nothing is known, generate de novo
    if not np.any(known_mask):
        return generate_de_novo("A" * n)

    # Find runs of unknown positions
    filled = coords.copy()

    # Find first and last known positions
    known_indices = np.where(known_mask)[0]
    first_known = known_indices[0]
    last_known = known_indices[-1]

    # Fill leading gap (before first known)
    if first_known > 0:
        direction = _get_extension_direction(filled, first_known, forward=False)
        for i in range(first_known - 1, -1, -1):
            filled[i] = filled[i + 1] + direction * C1_C1_DISTANCE
            # Add slight random deviation to avoid perfectly straight chains
            filled[i] += np.random.normal(0, 0.3, 3)

    # Fill trailing gap (after last known)
    if last_known < n - 1:
        direction = _get_extension_direction(filled, last_known, forward=True)
        for i in range(last_known + 1, n):
            filled[i] = filled[i - 1] + direction * C1_C1_DISTANCE
            filled[i] += np.random.normal(0, 0.3, 3)

    # Fill internal gaps by interpolation
    gap_start = None
    for i in range(n):
        if np.isnan(filled[i, 0]):
            if gap_start is None:
                gap_start = i
        else:
            if gap_start is not None:
                # We have a gap from gap_start to i-1
                _interpolate_gap(filled, gap_start, i - 1)
                gap_start = None

    return filled


def _interpolate_gap(coords: np.ndarray, start: int, end: int) -> None:
    """
    Fill a gap [start, end] using arc interpolation between
    coords[start-1] and coords[end+1], enforcing C1'-C1' distance.
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

    # Linear interpolation as starting point
    for i in range(gap_len):
        t = (i + 1) / total_segments
        coords[start + i] = p_left + t * (p_right - p_left)

    # Now enforce C1'-C1' distance constraints iteratively
    _enforce_distances(coords, left, right)


def _enforce_distances(coords: np.ndarray, left: int, right: int,
                       iterations: int = 20) -> None:
    """
    Iteratively adjust residue positions to enforce C1'-C1' distances
    between left and right (inclusive).
    """
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

            # Only move non-boundary points
            if i > left:
                coords[i] += correction
            if i + 1 < right:
                coords[i + 1] -= correction


def _get_extension_direction(coords: np.ndarray, idx: int,
                             forward: bool) -> np.ndarray:
    """Get a unit direction vector for extending beyond known coordinates."""
    # Try to use the local chain direction
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
