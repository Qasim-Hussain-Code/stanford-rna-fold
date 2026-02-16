"""
refine.py — Structure refinement through distance regularization and smoothing.

Applies physics-inspired post-processing to improve predicted coordinates:
  1. Gaussian smoothing to remove sharp kinks
  2. Distance regularization to enforce consistent C1'-C1' spacing
  3. Steric clash removal
"""

import numpy as np
from typing import Optional

C1_C1_DISTANCE = 5.9  # Angstroms
C1_C1_TOLERANCE = 1.0  # Allowed deviation from ideal distance


def refine_coordinates(
    coords: np.ndarray,
    target_distance: float = C1_C1_DISTANCE,
    smooth_window: int = 3,
    regularize_steps: int = 30,
    smooth_weight: float = 0.3,
) -> np.ndarray:
    """
    Refine predicted coordinates with smoothing and distance regularization.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)
        Predicted C1' coordinates.
    target_distance : float
        Target inter-residue distance.
    smooth_window : int
        Gaussian smoothing window size.
    regularize_steps : int
        Number of distance regularization iterations.
    smooth_weight : float
        Weight for smoothing (0 = no smoothing, 1 = full smoothing).

    Returns
    -------
    np.ndarray, shape (N, 3)
        Refined coordinates.
    """
    if len(coords) < 2:
        return coords.copy()

    refined = coords.copy()

    # Step 1: Light Gaussian smoothing
    refined = _smooth(refined, window=smooth_window, weight=smooth_weight)

    # Step 2: Distance regularization
    refined = _regularize_distances(refined, target_distance, regularize_steps)

    # Step 3: Remove steric clashes (residues too close together)
    refined = _remove_clashes(refined, min_distance=3.0)

    return refined


def _smooth(coords: np.ndarray, window: int = 3,
            weight: float = 0.3) -> np.ndarray:
    """Apply weighted Gaussian smoothing while preserving endpoints."""
    n = len(coords)
    if n <= window:
        return coords.copy()

    smoothed = coords.copy()
    half_w = window // 2

    for i in range(half_w, n - half_w):
        neighbors = coords[i - half_w:i + half_w + 1]
        avg = np.mean(neighbors, axis=0)
        smoothed[i] = (1 - weight) * coords[i] + weight * avg

    return smoothed


def _regularize_distances(
    coords: np.ndarray,
    target: float,
    iterations: int,
) -> np.ndarray:
    """
    Iteratively adjust positions to make consecutive C1'-C1' distances
    closer to the target value. Uses a spring-like force model.
    """
    refined = coords.copy()
    n = len(refined)

    for step in range(iterations):
        # Decrease learning rate over iterations
        lr = 0.3 * (1 - step / iterations)

        for i in range(n - 1):
            diff = refined[i + 1] - refined[i]
            dist = np.linalg.norm(diff)

            if dist < 1e-8:
                # Degenerate: add random displacement
                refined[i + 1] += np.random.normal(0, 0.5, 3)
                continue

            error = dist - target
            correction = lr * error * diff / dist

            # Move both residues (except endpoints which are more anchored)
            w_i = 0.3 if (i == 0) else 0.5
            w_j = 0.3 if (i + 1 == n - 1) else 0.5

            refined[i] += w_i * correction
            refined[i + 1] -= w_j * correction

    return refined


def _remove_clashes(
    coords: np.ndarray,
    min_distance: float = 3.0,
    iterations: int = 10,
) -> np.ndarray:
    """
    Remove steric clashes: push apart non-consecutive residues
    that are closer than min_distance.
    """
    refined = coords.copy()
    n = len(refined)

    if n < 4:
        return refined

    for _ in range(iterations):
        clash_found = False
        for i in range(n):
            for j in range(i + 3, min(i + 20, n)):  # Check nearby non-bonded
                diff = refined[j] - refined[i]
                dist = np.linalg.norm(diff)
                if dist < min_distance and dist > 1e-8:
                    clash_found = True
                    push = (min_distance - dist) / (2 * dist) * diff
                    refined[i] -= push * 0.5
                    refined[j] += push * 0.5
        if not clash_found:
            break

    return refined
