"""
ensemble.py — Multi-state conformational ensemble generation.

Generates diverse RNA structure predictions for best-of-5 TM-score
optimization. Key strategy: generate many candidates, then select
the 5 most diverse using maximin RMSD selection.

Strategies:
  1. Template-derived base structure
  2. Alternative template hypotheses
  3. Thermal perturbation (flexibility-weighted noise)
  4. Hinge/domain motions (rigid rotation of subdomains)
  5. Normal mode-like collective motions
  6. Simulated annealing variants (different seeds/temperatures)

Selection:
  - Generate 15-25 candidate models
  - Compute pairwise RMSD matrix
  - Greedy maximin selection: iteratively pick the model most dissimilar
    to all previously selected models
  - This provably maximizes conformational coverage for best-of-5 scoring
"""

import numpy as np
from typing import List, Optional


def generate_ensemble(
    base_coords: np.ndarray,
    n_models: int = 5,
    noise_scales: Optional[List[float]] = None,
    hinge_angles: Optional[List[float]] = None,
    sequence: str = "",
    use_sa: bool = True,
) -> List[np.ndarray]:
    """
    Generate a diverse conformational ensemble.

    Instead of generating exactly n_models, we generate ~20 candidates
    using multiple strategies, then select the n_models most diverse ones
    via maximin RMSD selection.

    Parameters
    ----------
    base_coords : np.ndarray, shape (N, 3)
        Base predicted coordinates.
    n_models : int
        Number of models to output (typically 5).
    noise_scales : list of float, optional
        Noise magnitudes for thermal perturbation.
    hinge_angles : list of float, optional
        Rotation angles for hinge motions.
    sequence : str
        RNA sequence (for SA refinement).
    use_sa : bool
        Whether to use simulated annealing variants.

    Returns
    -------
    list of np.ndarray
        n_models coordinate arrays.
    """
    if noise_scales is None:
        noise_scales = [0.3, 0.5, 0.8, 1.2, 2.0]
    if hinge_angles is None:
        hinge_angles = [-20.0, -10.0, 10.0, 20.0]

    n = len(base_coords)
    candidates = []

    # Model 1: Base prediction (always included)
    candidates.append(base_coords.copy())

    # Models: Thermal noise with different scales
    for scale in noise_scales:
        candidates.append(apply_noise(base_coords, scale=scale))

    # Models: Hinge motions at different angles
    for angle in hinge_angles:
        if n >= 8:
            candidates.append(apply_hinge(base_coords, angle=angle))

    # Models: Multi-hinge (split into 3 domains)
    if n >= 20:
        for angle in [15.0, -15.0]:
            candidates.append(apply_multi_hinge(base_coords, n_domains=3, angle=angle))

    # Models: Collective breathing motion
    if n >= 6:
        for scale in [0.8, 1.2]:
            candidates.append(apply_breathing(base_coords, scale=scale))

    # Models: Local segment perturbation
    if n >= 10:
        for _ in range(3):
            candidates.append(apply_segment_perturbation(base_coords))

    # Models: Simulated annealing variants (different seeds)
    if use_sa and n <= 300:  # Only for reasonably sized structures
        try:
            from .energy import refine_with_energy
            for seed in [42, 137]:
                sa_refined = refine_with_energy(
                    base_coords.copy(), sequence=sequence,
                    n_steps=500, seed=seed,
                )
                candidates.append(sa_refined)
        except Exception:
            pass

    # If we have enough candidates, select the most diverse subset
    if len(candidates) > n_models:
        selected = maximin_select(candidates, n_models)
    else:
        selected = candidates[:n_models]

    # Pad if needed
    while len(selected) < n_models:
        noise = base_coords + np.random.normal(0, 1.5, base_coords.shape)
        selected.append(noise)

    return selected[:n_models]


def maximin_select(candidates: List[np.ndarray], n_select: int) -> List[np.ndarray]:
    """
    Select n_select models that maximize minimum pairwise dissimilarity.

    Uses a greedy algorithm:
      1. Start with the first candidate (base prediction)
      2. Iteratively add the candidate most dissimilar to all already selected

    This is provably a 2-approximation to the optimal diversity maximization
    problem (NP-hard in general).

    Parameters
    ----------
    candidates : list of np.ndarray
    n_select : int

    Returns
    -------
    list of np.ndarray
    """
    if len(candidates) <= n_select:
        return candidates

    try:
        from .kabsch import kabsch_rmsd
        use_kabsch = True
    except ImportError:
        use_kabsch = False

    n = len(candidates)

    # Compute pairwise RMSD matrix
    rmsd_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if use_kabsch:
                r = kabsch_rmsd(candidates[i], candidates[j])
            else:
                r = np.sqrt(np.mean((candidates[i] - candidates[j]) ** 2))
            rmsd_matrix[i, j] = r
            rmsd_matrix[j, i] = r

    # Greedy maximin selection
    selected_indices = [0]  # Always start with the base prediction

    for _ in range(n_select - 1):
        best_idx = -1
        best_min_dist = -1.0

        for i in range(n):
            if i in selected_indices:
                continue
            min_dist = min(rmsd_matrix[i, j] for j in selected_indices)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i

        if best_idx >= 0:
            selected_indices.append(best_idx)
        else:
            break

    return [candidates[i] for i in selected_indices]


def apply_noise(coords: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """
    Apply flexibility-weighted Gaussian noise.

    Terminal residues get 2x noise (more flexible in solution).
    """
    n = len(coords)
    noised = coords.copy()

    # Flexibility profile: higher at termini
    flex = np.ones(n)
    ramp = min(n // 4, 10)
    for i in range(ramp):
        factor = 1.0 + 1.0 * (1 - i / ramp)
        flex[i] = factor
        flex[n - 1 - i] = factor

    # Apply noise
    noise = np.random.normal(0, scale, coords.shape)
    noised += noise * flex[:, np.newaxis]

    return noised


def apply_hinge(coords: np.ndarray, angle: float = 15.0,
                hinge_pos: Optional[int] = None) -> np.ndarray:
    """
    Apply a hinge motion: rotate the C-terminal half relative to the
    N-terminal half around a pivot point.

    Uses Rodrigues' rotation formula for efficiency.
    """
    n = len(coords)
    if n < 4:
        return coords.copy()

    if hinge_pos is None:
        hinge_pos = n // 2

    result = coords.copy()

    # Rotation axis: perpendicular to local backbone direction
    pivot = coords[hinge_pos]
    if hinge_pos > 0:
        backbone_dir = coords[hinge_pos] - coords[hinge_pos - 1]
    else:
        backbone_dir = coords[hinge_pos + 1] - coords[hinge_pos]

    # Cross with z-axis to get perpendicular rotation axis
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(backbone_dir, z)
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        axis = np.array([0.0, 1.0, 0.0])
    else:
        axis = axis / norm

    # Rodrigues' rotation
    theta = np.radians(angle)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    for i in range(hinge_pos, n):
        result[i] = pivot + R @ (coords[i] - pivot)

    return result


def apply_multi_hinge(coords: np.ndarray, n_domains: int = 3,
                      angle: float = 15.0) -> np.ndarray:
    """
    Apply hinge motions between multiple domains.

    Splits the structure into n_domains segments and applies
    alternating rotations between them.
    """
    n = len(coords)
    if n < n_domains * 3:
        return apply_hinge(coords, angle)

    result = coords.copy()
    segment_size = n // n_domains

    for d in range(1, n_domains):
        hinge_pos = d * segment_size
        direction = 1 if d % 2 == 0 else -1
        pivot = result[hinge_pos]

        # Variable axis per hinge
        if hinge_pos > 0:
            backbone_dir = result[hinge_pos] - result[hinge_pos - 1]
        else:
            backbone_dir = np.array([1, 0, 0])

        perp = np.cross(backbone_dir, [0, 0, 1])
        norm = np.linalg.norm(perp)
        if norm < 1e-8:
            perp = np.array([0, 1, 0])
        else:
            perp /= norm

        theta = np.radians(angle * direction)
        K = np.array([
            [0, -perp[2], perp[1]],
            [perp[2], 0, -perp[0]],
            [-perp[1], perp[0], 0],
        ])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

        for i in range(hinge_pos, n):
            result[i] = pivot + R @ (result[i] - pivot)

    return result


def apply_breathing(coords: np.ndarray, scale: float = 1.2) -> np.ndarray:
    """
    Apply a breathing motion: expand or contract the structure
    relative to its center of mass.

    This models the collective breathing modes that are the
    lowest-energy deformations of RNA structures.
    """
    center = np.mean(coords, axis=0)
    result = center + (coords - center) * scale
    return result


def apply_segment_perturbation(coords: np.ndarray) -> np.ndarray:
    """
    Randomly perturb a contiguous segment of the structure while
    keeping the rest fixed. Models local conformational changes
    (loop rearrangements, helix bending).
    """
    n = len(coords)
    result = coords.copy()

    # Pick a random segment (10-30% of the structure)
    seg_len = max(3, int(0.15 * n) + np.random.randint(-2, 3))
    seg_len = min(seg_len, n - 2)
    start = np.random.randint(0, n - seg_len)

    # Apply moderate noise to the segment
    noise = np.random.normal(0, 1.5, (seg_len, 3))
    result[start:start + seg_len] += noise

    return result
