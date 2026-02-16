"""
energy.py — Energy function and simulated annealing for RNA structure refinement.

Implements a coarse-grained energy function for C1' backbone coordinates with:
  - Bond length potential (C1'-C1' distance)
  - Bond angle potential (consecutive bond angles)
  - Steric exclusion (van der Waals-like repulsion)
  - Compactness bias (radius of gyration restraint)
  - Base-pair distance restraints (from Nussinov prediction)

Simulated annealing optimization explores conformational space more
effectively than simple gradient-based refinement.
"""

import numpy as np
from typing import List, Optional, Tuple


# RNA backbone geometry (C1' coarse-grained)
IDEAL_BOND = 5.9        # C1'-C1' distance (Angstroms)
IDEAL_ANGLE = 120.0     # Bond angle (degrees)
CLASH_DIST = 3.5        # Minimum non-bonded distance
BASE_PAIR_DIST = 10.5   # Ideal C1'-C1' distance for base-paired residues
BOND_K = 50.0           # Bond spring constant
ANGLE_K = 5.0           # Angle spring constant
CLASH_K = 100.0         # Clash penalty
PAIR_K = 2.0            # Base pair restraint strength
RG_K = 0.01             # Radius of gyration restraint


def compute_energy(
    coords: np.ndarray,
    base_pairs: Optional[List[Tuple[int, int]]] = None,
    target_rg: Optional[float] = None,
) -> float:
    """
    Compute the coarse-grained energy of an RNA C1' backbone.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)
    base_pairs : list of (i, j), optional
        Predicted base pairs from Nussinov.
    target_rg : float, optional
        Target radius of gyration. If None, estimated from sequence length.

    Returns
    -------
    float
        Total energy (lower is better).
    """
    n = len(coords)
    if n < 2:
        return 0.0

    energy = 0.0

    # 1. Bond length potential
    for i in range(n - 1):
        d = np.linalg.norm(coords[i + 1] - coords[i])
        energy += BOND_K * (d - IDEAL_BOND) ** 2

    # 2. Bond angle potential (i, i+1, i+2)
    ideal_angle_rad = np.radians(IDEAL_ANGLE)
    for i in range(n - 2):
        v1 = coords[i] - coords[i + 1]
        v2 = coords[i + 2] - coords[i + 1]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            continue
        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        angle = np.arccos(cos_angle)
        energy += ANGLE_K * (angle - ideal_angle_rad) ** 2

    # 3. Higher-order distance constraints (i, i+2 and i, i+3)
    # These enforce proper backbone curvature
    for i in range(n - 2):
        d = np.linalg.norm(coords[i + 2] - coords[i])
        # Expected i,i+2 distance for ~120° angle ≈ 10.2 Å
        ideal_d2 = 10.2
        energy += 2.0 * (d - ideal_d2) ** 2

    for i in range(n - 3):
        d = np.linalg.norm(coords[i + 3] - coords[i])
        # Expected i,i+3 distance ≈ 13.5 Å
        ideal_d3 = 13.5
        energy += 1.0 * (d - ideal_d3) ** 2

    # 4. Steric exclusion for non-bonded atoms
    # Only check within a window to keep it O(n) not O(n²)
    window = min(30, n)
    for i in range(n):
        for j in range(i + 4, min(i + window, n)):
            d = np.linalg.norm(coords[j] - coords[i])
            if d < CLASH_DIST:
                energy += CLASH_K * (CLASH_DIST - d) ** 2

    # 5. Base pair distance restraints
    if base_pairs:
        for i, j in base_pairs:
            if i < n and j < n:
                d = np.linalg.norm(coords[j] - coords[i])
                energy += PAIR_K * (d - BASE_PAIR_DIST) ** 2

    # 6. Radius of gyration restraint (compactness)
    if target_rg is None:
        # Empirical: Rg ≈ 3.0 * N^0.4 for compact RNA
        target_rg = 3.0 * (n ** 0.4)

    center = np.mean(coords, axis=0)
    rg = np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1)))
    energy += RG_K * n * (rg - target_rg) ** 2

    return energy


def simulated_annealing(
    coords: np.ndarray,
    base_pairs: Optional[List[Tuple[int, int]]] = None,
    n_steps: int = 2000,
    temp_start: float = 5.0,
    temp_end: float = 0.1,
    step_size: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Refine coordinates using simulated annealing with the coarse-grained
    energy function.

    This explores conformational space much more effectively than gradient
    descent, allowing escape from local minima.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)
        Initial coordinates.
    base_pairs : list of (i, j), optional
        Predicted base pairs for distance restraints.
    n_steps : int
        Number of SA steps.
    temp_start, temp_end : float
        Temperature schedule (exponential cooling).
    step_size : float
        Maximum perturbation per step (Angstroms).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (N, 3)
        Refined coordinates.
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(coords)
    if n < 2:
        return coords.copy()

    # Cap steps for very long sequences to stay within time budget
    if n > 500:
        n_steps = min(n_steps, 500)
    if n > 1000:
        n_steps = min(n_steps, 200)

    best = coords.copy()
    current = coords.copy()
    best_energy = compute_energy(best, base_pairs)
    current_energy = best_energy

    # Temperature schedule
    cooling_rate = (temp_end / temp_start) ** (1.0 / max(n_steps, 1))

    temp = temp_start
    accepted = 0

    for step in range(n_steps):
        # Pick a random residue (or small window) to perturb
        if np.random.random() < 0.3:
            # Move a contiguous block (3-7 residues)
            block_size = min(np.random.randint(3, 8), n)
            start_idx = np.random.randint(0, n - block_size + 1)
            perturbation = np.random.normal(0, step_size, (block_size, 3))
            trial = current.copy()
            trial[start_idx:start_idx + block_size] += perturbation
        else:
            # Move a single residue
            idx = np.random.randint(0, n)
            trial = current.copy()
            trial[idx] += np.random.normal(0, step_size, 3)

        trial_energy = compute_energy(trial, base_pairs)
        delta = trial_energy - current_energy

        # Metropolis criterion
        if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-8)):
            current = trial
            current_energy = trial_energy
            accepted += 1

            if current_energy < best_energy:
                best = current.copy()
                best_energy = current_energy

        temp *= cooling_rate

    return best


def refine_with_energy(
    coords: np.ndarray,
    sequence: str = "",
    n_steps: int = 2000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    High-level refinement: predict secondary structure, then run SA.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)
    sequence : str
        RNA sequence for Nussinov prediction.
    n_steps : int
    seed : int, optional

    Returns
    -------
    np.ndarray, shape (N, 3)
    """
    base_pairs = []
    if sequence:
        try:
            from .nussinov import predict_base_pairs
            # Guard against very long sequences (Nussinov is O(n³))
            if len(sequence) <= 500:
                base_pairs = predict_base_pairs(sequence)
        except Exception:
            pass

    return simulated_annealing(
        coords,
        base_pairs=base_pairs,
        n_steps=n_steps,
        seed=seed,
    )
