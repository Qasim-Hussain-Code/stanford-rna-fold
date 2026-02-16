"""
kabsch.py — Structural superposition via SVD (Kabsch algorithm).

Used for:
  - Aligning multiple templates before consensus averaging
  - Computing RMSD between candidate models for diversity selection
  - Superposing fragments during assembly
"""

import numpy as np
from typing import Tuple


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Compute RMSD between two coordinate sets after optimal superposition.

    Parameters
    ----------
    P, Q : np.ndarray, shape (N, 3)
        Two coordinate sets of equal size.

    Returns
    -------
    float
        RMSD after optimal rigid-body alignment.
    """
    P_aligned, Q_aligned, rmsd = kabsch_align(P, Q)
    return rmsd


def kabsch_align(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Optimally superpose P onto Q using the Kabsch algorithm (SVD).

    Parameters
    ----------
    P : np.ndarray, shape (N, 3)
        Mobile coordinate set (will be rotated/translated).
    Q : np.ndarray, shape (N, 3)
        Reference coordinate set (stays fixed).

    Returns
    -------
    P_aligned : np.ndarray, shape (N, 3)
        P after optimal rotation + translation.
    Q : np.ndarray, shape (N, 3)
        Reference (unchanged).
    rmsd : float
        RMSD after alignment.
    """
    n = len(P)
    if n == 0:
        return P.copy(), Q.copy(), 0.0
    if n != len(Q):
        min_n = min(n, len(Q))
        P = P[:min_n]
        Q = Q[:min_n]
        n = min_n

    # Center both sets
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # Compute cross-covariance matrix
    H = P_centered.T @ Q_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.eye(3)
    sign_matrix[2, 2] = np.sign(d)

    # Optimal rotation
    R = Vt.T @ sign_matrix @ U.T

    # Apply rotation and translation
    P_aligned = (P_centered @ R.T) + centroid_Q

    # RMSD
    diff = P_aligned - Q
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))

    return P_aligned, Q, rmsd


def compute_tm_score(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Compute an approximate TM-score between two structures.

    This is not the official TM-score (which requires optimal superposition
    over all possible subsets), but a fast approximation using the Kabsch
    alignment. Useful for self-assessment and ensemble selection.

    Parameters
    ----------
    P, Q : np.ndarray, shape (N, 3)
        Two coordinate sets.

    Returns
    -------
    float
        Approximate TM-score in [0, 1].
    """
    n = min(len(P), len(Q))
    if n < 3:
        return 0.0

    P = P[:n]
    Q = Q[:n]

    # Align
    P_aligned, _, _ = kabsch_align(P, Q)

    # TM-score formula
    # d0 depends on target length
    d0 = 1.24 * (n - 15) ** (1.0 / 3.0) - 1.8
    d0 = max(d0, 0.5)  # Floor

    # Per-residue distances after alignment
    di = np.sqrt(np.sum((P_aligned - Q) ** 2, axis=1))

    # TM-score
    tm = np.sum(1.0 / (1.0 + (di / d0) ** 2)) / n

    return float(tm)


def pairwise_rmsd_matrix(models: list) -> np.ndarray:
    """
    Compute pairwise RMSD matrix between a list of coordinate arrays.

    Parameters
    ----------
    models : list of np.ndarray, each shape (N, 3)

    Returns
    -------
    np.ndarray, shape (M, M)
        Symmetric RMSD matrix.
    """
    m = len(models)
    matrix = np.zeros((m, m))

    for i in range(m):
        for j in range(i + 1, m):
            rmsd = kabsch_rmsd(models[i], models[j])
            matrix[i, j] = rmsd
            matrix[j, i] = rmsd

    return matrix
