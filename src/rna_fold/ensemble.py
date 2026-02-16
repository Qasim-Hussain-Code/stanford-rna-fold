"""
ensemble.py — 5-state conformational ensemble generation.

Generates 5 structurally diverse models from a base prediction to
maximize the best-of-5 TM-score metric. Strategies:
  1. Base (anchor): the best prediction as-is
  2. Thermal noise (low): Gaussian perturbation σ=0.5 Å
  3. Thermal noise (high): Gaussian perturbation σ=1.0 Å
  4. Hinge open: rotate second domain by +15°
  5. Hinge close: rotate second domain by -15°
"""

import numpy as np
from typing import List, Optional


def generate_ensemble(
    base_coords: np.ndarray,
    n_models: int = 5,
    noise_scales: Optional[List[float]] = None,
    hinge_angles: Optional[List[float]] = None,
) -> List[np.ndarray]:
    """
    Generate a diverse ensemble of conformations from a base structure.

    Parameters
    ----------
    base_coords : np.ndarray, shape (N, 3)
        Base predicted coordinates.
    n_models : int
        Number of models to generate (must be 5 for this competition).
    noise_scales : list of float, optional
        Gaussian noise scales for thermal perturbation models.
    hinge_angles : list of float, optional
        Rotation angles (degrees) for hinge motion models.

    Returns
    -------
    list of np.ndarray
        Five coordinate arrays, each shape (N, 3).
    """
    if noise_scales is None:
        noise_scales = [0.5, 1.0]
    if hinge_angles is None:
        hinge_angles = [15.0, -15.0]

    ensemble = []

    # Model 1: Base structure (anchor — best prediction)
    ensemble.append(base_coords.copy())

    # Model 2-3: Thermal noise perturbations
    for scale in noise_scales:
        perturbed = apply_noise(base_coords, scale=scale)
        ensemble.append(perturbed)

    # Model 4-5: Hinge motions
    for angle in hinge_angles:
        hinged = apply_hinge(base_coords, angle=angle)
        ensemble.append(hinged)

    # Ensure we have exactly n_models
    while len(ensemble) < n_models:
        scale = 0.3 + 0.5 * np.random.random()
        ensemble.append(apply_noise(base_coords, scale=scale))

    return ensemble[:n_models]


def apply_noise(coords: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """
    Apply Gaussian thermal noise to coordinates.

    The noise is applied uniformly but the magnitude is modulated
    by the local flexibility — terminal regions get more noise.
    """
    n = len(coords)
    if n == 0:
        return coords.copy()

    result = coords.copy()

    # Flexibility profile: terminals are more flexible
    flex = np.ones(n)
    ramp_len = min(n // 4, 20)
    for i in range(ramp_len):
        factor = 1.0 + 0.5 * (1 - i / ramp_len)
        flex[i] *= factor
        flex[n - 1 - i] *= factor

    # Apply noise with flexibility modulation
    for i in range(n):
        noise = np.random.normal(0, scale * flex[i], 3)
        result[i] += noise

    return result


def apply_hinge(
    coords: np.ndarray,
    pivot_index: Optional[int] = None,
    angle: float = 15.0,
    axis: str = "auto",
) -> np.ndarray:
    """
    Apply hinge motion: rotate the second half of the structure
    around a pivot point, simulating domain opening/closing.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)
        Input coordinates.
    pivot_index : int, optional
        Index of the pivot residue. Default: midpoint.
    angle : float
        Rotation angle in degrees.
    axis : str
        Rotation axis: "auto" picks a perpendicular axis to the chain.
    """
    n = len(coords)
    if n < 4:
        return coords.copy()

    if pivot_index is None:
        pivot_index = n // 2

    pivot_index = max(2, min(n - 2, pivot_index))

    result = coords.copy()
    pivot_point = coords[pivot_index].copy()

    # Determine rotation axis
    if axis == "auto":
        # Use the perpendicular to the local chain direction
        local_dir = coords[min(pivot_index + 2, n - 1)] - coords[max(pivot_index - 2, 0)]
        local_dir_norm = np.linalg.norm(local_dir)
        if local_dir_norm < 1e-8:
            rot_axis = np.array([0.0, 0.0, 1.0])
        else:
            local_dir /= local_dir_norm
            # Find perpendicular axis
            if abs(local_dir[0]) < 0.9:
                perp = np.cross(local_dir, [1, 0, 0])
            else:
                perp = np.cross(local_dir, [0, 1, 0])
            rot_axis = perp / np.linalg.norm(perp)
    else:
        axes = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
        rot_axis = np.array(axes.get(axis, [0, 0, 1]), dtype=float)

    # Build rotation matrix (Rodrigues' formula)
    R = _rotation_matrix(rot_axis, np.radians(angle))

    # Rotate second half around pivot
    for i in range(pivot_index, n):
        relative = result[i] - pivot_point
        result[i] = pivot_point + R @ relative

    return result


def apply_jitter(coords: np.ndarray, max_shift: float = 2.0) -> np.ndarray:
    """Apply random rigid-body translation + rotation jitter."""
    result = coords.copy()
    center = np.mean(result, axis=0)

    # Small random rotation
    angle = np.random.uniform(-5, 5)
    axis = np.random.normal(0, 1, 3)
    axis /= np.linalg.norm(axis)
    R = _rotation_matrix(axis, np.radians(angle))

    # Small random translation
    shift = np.random.normal(0, max_shift / 3, 3)

    for i in range(len(result)):
        result[i] = center + R @ (result[i] - center) + shift

    return result


def _rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues' rotation formula: rotation matrix around axis by theta radians."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R
