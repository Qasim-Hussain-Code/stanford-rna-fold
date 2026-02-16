"""
pipeline.py — End-to-end RNA structure prediction pipeline.

Orchestrates the full TBM workflow with improvements:
  - Multi-template consensus coordinate blending
  - Template-diverse ensemble (different templates as different models)
  - Multi-chain handling
  - Multiprocessing for throughput
"""

import os
import sys
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count

from .parser import parse_cif, build_template_index
from .search import search_templates, get_template_coords
from .transfer import (
    transfer_coordinates,
    multi_template_consensus,
    generate_de_novo,
    C1_C1_DISTANCE,
)
from .refine import refine_coordinates
from .ensemble import generate_ensemble


# ============================================================
# Core prediction function (per target)
# ============================================================

def predict_structure(
    sequence: str,
    template_index: List[dict],
    n_models: int = 5,
    min_identity: float = 0.15,
    min_coverage: float = 0.30,
    verbose: bool = False,
) -> List[np.ndarray]:
    """
    Predict RNA 3D structure from sequence.

    Improvements over baseline:
      1. Multi-template consensus: blends top-3 templates (weighted by identity)
      2. Template-diverse ensemble: uses different templates as separate models
      3. Nussinov-guided gap filling (via transfer.py)
      4. Compact random walk de novo fallback (via transfer.py)

    Parameters
    ----------
    sequence : str
        RNA sequence (ACGU).
    template_index : list of dict
        Template index from build_template_index().
    n_models : int
        Number of models to generate (default 5).
    min_identity : float
        Minimum sequence identity threshold.
    min_coverage : float
        Minimum alignment coverage threshold.
    verbose : bool
        Print progress information.

    Returns
    -------
    list of np.ndarray
        n_models coordinate arrays, each shape (N, 3).
    """
    n = len(sequence)

    # Step 1: Search for templates (get more than we need for consensus)
    hits = []
    if template_index:
        hits = search_templates(
            query_sequence=sequence,
            template_index=template_index,
            top_n=10,  # Get more for consensus and diversity
            min_identity=min_identity,
            min_coverage=min_coverage,
            max_kmer_candidates=150,
        )

    if verbose:
        if hits:
            print(f"    Found {len(hits)} template hit(s), "
                  f"best identity={hits[0]['identity']:.2f}, "
                  f"coverage={hits[0]['coverage']:.2f}")
        else:
            print(f"    No template hits — using de novo fallback")

    # Step 2: Generate base coordinates
    if len(hits) >= 3:
        # Multi-template consensus: blend top-3 templates
        base_coords = _multi_template_predict(sequence, hits[:3], verbose)
    elif len(hits) >= 1:
        # Single template
        base_coords = _single_template_predict(sequence, hits[0], verbose)
    else:
        # De novo (compact random walk with Nussinov guidance)
        base_coords = generate_de_novo(sequence, geometry="compact")

    # Step 3: Refine base coordinates
    base_coords = refine_coordinates(base_coords)

    # Step 4: Generate ensemble
    if len(hits) >= 2:
        # Template-diverse ensemble: use multiple template hypotheses
        ensemble = _template_diverse_ensemble(
            sequence, base_coords, hits, n_models, verbose
        )
    else:
        # Standard perturbation-based ensemble
        ensemble = generate_ensemble(base_coords, n_models=n_models)

    # Clip coordinates to valid range
    for i in range(len(ensemble)):
        ensemble[i] = np.clip(ensemble[i], -999.999, 9999.999)

    return ensemble


def _multi_template_predict(sequence: str, hits: List[dict],
                            verbose: bool = False) -> np.ndarray:
    """Use multi-template consensus for base coordinate prediction."""
    template_results = []
    template_coords_list = []

    for hit in hits:
        try:
            t_coords = get_template_coords(hit)
            if t_coords is not None and len(t_coords[0]) > 0:
                template_results.append(hit)
                template_coords_list.append(t_coords)
        except Exception:
            continue

    if len(template_results) >= 2:
        if verbose:
            print(f"    Multi-template consensus from {len(template_results)} templates")
        return multi_template_consensus(
            sequence, template_results, template_coords_list
        )
    elif len(template_results) == 1:
        return _single_template_predict(sequence, template_results[0], verbose)
    else:
        return generate_de_novo(sequence, geometry="compact")


def _single_template_predict(sequence: str, hit: dict,
                              verbose: bool = False) -> np.ndarray:
    """Single template coordinate transfer."""
    try:
        result = get_template_coords(hit)
        if result is None:
            return generate_de_novo(sequence, geometry="compact")

        t_coords, t_resids = result
        coords = transfer_coordinates(sequence, t_coords, hit["mapping"])
        return coords
    except Exception:
        return generate_de_novo(sequence, geometry="compact")


def _template_diverse_ensemble(
    sequence: str,
    base_coords: np.ndarray,
    hits: List[dict],
    n_models: int,
    verbose: bool = False,
) -> List[np.ndarray]:
    """
    Generate diverse ensemble using different templates as different models.

    Strategy:
      Model 1: Multi-template consensus (best prediction)
      Models 2-3: Individual top templates (structural diversity)
      Models 4-5: Perturbations of base (noise + hinge)
    """
    ensemble = [base_coords.copy()]  # Model 1: consensus

    # Models 2-3: Individual templates (if available)
    used_templates = 0
    for hit in hits[1:]:  # Skip first (already in consensus)
        if used_templates >= 2:
            break
        try:
            coords = _single_template_predict(sequence, hit)
            coords = refine_coordinates(coords)
            ensemble.append(coords)
            used_templates += 1
        except Exception:
            continue

    # Fill remaining slots with perturbation-based models
    remaining = n_models - len(ensemble)
    if remaining > 0:
        perturbation_ensemble = generate_ensemble(
            base_coords, n_models=remaining + 1
        )
        # Skip the first (base), take the perturbations
        for model in perturbation_ensemble[1:]:
            if len(ensemble) >= n_models:
                break
            ensemble.append(model)

    # If still not enough, add noise variants
    while len(ensemble) < n_models:
        noised = base_coords + np.random.normal(0, 1.5, base_coords.shape)
        ensemble.append(noised)

    return ensemble[:n_models]


# ============================================================
# Multi-chain handling
# ============================================================

def predict_multichain(
    chains: Dict[str, str],
    template_index: List[dict],
    n_models: int = 5,
    verbose: bool = False,
) -> Dict[str, List[np.ndarray]]:
    """
    Predict structures for multi-chain RNA complexes.

    Each chain is folded independently, then positioned with
    reasonable inter-chain spacing.

    Parameters
    ----------
    chains : dict of {chain_id: sequence}
    template_index : list of dict
    n_models : int
    verbose : bool

    Returns
    -------
    dict of {chain_id: list of model coords}
    """
    chain_results = {}
    offset = np.zeros(3)

    for chain_id, sequence in chains.items():
        if verbose:
            print(f"  Chain {chain_id}: {len(sequence)} nt")

        models = predict_structure(
            sequence, template_index, n_models,
            verbose=verbose,
        )

        # Apply offset to position chains next to each other
        for i in range(len(models)):
            models[i] = models[i] + offset

        chain_results[chain_id] = models

        # Compute offset for next chain: place it beyond this chain's extent
        all_coords = np.vstack(models)
        max_extent = np.max(all_coords, axis=0) - np.min(all_coords, axis=0)
        offset += np.array([max_extent[0] + 20.0, 0, 0])  # 20 Å gap

    return chain_results


# ============================================================
# Submission writing
# ============================================================

_RESNAME_MAP = {
    "A": "A", "C": "C", "G": "G", "U": "U",
    "T": "U", "N": "A",
}


def _write_target(f, target_id: str, sequence: str,
                   models: List[np.ndarray]) -> None:
    """Write one target's predictions to the submission CSV."""
    n = len(sequence)
    for i in range(n):
        resname = _RESNAME_MAP.get(sequence[i].upper(), "A")
        resid = i + 1
        parts = [f"{target_id}_{resid}", resname, str(resid)]

        for model_idx in range(5):
            if model_idx < len(models) and i < len(models[model_idx]):
                x, y, z = models[model_idx][i]
                parts.extend([
                    f"{np.clip(x, -999.999, 9999.999):.3f}",
                    f"{np.clip(y, -999.999, 9999.999):.3f}",
                    f"{np.clip(z, -999.999, 9999.999):.3f}",
                ])
            else:
                parts.extend(["0.000", "0.000", "0.000"])

        f.write(",".join(parts) + "\n")


# ============================================================
# Worker function for multiprocessing
# ============================================================

def _predict_worker(args):
    """Worker function for parallel prediction."""
    target_id, sequence, template_index, verbose = args
    try:
        models = predict_structure(
            sequence=sequence,
            template_index=template_index,
            n_models=5,
            verbose=verbose,
        )
        return (target_id, sequence, models)
    except Exception as e:
        # Fallback to de novo on any error
        n = len(sequence)
        base = generate_de_novo(sequence, geometry="compact")
        models = generate_ensemble(base, n_models=5)
        return (target_id, sequence, models)


# ============================================================
# Main pipeline
# ============================================================

def run_pipeline(
    test_sequences: Dict[str, str],
    template_index: List[dict],
    output_path: str = "submission.csv",
    verbose: bool = True,
    n_workers: int = 0,
) -> None:
    """
    Run the complete prediction pipeline for all test targets.

    Parameters
    ----------
    test_sequences : dict of {target_id: sequence}
    template_index : list of dict
    output_path : str
    verbose : bool
    n_workers : int
        Number of parallel workers. 0 = auto (use all CPUs).
    """
    total_targets = len(test_sequences)
    header = ("ID,resname,resid,"
              "x_1,y_1,z_1,x_2,y_2,z_2,x_3,y_3,z_3,"
              "x_4,y_4,z_4,x_5,y_5,z_5\n")

    if verbose:
        total_residues = sum(len(s) for s in test_sequences.values())
        print(f"  Predicting {total_targets} targets "
              f"({total_residues} total residues)")

    # Determine parallelism
    if n_workers == 0:
        n_workers = min(cpu_count(), total_targets, 8)

    use_parallel = n_workers > 1 and total_targets > 1

    with open(output_path, "w") as f:
        f.write(header)

        if use_parallel and total_targets >= 4:
            # Parallel prediction
            if verbose:
                print(f"  Using {n_workers} parallel workers")

            work_items = [
                (tid, seq, template_index, False)
                for tid, seq in test_sequences.items()
            ]

            try:
                with Pool(n_workers) as pool:
                    results = pool.map(_predict_worker, work_items)

                for idx, (target_id, sequence, models) in enumerate(results):
                    _write_target(f, target_id, sequence, models)
                    if verbose:
                        print(f"  [{idx+1}/{total_targets}] {target_id}: "
                              f"{len(sequence)} nt — done")
            except Exception as e:
                if verbose:
                    print(f"  Parallel execution failed ({e}), "
                          f"falling back to sequential")
                # Fallback to sequential
                _run_sequential(f, test_sequences, template_index,
                               verbose, total_targets)
        else:
            # Sequential prediction
            _run_sequential(f, test_sequences, template_index,
                           verbose, total_targets)

    if verbose:
        print(f"  Submission written to {output_path}")


def _run_sequential(f, test_sequences, template_index,
                    verbose, total_targets):
    """Sequential fallback for prediction."""
    for idx, (target_id, sequence) in enumerate(test_sequences.items()):
        t0 = time.time()

        if verbose:
            print(f"  [{idx+1}/{total_targets}] {target_id}: "
                  f"{len(sequence)} nt", end="", flush=True)

        try:
            models = predict_structure(
                sequence=sequence,
                template_index=template_index,
                n_models=5,
                verbose=verbose,
            )
        except Exception as e:
            if verbose:
                print(f" — ERROR: {e}, using de novo")
            base = generate_de_novo(sequence, geometry="compact")
            models = generate_ensemble(base, n_models=5)

        _write_target(f, target_id, sequence, models)

        if verbose:
            elapsed = time.time() - t0
            print(f" — {elapsed:.1f}s")
