"""
pipeline.py — End-to-end TBM pipeline orchestrator.

Orchestrates the full prediction workflow:
  sequence → template search → alignment → coordinate transfer →
  gap filling → refinement → 5-state ensemble → output
"""

import numpy as np
import sys
import time
from typing import Dict, List, Optional, Tuple

from .parser import parse_cif, build_template_index
from .search import search_templates, get_template_coords
from .align import sequence_identity
from .transfer import transfer_coordinates, generate_de_novo
from .refine import refine_coordinates
from .ensemble import generate_ensemble


def predict_structure(
    sequence: str,
    template_index: List[dict],
    n_models: int = 5,
    min_identity: float = 0.20,
    min_coverage: float = 0.40,
    verbose: bool = False,
) -> List[np.ndarray]:
    """
    Predict 3D coordinates for an RNA sequence.

    Parameters
    ----------
    sequence : str
        RNA sequence (ACGU).
    template_index : list of dict
        Pre-built template index from build_template_index().
    n_models : int
        Number of ensemble models (5 for competition).
    min_identity : float
        Minimum sequence identity for template acceptance.
    min_coverage : float
        Minimum alignment coverage for template acceptance.
    verbose : bool
        Print progress info.

    Returns
    -------
    list of np.ndarray
        Five 3D coordinate arrays, each shape (N, 3).
    """
    n = len(sequence)
    if n == 0:
        return [np.empty((0, 3)) for _ in range(n_models)]

    if verbose:
        print(f"  Predicting structure for sequence of length {n}...", flush=True)

    # Step 1: Search for templates
    t0 = time.time()
    hits = search_templates(
        query_sequence=sequence,
        template_index=template_index,
        top_n=3,
        kmer_prefilter_n=min(100, len(template_index)),
        min_identity=min_identity,
        min_coverage=min_coverage,
    )
    t_search = time.time() - t0

    if verbose:
        if hits:
            best = hits[0]
            print(f"    Template found: identity={best['identity']:.2f}, "
                  f"coverage={best['coverage']:.2f}, "
                  f"search_time={t_search:.1f}s", flush=True)
        else:
            print(f"    No template found. Using de novo. "
                  f"search_time={t_search:.1f}s", flush=True)

    # Step 2: Generate base coordinates
    if hits:
        best_hit = hits[0]
        template_coords, template_resids = get_template_coords(best_hit)
        mapping = best_hit["mapping"]

        if len(template_coords) > 0:
            base_coords = transfer_coordinates(sequence, template_coords, mapping)
        else:
            base_coords = generate_de_novo(sequence, geometry="helix")
    else:
        base_coords = generate_de_novo(sequence, geometry="helix")

    # Step 3: Refine
    base_coords = refine_coordinates(base_coords)

    # Step 4: Generate ensemble
    ensemble = generate_ensemble(base_coords, n_models=n_models)

    # Light refinement on each ensemble member (except base)
    for i in range(1, len(ensemble)):
        ensemble[i] = refine_coordinates(
            ensemble[i],
            smooth_window=3,
            regularize_steps=15,
            smooth_weight=0.2,
        )

    return ensemble


def run_pipeline(
    test_sequences: Dict[str, Tuple[str, str]],
    template_index: List[dict],
    output_path: str = "submission.csv",
    verbose: bool = True,
) -> None:
    """
    Run the full prediction pipeline and write submission.csv.

    Parameters
    ----------
    test_sequences : dict
        Mapping from target_id to (sequence, temporal_cutoff or other info).
        Or simply target_id -> (sequence,).
    template_index : list of dict
        Pre-built template index.
    output_path : str
        Where to write submission CSV.
    verbose : bool
        Print progress.
    """
    if verbose:
        print(f"Running pipeline on {len(test_sequences)} targets...", flush=True)
        print(f"Template library: {len(template_index)} chains", flush=True)

    total = len(test_sequences)

    with open(output_path, "w") as f:
        f.write("ID,resname,resid,x_1,y_1,z_1,x_2,y_2,z_2,"
                "x_3,y_3,z_3,x_4,y_4,z_4,x_5,y_5,z_5\n")

        for idx, (target_id, seq_info) in enumerate(test_sequences.items()):
            if isinstance(seq_info, tuple):
                sequence = seq_info[0]
            else:
                sequence = seq_info

            if verbose:
                print(f"\n[{idx + 1}/{total}] Target: {target_id} "
                      f"(len={len(sequence)})", flush=True)

            t0 = time.time()
            try:
                models = predict_structure(
                    sequence=sequence,
                    template_index=template_index,
                    verbose=verbose,
                )
            except Exception as e:
                print(f"  ERROR: {e}. Using de novo fallback.", flush=True)
                de_novo = generate_de_novo(sequence, geometry="helix")
                models = [de_novo.copy() for _ in range(5)]

            elapsed = time.time() - t0
            if verbose:
                print(f"  Completed in {elapsed:.1f}s", flush=True)

            # Write to submission CSV
            _write_target(f, target_id, sequence, models)

    if verbose:
        print(f"\nSubmission written to {output_path}", flush=True)


def _write_target(f, target_id: str, sequence: str,
                  models: List[np.ndarray]) -> None:
    """Write rows for one target to the submission CSV."""
    n = len(sequence)
    for res_idx in range(n):
        row_id = f"{target_id}_{res_idx + 1}"
        resname = sequence[res_idx]
        resid = res_idx + 1

        coords_parts = []
        for model_idx in range(5):
            if model_idx < len(models) and res_idx < len(models[model_idx]):
                x, y, z = models[model_idx][res_idx]
                # Clip coordinates to allowed range
                x = np.clip(x, -999.999, 9999.999)
                y = np.clip(y, -999.999, 9999.999)
                z = np.clip(z, -999.999, 9999.999)
                coords_parts.append(f"{x:.3f},{y:.3f},{z:.3f}")
            else:
                coords_parts.append("0.000,0.000,0.000")

        f.write(f"{row_id},{resname},{resid},{','.join(coords_parts)}\n")
