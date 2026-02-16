"""
pipeline.py — End-to-end RNA structure prediction pipeline.

Orchestrates the full TBM workflow with maximum-impact improvements:
  - Multi-template consensus with Kabsch superposition
  - Template-diverse ensemble + maximin diversity selection
  - Simulated annealing energy-based refinement
  - Nussinov-guided gap filling
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

    Full pipeline:
      1. Template search (k-mer filtered + NW with RIBOSUM scoring)
      2. Multi-template consensus OR single template OR de novo
      3. Classical refinement (smoothing + distance regularization)
      4. Simulated annealing with coarse-grained energy function
      5. Template-diverse ensemble with maximin diversity selection
    """
    n = len(sequence)

    # Step 1: Search for templates
    hits = []
    if template_index:
        hits = search_templates(
            query_sequence=sequence,
            template_index=template_index,
            top_n=10,
            min_identity=min_identity,
            min_coverage=min_coverage,
            max_kmer_candidates=200,
        )

    if verbose:
        if hits:
            print(f"    Found {len(hits)} template hit(s), "
                  f"best identity={hits[0]['identity']:.2f}, "
                  f"coverage={hits[0]['coverage']:.2f}")
        else:
            print(f"    No template hits — using de novo fallback")

    # Step 2: Generate base coordinates
    has_template = len(hits) > 0
    if len(hits) >= 3:
        base_coords = _multi_template_predict(sequence, hits[:3], verbose)
    elif len(hits) >= 1:
        base_coords = _single_template_predict(sequence, hits[0], verbose)
    else:
        base_coords = generate_de_novo(sequence, geometry="compact")

    # Step 3: Classical refinement
    base_coords = refine_coordinates(base_coords)

    # Step 4: Simulated annealing refinement (physics-based)
    # More SA steps for de novo (more room to improve), fewer for template-based
    try:
        from .energy import refine_with_energy
        sa_steps = 500 if has_template else 2000
        if n > 500:
            sa_steps = min(sa_steps, 300)  # Time guard
        base_coords = refine_with_energy(
            base_coords, sequence=sequence,
            n_steps=sa_steps, seed=42,
        )
    except Exception:
        pass

    # Step 5: Generate diverse ensemble
    # Collect all template-derived alternatives for the ensemble
    template_alternatives = []
    if len(hits) >= 2:
        for hit in hits[1:4]:  # Up to 3 alternatives
            try:
                alt = _single_template_predict(sequence, hit)
                alt = refine_coordinates(alt)
                template_alternatives.append(alt)
            except Exception:
                continue

    # Build candidate pool: base + template alternatives + perturbations
    all_candidates = [base_coords.copy()] + template_alternatives

    # Generate perturbation-based candidates via the ensemble module
    # (which already generates ~20 candidates internally and selects via maximin)
    perturbation_models = generate_ensemble(
        base_coords,
        n_models=max(n_models, 5),
        sequence=sequence,
        use_sa=(n <= 200),  # SA variants only for small structures
    )

    # Merge all candidates
    for model in perturbation_models:
        all_candidates.append(model)

    # Final maximin diversity selection across ALL candidates
    from .ensemble import maximin_select
    ensemble = maximin_select(all_candidates, n_models)

    # Clip coordinates
    for i in range(len(ensemble)):
        ensemble[i] = np.clip(ensemble[i], -999.999, 9999.999)

    return ensemble


# ============================================================
# Template coordinate extraction
# ============================================================

def _multi_template_predict(sequence: str, hits: List[dict],
                            verbose: bool = False) -> np.ndarray:
    """
    Multi-template consensus with Kabsch superposition.

    Templates are first Kabsch-aligned to the best template before
    averaging, preventing coordinate smearing from misaligned frames.
    """
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

        # Kabsch-align all templates to the first (best) template
        try:
            from .kabsch import kabsch_align
            _kabsch_align_templates(
                template_results, template_coords_list
            )
        except Exception:
            pass

        return multi_template_consensus(
            sequence, template_results, template_coords_list
        )
    elif len(template_results) == 1:
        return _single_template_predict(sequence, template_results[0], verbose)
    else:
        return generate_de_novo(sequence, geometry="compact")


def _kabsch_align_templates(template_results, template_coords_list):
    """
    Structurally superpose templates 2..N onto template 1 using Kabsch.

    This ensures that when we average coordinates across templates,
    they are in the same reference frame. Without this, the average
    would smear the structure.
    """
    from .kabsch import kabsch_align

    if len(template_results) < 2:
        return

    ref_coords = template_coords_list[0][0]  # Best template coords
    ref_mapping = {t: q for q, t in template_results[0]["mapping"]}

    for i in range(1, len(template_results)):
        target_coords = template_coords_list[i][0]
        target_mapping = {t: q for q, t in template_results[i]["mapping"]}

        # Find common query positions between template 0 and template i
        ref_query_set = set(q for q, _ in template_results[0]["mapping"])
        target_query_set = set(q for q, _ in template_results[i]["mapping"])
        common_query = sorted(ref_query_set & target_query_set)

        if len(common_query) < 4:
            continue

        # Build corresponding coordinate arrays
        ref_map_inv = {q: t for q, t in template_results[0]["mapping"]}
        target_map_inv = {q: t for q, t in template_results[i]["mapping"]}

        ref_pts = []
        target_pts = []
        for q in common_query:
            rt = ref_map_inv[q]
            tt = target_map_inv[q]
            if rt < len(ref_coords) and tt < len(target_coords):
                ref_pts.append(ref_coords[rt])
                target_pts.append(target_coords[tt])

        if len(ref_pts) < 4:
            continue

        ref_arr = np.array(ref_pts)
        target_arr = np.array(target_pts)

        # Kabsch align
        aligned, _, rmsd = kabsch_align(target_arr, ref_arr)

        # Apply the same transformation to ALL target template coordinates
        centroid_target = np.mean(target_arr, axis=0)
        centroid_ref = np.mean(ref_arr, axis=0)
        H = (target_arr - centroid_target).T @ (ref_arr - centroid_ref)
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        sign_m = np.eye(3)
        sign_m[2, 2] = np.sign(d)
        R = Vt.T @ sign_m @ U.T

        # Transform all coordinates
        full_coords = template_coords_list[i][0]
        transformed = (full_coords - centroid_target) @ R.T + centroid_ref
        template_coords_list[i] = (transformed, template_coords_list[i][1])


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
    Each chain folded independently, positioned with inter-chain spacing.
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

        for i in range(len(models)):
            models[i] = models[i] + offset

        chain_results[chain_id] = models

        all_coords = np.vstack(models)
        max_extent = np.max(all_coords, axis=0) - np.min(all_coords, axis=0)
        offset += np.array([max_extent[0] + 20.0, 0, 0])

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
    """
    total_targets = len(test_sequences)
    header = ("ID,resname,resid,"
              "x_1,y_1,z_1,x_2,y_2,z_2,x_3,y_3,z_3,"
              "x_4,y_4,z_4,x_5,y_5,z_5\n")

    if verbose:
        total_residues = sum(len(s) for s in test_sequences.values())
        print(f"  Predicting {total_targets} targets "
              f"({total_residues} total residues)")

    if n_workers == 0:
        n_workers = min(cpu_count(), total_targets, 8)

    use_parallel = n_workers > 1 and total_targets > 1

    with open(output_path, "w") as f:
        f.write(header)

        if use_parallel and total_targets >= 4:
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
                    print(f"  Parallel failed ({e}), sequential fallback")
                _run_sequential(f, test_sequences, template_index,
                               verbose, total_targets)
        else:
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
