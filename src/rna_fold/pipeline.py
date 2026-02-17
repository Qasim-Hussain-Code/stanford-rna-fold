"""
pipeline.py — End-to-end RNA structure prediction pipeline.

Orchestrates the full TBM workflow, optimized for the 8-hour Kaggle time limit.
Key speed constraints:
  - Template index: ~15,000 CIF files → must use fast sequence-only extraction
  - Search: k-mer pre-filter to 30 candidates → NW alignment on those only
  - SA refinement: limited steps, skipped for long sequences
  - Ensemble: lightweight perturbation, no SA in ensemble generation
  - Per-target time budget: skip slow steps if running behind
"""

import os
import sys
import time
import numpy as np
from typing import Dict, List, Optional, Tuple

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
# Core prediction function (per target) — SPEED-OPTIMIZED
# ============================================================

def predict_structure(
    sequence: str,
    template_index: List[dict],
    n_models: int = 5,
    min_identity: float = 0.15,
    min_coverage: float = 0.30,
    verbose: bool = False,
    time_budget: float = 300.0,  # seconds per target
) -> List[np.ndarray]:
    """
    Predict RNA 3D structure from sequence.

    Speed-optimized pipeline:
      1. Template search (k-mer filter → NW on top 30 candidates)
      2. Multi-template consensus OR single template OR de novo
      3. Light refinement
      4. Brief SA refinement (skipped if time is short or sequence is long)
      5. Fast ensemble generation (no SA in ensemble, maximin on fewer candidates)
    """
    t_start = time.time()
    n = len(sequence)

    def _time_left():
        return time_budget - (time.time() - t_start)

    # Step 1: Search for templates (fast: k-mer filter to 30, NW on 30)
    hits = []
    if template_index:
        hits = search_templates(
            query_sequence=sequence,
            template_index=template_index,
            top_n=5,  # Only keep top 5 (was 10)
            min_identity=min_identity,
            min_coverage=min_coverage,
            max_kmer_candidates=30,  # Was 200 — this is the big speedup
        )

    if verbose:
        if hits:
            print(f"    Found {len(hits)} hit(s), "
                  f"best={hits[0]['identity']:.2f}/{hits[0]['coverage']:.2f}",
                  end="", flush=True)
        else:
            print(f"    No hits — de novo", end="", flush=True)

    # Step 2: Generate base coordinates
    has_template = len(hits) > 0
    if len(hits) >= 3:
        base_coords = _multi_template_predict(sequence, hits[:3], verbose)
    elif len(hits) >= 1:
        base_coords = _single_template_predict(sequence, hits[0], verbose)
    else:
        base_coords = generate_de_novo(sequence, geometry="compact")

    # Step 3: Classical refinement (always fast)
    base_coords = refine_coordinates(base_coords)

    # Step 4: SA refinement — only if we have time and sequence is short
    if _time_left() > 60 and n <= 300:
        try:
            from .energy import refine_with_energy
            sa_steps = 200 if has_template else 500
            base_coords = refine_with_energy(
                base_coords, sequence=sequence,
                n_steps=sa_steps, seed=42,
            )
        except Exception:
            pass

    # Step 5: Fast ensemble generation (NO SA inside ensemble)
    if len(hits) >= 2 and _time_left() > 30:
        # Use 1 template alternative + perturbations
        ensemble = _fast_template_ensemble(
            sequence, base_coords, hits, n_models
        )
    else:
        # Pure perturbation ensemble (fastest)
        ensemble = generate_ensemble(
            base_coords, n_models=n_models,
            sequence=sequence,
            use_sa=False,  # Critical: no SA in ensemble
        )

    # Clip coordinates
    for i in range(len(ensemble)):
        ensemble[i] = np.clip(ensemble[i], -999.999, 9999.999)

    return ensemble


# ============================================================
# Template coordinate extraction
# ============================================================

def _multi_template_predict(sequence: str, hits: List[dict],
                            verbose: bool = False) -> np.ndarray:
    """Multi-template consensus (Kabsch-aligned when possible)."""
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
        # Kabsch-align templates to the same reference frame
        try:
            _kabsch_align_templates(template_results, template_coords_list)
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
    """Structurally superpose templates 2..N onto template 1."""
    from .kabsch import kabsch_align

    if len(template_results) < 2:
        return

    ref_coords = template_coords_list[0][0]

    for i in range(1, len(template_results)):
        target_coords = template_coords_list[i][0]

        ref_query_set = set(q for q, _ in template_results[0]["mapping"])
        target_query_set = set(q for q, _ in template_results[i]["mapping"])
        common_query = sorted(ref_query_set & target_query_set)

        if len(common_query) < 4:
            continue

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

        # Compute rotation via SVD
        centroid_target = np.mean(target_arr, axis=0)
        centroid_ref = np.mean(ref_arr, axis=0)
        H = (target_arr - centroid_target).T @ (ref_arr - centroid_ref)
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        sign_m = np.eye(3)
        sign_m[2, 2] = np.sign(d)
        R = Vt.T @ sign_m @ U.T

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


def _fast_template_ensemble(
    sequence: str,
    base_coords: np.ndarray,
    hits: List[dict],
    n_models: int,
) -> List[np.ndarray]:
    """
    Fast ensemble: base + 1 template alternative + 3 perturbations.
    Uses maximin selection from a smaller candidate pool.
    """
    candidates = [base_coords.copy()]

    # One alternative template
    if len(hits) >= 2:
        try:
            alt = _single_template_predict(sequence, hits[1])
            alt = refine_coordinates(alt)
            candidates.append(alt)
        except Exception:
            pass

    # Fill with perturbations (fast, no SA)
    perturbations = generate_ensemble(
        base_coords,
        n_models=n_models + 2,
        sequence=sequence,
        use_sa=False,
    )
    candidates.extend(perturbations)

    # Maximin selection
    from .ensemble import maximin_select
    return maximin_select(candidates, n_models)


# ============================================================
# Multi-chain handling
# ============================================================

def predict_multichain(
    chains: Dict[str, str],
    template_index: List[dict],
    n_models: int = 5,
    verbose: bool = False,
) -> Dict[str, List[np.ndarray]]:
    """Predict structures for multi-chain RNA complexes."""
    chain_results = {}
    offset = np.zeros(3)

    for chain_id, sequence in chains.items():
        if verbose:
            print(f"  Chain {chain_id}: {len(sequence)} nt")

        models = predict_structure(
            sequence, template_index, n_models, verbose=verbose,
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
# Main pipeline — SEQUENTIAL ONLY (multiprocessing removed)
# ============================================================

def run_pipeline(
    test_sequences: Dict[str, str],
    template_index: List[dict],
    output_path: str = "submission.csv",
    verbose: bool = True,
    n_workers: int = 0,
) -> None:
    """
    Run the complete prediction pipeline.

    Runs sequentially to avoid multiprocessing overhead and ensure
    predictable memory usage within Kaggle constraints.
    """
    total_targets = len(test_sequences)
    header = ("ID,resname,resid,"
              "x_1,y_1,z_1,x_2,y_2,z_2,x_3,y_3,z_3,"
              "x_4,y_4,z_4,x_5,y_5,z_5\n")

    if verbose:
        total_residues = sum(len(s) for s in test_sequences.values())
        print(f"  Predicting {total_targets} targets "
              f"({total_residues} total residues)")

    # Compute per-target time budget
    # Reserve 30 minutes for index building overhead, use rest for predictions
    TOTAL_BUDGET = 5.5 * 3600  # 5.5 hours (out of 8, leaving margin)
    per_target = TOTAL_BUDGET / max(total_targets, 1)
    per_target = min(per_target, 600)  # Cap at 10 minutes per target

    if verbose:
        print(f"  Time budget: {per_target:.0f}s per target")

    with open(output_path, "w") as f:
        f.write(header)

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
                    time_budget=per_target,
                )
            except Exception as e:
                if verbose:
                    print(f" — ERROR: {e}, using de novo")
                base = generate_de_novo(sequence, geometry="compact")
                models = generate_ensemble(base, n_models=5, use_sa=False)

            _write_target(f, target_id, sequence, models)

            if verbose:
                elapsed = time.time() - t0
                print(f" — {elapsed:.1f}s")

    if verbose:
        print(f"  Submission written to {output_path}")
