"""
pipeline.py — End-to-end RNA structure prediction pipeline.

Orchestrates the full TBM workflow, aggressively optimized for 8-hour limit.
Time budget breakdown:
  - Index building: ~30-45 min (fast sequence-only parser)
  - Predictions: ~3-4 hours for 28 targets
  - Total target: < 5 hours, leaving 3 hours of margin

Key optimizations over v1:
  - Banded NW alignment for long sequences (O(n*band) vs O(n*m))
  - Length-adaptive candidate count (fewer NW alignments for long seqs)
  - Targets sorted shortest-first (long sequences get remaining budget)
  - Hard per-target timeout with de novo fallback
  - No SA refinement, no SA in ensemble
  - Sequential execution only
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
# Core prediction function — AGGRESSIVELY SPEED-OPTIMIZED
# ============================================================

def predict_structure(
    sequence: str,
    template_index: List[dict],
    n_models: int = 5,
    min_identity: float = 0.15,
    min_coverage: float = 0.30,
    verbose: bool = False,
    time_budget: float = 300.0,
) -> List[np.ndarray]:
    """
    Predict RNA 3D structure from sequence.

    Length-adaptive strategy:
      - Short (≤200 nt): full search + refinement + ensemble
      - Medium (201-500 nt): reduced search + light ensemble
      - Long (501-1500 nt): minimal search + perturbation only
      - Very long (>1500 nt): k-mer only (no NW) + de novo perturbation
    """
    t_start = time.time()
    n = len(sequence)

    def _time_left():
        return time_budget - (time.time() - t_start)

    # Length-adaptive parameters
    if n <= 200:
        kmer_candidates = 30
        top_n = 5
    elif n <= 500:
        kmer_candidates = 15
        top_n = 3
    elif n <= 1500:
        kmer_candidates = 8
        top_n = 2
    else:
        kmer_candidates = 3  # Very few NW alignments for huge sequences
        top_n = 1

    # Step 1: Template search
    hits = []
    if template_index and _time_left() > 30:
        try:
            hits = search_templates(
                query_sequence=sequence,
                template_index=template_index,
                top_n=top_n,
                min_identity=min_identity,
                min_coverage=min_coverage,
                max_kmer_candidates=kmer_candidates,
            )
        except Exception:
            hits = []

    if verbose:
        if hits:
            print(f"    Found {len(hits)} hit(s), "
                  f"best={hits[0]['identity']:.2f}/{hits[0]['coverage']:.2f}",
                  end="", flush=True)
        else:
            print(f"    No hits — de novo", end="", flush=True)

    # Step 2: Generate base coordinates
    has_template = len(hits) > 0
    if _time_left() < 20:
        # Emergency: almost out of time, go straight to de novo
        base_coords = generate_de_novo(sequence, geometry="compact")
    elif len(hits) >= 2 and n <= 500:
        # Multi-template consensus only for short/medium sequences
        base_coords = _multi_template_predict(sequence, hits[:2], verbose)
    elif len(hits) >= 1:
        base_coords = _single_template_predict(sequence, hits[0], verbose)
    else:
        base_coords = generate_de_novo(sequence, geometry="compact")

    # Step 3: Classical refinement (fast, always do it)
    base_coords = refine_coordinates(base_coords)

    # Step 4: SA refinement — ONLY for short sequences with plenty of time
    if _time_left() > 60 and n <= 150:
        try:
            from .energy import refine_with_energy
            sa_steps = 150 if has_template else 300
            base_coords = refine_with_energy(
                base_coords, sequence=sequence,
                n_steps=sa_steps, seed=42,
            )
        except Exception:
            pass

    # Step 5: Ensemble generation (no SA, fast perturbation only)
    if _time_left() > 10:
        ensemble = generate_ensemble(
            base_coords, n_models=n_models,
            sequence=sequence,
            use_sa=False,
        )
    else:
        # Emergency: just duplicate with tiny noise
        ensemble = [base_coords + np.random.randn(*base_coords.shape) * 0.5
                    for _ in range(n_models)]

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
# Main pipeline — SORTED SHORTEST-FIRST
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

    Sorts targets shortest-first so long sequences can use
    any remaining time budget. Writes results incrementally.
    """
    total_targets = len(test_sequences)
    header = ("ID,resname,resid,"
              "x_1,y_1,z_1,x_2,y_2,z_2,x_3,y_3,z_3,"
              "x_4,y_4,z_4,x_5,y_5,z_5\n")

    if verbose:
        total_residues = sum(len(s) for s in test_sequences.values())
        print(f"  Predicting {total_targets} targets "
              f"({total_residues} total residues)")

    # Sort targets: shortest first, so fast ones complete quickly
    # and long sequences can use remaining budget
    sorted_targets = sorted(test_sequences.items(), key=lambda x: len(x[1]))

    # Global time tracking
    TOTAL_BUDGET = 4.5 * 3600  # 4.5 hours for predictions (conservative)
    pipeline_start = time.time()

    if verbose:
        print(f"  Total prediction budget: {TOTAL_BUDGET/3600:.1f}h")

    # Store results keyed by target_id (we write in original order at the end)
    results = {}

    for idx, (target_id, sequence) in enumerate(sorted_targets):
        t0 = time.time()

        # Dynamic per-target budget: remaining time / remaining targets
        elapsed_total = time.time() - pipeline_start
        remaining_budget = TOTAL_BUDGET - elapsed_total
        remaining_targets = total_targets - idx
        per_target = remaining_budget / max(remaining_targets, 1)
        per_target = max(per_target, 30)  # At least 30 seconds
        per_target = min(per_target, 1200)  # At most 20 minutes

        if verbose:
            print(f"  [{idx+1}/{total_targets}] {target_id}: "
                  f"{len(sequence)} nt (budget: {per_target:.0f}s)",
                  end="", flush=True)

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

        results[target_id] = (sequence, models)

        if verbose:
            elapsed = time.time() - t0
            total_elapsed = time.time() - pipeline_start
            print(f" — {elapsed:.1f}s (total: {total_elapsed/60:.0f}m)")

    # Write results in ORIGINAL order (important for submission format)
    with open(output_path, "w") as f:
        f.write(header)
        for target_id, sequence in test_sequences.items():
            if target_id in results:
                seq, models = results[target_id]
                _write_target(f, target_id, seq, models)

    if verbose:
        total_elapsed = time.time() - pipeline_start
        print(f"  Submission written to {output_path}")
        print(f"  Total prediction time: {total_elapsed/60:.0f}m")
