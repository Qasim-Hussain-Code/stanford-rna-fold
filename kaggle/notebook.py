"""
Kaggle Submission Notebook — Stanford RNA 3D Folding Part 2

HelixForge: Hybrid TBM & Geometric Sampling for Multi-State RNA

This notebook implements a Template-Based Modeling (TBM) pipeline that:
  1. Builds a template index from PDB_RNA/ structures
  2. For each test target, searches for homologous templates
  3. Transfers coordinates via sequence alignment
  4. Fills gaps with geometric interpolation
  5. Refines with distance regularization
  6. Generates 5 diverse conformations for best-of-5 scoring

Competition: Stanford RNA 3D Folding Part 2
Approach: CPU-only, no deep learning, no internet required
"""

# ============================================================
# Cell 1: Install private package (offline)
# ============================================================
# !pip install /kaggle/input/my-private-rna-code/rna_fold-0.1.0-py3-none-any.whl --quiet

# ============================================================
# Cell 2: Imports and Configuration
# ============================================================
import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# If running from the repo directly (local testing), add src to path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
if os.path.exists(src_path):
    sys.path.insert(0, src_path)

from rna_fold.parser import parse_cif, build_template_index
from rna_fold.pipeline import run_pipeline, predict_structure
from rna_fold.transfer import generate_de_novo

# ============================================================
# Cell 3: Configuration
# ============================================================
# Paths — adjust for Kaggle vs local
KAGGLE_DATA_DIR = "/kaggle/input/stanford-rna-3d-folding-2"
LOCAL_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

if os.path.exists(KAGGLE_DATA_DIR):
    DATA_DIR = KAGGLE_DATA_DIR
    print("Running on Kaggle")
elif os.path.exists(LOCAL_DATA_DIR):
    DATA_DIR = LOCAL_DATA_DIR
    print("Running locally with test data")
else:
    DATA_DIR = "."
    print("WARNING: No data directory found. Using current directory.")

PDB_RNA_DIR = os.path.join(DATA_DIR, "PDB_RNA")
TEST_CSV = os.path.join(DATA_DIR, "test_sequences.csv")
INDEX_CACHE = "/tmp/template_index.pkl"
OUTPUT_FILE = "submission.csv"


def main():
    """Main entry point for the submission pipeline."""
    print("=" * 60)
    print("HelixForge: RNA 3D Structure Prediction Pipeline")
    print("=" * 60)
    total_start = time.time()

    # ========================================================
    # Step 1: Build or load template index
    # ========================================================
    print("\n[Step 1] Building template index from PDB_RNA/...")
    t0 = time.time()

    if os.path.exists(INDEX_CACHE):
        print("  Loading cached index...")
        with open(INDEX_CACHE, "rb") as f:
            template_index = pickle.load(f)
    elif os.path.exists(PDB_RNA_DIR):
        template_index = build_template_index(PDB_RNA_DIR)
        # Cache for reuse
        try:
            with open(INDEX_CACHE, "wb") as f:
                pickle.dump(template_index, f)
            print(f"  Cached index to {INDEX_CACHE}")
        except Exception:
            pass
    else:
        print("  WARNING: PDB_RNA/ not found. Will use de novo prediction only.")
        template_index = []

    print(f"  Template index: {len(template_index)} chains "
          f"({time.time() - t0:.1f}s)")

    # ========================================================
    # Step 2: Load test sequences
    # ========================================================
    print(f"\n[Step 2] Loading test sequences from {TEST_CSV}...")

    if not os.path.exists(TEST_CSV):
        print(f"  ERROR: {TEST_CSV} not found!")
        # Create a dummy submission so we don't crash
        pd.DataFrame(columns=[
            "ID", "resname", "resid",
            "x_1", "y_1", "z_1", "x_2", "y_2", "z_2",
            "x_3", "y_3", "z_3", "x_4", "y_4", "z_4",
            "x_5", "y_5", "z_5"
        ]).to_csv(OUTPUT_FILE, index=False)
        return

    test_df = pd.read_csv(TEST_CSV)
    print(f"  Loaded {len(test_df)} target(s)")
    print(f"  Columns: {list(test_df.columns)}")

    # Parse test data into dict: target_id -> sequence
    # Handle different possible column names
    if "target_id" in test_df.columns:
        id_col = "target_id"
    elif "ID" in test_df.columns:
        id_col = "ID"
    else:
        id_col = test_df.columns[0]

    if "sequence" in test_df.columns:
        seq_col = "sequence"
    elif "Sequence" in test_df.columns:
        seq_col = "Sequence"
    else:
        seq_col = test_df.columns[1]

    test_sequences = {}
    for _, row in test_df.iterrows():
        target_id = str(row[id_col])
        sequence = str(row[seq_col]).upper().replace("T", "U")
        test_sequences[target_id] = sequence

    print(f"  Parsed {len(test_sequences)} unique targets")

    # Show length distribution
    lengths = [len(s) for s in test_sequences.values()]
    print(f"  Sequence lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}")

    # ========================================================
    # Step 3: Run the pipeline
    # ========================================================
    print(f"\n[Step 3] Running TBM pipeline...")
    run_pipeline(
        test_sequences=test_sequences,
        template_index=template_index,
        output_path=OUTPUT_FILE,
        verbose=True,
    )

    # ========================================================
    # Step 4: Validate submission
    # ========================================================
    print(f"\n[Step 4] Validating submission...")
    sub_df = pd.read_csv(OUTPUT_FILE)
    print(f"  Rows: {len(sub_df)}")
    print(f"  Columns: {list(sub_df.columns)}")
    print(f"  Expected columns: 18, Got: {len(sub_df.columns)}")

    expected_rows = sum(len(s) for s in test_sequences.values())
    print(f"  Expected rows: {expected_rows}, Got: {len(sub_df)}")

    # Sanity check coordinates
    for model_idx in range(1, 6):
        x_col = f"x_{model_idx}"
        if x_col in sub_df.columns:
            x_vals = sub_df[x_col]
            print(f"  Model {model_idx}: x range=[{x_vals.min():.1f}, "
                  f"{x_vals.max():.1f}]")

    total_time = time.time() - total_start
    print(f"\nTotal pipeline time: {total_time:.1f}s "
          f"({total_time / 60:.1f} min)")
    print("DONE.")


if __name__ == "__main__":
    main()
