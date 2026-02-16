"""
parser.py — mmCIF file parser for extracting RNA C1' atom coordinates.

Parses .cif files from the competition PDB_RNA/ folder without external
dependencies (no Biopython required for offline Kaggle submission).
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Mapping from three-letter codes to single-letter RNA bases
_RESNAME_MAP = {
    "A": "A", "C": "C", "G": "G", "U": "U",
    "ADE": "A", "CYT": "C", "GUA": "G", "URA": "U",
    "DA": "A", "DC": "C", "DG": "G", "DT": "U",  # DNA → treat as RNA
}


def parse_cif(filepath: str) -> Dict[str, Tuple[str, np.ndarray, List[int]]]:
    """
    Parse an mmCIF file and extract C1' atom coordinates per chain.

    Parameters
    ----------
    filepath : str
        Path to the .cif file.

    Returns
    -------
    dict
        Mapping from chain_id to (sequence, coords, residue_ids).
        coords : np.ndarray of shape (N, 3) — C1' xyz in Angstroms.
        residue_ids : list of int — one-based residue indices.
    """
    filepath = str(filepath)
    chains: Dict[str, Dict[int, dict]] = {}

    in_atom_site = False
    column_names = []
    col_idx = {}

    with open(filepath, "r") as f:
        for line in f:
            line = line.rstrip()

            # Detect start of _atom_site loop
            if line.startswith("_atom_site."):
                if not in_atom_site:
                    in_atom_site = True
                    column_names = []
                    col_idx = {}
                col_name = line.strip().split(".")[1].split()[0]
                column_names.append(col_name)
                col_idx[col_name] = len(column_names) - 1
                continue

            # If we were in atom_site and hit a non-data line, we're done
            if in_atom_site and (line.startswith("#") or line.startswith("_")
                                 or line.startswith("loop_")):
                in_atom_site = False
                continue

            if not in_atom_site:
                continue

            # Skip empty lines
            if not line.strip():
                continue

            # Parse atom record
            tokens = _tokenize_cif_line(line)
            if len(tokens) < len(column_names):
                continue

            # We only want ATOM or HETATM records
            group = _get_col(tokens, col_idx, "group_PDB", "ATOM")
            if group not in ("ATOM", "HETATM"):
                continue

            # We only want C1' atoms
            atom_name = _get_col(tokens, col_idx, "label_atom_id", "")
            if atom_name not in ("C1'", "C1*"):
                continue

            # Get chain, residue info
            chain_id = _get_col(tokens, col_idx, "label_asym_id",
                                _get_col(tokens, col_idx, "auth_asym_id", "A"))
            resname = _get_col(tokens, col_idx, "label_comp_id",
                               _get_col(tokens, col_idx, "auth_comp_id", ""))
            resid_str = _get_col(tokens, col_idx, "label_seq_id",
                                 _get_col(tokens, col_idx, "auth_seq_id", "0"))

            # Skip non-numeric residue IDs (e.g. ".")
            try:
                resid = int(resid_str)
            except ValueError:
                continue

            # Get coordinates
            try:
                x = float(_get_col(tokens, col_idx, "Cartn_x", "0"))
                y = float(_get_col(tokens, col_idx, "Cartn_y", "0"))
                z = float(_get_col(tokens, col_idx, "Cartn_z", "0"))
            except ValueError:
                continue

            # Only keep the first alt conf
            alt_id = _get_col(tokens, col_idx, "label_alt_id", ".")
            if alt_id not in (".", "A", "?", ""):
                continue

            # Convert residue name
            base = _RESNAME_MAP.get(resname.upper(), None)
            if base is None:
                continue  # Skip non-RNA residues

            if chain_id not in chains:
                chains[chain_id] = {}
            if resid not in chains[chain_id]:
                chains[chain_id][resid] = {"base": base, "x": x, "y": y, "z": z}

    # Convert to output format
    result = {}
    for chain_id, residues in chains.items():
        if not residues:
            continue
        sorted_ids = sorted(residues.keys())
        seq = "".join(residues[rid]["base"] for rid in sorted_ids)
        coords = np.array(
            [[residues[rid]["x"], residues[rid]["y"], residues[rid]["z"]]
             for rid in sorted_ids],
            dtype=np.float64,
        )
        result[chain_id] = (seq, coords, sorted_ids)

    return result


def parse_cif_longest_chain(filepath: str) -> Tuple[str, np.ndarray, List[int]]:
    """Parse a CIF and return only the longest RNA chain."""
    chains = parse_cif(filepath)
    if not chains:
        return "", np.empty((0, 3)), []
    best_chain = max(chains.values(), key=lambda x: len(x[0]))
    return best_chain


def build_template_index(pdb_dir: str) -> List[dict]:
    """
    Scan all CIF files in a directory and build a sequence index.

    Parameters
    ----------
    pdb_dir : str
        Path to the PDB_RNA/ directory.

    Returns
    -------
    list of dict
        Each entry: {"file": path, "chain_id": str, "sequence": str, "length": int}
    """
    index = []
    pdb_path = Path(pdb_dir)
    for cif_file in sorted(pdb_path.glob("*.cif")):
        try:
            chains = parse_cif(str(cif_file))
            for chain_id, (seq, coords, resids) in chains.items():
                if len(seq) >= 10:  # Skip very short fragments
                    index.append({
                        "file": str(cif_file),
                        "chain_id": chain_id,
                        "sequence": seq,
                        "length": len(seq),
                    })
        except Exception:
            continue  # Skip malformed files
    return index


def _tokenize_cif_line(line: str) -> List[str]:
    """Tokenize a CIF data line, respecting quoted strings."""
    tokens = []
    i = 0
    n = len(line)
    while i < n:
        if line[i] in (' ', '\t'):
            i += 1
            continue
        if line[i] in ("'", '"'):
            quote_char = line[i]
            i += 1
            start = i
            while i < n and line[i] != quote_char:
                i += 1
            tokens.append(line[start:i])
            i += 1  # skip closing quote
        else:
            start = i
            while i < n and line[i] not in (' ', '\t'):
                i += 1
            tokens.append(line[start:i])
    return tokens


def _get_col(tokens: List[str], col_idx: Dict[str, int],
             col_name: str, default: str) -> str:
    """Safely get a column value from tokenized CIF line."""
    idx = col_idx.get(col_name)
    if idx is not None and idx < len(tokens):
        val = tokens[idx]
        return val if val != "?" else default
    return default
