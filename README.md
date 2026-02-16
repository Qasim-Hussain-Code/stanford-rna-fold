# stanford-rna-fold

A computational pipeline for predicting RNA tertiary structures and multi-state conformational ensembles, developed for the [Stanford RNA 3D Folding Part 2](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2) challenge. The approach integrates Template-Based Modeling (TBM) with physics-inspired geometric sampling to predict C1' atom coordinates from RNA sequences.

## Background

Predicting the three-dimensional structure of RNA from its nucleotide sequence remains one of the central unsolved problems in structural biology. Unlike proteins, for which deep learning methods such as AlphaFold have achieved near-experimental accuracy, RNA structure prediction continues to pose significant challenges due to the diversity of RNA folding motifs, the scarcity of experimentally determined structures, and the inherent conformational flexibility of RNA molecules.

Template-Based Modeling has demonstrated strong performance in this domain, as evidenced by its effectiveness in the first iteration of the Stanford RNA 3D Folding competition, where a hybrid TBM approach secured 1st place. The underlying principle is straightforward: RNA tends to conserve its three-dimensional architecture more robustly than its primary sequence across evolutionary timescales, making homology-driven coordinate transfer a reliable prediction strategy when suitable templates exist.

## Methodology

This pipeline implements a five-stage TBM workflow, operating entirely on CPU with no dependency on pre-trained deep learning models.

### Stage 1 -- Template Search

Given a query RNA sequence, the pipeline identifies candidate structural templates from the competition-provided PDB_RNA library using a two-phase strategy. An initial k-mer-based filter rapidly scores all templates by shared subsequence content, reducing the search space. The top candidates then undergo full Needleman-Wunsch global sequence alignment with affine gap penalties optimized for RNA (match: +2, mismatch: -1, gap open: -5, gap extend: -1). Templates are ranked by the product of sequence identity and alignment coverage, with configurable thresholds for acceptance.

### Stage 2 -- Coordinate Transfer

For matched alignment positions between the query and the best-scoring template, C1' atom coordinates are directly transferred. This step exploits the structural conservation principle inherent to homologous RNA molecules.

### Stage 3 -- Gap Filling

Query residues that lack a corresponding template position (insertions, deletions, or unaligned termini) are reconstructed using geometric interpolation. Internal gaps are filled by arc interpolation between flanking known coordinates, followed by iterative enforcement of the canonical C1'-C1' inter-residue distance of approximately 5.9 Angstroms. Leading and trailing gaps are extended along the local chain direction with slight stochastic perturbation to avoid degeneracy. When no template is found, a de novo A-form helical backbone is generated as a fallback.

### Stage 4 -- Refinement

The predicted coordinates undergo three post-processing steps: (i) Gaussian smoothing to attenuate sharp kinks introduced by gap filling, (ii) spring-like distance regularization to enforce consistent inter-residue spacing across 30 iterative cycles with a decaying learning rate, and (iii) steric clash removal for non-consecutive residues that fall within 3.0 Angstroms of each other.

### Stage 5 -- Conformational Ensemble Generation

The competition's evaluation metric (TM-score) selects the best of five submitted conformations per target. To exploit this, the pipeline generates five structurally diverse models from the base prediction:

| Model | Strategy | Description |
|-------|----------|-------------|
| 1 | Base | Unperturbed template-derived prediction |
| 2 | Thermal perturbation (low) | Gaussian noise, sigma = 0.5 Angstroms, weighted by local flexibility |
| 3 | Thermal perturbation (high) | Gaussian noise, sigma = 1.0 Angstroms |
| 4 | Hinge motion (open) | Rigid rotation of the C-terminal domain by +15 degrees |
| 5 | Hinge motion (closed) | Rigid rotation of the C-terminal domain by -15 degrees |

The noise perturbations are modulated by a flexibility profile that assigns higher amplitudes to terminal residues, reflecting the empirical observation that chain termini exhibit greater conformational variability in solution.

## Repository Structure

```
stanford-rna-fold/
    src/rna_fold/
        __init__.py          Package metadata
        parser.py            mmCIF parser; extracts C1' atom coordinates per chain
        align.py             Needleman-Wunsch global alignment with affine gap penalties
        search.py            Two-phase template search (k-mer filter + full alignment)
        transfer.py          Coordinate transfer and geometric gap interpolation
        refine.py            Distance regularization, smoothing, and clash removal
        ensemble.py          Five-state conformational ensemble generation
        pipeline.py          End-to-end pipeline orchestrator and submission writer
    kaggle/
        notebook.py          Kaggle submission notebook script
    tests/
        test_pipeline.py     Unit and integration tests (23 tests)
    setup.py
    LICENSE
```

## Installation

```bash
pip install -e .
```

To build a distributable wheel for offline deployment:

```bash
pip install build
python -m build
```

This produces `dist/rna_fold-0.1.0-py3-none-any.whl`, which can be installed on air-gapped systems (e.g., Kaggle with internet disabled) via `pip install rna_fold-0.1.0-py3-none-any.whl`.

## Testing

```bash
python -m pytest tests/ -v
```

All 23 tests validate the individual modules (alignment, coordinate transfer, gap filling, refinement, ensemble generation, CIF parsing) as well as end-to-end submission format correctness.

## Dependencies

- Python >= 3.8
- NumPy
- pandas

No external bioinformatics libraries (Biopython, MMseqs2, etc.) are required. The mmCIF parser and sequence alignment engine are implemented from scratch to ensure compatibility with the competition's offline execution environment.

## Limitations

This implementation is a pure TBM baseline. It performs well on targets for which a homologous structure exists in the provided PDB_RNA library, but its accuracy degrades substantially for novel folds lacking detectable sequence similarity to any known template. In such cases, the pipeline falls back to de novo A-form helix generation, which is unlikely to produce biologically meaningful predictions. The winning solution in the first iteration of this competition addressed this limitation by supplementing TBM with DRfold2, a deep learning-based structure prediction model, for low-homology targets. Incorporating a similar fallback mechanism would be a natural next step for improving the pipeline's coverage of the sequence-structure space.

## License

MIT. See [LICENSE](LICENSE) for details.
