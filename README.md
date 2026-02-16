# stanford-rna-fold

A computational pipeline for predicting RNA tertiary structures and multi-state conformational ensembles, developed for the [Stanford RNA 3D Folding Part 2](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2) challenge. The approach integrates Template-Based Modeling (TBM) with physics-inspired geometric sampling and secondary structure-guided coordinate reconstruction to predict C1' atom coordinates from RNA sequences.

## Background

Predicting the three-dimensional structure of RNA from its nucleotide sequence remains one of the central unsolved problems in structural biology. Unlike proteins, for which deep learning methods such as AlphaFold have achieved near-experimental accuracy, RNA structure prediction continues to pose significant challenges due to the diversity of RNA folding motifs, the scarcity of experimentally determined structures, and the inherent conformational flexibility of RNA molecules.

Template-Based Modeling has demonstrated strong performance in this domain, as evidenced by its effectiveness in the first iteration of the Stanford RNA 3D Folding competition, where a hybrid TBM approach secured 1st place. The underlying principle is straightforward: RNA tends to conserve its three-dimensional architecture more robustly than its primary sequence across evolutionary timescales, making homology-driven coordinate transfer a reliable prediction strategy when suitable templates exist.

## Methodology

This pipeline implements a six-stage TBM workflow, operating entirely on CPU with no dependency on pre-trained deep learning models.

### Stage 1 -- Template Search

Given a query RNA sequence, the pipeline identifies candidate structural templates from the competition-provided PDB_RNA library using a two-phase strategy. An initial k-mer-based filter rapidly scores all templates by shared subsequence content, reducing the search space. The top candidates then undergo full Needleman-Wunsch global sequence alignment with affine gap penalties optimized for RNA (match: +2, mismatch: -1, gap open: -5, gap extend: -1). Templates are ranked by the product of sequence identity and alignment coverage, with configurable thresholds for acceptance.

### Stage 2 -- Multi-Template Consensus

Rather than relying on a single best-scoring template, the pipeline blends coordinates from the top three templates using identity-weighted averaging. For each query position covered by multiple template alignments, the transferred coordinate is computed as the weighted mean across all contributing templates, where weights are the product of sequence identity and alignment coverage. This consensus approach, adapted from established practice in the CASP protein structure prediction community, substantially reduces noise attributable to any individual template.

### Stage 3 -- Gap Filling with Secondary Structure Guidance

Query residues that lack a corresponding template position are reconstructed using geometric interpolation, enhanced by RNA secondary structure prediction. The Nussinov dynamic programming algorithm is applied to the query sequence to identify canonical (AU, GC) and wobble (GU) base pairs. For gap regions that overlap with predicted helical stems, the pipeline applies helical interpolation geometry rather than simple linear interpolation, producing more physically realistic backbone conformations. Internal gaps are further refined through iterative enforcement of the canonical C1'-C1' inter-residue distance of approximately 5.9 Angstroms. When no template is found, a constrained random walk generates a compact de novo backbone that respects bond angle constraints (approximately 120 degrees between consecutive C1' atoms), steric exclusion (minimum 3.5 Angstroms between non-bonded atoms), and base-pair proximity derived from the Nussinov prediction.

### Stage 4 -- Refinement

The predicted coordinates undergo three post-processing steps: (i) Gaussian smoothing to attenuate sharp kinks introduced by gap filling, (ii) spring-like distance regularization to enforce consistent inter-residue spacing across 30 iterative cycles with a decaying learning rate, and (iii) steric clash removal for non-consecutive residues that fall within 3.0 Angstroms of each other.

### Stage 5 -- Template-Diverse Ensemble Generation

The competition's evaluation metric (TM-score) selects the best of five submitted conformations per target. To exploit this, the pipeline generates five structurally diverse models using a hybrid strategy:

| Model | Strategy | Description |
|-------|----------|-------------|
| 1 | Multi-template consensus | Weighted blend of top-3 templates |
| 2 | Alternative template | Coordinates from the second-best template |
| 3 | Alternative template | Coordinates from the third-best template |
| 4 | Thermal perturbation | Gaussian noise, flexibility-weighted at chain termini |
| 5 | Hinge motion | Rigid rotation of the C-terminal domain |

When multiple high-quality template hits are available, models 2 and 3 represent genuinely different structural hypotheses rather than mere perturbations of a single prediction. This approach explores a broader region of conformational space than noise-only diversification.

### Stage 6 -- Multi-Chain Handling

Part 2 of the competition includes multi-chain RNA complexes. The pipeline detects and processes each chain independently through the full prediction workflow, then positions chains with reasonable inter-chain spacing (20 Angstroms) to form the assembled complex.

## Repository Structure

```
stanford-rna-fold/
    src/rna_fold/
        __init__.py          Package metadata
        parser.py            mmCIF parser; extracts C1' atom coordinates per chain
        align.py             Needleman-Wunsch global alignment with affine gap penalties
        search.py            Two-phase template search (k-mer filter + full alignment)
        transfer.py          Multi-template consensus and geometric gap interpolation
        nussinov.py          RNA secondary structure prediction (Nussinov algorithm)
        refine.py            Distance regularization, smoothing, and clash removal
        ensemble.py          Conformational ensemble generation
        pipeline.py          End-to-end pipeline orchestrator with multiprocessing
    kaggle/
        notebook.py          Kaggle submission notebook script
    tests/
        test_pipeline.py     Unit and integration tests (32 tests)
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

All 32 tests validate the individual modules (alignment, coordinate transfer, multi-template consensus, Nussinov secondary structure, gap filling, refinement, ensemble generation, CIF parsing) as well as end-to-end submission format correctness.

## Dependencies

- Python >= 3.8
- NumPy
- pandas

No external bioinformatics libraries (Biopython, MMseqs2, etc.) are required. The mmCIF parser, sequence alignment engine, and secondary structure predictor are implemented from scratch to ensure compatibility with the competition's offline execution environment.

## Limitations

This implementation is a pure TBM baseline. It performs well on targets for which a homologous structure exists in the provided PDB_RNA library, but its accuracy degrades substantially for novel folds lacking detectable sequence similarity to any known template. In such cases, the pipeline falls back to the constrained random walk de novo backbone, which produces compact and sterically valid structures but cannot capture specific tertiary interactions. The winning solution in the first iteration of this competition addressed this limitation by supplementing TBM with DRfold2, a deep learning-based structure prediction model, for low-homology targets. Incorporating a similar fallback mechanism would be a natural next step for improving the pipeline's coverage of the sequence-structure space.

## License

MIT. See [LICENSE](LICENSE) for details.
