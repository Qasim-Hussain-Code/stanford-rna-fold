# stanford-rna-fold

This repository contains my submission to the [Stanford RNA 3D Folding Part 2](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2) competition on Kaggle. The competition asks participants to predict the three-dimensional coordinates of RNA molecules from nucleotide sequence alone, with the objective of maximizing the TM-score (best of five submitted conformations per target) against experimentally determined reference structures.

## Motivation

Predicting the three-dimensional structure of RNA from its primary sequence is one of the central unsolved problems in structural biology. Unlike proteins, for which deep learning methods such as AlphaFold have achieved near-experimental accuracy, RNA structure prediction continues to pose significant challenges owing to the diversity of RNA folding motifs, the relative scarcity of experimentally determined structures, and the conformational flexibility inherent to RNA molecules. I pursued a Template-Based Modeling approach for this competition because TBM has historically demonstrated strong performance in nucleic acid structure prediction, including in the first iteration of this challenge, where a hybrid TBM solution placed first.

## Approach

I implemented a six-stage computational pipeline operating entirely on CPU, with no dependency on pre-trained deep learning models. All algorithms are implemented from scratch using only NumPy and pandas.

### Template Search

Given a query RNA sequence, the pipeline identifies candidate structural templates from the competition-provided PDB_RNA library using a two-phase search strategy. An initial k-mer-based filter rapidly scores all templates by shared subsequence content, reducing the search space to approximately 200 candidates. These candidates then undergo full Needleman-Wunsch global sequence alignment with affine gap penalties. I incorporated a RIBOSUM-inspired substitution matrix that differentiates transitions from transversions and scores wobble base pairs (G-U) more favorably than true mismatches, reflecting the known structural conservation of wobble pairs across RNA families (Klein and Eddy, 2003). Templates are ranked by the product of their sequence identity and alignment coverage.

### Multi-Template Consensus

Rather than relying on a single best-scoring template, I blend coordinates from the top three templates using identity-weighted averaging. Before averaging, I structurally superpose all templates into a common reference frame using the Kabsch SVD algorithm (Kabsch, 1976). Without this superposition step, coordinates from different templates would be in different spatial orientations, and their average would produce a smeared, unphysical structure. The Kabsch-aligned consensus approach is adapted from established practice in the CASP protein structure prediction community.

### Gap Filling with Secondary Structure Guidance

Query residues that lack a corresponding template position are reconstructed using geometric interpolation. I apply the Nussinov dynamic programming algorithm (Nussinov et al., 1978) to the query sequence to predict canonical (A-U, G-C) and wobble (G-U) base pairs. For gap regions that overlap with predicted helical stems, the pipeline applies helical interpolation geometry rather than simple linear interpolation, producing more physically realistic backbone conformations. For sequences exceeding 500 nucleotides, I partition the sequence into overlapping windows and merge the predicted base pairs, converting the cubic time complexity into a linear-in-sequence-length operation while preserving local pairing accuracy.

When no template is available, a constrained random walk generates a compact backbone that respects bond angle constraints (approximately 120 degrees between consecutive C1-prime atoms), enforces steric exclusion (minimum 3.5 Angstroms between non-bonded atoms), and biases base-paired residues toward appropriate spatial proximity as predicted by the Nussinov algorithm.

### Refinement

I apply two stages of refinement. The first is a classical post-processing step consisting of Gaussian smoothing to attenuate sharp kinks from gap filling, spring-like distance regularization to enforce consistent inter-residue spacing, and steric clash removal for non-consecutive residues.

The second stage is a simulated annealing optimization (Kirkpatrick et al., 1983) using a coarse-grained energy function that I designed specifically for C1-prime backbone coordinates. The energy function includes six terms: bond length potential, bond angle potential, higher-order distance constraints (for residue pairs separated by two and three positions along the chain, enforcing proper backbone curvature), steric exclusion, base-pair distance restraints from the Nussinov prediction, and a radius-of-gyration restraint that biases the structure toward compact globular geometries. Simulated annealing explores conformational space more effectively than gradient-based refinement by accepting uphill moves according to the Metropolis criterion, enabling escape from local energy minima.

### Ensemble Generation and Diversity Selection

The competition evaluates TM-score on the best of five submitted conformations per target, making ensemble diversity a critical factor. I generate approximately twenty candidate models per target using six distinct perturbation strategies: Gaussian noise at multiple amplitude scales with flexibility weighting at chain termini, single-hinge and multi-hinge domain rotations using the Rodrigues rotation formula, collective breathing motions that uniformly expand or contract the structure relative to its center of mass, local segment perturbations, and simulated annealing variants initialized with different random seeds. When multiple high-quality template hits are available, I also include coordinates transferred independently from alternative templates, providing genuinely different structural hypotheses rather than mere perturbations of a single prediction.

From this pool of candidates, I select the five most diverse models using a greedy maximin algorithm: starting with the consensus prediction, I iteratively add the candidate whose minimum RMSD to all previously selected models is largest. Pairwise RMSD values are computed after optimal Kabsch superposition. This algorithm is a provable 2-approximation to the NP-hard maximum diversity selection problem, and it ensures that the five submitted models span as broad a region of conformational space as possible.

### Multi-Chain Handling

The competition includes multi-chain RNA complexes. I process each chain independently through the full prediction workflow and position the assembled complex with reasonable inter-chain spacing.

## Compliance and Ethics

I have taken care to ensure that this submission adheres strictly to the competition rules and to the broader principles of honest scientific practice.

- **Code originality.** All code in this repository is my own work, written from scratch. The underlying algorithms (Needleman-Wunsch alignment, Nussinov secondary structure prediction, Kabsch SVD superposition, simulated annealing) are well-established methods from the published scientific literature. Implementing known algorithms is standard and expected practice in computational biology.
- **No pre-trained models.** The pipeline does not use any pre-trained deep learning models, neural network weights, or externally trained parameters of any kind. All computations are deterministic mathematical operations (dynamic programming, singular value decomposition, affine transformations) or controlled stochastic procedures (simulated annealing, random walk generation) executed from first principles.
- **No external data.** The pipeline uses only the competition-provided PDB_RNA template library and the test sequence file. No additional databases, web resources, or external datasets are accessed at any point.
- **Offline execution.** The submission notebook is designed to run with internet access disabled, as required by Kaggle Code Competition rules.
- **Dependencies.** The only third-party dependencies are NumPy and pandas, both standard scientific computing libraries permitted in all Kaggle competitions.
- **No copying of other participants' work.** I have not viewed, copied, or derived from any other participant's solution, notebook, or discussion post relating to this specific competition.
- **Honest representation.** The methodology described in this document accurately reflects what the code does. I have not misrepresented the capabilities or limitations of this pipeline.

## Repository Structure

```
stanford-rna-fold/
    src/rna_fold/
        __init__.py          Package metadata
        parser.py            mmCIF parser; C1-prime coordinate extraction per chain
        align.py             Needleman-Wunsch alignment with RIBOSUM matrix
        search.py            Two-phase template search (k-mer filter + full alignment)
        transfer.py          Multi-template consensus and geometric gap filling
        nussinov.py          RNA secondary structure prediction (Nussinov DP)
        kabsch.py            Structural superposition via SVD
        energy.py            Coarse-grained energy function and simulated annealing
        refine.py            Distance regularization, smoothing, clash removal
        ensemble.py          Ensemble generation and maximin diversity selection
        pipeline.py          End-to-end pipeline orchestrator with multiprocessing
    kaggle/
        notebook.py          Kaggle submission notebook script
    tests/
        test_pipeline.py     Unit and integration tests (32 tests)
    setup.py
    LICENSE
```

## Installation and Usage

```bash
pip install -e .
python -m pytest tests/ -v
```

To build a distributable wheel for offline deployment on Kaggle:

```bash
pip install build
python -m build
```

This produces `dist/rna_fold-0.2.0-py3-none-any.whl`, which can be uploaded as a private Kaggle dataset and installed in a competition notebook via `pip install rna_fold-0.2.0-py3-none-any.whl`.

## Dependencies

- Python 3.8 or later
- NumPy
- pandas

No external bioinformatics libraries are required. The mmCIF parser, sequence alignment engine, substitution matrix, secondary structure predictor, structural superposition algorithm, and energy function are all implemented from scratch.

## Limitations

This implementation is a pure template-based modeling pipeline. It performs well on targets for which a homologous structure exists in the provided PDB_RNA library, but its accuracy degrades on novel folds that lack detectable sequence similarity to any known template. In such cases, the pipeline falls back to the constrained random walk backbone with simulated annealing refinement, which produces compact and sterically valid structures but cannot capture fold-specific tertiary interactions. The winning solution in Part 1 of this competition addressed this limitation by supplementing TBM with DRfold2, a deep learning-based structure prediction model, for low-homology targets.

I acknowledge this limitation openly and regard it as the primary area for future improvement. Incorporating a learned potential or a neural-network-based distance predictor as a fallback for templateless targets would be the most impactful single addition to this pipeline.

## References

- Kabsch, W. (1976). A solution for the best rotation to relate two sets of vectors. Acta Crystallographica Section A, 32(5), 922-923.
- Kirkpatrick, S., Gelatt, C. D., and Vecchi, M. P. (1983). Optimization by simulated annealing. Science, 220(4598), 671-680.
- Klein, R. J. and Eddy, S. R. (2003). RSEARCH: Finding homologs of single structured RNA sequences. BMC Bioinformatics, 4, 44.
- Needleman, S. B. and Wunsch, C. D. (1970). A general method applicable to the search for similarities in the amino acid sequence of two proteins. Journal of Molecular Biology, 48(3), 443-453.
- Nussinov, R., Pieczenik, G., Griggs, J. R., and Kleitman, D. J. (1978). Algorithms for loop matchings. SIAM Journal on Applied Mathematics, 35(1), 68-82.

## License

MIT. See LICENSE for details.
