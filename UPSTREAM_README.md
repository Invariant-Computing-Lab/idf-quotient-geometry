# IDF Invariant Geometry - Submission and Reproducibility Package

This is a personal academic research package for the manuscript:

**Quotient-Invariant Reformulation of Intrinsic Bosonic Geometry: A Falsification-Driven Synthetic Study**

Author: **Emirhan Yıldırım**  
Version: **1.0.0**  
Date: **12 August 2026**

## Scope

The package reformulates a legacy intrinsic-geometry model as a quotient/invariant/reconstructibility problem and tests the resulting mathematical invariance claims with deterministic synthetic experiments.

The package does **not** claim:

- validation of the Standard Model reinterpretation;
- discovery of new particles;
- derivation of measured particle masses or couplings;
- proof that fundamental bosons possess an intrinsic two-dimensional manifold;
- experimental support from collider or other physical datasets.

## Package contents

- `Main_Manuscript.pdf` / `.docx` / `.md` - main research article.
- `Supplementary_Appendix.pdf` / `.docx` / `.md` - derivations, experiment definitions, detailed results, claim audit, and reproducibility instructions.
- `code/run_all.py` - master deterministic runner.
- `code/transformation_and_quotient_suite.py` - topology, gauge, Picard, coordinate, unit, signature, and spectral-scale tests.
- `code/final_reconstructibility_suite.py` - fingerprint collisions, ablation, noise, finite-resolution, and critical-locus experiments.
- `results/*.csv` - raw numerical outputs.
- `results/figures/*.png` - manuscript figures generated from the final run.
- `run_log.txt` - final end-to-end run log.
- `requirements.txt` - Python dependencies.
- `SHA256SUMS.txt` - integrity manifest.

## Reproduction

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python code/run_all.py
```

All stochastic tests use seed `20260812`.

## Headline synthetic results

- Gauss-Bonnet topology remained stable to approximately `1e-14` under random geometric deformation while surface area drifted.
- SU(2) trace and Wilson-loop conjugation invariants remained stable to approximately `1e-15` while components changed substantially.
- Picard/divisor degree was preserved exactly under degree-zero representative shifts.
- A curvature-over-area observable was **not** protected by topology alone; it became stable only after area/polarization restoration.
- Raw Laplace eigenvalues failed global scale invariance, whereas `A*lambda_n` and `lambda_n/lambda_1` were stable to approximately machine precision.
- In a restricted 63-class synthetic benchmark, topology+gauge+degree reached only 28.57% exact-reconstruction ceiling; adding normalized spectrum reached 100%.
- The smallest full-recovery subset in that benchmark was gauge + degree + normalized spectrum. This is a benchmark-specific redundancy result, not a universal theorem.
- With eight normalized spectral values, seven-template geometry recovery remained 100% through 2% multiplicative spectral noise, 99.74% at 5%, and 92.06% at 10%.

## Research-integrity note

Synthetic mathematical validation can establish that a proposed quantity has the claimed transformation behavior inside the tested construction. It cannot establish that the construction describes nature. Any future physical manuscript must separately derive a consistent action, effective four-dimensional observables, quantum consistency, and externally testable deviations from established theory.
