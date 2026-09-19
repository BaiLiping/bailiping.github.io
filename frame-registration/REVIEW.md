# Frame-registration accuracy review — 7 September 2026

## Scope

This review covers `frame-registration/index.html`, the 20-slide Bento document
in `frame-registration-slides/index.html`, and its live lab. It preserves the
existing lesson sequence, visual language, embedded runtime, and interactive
examples. The browser solvers are educational 2-D implementations, not official
paper reproductions. None of the numerical tests establishes a universal
ranking or a global-correctness guarantee.

## Corrections

- Explicit source-to-target convention, proper rotations, determinant correction,
  2-D versus 3-D observability, and rigid versus similarity models.
- RANSAC's ceiling and sampling assumptions; objective-value convergence versus
  pose convergence and true alignment; robust-loss and GICP covariance caveats.
- NDT's hard point-to-cell assignments, within-cell derivative assumptions,
  unnormalized score, and covariance regularization. The slide lab is
  translation-only direct search; the article implements damped Newton.
- CPD mixture normalization, original clutter convention, optional similarity,
  variance estimation (not guaranteed annealing), and optional full matrix storage.
- FilterReg supports analytic variance estimation. The walkthrough uses sparse
  Cartesian filtering, not the paper's custom permutohedral lattice.
- D random frequencies produce 2D features. Finite features need not identify a
  distribution uniquely. Implicit differentiation requires a regular selected
  optimum, not merely a smooth objective.
- The heuristic soft-target race is not a complete CPD implementation. The old
  PMBM-labeled race is now a hypothesis-mixture toy, not a PMBM filter. The small
  learned-flow network is not RAP and supplies no evidence about RAP's performance.
- The benchmark reports rotation-only and joint rotation/translation success;
  unsupported static leaderboards and universal capture-angle claims were removed.
- The NN landscape now plots the capped squared-distance objective actually
  decreased by the translation-only gated update, with a fixed denominator.
- NDT integer-key aliasing and its hover decoder were fixed together, and all four
  walkthrough auto flags reset consistently.

## Reproducible checks

```sh
python -m pip install matplotlib playwright
python -m playwright install chromium
python scripts/frame-registration-audit.py
python scripts/finalize-frame-registration.py
node --test tests/frame-registration.test.cjs
python tests/frame-registration-browser.py
```

The one-time migration checks source matches, rolls back if a patch stage fails,
then leaves directly editable deployed HTML. Subsequent runs do not overwrite
manual edits to a marked article. The accompanying JSON files hold the reviewed
prose and slide text. CI checks idempotence and captures browser screenshots and
a JSON report.

The Node tests execute functions extracted from the deployed inline scripts:
paired rigid recovery, composition, reflection exclusion, regularized NDT cells,
non-aliasing cell keys, RFF Jacobian finite differences, fixed-denominator capped
cost, descent of its gated translation step, finiteness of nine race solvers on
six seeded settings, RANSAC initialization invariance, NDT gradient/Hessian finite
differences, accepted Newton score improvement, and article/deck consistency.
Browser checks cover MathJax, controls, NDT hover, the complete 25-seed benchmark,
20 slides, equation images, text bounds, local resources, and all live lab tabs.
Screenshots include desktop and mobile views. They are smoke tests, not an
exhaustive accessibility or device-compatibility certification.

## Primary sources used

- Arun, Huang, and Blostein, *Least-Squares Fitting of Two 3-D Point Sets*, IEEE
  TPAMI, 1987. DOI: 10.1109/TPAMI.1987.4767965.
- Umeyama, *Least-Squares Estimation of Transformation Parameters Between Two
  Point Patterns*, IEEE TPAMI, 1991. DOI: 10.1109/34.88573.
- Fischler and Bolles, *Random Sample Consensus*, CACM, 1981.
  DOI: 10.1145/358669.358692.
- Besl and McKay, *A Method for Registration of 3-D Shapes*, IEEE TPAMI, 1992.
  DOI: 10.1109/34.121791.
- Chen and Medioni, *Object Modelling by Registration of Multiple Range Images*,
  Image and Vision Computing, 1992. DOI: 10.1016/0262-8856(92)90066-C.
- Segal, Haehnel, and Thrun, *Generalized-ICP*, RSS, 2009.
  https://doi.org/10.15607/RSS.2009.V.021
- Biber and Straßer, *The Normal Distributions Transform: A New Approach to
  Laser Scan Matching*, IROS, 2003. DOI: 10.1109/IROS.2003.1249285.
- Myronenko and Song, *Point Set Registration: Coherent Point Drift*, TPAMI, 2010.
  https://arxiv.org/abs/0905.2635
- Gao and Tedrake, *FilterReg: Robust and Efficient Probabilistic Point-Set
  Registration Using Gaussian Filter and Twist Parameterization*, CVPR, 2019.
  Section 3.3 explicitly discusses optimized variance.
  https://arxiv.org/html/1811.10136v3
- Crane et al., *MMD-Reg*, 2026. https://arxiv.org/abs/2606.27818
- Gretton et al., *A Kernel Two-Sample Test*, JMLR, 2012.
  https://www.jmlr.org/papers/v13/gretton12a.html
- *Register Any Point*, https://arxiv.org/abs/2512.01850
- Vizzo et al., *KISS-ICP*, https://arxiv.org/abs/2209.15397

Specific algorithm claims should be checked against their cited model and paper
version. The review does not reproduce the training of the embedded MLP, the
published RAP/FilterReg/MMD-Reg evaluations, or a production 3-D SLAM pipeline.
