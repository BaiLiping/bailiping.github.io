# Learning Latent Fields in Multiobject Tracking

28 native Bento slides explaining [Wei, Rao and Meyer, arXiv:2610.38898v1](https://arxiv.org/html/2610.38898v1), with two embedded interactive laboratories. The presentation uses the site's established serif, cream and teal design, native keyboard navigation, reading notes and annotation support.

## Edit and build

Edit `bento-deck.mjs`, then run `node latent-field-tracking/build.mjs` from the repository root. The builder preserves the native Bento runtime from the existing radio-measurement deck and generates `index.html`, `deck.json`, `live-demos.json` and `study.html`. It does not modify the homepage. The two original SVG diagrams are in `figures/`.

Run `node --test latent-field-tracking/tests/model.test.mjs` for deterministic numerical checks of the complex likelihood, covariance, Bayesian normalization and separation of hidden truth from training. After staging changed HTML, regenerate discovery with `python3 scripts/build-search-index.py` and verify it with `python3 scripts/build-search-index.py --check`.

## Content and demonstrations

The deck explains the coherent observation model and covariance likelihood, the shared conditional signal-power density, conditional normalizing flows, approximate particle BP, and the existence-weighted ELBO learning objective. It reproduces selected numerical facts from the paper's Tables I–III and Section V-C5 with metric, hardware and comparison context.

- `live/?lab=covariance`: a six-sensor, two-source synthetic complex-array example. The full covariance likelihood scans the second bearing while the first bearing and both powers are known. Beam power uses the same samples but less scene knowledge. Sensor-wise powers alone have no bearing information in this example. This is not an end-to-end tracker comparison.
- `live/?lab=learning`: one known-present object, finite-grid Bayesian filtering and a shared conditional lognormal power prior. Frozen preceding-position contexts and posterior expected log powers train a regularized mean model. Ground truth is used only for plots and evaluation. The field plot shows conditional median power. The fixed log-scale deliberately differs from the synthetic truth, so the restricted learner cannot recover its full density.

These labs are computed teaching reductions. They do not reproduce the paper's full multiobject BP/neural-flow algorithm, real datasets, or benchmark results. Assumptions and equations are documented inside the full lab pages and per-slide reading notes.

## Sources

All paper references use version 1. Original diagram assets and lab plots are authored for this companion; the paper's figures are not republished. Existing native Bento and bundled dependency license notices remain in the generated page.
