# Multidensity Fusion

A 29-slide Bento deck with six deterministic live labs, linked from Random thoughts.

## Files and editing

- `bento-deck.mjs`: readable slide content, layout helpers, equations, source references, speaker notes, and inline lab map. Edit this file directly; no generated JSON to maintain.
- `index.html`: metadata, MathJax configuration, accessibility/loading fallback, and Bento document containers.
- `boot.mjs`: installs this deck's document and loads **the exact compressed Bento engine already shipped in `../kalman-filter-derivations/index.html`**. It extracts only `bento-rt-css` and `bento-rt`; it does not import that deck's content or run its other scripts. This intentionally reuses the backend rather than copying or forking it. The reference URL and these two block IDs must remain available.
- `math.mjs`: pure numerical functions, separate from interface code.
- `live/`: responsive standalone experiments, also embedded by the existing `assets/bento-inline-live.js` host.
- `test.mjs`: deterministic numerical, edge-case, and document-structure tests.
- `test-browser.py`: end-to-end Chromium checks, including the real shared runtime, iframe navigation, six labs, and desktop/mobile screenshots.

No build step, npm dependencies, server backend, API keys, or new deployment workflow is required. A read-only, path-scoped GitHub Actions workflow runs the numerical and browser checks when this deck or its shared dependencies change. GitHub Pages serves the files as it does the other decks. The first deck load fetches the existing reference HTML to reuse its runtime. MathJax uses the same pinned CDN version and dynamic-typesetting helper as the reference deck. Direct labs do not require MathJax or Bento.

## Local checks

From the repository root:

```sh
node --test multidensity-fusion/test.mjs
python3 -m http.server 8080
```

Open `http://localhost:8080/multidensity-fusion/`. The reference deck and shared assets must be present for the full slide engine. For the optional browser test, install `playwright==1.55.0`, run `python -m playwright install chromium`, then run `python multidensity-fusion/test-browser.py`. Labs can also run independently:

- `live/?demo=prior`: common-prior accounting.
- `live/?demo=correlation`: actual versus reported scalar error variance.
- `live/?demo=geometry`: two-dimensional covariance intersection and a grid search over its weight.
- `live/?demo=pooling`: arithmetic/geometric pools, mixtures, and disjoint supports.
- `live/?demo=bernoulli`: full Bernoulli set normalization and spatial overlap.
- `live/?demo=rumors`: duplicated messages versus independent estimation errors.

The frames contain only trusted first-party code and use same-origin messaging with the shared host. Native slider keys remain available; Page Up/Down outside controls navigate slides. `unloadWhenHidden: false` retains lab inputs when navigating away and back. Every lab has an explicit reset button.

## Mathematical boundaries

Bayesian fusion assumes the displayed conditional-independence/common-information factorization. Correlated linear fusion assumes a valid joint error covariance. CI's covariance-bound statement requires unbiased, individually consistent inputs and appropriate fixed/conditional weighting assumptions. General GCI does not automatically inherit all Gaussian CI guarantees. AA density covariance is distinguished from the error covariance of an averaged estimator. Wasserstein barycenters answer a separate transport problem.

Bernoulli and PPP formulas use full finite-set normalization. AA of PPP densities is generally a mixture of PPPs, while its intensity is the exact arithmetic average. An intensity-matched PPP is explicitly called an approximation. Association, coordinate alignment, labels, and fields of view are modeling prerequisites, not solved by the pooling formula.

The density lab uses 1201 equally spaced points on `[-12,12]`, trapezoidal quadrature, and log-domain normalization. It reports a zero normalizer as undefined rather than silently adding a density floor. Input densities are numerically normalized on that finite window; tail truncation is not a physical observation. The small explicit 2x2 matrix formulas are for teaching; use factorization-based solves in a production estimator. These experiments are illustrations, not empirical tracking results.

Primary paper links are attached to relevant slides and listed on the final two reference slides. The shared Bento runtime remains under its existing MIT and third-party notices; see `index.html` and the reference deck.
