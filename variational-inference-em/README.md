# One Bound, Two Algorithms

A 26-slide Bento presentation on variational inference and expectation–maximization, with the same renderer, cream/green/rust visual language, equation panels, navigation, and inline live regions as the site's **One Filter, Many Derivations** deck.

Published route: `/variational-inference-em/`.

## Content

The deck derives the ELBO from KL and Jensen, derives mean-field coordinate updates, explains exact EM and its likelihood-monotonicity proof, works through Gaussian-mixture responsibilities and parameter updates, and distinguishes variational EM from fully Bayesian VI. Equation-summary slides, speaker notes, clickable primary references, a conjugate Bayesian-mixture example, and a reparameterization extension are included.

## Interactive examples

- `live/?demo=meanfield`: a normalized correlated Gaussian target, current mean-field Gaussian, product of exact marginals, marginal-variance comparison, exact reverse KL, and sequential CAVI. The default view is the optimum; **Start CAVI from offset** begins the iteration. Correlation changes reset the objective. Contours have Mahalanobis radius 2, not 95% probability coverage.
- `live/?demo=em`: a seeded, one-dimensional, two-component Gaussian mixture. The **E-step** changes responsibilities only; the **M-step** changes parameters only. The likelihood, ELBO, and their exact KL gap are displayed per observation. Controls vary data separation, initialization, seed, and whether variances are learned. Identical means deliberately expose a symmetric fixed point.

Data are synthetic, with 120 observations, generating weights 0.4/0.6, and standard deviation 0.8. Fixed-variance EM uses variance 0.64. When variances are learned, the explicit constraint is variance >= 0.09; the clipped weighted-residual update is the exact one-dimensional constrained M-step. These are teaching examples, not empirical performance claims.

The browser code has no external numerical dependencies, data services, keys, or telemetry. Classic scripts are intentional: they work in the host's opaque-origin `allow-scripts` iframe sandbox. MathJax and the Bento host assets follow the existing site's setup. Math rendering therefore still needs the existing CDN dependency.

## Editing and rebuilding

Edit `bento-deck.mjs` for content and `live/` for the examples. Do not manually edit generated `index.html`.

```sh
node variational-inference-em/tests.mjs
node variational-inference-em/build.mjs
python3 -m http.server 8765
```

The builder copies the existing `kalman-filter-derivations/index.html` **at build time**, replaces its document and live-map blocks, rewrites routes/metadata, and retains the renderer and third-party notices. It does not change the Kalman deck. The generated presentation does not fetch the Kalman deck at runtime.

For browser validation, in a separate shell while the local server is running:

```sh
pip install playwright==1.55.0 Pillow==11.3.0
python -m playwright install chromium
python variational-inference-em/browser-test.py
```

`.github/workflows/build-vi-em.yml` runs syntax checks, numerical tests, a browser pass through all slides, both interactive controls, sandbox loading, and mobile layouts. It uploads screenshots and reports, then commits only the generated presentation after success. The homepage card is maintained separately. GitHub Pages uses the repository's existing deployment configuration.

The numerical tests cover seven correlations, 54 mixture configurations, and 4,320 E/M half-steps. They check normalization, bound tightness after E, bound and likelihood ascent after M, the gap identity, variance constraints, symmetric initialization, and deterministic data generation.

## Primary references

1. Blei, Kucukelbir & McAuliffe (2017), *Variational Inference: A Review for Statisticians*, JASA 112(518), 859–877. https://arxiv.org/abs/1601.00670
2. Dempster, Laird & Rubin (1977), *Maximum Likelihood from Incomplete Data via the EM Algorithm*, JRSS B 39(1), 1–22. https://doi.org/10.1111/j.2517-6161.1977.tb01600.x
3. Neal & Hinton (1998), *A View of the EM Algorithm that Justifies Incremental, Sparse, and Other Variants*, in Learning in Graphical Models, pp. 355–368. https://www.cs.toronto.edu/~hinton/absps/em.htm
4. Kingma & Welling (2014), *Auto-Encoding Variational Bayes*, ICLR. https://arxiv.org/abs/1312.6114
