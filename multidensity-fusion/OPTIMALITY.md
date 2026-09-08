# Optimality companion

The displayed deck now has 40 slides: the original 29 plus 11 companion slides.
`boot.mjs` applies the pure `withOptimality` transformation from `optimality.mjs`
before serializing the document for Bento. The original `bento-deck.mjs` and all
six live experiments remain unchanged. Page numbers, progress bars, and the
live-map indexes are recomputed after insertion.

## Content

The additions are a criterion map and proofs for Bayesian fusion, known-error-
correlation fusion, covariance intersection, Gaussian moment projection,
one-dimensional Wasserstein barycenters, full-set AA/GCI, Bernoulli fusion,
Poisson GCI, Poisson approximation of an arithmetic mixture, and geometric
consensus. The existing AA and GCI proof cards gain explicit equality conditions.

Every result specifies its objective and admissible family. In particular, CI
weight selection is described as optimizing a chosen bound within the CI family,
not minimizing an unknown actual MSE. Gaussian and Poisson projections are not
claimed to preserve the full arithmetic mixture. Notes cover integrability,
support, boundary, and covariance-consistency conditions. Each companion inherits
the primary references from its preceding method slide.

## Tests

From the repository root:

```sh
node --test multidensity-fusion/test.mjs multidensity-fusion/test-optimality.mjs
python multidensity-fusion/test-browser.py
```

The new Node suite checks the mathematical objective-gap identities and the
40-slide document/live-map transformation. The browser test expects 40 slides
and verifies each of the six remapped live entries.

During preparation, the nine new tests passed using an explicitly identified
29-slide structural fixture for the document test. All 60 expressions on the
11 added slides compiled with MathJax, and their standalone Chromium previews
had no detected text or equation overflow. The complete Bento runtime and live
iframe regression were not run in the preparation environment; the browser
command above remains the required end-to-end check in a full checkout.
