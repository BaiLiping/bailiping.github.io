# iSAM & MH-iSAM2 — One Graph, Many Hypotheses

A 25-slide native Bento presentation for the site's **Random thoughts** section. Public route: `/isam-mh-isam/`.

## Files and build

- `bento-deck.mjs`: base native Bento slide document, LaTeX, source links, notes, static laboratory fallbacks and live-region map.
- `teaching-revisions.mjs`: the equation clarifications and fourth laboratory; applied by the builder without replacing the existing Bento presentation.
- `build-bento.mjs`: validates and compiles both sources into `index.html` using the shared Bento runtime.
- `index.html`: generated, deployable presentation. Do not edit it directly.
- `engine.js`: numerical and symbolic models for the QR, tree-locality and hypothesis laboratories.
- `live/index.html` and `live/app.js`: routes for those three laboratories.
- `live/dp.html`, `live/dp-math.js`, and `live/dp.js`: separator-summary / dynamic-programming laboratory.
- `test-engine.js`, `test-revisions.mjs`, and `test-published-browser.py`: numerical, structural and browser checks.

From the repository root:

```sh
node isam-mh-isam/test-engine.js
node isam-mh-isam/test-revisions.mjs
node isam-mh-isam/build-bento.mjs
```

`.github/workflows/build-isam-slides.yml` runs these checks, builds the deck, exercises its browser integration and commits only the generated presentation after validation. Browser checks use a local static server and Playwright; screenshots are kept as workflow artifacts.

The generated deck uses the site's pinned MathJax 3.2.2 and shared Bento adapters. All experiment computation is local; there are no analytics, data uploads, API keys or persistent storage.

## Presentation behavior

Bento owns slide navigation, overview, notes, fullscreen and print. Each live laboratory immediately follows its concept slide and mounts automatically in a fixed region. The native static fallback remains available in the document for print and when the live laboratory does not mount.

Direct routes:

- `live/?demo=qr`: incremental QR and loop closure.
- `live/dp.html`: cached separator summary and child reconstruction.
- `live/?demo=tree`: Bayes-tree update locality.
- `live/?demo=mh`: delayed disambiguation and pruning.

Named slide links include `#objective`, `#linearize`, `#isam`, `#qr-demo`, `#dp-demo`, `#tree-demo` and `#mh-demo`. Inside an embedded lab, Escape returns focus to Bento and Page Up / Page Down move between slides.

## Scientific scope

The principal sources are the supplied **ICRA 2011 iSAM2** paper by Kaess et al. and **ICRA 2019 MH-iSAM2** paper by Hsiao and Kaess. Footers identify relevant sections and the references slide links to author-hosted publication pages. The source PDFs are not republished. The original 2008 iSAM paper is a background reference. Mathematical expansions and original teaching examples are labeled separately from paper algorithms and experiments.

1. **Incremental QR:** actual Givens factorization updates on a 2D translation-only graph with p0 fixed; each result is compared with a fresh batch solve. No manifold rotations or nonlinear relinearization. Row counts are not timing benchmarks.
2. **Separator-summary reuse:** an original quadratic min-sum example shows a cached function of separator s and the reconstruction rule u*(s). Changing only external evidence changes both estimates without recomputing the child summary. This is a dynamic-programming analogy, not a full iSAM2 update. For this fixed-curvature Gaussian example, marginalization and minimization give the same separator-dependent quadratic up to a constant; that equivalence must not be assumed for arbitrary models.
3. **Bayes-tree locality:** computed symbolic elimination, fill-in and clique construction for nine existing variable blocks. Highlights are computed on the pre-update tree. The explicit teaching orderings are not CCOLAMD. No numerical incremental Bayes-tree solve is claimed.
4. **Multi-hypothesis:** exact batch linear least squares per mode assignment, prefix-tree bookkeeping, a positive-DoF 95% chi-square gate and irreversible capacity pruning. Equal branch weights, covariances and dimensions make the paper's fewer-DoF preference inactive. This does not implement shared numerical MH-Bayes-tree factorization or report normalized posterior hypothesis probabilities.

Default MH experiment: 1 -> 2 -> 4 -> 2 -> 1 surviving hypotheses. A cap of one discards the needed branch before later evidence and can leave an empty set. No pruned branch is silently restored. Changing any experiment setting resets its stream.
