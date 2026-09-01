# iSAM & MH-iSAM2 — One Graph, Many Hypotheses

A 24-slide Bento presentation for the site's **Random thoughts** section. Public route: `/isam-mh-isam/`.

## Files

- `bento-deck.mjs`: canonical native Bento slide document, LaTeX, notes, links, static lab fallbacks, and live-region map.
- `build-bento.mjs`: validates and compiles the canonical document into `index.html` using the shared Bento runtime.
- `index.html`: generated, deployable presentation. Do not edit it directly.
- `engine.js`: dependency-free numerical and symbolic models, shared by the browser and Node tests.
- `live/index.html` and `live/app.js`: direct and embedded routes for the three deterministic laboratories.
- `test-engine.js`: numerical and bookkeeping regression tests.

Rebuild from the repository root with:

```sh
node isam-mh-isam/build-bento.mjs
```

The generated deck uses the site's pinned MathJax 3.2.2 and shared Bento adapters. All experiment computation is local; there are no analytics, uploads, API keys, or persistent storage.

## Presentation behavior

Bento owns slide navigation, overview, notes, fullscreen, accessibility, and print. Each live lab follows its concept slide and mounts automatically in a fixed region—there are no launch buttons or hidden state slides. The underlying native fallback remains useful in print and when JavaScript is unavailable.

Direct routes remain available at `live/?demo=qr`, `live/?demo=tree`, and `live/?demo=mh`. In an embedded lab, Escape returns focus to Bento and Page Up / Page Down move between slides.

## Scientific scope

The principal sources are the supplied **ICRA 2011 iSAM2** paper by Kaess et al. and **ICRA 2019 MH-iSAM2** paper by Hsiao and Kaess. Footers link to author-hosted publication pages; the source PDFs are not republished. The 2008 iSAM paper is a background reference. Mathematical expansions and original teaching examples are labeled separately from paper algorithms and experiments.

1. **Incremental QR:** actual Givens factorization updates on a 2D translation-only graph with p0 fixed; each result is compared with a fresh batch solve. No manifold rotations or nonlinear relinearization. The displayed row counts are not timing benchmarks.
2. **Bayes tree:** computed symbolic elimination, fill-in and clique construction for nine existing variable blocks. Highlighted ancestors are computed on the pre-update tree. The alternative orderings are explicit teaching choices, not CCOLAMD. No numerical incremental Bayes-tree solve is claimed.
3. **Multi-hypothesis:** exact batch linear least squares per mode assignment, prefix-tree bookkeeping, a positive-DoF 95% chi-square gate and irreversible capacity pruning. Equal branch weights, covariance and dimensions make the paper's fewer-DoF preference inactive. This does not implement shared numerical MH-Bayes-tree factorization and does not report normalized posterior hypothesis probabilities.

Default MH experiment: 1 -> 2 -> 4 -> 2 -> 1 surviving hypotheses. A cap of one discards the needed branch before later evidence and can leave an empty set. No pruned branch is silently restored. Changing any experiment setting resets its data stream.

Run `node isam-mh-isam/test-engine.js` from the repository root. Run the strict structural audit with:

```sh
python3 ~/.codex/skills/build-interactive-slides/scripts/audit_bento.py isam-mh-isam/index.html --strict
```
