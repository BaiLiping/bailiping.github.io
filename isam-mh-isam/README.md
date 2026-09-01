# iSAM & MH-iSAM2 — One Graph, Many Hypotheses

A 24-slide, responsive, light/card-based deck for the site's **Random thoughts** section. Public route: `/isam-mh-isam/`.

## Files

- `index.html`: accessible page shell and pinned MathJax 3.2.2 loader.
- `styles.css`: light card theme, responsive layout and print rules.
- `deck.js`: slide content, LaTeX, source footers, speaker notes and original diagrams.
- `engine.js`: dependency-free numerical/symbolic teaching models (also loadable with Node).
- `demos.js`: SVG views and live controls.
- `app.js`: direct slide links, keyboard/touch navigation, overview, notes and fullscreen.
- `test-engine.js`: numerical and bookkeeping regression tests.

No build is needed. Open through any static server; the only network dependency is MathJax from the pinned CDN. Equations are typeset once; all demo computation is local. No analytics, data uploads or user storage are added.

## Navigation

Arrow keys / Page Up / Page Down; Home / End; O for overview; N for notes; F for fullscreen. Inputs retain their normal keyboard behavior. Swipe outside a control, equation or plot to navigate on mobile. Links such as `#qr-demo`, `#tree-demo` and `#mh-demo` open experiments directly. Browser Print renders all slides with the current demo states.

## Scientific scope

The principal sources are the supplied **ICRA 2011 iSAM2** paper by Kaess et al. and **ICRA 2019 MH-iSAM2** paper by Hsiao and Kaess. Footers link to author-hosted publication pages; the source PDFs are not republished. The 2008 iSAM paper is a background reference. Mathematical expansions and original teaching examples are labeled separately from paper algorithms and experiments.

1. **Incremental QR:** actual Givens factorization updates on a 2D translation-only graph with p0 fixed; each result is compared with a fresh batch solve. No manifold rotations or nonlinear relinearization. The displayed row counts are not timing benchmarks.
2. **Bayes tree:** computed symbolic elimination, fill-in and clique construction for nine existing variable blocks. Highlighted ancestors are computed on the pre-update tree. The alternative orderings are explicit teaching choices, not CCOLAMD. No numerical incremental Bayes-tree solve is claimed.
3. **Multi-hypothesis:** exact batch linear least squares per mode assignment, prefix-tree bookkeeping, a positive-DoF 95% chi-square gate and irreversible capacity pruning. Equal branch weights, covariance and dimensions make the paper's fewer-DoF preference inactive. This does not implement shared numerical MH-Bayes-tree factorization and does not report normalized posterior hypothesis probabilities.

Default MH experiment: 1 -> 2 -> 4 -> 2 -> 1 surviving hypotheses. A cap of one discards the needed branch before later evidence and can leave an empty set. No pruned branch is silently restored. Changing any experiment setting resets its data stream.

Run `node isam-mh-isam/test-engine.js` from the repository root. Browser QA was also run at 1440x900, 1280x768 and 390x844 with 50 rendered equations; no JavaScript errors or overflowing content cards were found. Numerical results were independently checked against NumPy least squares and Cholesky during authoring.
