# Grassmannian manifold

Public routes: `/grassmannian-slides/` (presentation), `/grassmannian/` (companion
guide), `/grassmannian/lab.html?lab=association` (the requested paper demo), and
`/grassmannian-slides/grassmannian.pdf` (static slide export).

Audience: visual intuition plus graduate-level mathematics. `deck.mjs` is the
editable source; `build.mjs` preserves the existing site's licensed Bento runtime
without editing its compressed code. It generates the presentation, `deck.json`,
and a separate static print layout. Original SVG illustrations come from
`../grassmannian/figures.mjs`.

```sh
node grassmannian/figures.mjs
node --test grassmannian/tests/*.test.mjs
node grassmannian-slides/build.mjs
python3 -m http.server 8767 --bind 127.0.0.1
```

Five shared labs: basis invariance, principal angles, geodesic interpolation,
PCA, and affine line/plane data association. Each TRY LIVE control opens a lazy
dialog; Back and Escape remove the frame and restore focus. Mobile labs scroll
vertically. The normal slides include static examples and source notes.
`print.html` exports at 1280 × 720 CSS pixels per page. Wait for local MathJax
typesetting before generating the PDF and verify its page count against the deck.

The association engine follows Lusk & How, arXiv:2205.08556v1, equations (2)–(5),
(8)–(9), Proposition 1. It evaluates every feasible clique in a tiny synthetic
24-vertex graph, including nonmaximal cliques. This solves the displayed discrete
weighted density objective; it does not implement CLIPPER, feature extraction,
the paper's pose registration, or its KITTI evaluation. Mixed-dimensional
principal-angle norms are used as dissimilarities, not claimed to separate
different-dimensional subspaces. Ground-truth IDs are only used to evaluate the
selected associations. The tests verify this separation and pose invariance.

The real manifold is denoted Gr(k,n) with k first. The handbook reverses those
arguments. Geodesic distance uses the canonical horizontal Frobenius metric;
projector distance includes 1/√2. Complex analogues are discussed but not used in
the numerical laboratories. Numerical source and tests live in `../grassmannian/`.
