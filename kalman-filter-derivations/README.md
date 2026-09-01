# Kalman filter: four derivation families

The public URL serves a 17-slide interactive Bento presentation. There is no separate reading view.

- Edit `bento-deck.mjs` for slide content, notes, order, and inline-live mappings.
- Run `node kalman-filter-derivations/build-bento.mjs` from the repository root to generate the canonical `index.html` and compatibility redirects.
- `slides.html` and `consolidated-slides.html` redirect to the canonical deck while preserving query strings and hashes.
- `legacy-slides.html` preserves the earlier 26-slide Bento deck. Its source remains in `deck.mjs` and `routes.mjs`; `build.mjs` rebuilds only that archive.

The deck preserves the original 14 topics and adds three ordinary live slides. Each experiment immediately follows its static introduction:

- `model` → `model-live`: scalar Gaussian fusion
- `mse` → `mse-live`: covariance geometry
- `implementations` → `implementations-live`: finite-precision forms

The live routes remain available under `live/` for direct QA. Inline regions mount automatically only on their active slide, keep deterministic defaults, hand navigation back to Bento, and leave a complete static/print fallback underneath.

The taxonomy consolidates synonymous derivations and separates statistical principles from implementations. KL updating is explicitly identified as a variational form of Bayes, not an independent probability law.
