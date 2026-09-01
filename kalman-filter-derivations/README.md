# Kalman filter: four derivation families

The public URL serves the 14-slide presentation directly. There is no separate reading view.

- Edit `index.html` for equations, copy, references, and slide order; it is the canonical deck.
- Edit `consolidated.css` and `consolidated.js` for layout and navigation.
- Run `node kalman-filter-derivations/build-consolidated.mjs` from the repository root to synchronize the compatibility URL at `consolidated-slides.html`.
- `slides.html` redirects to the canonical deck.
- `legacy-slides.html` preserves the earlier deck and interactive experiments. Its older source remains in `deck.mjs` and `routes.mjs`; `build.mjs` now rebuilds only the archive.

The 14-page structure is overview, model, four consecutive idea/equation pairs, Gaussian elimination, numerical forms/control duality, equivalence, and references. The print button prints only the four equation sheets, one A4 landscape page per family.

The taxonomy consolidates synonymous derivations and separates statistical principles from implementations. KL updating is explicitly identified as a variational form of Bayes, not an independent probability law.
