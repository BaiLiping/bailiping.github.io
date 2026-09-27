# Variational inference

The public URL serves a 15-slide interactive Bento presentation on variational inference.

- Edit `bento-deck.mjs` for slide content, notes, order, and the inline-live mapping.
- Edit `live/model.js` for the pure Gaussian-mixture EM model and `live/app.js` for controls and rendering.
- Run `node variational-inference/build-bento.mjs` from the repository root to regenerate `index.html`.

The interactive teaching unit is a pair of ordinary slides:

- `em` introduces EM as coordinate ascent on a variational lower bound.
- `em-live` mounts the deterministic Gaussian-mixture experiment automatically.

The direct QA route is `live/`. The embedded region contains no duplicate slide chrome, pauses when hidden, returns keyboard focus to Bento on Escape, and preserves a complete static/print fallback.
