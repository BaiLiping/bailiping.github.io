# RadarSplat interactive note

Public route: `/radarsplat/`. Independent educational implementation of Kung et al., arXiv:2506.01379v1, especially §§3.5, 3.6 and 8.4. This is not the authors' CUDA implementation and does not claim their reconstruction performance.

No dependencies or build step. Serve the repository root with a static server. `index.html`, `styles.css`, `model.mjs`, and `app.mjs` form the page. Navigation remains in the same tab. No telemetry, remote fonts, network inference, or dataset downloads.

Run the numerical tests with:

```sh
node radarsplat/test-model.mjs
```

The renderer computes 3D covariance, sensor-frame transformation, first-order spherical covariance, elevation-weighted additive power, circular azimuth convolution and Gaussian range leakage. Its image is 400 azimuth rows × 240 range bins, with Q = 4. The UI exposes geometry, return decomposition, sensor parameters, a beam slice, and an explicitly labeled simple peak detector. The training lab performs actual finite-difference, backtracking optimization on a synthetic one-dimensional problem.

Deliberate approximations are displayed on the page: normalized analytic antenna kernels, a restricted first-order SH slice, fixed relative power units with a 10 m reference, covariance pixel-variance floors, coarse range bins, and a one-beam training objective distinct from the paper. Azimuth width is limited to its fixed 1.8-degree kernel support. Multipath-source modeling, preprocessing, Doppler and semantic detection are omitted. Occupancy rendering is a weighted image, not a calibrated probability. The ReLU probability penalty is an inequality penalty, not an equality constraint.

Page discoverability is maintained by `scripts/build-search-index.py`; after changing title/description/routes, stage new HTML and regenerate `sitemap.xml` and `site-map/index.html`.
