# Advanced State Variable Representations

A 24-slide native Bento presentation for the site's **Random thoughts** section. Public route: `/advanced-state-representations/`.

The deck is a concise interpretation of Chapter 2, “Advanced State Variable Representations,” from the supplied 2026 *SLAM Handbook*. It covers manifold-valued rotations and poses, tangent-space optimization, Lie-group uncertainty and Jacobians, splines, SDE-derived Gaussian processes, and continuous-time trajectories on Lie groups. The source PDF is not copied into the repository.

## Files and build

- `bento-deck.mjs`: canonical Bento slide document, LaTeX, speaker notes, source links, complete static lab fallbacks, and live-region map.
- `build.mjs`: validates the authored document and compiles `index.html` using the site's shared Bento runtime.
- `index.html`: generated deployable deck; do not edit it directly.
- `model.js`: dependency-free deterministic models shared by the browser and Node tests.
- `live/index.html`, `live/styles.css`, and `live/app.js`: direct and embedded routes for all three labs.
- `test-model.cjs`: numerical and invariance checks.

From the repository root:

```sh
node advanced-state-representations/test-model.cjs
node advanced-state-representations/build.mjs
python3 ../Skills/build-interactive-slides/scripts/audit_bento.py advanced-state-representations/index.html --strict
```

## Presentation behavior

Bento owns slide identity, navigation, overview, notes, scaling, and print. Each live lab immediately follows its concept slide and mounts automatically over the exact static fallback region. There are no launcher buttons or hidden state slides.

Direct lab routes:

- `live/?demo=manifold`: repeated planar rotation updates and constraint drift.
- `live/?demo=spline`: local support in linear and cubic open-uniform B-splines.
- `live/?demo=gp`: scalar random-walk GP smoothing with asynchronous measurements and a tridiagonal information matrix.

Inside an embedded lab, Escape returns focus to Bento and Page Up / Page Down navigate the deck.

## Scientific scope

The three browser labs are original, deterministic teaching models. They reproduce no handbook dataset and report no performance benchmark.

1. **Manifold lab:** exact planar rotation composition is compared with deliberately crude repeated first-order matrix accumulation to expose loss of orthogonality.
2. **Spline lab:** basis weights use Cox–de Boor recursion. The displayed speed is a numerical derivative used only as a teaching readout.
3. **GP lab:** an actual linear Gaussian solve estimates eight scalar control states from a random-walk prior and asynchronous interpolated measurements. The pose-valued Lie-group construction is explained in the static deck, not claimed by the scalar lab.
