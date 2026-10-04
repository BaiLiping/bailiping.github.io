# Extended-object MTT slides

Build with `node eo-mtt-slides/build.mjs`. The build preserves the companion
links, computes inline slide indexes, and generates the model-derived
`live/extent-fallback.svg` for the printable slide.

Slide 3 (`s-extent-live`, `#/2`) is a moving 2D extended-object tracking
experiment. Its standalone lab is `live/extent.html`.

## Measurement and tracking model

Each scan contains independent source points uniformly distributed over an
elliptical object's area, followed by additive Gaussian sensor noise.
Length and width controls are the **full physical axes in metres**. The
orientation control rotates the body. A fixed count is used per scan.

The estimator is a factorized Gaussian–inverse-Wishart random-matrix filter:
its unknowns are position/velocity and a positive-definite extent matrix X.
It uses the approximate measurement update and forgetting prediction in
[Granström, Baum & Reuter, Eq. (14), Tables IV and IX](https://arxiv.org/abs/1604.00970),
which summarize the Feldmann et al. method. This implementation uses symmetric
positive-definite matrix square roots.

- The physical ellipse is defined by `(z − center)' X⁻¹ (z − center) ≤ 1`.
- Uniform-area source covariance is `X/4`; the filter uses the Gaussian
  measurement approximation `N(z; Hx, X/4 + R)`.
- A scan's centroid updates motion. Its scatter and centroid innovation update
  the inverse-Wishart extent parameters, correcting for known sensor noise.
- In the reference's inverse-Wishart convention, `E[X] = V/(nu − 6)` in 2D.
- The fixed circular extent prior is 7 m × 7 m for every truth setting.
  `update(prior, detections, R)` has no access to the true shape or trajectory.
- A separate dotted contour depicts the Gaussian 95% center region,
  using `5.991464547 * P_position`. It is not a body-size estimate.

This is a synthetic single-object example with known association, not a
multi-object association implementation or a sensor-data benchmark.
The approximation and scope are visible in the lab's model disclosure.

## Interaction and verification

Play/Next scan propagate the filter through new scans. The timeline replays
up to 40 scans; sliders replay the same seeded random draws with the new
physical settings. Reset restores the defaults. Truth, noise-free sources,
and the center-uncertainty contour can be toggled independently.

The model is in `live/extent-model.js`; `live/extent-view.js` shares the scene
renderer between the live lab and the print fallback.

Run `node --test eo-mtt-slides/tests/extent-model.test.cjs`. Tests cover
uniform-area sampling and noise variance, an independently hand-calculated
update, shape learning from equal-centroid clouds, rotation equivariance,
truth-free inference, learning over multiple seeds, sensor-noise correction,
positive-definite covariances across the control range, and deterministic replay.

Open `?print-pdf` for the 19-slide print edition. Leaving a live slide unloads
its iframe. Playback also stops on document hiding, the Bento pause event, and
page unload; re-entry starts paused.
The print route suppresses Bento's automatic fullscreen request: browser
fullscreen rules otherwise constrain the complete deck to one viewport after
interaction. Its static layout also supports an ordinary screen preview.

## Two same-side returns: slide 4

Slide 4 (`s-two-return-live`, `#/3`) adds a second experiment after the 2D
tracker. It follows the prior/likelihood/posterior curves on page 4 of
`kalman-filter-derivations/`, while **jointly estimating center and extent**.
The standalone lab is `live/two-return.html`.

A finite 1D body occupies `[c − L/2, c + L/2]`. Each detection has its own
latent source `u_i ~ Uniform[-L/2,L/2]` and independent Gaussian sensor noise.
Integrating the source out gives the exact single-detection likelihood
`[Phi((z_i−c+L/2)/sigma) − Phi((z_i−c−L/2)/sigma)] / L`.
This is a segment specialization of the spatial-source convolution in
[Granström, Baum & Reuter, Eq. (6)](https://arxiv.org/abs/1604.00970).

The Gaussian center prior and lognormal length prior are initially independent.
The likelihood product creates a joint posterior; midpoint quadrature in
`(c, log L)` retains that dependence. The length-density conversion includes
the `1/L` Jacobian. Integration bounds include distant length tails for
conflicting priors and observations. The two plotted posteriors are marginals
of this joint result, with equal-tailed 95% credible intervals. Each orange
curve integrates the other variable's **prior** and is scaled for display;
it is not a plug-in likelihood evaluated at an estimated size or center.
The optional joint view displays the coupled posterior directly.

The illustrated pair is deliberately chosen at 25% and 85% of one half of
the body: default observations `0.75, 2.55 m`, true center `0 m`, full extent
`6 m`. Both detections stay on the selected side and within the body as its
center and extent change. They are chosen observations, not random playback;
the noise control changes uncertainty without resampling them. The estimator
receives only observations, sensor noise, and independently controlled priors.
It does not receive true center, true size, source positions, or source-side
information. Two returns can leave a broad set of plausible centers and sizes.

Numerical logic and SVG views are separate in `two-return-model.js` and
`two-return-view.js`. The build generates `two-return-fallback.svg` from the
same numerical result. Run both suites with
`node --test eo-mtt-slides/tests/*.test.cjs`.
