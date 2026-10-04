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

Open `?print-pdf` for the 18-slide print edition. Leaving the live slide unloads
its iframe. Playback also stops on document hiding, the Bento pause event, and
page unload; re-entry starts paused.
