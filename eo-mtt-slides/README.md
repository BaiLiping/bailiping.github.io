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

Slide 4 (`s-two-return-live`, `#/3`) follows the prior/likelihood/posterior
visualization on page 4 of `kalman-filter-derivations/`. Its main experiment is
**changing true extent with fixed priors**, while jointly estimating center and
extent. The standalone lab is `live/two-return.html`.

The single prominent slider changes true full size from 1 to 8 m. Small/large
presets select 2 and 8 m. The two chosen observations stay in the right half:
`z1 = 0.125 L*`, `z2 = 0.425 L*`. True center remains 0 m and sensor standard
deviation remains 0.3 m. These are controlled illustrative measurement sets,
not random playback. Each update restarts from the same priors; dragging does
not accumulate observations. Truth, source fractions and source side are not
passed to inference.

A finite 1D body occupies `[c − L/2, c + L/2]`. Each detection has an independent
latent source `u_i ~ Uniform[-L/2,L/2]` and independent Gaussian sensor noise.
Integrating the source out gives the single-detection likelihood
`[Phi((z_i−c+L/2)/sigma) − Phi((z_i−c−L/2)/sigma)] / L`.
This is a segment specialization of the spatial-source convolution in
[Granström, Baum & Reuter, Eq. (6)](https://arxiv.org/abs/1604.00970).

Fixed priors: center `N(0, 1.2²)` m; length lognormal with median 4 m and log
standard deviation 0.5. Numerical quadrature in `(c, log L)` retains dependence
in the full joint posterior. The length-density conversion includes the `1/L`
Jacobian. Integration bounds retain distant tails when observations disagree
with the priors. Both center and extent are estimated at every slider position;
the Center/Extent buttons only switch the plotted marginal.

Each orange curve integrates the other variable using its **prior**. It is
normalized to unit area over the fixed visible interval for comparing curve
shapes; it is not a posterior or a plug-in likelihood at an estimated size.
This display scaling never affects inference. Prior and posterior curves keep
their density values. Horizontal and vertical scales remain fixed across all
true sizes, and the prior is evaluated on an invariant plotting grid so it
stays visually identical. The body sketch also uses a fixed metre scale.
Posterior means, standard deviations and equal-tailed 95% intervals update
with the observations. Two same-side detections do not identify the full body.

`two-return-model.js` contains inference; `two-return-view.js` supplies the
shared SVG views. The generated `two-return-fallback.svg` compares the 2 m
and 8 m examples, with center and extent curves for each and matched axes.
Run the numerical and comparison checks with
`node --test eo-mtt-slides/tests/*.test.cjs`.
