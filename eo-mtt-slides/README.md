# Extended-object MTT slides

Build the presentation with `node eo-mtt-slides/build.mjs`. The build retains
the companion-links slide and calculates each inline lab's slide index.

Slide 3 (`s-extent-live`, `#/2`) adapts the scalar Gaussian fusion experiment
from Kalman Filter Derivations to eight extended-target returns. Its standalone
lab is `live/extent.html`; `live/extent-model.js` contains the numerical model.

The extent slider is the known Gaussian spatial standard deviation, separate
from sensor noise. All eight returns are assigned to one object and are
conditionally independent. The centroid variance is
`(extentSigma² + noiseSigma²) / 8`; the center posterior is exact under these
assumptions. The lab does not estimate extent or perform association. Its fixed
illustrative cloud keeps the measured centroid unchanged when spread changes.
The spatial model follows Granström, Baum & Reuter,
[Eq. (14)](https://arxiv.org/abs/1604.00970), specialized to one dimension.

Run `node --test eo-mtt-slides/tests/extent-model.test.cjs` for sequential/batch
agreement, numerical integration of the full cloud likelihood, extent
monotonicity, independent-return scaling, and the zero-extent limit.

The native slide includes complete equations and default numerical values
beneath its active iframe for static/print viewing. Open `?print-pdf` for the
18-slide printable deck. Live frames unload when
leaving their slide; the new SVG lab only redraws on input or resize.
