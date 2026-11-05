# Splat-LOAM paper companion

Public route: `/splat-loam/`, linked under **Random thoughts** on the homepage.
The deck contains 25 native Bento slides, two embedded numerical laboratories,
speaker notes, a responsive reading mode and original credited paper figures.
It follows the final ICCV 2025 paper by Giacomini et al. and its supplement.

## Edit and build

Edit `bento-deck.mjs`, then run from the repository root:

```sh
node splat-loam/build.mjs
node --test splat-loam/tests/*.test.mjs
git add splat-loam/*.html splat-loam/live/index.html
python3 scripts/build-search-index.py
python3 scripts/build-search-index.py --check
```

The builder uses the existing native Bento runtime in
`mpc-detection-to-bounce-count-slides/index.html`, the shared local MathJax
bundle, and `withAuthoring` from `scripts/authoring-build.mjs`. Keep any future
`authoring.json` changes and stable slide/object IDs. It generates `index.html`,
`deck.json`, `live-demos.json` and `study.html`. Shared theme and annotation
assets remain unchanged.

## Interactive laboratories

- `live/?lab=rendering` performs exact 3D ray-plane intersections with three
  flat Gaussian surface patches, per-ray sorting and front-to-back alpha
  compositing. Controls change tilt, scale, opacity and sensor yaw. The image
  covers a 110-degree azimuth window. It exposes accumulated range,
  opacity-normalized expected range and opacity separately. It does not model
  the released renderer's optional median-depth mixture, CUDA tiling, seam
  bounds, backpropagation or mapping.
- `live/?lab=registration` is a deterministic SE(2) point-to-line analogue of
  the geometric part of the tracker. It recomputes associations and performs
  robust damped Gauss–Newton steps. A corner constrains the pose; a corridor
  leaves along-wall translation unobservable. Synthetic truth generates the
  data and evaluates errors but is never an optimizer input. This lab omits
  the paper's PCA kd-tree, range-image residual and full SE(3) mapping loop.

The `embed=1` variant fits the 1136 × 475 slide region. The shared
`bento-inline-frame.js` bridge handles readiness, visibility and deck keyboard
navigation. Standalone pages retain all controls and explanatory scope.

The rendering lab defaults to a discrete grayscale range image. False color is
an optional display encoding of the same metric range, not learned RGB or
reflectivity. Its fixed range scale does not mix in opacity. Each selected cell
uses exactly its center ray in the 3D view and hit table. The 72 × 32 and 144 × 64
grids sample a cropped 110° × 48° azimuth–elevation window for teaching; they do
not prescribe a real sensor's beam pattern.

The front-splat count measures forward rays crossing the chosen ellipse
`u² + v² ≤ 9` (a 3σ contour in normalized tangent coordinates), before opacity
or visibility weighting. It is not a count of reflected/detected returns. The
Gaussian renderer retains its tails outside this counting contour. Missing
expected range at accumulated opacity `O ≤ 1e-6` is shown as neutral background,
not zero metres. The optional median-range mixture remains omitted.

## Scientific distinctions

The paper's frame-to-model tracker combines point-to-plane geometry with
range-image warping. It is separate from Gaussian map optimization through the
differentiable spherical rasterizer. It is not the NDT frontend used in a
separate radio adaptation. “Photometric” here refers to range values, not RGB.

Paper Eq. (5) gives the accumulated range channel. The released renderer divides
by accumulated opacity for expected range and can combine this with median
range. The deck and lab label the distinction. Paper Eq. (17) appears to reverse
the intended scale-penalty sign; the released mapper implements a positive
excess-scale penalty, consistent with the prose.

Reconstruction scores in final Table 1 use ground-truth poses, 20 cm voxels and
a 20 cm F-score threshold. The deck uses the explicitly percent-labeled average
RPE column of supplementary Table B. Mapping-iteration frequency is not full
SLAM frame throughput. No full benchmark execution is claimed.

## Sources and figure credits

- [Final ICCV paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Giacomini_Splat-LOAM_Gaussian_Splatting_LiDAR_Odometry_and_Mapping_ICCV_2025_paper.pdf)
- [Final supplementary material](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Giacomini_Splat-LOAM_Gaussian_Splatting_ICCV_2025_supplemental.pdf)
- [User fork](https://github.com/BaiLiping/Splat-LOAM)
- [Authors' implementation](https://github.com/rvp-group/Splat-LOAM)

`figures/sources.json` records original figure numbers, proceedings pages and
source URLs. Figure 2 is the method overview; Figure 6 supplies the full
reconstruction comparison and a labeled cover crop; Figure 1 supplies the
performance overview. These are crops of source figures, not recreated results.
Existing Bento and bundled-dependency license notices remain intact.

