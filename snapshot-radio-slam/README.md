# Snapshot Radio SLAM: Initial Pose and Clock

Public educational companion to [Shen et al., arXiv:2607.04847v2](https://arxiv.org/abs/2607.04847v2), **Amplitude-Independent Robust Snapshot 6-D Radio SLAM via a Unified Angle-Delay Formulation**.

The native Bento deck contains 25 focused slides with equation-specific references, detailed presenter notes and three embedded laboratories. It explains the physical measurements, shared receiver clock, scatterer elimination, conditional linear system, singular value decomposition, orientation search, the analytic Gauss–Newton rotation increment, exhaustive subset consensus, physical refinement and QAIC model selection.

## Build

```sh
node snapshot-radio-slam/build.mjs
node snapshot-radio-slam/tests/math.test.mjs
```

Run from the site repository root. The build writes `index.html`, `deck.json`, and `live-demos.json`. It validates slide boundaries, unique identifiers, internal links, panel font sizes, lab placement and the required preceding lab introductions. The source of the slide content is `bento-deck.mjs`.

The build reuses the existing public radio deck's native Bento runtime and its shared MathJax file. It preserves that runtime's licensing and normal navigation, presenter and live-iframe behavior. No new CDN dependency is introduced by the slide build.

## Public routes

- Deck: `/snapshot-radio-slam/`
- Standalone laboratories: `/snapshot-radio-slam/live/`
- Geometry: `/snapshot-radio-slam/live/?lab=geometry`
- Matrix and SVD: `/snapshot-radio-slam/live/?lab=linear`
- Orientation: `/snapshot-radio-slam/live/?lab=orientation`

Embedded labs add `&embed=1`. All three iframe regions occupy the Bento coordinates `x=72, y=180, width=1136, height=475`.

## Mathematical conventions

Angles use radians within the mathematical model. Rotation matrices map local vectors into the global frame. AoA is the direction from the receiver toward the previous interaction or BS. With `t = p_UE - p_BS`, `rho = c tau`, `B = c b`, and `w = v - u`, the eliminated residual is

```text
r = M t - (rho - B) w
M = v u^T + u v^T - (1 + u^T v) I
```

The paper stacks the signed clock coordinate `y = [t; -B]` and uses `A_i = [M_i, -w_i]`. The browser uses the exactly equivalent positive clock coordinate `q = [t; B]` with `A_i = [M_i, +w_i]`. Both share the right-hand side `rho_i w_i`. The slides explicitly distinguish these conventions.

The numerical example on slide 12 comes from the browser's noiseless scene with BS `[0,0,2]` m, UE `[6,4,3.5]` m, clock `10 ns`, six single-bounce paths and LoS. It explicitly supplies the correct orientation as a **conditional algebra check**, not as an initializer for unknown-pose estimation. The true orientation convention is `Rz(30°) Ry(12°) Rx(-8°)`. The linear unknown is displacement `[6,4,1.5]` m and distance-clock `2.99792458 m`.

## Scope

The labs expose three parts of the model: geometry, the conditional matrix/SVD solve and orientation optimization. They do **not** execute the complete paper's exhaustive minimum-subset consensus or final IRLS/QAIC model selection. Those stages are explained in the deck and are part of the separate Python replication.

The full paper estimates LoS and single-bounce inliers and rejects incompatible multi-bounce measurements. It does not reconstruct double-bounce paths. The notes distinguish conditional rank from full-state identifiability, the formulation-specific four-path minimum from other solvers' requirements, and local convergence from global correctness.

Numerical implementation checks and browser interaction/rendering checks should be run after changing either the mathematical engine or the live UI. The deck build checks structure rather than visual overflow or algorithmic correctness.

## Numerical engine

`live/math.mjs` is a dependency-free ES module. Its one-sided Jacobi SVD returns singular values and each projection/reconstruction mode. The conditional solver withholds a unique state when rank is below four. `live/scene3d.mjs` renders a z-up perspective scene with orbit, wheel/pinch zoom, keyboard controls, and clipped off-screen estimates. `live/app.mjs` connects the controls to these calculations.

The default candidate orientation is `(15°,0°,0°)`, independently of the true `(30°,12°,-8°)` used to generate the snapshot. The explicit **True orientation (test)** button creates a labeled conditional test. The noise slider scales standard deviations, not the covariance multiplier gamma in the paper. Noise is reproducible by path ID. The optional double-bounce path is a physical three-segment example, rather than the paper’s synthetic random-angle outlier distribution.

The Node checks cover known SVD systems, exact conditional recovery, a shared delay shift, rank deficiency, outlier exclusion, deterministic noise, finite-difference agreement for the orientation Jacobian, and local orientation recovery.
