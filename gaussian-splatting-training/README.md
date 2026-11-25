# How Pixels Teach Gaussian Splats

39 native Bento slides, three computed browser laboratories, and expanded reading notes. The lesson starts with the original 2023 3DGS objective and continues into joint camera-bearing UE pose and Gaussian-splat estimation, comparing the online MonoGS objectives and tracing both gradient paths.

## Build

From the repository root:

```sh
node gaussian-splatting-training/build-static-figures.mjs
node gaussian-splatting-training/build-figures.mjs
node gaussian-splatting-training/build.mjs
node --test gaussian-splatting-training/tests/*.test.mjs
```

The builder preserves the established Bento runtime from `mpc-detection-to-bounce-count-slides/index.html`, its dependency license notices, shared theme and authoring support. Edit `bento-deck.mjs`, `joint-section.mjs` and the lab sources. Existing `authoring.json` edits, when present, pass through `withAuthoring`; do not discard them. The build produces `index.html`, `deck.json`, `live-demos.json` and `study.html`. All diagrams are original explanatory figures. The training-preview and scalar-gradient plots use computed numerical results.

After staging new HTML, regenerate and check search discovery:

```sh
python3 scripts/build-search-index.py
python3 scripts/build-search-index.py --check
```

## Joint pose and splat section

Slides 22–35 are an integrated continuation after the multi-view lab:

1. Fixed camera poses versus trainable camera/UE poses, with the same renderer and observations.
2. Original L1 + SSIM versus MonoGS online RGB L1 + isotropic shape regularization. Making pose trainable does not require dropping SSIM.
3. Optional measured-depth residuals versus shape priors; units, masks and reductions determine the meaning of the weights.
4. The shared image residual and its separate scene and pose Jacobians.
5. A computed one-pixel joint update with opposite camera-position and splat-position derivatives.
6. Translation-first, left SE(3) increments, including both projected-center and covariance derivative paths.
7. Local differentiability, L1 subgradients and discrete visibility/density/keyframe decisions.
8. Tracking-only, map-only, simultaneous joint and alternating update schedules.
9. Coordinate gauge, monocular scale and the distinction between a gradient and identifiability.

The UE notation assumes a rigid camera with known extrinsic calibration: T_AB maps B into A and T_CW = T_CU T_UW. Radio-only UE measurements would require a different forward model and likelihood. The two original labs remain fixed-camera teaching models. The third lab computes actual joint updates using L1 + SSIM. The scalar joint example uses squared error for transparent arithmetic and is explicitly distinguished from MonoGS online L1. With x=1, sigma=1, mu=t=0 and observed intensity 0.8, the initial loss is 0.0187151928 and the two gradients are -0.1173450866 and +0.1173450866. A simultaneous step of 0.1 lowers loss to 0.0160630031. Central differences verify both derivatives.

The standalone joint deck and its earlier additive-MSE lab have been removed. The new joint lab is integrated here and uses the same L1 + SSIM convention as the fixed-camera labs. The training deck is the single entry point; stable section links include `#joint-start`, `#joint-loss` and `#joint-differentiability`.

## Laboratories

### `live/?lab=training`

Five anisotropic 2D Gaussians render a 32×32 RGB image through front-to-back alpha compositing, including the background contribution. The objective is mean RGB L1 plus `lambda * (1 - SSIM)`, with the L1 term weighted by `1 - lambda`. Local SSIM uses an 11×11 Gaussian window, standard deviation 1.5, zero padding, and constants 0.01² and 0.03². There is no additional factor of one-half.

Analytic gradients and Adam update centers, log scales, rotation, opacity logits and constant RGB logits. Each parameter group can be frozen. The target image is fixed, and the optimizer never reads generating splat parameters. The pixel inspector displays exact alpha, transmittance, compositing weight, RGB contribution, background and the full-image loss gradient at a selected pixel. Loss-history phases restart when lambda or trainable groups change.

This is an explicit teaching reduction: fixed depth order, full untruncated footprints, constant RGB, numerical stability bounds and a decaying group-specific learning-rate schedule. It does not implement 3D perspective projection, spherical harmonics, CUDA tiling/culling, adaptive density control or original benchmark experiments.

### `live/?lab=multiview`

One isotropic Gaussian has an unknown 3D center and known shape, opacity and color. Camera 1 observes x/y, camera 2 observes z/y, and an evaluation camera observes a 45-degree mixture of x and z. Calibrated orthographic projections and the same image objective isolate the missing depth information of one camera. One-view training cannot change z. Adding camera 2 supplies an image-loss gradient in that direction. A separate held-out view never contributes to training.

The displayed reference center is evaluation-only. Tests verify that changing it does not change the optimization. The model is noise-free and deliberately constrained; success does not establish unique reconstruction for arbitrary scenes. The first two lab routes accept `&embed=1` and participate in the shared Bento pause, ready and keyboard-navigation protocol.

### `live/joint.html`

Four anisotropic 2D Gaussian centers and one camera's SE(2) transform are estimated from two fixed 24×24 RGB observations. Camera 0 is anchored at identity. Shapes, RGB, opacity, layer order and orthographic scale remain known. The forward transform is `m = 0.5 + R(theta)(mu - 0.5) + t`; covariance rotates with the camera. This is a controlled 2D teaching reduction, not full SE(3) SLAM or evidence of metric-scale recovery from arbitrary monocular imagery.

The objective averages `(1-lambda)*mean_RGB_L1 + lambda*(1-SSIM)` over both images. The default lambda is 0.2. Both analytic gradient blocks include both loss terms. No true pose or generating center list is retained in optimizer state. The photo is known even if its true pose is not: compare its fixed RGB with the render at the current estimate on the same pixel grid.

Controls select joint, pose-only or splat-only updates, simultaneous or alternating schedules, and lambda from 0 (L1 alone) to 1 (SSIM alone). Backtracking gradient descent accepts image-loss reductions. The full view includes a world-map/camera plot, actual loss history, component arithmetic, gradient norms and a keyboard-accessible pixel inspector. Reset keeps the selected controls. History starts a fresh phase when controls change. `?embed=1` provides the compact third inline lab with the shared pause/ready/keyboard protocol.

Stable new lesson links: `#joint-observation`, `#joint-l1-pixel`, `#joint-image-loss`, `#joint-setup`, `#joint-live`. The original `#joint-loss` distinction remains unchanged.

## Verification

The 18-test numerical suite checks alpha/background compositing, unnormalized footprint behavior, the loss convention, an independent direct local-SSIM implementation, all 45 analytic parameter derivatives against central differences, actual convergence, parameter freezing, camera derivatives, one-view depth ambiguity, held-out recovery and independence from displayed reference geometry.

New joint checks verify all 11 center/pose derivatives with central differences at lambda 0, 0.2 and 1, including the anisotropic covariance rotation path. They check both schedules, exact frozen blocks, anchored camera behavior, fixed observations, independence from fabricated truth fields and loss-history phases. At default settings, joint loss is 0.127435 initially and 0.00001997 after 150 simultaneous steps.

Observed deterministic fixed-camera defaults: image loss 0.140337 initially, 0.001265 after 100 steps, and 0.000318 after 200. A fresh single-camera run reaches low training loss while z remains 0.27 instead of the generating 0.70. Adding the second camera corrects z and improves the held-out image. These are outputs of this teaching model, not measurements from the original paper.

## Primary references

- [Matsuki et al., Gaussian Splatting SLAM, CVPR 2024, with supplement](https://arxiv.org/html/2312.06741v2)
- [MonoGS online losses](https://github.com/muskie82/MonoGS/blob/main/utils/slam_utils.py)
- [MonoGS mapping and separate color refinement](https://github.com/muskie82/MonoGS/blob/main/utils/slam_backend.py)
- [MonoGS pose updates](https://github.com/muskie82/MonoGS/blob/main/utils/pose_utils.py)
- [Pose-aware rasterizer](https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose)

- [Kerbl et al., original 2023 paper](https://arxiv.org/html/2308.04079v1)
- [Authors’ project](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Official training loop](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/train.py)
- [Official L1 and SSIM](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/loss_utils.py)
- [Gaussian parameterization and density control](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py)
- [Optimization defaults](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/arguments/__init__.py)
- [Forward rasterizer](https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu)
- [Backward rasterizer](https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/backward.cu)

The current official repository includes later optional additions. This lesson separates those from the original photometric objective. The independent educational model does not copy the CUDA implementation.

