# LiDAR Gaussian Splatting SLAM

A 28-slide native Bento reading of **Real-Time LiDAR Gaussian Splatting SLAM via Geometry-Aware Covariance Coupling**, with three embedded interactive laboratories and expanded reading notes.

- Presentation: <https://bailiping.com/lidar-gs-slam/>
- Reading notes: <https://bailiping.com/lidar-gs-slam/study.html>
- Standalone laboratories: <https://bailiping.com/lidar-gs-slam/live/>

## Sources and version

SeungJun Tak, Yewon Jeon, Jaeik Hwang, Suk Min Hwang, Seongbo Ha and Hyeonwoo Yu. Tak and Jeon contributed equally. ECCV 2026, pp. 620–637.

- Paper: <https://arxiv.org/html/2607.04127v1>
- Published version: <https://doi.org/10.1007/978-3-032-37595-7_34>
- Repository: <https://github.com/Lab-of-AI-and-Robotics/LiDAR-GS-SLAM>
- Inspected source commit: c51abf31e82a4bf628df2965702ab36177d089cd
- Project: <https://lab-of-ai-and-robotics.github.io/GS-SLAM-Family/>

The canonical slide text and speaker notes are in bento-deck.mjs. Each slide carries source references. Paper benchmarks are author-reported and were not reproduced for this presentation. Dataset, units, table context and hardware remain explicit.

## Build

From the site root:

    node lidar-gs-slam/build.mjs
    node --test lidar-gs-slam/tests/model.test.mjs

The builder validates slide bounds, IDs, links, figure paths and live-slide ordering. It reuses the unchanged Bento runtime and license notices from mpc-detection-to-bounce-count-slides/index.html, the site's existing MathJax asset, and scripts/authoring-build.mjs. It writes index.html, deck.json, live-demos.json and study.html.

The shared authoring helper preserves a future authoring.json sidecar. Read and retain that file before rebuilding if it exists. No build input depends on research files outside the committed website.

After staging new HTML pages, update site discovery:

    python3 scripts/build-search-index.py
    python3 scripts/build-search-index.py --check

## Laboratory scope

The demo runs locally in the browser with deterministic synthetic data.

1. **Covariance** estimates a 3×3 empirical covariance and its eigensystem, then illustrates an oriented surface disk and range-adaptive initialization.
2. **Tracking** performs real SE(2) Gauss–Newton updates with fixed correspondences. It uses the paper's normalized scale ratio and clipped reciprocal geometry weight.
3. **Budget** illustrates score-driven planar grouping and symmetric splitting with explicit toy thresholds, caps and a toy coverage metric.

These models do not reproduce the complete SLAM pipeline, CUDA spherical rasterization, the three-dimensional multi-pass management implementation or paper benchmark performance. The visible lab descriptions identify the simplifications.

## Interpretation details retained

- The 86.78% F-score and 22.15 system FPS headline is Newer College **Quad**, not the four-sequence mean.
- The four-sequence means are 69.51% F-score and 21.24 FPS.
- The map control score and the tracking scale-ratio weight are separate signals.
- The inspected release normalizes its tracking ratio with a fixed interval. The demo follows paper Eq. (3).
- Final map storage can fall while peak VRAM rises. The Quad full-management Table 4 result is 5.0 MB of map storage and 4.5 GB peak VRAM, versus 9.6 MB and 2.9 GB for opacity pruning alone.
- Table 1 and the ablation tables use distinct runs. Their Keble ATE values must not be mixed.
- Loop closure applies keyframe-wise rigid map corrections. It does not globally reoptimize every Gaussian.
- The inspected Newer College config enables splitting but disables plane management. Both flags are required to request the full map-management variant.

## Figure attribution

figures/paper-control-score.png reproduces **Figure 3** from the paper:

<https://arxiv.org/html/2607.04127v1/Control_score.png>

figures/paper-budget.png reproduces **Figure 5**:

<https://arxiv.org/html/2607.04127v1/roi8_groupcaps_toplabels_tnr_v47.png>

Both figures are credited to Tak et al. and reproduced unchanged under the paper's [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/). Scaling in the slide preserves the original aspect ratio. The remaining SVG diagrams are original conceptual teaching illustrations, not experimental data.

The embedded Bento runtime retains its existing MIT and third-party notices. Shared fonts and MathJax retain their existing license files.
