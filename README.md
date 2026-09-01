# bailiping.com

GitHub Pages site for `bailiping.com`.

## Structure

- `/` is the main website entry point.
- `/sales/` is the static English second-hand sale catalog.
- `/handover/` is the target-handover project page with paper, repository, and result animation links.
- `/vslam/` is an interactive step-by-step visual SLAM explainer (tracking, loop closure, pose graph optimization, bundle adjustment) with live solvers.
- `/bp-vs-pmbm/` is an interactive side-by-side comparison of belief propagation and PMBM data association for multi-target tracking.
- `/eo-mtt/` is an interactive note on partition uncertainty in extended-object multi-target tracking.
- `/frame-registration-slides/` is the interactive Bento slide deck for frame-registration methods, with live RANSAC, ICP, and NDT labs.
- `/target-handover-slides/` is the interactive Bento slide deck for point-target handover, with live decision-rule and timeline labs.
- `/bp-vs-pmbm-slides/` is the interactive Bento slide deck for normalized data association, with live shared-weight, BP, and joint-hypothesis labs.
- `/eo-mtt-slides/` is the interactive Bento slide deck for extended-object partition uncertainty, with live candidate-partition, hypothesis-management, and inference labs.
- `/gaussian-splatting/` is the original interactive Gaussian Splatting and GS-SLAM note.
- `/3dgs/` is an interactive explainer of the original 3D Gaussian Splatting paper, followed by bridges to visual Gaussian-splatting SLAM and unknown-UE radio multipath optimization.
- `/splatting-graph-slam/` remains the deeper factor-graph walkthrough linked from the integrated 3DGS chapter.
- `/differentiable-ray-tracing/` is an interactive explanation of forward ray simulation, smooth path derivatives, visibility discontinuities, and optimization through a renderer.
- `/graph-slam/` is a Jupyter-notebook-style walkthrough of graph SLAM, incremental smoothing with iSAM/iSAM2, bundle adjustment, and structure from motion. Its numbered numerical cells were executed; the printed outputs and figures in `graph-slam/assets/` are reproducible results, while external-library cells are marked as reference snippets.
- `/cir-to-taps/` is an interactive communication-basics lab showing how continuous-delay channel paths become discrete complex channel taps, plus a status-labeled, field-level inventory of the radio-SLAM experiment data and its estimator boundary.
- `/mpc-detection-to-bounce-count/` is the interactive note for converting resolved MPC delay, AoA, AoD, and path-loss evidence into bounce-count hypotheses under three map/pose knowledge regimes.
- `/mpc-detection-to-bounce-count-slides/` is the 11-slide Bento presentation of that note, with one consolidated concept/live pair for each map/pose regime and printable static fallbacks.
- `/variational-inference/` is a 15-slide interactive Bento deck on ELBOs, mean-field inference, coordinate ascent, and stochastic gradients, with a deterministic step-by-step EM experiment.

## Edit sale items

Update `sales/data/items.js`. The data is grouped by seller:

- `status`: `available`, `reserved`, or `sold`
- `images`: one or more image URLs
- `price`: display text, so currencies can be written exactly as needed

The Feishu link provided by the user is stored in `sourceDocument`, but it was not readable without Feishu login from this environment.
