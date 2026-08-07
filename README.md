# bailiping.com

GitHub Pages site for `bailiping.com`.

## Structure

- `/` is the main website entry point.
- `/sales/` is the static English second-hand sale catalog.
- `/handover/` is the target-handover project page with paper, repository, and result animation links.
- `/vslam/` is an interactive step-by-step visual SLAM explainer (tracking, loop closure, pose graph optimization, bundle adjustment) with live solvers.
- `/bp-vs-pmbm/` is an interactive side-by-side comparison of belief propagation and PMBM data association for multi-target tracking.
- `/eo-mtt/` is an interactive note on partition uncertainty in extended-object multi-target tracking.
- `/gaussian-splatting/` is the original interactive Gaussian Splatting and GS-SLAM note.
- `/3dgs/` is an interactive explainer of the original 3D Gaussian Splatting paper and training loop.
- `/differentiable-ray-tracing/` is an interactive explanation of forward ray simulation, smooth path derivatives, visibility discontinuities, and optimization through a renderer.
- `/graph-slam/` is a Jupyter-notebook-style walkthrough of graph SLAM, incremental smoothing with iSAM/iSAM2, bundle adjustment, and structure from motion. Its numbered numerical cells were executed; the printed outputs and figures in `graph-slam/assets/` are reproducible results, while external-library cells are marked as reference snippets.
- `/cir-to-taps/` is an interactive communication-basics lab showing how continuous-delay channel paths become discrete complex channel taps, plus a status-labeled, field-level inventory of the radio-SLAM experiment data and its estimator boundary.

## Edit sale items

Update `sales/data/items.js`. The data is grouped by seller:

- `status`: `available`, `reserved`, or `sold`
- `images`: one or more image URLs
- `price`: display text, so currencies can be written exactly as needed

The Feishu link provided by the user is stored in `sourceDocument`, but it was not readable without Feishu login from this environment.
