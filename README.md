# bailiping.com

GitHub Pages site for `bailiping.com`.

## Structure

- `/message-passing-tracking/` is an interactive Bento walkthrough of Figure 4 in Meyer et al. (2018), following prediction, association messages, and target existence updates. It is listed under Random thoughts.

- `/factor-graphs/` is an interactive Bento reading of *Factor Graphs and the Sum-Product Algorithm*, with the paper's five-step example, a message inspector, and a factor-sum workbench. It is listed under Random thoughts.

- `/` is the main website entry point.
- For topics with both slides and an article, the homepage lists the deck. Companion articles appear on each deck's
  final **Extensions** slide, with an **Extensions** shortcut throughout the deck.
  Article headers link back to their slides; site navigation uses the same tab.
- **Work** lists point-target handover, extended-target handover, extended-target tracking,
  density fusion, and joint-PDF derivations.
- **SLAM** includes the Visual SLAM and Graph SLAM notes as extensions.
- Companion mappings and appendix layouts live in `assets/deck-extensions.mjs`.
  Run `node scripts/install-deck-extensions.mjs` after editing a static deck.
  The BP vs PMBM builder and the Density Fusion source apply the helper directly.
- `/sales/` is the static English second-hand sale catalog.
- `/handover/` is the target-handover project page with paper, repository, and result animation links.
- `/vslam/` is an interactive step-by-step visual SLAM explainer (tracking, loop closure, pose graph optimization, bundle adjustment) with live solvers.
- `/bp-vs-pmbm/` is an interactive side-by-side comparison of belief propagation and PMBM data association for multi-target tracking.
- `/eo-mtt/` is an interactive note on partition uncertainty in extended-object multi-target tracking.
- `/frame-registration-slides/` is the interactive Bento slide deck for frame-registration methods, with live RANSAC, ICP, and NDT labs.
- `/target-handover-slides/` is the interactive Bento slide deck for point-target handover, with a decision-rule lab using the Figure 3 trajectories, links to the IEEE paper, and the project page as an extension.
- `/et-handover/` is the public Extended-Target Handover presentation, with 17 slides,
  two live demos, and links to its supporting presentations and GrBP derivation.
  Legacy `/eo-handover-slides/` URLs redirect here.
- `/eo-derivation/` and `/eo-derivation-slides/` contain the public joint-PDF and GrBP
  derivations, their 50-slide companion, source figures, and PDF. Links in the
  handover presentation and derivations navigate in the current tab.
- `/bp-vs-pmbm-slides/` is the interactive Bento slide deck for normalized data association, with live shared-weight, BP, and joint-hypothesis labs.
- `/eo-mtt-slides/` is the interactive Bento slide deck for extended-object partition uncertainty, with live candidate-partition, hypothesis-management, and inference labs.
- `/gaussian-splatting/` is the original interactive Gaussian Splatting and GS-SLAM note.
- `/3dgs/` is an interactive explainer of the original 3D Gaussian Splatting paper, followed by bridges to visual Gaussian-splatting SLAM and unknown-UE radio multipath optimization.
- `/splatting-graph-slam/` remains the deeper factor-graph walkthrough linked from the integrated 3DGS chapter.
- `/differentiable-ray-tracing/` is an interactive explanation of forward ray simulation, smooth path derivatives, visibility discontinuities, and optimization through a renderer.
- `/graph-slam/` is a Jupyter-notebook-style walkthrough of graph SLAM, incremental smoothing with iSAM/iSAM2, bundle adjustment, and structure from motion. Its numbered numerical cells were executed; the printed outputs and figures in `graph-slam/assets/` are reproducible results, while external-library cells are marked as reference snippets.
- `/cir-to-taps/` is an interactive communication-basics lab showing how continuous-delay channel paths become discrete complex channel taps, plus a status-labeled, field-level inventory of the radio-SLAM experiment data and its estimator boundary.
- `/variational-inference/` is a 15-slide interactive Bento deck on ELBOs, mean-field inference, coordinate ascent, and stochastic gradients, with a deterministic step-by-step EM experiment.
- `/advanced-state-representations/` is a 24-slide interactive Bento deck on manifold-valued states, Lie-group optimization, splines, and sparse continuous-time Gaussian processes, with three deterministic live labs.

## Edit sale items

Update `sales/data/items.js`. The data is grouped by seller:

- `status`: `available`, `reserved`, or `sold`
- `images`: one or more image URLs
- `price`: display text, so currencies can be written exactly as needed

The Feishu link provided by the user is stored in `sourceDocument`, but it was not readable without Feishu login from this environment.

- `/radar-slam/` is a 33-slide Bento presentation with seven inline labs covering range, CFAR, coordinate frames, Doppler velocity, ICP, and pose-graph correction. Build and QA commands are in `radar-slam/README.md`.
