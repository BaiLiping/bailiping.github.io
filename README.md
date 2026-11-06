# bailiping.com

GitHub Pages site for `bailiping.com`.

## Homepage

The homepage has one horizontal row of category tabs. Work is selected by
default; selecting a tab shows only that category's pages. URLs such as
`/#life` and `/#random-thoughts` restore the selected tab on reload, and browser
Back/Forward restores earlier selections. Arrow keys, Home, and End navigate
the tabs. On narrow screens the tab row scrolls horizontally. Without JavaScript,
the category links jump to the corresponding sections and all pages stay visible.

Work uses research rows with an existing project figure, title, author details,
and links to public materials. Edit the entries in `index.html`; the homepage
styles and tab behavior are in `assets/home.css` and `assets/home.js`.
The small WebP preview in `assets/home/` comes from the existing public handover
figure. The nuScenes preview uses the project's existing looping crossing GIF
at 2× speed. The full figures and animation remain available on their project pages.

## Edit presentations visually

Open **Bailiping Editor.command** on the Mac desktop, or run
`scripts/start-authoring.command`. Choose a presentation, edit it in Bento, and
use **Save changes** (Command-S). Visual edits are kept in each deck's
`authoring.json` and survive generator rebuilds. Use **Copy feedback** to bring
comments back to Codex for review and publication. See
[the authoring guide](authoring/README.md) for preview, backup, and recovery.

## Search discovery

`robots.txt` permits crawling and advertises the site-wide `sitemap.xml`.
The homepage links to `/site-map/`, a static directory of all public pages with
ordinary same-tab links, including companion articles, demos, and study notes.

After adding, removing, renaming, or changing the title or description of a page,
stage new HTML files with Git and regenerate both discovery files:

```sh
python3 scripts/build-search-index.py
python3 scripts/build-search-index.py --check
```

The generator uses tracked HTML in this public repository. It excludes redirects,
HTML fragments, verification files, support directories, pages marked `noindex`,
and aliases whose canonical URL points elsewhere. It does not read private
repositories. The Search discovery workflow checks that the generated files
remain current when public HTML changes.

The Google verification file at `/google1add1f3ff1d21ef7.html` must remain unchanged.
After verifying `https://bailiping.com/` in Google Search Console, submit
`https://bailiping.com/sitemap.xml` in **Sitemaps**. Submission helps discovery;
Google decides which pages to index and when.

## Career

The homepage **Career** tab opens `/career/`. Its **Potential Jobs** page at
`/career/potential-jobs/` collects jobs and internships in radio localization,
radar, robotics, SLAM, and integrated sensing and communications. The original
Qualcomm reference and NVIDIA Aerial ISAC internship remain saved.

Swedish opportunities are included through selected official employer feeds.
The **Sweden** location filter is available alongside **Europe**, which continues
to include Swedish roles. The watch is civilian-only: military, defence, and
explicitly dual-use roles are excluded.

The **Update potential jobs** GitHub Actions workflow refreshes configured
official employer feeds on Mondays around **09:17 Europe/Stockholm** and publishes
the generated page. It can also be run from the Actions tab. Each listing shows
its check status; failed requests retain previous results. Search and filters
run in the browser, while the listings themselves are static HTML.

See [the job watch documentation](career/README.md) for sources, local commands,
status rules, and maintenance. The homepage also links directly to Potential
Jobs under the Career heading.

## Structure

- `/gaussian-splatting-training/` is a 34-slide Bento lesson on the original 3DGS photometric loss, alpha-compositing gradients, continuous optimization, adaptive density control and joint pose–splat estimation. The joint section compares the original photometric objective with MonoGS tracking and mapping losses, traces gradients into pose and splat parameters, and explains differentiability limits and gauge freedoms. Two computed labs fit anisotropic 2D splats and resolve a single-camera depth ambiguity. Expanded reading notes and a per-pixel contribution inspector explain the equations. It is listed under Random thoughts and complements the existing `/3dgs/` overview. Build commands and numerical checks are in `gaussian-splatting-training/README.md`.

- `/latent-field-tracking/` is a paper walkthrough of Wei, Rao and Meyer, arXiv:2610.38898v1, with embedded complex-array likelihood and online power-field learning demos. It is listed under Random thoughts. The browser examples are simplified teaching models, distinct from the paper's tracker and benchmark experiments. Build instructions and numerical checks are in `latent-field-tracking/README.md`.

- `/engine-lab/` compares how exhaust drives turbines in jet and turbocharged car engines, with animated cutaways, a four-stroke walkthrough, and gas-flow and shaft-power diagrams. It is listed under Random thoughts.

- `/cramer-rao-bound/` is a 23-slide offline-capable presentation on the Cramér–Rao Bound, with live Gaussian-sampling, bias–variance, and localization-geometry labs, presenter notes, PDF and PowerPoint downloads. It is listed under Random thoughts and is separate from the radio-measurements CRB deck. Build and test commands are in `cramer-rao-bound/README.md`.


- `/snapshot-radio-slam/` is a 25-slide Bento reading of Shen et al., arXiv:2607.04847v2, with three embedded views of a 3D equation lab. It explains the conditional position/clock pseudoinverse, nonlinear orientation search, robust consensus, physical refinement, and QAIC. The live lab exposes its measured snapshot, actual stacked matrix, SVD modes, and local SO(3) updates. It is listed under Random thoughts.

- `/message-passing-tracking/` is an interactive Bento walkthrough of Figure 4 in Meyer et al. (2018), following prediction, association messages, and target existence updates. It is listed under Random thoughts.

- `/factor-graphs/` is an interactive Bento reading of *Factor Graphs and the Sum-Product Algorithm*, with the paper's five-step example, a message inspector, and a factor-sum workbench. It is listed under Random thoughts.

- `/` is the main website entry point.
- For topics with both slides and an article, the homepage lists the deck. Companion articles appear on each deck's
  final **Extensions** slide, with an **Extensions** shortcut throughout the deck.
  Article headers link back to their slides; site navigation uses the same tab.
- **Work** lists Extended-Target Handover. Its supporting pages—point-target handover,
  extended-target tracking, density fusion, BP vs PMBM, and the joint-PDF derivations—
  remain public and are reached through the handover presentation's links. They do
  not have separate homepage entries.
- **SLAM** includes the Visual SLAM and Graph SLAM notes as extensions.
- Every Bento deck links `assets/deck-theme.css` (shared fonts, panel shadow, presentation frame) through `withAuthoring`; see [the authoring guide](authoring/README.md).
- Companion mappings and appendix layouts live in `assets/deck-extensions.mjs`.
  Run `node scripts/install-deck-extensions.mjs` after editing a static deck.
  The BP vs PMBM builder and the Density Fusion source apply the helper directly.
- `/sales/` is the static English second-hand sale catalog.
- `/books/` is **Books & Podcast** under **Life**, with books and a section for podcast episodes and reflections. Keep the supplied order and leave missing summaries blank; cover sources are recorded in `books/README.md`.
- `/handover/` is the target-handover project page with paper, repository, and result animation links.
- `/vslam/` is an interactive step-by-step visual SLAM explainer (tracking, loop closure, pose graph optimization, bundle adjustment) with live solvers.
- `/bp-vs-pmbm/` is an interactive side-by-side comparison of belief propagation and PMBM data association for multi-target tracking.
- `/eo-mtt/` is an interactive note on partition uncertainty in extended-object multi-target tracking.
- `/frame-registration-slides/` is the 21-slide interactive Bento deck for rigid frame registration, with live RANSAC, ICP, and NDT labs. It is generated by `frame-registration-slides/build-bento.mjs` in the shared slide system used by `/radar-slam/`.
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
- `/splat-loam/` is an interactive Bento walkthrough of Giacomini et al., *Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping* (ICCV 2025), with spherical-rendering and registration laboratories. It is listed under Random thoughts. The synthetic browser experiments explain selected mechanisms and do not reproduce the authors' CUDA implementation or benchmark results. Build instructions and numerical checks are in `splat-loam/README.md`.
- `/lidar-gs-slam/` is a 28-slide paper walkthrough of *Real-Time LiDAR Gaussian Splatting SLAM via Geometry-Aware Covariance Coupling* (ECCV 2026), with live covariance, registration, and map-management labs, expanded reading notes, and source-specific benchmark comparisons. The browser labs use synthetic teaching models. Build and numerical verification instructions are in `lidar-gs-slam/README.md`.
- `/3dgs/` is an interactive explainer of the original 3D Gaussian Splatting paper, followed by bridges to visual Gaussian-splatting SLAM and unknown-UE radio multipath optimization.
- `/splatting-graph-slam/` remains the deeper factor-graph walkthrough linked from the integrated 3DGS chapter.
- `/differentiable-ray-tracing/` is an interactive explanation of forward ray simulation, smooth path derivatives, visibility discontinuities, and optimization through a renderer.
- `/graph-slam/` is a Jupyter-notebook-style walkthrough of graph SLAM, incremental smoothing with iSAM/iSAM2, bundle adjustment, and structure from motion. Its numbered numerical cells were executed; the printed outputs and figures in `graph-slam/assets/` are reproducible results, while external-library cells are marked as reference snippets.
- `/cir-to-taps/` is an interactive communication-basics lab showing how continuous-delay channel paths become discrete complex channel taps, plus a status-labeled, field-level inventory of the radio-SLAM experiment data and its estimator boundary.
- `/variational-inference/` is a 15-slide interactive Bento deck on ELBOs, mean-field inference, coordinate ascent, and stochastic gradients, with a deterministic step-by-step EM experiment.
- `/advanced-state-representations/` is a 24-slide interactive Bento deck on manifold-valued states, Lie-group optimization, splines, and sparse continuous-time Gaussian processes, with three deterministic live labs.
- `/sampling-playground/` is a 27-slide interactive Bento deck on Monte Carlo sampling (Gibbs, Metropolis–Hastings, HMC, slice, rejection, importance sampling, and a particle filter checked against the exact Kalman filter), with seven seeded live labs. It is listed under Random thoughts and generated by `sampling-playground/build-bento.mjs` in the shared slide system used by `/radar-slam/`; build and QA commands are in `sampling-playground/README.md`.

## Edit sale items

Update `sales/data/items.js`. The data is grouped by seller:

- `status`: `available`, `reserved`, or `sold`
- `images`: one or more image URLs
- `price`: display text, so currencies can be written exactly as needed

The Feishu link provided by the user is stored in `sourceDocument`, but it was not readable without Feishu login from this environment.

- `/radar-slam/` is a 33-slide Bento presentation with seven inline labs covering range, CFAR, coordinate frames, Doppler velocity, ICP, and pose-graph correction. Build and QA commands are in `radar-slam/README.md`.
