# Radar SLAM — Bento presentation

`index.html` is generated from `bento-deck.mjs` using the existing site's Bento
runtime in `../kalman-filter-derivations/index.html`. Compressed runtime payloads
are preserved verbatim. Do not edit generated slide JSON by hand.

```sh
node radar-slam/build-bento.mjs
node --test radar-slam/test-model.cjs radar-slam/test-controls.cjs
python3 -m http.server 8000
```

Open `/radar-slam/`. Stable topic hashes (for example `#velocity`) lead to the
introduction; the immediately following regular slide (`#velocity-live`) mounts
the experiment automatically. There are 33 slides, no hidden state slides, and
seven introduction/experiment pairs. Bento owns navigation, overview and notes.

- `model.js`: pure deterministic signal, registration and graph models.
- `live/labs.js`: controls, labels and experiment presets.
- `live/app.js`: independent controllers, derived metrics and animation lifecycle.
- `live/render.js`: canvas geometry and plots.
- `live/frame.js`: keyboard/focus bridge and combined host/document visibility.
- `fallback.mjs`: vector initial states computed from the same numerical models.
- `print.js`: makes the browser Print command use Bento's native print-page
  structure, preserving typeset equations and removing live frames.

Direct lab URLs use `/radar-slam/live/?demo=velocity` (also `whole-run`, `range`,
`cfar`, `frames`, `icp`, `optimize`). Narrow screens use a scrolling layout.
No autoplay occurs. Hidden slides pause, and returning resumes user-started
animation from its existing state. Page Up/Down navigate from controls; Escape
returns focus to Bento. Input controls retain arrow keys and Space.

The graph lab compares before/after poses and stored scans, shows objective
history and true position RMSE separately, and exports the current run as JSON.
Truth is used only for synthetic observation generation and evaluation. The
loop candidate is supplied, not retrieved automatically. These are teaching
models, not benchmark reproductions or a complete raw-I/Q-to-SLAM pipeline.

Math uses the site's pinned MathJax 3.2.2 SVG renderer. Live numerical labs have
no network dependency. The static fallback is shown under every live frame and
in print; a text/figure reading view is also included for disabled JavaScript.

QA covers deterministic data, coordinate conventions, outlier robustness,
observable geometry, registration, monotone graph steps, actual control handlers,
resets, export, animation pause/resume and bounds/pairing audits. Browser review
also checks native navigation, presentation layout, overview, and the 33-page
print preview with rendered formulas and static lab graphics.
