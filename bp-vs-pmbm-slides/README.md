# Belief-Propagation MTT

Public deck: <https://bailiping.com/bp-vs-pmbm-slides/>. Linked from handover page 2,
with a return link on every slide. The companion article stays under Extensions.

The September 2026 redesign follows the handover deck's white background, Arial
headings, navy text, and green accent. Seventeen regular Bento slides progress
from one-to-one assignment and local evidence to the factor graph, BP messages,
state updates, joint hypotheses, and the PMBM / TOMB/P connection.

Each of the three experiments has an immediately preceding static introduction:
weights (3–4), BP messages (6–7), and retained hypotheses (9–10). Live views replace
only the 1136 × 435 content region. Static screenshots remain visible in print and
if an iframe is unavailable. Page Up / Down navigate from controls; Escape restores
deck focus. Standalone experiments stack and scroll on small screens.

## Source and build

- `build-deck.mjs`: authored text, equations, speaker notes, native slide elements,
  references, and introduction/live mapping; preserves the checked-in Bento runtime.
- `figures.mjs`: deterministic SVG diagrams, with build-time MathJax labels.
- `live/app.js`: canonical control state and interaction handlers.
- `live/render.js`: SVG and table rendering, independent of numerical inference.
- `../bp-vs-pmbm/association-model.js`: unchanged shared numerical model.
- `assets/`: local MathJax 3.2.2 and license, diagrams, and static live fallbacks.

From the repository root:

```sh
npm ci --prefix bp-vs-pmbm-slides --ignore-scripts --no-audit --no-fund
node bp-vs-pmbm-slides/build-deck.mjs
node --test tests/bp-pmbm.test.cjs
python tests/bp-pmbm-browser.py
```

The browser check uses Python Playwright 1.57.0. `CHROMIUM_PATH` can select an
existing Chrome executable. It checks numerical parity, preset and parameter
changes, playback, exact-reference display, truncation endpoints, actual sandboxed
embeds, typeset mathematics, and layout. Additional lifecycle checks cover keyboard
handoff, inactive playback, overview, responsive layouts, print, and return links.

## Mathematical scope

The experiments use three certain existing point targets, Gaussian predicted
measurements, Poisson clutter, and no undetected PPP. Gates truncate pair weights
without redefining detection probability. Exact means summing all legal assignments
for those same weights. The default example has 22 positive-weight events; BP meets
its tolerance after 30 sweeps with a 6.9262 percentage-point maximum marginal gap.

The state-update mixture illustrates how association marginals enter tracking.
Existence and birth require further components. PMBM is a posterior family; BP is
an inference method. The PMBM evidence slide conditions on one predicted parent;
new existence is conditional on an unassigned detection. The dependence example
separates marginalization from an independent product approximation. TOMB/P retains
the undetected PPP while approximating the detected-target MBM by one MB.

All original mathematical audit tests are retained. The deck links the primary
Williams–Lau, García-Fernández et al., and Williams papers alongside relevant slides.
