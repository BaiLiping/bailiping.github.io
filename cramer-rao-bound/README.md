# Cramér–Rao Bound — Precision Has a Floor

A native 23-slide Bento presentation with three compact interactive labs, a complete reading guide, and presenter notes. Public route: https://bailiping.com/cramer-rao-bound/ . This introductory deck is separate from the existing `/radio-cramer-rao-slides/` presentation.

## Open and present

The public route uses the same Bento runtime, typography, slide canvas, and
navigation as the site's technical presentations. Use arrow keys to advance;
Escape opens Bento's overview. Existing semantic links such as `#geometry-lab`
continue to select their corresponding slide.

Normal slides contain the full explanation and static vector figures. **TRY
LIVE** opens only the selected compact experiment in a modal. Back, Escape,
and dialog close share one cleanup path: the iframe is removed and keyboard
focus returns to the originating control. Labs load only when requested.

The [interactive guide](./guide/) provides the complete original explanations,
references, and working experiments in a responsive reading view. The
[standalone labs](./live/) contain Gaussian sampling, shrinkage bias, and sensor
geometry. Both link back to the presentation in the same tab.

The deck, guide, and labs use local code and embedded SVG equations, without
remote fonts or runtime libraries. The source package works from the filesystem,
including its compact labs. PDF and PowerPoint are static snapshots; the
PowerPoint includes presenter notes. The HTML is the interactive master.

## Published downloads

- [PDF](./Cramer-Rao-Bound.pdf)
- [PowerPoint](./Cramer-Rao-Bound.pptx)
- [Presenter notes](./presenter-notes.md)
- [Source package](./Cramer-Rao-Bound-site-package.zip)

## Build and test

From the repository root:

```sh
python3 cramer-rao-bound/build.py
node cramer-rao-bound/tests/numerics.cjs
node et-handover/validate.mjs cramer-rao-bound/index.html
```

`build.py` regenerates the guide and compact labs, then invokes `build.mjs` to
produce the native Bento document and an independent static print layout.
`deck.mjs` authors native text, shapes, vector charts, equations, and links.
The checked-in Bento runtime matches the site's technical decks and is preserved
verbatim in `index.html`; the standalone package can rebuild from that shell.

Install optional figure, browser-QA, and export tools:

```sh
npm install --prefix cramer-rao-bound --ignore-scripts
npx --prefix cramer-rao-bound playwright install chromium
```

After editing equations or plotted models, regenerate their cached assets:

```sh
node cramer-rao-bound/tools/render-formulas.cjs
python3 cramer-rao-bound/build.py --guide-only
node cramer-rao-bound/tools/capture-figures.cjs
python3 cramer-rao-bound/build.py
```

The 11 static vector figures come from the same calculations as the live labs;
`assets/lab-states.json` records their default numerical state.

Run a local HTTP server from the repository root on port 8772, then:

```sh
node cramer-rao-bound/tests/bento.cjs
```

`CRB_BASE_URL` overrides the served route. `CHROMIUM_EXECUTABLE` optionally selects
a system browser; `PLAYWRIGHT_MODULE` optionally selects a local Node Playwright
installation. Integration checks cover all 23 slides, image/formula loading,
desktop/mobile overflow, numerical controls, pointer/keyboard geometry changes,
lazy loading, Back/Escape, focus return, teardown during delayed loading, legacy
hashes, and offline file access. The numerical suite contains 348 assertions.
The earlier full-guide regressions remain in `tests/browser.py` (Python Playwright).

Export the actual Bento slides and static print layout:

```sh
node cramer-rao-bound/tools/export.cjs
python -m pip install python-pptx==1.0.2
python cramer-rao-bound/tools/export.py
```

The Python exporter invokes the Node exporter and adds a PowerPoint with notes.
Exports go to the ignored `exports/` directory. Inspect the real 23-page PDF,
then copy the PDF and PPTX into this directory before publication. Run
`python3 cramer-rao-bound/tools/package.py` to rebuild the source ZIP without
dependencies, caches, temporary exports, or the ZIP itself.

After staging new public routes, refresh discovery:

```sh
git add cramer-rao-bound/index.html cramer-rao-bound/guide/index.html cramer-rao-bound/live/index.html
python3 scripts/build-search-index.py
python3 scripts/build-search-index.py --check
```

## Source layout

- `deck.mjs`, `build.mjs`, `deck.js`, and `deck.css`: native Bento content,
  static print generation, accessible lab lifecycle, and presentation styling.
- `src/deck.json`: the full guide text, primary references, and presenter notes.
- `src/formulas.json` and `src/formula-svg.json`: editable TeX and cached SVG.
- `src/math.js`: unchanged, independently tested numerical models.
- `src/app.js`, `src/style.css`, `src/guide.css`, and `src/lab.css`: interactive
  guide and compact lab rendering.
- `assets/`: generated vector figures and their recorded default lab state.

## Statistical interpretation

The scalar bound is pointwise and applies to regular models and locally unbiased estimators. The Gaussian lab simulates complete, independent datasets; empirical variance uses the sample-variance divisor and can fluctuate below the theoretical floor. The analytic reference curve is the efficient Gaussian sample-mean distribution, not a distribution implied by every CRB.

The bias lab holds the shrinkage factor fixed. Its unbiased CRB is not a universal MSE lower bound. The biased variance bound uses the derivative of the estimator's bias and is displayed separately.

The geometry lab assumes independent additive Gaussian range noise with known, position-independent variance and known anchors. It evaluates local information at the true target, not at an estimated solution. Its ellipse has unit Mahalanobis radius and is not a confidence region. Rank-deficient information produces no finite ordinary full-state bound; no pseudoinverse is disguised as one. Target-anchor distances below 0.08 m are flagged to avoid the undefined derivative at coincidence. Unknown common offsets are eliminated with the exact Schur complement. Distant reflection ambiguities are not excluded by finite local information.

## References

- Stanford STATS 200, Lecture 15: https://web.stanford.edu/class/archive/stats/stats200/stats200.1172/Lecture15.pdf
- Ly et al. (2017), *A Tutorial on Fisher Information*: https://arxiv.org/abs/1705.01064
- Shen and Win (2010), *Fundamental Limits of Wideband Localization—Part I: A General Framework*: https://arxiv.org/abs/1006.0888
- Stanford STATS 200, Lecture 14: https://web.stanford.edu/class/archive/stats/stats200/stats200.1172/Lecture14.pdf

The independent Gaussian-range lab is a simplified teaching model, not a reproduction of the waveform-level localization paper. Bias, uniform-maximum, and linear-Gaussian examples are independently derived, with explanations in the presenter notes.
