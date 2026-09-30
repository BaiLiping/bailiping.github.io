# Cramér–Rao Bound — Precision Has a Floor

An offline-capable, 23-slide presentation with three interactive labs and presenter notes. Public route: https://bailiping.com/cramer-rao-bound/ . This introductory deck is separate from the existing `/radio-cramer-rao-slides/` presentation.

## Open and present

Open `index.html` in a browser, including directly from the filesystem. No runtime libraries, external fonts, analytics, or network requests are required. Equations are embedded SVG. Links to references and related site pages navigate in the same tab.

Use the arrow keys, Home/End, or the navigation buttons to change slides. O opens the overview, N toggles presenter notes, and F enters fullscreen. Read view exposes the whole deck and is the default on phones. Print produces static slides.

Direct lab routes:
- `#gaussian-lab`: independent Gaussian datasets, the efficient sample mean versus discarding measurements, a seeded Monte Carlo histogram, exact variance, and the unbiased CRB.
- `#bias-lab`: fixed-factor shrinkage toward zero, with exact variance, squared bias, and MSE. At zero shrinkage factor the distribution is correctly represented as a point mass.
- `#geometry-lab`: draggable anchors and target, information eigenvalues, a local covariance-bound ellipse, and optional Schur elimination of an unknown common range offset. Coordinate sliders and keyboard controls provide alternatives to dragging.

The PDF and PowerPoint are static snapshots. The PowerPoint includes presenter notes; its slide controls are images, not interactive controls. The HTML is the interactive master.

## Published downloads

- [PDF](./Cramer-Rao-Bound.pdf)
- [PowerPoint](./Cramer-Rao-Bound.pptx)
- [Presenter notes](./presenter-notes.md)
- [Source package](./Cramer-Rao-Bound-site-package.zip)

## Build and test

From the repository root, rebuilding the cached, self-contained HTML requires only Python:

```sh
python3 cramer-rao-bound/build.py
node cramer-rao-bound/tests/numerics.cjs
```

After editing `src/formulas.json`, regenerate the SVG cache first:

```sh
npm install --prefix cramer-rao-bound --ignore-scripts
node cramer-rao-bound/tools/render-formulas.cjs
python3 cramer-rao-bound/build.py
```

The optional build dependency is pinned to `mathjax-full` 3.2.2. It is not loaded by the published page. Do not commit `node_modules` or any font files.

Browser regression tests exercise all 23 slides, layout overflow, seeded resets, controls, singular and coincidence cases, dragging, keyboard navigation, notes, and phone reading view:

```sh
python -m pip install playwright==1.57.0 python-pptx==1.0.2
python -m playwright install --with-deps chromium
python cramer-rao-bound/tests/browser.py
python cramer-rao-bound/tools/export.py
```

`CHROMIUM_EXECUTABLE` optionally selects a system browser. Exports go to the ignored `exports/` directory. Copy the PDF and PPTX to this directory when publishing. The source package excludes dependencies, fonts, and temporary export files.

After staging new or modified public HTML, regenerate site discovery:

```sh
git add cramer-rao-bound/index.html
python3 scripts/build-search-index.py
python3 scripts/build-search-index.py --check
```

## Source layout

`src/deck.json` contains slide text, HTML fragments, references, and presenter notes. `src/formulas.json` holds editable TeX; `src/formula-svg.json` is its generated SVG cache. `src/math.js` contains the pure numerical models; `src/app.js` contains presentation behavior and SVG demos. `src/style.css` defines desktop, mobile, and print layouts. The deck uses its own lightweight offline renderer and does not change the site's shared Bento runtime.

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
