# Joint PDF factorization and Derivation for GrBP

Public companion at <https://bailiping.com/eo-derivation-slides/>.
The 50 slides cover the original four point/extended and known/unknown-count
scenarios, followed immediately after S4 by **Derivation for GrBP** (17 slides,
starting at slide 31, `#grbp-preclustering`).

The GrBP section follows Appendix I, Eqs. (A.1)–(A.9), from
`BaiLiping/EO_Writing_Long_Version` at commit
`d0a380a9caf81a36b88886efa91b5afd9bb7bdb5`. It fixes a preclustered partition,
then derives group-level association, count cancellation, group likelihoods,
legacy/newborn factors, and the approximate joint PDF. The original S1–S4
teaching slides and handwritten source figures are preserved.

The article is <https://bailiping.com/eo-derivation/#grbp>. The source map and unaltered appendix
excerpt are in `../eo-derivation/source/`.

## Build and review

```sh
cd eo-derivation/source
npm ci --ignore-scripts --no-audit --no-fund
npm run build
```

`content.cjs` authors the original scenarios and overview; `appendix.cjs` authors
the GrBP addition. `build.cjs` verifies the article and appendix hashes, renders
all LaTeX as SVG, and embeds the five figures. The existing presentation runtime,
notes, overview, direct links, and print support are retained. The builder never
modifies the article. The deck and its matching `slides.pdf` work offline.

From the repository root, using Playwright and an installed Chrome:

```sh
CHROME_BIN=/path/to/chrome node eo-derivation-slides/qa.cjs --pdf
```

If Playwright is installed outside Node's search path, set `PLAYWRIGHT_MODULE`
to its package directory. `EO_QA_DIR` controls screenshots and
`EO_QA_SCREENSHOTS=all` captures every slide. QA checks layout, rendered math,
images, scenario links, keyboard controls, notes, overview, mobile width, and
absence of external runtime requests. Reports are `build.json`, `qa-results.json`,
and `pdf-results.json`.

The GrBP section ends with the manuscript's factor graph at `#grbp-graph`
(slide 47). Its SVG is embedded in the deck and remains sharp in the PDF.
To regenerate the figure from the preserved TikZ source, run
`python3 eo-derivation/source/build-grbp-graph.py` from the repository root
(requires a TeX installation with TikZ/standalone and Poppler), then rebuild
the slides. Only the count labels are aligned with Appendix I.
