# Rigid frame registration — Bento presentation

`index.html` is generated from `bento-deck.mjs` with the site's shared Bento
runtime in `../kalman-filter-derivations/index.html`, the same pipeline and
slide system as `/radar-slam/`. Do not edit the generated slide JSON by hand.

```sh
node frame-registration-slides/build-bento.mjs
node --test tests/frame-registration.test.cjs
python3 tests/frame-registration-browser.py   # needs Playwright
```

21 slides: cover, contents, 15 concept slides, three intro → live lab pairs
(`ransac`, `icp`, `ndt`), and the shared Extensions appendix from
`assets/deck-extensions.mjs`. Stable hashes such as `#icp` open the
introduction; the next slide (`#icp-live`) mounts the lab. The contents slide,
cover page count and page numbers are derived from the slide list at build time.

- `live/`: the dependency-free RANSAC / ICP / NDT lab (`?demo=ransac|icp|ndt`,
  `&embed=region` inside the deck). Colours follow the site palette.
- `fallback/*.png`: the lab's deterministic initial state, shown under each live
  frame and in print. Regenerate after changing the lab:
  `"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless
  --force-device-scale-factor=2 --window-size=1136,475
  --screenshot=frame-registration-slides/fallback/icp.png
  "http://localhost:8000/frame-registration-slides/live/?demo=icp&embed=region"`.
- `print.js`: routes the browser Print command through Bento's print pages.

Equations are LaTeX typeset by the pinned MathJax 3.2.2 bundle. Bento table
cells are not re-typeset, so keep LaTeX out of `nativeTable` rows.
The companion article is `/frame-registration/`.
