# See how samplers think — Bento presentation

`index.html` is generated from `bento-deck.mjs` with the site's shared Bento
runtime in `../kalman-filter-derivations/index.html`, the same pipeline and
slide system as `/radar-slam/` and `/frame-registration-slides/`. Do not edit
the generated slide JSON by hand. This deck replaced the earlier single-page
Sampling Playground at the same URL (24 Sep 2026).

```sh
node sampling-playground/build-bento.mjs
node --test tests/sampling-playground.test.cjs
python3 tests/sampling-playground-browser.py   # needs Playwright; PW_CHANNEL=chrome uses installed Chrome
python3 ~/.codex/skills/build-interactive-slides/scripts/audit_bento.py --strict sampling-playground/index.html
```

27 slides: cover, contents, three framing slides, seven intro → live lab pairs
(`gibbs`, `mh`, `hmc`, `slice`, `rejection`, `importance`, `smc`), a Gibbs-as-MH
derivation, a measured chain scorecard, comparison, chooser, diagnostics,
connections, takeaways and references. Stable hashes such as `#hmc` open the
introduction; the next slide (`#hmc-live`) mounts the lab. The contents slide,
cover counts and page numbers are derived from the slide list at build time.

- `model.js`: every sampler, seeded. The live lab, the deck build and the tests
  all run this file, so every number on a slide (acceptance, ESS, envelope
  constant, weight ESS, particle-filter errors) is computed, not typed.
- `live/`: the dependency-free lab (`?demo=gibbs|mh|hmc|slice|rejection|importance|smc`,
  `&embed=region` inside the deck). Colours follow the site palette; the theme
  block is `body[data-interactive-slides="sampling"]` in
  `../assets/interactive-slide-system.css`.
- `fallback/*.png`: each lab's deterministic initial state, shown under the live
  frame and in print. Regenerate after changing the lab (then quantize to 128 colours):
  `"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless
  --force-device-scale-factor=2 --window-size=1136,475
  --screenshot=sampling-playground/fallback/gibbs.png
  "http://localhost:8000/sampling-playground/live/?demo=gibbs&embed=region"`.
- `print.js`: routes the browser Print command through Bento's print pages.

Corrections relative to the earlier page: the slice sampler now uses Neal's
randomized stepping-out budget (J = floor(mV), K = m − 1 − J); MH acceptance no
longer counts the start point; contours are true Mahalanobis radii (39% and 86%
of the mass) drawn at the samples' scale; the chain comparison reports cost per
move and long-run Geyer ESS instead of a lag-1 proxy alone; the importance-sampling
z-score is reported from the model (the old page said 1.9; it was 1.8); the particle
filter is checked against the exact Kalman posterior; and the "on this site"
connections no longer overstate what the linked pages implement.
