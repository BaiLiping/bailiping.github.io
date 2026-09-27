# Scalable Extended-Target Handover

Public presentation at <https://bailiping.com/et-handover/>.
The 22 regular Bento slides include two introduction/live pairs,
GrBP processing diagrams, a simulation-environment animation, results, and
five failure-analysis slides immediately before the conclusion.
The previous `/eo-handover-slides/` and `/eo-handover-slides/live/` URLs redirect
here, preserving query parameters and slide bookmarks.

## Edit and build

`deck.mjs` is the slide authoring source. `build.mjs` replaces the document and
live-demo mapping in `index.html`, preserving its bundled Bento runtime.
All required assets and interactive demonstrations live in this directory.
Links to supporting presentations use site-relative URLs and navigate in the
current browser tab, including when previewing the site locally.
The local MathJax 3.2.2 license is in `assets/vendor/MathJax-LICENSE`.

From the repository root:

```sh
node et-handover/build.mjs
node et-handover/validate.mjs
python3 -m http.server 8767 --bind 127.0.0.1
```

In another terminal on Linux (Node.js 20):

```sh
CHROME_BIN=/usr/bin/google-chrome node --experimental-websocket et-handover/qa.mjs http://127.0.0.1:8767
```

QA uses local headless Chrome (`CHROME_BIN` overrides its macOS default path).
It checks every slide, rendered mathematics, both demos, payload accounting,
ownership transfer, responsive layouts, keyboard navigation, and print fallbacks.
It regenerates the two fallback PNGs in `assets/`; screenshots and the report go
to `/tmp/eo-grbp-qa` (`BENTO_QA_DIR` overrides that path).

`live/demo.js` contains the deterministic timeline/model, toy renderer and
controls, and score widget. `live/index.html` is their standalone route;
`?slide-embed=%23toy` or `?slide-embed=%23handover-score-demo` selects a focused
region. Inactive deck demos unload; Page Up / Page Down return navigation to Bento.
The slide preview stays visible with a loading status until the focused demo's
layout and equations are ready. The full source page is hidden during setup.
With Playwright installed, run the cold-load regression using
`CHROME_BIN=/usr/bin/google-chrome node et-handover/qa-loading.cjs http://127.0.0.1:8767`
(`PLAYWRIGHT_MODULE` can point to an existing Playwright package).
It delays the embed stylesheet and demo script at desktop and mobile widths,
then checks controls, navigation away during loading, revisits, and print previews.

## Preserved presentation details

- The cover uses the title **Scalable Extended-Target Handover**,
  and the same white background and dark text as the rest of the deck. Its
  centered Figure 1 is copied unchanged from
  `Drawings/Target Handover.png` at writing revision `7d88f02` to
  `assets/manuscript-figure-1.png`. The central unit depicts the coordinated baseline.
- The cover link **extention of point-target handover** opens the matching
  point-target presentation in the same tab.
- The cover's **Paper · arXiv:2609.25737** link opens
  <https://arxiv.org/abs/2609.25737> in the same tab.
- Slide 2, **Scalability is core design objective for DISAC**, links to the Extended-Target
  Tracking and Density Fusion presentations through two full-box links that
  open in the same tab. Use the browser Back button to return from these public decks. Their hover and keyboard-focus styles live in `slides.css`;
  `topic-links.js` restores the authored page links after Bento renders.
  A **Belief-Propagation MTT** link below **Solution: GrBP** opens <https://bailiping.com/bp-vs-pmbm-slides/>.
- Slide 14 marks the target leaving BS A's field of view. The transfer arrow is
  highlighted only during the request and acknowledgment. BS A owns `(A, 3)`
  until acknowledgment arrives; then BS C owns `(C, 7)`. The original sidebar
  and normal playback speed remain.
- Slide 15 introduces the seven-base-station environment using the supplied
  `centralized_animation (2).gif`. The presentation copy at
  `assets/simulation-environment.gif` is rendered from the same archived
  `mc_0088` results with the in-plot legend removed. The nine legend entries
  sit beneath **Simulation environment**, with symbols from the original
  plotting code in `assets/simulation-legend-symbols.svg` and native slide text.
  Native slide elements replace the static
  animation heading with **GrBP Tracking / mc_0088**, preserving the
  animated frame counter and 10-second playback. Results follow on slide 16.
- These are explanatory demonstrations, not a numerical implementation of GrBP
  inference. Diagram and benchmark content follows the current
  *Scalable Extended-Target Handover in Distributed Integrated Sensing and
  Communication* manuscript. Experiment payloads exclude protocol headers;
  the toy additionally counts illustrative 24-byte control messages.

- Slide 3 links to **Derivation for GrBP** at <../eo-derivation/#grbp>, with
  the corresponding offline slide deck.

## Failure-analysis animations

Slides 17 and 18 are **Extent Expansion** (the animations on pages 11 and 12
of `Sep 11.pptx`, side by side) and **Multiple Initiation** (page 13).
The conclusion is slide 22. The animations keep their original 60, 61, and
62 frames, 100 ms frame delays, and infinite looping.

The original plots shifted horizontally by 39, 42, and 33 pixels across their
frames. `prepare-failure-gifs.py` measures the vertical plot borders and uses
FFmpeg to translate whole frames onto a fixed canvas. It clears the embedded
title and crops the top 36 rows, keeps the axis labels and annotations,
and encodes opaque full frames with one global palette and no dithering.
No temporal smoothing or frame interpolation is applied. Source hashes,
frame translations, and output checks are recorded in
`assets/failure-analysis-provenance.json`.

To regenerate from the supplied PowerPoint (requires FFmpeg, Pillow, and NumPy):

```sh
python3 et-handover/prepare-failure-gifs.py '/path/to/Sep 11.pptx'
node et-handover/build.mjs
node et-handover/validate.mjs
```

## Underlying Multimodal distribution

Slides 19–21 share the requested title **Underlying Multimodal distribution**
and use the pictures from slides 6–8 of `Oct 2.pptx`, in that order. Each
picture has its own slide, with both panels enlarged beneath the heading.
The examples are BS5 at frame 182/200, BS5 at frame 91/200, and BS3 at
frame 139/200. Each shows the tracking scene on the left and particle clouds
on the right, including the original red annotation highlighting the case.

`prepare-multimodal-figures.py` resolves PowerPoint slide order and selects
each slide's plot and red ink annotation. It embeds both original PNGs
unchanged in a self-contained SVG, retaining their PowerPoint placement.
A cropped viewport removes both embedded panel titles and unused margins.
On the final figure, a header-only SVG mask clears the title's last
antialiased row without clipping the top axis tick.
Axis labels, trajectories, particle clouds, and red highlights remain intact.
Source hashes, image placement, and crop bounds are recorded in
`assets/multimodal-provenance.json`.

```sh
python3 et-handover/prepare-multimodal-figures.py '/path/to/Oct 2.pptx'
node et-handover/build.mjs
node et-handover/validate.mjs
```

## Multi-target factor graphs

Slides 3, 5, 6, and 7 share the expanded graph authored in
`drawings/build-graphs.cjs`. Each local update shows the first and last legacy
chains and the first and last group/newborn chains, with ellipses between them.
The shared consistency factor couples every target and group association.
This adapts the factor graph preserved in `drawings/graph-drawing-original.tex`.

The diagrams use the deck's colors and render LaTeX labels as self-contained SVG.
The factor labels use `f(\cdot)`, `\underline l(\cdot)`, and `\overline l(\cdot)`.
Arguments are abbreviated by a dot; variable nodes retain their indices.
Orange inputs show the current prediction `\mathcal F_k^-`, with a
BS index in the distributed diagram; blue outputs show the current posterior.
Sequential stages retain the preceding posterior as their input density.
The time index is suppressed. The parallel diagram retains one common
legacy count and BS-specific group counts. Sequential stages take all incoming
components, including the preceding stage's newborn proposals. Dark edges show
factor dependencies; colored arrows between cards show processing flow.

Regenerate the four vector figures before rebuilding the deck:

```sh
npm ci --prefix et-handover/drawings --ignore-scripts --no-audit --no-fund
node et-handover/drawings/build-graphs.cjs
node et-handover/build.mjs
```

The handover-variant packet diagram depicts a shared track's messages, so it
does not use the former one-row factor-graph shorthand.
