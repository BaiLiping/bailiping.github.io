# JPDA teaching lab and companion slides

The lab at `/jpda/` and the Bento deck at `/jpda-slides/` share the probability
model in `math.mjs` and plot renderer in `scene.mjs`. They are linked from page 4
of `/et-handover/`. Links use the current tab; deck demos load only when opened.

Primary references are the user-provided `55JOE.pdf` (Fortmann, Bar-Shalom and
Scheffe, 1983) and `358CSM.pdf` (Bar-Shalom, Daum and Huang, 2009), already hosted
in `et-handover/papers/`. Source equations and modeling assumptions are identified
in the article and speaker notes. This is a synthetic one-scan example, not a
replication of either paper's experiments.

Run from the repository root:

```sh
node --test jpda/math.test.mjs
node jpda-slides/build.mjs
node et-handover/build.mjs
node et-handover/validate.mjs et-handover/index.html jpda-slides/index.html
python3 -m http.server 8767 --bind 127.0.0.1
```

The slide builder reuses the checked-in Bento runtime and license notices from
`et-handover/index.html`; author content in `jpda-slides/deck.mjs`, not the
generated HTML. Static figures are regenerated from the same presets as the lab.
MathJax is shared with the existing handover deck and needs no external CDN.

Browser checks should cover all four presets; sliders, pointer and keyboard
movement; event selection; desktop and mobile layout; the handover-page link;
lazy demo loading; Back/Escape and restored focus; iframe removal; and printed
static explanations. Position controls provide an alternative to dragging.
