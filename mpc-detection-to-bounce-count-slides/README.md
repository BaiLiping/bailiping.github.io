# Radio Measurements → Radio Map

Public presentation: <https://bailiping.com/mpc-detection-to-bounce-count-slides/>.
The homepage links to it under **Random thoughts**.

The 14-slide presentation preserves original slides 1–10, 16, 18, 31, and 33
from the private deck at revision `b84536c`. The other 19 slides were removed
before publication. Slide IDs remain stable; visible numbers and live-demo
indices follow the new order.

`deck.json` is the editable slide document; `live-demos.json` connects its five
interactive slides to the geometry and wall-map demos in `live/`.
`build.mjs` updates the readable JSON blocks in `index.html`, retaining the
bundled Bento runtime. It also rebuilds the incidence-point scripts, CSS, and
static illustration from their model, renderer, and control sources.

From the repository root:

```sh
node mpc-detection-to-bounce-count-slides/build.mjs
node --test mpc-detection-to-bounce-count-slides/tests/*.test.cjs
node mpc-detection-to-bounce-count-slides/tests/geometry-regression.mjs
python3 -m http.server 8767 --bind 127.0.0.1
```

Open <http://127.0.0.1:8767/mpc-detection-to-bounce-count-slides/>.
Arrow keys navigate the deck. Escape returns focus from a demo to the deck;
Page Up and Page Down navigate from inside the demo. Inactive iframes unload,
and slide-native explanations and illustrations remain available for print.

The required demo files and MathJax distribution are included locally.
Shared Bento bridge files live in the repository's `assets/` directory.
The MathJax license is in `assets/vendor/MathJax-LICENSE`.
