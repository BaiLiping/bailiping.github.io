# Target handover presentation

The readable `bento-doc` JSON in `index.html` is the slide source. Preserve the
embedded Bento runtime when editing it. The 15 regular slides include the
handover rule immediately followed by the interactive `live/` experiment, and
a final Extensions slide linking to the project page. The deck is listed under Work; each content slide links back to all topics.
Run `node scripts/install-deck-extensions.mjs` after editing the static document
to refresh the companion links from `assets/deck-extensions.mjs`.
The cover and conclusion link to the IEEE Xplore paper. `paper-links.js`
restores those anchors after Bento sanitizes rich text and creates slide copies.

The deck matches the EO handover presentation: white pages, Arial typography,
navy text (`#16273e`), teal accents (`#087f68`), 72 px title margins, and thin
header rules. The live region occupies `(40, 178, 1200, 486)` on the 1280 × 720 canvas.
The explanatory content under the iframe remains available for print and loading.

## Trajectory source

`live/trajectory.js` contains the ground-truth positions for the simulation
trial used in Figure 3: `all_true_tracks[27][:, :2, :]` from `data_generation.pkl`
in [BPTargetHandover at d78dbc7](https://github.com/BaiLiping/BPTargetHandover/tree/d78dbc7835b4c59a64b4db4fa0ccf1dfc22fd4fd).
`visualize_simulation_scenario.py` selects `trial_0027.pkl`; `centralized.py`
names trials by their zero-based index. There are 101 stored samples at
`t = 0…100 s`, rounded here to six decimal places in metres.

Target A runs from the upper left to the lower right (BS1 → BS2); target B
runs from the upper right to the lower left (BS2 → BS1). The paths and moving
markers use the same coordinate lookup, so scrubbing and playback stay aligned.
The SVG also contains those paths and initial positions as a static fallback.
Handover gates, measurement offsets, and receiver uncertainty remain a teaching
illustration; they are not BP tracking estimates. Particle playback is removed.

Serve the repository root with `python3 -m http.server 8914` and open
`http://localhost:8914/target-handover-slides/`. No build is required.
`scripts/convert-bento-inline-live.mjs` retains the single intro/live mapping.

When checking changes, verify first/crossing/last positions against the source
trial, alignment between each marker and its path at every frame, threshold
controls, playback and seeking, Page Up/Page Down and Escape after iframe
interaction, working paper links, and the static print fallback.
