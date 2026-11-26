# Rebuilding the 2024 F-150 Raptor R model

This build applies local geometry corrections to the verified articulated model. It preserves the named pivots, running gear, cab/bed separation, and closed tonneau-cover alignment. It does **not** reuse the later global silhouette warp, which displaced rigid components and broke those relationships.

## Source provenance

- Baseline commit: `007d1fc7ff1f939016f3ddcc9330afb3a02dfca5`
- Baseline asset: `ford-f150-raptor-r/ford_f150_raptor_r.glb`
- SHA-256 of that compressed predecessor asset: `079dc5c21f3db2dc80411b90d5b1ab3a73d1d1692e76cdd8c4c4b3f847bfb194`

Always rebuild from this baseline. Applying the correction modules to an already corrected or globally warped GLB is unsupported.

## Build

Requirements: Node.js/npm, Blender 4.3 or compatible, and this repository's Git history. Blender supplies the Python dependencies. The decoder uses the checked-in `vendor/draco_decoder.js`; compression dependencies are pinned in `package.json`.

Run from the repository root. The final command replaces the served GLB.

```bash
raptor_build_dir="$(mktemp -d -t raptor-r-build.XXXXXX)"

git show 007d1fc7ff1f939016f3ddcc9330afb3a02dfca5:ford-f150-raptor-r/ford_f150_raptor_r.glb > "$raptor_build_dir/baseline.glb"

printf '%s  %s\n' '079dc5c21f3db2dc80411b90d5b1ab3a73d1d1692e76cdd8c4c4b3f847bfb194' "$raptor_build_dir/baseline.glb" | sha256sum -c -

node ford-f150-raptor-r/model/decode_model.cjs \
  "$raptor_build_dir/baseline.glb" "$raptor_build_dir/baseline-decoded.glb"

blender -b --python-exit-code 1 \
  --python ford-f150-raptor-r/model/build_model.py -- \
  "$raptor_build_dir/baseline-decoded.glb" "$raptor_build_dir/corrected-uncompressed.glb"

npm --prefix ford-f150-raptor-r/model install

node ford-f150-raptor-r/model/compress_model.mjs \
  "$raptor_build_dir/corrected-uncompressed.glb" \
  ford-f150-raptor-r/ford_f150_raptor_r.glb
```

`build_model.py` restores the original `_STUDIO_VISIBILITY` data, applies `correct_cabin.py`, `correct_rear.py`, and `correct_wheels.py`, then exports a GLB and a sibling `corrected-uncompressed.build.json` report. It checks unchanged articulation pivots, a 3.6932 m wheelbase, cab/bed clearance, cover/rail alignment, and rear-differential clearance. `compress_model.mjs` uses Draco with 16-bit position quantization.

Before publishing, inspect the rebuilt model from the front, side, rear, and above, and exercise the doors, hood, tailgate, tonneau cover, and suspension controls. Update the model URL's cache version in `index.html` when replacing the deployed asset. Reproduction is at the source/geometry level; different Blender or dependency versions may produce different binary bytes.

## References and limits

- [Ford's 2024 Raptor/Raptor R technical specifications](https://www.fromtheroad.ford.com/content/dam/fordmediasite/us/en/library/2024/specs/2024-Ford-Raptor-and-Raptor-R-Tech-Specs.pdf), pages 3–4: Raptor R wheel/tire specification, overall dimensions, axle tracks, and clearance. These Raptor-specific figures take precedence over generic F-150 dimensions.
- [Ford Pro 2024 F-150 Body Builders Layout Book](https://api-prd.fordpro.com/pro/bbas/download/BBLB/661025b789417d202fd69d4f), pages 6–8: dimensioned plan/side views, cab layout, and cargo-box drawings. This is a **common F-150** drawing with multiple cab configurations, not a Raptor R exterior blueprint. It supports shared structural dimensions, not the Raptor's fenders, bumpers, lamps, hood, or ride height. [Ford Pro publication index](https://www.fordpro.com/en-us/upfit/publications/).
- [Ford Mexico's 2024 Lobo Raptor brochure](https://www.ford.mx/content/dam/Ford/website-assets/latam/mx/nameplate/raptor-2018/lobo-raptor-2024/pdfs/ford-lobo-raptor-2024-catalogo-descargable.pdf), page 14: side/rear dimension schematic and supplementary bed/overhang dimensions. Its outline is also generic. Its 2,049 mm width conflicts with the dedicated US Raptor R specification of 2,209.8 mm, so it is not used for body width.
- [Ford's 2024 Raptor R announcement and exterior photograph](https://www.fromtheroad.ford.com/us/en/articles/2023/dashing-through-the-sand--new-ford-f-150-raptor-r-cranks-out-720), plus the existing photo comparisons under `../frame-by-frame/frames/`, guide visible styling details. Photo-derived mirror, handle, window, rear-body, and tread adjustments are approximations rather than factory part measurements.

The result is a reference-informed visual model. These public sources do not provide complete Raptor R surface CAD or establish an exact replica of every component.
