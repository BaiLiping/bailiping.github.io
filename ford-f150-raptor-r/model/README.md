# Rebuilding the 2024 F-150 Raptor R model

[Download the editable Blender master](ford_f150_raptor_r.blend). It includes the articulated truck in metres, packed textures and 50 cameras named for the Ford orbit frames and detail photographs. Each camera stores its source, reference pixel dimensions and the viewer pose. Set the scene resolution to the camera’s reference dimensions when rendering a detail crop.

The served model is built in two stages.

1. **Rig, running gear, engine bay and cabin.** These come from the assembly-corrected model of commit `a391343`. `build_model.py` built it from the earlier `007d1fc` asset (see [Stage 1](#stage-1-assembly-corrected-baseline)).
2. **Exterior body.** `exterior/build_raptor.py` removes the baseline's 444 exterior meshes and builds a new body fitted to Ford's official F-150 Raptor 360° renders, Ford's Raptor R specifications and close reference photographs. The doors, hood, tailgate, tonneau and fuel door are reparented to relocated hinge pivots. The cabin trim and the top of the engine are trimmed to clear the new glass and hood.

## Build

Requirements: Node.js/npm, Blender (the served asset was built with Blender 5.2.1 LTS), and this repository's Git history. Blender supplies NumPy. The decoder uses the checked-in `vendor/draco_decoder.js`; compression dependencies are pinned in `package.json`.

Run from the repository root. The final command replaces the served GLB.

```bash
raptor_build_dir="$(mktemp -d -t raptor-r-build.XXXXXX)"

git show a391343:ford-f150-raptor-r/ford_f150_raptor_r.glb > "$raptor_build_dir/baseline.glb"

printf '%s  %s\n' '5941cd6de5d2a5fbdaedf896bcb8f81615c26b7ac734ed3dcb6df3c9ca35cc3a' "$raptor_build_dir/baseline.glb" | shasum -a 256 -c -

node ford-f150-raptor-r/model/decode_model.cjs \
  "$raptor_build_dir/baseline.glb" "$raptor_build_dir/baseline-decoded.glb"
# decoded SHA-256: bcb990a30cb36c676570fcc013323c06d539c15654c99f297e83fb091a94ed3d

blender -b --python-exit-code 1 \
  --python ford-f150-raptor-r/model/exterior/build_raptor.py -- \
  "$raptor_build_dir/baseline-decoded.glb" "$raptor_build_dir/rebuilt.glb" --blend "$raptor_build_dir/editable.blend"

blender -b "$raptor_build_dir/editable.blend" --python-exit-code 1 \
  --python ford-f150-raptor-r/model/save_reference_master.py -- \
  ford-f150-raptor-r/model/ford_f150_raptor_r.blend

npm --prefix ford-f150-raptor-r/model install

node ford-f150-raptor-r/model/compress_model.mjs \
  "$raptor_build_dir/rebuilt.glb" ford-f150-raptor-r/ford_f150_raptor_r.glb
```

The reviewed Draco GLB has SHA-256 `74643b5dde41cf3b4d660a5a6a714b9d83b283cd988a6420188680e6a7de0f88`. Blender and dependency versions can affect the encoded bytes.

`build_raptor.py` also writes `rebuilt.build.json`. It reports the restored `_STUDIO_VISIBILITY` primitives, the removed and added meshes per pivot, the overall dimensions and the wheelbase assertion (3.6932 m).

## Checks

```bash
blender -b --python-exit-code 1 --python ford-f150-raptor-r/frame-by-frame/measure_model.py -- \
  ford-f150-raptor-r/ford_f150_raptor_r.glb measurements.json
blender -b --python-exit-code 1 --python ford-f150-raptor-r/model/exterior/check_offroad_angles.py -- \
  ford-f150-raptor-r/ford_f150_raptor_r.glb
```

The served asset measures 5,878 × 2,210 × 2,049 mm (2,440 mm with mirrors), with a 3,693 mm wheelbase, 1,887/1,877 mm tracks and 333 mm ground clearance. Its approach, departure and breakover angles are 33.3°, 24.9° and 23.8°; Ford gives 33.1°, 24.9° and 24.4°. No painted panel touches the headlamp lenses or grille frame, and the hood clears the fixed front panels from closed to 48°. The length is 30 mm under Ford's published 5,908 mm. Ford's calibrated side renders place both bumper faces within about 2 cm of the model at their most prominent heights, so the bumpers were not stretched to the published figure.

Before publishing, inspect the model from the front, side, rear and above. Exercise the doors, hood, tailgate, tonneau, fuel door, exploded view and suspension controls. Update the model URL's cache version in `index.html`.

## Photo-led refinements

Ford's driver and passenger profiles and the auction truck's three-quarter photographs show a pronounced lower door stamping, a painted window header around the black weatherstrip, and a shadow under each grab handle. The exterior skins and card clearance reproduce those features on all four hinges. The driver's bedside has a cutout for the filler so opening the flap reveals the neck.

The auction truck's illuminated and unlit front close-ups were used for the upright and upper amber diffusers, the two deep projector reflectors, the narrow undulating grille webs and the horizontal orange recovery loops in the bumper notches. The skid plate retains its exposed alloy finish. The auction truck has no tall antenna mast at its passenger fender. Ford's standard Raptor orbit includes one.

The same-year side photograph places the heat extractor 0.30 m farther forward than the earlier mesh. The three pods and their shared tray now clear the graphic-covered crown in the low front view. The elevated auction photograph constrains the deeper cups, four base slots per cup, and the position of the two outboard badges.

The 2024 interior photos guided the contoured front cushions, tapered backrest bolsters, perforated inserts and orange welt stitching. Each original seat frame retains its headrest, badge, switches and belt latch. Both front door cards now have an open lower storage bin, molded middle shoulder, padded armrest and switch bridge. The passenger-side card shape is inferred from the driver-side close-up.

## Exterior sources (`exterior/`)

| File | Contents |
| --- | --- |
| `design.py` | Measured body data: plan and section curves, door and window outlines, greenhouse, roof, windshield, hood and Power Dome, flares, bed. |
| `geo.py`, `mk.py` | Monotone interpolation and splines; panels on analytic surfaces with rolled hems; sweeps and mirroring that preserves normals. |
| `parts_body.py` | Side panels, fender decks, doors, greenhouse, roof (with its stamped ribs), cab back and hood. |
| `parts_front.py` | Grille and FORD letters, valance, front bumper, LED fog modules, skid plate and recovery hooks. |
| `parts_rear.py` | Tailgate, rear bumper with exhaust outlets and receiver, bed box. |
| `parts_misc.py`, `parts_extra.py` | Flares, wheelhouse liners, mirrors, handles, running boards, fuel door, fender vents, wipers, decals, tonneau, bed details, badges. |
| `parts_detail.py` | Second-pass details from close-ups: headlamps with C-clamp light guides, taillamps, bed rear pillars, hood nose, extractor, CHMSL, roof antenna pods, C-pillar trims, wheel-well aprons. |
| `seat_trim.py`, `door_trim.py` | Front seat upholstery and front door-card moldings from the 2024 interior photographs. |
| `build_raptor.py` | Removes the old exterior, builds and parents the new one, fits kept parts inside it and exports the GLB. |
| `fordcam.py`, `ford360_cameras.json` | Calibrated cameras for Ford's 36 renders, to overlay the model on the official views. |
| `check_offroad_angles.py` | Approach, departure and breakover angles of an exported GLB. |

The views tab includes all 36 native 2160 × 926 Ford studio frames. `fordcam.project(k, points)` maps model coordinates onto frame `k` (01–36, 2160 × 926) of Ford's F-150 Raptor 360° view. Ford renders the body 62 mm lower on its wheels than the published ride height. The cameras compensate, so overlays line up with the body.

## References and limits

- [Ford F-150 Raptor 360° view](https://www.ford.com/trucks/f150/models/raptor/): 36 official studio renders per paint colour. Their cameras were solved from the wheel hubs and lug-pattern hexagons, with the wheelbase fixing the scale, a mirror-symmetric rig and silhouette refinement. They define the body silhouettes and panel lines: cab and tumblehome, windshield rake, hood and fender line, lamp outlines, the bed's plan taper and both bumpers. The gallery's 2027 visualizer shows an R-badged Ruby Red truck with no tonneau. Compare shared exterior features against that orbit and use the 2024 photographs for year-specific trim. The model retains the 2024 configuration in Rapid Red.
- [Ford's 2024 Raptor/Raptor R technical specifications](https://www.fromtheroad.ford.com/content/dam/fordmediasite/us/en/library/2024/specs/2024-Ford-Raptor-and-Raptor-R-Tech-Specs.pdf), pages 3–4: overall width and height, tracks, wheelbase, ground clearance, and approach, departure and breakover angles. These Raptor R figures take precedence over generic F-150 dimensions.
- [Ford Pro 2024 F-150 Body Builders Layout Book](https://api-prd.fordpro.com/pro/bbas/download/BBLB/661025b789417d202fd69d4f), pages 6–8: dimensioned plan and side views, cab layout and cargo-box drawings. It is a **common F-150** drawing and does not cover the Raptor's fenders, bumpers, lamps, hood or ride height. [Ford Pro publication index](https://www.fordpro.com/en-us/upfit/publications/).
- [Ford Mexico's 2024 Lobo Raptor brochure](https://www.ford.mx/content/dam/Ford/website-assets/latam/mx/nameplate/raptor-2018/lobo-raptor-2024/pdfs/ford-lobo-raptor-2024-catalogo-descargable.pdf), page 14: a generic side and rear outline. Its 2,049 mm width conflicts with the US Raptor R specification of 2,210 mm, so it is not used for body width.
- [Bring a Trailer lot 181611](https://bringatrailer.com/listing/2024-ford-f-150-raptor-r-18/) (125 photographs of a 2024 Raptor R) for the lamp internals, extractor, tailgate lettering and layout, bumper, exhaust outlets, hitch and CHMSL. Plate-scaled measurements from the rear photograph set the tailgate layout and the bed taper at the lamps. These details are photo-derived approximations, not factory part measurements.

The result is a reference-fitted visual model. These public sources give silhouettes, panel lines and dimensions, not Raptor R surface CAD, so the surfaces between those constraints are modeled.

## Stage 1: assembly-corrected baseline

`build_model.py` applied local geometry corrections (`correct_cabin.py`, `correct_rear.py`, `correct_wheels.py`) to the verified articulated model at commit `007d1fc7ff1f939016f3ddcc9330afb3a02dfca5`. Its compressed input has SHA-256 `079dc5c21f3db2dc80411b90d5b1ab3a73d1d1692e76cdd8c4c4b3f847bfb194`. It restored the original `_STUDIO_VISIBILITY` data and preserved the named pivots, running gear and cab/bed separation. It did **not** reuse the later global silhouette warp, which displaced rigid components. Its output, served at commit `a391343`, is the baseline for Stage 2:

```bash
git show 007d1fc7ff1f939016f3ddcc9330afb3a02dfca5:ford-f150-raptor-r/ford_f150_raptor_r.glb > "$raptor_build_dir/stage1-input.glb"
node ford-f150-raptor-r/model/decode_model.cjs "$raptor_build_dir/stage1-input.glb" "$raptor_build_dir/stage1-decoded.glb"
blender -b --python-exit-code 1 --python ford-f150-raptor-r/model/build_model.py -- \
  "$raptor_build_dir/stage1-decoded.glb" "$raptor_build_dir/corrected-uncompressed.glb"
```
