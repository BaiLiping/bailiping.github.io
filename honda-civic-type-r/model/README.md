# 2020 Civic Type R (FK8) model: photo-matched exterior refinement

The live model is `../honda_civic_type_r.glb`. It is built from the previous, CAD-aligned model (commit 91d619b), which already matched Honda's official P24 three-view outline to within millimetres. This pass corrects the shape *inside* that outline.

## References
- **Honda factory data:** the Oct 2020 factbook P24 vector three-view (1 px = 1 mm plates), the P09 studio front/rear, and the USDM body repair manual dimensions.
- **American Honda's official 2020 Boost Blue 360° colour-selector renders:** 36 frames from automobiles.honda.com, recovered from the Internet Archive. Their cameras are solved from the renders. The CGI body sits 49 mm lower on its wheels than the real car.
- **A 36-frame turntable spin of a stock 2017 FK8** (Clutch.ca). Its cameras are solved from silhouettes and anchored to the wheel centre caps, at the official wheelbase and track.
- **Stock 2020-2021 FK8 photo sets** for the facelift front.

## What changed
- **Side:** doors re-cut to Honda's P24 shut lines; FK8 shoulder, lower scoop and rear sweep creases; body-colour sills; B-pillar leans back; front door handles and the fuel door moved to the CAD positions; front door hinges moved to the new door edge.
- **Rear:** A-frame wing uprights in body colour and a gloss-black aerofoil wing; the forward wrap of the boomerang taillamp onto the quarter; body-colour roof vortex generators.
- **Front:** 2020 grille frame and honeycomb opening; gloss-black lower bar; headlamp top line; bonnet character lines; fog lamps and hood scoop opening moved to the CAD; black brow over the lamps; see-through gaps into the engine bay closed.

## Later passes (2026-11-27)
- **Rear proportions:** a smooth (C2 B-spline) warp of the rear section, fitted to Honda's 360° frames with the lamp and badge positions anchored, moves the upper tail 20–40 mm forward and rounds the upper rear corners in plan.
- **Side skin:** the door and quarter skins are refitted to a smooth half-width map that carries the P24 crease lines, which removes the dents in the doors.
- **Smooth reflections:** the body mesh is uneven at a 10–30 cm scale, and glossy paint showed every ripple. The large painted panels now carry shading normals computed from a smoothed copy of each panel: 12 cm radius, 20 cm on the hood, front fenders, roof and rear quarter, never across a crease sharper than 25°. Positions are untouched, so the outline is unchanged. Small sharp-edged parts keep their own normals.
- **Boost Blue paint:** a metallic flake base (metalness 1, roughness 0.30) under a light clear coat (0.12). In the viewer the paint reflects its own studio, `../paint-studio.js`, not the room panorama the rest of the car uses. That studio was recovered from Honda's renders: each paint pixel's reflection direction comes from the solved camera and the model's normals, and the radiance per direction was fitted until the viewer's paint matched Honda's brightness, saturation and hue in all twelve full frames, then clamped and smoothed.
- An intermediate rear rebuild (published in commit ea062ed) replaced the quarter, hatch and bumper with one fitted surface textured from Honda's photos. It is withdrawn: up close it showed staircase paint edges, seams and flat sticker lamps, and the fuel door no longer opened. It matched Honda's rear three-quarter outlines about 0.5 percentage points better (intersection over union, IoU).

## Rebuild
```
blender -b --python build.py -- <base model glb (commit 91d619b)> none side,rear,front out.glb
node compress_model.mjs out.glb honda_civic_type_r.glb    # Draco, needs @gltf-transform/* and draco3d
```
Each region script may change only the parts listed for it in `ownership.json`; `build.py` checks this. `render_views.py` renders the model from any solved camera json, for overlays. The rear warp, side skin, normal smoothing and paint steps run in the build workspace (`work/render-fix/build.sh`); their scripts are not in this folder.

## Known differences
- The headlamp is about 12 cm short of the CAD at its rear end (side view).
- The rear-quarter crease behind the rear door is not modelled.
- The wheel-arch flares are unchanged.
- The front fender and bumper corner are still slightly uneven; the smoothing hides most of it in the reflections, not in the outline.
- The taillamp's red light guide and the rear bumper vents' red blades are bolder than Honda's; the taillamp has no outer smoked lens.
- In Honda's rear three-quarter frames the outline matches to IoU 0.949–0.964 (body only); the upper rear corners are still slightly too full.
