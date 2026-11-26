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

## Rebuild
```
blender -b --python build.py -- <base model glb (commit 91d619b)> none side,rear,front out.glb
node compress_model.mjs out.glb honda_civic_type_r.glb    # Draco, needs @gltf-transform/* and draco3d
```
Each region script may change only the parts listed for it in `ownership.json`; `build.py` checks this. `render_views.py` renders the model from any solved camera json, for overlays.

## Known differences
- The headlamp is about 12 cm short of the CAD at its rear end (side view).
- The rear-quarter crease behind the rear door is not modelled.
- The wheel-arch flares are unchanged.
