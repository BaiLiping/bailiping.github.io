# Porsche 911 Turbo S Cabriolet (992.2)

The Gentian Blue Metallic car uses the EU launch specification, 20/21-inch Turbo S centre-lock wheels, yellow PCCB brakes, a black textile roof and a left-hand-drive four-seat cabin. Porsche’s October 2025 MY T sheet identifies the 2026 model.

## Build

Extract the rebuild archive. From its root, run the following with Blender 5.2.1 and its bundled Draco exporter.

```sh
blender -b --python-exit-code 1 -P blender/build_master.py -- --out-prefix final
```

The build writes the editable metre-scale master and the viewer-scale GLB to `blender/out/`. In the master, +X points forward, +Y to the driver's left and +Z up. The origin is on the ground halfway between the axles. One GLB unit represents 18 real millimetres. Each assembly has a named pivot. The master includes both positions of the fabric hood and 62 named reference cameras. Each camera records the reference dimensions and intended roof state.

## How the body follows the photographs

The body is a parametric shell moved by a silhouette-fitted displacement (`blender/exterior/surface_fit.py`, field in `silhouette_field.npz`). Four groups of parts are built to targets measured on the Porsche studio frames through their calibrated cameras:

- **Greenhouse** (`blender/exterior/greenhouse.py`). The side-window top line was measured in the left profile, and the screen header and roof edge in the overhead frame. The canvas edge, the door and quarter glass and the slanted B-seal are built in assembled coordinates on that line. The header corners sit at the A-pillar tops, about 0.60 m from the centreline, and the A-pillar is a flush strip rather than a tube.
- **Front end** (`_front_design_z` in `blender/exterior/body_shell.py`, `HOOD_SIDE_*`, `LID_*` in `config.py`). The lid's shut line is triangulated from the overhead and straight-on frames (`tools/measure_lid.py`, 5 mm rms). It narrows to rounded corners and a front edge that bows forward ahead of the crest. The lid is domed above that edge. The fender barrels rise from the shut line to the crown line measured in both profiles (`tools/front_targets.py`). The design is made in assembled coordinates and carried back to rest space through the inverse of the body deformation, so no corrective deltas are stacked on the front. The silhouette field is dropped on the deck, round the headlamps and on the nose. On the fender flanks round the front wheel it still shapes the straight-on outline.
- **Headlamps** (`blender/exterior/headlamp_fit.json`). Each lamp frame is the planar ellipse fitted to its outline in eleven front, three-quarter, side and overhead frames (`tools/fit_lamp.py`). The bezel and lens are raycast onto the fender barrel along the lamp axis, so the ring follows the barrel as on the car.
- **Fuel door** (`blender/exterior/fuel_fit.json`). Its outline is an ellipse fitted to the shut line in ten right-side and overhead frames (`tools/fit_fuel.py`), cut exactly into the fender. It opens outward about its forward edge (`hingeAxis` on `Pivot_Fuel_Flap`) onto a dark bowl with a rubber lip and the filler cap.

Painted panels get faired shading normals (`blender/exterior/fair_normals.py`): a feature-preserving diffusion over 15 cm. The silhouettes and shut lines stay where the fit put them, and the reflections run fair. The browser renders Gentian Blue as a clear-coated metallic under its own studio environment, whose parameters were fitted to 20 studio frames (`tools/fit_paint.py`). Grazing reflectance is capped, as in the studio frames, and the key light leaves no hot spots in the lacquer.

Measured against the 52 studio masks, the exterior silhouette overlaps by 97.1% on average. The median boundary distance is 6.4 px and the 95th percentile 27 px at 3840 px width (`tools/silscore.py`). Per-frame statuses, notes and measurements are in `frame-review.json`.

## Known differences

The bumper has a rolled upper edge where the car has a crisp shelf crease. The outer intakes sit lower with thinner flaps, and the central opening has no radiator grille. Straight on, the headlamps read slightly smaller and flatter than on the car. The fender round the fuel door faces slightly more sideways than the car. The rear reflectors, valance ribs and parking sensors are simplified, and the tyre lettering is larger than Pirelli's. The gearbox's internal tooth profiles, the backs of component castings and concealed pipe runs are inferred. The photographed sides of the engine and gearbox combine mesh detail with surface textures derived from Porsche’s illustrations.

The photo references and illustrations are credited to their respective publishers in `inventory.json`. The [Car_Photos catalog](https://github.com/BaiLiping/Car_Photos/tree/726ad07ff53ea2797bf83041ad73aba26ede6b18) includes third-party meshes of a 2019 Carrera 4S Cabriolet and a 2025 GT3. They were not used: the Turbo S Cabriolet body, roof and bumper geometry follow the Turbo S studio photographs.
