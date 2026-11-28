# Porsche 911 Turbo S Cabriolet (992.2)

The Gentian Blue Metallic car uses the EU launch specification, 20/21-inch Turbo S centre-lock wheels, yellow PCCB brakes, a black textile roof and a left-hand-drive four-seat cabin. Porsche’s October 2025 MY T sheet identifies the 2026 model.

## Build

Extract the rebuild archive. From its root, run the following with Blender 5.2.1 and its bundled Draco exporter.

```sh
blender -b --python-exit-code 1 -P blender/build_master.py -- --out-prefix final
```

The build writes the editable metre-scale master and the viewer-scale GLB to `blender/out/`. In the master, +X points forward, +Y to the driver's left and +Z up. The origin is on the ground halfway between the axles. One GLB unit represents 18 real millimetres. Each assembly has a named pivot. The master includes both positions of the fabric hood and 62 named reference cameras. Each camera records the reference dimensions and intended roof state.

## How the body follows the photographs

The body is a parametric shell moved by a silhouette-fitted displacement (`blender/exterior/surface_fit.py`, field in `silhouette_field.npz`). Three groups of parts are placed against targets measured on the Porsche studio frames through their calibrated cameras:

- **Greenhouse** (`blender/exterior/greenhouse.py`). The side-window top line was measured in the left profile, and the screen header and roof edge in the overhead frame. The canvas edge, the door and quarter glass and the slanted B-seal are built in assembled coordinates on that line. The header corners sit at the A-pillar tops, about 0.60 m from the centreline, and the A-pillar is a flush strip rather than a tube.
- **Front lid** (`HOOD_HW_*` in `blender/exterior/config.py`). Its outline comes from the overhead frame: the lid's shut lines were raycast onto the body and inverted through the fit, giving the strong forward taper, the rounded corners and the level nose edge.
- **Headlamps** (`blender/exterior/headlamp_fit.json`). Each lamp is a planar ellipse fitted to its outline in eleven front, three-quarter, side and overhead frames (`tools/fit_lamp.py`). The lamp is carried onto it with one affine map, and the paint around the opening blends back to the fit within 1.9 lamp radii.

Painted panels get faired shading normals (`blender/exterior/fair_normals.py`): a feature-preserving diffusion over 15 cm. The silhouettes and shut lines stay where the fit put them, and the reflections run fair. The browser renders Gentian Blue as a clear-coated metallic under its own studio environment, whose parameters were fitted to 20 studio frames (`tools/fit_paint.py`). Grazing reflectance is capped, as in the studio frames, and the key light leaves no hot spots in the lacquer.

Measured against the 52 studio masks, the exterior silhouette overlaps by 97.2% on average. The median boundary distance is 6.2 px and the 95th percentile 26 px at 3840 px width (`tools/silscore.py`). Per-frame statuses, notes and measurements are in `frame-review.json`.

## Known differences

The bumper shelf under the lid is softer than on the car. The outer intakes have thinner flaps and the central opening has no radiator grille. The fender around the fuel flap faces more sideways than the studio car. The rear reflectors, valance ribs and parking sensors are simplified, and the tyre lettering is larger than Pirelli's. The gearbox's internal tooth profiles, the backs of component castings and concealed pipe runs are inferred. The photographed sides of the engine and gearbox combine mesh detail with surface textures derived from Porsche’s illustrations.

The photo references and illustrations are credited to their respective publishers in `inventory.json`. The [Car_Photos catalog](https://github.com/BaiLiping/Car_Photos/tree/726ad07ff53ea2797bf83041ad73aba26ede6b18) includes third-party meshes of a 2019 Carrera 4S Cabriolet and a 2025 GT3. They were not used: the Turbo S Cabriolet body, roof and bumper geometry follow the Turbo S studio photographs.
