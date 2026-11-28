# Porsche 911 Turbo S Cabriolet (992.2)

The Gentian Blue Metallic car uses the EU launch specification, 20/21-inch Turbo S centre-lock wheels, yellow PCCB brakes, a black textile roof and a left-hand-drive four-seat cabin. Porsche’s October 2025 MY T sheet identifies the 2026 model.

## Build

Extract the rebuild archive. From its root, run the following with Blender 5.2.1 and its bundled Draco exporter.

```sh
blender -b --python-exit-code 1 -P blender/build_master.py -- --out-prefix porsche_911_turbo_s_cab_refined
```

The build writes the editable metre-scale master and the viewer-scale GLB to `blender/out/`. In the master, +X points forward, +Y to the driver's left and +Z up. The origin is on the ground halfway between the axles. One GLB unit represents 18 real millimetres. Each assembly has a named pivot. The master includes both positions of the fabric hood.

The archive includes the photo-fitted displacement field, packed optical and mechanical finishes, and the component build scripts. The browser adds the studio illumination, registered cameras and assembly motions. The `viewer/` directory contains the studio shader maps used on the painted panels. The photo references and illustrations are credited to their respective publishers in `inventory.json`.

The gearbox's internal tooth profiles, the backs of component castings and concealed pipe runs are inferred. The photographed sides of the engine and gearbox combine mesh detail with surface textures derived from Porsche’s illustrations. Factory geometry checks and per-frame measurements are in `frame-review.json`.
