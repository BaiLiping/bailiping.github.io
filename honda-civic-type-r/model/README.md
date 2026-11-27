# 2020 Civic Type R (FK8): exterior rebuilt from Honda's references

The live model is `../honda_civic_type_r.glb`. The exterior panels were rebuilt from Honda’s references in November 2026 and fitted around the previous model’s running gear, brakes, engine, gearbox and cabin.

## References
- **Honda factory three-view (P24).** The October 2020 Japanese factbook vector drawing: side, top and front at 1 mm per pixel. Every closed area between Honda's lines was labelled as a region. Panels, lamps, openings and creases are taken from these regions and from the strokes between them.
- **American Honda's 2020 Boost Blue 360° colour-selector renders.** 36 frames, recovered from the Internet Archive, with a camera solved for each frame. Their alpha masks carve the car's volume. The rear frame supplies the rear-face layout and the traced lamp and vent outlines, since there is no rear drawing.
- **Honda's factbook studio photos** of the front and rear, for details the drawings do not show.

## How the body is made
1. **Volume.** The 36 masks form a visual hull, a signed-distance field on a 6 mm grid. It is intersected with the drawing's side, top and front outlines.
2. **Designed sections.** Some surfaces are never seen edge-on, so they are designed from the drawing's crease lines:
   - the door section, rear quarters, sill and skirt;
   - the valley-shaped bonnet with its scoop;
   - the C-pillars and tail;
   - the depth of every front opening.
3. **Mesh.** The field is meshed twice, at 12 mm for large panels and 6 mm for detailed ones. Vertices are snapped to the surface and normals come from the field, so reflections stay smooth.
4. **Panels.** Each panel is a combination of drawing regions in the view it faces, and is cut out of the mesh exactly along Honda's strokes. The shut lines therefore follow the drawing, with even gaps between panels.
5. **Parts and assembly.** Mirrors, wing, endplates, handles, honeycomb grilles, lamp layers, badges, rear wiper and vortex generators are built from their outlines in the drawing and from Honda's frames. Opened-state surfaces sit hidden behind the closed skin: door jambs, shut faces, sill plates, the hood underside and the hatch trim. Kept cabin and engine parts are fitted at least 4 mm inside the new skin.
6. **Surface refinement.** The finished mesh uses the roof pressing to form the cabin headliner. Cubic patches smooth the glazing, while the cut glass perimeters stay against their seals. The hood ridges, two rows of headlamp optics and recessed bumper vents follow the front three-quarter reference. The rear fascia uses four diffuser fins and three tapered exhaust outlets fitted to the rear and low-angle views. The paint uses a blue dielectric base with a clear coat.

## Match to Honda's renders
The original visual-hull build recorded these silhouette overlaps against Honda’s alpha masks, using the solved cameras and intersection over union (IoU).
- **Frame 01** (driver side): 0.968
- **Frame 10** (front): 0.958
- **Frame 19** (passenger side): 0.966
- **Frame 28** (rear): 0.955

The frame-by-frame page shows every comparison. The model is about 7 MB with Draco compression.

## Known differences
- Taillamp inner cells and the rear vent detail are simpler than Honda's.
- Faint waviness remains on the hatch near the lamps and on the quarter behind the fuel door.
- The opened hood's inner panel is plain, without ribs.
- Honda's renders place the wheels and arches about 2–3 cm further forward than the drawing's axle positions. The model follows the drawing.

## Rebuild
The build scripts are in `rebuild/`. Honda's drawing and renders are not included, so the scripts document the method rather than run on their own.

`rebuild/build.sh <tag>` builds the body, assembles the rig, compresses it with `compress_model.mjs` and runs the final surface refinement. To repeat the last step with an unrefined compressed assembly, run `python rebuild/refine_exterior.py <assembly.glb> <output.glb>` from this directory with NumPy, SciPy and DracoPy 2.2 or newer.

To reproduce the published refinement from the archived assembly without the original drawings, run these commands at the repository root.

```sh
git show b436dba528:honda-civic-type-r/honda_civic_type_r.glb > /tmp/fk8-assembly.glb
python honda-civic-type-r/model/rebuild/refine_exterior.py /tmp/fk8-assembly.glb /tmp/fk8-refined.glb
```

The viewer uses a separate pair of ambient visibility values per vertex for the closed and open assemblies. To regenerate it from the published GLB, install NumPy, SciPy, DracoPy and Mitsuba 3 and run the following from the viewer directory. Add `--variant cuda_ad_rgb` to use an NVIDIA GPU.

```sh
python model/rebuild/bake_visibility.py honda_civic_type_r.glb studio-visibility.bin.gz
```
