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
3. **Mesh.** The body starts as a single surface sampled at 12 mm. Edges near fine details are bisected before the panels are cut. Both triangles that meet at an edge share its midpoint.
4. **Panels.** Honda's drawing regions divide the surface into panels. Dark outlines follow the side door openings, window seals and fuel flap. The hood ridges displace all adjoining pieces together. A single curve shared by the body, the hatch and the weatherstrip defines the 3.6 mm hatch shut line. A depth cut keeps the rear wheelhouses on the body when the hatch lifts.
5. **Parts and assembly.** Mirrors, wing, endplates, handles, honeycomb grilles, lamp layers, badges, rear wiper and vortex generators are built from their outlines in the drawing and from Honda's frames. Opened-state surfaces sit hidden behind the closed skin: door jambs, shut faces, sill plates, the hood underside and the hatch trim. Kept cabin and engine parts are fitted at least 4 mm inside the new skin.
6. **Surface refinement.** The finished mesh uses the roof pressing to form the cabin headliner. Cubic patches smooth the glazing, while the cut glass perimeters stay against their seals. The hood ridges, two rows of headlamp optics and recessed bumper vents follow the front three-quarter reference. The rear fascia uses four diffuser fins and three tapered exhaust outlets fitted to the rear and low-angle views. Each door grip has two feet and a painted finger cup fitted to the door surface. The rear reflectors have a solid lens on each bumper insert. The painted rail above each front fog lamp follows the factory drawing and stands clear of its black backing. The paint uses a blue dielectric base with a clear coat.

## Moving panels

The rear opening has a continuous painted flange, a rubber seal, a sill plate and a carpeted load floor. The wing, upper spoiler, glass and inner lamp segments follow the hatch through its 88° travel. Two telescoping supports aim at the body and hatch sockets throughout the motion. A 2021 U.S. FK8 in Boost Blue supplies the open-hatch reference. Its rear seatbacks are folded. [Matched open-hatch photograph and model](https://bailiping.com/honda-civic-type-r/frame-by-frame/#photo-fk8-open-hatch)

The hood seam follows Honda’s P24 plan across the painted strip ahead of the bonnet and along the upper edges of both headlamps. The bumper garnish and lights stay on the body. The moving skin folds into a continuous inner pressing, with four recesses, a crosswise scoop air guide and the latch. A pair of cowl hinges carries the panel through approximately 52° of travel. The passenger-side support rod swings out of its clip and reaches a socket in the hood. The open angle and hinge depth are estimated from photographs of a 2021 Boost Blue FK8. [Front photograph and model](https://bailiping.com/honda-civic-type-r/frame-by-frame/#photo-fk8-open-hood) · [Side photograph](https://bringatrailer.com/wp-content/uploads/2023/05/untitled-98-09851-scaled.jpg) · [Honda hood operation](https://owners.honda.com/utility/download?path=%2Fstatic%2Fpdfs%2F2020%2FCivic+Hatchback%2F2020_Civic_5D_Hood_Release.pdf)

Honda's accessory installation drawing locates the guide and its fasteners. A photograph of a bare FK8 hood supplies the pressing outline. [American Honda installation drawing](https://www.bernardiparts.com/Images/Install/2020_CivicTypeR_AirIntakeTrim_AII11806-113.pdf) · [Redline Auto Parts FK8 hood](https://www.redlineautoparts.com/honda/2017-2021-civic-type-r/2017-2021-fk8-honda-civic-type-r-oem-hood-panel-w-scoop-tr105/)

## Match to Honda's renders
The original visual-hull build recorded these silhouette overlaps against Honda’s alpha masks, using the solved cameras and intersection over union (IoU).
- **Frame 01** (driver side): 0.968
- **Frame 10** (front): 0.958
- **Frame 19** (passenger side): 0.966
- **Frame 28** (rear): 0.955

The public viewer includes 30 matched views. The final compressed model was checked against all 36 original studio frames at 1920 × 1080. With the studio floor shadow removed, the mean silhouette overlap is 0.948 and the mean distance between boundaries is 4.6 pixels. The model measures 4564.6 mm long, 1876.0 mm wide across the body and 1456.4 mm high including the roof antenna. The corresponding U.S. factory dimensions are 4556.8, 1877.1 and 1435.1 mm. The model’s wheelbase matches the 2700 mm factory figure. [Honda U.S. specifications](https://hondanews.com/en-US/honda-automobiles/releases/release-9349200e0db74014eece042d0a003399-2020-civic-type-r-specifications-features) · [Honda Japan three-view drawing](https://www.honda.co.jp/factbook/auto/CIVIC_TYPE_R/202010/P24.pdf)

## Known differences
- Taillamp inner cells and the rear vent detail are simpler than Honda's.
- Faint waviness remains on the hatch near the lamps and on the quarter behind the fuel door.
- The hood pressing depth, rear seatbacks, carpet, cargo-side pockets and small engine-bay fittings are approximate.
- Honda's renders place the wheels and arches about 2–3 cm further forward than the drawing's axle positions. The model follows the drawing.

## Rebuild
The build scripts are in `rebuild/`. Honda's drawing and renders are not included, so the scripts document the method rather than run on their own.

`rebuild/build.sh <tag>` builds the body, assembles the rig, compresses it with `compress_model.mjs` and runs the final surface refinement. To repeat the last step with an unrefined compressed assembly, run `python rebuild/refine_exterior.py <assembly.glb> <output.glb>` from this directory with NumPy, SciPy, OpenCV and DracoPy 2.2 or newer.

Rebuilding the panel edges requires the body field and Honda drawing masks used by `cutpanels.py`. The compressed assembly passed to the final refinement must already contain those panel cuts.

The viewer uses a separate pair of ambient visibility values per vertex for the closed and open assemblies. To regenerate it from the published GLB, install NumPy, SciPy, DracoPy and Mitsuba 3 and run the following from the viewer directory. Add `--variant cuda_ad_rgb` to use an NVIDIA GPU.

```sh
python model/rebuild/bake_visibility.py honda_civic_type_r.glb studio-visibility.bin.gz
```
