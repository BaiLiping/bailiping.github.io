# Notes on the 930/60 model

What the model is, where it is plainer than the engine, what was not built and which numbers are still unverified.
`PARTS.md` has the dimensions and their sources.

## What it is

One Blender script builds the whole engine from nothing: 547 parts, about 120,000 unique triangles (291,000
drawn, because barrels, heads, pistons, rods, valves, rockers, chain links and fasteners are instanced), a
3.8 MB GLB with the crank cycle baked into one animation. `spec.py` holds every dimension and motion law with no
Blender dependency; `kit.py` is a small builder kit after anymachine's `hs.py`; `engine_core.py` builds the
mechanism and `engine_dress.py` everything bolted round it.

```
blender -b --factory-startup --python build.py                 # writes ../engine930.glb and ../engine930.json
blender -b --factory-startup --python build.py -- --renders    # also check renders into _renders/
blender -b --factory-startup --python build.py -- --blend      # also engine930.blend (not committed)
```

The build fails if a valve is open at its cylinder's firing TDC, if a piston comes within 0.5 mm of a valve, if a
cam lobe would touch a rocker shaft or could not pass through the bearing bores, or if the triangle budget is
exceeded. It prints triangles per group and the bounding box.

## Checked against the references

Rear view against the 1978 engine photograph: the intercooler, the charge pipe down the far left, the air cleaner
and its snorkel, the fan, the distributor, the carrier and the turbocharger sit where the photograph has them.
Side and three-quarter views against the museum cutaway: the K-Jetronic unit, the oil cooler, the air pump and its
belt, the heat exchangers and the muffler agree in place; shapes are simpler.

## What differs from the real engine

- **Castings are plain.** The crankcase is assembled from flat plates, the heads are a domed ring and a block
  with fins, the cam housings are folded plates. Real castings have ribs, bosses, drillings and draft.
- **The fan shroud** is a smooth loft. The real moulding has ducts to each bank, cut-outs and the upper air
  guides; none of the engine tin under and round the cylinders is built.
- **The fan** has 11 flat pitched blades on a back plate; the real blades are curved and shrouded.
- **Fins**: the barrel's fin count and the heads' fin layout are estimates.
- **The turbocharger** uses the later K27-7006's wheel sizes and a generic volute. Blade shapes are flat plates.
- **The exhaust** is drawn from the parts-catalogue illustration and one product photograph; pipe diameters are
  estimates and the heat exchangers show a shell and three stubs, not three pipes inside a shell.
- **The inlet duct** follows a hose seen in one photograph; its lower run is a guess.
- **Pistons** have a simple dish. The real crown shape for 7.0 : 1 was not found.
- **Valve lift** between the published checking points is a raised cosine, not the factory lift curve.
- **Chain housings** are a wall round the chain path. The real ones are deeper castings with the tensioner and
  ramp mounts cast in.
- **Fuel lines and plug leads** run plausible routes, not the factory's.
- **Sizes of small hardware** follow thread size, not the parts catalogue.

## Not built

Engine tin and cylinder air baffles; oil lines, the S-hose and the oil tank (it is in the car's body); the
K-Jetronic's auxiliary air valve, cold-start valve, thermo-time switch and vacuum lines; the heater blower and its
hoses; the air-conditioning compressor; wiring; gaskets and seals; the starter and gearbox.

## Still unverified

- Which bank leads. The left bank is drawn 59 mm ahead, from "cylinder 1 is nearest the flywheel"; a top view or a
  crank drawing would settle it, and it moves every cylinder.
- Main journal diameter (60 mm used; one table says 64).
- The intermediate shaft's 84 mm drop and the tooth counts come from a secondary source.
- The camshaft's 329.5 mm distance is derived from the chain length, not measured.
- Head height, head stud spacing, fin counts, valve seat positions.
- The 930/60's own turbocharger wheel sizes.
- Whether a Europe-specification 1978 car carried the air injection pump; the museum's engine does.
- The engine's overall size: the model is 796 mm across the valve cover nuts and 588 mm from the flywheel flange to
  the back of the pulley; no dimensioned drawing was found to compare.

## Motion a plain viewer will not show

The GLB's animation carries the crank, rods, pistons, camshafts, rockers, valves and springs. The intermediate
shaft, idlers, fan, distributor rotor, chains, turbo rotor, throttle, wastegate and bypass valve are still in a
plain viewer; the page drives them from the `spin` and `param` extras on each node and the chain paths in
`engine930.json`.
