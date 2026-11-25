# Parts breakdown: Porsche 911 Turbo 3.3 (930), engine Type 930/60

Written before the geometry, kept current with it. `spec.py` holds the same numbers as code; where the two
disagree, `spec.py` is what the model was built from and this file is wrong.

**What is modelled:** a 1978 Europe-specification 930/60: air-cooled 3.3 litre flat-six, one KKK turbocharger,
intercooler, Bosch K-Jetronic, the twin-outlet muffler whose left pipe carries wastegate gas. The model is original
work. It was built from published dimensions and from photographs studied as reference; no drawing, scan or
photograph is reproduced here. It is unofficial and not affiliated with Porsche.

Tags: **[S]** a source (list at the end, or `research.json`), **[D]** derived by arithmetic, **[P]** read off a
photograph, **[E]** an estimate chosen to fit, **[M]** from memory and unverified.

## Axes, origin, units

- +X toward the front of the car (flywheel end), +Y the car's left (cylinders 1 to 3), +Z up.
- Origin on the crank axis at the centre of main bearing 4.
- Built in millimetres, exported in metres with +Y up (glTF).
- Head frame, used for everything that repeats per cylinder: `u` outward along the cylinder axis from the crank
  axis, `w` up, `x` along the crank. The right bank's frame is the left's turned half a turn about Z, so heads,
  valves, rockers, barrels, pistons and rods are one mesh each, instanced, as the real parts are.

## References studied

- Porsche Museum cutaway of a 930/60 with gearbox, three views on Wikimedia Commons (photographer's "left" is the
  car's right: the pulley end is that engine's front). Shows the air cleaner and K-Jetronic unit on the car's
  **right**, the engine-mounted oil cooler at the right front, the air pump at the left front driven by a toothed
  belt from the left camshaft, the muffler's two inner tubes and two outlets.
- A 1978 930 engine out of the car, seen from the rear (auction listing photograph): intercooler across the top,
  the charge pipe straight down the far left to the compressor, the distributor left of the fan, the engine
  carrier across behind the pulley, the turbocharger low at the left rear with its turbine outlet facing
  rearward, a large hose dropping down the right rear.
- The factory parts-catalogue illustration of the exhaust: two heat exchangers, a crossover, the pipe to the
  turbine with the wastegate branch, the wastegate, the muffler.
- A stock inlet manifold photographed on a bench: central throttle body, a transverse plenum, a flat log over
  each bank.

Where these contradicted the handover's memory notes, the photographs won: the K-Jetronic unit is on the right,
not the left, and the shroud is black on these engines.

## Decisions where sources disagree

| Item | Used | Why |
|---|---|---|
| Model year | 1978 Europe, 930/60 | It is the engine named in the brief and the one both photograph sets show. The muffler with a wastegate connection and two 44.5 mm tailpipes is listed for 1978 to 1982 [S]. |
| Cam timing at 1 mm | IO 3° ATDC, IC 37° ABDC, EO 27° BBDC, EC 5° BTDC | Of three quoted sets it is the only one that reproduces the cam grinder's figures: 214°/202° duration and 108° lobe separation [S][D]. The model then gives 0.81 mm intake lift at overlap TDC, which was not an input; the factory timing check is quoted as 0.65 to 0.80 mm, so it lands 0.01 mm above the top of that range. |
| Main journals | 60 mm | A workshop-manual figure quoted for this crank family and the tech wiki agree; one vendor table says 64 mm [S]. |
| Rod length | 127.0 mm | Maker's application guide [S]. With the 112.5 mm deck and 85.4 mm barrel it leaves 0.9 mm deck clearance; 127.75 mm would leave 0.15 mm. |
| Bore | 97.0 mm | 97.0 gives 3,298.8 cm³; one wiki prints 97.6 [S][D]. |
| Turbocharger wheels | 52.5/76.0 mm compressor (6+6 blades), 76.1/64.0 mm turbine (12 blades) | These are the later K27-7006's [S]. The 930/60's own unit is described as a KKK 3LDZ or K27 depending on the source; its wheel sizes were not found. |
| Fan ratio | 1.67 | Workshop manual [S]; some years are quoted at 1.8. |
| Shroud colour | black | Both photograph sets [P]; parts vendors sell a red one for other years. |

## Crank train

| Item | Value | Tag |
|---|---|---|
| Bore × stroke | 97.0 × 74.4 mm, 3,299 cm³ | [S] |
| Crank radius | 37.2 mm | [D] |
| Rod | 127.0 mm centres, 58 mm big-end bore, 22 mm journal width | [S] |
| Piston | compression height 32.8 mm, 23 mm pin, rings 1.2 / 1.2 / 3.0 mm, dished crown | [S], dish [E] |
| Journals | mains 60 mm, pins 55 mm, 7 mains and a nose bearing | [S] |
| Bore pitch, stagger | 118 mm, left bank 59 mm ahead | [S], stagger [D] from the throw order |
| Throw order from the flywheel | 1, 4, 2, 5, 3, 6 | [D] from "no. 1 nearest the flywheel" |
| Pin angles at θ = 0 | 1: 0°, 2: 120°, 3: 240°, 4: 180°, 5: 300°, 6: 60° (from +Y toward +Z) | [D] from the firing order 1-6-2-4-3-5 |
| Webs | 7.5 mm thick, each with a counterweight lobe | [D] from 59 mm pin pitch, 22 mm pin, 22 mm main |
| Deck, barrel | 112.5 mm crank axis to barrel seat, 85.4 mm barrel, head face at 197.9 mm | [S], [D] |
| Flywheel | 9 bolts, 135-tooth ring gear, 240 mm clutch | [S] |
| Rotation | clockwise seen from the pulley end, which is positive about +X | [S] |

Kinematics, one parameter θ (crank angle, 0 to 720°): piston pin at `r·cos β + sqrt(L² − r²·sin² β)` from the
crank axis, β the pin's angle from its own cylinder axis. Firing TDC: 1 at 0°, 6 at 120°, 2 at 240°, 4 at 360°,
3 at 480°, 5 at 600°.

## Crankcase, cylinders, heads

- Two halves split on the crank's vertical plane [S]. Each half is built from plates so that a cut shows a hollow
  case: top and bottom walls, a side wall bored for three 105 mm barrel spigots [S], seven bearing webs, end walls.
  Joint nuts along the split line; 11 M10 through-bolts [S] across the webs.
- The rear of the case is lower than the body so the fan housing can sit on it [E].
- Barrels: Nikasil bores, fins on the lower part, a plain band at the top, as the turbo's are described [S]. Fin
  outline flattened where neighbours sit 118 mm apart. Fin count [E].
- 24 head studs, four per cylinder at ±44 mm [S count, E spacing].
- Six separate heads [S]: domed chamber, intake port on top, exhaust port underneath, one plug [S]. Head height
  85 mm [M].

## Valve gear

| Item | Value | Tag |
|---|---|---|
| Valves | intake 49 mm head, 110.1 mm long; exhaust 41.5 mm, 108.4 mm; 9 mm stems | [S] |
| Valve angles from the cylinder axis | intake 25.5°, exhaust 30.25° | [S] forum measurements |
| Valve lift | 9.35 mm intake, 8.61 mm exhaust | [S] cam grinder's table |
| Rocker ratio | 1.4 | [S] |
| Rocker shafts | 18 mm, one per rocker | [S] |
| Springs | two per valve, 41 mm installed | [S] |
| Cam housing | one per bank, four 49 mm bearing bores | [S] |
| Cam axis | 329.5 mm from the crank axis | [D], below |
| Lift law between the checking points | raised cosine | placeholder, not the factory curve |

The camshaft's distance is not published. It follows from the chain: 98 links of 3/8 in pitch [S] round a 24-tooth
and a 28-tooth sprocket leave 340 mm between centres once the tensioner's idler has taken up the slack, and with
the intermediate shaft 84 mm below the crank [S secondary] that puts the cam 329.5 mm out. The same number gives an
engine about 770 mm across the valve covers, which is the figure usually quoted.

Each rocker's layout is solved in `spec.rocker_geom`: the pad arm square to the cam's push, the valve arm square to
the stem, arms in the ratio 1.4. The lobes are cut from the lift law and phased from the firing order, so the cam,
rocker and valve agree at every angle. Checked in the build: valves seated at firing TDC, 2.5 mm between piston and
valve at the closest approach, lobes clear the rocker shafts and pass through the bearing bores.

## Timing drive

- Crank gear 28 teeth, intermediate-shaft gear 48, shaft sprockets 24, cam sprockets 28, idlers 19 [S secondary]:
  (28/48) × (24/28) = 1/2. No idler gear between crank and intermediate shaft [S], so the shaft turns against the
  crank and the cams turn against the crank at half speed.
- The intermediate shaft lies behind the last connecting rod; a thin shaft carries its drive forward to the oil
  pump in the sump. A full-length shaft 84 mm below the crank would foul the rods [D].
- Two duplex chains at the pulley end, 27 mm apart on the shaft, the left one forward [E]. The slack run is on top
  on the left bank and underneath on the right, so the tensioners sit there [D] from the shaft's rotation.
- The chain path is solved in `spec.chain_path` so its length is exactly 98 pitches.
- Three guide ramps per chain, one brown [S for five black and one brown].

## Cooling

- Fan 245 mm, 11 blades [S], at 1.67 times crank speed [S]. Its height, 181 mm above the crank, follows from the
  710 mm belt and the two pulleys [D].
- Alternator inside the fan hub [S]. Housing strap zinc plated [S].
- Shroud: a moulded shell from the fan ring forward over both banks [P]. Its shape is a loft between stations
  chosen by eye.
- Engine oil cooler at the right front, where the stagger leaves room ahead of cylinder 4 [P].

## Induction and charge air

Air cleaner (right, top) → airflow sensor with the fuel distributor on it → inlet duct down the right rear and
across low behind the crankcase → compressor → charge pipe up the far left → intercooler (core 279 × 292 × 57 mm
[S]) → intermediate manifold, which carries the bypass valve [S] → throttle body (68 mm [S]), raised 35° [S] →
plenum → a log over each bank → six runners → ports. Six injectors at the runners with a steel line each [S].
A recirculation hose returns the bypass valve's air to the inlet duct [S].

The inlet duct's route is the least certain thing in the model: it is drawn where a large hose is seen in the
rear photograph, and its lower run was not seen at all [P][E].

## Turbocharger and exhaust

- Turbocharger low at the left rear, behind the left chain housing, rotor along the car, turbine rearward [P].
- Each bank's three ports feed a heat exchanger under the bank; the right one's outlet crosses behind the
  crankcase to join the left; one pipe rises to the turbine inlet and branches to the wastegate [S diagram].
- Wastegate: poppet valve under a spring and diaphragm, opening at 0.8 bar [S]. Its pipe enters the muffler and
  its gas leaves by the left tailpipe only [S].
- Muffler across the car behind the engine, two 44.5 mm tailpipes on the left [S].
- Turbo oil: feed from the case, drain to a small catch tank, scavenge pump on the left camshaft's front end [S].

## What moves

| Part | Law | Carried by |
|---|---|---|
| Crankshaft, flywheel, clutch, pulley | θ about +X | baked action |
| Rods, pistons | slider-crank | baked action |
| Camshafts and sprockets | −θ/2 | baked action |
| Rockers, valves, springs | from the lift law | baked action |
| Intermediate shaft, oil pump shaft | −28/48 θ | `spin` extra, driven by the page |
| Idler sprockets | −(28/48)(24/19) θ | `spin` extra |
| Fan | 1.67 θ | `spin` extra |
| Distributor rotor | θ/2 | `spin` extra |
| Chains | 14 pitches per crank turn along the exported path | `chains` in `engine930.json` |
| Turbo rotor | illustrative wheel speed | `spin` axis |
| Throttle plate, wastegate valve, bypass valve, airflow plate | fixed pose in the comparison | model includes `param` extras |

The baked action is one glTF animation of 181 linear keys, 4° apart, so any viewer can play the crank cycle.
The shafts whose ratio to the crank is not a whole number of turns in 720° cannot loop in a baked clip, so the
page turns them itself.

## Materials

Cast aluminium (case, heads, barrels, housings), machined aluminium (covers, pulleys, fan), steel (crank, rods,
cams, valves), dark steel (chain, springs, flywheel), zinc-plated hardware, cast iron (turbine housing,
wastegate), painted steel (exhaust), black paint (air cleaner, intercooler tanks, carrier, muffler), black
moulding (shroud), rubber (hoses, belts), brass (injectors, throttle plate), ceramic (plug insulators).

## Sources

`research.json` beside this file holds the 204 sourced facts gathered for the model, each with its URL and the kind
of source. The principal ones:

- [S1] ultimatespecs.com, Porsche 911 Coupe 3.3 Turbo (1978 to 1989): https://www.ultimatespecs.com/car-specs/Porsche/3553/Porsche-911-Coupe-(911-series)-33-Turbo.html
- [S2] auto-data.net, 911 Coupe (Type 930) 3.3 Turbo: https://www.auto-data.net/en/porsche-911-coupe-type-930-3.3-turbo-300hp-manual-5-speed-6654
- [S3] stuttcars.com, Porsche 911 Turbo 3.3 (930): https://stuttcars.com/porsche-911-turbo-3-3-930/
- [S4] supercarnostalgia.com, guide to the 911 3.3 Turbo (930): https://supercarnostalgia.com/blog/porsche-930-33
- [S5] elferspot.com, Porsche 930 Turbo 3.3 buyer's guide: https://www.elferspot.com/en/?p=1722792
- [S6] Forum threads on the wastegate and bypass valve: https://rennlist.com/forums/911-turbo-930-forum/820277-backdating-76-930-to-original-non-intercooled-setup-what-should-i-change.html (bypass valve; same wastegate on 3.0 and 3.3), https://rennlist.com/forums/911-turbo-930-forum/402107-k27-turbo-set-up.html (0.8 bar spring), https://forums.pelicanparts.com/911-930-turbo-super-charging-forum/453046-2years-turbo-install-runs-crappy-need-help-2.html (about 0.7 bar)
- [S7] Porsche Classic, 911 Turbo (G series): https://www.porsche.com/australia/accessoriesandservice/classic/models/911-g/911-turbo
- [S8] Hagerty, the air-cooled flat-six over three decades: https://www.hagerty.com/media/magazine-features/hagerty-magazine/porsches-air-cooled-flat-six-engine-thrived/
- [S9] Porsche Newsroom, "Heart and Soul" (boxer family, 118 mm spacing): https://newsroom.porsche.com/en/2020/technology/porsche-boxer-engine-911-turbo-s-christophorus-394-20340.html
- [S10] tech-racingcars wiki, Porsche 930: http://tech-racingcars.wikidot.com/porsche-930
- [S11] Rennlist, "Need 930 3.3L rods dimensions": https://rennlist.com/forums/911-turbo-930-forum/7554-need-930-3-3l-rods-dimensions.html
- [S12] carscounsel.com, Porsche 911 firing order: https://carscounsel.com/porsche-911-firing-order/
- [S13] Rennlist, "Plug Wire Question": https://rennlist.com/forums/911-forum/6085-plug-wire-question.html
- [S14] Pelican Parts, valves and guides for the 930 3.3: https://www.pelicanparts.com/byv-911-930-turbo-1978-3-3l/c-engine_c-cylinder-head-valvetrain_c-valves-guides
- [S15] Pelican forum, "Engine Fan Drive Ratio": https://forums.pelicanparts.com/porsche-911-technical-forum/170042-engine-fan-drive-ratio.html
- [S16] Pelican forum, "Engine cooling fan": https://forums.pelicanparts.com/porsche-911-technical-forum/428673-engine-cooling-fan.html
- [S17] Design 911, 930 air cooling parts: https://design911.co.uk/porsche/930--911--turbo-3-0l/air-cooling-parts
- [S18] A third-party 911 SC engine model in code, pull request 20 (tooth counts, 84 mm centres): https://github.com/josephschneider77-sys/porsche-911sc-engine/pull/20 The repository itself could not be opened from the cloud session. It models the naturally aspirated sibling of this engine and may be a useful cross-check. Read its licence before reusing anything from it.
- [S19] Stoddard, 911 camshaft drive and timing parts: https://www.stoddard.com/porsche-911/porsche-911-engine-early/timing.html
- [S20] LN Engineering, VW and Porsche journal diameters (conflicts with [S10]): https://docs.lnengineering.com/article/66-vw-and-porsche-main-and-rod-journal-diameters
- [S21] Porsche 911 Turbo workshop manual, 177 pages, on ManualsLib: https://www.manualslib.com/manual/392505/Porsche-911-Turbo-1989.html
- [S22] Rennlist, 934 race-engine data (not the road engine): https://rennlist.com/forums/911-turbo-930-forum/198608-934-and-934-5-photos-2.html
- [S23] stuttcars.com, 1991 911 Turbo 3.3 (964) technical specifications: https://www.stuttcars.com/?p=434528
- Wikimedia Commons, "Porsche 930-60 engine" at the Porsche Museum, three views (CC BY-SA 3.0, studied only): https://commons.wikimedia.org/wiki/File:Porsche_930-60_engine_left_Porsche_Museum.jpg
- Dougherty Racing Cams, Porsche profile table (stock 930 turbo cam: durations, lift, lobe separation), via research.json
- Mahle Motorsport application guide, Porsche 3.0 to 3.3 litre (compression height, pin, rings, rod length, spigot register), via research.json
- Method: https://github.com/BaiLiping/anymachine (PIPELINE.md, hardsurface/NOTES.md)
