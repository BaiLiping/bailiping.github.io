"""Writes cut_rear.json and parts_rear.json (the rear worker's panel and part specs).

Run: python3 rear_specs.py   (in the rebuild-rear directory)
Rear-facing surfaces are classified with Honda's rear frame 28 colour classes (h28 terms); side-facing ones with
the P24 side-view regions. Order = priority. Every group carries an x limit (XR or tighter), since h28 and view terms
extrude through the whole car. Thin features use the fine (step-1) mesh ("mesh": "fine", sides_run.sh / rear_run.sh).
"""
import json

FR = ['face', 'rear', 0.2]
FS = ['face', 'side', 0.05]
XR = ['box', 'x', -3.0, -1.75]          # every rear term: h28 and side-view terms extrude through the whole car
# hatch bottom edge (frame 28 shading): horizontal at y 0.665 across |z| < 0.42, then a diagonal up to the lamp
# bottoms (|z| 0.58, y 0.778). The panel follows it as a staircase of 4 mm boxes on the fine mesh; the visible shut
# line is a modelled ribbon (rear_parts.py hatch_shutline), so it stays continuous and even
HATCH_DIAG = [[FR, XR, ['box', 'az', 0.414 + 0.004 * k, 0.62], ['box', 'y', 0.665 + 0.00283 * k, 1.0]] for k in range(41)]

cut = [
    # ---- taillamps, badges, vent mesh and reflectors: modelled parts (rear_parts.py, rear_badges.py)
    # ---- hatch: lower glass under the lip, black spoiler bar (rear face, top strip 30, side ends), panel
    {'name': 'Civic_Hatch_Lower_Glass', 'mat': 'privacy', 'face': 'none', 'pivot': 'Pivot_Hatch_Assembly', 'mesh': 'fine', 'any': [
        [FR, XR, ['h28', ['dark'], 'neg'], ['box', 'y', 0.995, 1.098], ['box', 'az', 0, 0.48]],
        [['h28', ['clear']], FR, XR, ['box', 'y', 0.99, 1.10], ['box', 'az', 0, 0.56]],
    ]},
    {'name': 'Civic_Hatch_Bar', 'mat': 'gloss_black', 'face': 'none', 'pivot': 'Pivot_Hatch_Assembly', 'mesh': 'fine', 'any': [
        [['h28', ['paint'], 'neg'], ['face', 'rear', 0.1], ['box', 'y', 0.99, 1.16], ['box', 'az', 0, 0.70], ['box', 'x', -2.4, -1.9]],
        [['top', [30]], ['face', 'top', 0.3], ['box', 'y', 1.0, 1.2], XR],
        [['box', 'x', -2.4, -1.955], ['box', 'y', 1.095, 1.2], ['box', 'az', 0, 0.66]],
        [['side', [96, 121, 133, 152]], FS, XR],
    ]},
    {'name': 'Civic_Hatch_Panel', 'mat': 'paint', 'face': 'none', 'pivot': 'Pivot_Hatch_Assembly', 'nsmooth': 4.0,
     'mesh': 'fine', 'any': [
        [FR, XR, ['box', 'y', 0.665, 1.0], ['box', 'az', 0, 0.418]],
        [['side', [32]], FS, XR],
    ] + HATCH_DIAG},
    # ---- bumper: red pinstripe along the sides, the central diffuser (frame 28), paint; the corner vents, their
    # black trim, the lower band at the corners and the pinstripe across the back are modelled skins (rear_parts.py)
    # red pinstripe: across the back a modelled ribbon (rear_parts.py); along the sides the side-view strip 117
    {'name': 'Civic_Rear_Bumper_Red_Strips', 'mat': 'red', 'face': 'none', 'mesh': 'fine', 'any': [
        [['side', [117]], FS, XR],
    ]},
    {'name': 'Civic_Rear_Diffuser', 'mat': 'black', 'face': 'none', 'mesh': 'fine', 'any': [
        [['h28', ['dark']], ['face', 'rear', 0.0], XR, ['box', 'y', 0.05, 0.56], ['box', 'az', 0, 0.42]],
        [['box', 'x', -2.4, -2.05], ['box', 'y', 0.20, 0.40], ['box', 'az', 0, 0.25]],
        [['box', 'x', -2.4, -1.9], ['box', 'y', 0.05, 0.27], ['box', 'az', 0, 0.80]],
        [['side', [82, 100, 116]], FS, XR],
    ]},
    {'name': 'Civic_Rear_Bumper', 'mat': 'paint', 'face': 'none', 'nsmooth': 4.0, 'mesh': 'fine', 'any': [
        [['face', 'rear', 0.0], ['box', 'y', 0.15, 0.86], ['box', 'x', -2.4, -1.85]],
        [['side', [10, 53, 81, 103, 97, 26]], FS, ['box', 'x', -3.0, -1.50]],
    ]},
    # ---- everything else on the tail that is not down-facing: paint (no stray seam specks); the down-facing
    # step of the plate recess and the underbody stay dark seams
    {'name': 'Civic_Rear_Body_Fill', 'mat': 'paint', 'face': 'none', 'mesh': 'fine', 'any': [
        [['box', 'x', -2.4, -1.75], ['box', 'y', 0.28, 1.35], ['face', 'top', -0.4]],
        [['box', 'x', -2.4, -1.62], ['box', 'y', 0.80, 1.12], ['face', 'top', -0.4], ['box', 'az', 0.55, 1.0]],
    ]},
]

# the wing, endplates, pylons and the lamp skins are built by rear_parts.py (analytic outlines, no raster steps),
# the badges by rear_badges.py
parts = [
    {'name': 'Civic_Roof_SharkFin', 'side': [86], 'top': [88], 'mat': 'paint', 'k': 0.004, 'h': 0.002, 'mirror': False},
]

json.dump(cut, open('cut_rear.json', 'w'), indent=1)
json.dump(parts, open('parts_rear.json', 'w'), indent=1)
print('wrote cut_rear.json', len(cut), 'specs; parts_rear.json', len(parts), 'parts')
