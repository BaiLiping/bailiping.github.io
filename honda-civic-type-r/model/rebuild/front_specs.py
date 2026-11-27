"""Generate the front worker's panel and part specs: cut_front.json and parts_front.json.

Run: venv/bin/python front_specs.py            (writes both files next to this script)

Front-view ids are the image-left (passenger, +Z) half of Honda's P24 front plate; cutpanels mirrors them.
Region roles (checked against Honda frames 04/07/10/13/16 and the JP factbook front studio photo):
  brow bar 11 (+ top-view brow top 23/35/45, side 80/104), grille band 19, slot band 47 with slots
  72/76/85/86 .., honeycomb 10, black chin 22, wings 41, inner lamp corner 78           -> gloss black / grille
  headlamp: lens 17 / side 41 (reflector), upper housing 38 / side 73 / top 63,74 (black bowl),
            LED cubes 112-116 and a light-guide strip along the 38|17 line (parts, bright)
  bumper paint: 7, 30, 84, the L band 28 + 89 around the pod, upper part of plate box 6 (US car: no plate),
            side 16, 30, 56, 112, 123, 146; top 28, 61, 110, 104, 96
  fog pod 12, 25, 53, 75 / side 88, 45, 76 (gloss black), blade 54 / side 114 (paint), fog lamp 60,
            vents 100-109 / side 163-167
  lower grille 4 + lower part of 6 (honeycomb), splitter 9, 14, 44, 34, 63, 66, 98, 119, 124, 125 /
            side 85, 89, 120, 122, 138, 142, 154, 160, 165, 39, 64 (+ its top face), red pinstripe along
            the splitter's lower front edge, side marker 91
Every front/top term is limited to x > XMIN so that extruded regions never claim surfaces further back.
"""
import json
from pathlib import Path
import numpy as np
import p24, lines as L

HERE = Path(__file__).resolve().parent
XMIN = 1.60
PLATE_SPLIT_Y = 0.385          # US car: body-colour band over the top of plate box 6, honeycomb below (30|4 line)
STRIPE_Y = (0.2065, 0.2165)    # red pinstripe band on the splitter's lower front edge
SPLITTER_Y = (0.195, 0.262)    # splitter shelf (its top face is seen by no view)
LENS_RIM, LENS_LOW, LENS_LOW_F = 0.004, 0.022, 0.045   # headlamp: dark rim around the bright LED optics, and the
    # smoked lower band of the lens above the bumper, measured in the side / front view (m; Honda frame 10: the
    # bright LED strip is the upper ~40% of the lens seen from ahead)
HOOD_LIP_Y = 0.727             # front region 11 above this is the hood's painted front lip (Honda 10/13: blue
                               # down to 0.73 at |z| 0.15-0.35); below it the dark gap over the brow bar 47
HOOD_FINE_X = 2.03             # hood surface ahead of this is cut from the fine mesh with the fascia
CHIN_Y = 0.503                 # black grille chin ends here also across plate box 6 (Honda 10/13: dark to 0.506)


def F(ids, face=0.55, xmin=XMIN):
    """Front-view footprint on surface facing forward (face = min N.x; negative also takes opening walls)."""
    return [['front', ids], ['face', 'front', face], ['box', 'x', xmin, 3.0]]


def S(ids, face=0.55, xmin=1.45):
    """Side-view footprint on surface facing outboard (|N.z| >= face)."""
    return [['side', ids], ['face', 'side', face], ['box', 'x', xmin, 3.0]]


def T(ids, front_ids, face=0.55, xmin=XMIN):
    """Top-view footprint on upward surface, kept inside a front-view footprint too (the plan alone also hits
    other upward surfaces at other heights)."""
    return [['top', ids], ['front', front_ids], ['face', 'top', face], ['box', 'x', xmin, 3.0]]


def lr(base):
    return [base + '_L', base + '_R']


LAMP_F = [17, 38, 78, 112, 114, 116]
# smoked lower band of the lens: away from the bumper below it in BOTH views (wider toward the outboard tail)
LOW_BAND = [['front', [7], 'neg', LENS_LOW_F], ['side', [16], 'neg', LENS_LOW]]
LAMP_S = [41, 73, 131, 124, 147, 148, 150, 155, 162]
BROW_F = [11, 47, 19, 72, 76, 85, 86]   # the brow bar is 47 (+ the lower edge of 11)
BUMPER_F = [7, 30, 84, 28, 89]
# large plain panels stay on the coarse mesh (the grille floors are hidden by the honeycomb meshes)
COARSE = {'Civic_Hood_Panel', 'Civic_Front_Grille', 'Civic_Front_Grille_Lower', 'Civic_Front_Splitter',
          'Civic_Front_Bumper_Fog_Surrounds'}
FILL_NEAR = [4, 9, 10, 11, 12, 14, 19, 22, 25, 34, 41, 44, 46, 47, 53, 54, 60, 63, 66, 72, 75, 76, 85, 86, 98,
             119, 124, 125]          # the black fascia regions (not the pod-side vents, not the plate box)
NOT_FASCIA = [1, 3, 5, 15, 17, 20, 31, 38, 112, 114, 116]   # windscreen, hood, fender, lamp (front view)


def cut_specs():
    G = []
    G.append({'names': lr('Civic_Front_Fog_Lights'), 'any': [F([60], -0.3, 1.93)], 'face': 'none', 'mat': 'Mat_FK8_Fog_Satin'})
    G.append({'name': 'Civic_Front_Grille_Emblem_Base', 'any': [F([46], -0.3, 2.05)], 'face': 'none', 'mat': 'gloss_black'})
    # lamp: front-facing surface by the front view, outboard-facing by the side view, and oblique surface where
    # both views agree
    # lamp internals as panels on the fine body mesh (smooth field normals, no follow-skin facets): LED cubes
    # (their drawn outlines stay dark), the bright LED optics filling the lens (LENS_RIM in from its outline, LENS_LOW above the bumper in
    # both views: the smoked lower band, wider toward the outboard tail) and the dark bowl for the rest
    G.append({'names': lr('Civic_Headlamp_LED'), 'any': [F([112, 114, 116], 0.2)], 'face': 'none', 'mat': 'fog'})
    G.append({'names': lr('Civic_Headlamp_Optics_Cells'), 'any': led_dividers(), 'face': 'none',   # dividers + taper
              'mat': 'Mat_Headlamp_Black_Bowl', 'grow': 0.0})
    G.append({'names': lr('Civic_Headlamp_Optics'),
              'any': [g + LOW_BAND for g in (F([17], 0.2) + [['front', [17], 'pos', -LENS_RIM]],
                                             S([41], 0.2) + [['side', [41], 'pos', -LENS_RIM]],
                                             [['front', [17], 'pos', -LENS_RIM], ['side', [41], 'pos', -LENS_RIM],
                                              ['box', 'x', XMIN, 3.0]])],
              'face': 'none', 'mat': 'fog'})
    G.append({'names': lr('Civic_Headlamp_Reflector'),
              'any': [F([17]), S([41]), [['front', [17]], ['side', [41]], ['box', 'x', XMIN, 3.0]]], 'face': 'none',
              'mat': 'Mat_Headlamp_Black_Bowl'})
    G.append({'names': lr('Civic_Headlamp_Housing'),
              'any': [F([38, 78, 112, 114, 116]), S(LAMP_S), T([63, 74, 159, 193], LAMP_F),
                      [['front', LAMP_F], ['side', LAMP_S], ['box', 'x', XMIN, 3.0]]],
              'face': 'none', 'mat': 'Mat_Headlamp_Black_Bowl'})
    G.append({'names': lr('Civic_Front_Bumper_Vents'), 'any': vent_slats(), 'face': 'none', 'mat': 'seam', 'grow': 0.0})
    G.append({'name': 'Civic_Hood_Scoop_Mouth', 'any': [F([45, 82], -0.7, 1.55) + [['box', 'x', 1.55, 1.75]]], 'face': 'none',
              'mat': 'seam', 'pivot': 'Pivot_Hood_Assembly'})
    G.append({'names': lr('Civic_Front_Bumper_Side_Marker'), 'any': [S([91])], 'face': 'none', 'mat': 'Mat_Side_Marker_Amber'})
    G.append({'name': 'Civic_Front_Grille_Lower',
              'any': [F([4, 6], -0.5, 2.0) + [['box', 'y', 0.0, PLATE_SPLIT_Y]], F([4], -0.5, 2.0)], 'face': 'none', 'mat': 'seam'})
    G.append({'names': lr('Civic_Front_Bumper_Fog_Surrounds'), 'any': [F([12, 25, 53, 54, 60, 75], -0.5, 1.93) + [['box', 'az', 0.0, 0.80]],
                      F([12, 25, 53, 54, 60, 75], -0.5, 1.93) + [['box', 'az', 0.80, 0.835], ['box', 'y', 0.0, 0.538]],
                      F([12, 25, 53, 54, 60, 75], -0.5, 1.93) + [['box', 'az', 0.835, 1.0], ['face', 'side', 0.6, 'neg']]],
              'face': 'none', 'mat': 'Mat_FK8_Fog_Satin'})
    G.append({'names': lr('Civic_Front_Bumper_Upper'),
              'any': [F(BUMPER_F, 0.25), F([30, 84, 6, 22], 0.25) + [['box', 'y', PLATE_SPLIT_Y - 0.004, CHIN_Y]],
                      S([16, 30, 56, 112, 123, 146, 114, 45, 163, 164, 166, 167, 88, 76], 0.45), T([28, 61, 110, 104, 96], BUMPER_F + [6])],
              'face': 'none', 'mat': 'paint'})
    G.append({'name': 'Civic_Front_Grille', 'any': [F([10], -0.5, 2.0)], 'face': 'none', 'mat': 'seam'})
    G.append({'name': 'Civic_Front_Grille_Slots', 'any': [F([72, 76, 85, 86], -0.5, 1.95)], 'face': 'none', 'mat': 'seam'})
    G.append({'name': 'Civic_Front_Grille_Frame',
              'any': [F([19, 47, 41], -0.5, 1.95), F([11, 47], -0.5, 1.95) + [['box', 'y', 0.0, HOOD_LIP_Y]],
                      F([22, 6, 41], -0.5, 1.95) + [['box', 'y', CHIN_Y, 1.0]], S([80, 104, 118]), T([35, 45], BROW_F)],
              'face': 'none', 'mat': 'gloss_black'})
    G.append({'name': 'Civic_Front_Splitter_Red_Line',
              'any': [[['box', 'y', STRIPE_Y[0], STRIPE_Y[1]], ['box', 'x', 1.74, 3.0], ['box', 'az', 0.0, 0.93],
                       ['face', 'top', -0.75]]], 'face': 'none', 'mat': 'red', 'grow': 0.0})
    G.append({'name': 'Civic_Front_Splitter',
              'any': [F([9, 14, 44, 34, 63, 66, 98, 119, 124, 125], -0.3) + [['face', 'top', -0.6]],
                      S([85, 89, 120, 122, 138, 142, 154, 160, 165, 39, 64]),
                      S([39, 64, 120, 85], 0.05) + [['face', 'top', -0.6], ['box', 'az', 0.72, 1.0]],
                      [['box', 'y', SPLITTER_Y[0], SPLITTER_Y[1]], ['box', 'x', 1.95, 3.0], ['box', 'az', 0.0, 0.93],
                       ['face', 'top', -0.6]]],
              'face': 'none', 'mat': 'carbon'})
    # fill: whatever is still unclaimed on the fascia below the hood lip (the drawn strokes between colour regions,
    # ledges of the openings that project onto a stroke) goes dark here instead of to the catch-all, which paints
    # faces that look up. 'neg' of the non-fascia regions covers every stroke, however wide it is drawn.
    G.append({'name': 'Civic_Front_Fascia_Trim_Fill',
              'any': [[['front', NOT_FASCIA, 'neg'], ['front', FILL_NEAR, 'pos', 0.010], ['face', 'front', -0.6],
                       ['face', 'top', -0.5], ['box', 'x', 1.93, 3.0],
                       ['box', 'y', 0.19, HOOD_LIP_Y], ['box', 'az', 0.0, 0.87]]],
              'face': 'none', 'mat': 'Mat_FK8_Fog_Satin'})
    G.append({'names': lr('Civic_Body_Shell_FrontFender_Tip'), 'any': [F([15], 0.3, 1.75)], 'face': 'none', 'mat': 'paint'})
    # hood: its plan regions, plus the painted front lip (front region 11 above HOOD_LIP_Y, and the plan's 23
    # strip ahead of the 3|23 crease)
    G.append({'name': 'Civic_Hood_Panel_Front', 'mesh': 'fine',
              'any': [[['front', [11]], ['box', 'y', HOOD_LIP_Y, 1.0], ['face', 'front', -0.3], ['box', 'x', 1.9, 3.0]],
                      [['top', [23]], ['front', [3, 11]], ['box', 'y', HOOD_LIP_Y, 1.0], ['face', 'top', 0.0],
                       ['box', 'x', 1.9, 3.0]]],
              'face': 'none', 'mat': 'paint', 'pivot': 'Pivot_Hood_Assembly'})
    G.append({'name': 'Civic_Hood_Panel',
              'any': [[['top', [3, 9, 13, 24, 25, 106, 123]], ['face', 'top', 0.15]],
                      [['front', [11]], ['box', 'y', HOOD_LIP_Y, 1.0], ['face', 'front', -0.3], ['box', 'x', 1.9, 3.0]],
                      [['top', [3, 23]], ['front', [3, 11]], ['box', 'y', HOOD_LIP_Y, 1.0], ['face', 'top', 0.0],
                       ['box', 'x', 1.9, 3.0]]],
              'face': 'none', 'mat': 'paint', 'pivot': 'Pivot_Hood_Assembly'})
    for g in G:                      # the fascia is cut from the fine (6 mm) body mesh when the build provides one
        if g.get('name', '') not in COARSE and 'Bumper_Upper' not in str(g.get('names', '')) and \
                not any(n[:-2] in COARSE for n in g.get('names', [])):
            g['mesh'] = 'fine'                         # (the hood's front strip is a fine spec of its own)
        if 'Bumper_Upper' in str(g.get('names', '')):
            g['close'] = 15          # swallow the small strokes inside the painted bumper (vent and blade outlines)
            g['grow'] = 0.003        # and reach into the strokes it shares with the black trim (even edges)
        if 'Headlamp' in str(g.get('names', '')):
            g['close'] = 25          # the stroked box around the LED cubes belongs to the lamp
        if 'Fill' not in str(g.get('name', '')) and 'grow' not in g:
            g['grow'] = 0.003        # fine panels reach 3 mm into Honda's strokes: the coarse seam left between
                                     # two of them is a thin, even shut line instead of a ragged 12 mm band
    return G


VENT_PAIRS = [(109, 167), (103, 164), (102, 163), (100, 166)]   # front-view slot, side-view slot of each slat


def vent_slats():
    """The four pod-side vents as slanted slats wrapping the bumper corner: for each, a band between two
    side-view lines through the slot's lower/upper edge where the front view draws it (at the envelope's x there)
    and where the side view draws it, kept between the front slot's inner end and the side slot's rear end."""
    labf, labs = p24.labels('front'), p24.labels('side')
    groups = []
    for fr, sr in VENT_PAIRS:
        vv, uu = np.nonzero(labf == fr); keep = uu < 1150; vv, uu = vv[keep], uu[keep]
        zf_in, zf_out = 1.15 - uu.max() / 1000, 1.15 - uu.min() / 1000
        yf_lo, yf_hi = 1.65 - vv.max() / 1000, 1.65 - vv.min() / 1000
        xf = float(env_front_x([0.5 * (yf_lo + yf_hi)], [0.5 * (zf_in + zf_out)])[0])
        vv, uu = np.nonzero(labs == sr)
        xs_rear, xs_c = 2.5 - uu.max() / 1000, 2.5 - uu.mean() / 1000
        ys_lo, ys_hi = 1.65 - vv.max() / 1000, 1.65 - vv.min() / 1000
        m_lo = (ys_lo - yf_lo) / (xs_c - xf); m_hi = (ys_hi - yf_hi) / (xs_c - xf)
        yf_lo, yf_hi, ys_lo, ys_hi = yf_lo + 0.002, yf_hi - 0.002, ys_lo + 0.003, ys_hi - 0.003     # slats a touch thinner
        m_lo = (ys_lo - yf_lo) / (xs_c - xf); m_hi = (ys_hi - yf_hi) / (xs_c - xf)
        groups.append([['hp', round(m_lo, 4), -1.0, round(yf_lo - m_lo * xf, 5)],
                       ['hp', round(-m_hi, 4), 1.0, round(-(yf_hi - m_hi * xf), 5)],
                       ['box', 'az', round(zf_in - 0.004, 4), 0.95], ['box', 'x', round(xs_rear - 0.006, 4), 2.10],
                       ['face', 'front', -0.7]])
    return groups


def band_poly(P, a0, a1, d0, d1, n=40):
    """Points (a, b) of a line; returns a closed polygon [[a, b - d0] ... [a, b - d1] reversed] over a0..a1."""
    o = np.argsort(P[:, 0]); a = np.linspace(a0, a1, n); b = np.interp(a, P[o, 0], P[o, 1])
    top = np.stack([a, b - d0], 1); bot = np.stack([a, b - d1], 1)[::-1]
    return np.vstack([top, bot]).round(4).tolist()


def env_front_x(ys, zs):
    """Frontmost x of the envelope field at points (y, z)."""
    d = np.load(HERE / 'env6.npz'); Fe = d['F']; x0, y0, z0, h = (float(d[k]) for k in ('x0', 'y0', 'z0', 'h'))
    xs = x0 + h * np.arange(Fe.shape[0]); out = []
    for y, z in zip(ys, zs):
        j = int(round((y - y0) / h)); k = int(round((z - z0) / h)); col = Fe[:, j, k]; i = np.nonzero(col < 0)[0].max()
        out.append(xs[i] + h * col[i] / (col[i] - col[i + 1]))
    return np.array(out)


def blade_profile():
    """Side-view band for the painted pod blade (front region 54): along the blade the envelope face recedes
    from x 2.20 (inner end) to 2.07 (outer end); the blade face sits 4 mm behind it and is 30 mm deep."""
    lab = p24.labels('front')
    ys = np.linspace(0.426, 0.530, 14); zc = []
    for y in ys:
        r = int(round((1.65 - y) * 1000)); c = np.nonzero(lab[r - 3:r + 4, :1150] == 54)[1]
        zc.append(1.15 - c.mean() / 1000 if len(c) else np.nan)
    zc = np.array(zc); ok = np.isfinite(zc); zc = np.interp(ys, ys[ok], zc[ok])
    xe = env_front_x(ys, zc)
    ys = np.concatenate([[ys[0] - 0.012], ys, [ys[-1] + 0.012]]); xe = np.concatenate([[xe[0]], xe, [xe[-1]]])
    front = np.stack([xe - 0.004, ys], 1); back = np.stack([xe - 0.034, ys], 1)[::-1]
    return np.vstack([front, back]).round(4).tolist()


def lattice_profile(mask, pu=48, pv=28, t=6, inset=5):
    """Honeycomb (diamond mesh) as ONE polygon for parts3v's front_profile: the opening's outline with the
    diamond holes linked in by zero-width bridges (cv2.fillPoly fills even-odd, so they become holes).
    mask: front-plate bool image of the opening; pu/pv: horizontal/vertical cell pitch (px = mm); t: bar width.
    Holes in each column (every other row) are chained top to bottom: down the left halves, back up the right
    halves; each chain's top hole is bridged straight up to the outline. Returns [[z, y], ...]."""
    import cv2
    ell = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m = cv2.erode(mask.astype(np.uint8), ell(inset))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    C = max(cs, key=cv2.contourArea)[:, 0].tolist()
    m = np.zeros_like(m); cv2.fillPoly(m, [np.array(C, np.int32)], 1)          # outline without inner islands
    safe = cv2.erode(m, ell(t // 2 + 1)) > 0
    hu, hv = pu // 2, pv // 2
    d = hu * hv / np.hypot(hu, hv); sc = 1 - t / (2 * d)
    a, b = int(round(hu * sc)), int(round(hv * sc))
    vv, uu = np.nonzero(m)
    holes = {}
    for r in range(int(vv.min() // hv) - 1, int(vv.max() // hv) + 2):
        vc = r * hv
        off = 0 if r % 2 == 0 else hu
        for k in range(-40, 41):
            uc = 1150 + off + k * pu
            pts = [(uc - a, vc), (uc + a, vc), (uc, vc - b), (uc, vc + b), (uc, vc)]
            if all(0 <= x < m.shape[1] and 0 <= y < m.shape[0] and safe[y, x] for x, y in pts):
                holes.setdefault(uc, []).append(vc)
    runs = []
    for uc, vlist in holes.items():
        vlist = sorted(vlist); run = [vlist[0]]
        for v in vlist[1:]:
            seg = m[run[-1] + b:v - b + 1, uc]
            if seg.all(): run.append(v)
            else: runs.append((uc, run)); run = [v]
        runs.append((uc, run))
    poly = [tuple(p) for p in C]
    inserts = {}
    for uc, run in runs:
        top = run[0] - b
        v = top
        while v > 0 and m[v - 1, uc]: v -= 1                                    # straight up to the outline
        bp = (uc, v)
        k = int(np.argmin([(px - uc) ** 2 + (py - v) ** 2 for px, py in poly]))
        path = []
        for vc in run: path += [(uc, vc - b), (uc - a, vc), (uc, vc + b)]       # down: T, L, B
        path += [(uc + a, run[-1]), (uc, run[-1] - b)]                         # last hole: R, T
        for vc in run[-2::-1]: path += [(uc, vc + b), (uc + a, vc), (uc, vc - b)]   # up: B, R, T
        inserts.setdefault(k, []).append([bp] + path + [bp])
    out = []
    for i, pnt in enumerate(poly):
        out.append(pnt)
        for det in inserts.get(i, []):
            out += det + [pnt]
    n = sum(len(r) for _, r in runs)
    print(f'  lattice: {n} holes in {len(runs)} chains, polygon {len(out)} points')
    return [[round(1.15 - (u + 0.3) / 1000, 5), round(1.65 - (v + 0.3) / 1000, 5)] for u, v in out]


def grille_masks():
    """Front-plate masks (both halves) of the upper honeycomb (10) and the lower grille (4 + plate box below
    PLATE_SPLIT_Y)."""
    import outlines
    up = outlines.region_mask('front', [10], sym=True)
    lo = outlines.region_mask('front', [4], sym=True) | outlines.region_mask('front', [4, 6], sym=True) & \
        (np.arange(up.shape[0])[:, None] >= (1.65 - PLATE_SPLIT_Y) * 1000)
    return up, lo


def lens_band(view, frac=0.62):
    """Upper part of the headlamp lens (front region 17 / side region 41): the bright LED optics band, from
    3 mm under the lens top down to `frac` of the lens height, as a polygon in that view's coordinates. The lens
    top is the housing|lens line (38|17, 73|41) bridged across the stroked LED-cube box; both edges are
    median-smoothed so the band has no teeth."""
    from scipy.ndimage import median_filter, uniform_filter1d
    lab = p24.labels(view); rid = 17 if view == 'front' else 41
    vv, uu = np.nonzero(lab == rid)
    if view == 'front': keep = uu < 1150; vv, uu = vv[keep], uu[keep]
    cols = np.arange(uu.min() + 3, uu.max() - 2, 6); bot = []
    for c in cols:
        v = vv[np.abs(uu - c) <= 1]; bot.append(v.max())
    line = L.boundary(view, 38, 17) if view == 'front' else L.boundary(view, 73, 41)
    if view == 'front': line = line[line[:, 0] < 1150]
    o = np.argsort(line[:, 0]); top = np.interp(cols, line[o, 0], line[o, 1])
    top = uniform_filter1d(median_filter(top, 5, mode='nearest'), 5, mode='nearest')
    bot = uniform_filter1d(median_filter(np.array(bot, float), 5, mode='nearest'), 5, mode='nearest')
    lo = top + 3; hi = top + frac * (bot - top)
    pts = np.concatenate([np.stack([cols, lo], 1), np.stack([cols, hi], 1)[::-1]])
    if view == 'front': out = np.stack([1.15 - pts[:, 0] / 1000, 1.65 - pts[:, 1] / 1000], 1)
    else: out = np.stack([2.5 - pts[:, 0] / 1000, 1.65 - pts[:, 1] / 1000], 1)
    return out.round(4).tolist()


def fog_profile():
    """Side-view band for the fog lamp drum: from the carved pod floor (envelope - 45 mm) 25 mm forward."""
    lab = p24.labels('front'); vv, uu = np.nonzero(lab == 60); keep = uu < 1150
    zc = 1.15 - uu[keep].mean() / 1000; yc = 1.65 - vv[keep].mean() / 1000
    xe = float(env_front_x([yc], [zc])[0])
    return [[xe - 0.047, 0.33], [xe - 0.020, 0.33], [xe - 0.020, 0.41], [xe - 0.047, 0.41]]


# "TYPE R" grille badge (driver side of the honeycomb, Honda frame 10: R at z -0.245..-0.289, y 0.577..0.604):
# a blocky italic red R and a thin bright "TYPE" bar, in (image-right offset, up) metres from the R's corner
R_GLYPH = [(0.000, 0.000), (0.009, 0.000), (0.012, 0.010), (0.020, 0.010), (0.025, 0.000), (0.037, 0.000),
           (0.029, 0.012), (0.036, 0.016), (0.040, 0.027), (0.008, 0.027)]
TYPE_BAR = [(-0.062, 0.002), (-0.005, 0.002), (-0.003, 0.010), (-0.060, 0.010)]
BADGE_Z0, BADGE_Y0 = -0.245, 0.577


def badge_poly(glyph):
    return [[round(BADGE_Z0 - a, 4), round(BADGE_Y0 + b, 4)] for a, b in glyph]


# tapered inner end of the LED band (front view, passenger side): below this edge the band is dark, so its lower
# edge sweeps up to the lens top at the inner end
LAMP_TAPER = [[0.425, 0.585], [0.425, 0.674], [0.455, 0.667], [0.49, 0.657], [0.53, 0.648], [0.57, 0.643], [0.60, 0.641],
              [0.62, 0.585]]
LED_PITCH, LED_GAP = 0.017, 0.0028     # LED cell dividers across the bright band (m)


def led_dividers():
    """Thin dark dividers across the bright LED band: planes of constant |z| on the front-facing lens, of constant x
    on the side-facing tail (exact plane cuts). Each group repeats the optics' own membership."""
    lens_f = [['front', [17], 'pos', -LENS_RIM]] + LOW_BAND
    lens_s = [['side', [41], 'pos', -LENS_RIM]] + LOW_BAND
    G = []
    for z0 in np.arange(0.455, 0.83, LED_PITCH):
        G.append(F([17], 0.45) + lens_f + [['face', 'side', 0.7, 'neg'], ['box', 'az', round(z0, 4), round(z0 + LED_GAP, 4)]])
    for x0 in np.arange(1.72, 2.06, LED_PITCH):
        G.append(S([41], 0.7) + lens_s + [['box', 'x', round(x0, 4), round(x0 + LED_GAP, 4)]])
    # tapered inner end: below the LAMP_TAPER edge the band is dark (5 mm steps in |z|, ~1 mm in height)
    T = np.array(LAMP_TAPER[1:-1])
    for z0 in np.arange(T[0, 0], T[-1, 0], 0.005):
        yt = float(np.interp(z0 + 0.0025, T[:, 0], T[:, 1]))
        G.append([['front', [17], 'pos', 0.003], ['box', 'x', XMIN, 3.0], ['box', 'az', round(z0, 4), round(z0 + 0.005, 4)],
                  ['box', 'y', 0.0, round(yt, 4)]])
    return G


H_LOGO = [[-0.030, 0.590], [-0.019, 0.590], [-0.019, 0.611], [0.019, 0.611], [0.019, 0.590], [0.030, 0.590],
          [0.031, 0.642], [0.019, 0.640], [0.019, 0.621], [-0.019, 0.621], [-0.019, 0.640], [-0.031, 0.642]]


def part_specs():
    P = []
    # hood scoop: a hump on the hood up to the front plate's scoop top (0.942), its mouth (front region 45) black
    # (hood scoop: now a hump in the body field, design_front.apply_scoop; lamp internals: cut panels)
    P.append({'name': 'Civic_Front_Grille_Emblem', 'front': [46],
              'profile': [[1.95, 0.55], [2.35, 0.55], [2.35, 0.68], [1.95, 0.68]],
              'follow': 0.010, 'k': 0.003, 'h': 0.0035, 'mat': 'red', 'mirror': False})
    P.append({'name': 'Civic_Front_Grille_Emblem_H', 'front_profile': H_LOGO,
              'profile': [[1.95, 0.55], [2.35, 0.55], [2.35, 0.68], [1.95, 0.68]],
              'follow': 0.0135, 'k': 0.0015, 'h': 0.0024, 'mat': 'chrome', 'mirror': False})
    for nm, gl, mat in (('Civic_Front_Grille_Badge_TypeR', R_GLYPH, 'red'), ('Civic_Front_Grille_Badge_Type', TYPE_BAR, 'chrome')):
        P.append({'name': nm, 'front_profile': badge_poly(gl), 'profile': [[2.05, 0.55], [2.30, 0.55], [2.30, 0.65], [2.05, 0.65]],
                  'follow': 0.012, 'k': 0.0015, 'h': 0.002, 'mat': mat, 'mirror': False})
    up, lo = grille_masks()
    P.append({'name': 'Civic_Front_Grille_Mesh', 'front_profile': lattice_profile(up, 46, 26, 6, 3),
              'profile': [[2.06, 0.53], [2.24, 0.53], [2.24, 0.64], [2.06, 0.64]],
              'follow': 0.004, 'k': 0.002, 'h': 0.0035, 'mat': 'grille', 'mirror': False})
    P.append({'name': 'Civic_Front_Grille_Lower_Mesh', 'front_profile': lattice_profile(lo, 54, 30, 7, 4),
              'profile': [[2.08, 0.22], [2.26, 0.22], [2.26, 0.40], [2.08, 0.40]],
              'follow': 0.0035, 'k': 0.0018, 'h': 0.004, 'mat': 'grille', 'mirror': False})
    P.append({'names': ['Civic_Front_Bumper_Painted_Blades_R', 'Civic_Front_Bumper_Painted_Blades_L'], 'front': [54],
              'profile': blade_profile(), 'k': 0.003, 'h': 0.004, 'mat': 'paint'})
    # fog lamp: a 25 mm drum standing on the pod floor (the pod is carved 45 mm, the lamp's own drawing region 60)
    P.append({'names': ['Civic_Front_Fog_Lamp_Lens_R', 'Civic_Front_Fog_Lamp_Lens_L'], 'front': [60],
              'profile': fog_profile(), 'k': 0.003, 'h': 0.003, 'mat': 'Mat_FK8_Optical_Aluminum'})
    return P


if __name__ == '__main__':
    json.dump(cut_specs(), open(HERE / 'cut_front.json', 'w'), indent=1)
    json.dump(part_specs(), open(HERE / 'parts_front.json', 'w'), indent=1)
    print('wrote cut_front.json, parts_front.json')
