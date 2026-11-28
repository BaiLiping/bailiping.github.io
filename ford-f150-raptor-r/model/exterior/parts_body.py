"""Sheet metal: side panels, fender decks, greenhouse, roof, hood, bed and tailgate shells."""
import numpy as np, bpy
from geo import pchip, spline, resample
from mk import Surf, outline_panel, make_obj, grid_obj, mirror_y
import design as D

PAINT = 'Mat_Rapid_Red_Paint'
BLACK = 'Mat_Trim_Matte_Black'
GLOSS = 'Mat_Gloss_Black_Pillar'

SIDE = Surf('side', D.Y_SIDE)

def lr(ob, base):
    """name left object and make the mirrored right copy"""
    ob.name = base + '_L'; ob.data.name = base + '_L'
    r = mirror_y(ob, base + '_R')
    return ob, r

def poly_from(pts, n=None):
    P = np.asarray(pts, float)
    return P if n is None else resample(P, n, closed=True)

def side_panels(col):
    out = {}
    # ---- front fender, side part (below the shoulder roll) ----
    xs = np.linspace(2.500, 1.194, 38)
    top = [(x, D.Z_SHOULDER(x)) for x in xs]
    fin = [p for p in D.FLARE_F_IN if p[0] > 1.29]
    # lamp cut-out: 8 mm clear of the lens' top and slanted rear edge (lens edge from parts_detail.LAMP_ZB)
    import parts_detail as PD
    ex = np.linspace(2.234, 2.440, 24); ez = PD.LAMP_ZB(ex)
    t = np.gradient(np.stack([ex, ez], 1), axis=0); t /= np.linalg.norm(t, axis=1, keepdims=True)
    nrm = np.abs(np.stack([t[:, 1], -t[:, 0]], 1))   # unit normal of the slanted edge; offset rearward and down
    cut = [(x - 0.008 * n[0], z - 0.008 * n[1]) for (x, z), n in zip(np.stack([ex, ez], 1), nrm)]
    cut = [p for p in cut if p[1] < 1.290]
    outline = ([(1.194, 0.80)] + [(1.194, D.Z_SHOULDER(1.194))] + top[::-1][1:] +
               [(2.500, 1.300), (2.224, 1.300)] + cut + [(2.452, 0.928), (2.458, 0.862), (2.300, 0.862), (2.262, 0.870)] +
               [(x + 0.012, z + 0.012) for (x, z) in reversed(fin) if x < 2.24] + [(1.30, 0.80)])
    ob = outline_panel('RaptorR_Body_Fender_F', outline, SIDE, h=0.022, mat=PAINT, collection=col)
    out['fender'] = lr(ob, 'RaptorR_Body_Fender')
    # ---- fender deck: swept roll-over from the shoulder to the hood gap ----
    xs = np.linspace(1.194, 2.556, 70)
    rows = []
    for x in xs:
        zs = D.Z_SHOULDER(x); ys = D.Y_SIDE(np.array([x]), np.array([zs]))[0]
        zg = D.Z_DECK(x); yg = D.Y_GAP(x) + 0.002
        # section: quarter-round shoulder (radius from geometry) then gently rising deck to the gap
        ang = np.linspace(0, np.pi / 2, 14)
        r_v = zg - zs - 0.012; r_h = 0.060
        sec = [(ys - r_h * (1 - np.cos(a)), zs + r_v * np.sin(a)) for a in ang]
        y1, z1 = sec[-1]
        for t in np.linspace(0, 1, 10)[1:]:
            sec.append((y1 + (yg - y1) * t, z1 + (zg - z1) * t))
        rows.append([(x, y, z) for (y, z) in sec])
    P = np.array(rows)
    ob = grid_obj('RaptorR_Body_FenderDeck_F', P, mat=PAINT, collection=col)
    out['deck'] = lr(ob, 'RaptorR_Body_FenderDeck')
    # ---- doors ----
    out['door_f'] = lr(outline_panel('RaptorR_Door_F_Skin', D.FRONT_DOOR, SIDE, h=0.022, mat=PAINT, collection=col), 'RaptorR_Door_Skin_F')
    out['door_r'] = lr(outline_panel('RaptorR_Door_R_Skin', D.REAR_DOOR, SIDE, h=0.022, mat=PAINT, collection=col), 'RaptorR_Door_Skin_R')
    out['quarter'] = lr(outline_panel('RaptorR_Body_CabQuarter', D.CAB_QUARTER, SIDE, h=0.02, mat=PAINT, collection=col), 'RaptorR_Body_CabQuarter')
    # ---- bedside (side part up to the roll) ----
    rout = [p for p in D.FLARE_R_IN]
    z_roll = 1.405
    outline = ([(D.BED_FRONT_X, 0.640), (D.BED_FRONT_X, z_roll), (-2.955, z_roll), (-2.960, 1.330)] +
               [(-2.962, 1.312), (-2.728, 1.312), (-2.712, 1.290), (-2.728, 1.200), (-2.758, 1.130), (-2.798, 1.060),
                (-2.848, 0.990), (-2.888, 0.952), (-2.930, 0.935), (-2.958, 0.930), (-2.958, 0.845), (-2.610, 0.845),
                (-2.480, 0.760)] +
               [(x - 0.012, z + 0.012) for (x, z) in reversed(rout)][::-1][::-1] + [(-1.40, 0.640)])
    # (bedside outline: front edge, roll line, rear edge around the tail lamp, bottom, arch under the flare)
    # Only the driver side carries the fuel filler. Cutting the bedside also
    # exposes the recessed neck behind the animated flap.
    fuel_cutout = [(-1.375 + 0.084 * np.cos(a), 1.205 + 0.084 * np.sin(a))
                   for a in np.linspace(0, 2 * np.pi, 64, endpoint=False)]
    left = outline_panel('RaptorR_Body_BedSide_L', outline, SIDE, holes=[fuel_cutout],
                         h=0.022, mat=PAINT, collection=col)
    solid = outline_panel('BedsideTemplate', outline, SIDE, h=0.022, mat=PAINT, collection=col)
    right = mirror_y(solid, 'RaptorR_Body_BedSide_R')
    bpy.data.objects.remove(solid, do_unlink=True)
    out['bedside'] = (left, right)
    # ---- bed rail roll: from the side roll line over to the rail cap ----
    xs = np.linspace(D.BED_FRONT_X, -2.955, 80)
    rows = []
    for x in xs:
        ys = D.Y_SIDE(np.array([x]), np.array([z_roll]))[0]
        ang = np.linspace(0, np.pi / 2, 12)
        sec = [(ys - 0.045 * (1 - np.cos(a)), z_roll + (D.BED_RAIL_Z - 0.010 - z_roll) * np.sin(a)) for a in ang]
        y1, z1 = sec[-1]
        for t in np.linspace(0, 1, 6)[1:]:
            sec.append((y1 + (0.905 - y1) * t, z1))
        rows.append([(x, y, z) for (y, z) in sec])
    out['bedroll'] = lr(grid_obj('RaptorR_Body_BedRoll', np.array(rows), mat=PAINT, collection=col), 'RaptorR_Body_BedRoll')
    return out

def greenhouse(col):
    out = {}
    def Yg(x, z):
        t = np.clip((z - D.Z_BELT) / (D.Z_DRIP - D.Z_BELT), 0, 1.2)
        yb, yd = D.Y_BELT(x), D.Y_DRIP(x)
        return yb - (yb - yd) * t ** D.P_TUMBLE(x)
    GS = Surf('side', Yg)
    # door frames (painted outer frame band) with DLO holes; glass separately
    def frame_outline(dlo, grow):
        P = np.asarray(dlo, float); c = P.mean(0)
        return [tuple(c + (p - c) * (1 + grow / np.linalg.norm(p - c))) for p in P]
    for tag, dlo, (x0, x1) in (('F', D.FRONT_DLO, (-0.034, 1.150)), ('R', D.REAR_DLO, (-1.007, -0.046))):
        # outer region of the door upper: from belt to drip rail across the door's x range
        if tag == 'F':
            outer = [(-0.034, D.Z_BELT), (1.10, D.Z_BELT), (1.174, 1.40), (1.162, 1.52), (1.066, 1.62), (0.906, 1.75),
                     (0.746, 1.86), (0.586, 1.928), (0.425, 1.950), (0.20, 1.957), (-0.034, 1.958)]
        else:
            outer = [(-0.046, D.Z_BELT), (-1.007, D.Z_BELT), (-1.012, 1.70), (-1.005, 1.90), (-0.985, 1.945),
                     (-0.955, 1.958), (-0.046, 1.958)]
        # Keep the gloss-black B-pillar applique and window weatherstrip, with
        # body-colour metal across the header and down the curved outer stile.
        if tag == 'F':
            outer[:2] = D.FRONT_DOOR[9:19]
        ob = outline_panel(f'RaptorR_Door_Frame_{tag}', outer, GS, holes=[dlo], h=0.018, bnd=0.006,
                           hem=0.010, mat=PAINT, collection=col, offset=0.004)
        ob.data.materials.append(bpy.data.materials[GLOSS])
        window = np.asarray(dlo)
        starts = window
        spans = np.roll(window, -1, axis=0) - window
        for poly in ob.data.polygons:
            x, _, z = poly.center
            relative = [x, z] - starts
            t = np.clip((relative * spans).sum(axis=1) / (spans * spans).sum(axis=1), 0, 1)
            distance = np.linalg.norm(relative - t[:, None] * spans, axis=1).min()
            if distance < 0.014 or (-0.16 < x < 0.068 and z < 1.925):
                poly.material_index = 1
        out['frame_' + tag] = lr(ob, f'RaptorR_Door_Frame_{tag}')
        g = outline_panel(f'RaptorR_Glass_Door_{tag}', dlo, GS, h=0.03, hem=0.0, hem_outer=False,
                          mat='Mat_Automotive_Glass' if tag == 'F' else 'Mat_Privacy_Glass', collection=col, offset=-0.004)
        out['glass_' + tag] = lr(g, f'RaptorR_Glass_Door_{tag}')
    # A-pillar strip (painted) between windshield side edge and front door frame
    # C-pillar (painted) behind the rear door up to the roof
    cp = [(-1.012, D.Z_BELT), (-1.150, D.Z_BELT), (-1.147, 1.60), (-1.138, 1.75), (-1.124, 1.85), (-1.104, 1.915), (-1.080, 1.948), (-0.955, 1.958),
          (-0.985, 1.945), (-1.005, 1.90), (-1.012, 1.70)]
    out['cpillar'] = lr(outline_panel('RaptorR_Body_Cab_CPillar', cp, GS, h=0.02, mat=PAINT, collection=col, hem_outer=False), 'RaptorR_Body_Cab_CPillar')
    # roof
    # roof with the stamped longitudinal stiffening ribs seen in Ford's elevated rear renders (11 ribs, ~110 mm pitch)
    xr = np.linspace(D.ROOF_REAR_X, D.ROOF_FRONT_X, 90); vr = np.linspace(-1, 1, 181)
    P = np.zeros((len(xr), len(vr), 3))
    ribs = np.arange(-5, 6) * 0.110
    for i, x in enumerate(xr):
        ym = D.Y_DRIP(x)
        ys = vr * ym
        ex = np.clip((x - D.ROOF_REAR_X - 0.05) / 0.12, 0, 1) * np.clip((D.ROOF_FRONT_X - 0.06 - x) / 0.14, 0, 1)
        r = np.zeros_like(ys)
        for yc in ribs:
            r = np.maximum(r, np.clip(1 - np.abs(ys - yc) / 0.016, 0, 1) ** 2 * (3 - 2 * np.clip(1 - np.abs(ys - yc) / 0.016, 0, 1)))
        r *= np.clip((0.62 - np.abs(ys)) / 0.05, 0, 1)
        P[i, :, 0] = x; P[i, :, 1] = ys; P[i, :, 2] = D.Z_ROOF(np.full_like(ys, x), ys) + 0.0035 * ex * r
    out['roof'] = grid_obj('RaptorR_Body_Cab_RoofPanel', P, mat=PAINT, collection=col)
    # windshield glass: loft from base edge to top edge
    v = np.linspace(-1, 1, 33); u = np.linspace(0, 1, 26)
    W = np.zeros((len(u), len(v), 3))
    for j, s in enumerate(v):
        yb = s * D.WS_HALF_BASE; yt = s * D.WS_HALF_TOP
        pb = np.array([D.WS_BASE_X(abs(yb)), yb, D.WS_BASE_Z(abs(yb))]); pt = np.array([D.WS_TOP_X(abs(yt)), yt, D.WS_TOP_Z(abs(yt))])
        for i, t in enumerate(u):
            p = pb + (pt - pb) * t
            # crown: bulge outward (forward/up) toward the middle
            bow = 0.018 * np.sin(np.pi * t) + 0.030 * (1 - s * s) * 0.5
            nrm = np.array([0.79, 0, 0.62])
            W[i, j] = p + nrm * bow
    out['windshield'] = grid_obj('RaptorR_Glass_Windshield', W, mat='Mat_Automotive_Glass', collection=col)
    # A-pillars: strip from windshield edge outward to the door frame front
    for side in (1,):
        rows = []
        for t in np.linspace(0, 1, 26):
            pb = np.array([D.WS_BASE_X(D.WS_HALF_BASE), D.WS_HALF_BASE, D.WS_BASE_Z(D.WS_HALF_BASE)])
            pt = np.array([D.WS_TOP_X(D.WS_HALF_TOP), D.WS_HALF_TOP, D.WS_TOP_Z(D.WS_HALF_TOP)])
            p_in = pb + (pt - pb) * t + np.array([0.79, 0, 0.62]) * 0.018 * np.sin(np.pi * t)
            z = p_in[2]
            # outer edge: door frame front edge at this height
            fx = pchip([1.40, 1.52, 1.62, 1.75, 1.86, 1.928, 1.958], [1.174, 1.162, 1.066, 0.906, 0.746, 0.586, 0.470])(min(max(z, 1.40), 1.958))
            p_out = np.array([fx + 0.006, Yg(np.array([fx]), np.array([z]))[0] + 0.004, z])
            mid = (p_in + p_out) / 2 + np.array([0.012, 0.012, 0.010])
            rows.append([p_in, mid, p_out])
        out['apillar'] = lr(grid_obj('RaptorR_Body_Cab_APillar', np.array(rows), mat=PAINT, collection=col), 'RaptorR_Body_Cab_APillar')
    # cab back wall with rear window opening
    def Xb(y, z): return -1.152 - 0.004 * (y / 0.9) ** 2 + 0.135 * np.clip(z - 1.45, 0, None) ** 1.15
    BS = Surf('front', Xb, sign=-1.0)
    back_outer = [(-0.940, 0.62), (0.940, 0.62), (0.940, 1.445), (0.910, 1.60), (0.832, 1.80), (0.745, 1.925),
                  (0.40, 1.932), (0.0, 1.930), (-0.40, 1.932), (-0.745, 1.925), (-0.832, 1.80), (-0.910, 1.60), (-0.940, 1.445)]
    rw = [(-0.805, 1.455), (0.805, 1.455), (0.770, 1.70), (0.725, 1.868), (-0.725, 1.868), (-0.770, 1.70)]
    # CHMSL / camera housing at the rear of the roof
    rows = []
    for t in np.linspace(-1, 1, 15):
        y = 0.205 * t
        zr = D.Z_ROOF(np.array([-1.00]), np.array([y]))[0]
        rows.append([(-1.098, y, 1.925), (-1.090, y * 0.98, 2.010), (-1.060, y * 0.96, 2.036), (-0.990, y * 0.95, 2.040), (-0.930, y * 0.95, zr + 0.002)])
    out['chmsl'] = grid_obj('RaptorR_Body_CHMSL_Housing', np.array(rows), mat='Mat_Gloss_Black_Pillar', collection=col)
    out['back'] = outline_panel('RaptorR_Body_Cab_RearWall', back_outer, BS, holes=[rw], h=0.025, mat=PAINT, collection=col, hem_outer=False)
    out['rearglass'] = outline_panel('RaptorR_Glass_RearCab', rw, Surf('front', lambda y, z: Xb(y, z) + 0.004, sign=-1.0), h=0.04,
                                     hem=0.0, hem_outer=False, mat='Mat_Privacy_Glass', collection=col)
    return out

def hood(col):
    out = {}
    HS = Surf('top', D.Z_HOOD)
    xs = np.linspace(D.HOOD_REAR_X, 2.62, 60)
    front = [(2.640 - 0.050 * (y / 0.85) ** 2, y) for y in np.linspace(-0.845, 0.845, 33)]
    left = [(x, D.Y_GAP(x) - 0.003) for x in xs]
    right = [(x, -D.Y_GAP(x) + 0.003) for x in xs]
    outline = [(D.HOOD_REAR_X, -0.837)] + right[1:] + front + left[::-1][1:] + [(D.HOOD_REAR_X, 0.837)]
    out['hood'] = outline_panel('RaptorR_Hood_Panel', outline, HS, h=0.025, mat=PAINT, collection=col, hem=0.014)
    return out
