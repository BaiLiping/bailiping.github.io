"""Flares, wheel-well liners, mirrors, running boards, handles, fuel door, fender vents, hood extractor, cowl, wipers, decals."""
import numpy as np, bpy, bmesh
from geo import pchip, resample, spline
from mk import Surf, outline_panel, make_obj, grid_obj, mirror_y, extrude_profile, inside
from parts_front import rrect
import design as D

PAINT = 'Mat_Rapid_Red_Paint'

def flare(col, name, outer, inner, hub_x, yout=D.FLARE_Y_OUT, yout_fn=None):
    """Sweep the flare between the body-side outer edge and the wheel-opening inner edge."""
    n = 90
    O = spline(outer, n); I = spline(inner, n)
    rows = []
    for k in range(n):
        xo, zo = O[k]; xi, zi = I[k]
        yb = D.Y_SIDE(np.array([xo]), np.array([zo]))[0] + 0.002
        # radial direction away from the wheel centre (for the return flange)
        cz = D.HUB_Z + 0.068
        rx, rz = xi - hub_x, zi - cz; rl = np.hypot(rx, rz) + 1e-9; rx /= rl; rz /= rl
        yo = yout if yout_fn is None else float(yout_fn(xi))
        sec = []
        for t in np.linspace(0, 1, 9):
            x = xo + (xi - xo) * t; z = zo + (zi - zo) * t
            bulge = yb + (yo - yb) * np.sin(min(1.0, t / 0.62) * np.pi / 2) ** 0.45
            sec.append((x, bulge, z))
        # lip roll and return into the wheel well
        sec.append((xi - rx * 0.006, yo - 0.012, zi - rz * 0.006))
        sec.append((xi - rx * 0.010, yo - 0.045, zi - rz * 0.010))
        sec.append((xi + rx * 0.010, yo - 0.110, zi + rz * 0.010))
        rows.append(sec)
    P = np.array(rows)
    ob = grid_obj(name + '_L', P, mat='Mat_Fender_Flare_Satin', collection=col)
    return ob, mirror_y(ob, name + '_R')

def liner(col, name, inner, hub_x, y0=0.40, y1=1.06):
    n = 60; I = spline(inner, n)
    cz = D.HUB_Z + 0.068
    rows = []
    for (xi, zi) in I:
        rx, rz = xi - hub_x, zi - cz; rl = np.hypot(rx, rz); rx /= rl; rz /= rl
        rows.append([(xi + rx * 0.02, y, zi + rz * 0.02) for y in np.linspace(y1, y0, 8)])
    ob = grid_obj(name + '_L', np.array(rows), mat='Mat_Trim_Matte_Black', collection=col,
                  outward=lambda p: np.stack([hub_x - p[:, 0], np.zeros(len(p)), D.HUB_Z + 0.068 - p[:, 2]], 1))
    return ob, mirror_y(ob, name + '_R')

def mirrors(col):
    """Door mirror (left): painted cap, black housing, glass, turn lamp, sail."""
    out = {}
    # head as a lofted superellipse box: x (fore-aft) 0.935..1.105, y 1.005..1.218, z 1.362..1.562
    def head(x0, x1, y0, y1, z0, z1, nx=10, nc=40, pw=5.0):
        ang = np.linspace(0, 2 * np.pi, nc, endpoint=False)
        rows = []
        for t in np.linspace(0, 1, nx):
            x = x0 + (x1 - x0) * t
            sh = 1 - 0.20 * (t ** 2)        # taper toward the front
            cy, cz = (y0 + y1) / 2, (z0 + z1) / 2; ry, rz = (y1 - y0) / 2 * sh, (z1 - z0) / 2 * (0.94 + 0.06 * sh)
            ring = []
            for a in ang:
                c, s = np.cos(a), np.sin(a)
                ring.append((x, cy + ry * np.sign(c) * abs(c) ** (2 / pw), cz + rz * np.sign(s) * abs(s) ** (2 / pw)))
            rows.append(ring)
        return np.array(rows)
    P = head(0.938, 1.105, 1.010, 1.218, 1.362, 1.562)
    # split ring into cap (upper/front) and housing (lower) by assigning materials later; build two pieces
    upper = P[:, :21, :]; lower = np.concatenate([P[:, 20:, :], P[:, :1, :]], axis=1)
    cap = grid_obj('RaptorR_Mirror_Cap_FL', upper, mat=PAINT, collection=col)
    hous = grid_obj('RaptorR_Mirror_HousingLower_FL', lower, mat='Mat_Trim_Matte_Black', collection=col)
    # front closing cap (nose) and rear glass
    nose = P[-1]; cen = nose.mean(0) + np.array([0.012, 0, 0])
    V = np.vstack([nose, cen]); F = [(i, (i + 1) % len(nose), len(nose)) for i in range(len(nose))]
    nose_ob = make_obj('RaptorR_Mirror_Nose_FL', V, F, mat=PAINT, collection=col)
    back = P[0]; c2 = back.mean(0)
    gl = np.array([c2 + (p - c2) * 0.86 for p in back]) - np.array([0.004, 0, 0])
    V = np.vstack([back, gl]); m = len(back)
    F = [(i, (i + 1) % m, m + (i + 1) % m, m + i) for i in range(m)]
    bez = make_obj('RaptorR_Mirror_GlassBezel_FL', V, F, mat='Mat_Trim_Matte_Black', collection=col)
    V = np.vstack([gl, gl.mean(0)[None]]); F = [(i, (i + 1) % m, m) for i in range(m)]
    glass = make_obj('RaptorR_Mirror_Glass_FL', V, F, mat='Mat_Mirror_Chrome_Glass', collection=col)
    # turn signal strip on the outboard face, puddle/zone lamp under
    ts = extrude_profile('RaptorR_Mirror_TurnSignal_FL', [(0.95, 1.40), (1.07, 1.43), (1.07, 1.46), (0.95, 1.44)], 'y', 0.006,
                         origin=(0, 1.214, 0), mat='Mat_Taillight_Clear_Accent', collection=col)
    # sail / arm to the door
    sail = extrude_profile('RaptorR_Mirror_Sail_FL', [(1.040, 1.330), (1.165, 1.340), (1.175, 1.400), (1.120, 1.470), (1.040, 1.455)],
                           'y', 0.085, origin=(0, 0.935, 0), mat='Mat_Trim_Matte_Black', bevel=0.006, collection=col)
    arm = extrude_profile('RaptorR_Mirror_Arm_FL', [(0.985, 1.375), (1.090, 1.380), (1.090, 1.430), (0.985, 1.425)], 'y', 0.10,
                          origin=(0, 0.995, 0), mat='Mat_Trim_Matte_Black', bevel=0.008, collection=col)
    parts = [cap, hous, nose_ob, bez, glass, ts, sail, arm]
    out['L'] = parts
    out['R'] = [mirror_y(p, p.name.replace('_FL', '_FR')) for p in parts]
    return out

def running_boards(col):
    out = []
    for s, nm in ((1, 'L'), (-1, 'R')):
        rows = []
        for x in np.linspace(-1.12, 1.12, 40):
            taper = 1 - 0.6 * np.clip((abs(x) - 0.98) / 0.14, 0, 1)
            rows.append([(x, s * 0.90, 0.535), (x, s * (0.90 + 0.18 * taper), 0.538), (x, s * (0.90 + 0.19 * taper), 0.522),
                         (x, s * (0.90 + 0.175 * taper), 0.500), (x, s * 0.90, 0.498)])
        out.append(grid_obj(f'RaptorR_RunningBoard_{nm}', np.array(rows), mat='Mat_Running_Board_CastBlack', collection=col, flip=(s < 0)))
        # rocker panel molding (black) below the doors
        rows = []
        for x in np.linspace(-1.15, 1.17, 40):
            yb = D.Y_SIDE(np.array([x]), np.array([0.556]))[0]
            rows.append([(x, s * (yb - 0.004), 0.556), (x, s * (yb - 0.012), 0.530), (x, s * (yb - 0.06), 0.505)])
        out.append(grid_obj(f'RaptorR_Body_Cab_Rocker_{nm}', np.array(rows), mat='Mat_Trim_Matte_Black', collection=col, flip=(s < 0)))
    return out

def handles(col):
    out = []
    S = Surf('side', lambda x, z: D.Y_SIDE(x, z) + 0.016)
    for nm, (x0, x1) in (('F', (0.060, 0.270)), ('R', (-0.965, -0.760))):
        P = rrect(x0, x1, 1.205, 1.238, 0.014)
        ob = outline_panel(f'RaptorR_Door_Handle_{nm}', P, S, h=0.01, hem=0.016, roll=0.006, mat=PAINT, collection=col)
        cup = outline_panel(f'RaptorR_Door_HandleCup_{nm}', rrect(x0 + 0.02, x1 - 0.02, 1.196, 1.246, 0.016),
                            Surf('side', lambda x, z: D.Y_SIDE(x, z) - 0.006), h=0.01, hem=0.0, hem_outer=False, mat='Mat_Trim_Matte_Black', collection=col)
        out.append((ob, mirror_y(ob, ob.name + '_R'))); out.append((cup, mirror_y(cup, cup.name + '_R')))
        ob.name += '_L'; cup.name += '_L'
    return out

def fuel_door(col):
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    P = [(-1.375 + 0.080 * np.cos(a), 1.205 + 0.080 * np.sin(a)) for a in ang]
    S = Surf('side', lambda x, z: D.Y_SIDE(x, z) + 0.001)
    return outline_panel('RaptorR_Fuel_Door', P, S, h=0.015, hem=0.010, roll=0.003, mat=PAINT, collection=col)

def fender_vents(col):
    out = []
    bez = [(1.212, 1.292), (1.290, 1.300), (1.400, 1.335), (1.478, 1.372), (1.470, 1.390), (1.380, 1.385), (1.300, 1.360), (1.215, 1.312)]
    S = Surf('side', lambda x, z: D.Y_SIDE(x, z) + 0.006)
    b = outline_panel('RaptorR_Fender_Vent_Bezel', bez, S, h=0.01, hem=0.010, roll=0.004, mat='Mat_Fender_Flare_Satin', collection=col)
    th = [(1.228, 1.300), (1.300, 1.310), (1.400, 1.342), (1.455, 1.372), (1.380, 1.376), (1.300, 1.352), (1.230, 1.310)]
    t = outline_panel('RaptorR_Fender_Vent_Throat', th, Surf('side', lambda x, z: D.Y_SIDE(x, z) - 0.010), h=0.01, hem=0.0, hem_outer=False,
                      mat='Mat_Grille_Charcoal_Black', collection=col)
    out = [(b, mirror_y(b, 'RaptorR_Fender_Vent_Bezel_R')), (t, mirror_y(t, 'RaptorR_Fender_Vent_Throat_R'))]
    b.name = 'RaptorR_Fender_Vent_Bezel_L'; t.name = 'RaptorR_Fender_Vent_Throat_L'
    return out

def hood_extractor(col):
    """Raptor R power-dome heat extractor: four vented pods at the rear of the dome."""
    out = []
    for yc, w in ((-0.345, 0.150), (-0.115, 0.170), (0.115, 0.170), (0.345, 0.150)):
        xs = np.linspace(1.292, 1.640, 22); ys = np.linspace(yc - w / 2, yc + w / 2, 15)
        P = np.zeros((len(xs), len(ys), 3))
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                zb = D.Z_HOOD(np.array([x]), np.array([y]))[0]
                ty = 1 - ((y - yc) / (w / 2)) ** 4
                tx = np.clip((x - 1.292) / 0.05, 0, 1) * np.clip((1.640 - x) / 0.22, 0, 1) ** 0.7
                P[i, j] = (x, y, zb + 0.003 + 0.010 * max(ty, 0) * tx)
        out.append(grid_obj('RaptorR_Hood_Extractor_Pod', P, mat='Mat_Hood_Extractor_SatinGray', collection=col))
        # rear-facing vent mouth (dark)
        rows = []
        for y in np.linspace(yc - w * 0.38, yc + w * 0.38, 9):
            zb = D.Z_HOOD(np.array([1.300]), np.array([y]))[0]
            rows.append([(1.296, y, zb + 0.004), (1.298, y, zb + 0.013)])
        out.append(grid_obj('RaptorR_Hood_Extractor_Vent', np.array(rows), mat='Mat_Grille_Charcoal_Black', collection=col))
    return out

def cowl_and_wipers(col):
    out = []
    rows = []
    for y in np.linspace(-0.86, 0.86, 35):
        xb = D.WS_BASE_X(abs(y)); zb = D.WS_BASE_Z(abs(y))
        rows.append([(D.HOOD_REAR_X - 0.004, y, D.Z_HOOD(np.array([D.HOOD_REAR_X]), np.array([y]))[0] - 0.006), (xb + 0.004, y, zb - 0.004)])
    out.append(grid_obj('RaptorR_Cowl_Grille', np.array(rows), mat='Mat_Trim_Matte_Black', collection=col))
    for (y0, y1) in ((-0.80, -0.05), (0.02, 0.62)):
        pts = []
        for t in np.linspace(0, 1, 12):
            y = y0 + (y1 - y0) * t; xb = D.WS_BASE_X(abs(y)) - 0.03 * t; zb = D.WS_BASE_Z(abs(y)) + 0.04 * t + 0.008
            pts.append((xb, y, zb))
        P = np.array([[p, p + np.array([-0.012, 0, 0.014])] for p in pts])
        out.append(grid_obj('RaptorR_Cowl_WiperBlade', P, mat='Mat_Trim_Matte_Black', collection=col))
    return out

def decal(col, name, mat, x0, x1, z0, z1, side, u_of, v_of, clip=None, h=0.02):
    """UV-mapped decal patch floating 1.5 mm off the body side; `clip` = flare outer edge (x,z) to stay above."""
    sgn = 1 if side == 'L' else -1
    S = Surf('side', lambda x, z: D.Y_SIDE(x, z) + 0.0015)
    P = [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]
    if clip is not None:
        C = np.array(sorted(clip)); zf = lambda x: float(np.interp(x, C[:, 0], C[:, 1])) + 0.006
        xs = np.linspace(x0, x1, 40)
        lower = [(x, max(z0, min(zf(x), z1 - 0.004))) for x in xs]
        P = lower + [(x1, z1), (x0, z1)]
    ob = outline_panel(name, P, S, h=h, hem=0.0, hem_outer=False, mat=mat, collection=col)
    me = ob.data
    if side == 'R':
        for v in me.vertices: v.co.y = -v.co.y
        me.flip_normals()
    uvl = me.uv_layers.new(name='UVMap')
    for poly in me.polygons:
        for li in poly.loop_indices:
            co = me.vertices[me.loops[li].vertex_index].co
            uvl.data[li].uv = (u_of(co.x, co.z), v_of(co.x, co.z))
    return ob

def decals(col):
    out = []
    out.append(decal(col, 'RaptorR_Graphic_Bedside_L', 'Mat_Graphic_Bedside_R_L', -2.972, -1.660, 0.858, 1.400, 'L',
                     lambda x, z: -0.762 * x - 1.265, lambda x, z: 1.792 * z - 1.537, clip=D.FLARE_R_OUT))
    out.append(decal(col, 'RaptorR_Graphic_Bedside_R', 'Mat_Graphic_Bedside_R_R', -2.972, -1.660, 0.858, 1.400, 'R',
                     lambda x, z: 0.762 * x + 2.265, lambda x, z: 1.792 * z - 1.537, clip=D.FLARE_R_OUT))
    for side in ('L', 'R'):
        out.append(decal(col, f'RaptorR_Graphic_Fender_F{side}', 'Mat_Graphic_Fender_Dash', 1.200, 1.640, 0.740, 1.250, side,
                         lambda x, z: -1.996 * x + 3.304, lambda x, z: 1.919 * z - 0.030 * x - 0.157 * 1.0 - 1.175,
                         clip=[p for p in D.FLARE_F_OUT if p[0] < 1.85]))
    return out

def hood_decal(col):
    HS = Surf('top', lambda x, y: D.Z_HOOD(x, y) + 0.0015)
    P = [(1.645, -0.44), (2.40, -0.38), (2.40, 0.38), (1.645, 0.44)]
    ob = outline_panel('RaptorR_Graphic_Hood', P, HS, h=0.03, hem=0.0, hem_outer=False, mat='Mat_Graphic_Hood_PowerDome', collection=col)
    me = ob.data; uvl = me.uv_layers.new(name='UVMap')
    for poly in me.polygons:
        for li in poly.loop_indices:
            co = me.vertices[me.loops[li].vertex_index].co
            uvl.data[li].uv = (-1.129 * co.y + 0.504, 0.781 * co.x - 1.007)
    return ob
