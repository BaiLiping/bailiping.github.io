"""Rear parts the 3-view builder cannot make cleanly: taillamp graphics, vent liners, honeycomb and reflectors, the
diffuser pinstripe, the rear wiper, the high-mount stop lamp, the roof vortex generators, the wing, its endplates
and pylons.

Run: venv/bin/python rear_parts.py <body field.npz> <out.npz>
Writes per-part <name>/V, /F, /N (glTF axes, float32/int32) plus an 'index' JSON list of {name, mat, pivot}, the
format of parts3v.py, for assemble.py --parts.

Taillamps: thin skins on the body surface, cut exactly along the polygons of rear_lamp_gfx.json, which are drawn on
the lamp-corner chart (rear_chart.py: u = arc length round the tail corner, y = height). A black housing base
(1.5 mm proud) carries the red lens and the silver inner cells (2.5 mm). Pieces above the hatch shut line (the red
strips under the spoiler bar) go on the hatch pivot.
Vents: on the vent-corner chart (rear_vent_gfx.json), a black liner on the recess floor (design_rear.py carves the
opening), a honeycomb grille of hexagonal rings 6 mm off the floor clipped to the opening, and the red reflector.
Wing, endplates, pylons: smooth intersections of analytic 2D outlines (side x-y, front z-y, plan x-z polygons),
so the faces are flat with no raster stair-steps; outlines as in rear_specs.py.
"""
import sys, json
import numpy as np
import cv2
from skimage import measure
import p24, outlines
from rear_chart import Field, Chart, LAMP

HERE = __file__.rsplit('/', 1)[0] if '/' in __file__ else '.'


# ---------------------------------------------------------------- mesh helpers
def cut(V, F, s, extra=None):
    """Keep the part of a triangle mesh where s < 0, splitting crossed triangles exactly. extra: per-vertex arrays
    interpolated like V. Returns V2, F2, s2 (s at the new vertices), extra2."""
    inside = s < 0
    cnt = inside[F].sum(1)
    keep = F[cnt == 3]; mixed = F[(cnt > 0) & (cnt < 3)]
    newv = {}; ev = []; et = []
    def ep(a, b):
        k = (a, b) if a < b else (b, a)
        if k not in newv:
            t = s[a] / (s[a] - s[b]); ev.append((a, b, t)); newv[k] = len(V) + len(ev) - 1
        return newv[k]
    tris = []
    for f in mixed:
        poly = []
        for i in range(3):
            a, b = f[i], f[(i + 1) % 3]
            if inside[a]: poly.append(a)
            if inside[a] != inside[b]: poly.append(ep(a, b))
        for k in range(1, len(poly) - 1):
            tris.append((poly[0], poly[k], poly[k + 1]))
    def interp(A):
        if not ev: return A
        a = np.array([e[0] for e in ev]); b = np.array([e[1] for e in ev]); t = np.array([e[2] for e in ev])
        t = t.reshape(-1, *([1] * (A.ndim - 1)))
        return np.concatenate([A, A[a] + t * (A[b] - A[a])])
    V2 = interp(V); S2 = interp(s); X2 = [interp(x) for x in (extra or [])]
    F2 = np.vstack([keep] + ([np.array(tris)] if tris else [])) if (len(keep) or tris) else np.zeros((0, 3), int)
    used = np.unique(F2); remap = -np.ones(len(V2), int); remap[used] = np.arange(len(used))
    return V2[used], remap[F2], S2[used], [x[used] for x in X2]


def project(fld, P, it=3):
    """Newton steps onto the field's zero level."""
    for _ in range(it):
        n = fld.grad(P); P = P - fld(P)[:, None] * n
    return P


def refine(fld, V, F, maxlen=0.005, off=0.0):
    """Split triangles until no edge is longer than maxlen, put the vertices back on the body surface and lift
    them off along the normals by off. Flat skin triangles spanning a curved surface sink into it (a 30 mm edge on
    a 50 mm radius sags 2 mm), and chart rows stretch where the surface turns horizontal. Returns V, F, N."""
    import trimesh
    if len(F):
        V, F = trimesh.remesh.subdivide_to_size(V, F, max_edge=maxlen, max_iter=8)
    V = project(fld, np.asarray(V, float)); N = fld.grad(V)
    return V + N * off, np.asarray(F), N


def mirror(V, F, N):
    return V * [1, 1, -1], F[:, ::-1].copy(), N * [1, 1, -1]


# ---------------------------------------------------------------- 2D signed distances
def poly_sdf(X, Y, poly):
    """Exact signed distance (negative inside) of points (X, Y arrays) to a closed polygon [[x, y], ...]."""
    P = np.asarray(poly, float); A = P; B = np.roll(P, -1, 0)
    x = X.ravel()[:, None]; y = Y.ravel()[:, None]
    ex, ey = (B - A)[:, 0][None], (B - A)[:, 1][None]
    wx, wy = x - A[:, 0][None], y - A[:, 1][None]
    t = np.clip((wx * ex + wy * ey) / np.maximum(ex * ex + ey * ey, 1e-18), 0, 1)
    d = np.sqrt(((wx - t * ex) ** 2 + (wy - t * ey) ** 2).min(1))
    c1 = (A[:, 1][None] <= y) & (B[:, 1][None] > y); c2 = (A[:, 1][None] > y) & (B[:, 1][None] <= y)
    xi = A[:, 0][None] + (y - A[:, 1][None]) * ex / np.where(ey == 0, 1e-18, ey)
    inside = (((c1 | c2) & (x < xi)).sum(1) % 2) == 1
    return np.where(inside, -d, d).reshape(X.shape)


def raster_sdf(polys, u0, y0, res, W, H, evenodd=False):
    """Signed distance (m) image of chart polygons, res m/px: their union, or even-odd (holes) with evenodd."""
    m = np.zeros((H, W), np.uint8)
    pts = [np.round(np.array([[(u - u0) / res * 16, (y - y0) / res * 16] for u, y in p])).astype(np.int32) for p in polys]
    if evenodd: cv2.fillPoly(m, pts, 1, lineType=cv2.LINE_8, shift=4)
    else:
        for q in pts: cv2.fillPoly(m, [q], 1, lineType=cv2.LINE_AA, shift=4)
    return p24.sdf2d(m > 0) * res


def smax(a, b, k):
    return 0.5 * (a + b + np.sqrt((a - b) ** 2 + k * k))


# ---------------------------------------------------------------- taillamps
LAMP_LAYERS = [('housing', 'gloss_black', 0.0015, 'Civic_Taillamp_Housing'),
               ('red', 'taillamp', 0.0025, 'Civic_Taillamp'),
               ('wedge', 'Mat_TypeR_Red_Alcantara', 0.0022, 'Civic_Taillamp_Wedge'),     # dim red light guide
               ('clear', 'Mat_FK8_Taillamp_Silvered_Cells', 0.0022, 'Civic_Taillamp_Clear'),
               ('cells', 'Mat_FK8_Ruby_Optical_Ribs', 0.0029, 'Civic_Taillamp_Cells'),
               ('bezel', 'chrome', 0.0030, 'Civic_Taillamp_Bezel'),
               ('dots', 'gloss_black', 0.0032, 'Civic_Taillamp_Sockets')]
LAMP_GRID = 0.003
LAMP_SIDE_IDS = [51, 59, 83, 94, 102, 135, 141, 144]
HATCH_Y, HATCH_Z = 0.995, 0.665          # lamp pieces above y and inboard of z ride on the hatch


def lamps(fld):
    """Returns [(name, mat, pivot, V, F, N)] for the passenger-side lamp (mirrored later)."""
    ch = Chart(fld, **LAMP, dy=0.002)
    gfx = json.load(open(f'{HERE}/rear_lamp_gfx.json'))
    allp = np.concatenate([np.array(p) for k, v in gfx.items() if not k.startswith('_') for p in v])
    u0, y0 = allp.min(0) - 0.01; u1, y1 = allp.max(0) + 0.01
    res = 0.0005; W = int((u1 - u0) / res) + 1; H = int((y1 - y0) / res) + 1
    sdf = {k: raster_sdf(v, u0, y0, res, W, H) for k, v in gfx.items() if not k.startswith('_')}
    us = np.arange(u0, u1 + 1e-9, LAMP_GRID); ys = np.arange(y0, y1 + 1e-9, LAMP_GRID)
    UU, YY = np.meshgrid(us, ys)
    nu = len(us)
    q = np.arange(len(us) * len(ys)).reshape(len(ys), nu)
    a, b, c, d = q[:-1, :-1].ravel(), q[:-1, 1:].ravel(), q[1:, :-1].ravel(), q[1:, 1:].ravel()
    F = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])
    UV = np.stack([UU.ravel(), YY.ravel()], 1)
    P = ch.point(UV[:, 0], UV[:, 1])
    def samp(img, uv):
        return p24.sample(img, (uv[:, 0] - u0) / res, (uv[:, 1] - y0) / res, outside=1.0)
    # the chart runs on round the corner onto the shoulder and C-pillar flank: the lamp ends where Honda's side drawing
    # ends it (P24 side regions of the lamp, dilated 4 mm); the band on the hatch ends at the hatch (x < -1.90)
    side_sd = p24.sdf2d(outlines.region_mask('side', LAMP_SIDE_IDS, sym=False)) / 1000.0
    su, sv = p24.to_px('side', P[:, 0], P[:, 1], P[:, 2])
    hz = (P[:, 1] > HATCH_Y) & (P[:, 2] < HATCH_Z)
    clip = np.where(hz, P[:, 0] + 1.90, p24.sample(side_sd, su, sv, outside=1.0) - 0.004)
    out = []
    for layer, mat, off, lname in LAMP_LAYERS:
        if layer not in sdf: continue
        s = np.maximum(samp(sdf[layer], UV), clip)
        if layer != 'housing': s = np.maximum(s, samp(sdf['housing'], UV) + 0.0005)
        V1, F1, _, (UV1,) = cut(P, F, s, [UV])
        if not len(F1): continue
        V1 = project(fld, V1)
        for piv, sgn in (('', 1), ('Pivot_Hatch_Assembly', -1)):
            sh = sgn * -np.maximum(HATCH_Y - V1[:, 1], V1[:, 2] - HATCH_Z)          # < 0: this pivot's part
            V2, F2, _, _ = cut(V1, F1, sh)
            if not len(F2): continue
            V2, F2, N2 = refine(fld, V2, F2, 0.004, off)
            nm = lname.replace('Civic_Taillamp', 'Civic_Hatch_Taillamp') if piv else lname
            out.append((nm, mat, piv, V2, F2, N2))
    return out


# ---------------------------------------------------------------- bumper corner vents
HEX_FLAT, HEX_WALL, HEX_LIFT = 0.013, 0.0022, 0.006     # honeycomb cell across flats, wall, height off the floor


def chart_grid(u0, u1, y0, y1, g):
    """Regular (u, y) grid triangulation: UV (N,2), F (M,3)."""
    us = np.arange(u0, u1 + 1e-9, g); ys = np.arange(y0, y1 + 1e-9, g)
    UU, YY = np.meshgrid(us, ys); UV = np.stack([UU.ravel(), YY.ravel()], 1)
    q = np.arange(len(UV)).reshape(len(ys), len(us))
    a, b, c, d = q[:-1, :-1].ravel(), q[:-1, 1:].ravel(), q[1:, :-1].ravel(), q[1:, 1:].ravel()
    return UV, np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])


def skin(fld, ch, UV, Fg, s, off):
    """Chart-grid skin where s < 0, on the body surface, lifted off along the normals. Returns V, F, N."""
    sel = (s[Fg] < 0.003).any(1)
    used = np.unique(Fg[sel])
    P = np.zeros((len(UV), 3)); P[used] = ch.point(UV[used, 0], UV[used, 1])
    V1, F1, _, _ = cut(P, Fg[sel], s)
    return refine(fld, project(fld, V1), F1, 0.005, off)


def flat_plate(fld, ch, poly, lift, step=0.004):
    """A flat plate: the (convex) chart polygon's points on the surface, a least-squares plane through them, the
    polygon (edges subdivided every step, fanned from its centre) projected onto that plane and lifted along its
    outward normal. Returns V, F, N."""
    Q = np.array(poly, float)
    E = [Q[i] + t * (Q[(i + 1) % len(Q)] - Q[i]) for i in range(len(Q))
         for t in np.linspace(0, 1, max(int(np.linalg.norm(Q[(i + 1) % len(Q)] - Q[i]) / step), 1), endpoint=False)]
    E = np.array(E); P = ch.point(E[:, 0], E[:, 1])
    c = P.mean(0); _, _, vt = np.linalg.svd(P - c); n = vt[2]
    if n @ fld.grad(c[None])[0] < 0: n = -n
    a = vt[0]; b = np.cross(n, a)
    loop = np.stack([(P - c) @ a, (P - c) @ b], 1)
    V2 = np.vstack([loop.mean(0)[None], loop]); m = len(loop)
    F2 = np.stack([np.zeros(m, int), 1 + np.arange(m), 1 + (np.arange(m) + 1) % m], 1)
    V = c + V2[:, :1] * a + V2[:, 1:2] * b + n * lift
    fn = np.cross(V[F2[:, 1]] - V[F2[:, 0]], V[F2[:, 2]] - V[F2[:, 0]]) @ n
    if fn.sum() < 0: F2 = F2[:, ::-1]
    return V, F2, np.repeat(n[None], len(V), 0)


def vents(fld):
    """Returns [(name, mat, pivot, V, F, N)] for the passenger-side bumper corner: a black liner on the vent recess
    floor, a honeycomb grille standing HEX_LIFT off it, the red reflector, and the black corner trim (vent surround
    and lower band) with the corner's red pinstripe (designed polygons in rear_vent_gfx.json on the VENT chart)."""
    from rear_chart import VENT
    ch = Chart(fld, **VENT, dy=0.002)          # chart of the carved body: the graphics were drawn on it
    gfx = json.load(open(f'{HERE}/rear_vent_gfx.json'))
    U0, U1, Y0, Y1 = -0.20, 0.64, 0.25, 0.74
    def union(polys, uv):
        return np.min([poly_sdf(uv[:, 0], uv[:, 1], p) for p in polys], axis=0)
    def blade(uv): return poly_sdf(uv[:, 0], uv[:, 1], gfx['blade'])
    def opening(uv):
        return np.maximum(poly_sdf(uv[:, 0], uv[:, 1], gfx['opening']), -blade(uv))
    def refl(uv): return poly_sdf(uv[:, 0], uv[:, 1], gfx['reflector'])
    ln = np.array(gfx['pinstripe']['line']); wd = gfx['pinstripe']['width']
    def pin(uv):                                          # distance to the polyline minus half the width
        A = ln[:-1][None]; B = ln[1:][None]; X = uv[:, None, :]
        t = np.clip(((X - A) * (B - A)).sum(-1) / ((B - A) ** 2).sum(-1), 0, 1)
        return np.linalg.norm(X - (A + t[..., None] * (B - A)), axis=-1).min(1) - wd / 2
    out = []
    UV, Fg = chart_grid(U0, U1, Y0, Y1, 0.003)
    out.append(('Civic_Rear_Bumper_Vent_Liner', 'seam', '', *skin(fld, ch, UV, Fg, opening(UV) + 0.002, 0.001)))
    # black lining over the recess walls and rim (8 mm outside the opening, lifted 3.5 mm): the 6 mm body mesh cuts
    # the concave wall foot and pokes through a thinner skin
    out.append(('Civic_Rear_Bumper_Vent_Wall', 'seam', '', *skin(fld, ch, UV, Fg,
                np.maximum(opening(UV) - 0.008, -(opening(UV) + 0.004)), 0.0035)))
    # black trim: the surround and lower band (exact polygon edges), reaching 6 mm into the opening so it lines the
    # recess walls (hides the field's 6 mm stair-steps there); the blade stays body colour
    trim = np.maximum(union(gfx['trim'], UV), np.maximum(-(opening(UV) + 0.006), -blade(UV)))
    out.append(('Civic_Rear_Bumper_Corner_Trim', 'black', '', *skin(fld, ch, UV, Fg, trim, 0.0025)))
    UV2, Fg2 = chart_grid(U0, U1, Y0, Y1, 0.002)
    out.append(('Civic_Rear_Bumper_Vent_Reflector', 'Mat_Rear_Reflector_Red', '', *flat_plate(fld, ch, gfx['reflector'], HEX_LIFT + 0.002)))
    out.append(('Civic_Rear_Bumper_Corner_Pinstripe', 'red', '', *skin(fld, ch, UV2, Fg2, pin(UV2), 0.0026)))
    # honeycomb: one hexagonal ring per lattice cell (outer = cell boundary, inner = wall inset), clipped to the
    # opening minus the blade and the reflector
    def region(uv):
        return np.maximum(opening(uv) + 0.002, -(refl(uv) - 0.002))
    R = HEX_FLAT / np.sqrt(3)
    Ri = (HEX_FLAT / 2 - HEX_WALL / 2) / np.cos(np.pi / 6)
    ang = np.radians(30 + 60 * np.arange(6))                   # pointy-top hexagons
    allp = np.array(gfx['opening']); (u0, y0), (u1, y1) = allp.min(0) - 0.01, allp.max(0) + 0.01
    cells = [(uc, yc) for j, yc in enumerate(np.arange(y0, y1, 1.5 * R))
             for uc in np.arange(u0 + ((HEX_FLAT / 2) if j % 2 else 0.0), u1, HEX_FLAT)]
    cells = np.array(cells); cells = cells[region(cells) < HEX_FLAT]
    Vs, Fs = [], []
    for k, (uc, yc) in enumerate(cells):
        o = np.stack([uc + R * np.cos(ang), yc + R * np.sin(ang)], 1); i = np.stack([uc + Ri * np.cos(ang), yc + Ri * np.sin(ang)], 1)
        Vs.append(np.vstack([o, i])); base = 12 * k
        for m in range(6):
            n = (m + 1) % 6
            Fs += [(base + m, base + n, base + 6 + n), (base + m, base + 6 + n, base + 6 + m)]
    UVh = np.vstack(Vs); Fh = np.array(Fs)
    Ph = ch.point(UVh[:, 0], UVh[:, 1])
    V2, F2, _, (UVc,) = cut(Ph, Fh, region(UVh), [UVh])
    V2 = ch.point(UVc[:, 0], UVc[:, 1])
    out.append(('Civic_Rear_Bumper_Vent_Mesh', 'grille', '', *refine(fld, project(fld, V2), F2, 0.004, HEX_LIFT)))
    return out


# ---------------------------------------------------------------- diffuser pinstripe
# centre height of the red pinstripe along |z| in Honda's rear frame 28 (red pixels resampled on x = -2.25): flat at
# 0.452 to |z| 0.32, falling to 0.352 by 0.48, 0.348 at 0.80, 0.340 at 0.86; 10 mm wide
PIN_Z = [0.0, 0.32, 0.36, 0.40, 0.44, 0.48, 0.52, 0.535]          # outboard of |z| 0.53 the corner piece takes over
PIN_Y = [0.452, 0.452, 0.432, 0.402, 0.374, 0.356, 0.352, 0.352]
PIN_W = 0.010


def rear_face_x(fld, Y, Z):
    """x of the first surface crossing along +x from behind, for points (Y, Z) (1D arrays)."""
    xs = np.arange(-2.40, -1.70, 0.001)
    P = np.stack([np.repeat(xs[None], len(Y), 0).ravel(), np.repeat(Y, len(xs)), np.repeat(Z, len(xs))], 1)
    f = fld(P).reshape(len(Y), len(xs))
    k = np.argmax(f < 0, 1); k0 = np.maximum(k - 1, 0)
    f0 = f[np.arange(len(Y)), k0]; f1 = f[np.arange(len(Y)), k]
    return xs[k0] + np.clip(f0 / np.maximum(f0 - f1, 1e-9), 0, 1) * 0.001


def pinstripe(fld):
    """Returns [(name, mat, pivot, V, F, N)]: the red pinstripe as one ribbon across the bumper, on the rear face."""
    zs = np.arange(-PIN_Z[-1], PIN_Z[-1] + 1e-9, 0.003)
    yc = np.interp(np.abs(zs), PIN_Z, PIN_Y)
    from scipy.ndimage import gaussian_filter1d
    yc = gaussian_filter1d(yc, 4)
    rows = [yc - PIN_W / 2, yc, yc + PIN_W / 2]
    Y = np.concatenate(rows); Z = np.tile(zs, 3)
    P = np.stack([rear_face_x(fld, Y, Z), Y, Z], 1)
    P = project(fld, P)
    n = len(zs); q = np.arange(3 * n).reshape(3, n)
    a, b, c, d = q[:-1, :-1].ravel(), q[:-1, 1:].ravel(), q[1:, :-1].ravel(), q[1:, 1:].ravel()
    F = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])
    return [('Civic_Rear_Diffuser_Pinstripe', 'red', '', *refine(fld, P, F, 0.003, 0.0028))]       # above the trim skin


# ---------------------------------------------------------------- hatch bottom shut line
SHUT_ZY = [(0.0, 0.665), (0.40, 0.665), (0.418, 0.6665), (0.432, 0.674), (0.58, 0.778)]
SHUT_W = 0.0045


def hatch_shutline(fld):
    """Returns [(name, mat, pivot, V, F, N)]: the hatch bottom shut line as a dark ribbon on the rear face, from the
    driver-side lamp bottom across to the passenger side (SHUT_ZY mirrored)."""
    from scipy.ndimage import gaussian_filter1d
    zy = np.array(SHUT_ZY)
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(zy, axis=0), axis=1))])
    t = np.arange(0, L[-1], 0.003)
    half = np.stack([np.interp(t, L, zy[:, 0]), np.interp(t, L, zy[:, 1])], 1)
    line = np.concatenate([half[::-1] * [-1, 1], half[1:]])
    line[:, 1] = gaussian_filter1d(line[:, 1], 1.5)
    d = np.gradient(line, axis=0); d /= np.linalg.norm(d, axis=1, keepdims=True); nrm2 = np.stack([-d[:, 1], d[:, 0]], 1)
    rows = [line - nrm2 * SHUT_W / 2, line, line + nrm2 * SHUT_W / 2]
    Z = np.concatenate([r[:, 0] for r in rows]); Y = np.concatenate([r[:, 1] for r in rows])
    P = project(fld, np.stack([rear_face_x(fld, Y, Z), Y, Z], 1))
    n = len(line); q = np.arange(3 * n).reshape(3, n)
    a, b, c, e = q[:-1, :-1].ravel(), q[:-1, 1:].ravel(), q[1:, :-1].ravel(), q[1:, 1:].ravel()
    F = np.concatenate([np.stack([a, b, e], 1), np.stack([a, e, c], 1)])
    return [('Civic_Hatch_Shutline', 'seam', 'Pivot_Hatch_Assembly', *refine(fld, P, F, 0.003, 0.0006))]


# ---------------------------------------------------------------- high-mount stop lamp
# Honda frame 28 on x = -2.09: a black pod hanging under the spoiler bar at the centre of the lower hatch glass, a
# rounded trapezoid z +-0.168 at the bar (y 1.104) narrowing to +-0.128 at its bottom (y 1.064), with a light lens
# strip z -0.10..0.10, y 1.075..1.093 in its rear face (flush with the bar face, x -2.104)
HMSL_FRONT = [[-0.168, 1.106], [0.168, 1.106], [0.150, 1.076], [0.128, 1.064], [-0.128, 1.064], [-0.150, 1.076]]
HMSL_SIDE = [[-2.105, 1.106], [-2.102, 1.064], [-2.045, 1.058], [-2.030, 1.106]]
HMSL_LENS_FRONT = [[-0.100, 1.093], [0.100, 1.093], [0.100, 1.075], [-0.100, 1.075]]
HMSL_LENS_SIDE = [[-2.1075, 1.095], [-2.1075, 1.073], [-2.098, 1.073], [-2.098, 1.095]]


def hmsl():
    V, F, N = solid(HMSL_SIDE, HMSL_FRONT, None, ((-2.11, -2.02), (1.05, 1.11), (-0.18, 0.18)), h=0.0015, k=0.002)
    V2, F2, N2 = solid(HMSL_LENS_SIDE, HMSL_LENS_FRONT, None, ((-2.11, -2.095), (1.07, 1.10), (-0.105, 0.105)), h=0.001, k=0.0012)
    return [('Civic_Hatch_HighMount_StopLamp', 'gloss_black', 'Pivot_Hatch_Assembly', V, F, N),
            ('Civic_Hatch_HighMount_StopLamp_Lens', 'Mat_FK8_Optical_Aluminum', 'Pivot_Hatch_Assembly', V2, F2, N2)]


# ---------------------------------------------------------------- rear wiper
# Honda frame 28 resampled on x = -1.93 (US car, LHD): parked along the bottom of the upper hatch glass, pivot at the
# centre (z 0.005, y 1.146), the arm out to z -0.255 (y 1.150), the blade from z -0.10 (y 1.142) to -0.445 (y 1.133).
# The (z, y) points are put on the glass along frame 28's camera rays.
WIPER = [('arm', [(0.012, 1.146), (-0.255, 1.151)], 0.012, 0.003, 0.012),       # name, (z, y) line, width, h0, h1
         ('blade', [(-0.100, 1.142), (-0.445, 1.133)], 0.006, 0.0, 0.011)]
WIPER_PIVOT = ((0.005, 1.146), 0.014, 0.016)                                   # (z, y), radius, height


def on_body_from_cam(fld, zy, fr=28, x_plane=-1.93):
    """Surface points hit by Honda camera fr's rays through (x_plane, y, z)."""
    cams, w, h = p24.rig(); K, R, C = cams[fr]
    out = []
    for z, y in zy:
        Q = np.array([x_plane, y, z]); d = (Q - C) / np.linalg.norm(Q - C)
        ts = np.arange(np.linalg.norm(Q - C) - 0.6, np.linalg.norm(Q - C) + 0.6, 0.0005)
        f = fld(C[None] + ts[:, None] * d[None])
        k = int(np.argmax(f < 0)); t = ts[k - 1] + (ts[k] - ts[k - 1]) * f[k - 1] / (f[k - 1] - f[k])
        out.append(C + t * d)
    return np.array(out)


def sweep_box(fld, P0, P1, w, h0, h1, step=0.004):
    """Rectangular bar from surface point P0 to P1: width w across, from h0 to h1 off the surface. Flat-shaded
    sides (vertices duplicated per side). Returns V, F, N."""
    n = max(int(np.linalg.norm(P1 - P0) / step), 2)
    S = P0[None] + np.linspace(0, 1, n + 1)[:, None] * (P1 - P0)[None]
    S = project(fld, S); Nn = fld.grad(S)
    T = np.gradient(S, axis=0); T -= (T * Nn).sum(1, keepdims=True) * Nn; T /= np.linalg.norm(T, axis=1, keepdims=True)
    B = np.cross(Nn, T)
    corners = [(-w / 2, h0), (w / 2, h0), (w / 2, h1), (-w / 2, h1)]
    ring = np.stack([S + B * a + Nn * b for a, b in corners], 1)            # [n+1, 4, 3]
    V, F, N = [], [], []
    for s in range(4):                                                       # bottom, right, top, left
        a, c = s, (s + 1) % 4
        nrm = {0: -Nn, 1: B, 2: Nn, 3: -B}[s]
        base = len(V) and sum(len(v) for v in V)
        Va = np.concatenate([ring[:, a], ring[:, c]]); Na = np.concatenate([nrm, nrm])
        i = np.arange(n); k = n + 1
        Fa = np.concatenate([np.stack([i, i + 1, k + i + 1], 1), np.stack([i, k + i + 1, k + i], 1)]) + base
        V.append(Va); N.append(Na); F.append(Fa)
    for end, sgn in ((0, -1), (n, 1)):                                       # end caps
        base = sum(len(v) for v in V)
        V.append(ring[end]); N.append(np.repeat((sgn * T[end])[None], 4, 0)); F.append(np.array([[0, 1, 2], [0, 2, 3]]) + base)
    return np.concatenate(V), np.concatenate(F), np.concatenate(N)


def cylinder(fld, P, r, h, seg=20):
    """Cap cylinder standing on surface point P along its normal. Returns V, F, N."""
    P = project(fld, P[None])[0]; Nn = fld.grad(P[None])[0]
    a = np.cross(Nn, [1.0, 0, 0]); a /= np.linalg.norm(a); b = np.cross(Nn, a)
    t = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    rim = (np.cos(t)[:, None] * a + np.sin(t)[:, None] * b)
    bot = P + rim * r - Nn * 0.002; top = P + rim * r + Nn * h
    V = np.concatenate([bot, top, top, [P + Nn * h]]); N = np.concatenate([rim, rim, np.repeat(Nn[None], seg + 1, 0)])
    i = np.arange(seg); j = (i + 1) % seg
    F = np.concatenate([np.stack([i, j, seg + j], 1), np.stack([i, seg + j, seg + i], 1),
                        np.stack([2 * seg + i, 2 * seg + j, np.full(seg, 3 * seg)], 1)])
    return V, F, N


def wiper(fld):
    parts = []
    for nm, zy, w, h0, h1 in WIPER:
        Q = on_body_from_cam(fld, zy)
        parts.append(sweep_box(fld, Q[0], Q[1], w, h0, h1))
    (zp, yp), r, h = WIPER_PIVOT
    parts.append(cylinder(fld, on_body_from_cam(fld, [(zp, yp)])[0], r, h))
    V = np.concatenate([p[0] for p in parts]); N = np.concatenate([p[2] for p in parts])
    offs = np.cumsum([0] + [len(p[0]) for p in parts[:-1]])
    F = np.concatenate([p[1] + o for p, o in zip(parts, offs)])
    return [('Civic_Hatch_RearWiper', 'black', 'Pivot_Hatch_Assembly', V, F, N)]


# ---------------------------------------------------------------- roof vortex generators
# four trapezoidal fins ahead of the roof spoiler: plan = P24 top-view regions 138 (|z| 0.199..0.220) and 142
# (0.071..0.091), x -1.354..-1.277 (882..997 mm ahead of the bumper tip), mirrored; in side view they rise from 0 at
# the front to VG_H at the rear edge, standing on the roof
VG_H = 0.020


def vortex_generators(fld, fldL):
    pieces = []
    for ids in ([138], [142]):
        plan = plan_poly(ids)                                                # [[x, z], ...] passenger side
        P = np.array(plan); x0, x1 = P[:, 0].min(), P[:, 0].max(); z0, z1 = P[:, 1].min(), P[:, 1].max()
        for f, sgn in ((fld, 1), (fldL, -1)):
            h = 0.001
            xs = np.arange(x0 - 0.004, x1 + 0.004, h); zs = np.arange(z0 - 0.004, z1 + 0.004, h)
            yc = rear_face_top(f, 0.5 * (x0 + x1), 0.5 * (z0 + z1))
            ys = np.arange(yc - 0.03, yc + VG_H + 0.02, h)
            X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
            Pt = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
            d_out = f(Pt).reshape(X.shape)                                   # ~ height above the roof
            Hx = VG_H * np.clip((X - x1) / (x0 - x1), 0, 1)
            prof = np.maximum(-d_out - 0.003, d_out - Hx)
            Xp, Zp = np.meshgrid(xs, zs, indexing='ij')
            pl = poly_sdf(Xp, Zp, plan)[:, None, :]
            fv = smax(prof, pl, 0.0015)
            V, Fc, _, _ = measure.marching_cubes(fv.astype(np.float32), 0.0)
            from scipy import ndimage
            g = np.gradient(fv); Nm = np.stack([ndimage.map_coordinates(gi, V.T, order=1, mode='nearest') for gi in g], 1)
            Nm /= np.maximum(np.linalg.norm(Nm, axis=1, keepdims=True), 1e-12)
            V = V * h + [xs[0], ys[0], zs[0]]
            if sgn < 0: V, Fc, Nm = mirror(V, Fc, Nm)
            pieces.append((V, Fc, Nm))
    V = np.concatenate([p[0] for p in pieces]); N = np.concatenate([p[2] for p in pieces])
    offs = np.cumsum([0] + [len(p[0]) for p in pieces[:-1]])
    F = np.concatenate([p[1] + o for p, o in zip(pieces, offs)])
    return [('Civic_Roof_Hatch_Vortex_Generators', 'paint', '', V, F, N)]


def rear_face_top(fld, x, z):
    """Roof height at (x, z): first surface crossing coming down from above."""
    ys = np.arange(1.50, 1.20, -0.0005)
    f = fld(np.stack([np.full_like(ys, x), ys, np.full_like(ys, z)], 1))
    k = int(np.argmax(f < 0)); return ys[k - 1] + (ys[k] - ys[k - 1]) * f[k - 1] / (f[k - 1] - f[k])


# ---------------------------------------------------------------- wing, endplates, pylons
def solid(side, front, plan, box, h=0.002, k=0.003):
    """Marching-cubes mesh of smax(side(x,y), front(z,y), plan(x,z)) polygons inside box ((x0,x1),(y0,y1),(z0,z1))."""
    xs, ys, zs = (np.arange(lo - 3 * h, hi + 3 * h, h) for lo, hi in box)
    terms = []
    if side is not None:
        X, Y = np.meshgrid(xs, ys, indexing='ij'); terms.append(poly_sdf(X, Y, side)[:, :, None])
    if front is not None:
        Y, Z = np.meshgrid(ys, zs, indexing='ij'); terms.append(poly_sdf(Z, Y, front)[None, :, :])
    if plan is not None:
        X, Z = np.meshgrid(xs, zs, indexing='ij'); terms.append(poly_sdf(X, Z, plan)[:, None, :])
    f = terms[0] * np.ones((len(xs), len(ys), len(zs)))
    for t in terms[1:]: f = smax(f, t, k)
    from scipy import ndimage
    import open3d as o3d
    V, Fc, _, _ = measure.marching_cubes(f.astype(np.float32), 0.0)
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V.astype(np.float64)), o3d.utility.Vector3iVector(Fc.astype(np.int32)))
    m = m.simplify_quadric_decimation(max(len(Fc) // 8, 6000))
    m.remove_degenerate_triangles(); m.remove_unreferenced_vertices()
    V, Fc = np.asarray(m.vertices), np.asarray(m.triangles)
    g = np.gradient(f)
    N = np.stack([ndimage.map_coordinates(gi, V.T, order=1, mode='nearest') for gi in g], 1)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)            # outward (+gradient)
    V = V * h + [xs[0], ys[0], zs[0]]
    return V, Fc, N


def plan_poly(ids):
    o = outlines.outline({'name': 'p', 'view': 'top', 'ids': ids})
    return o['outer']


# wing section: P24 side region 93 (flat underside, rounded leading edge, upturned trailing edge, 37 mm thick),
# simplified to a polygon; across the span it rides on the gull line seen in Honda's frame 28: the section's top
# (1.335 in the side view) at 1.348 on the centre line, 1.335 at |z| 0.45, dropping to 1.29 at the tips (0.625)
WING_SECTION = [[-1.979, 1.2953], [-1.990, 1.3060], [-2.006, 1.3220], [-2.0286, 1.3345], [-2.1576, 1.3350],
                [-2.1618, 1.3325], [-2.1362, 1.3032], [-2.1289, 1.2982], [-2.06, 1.2965]]
WING_GULL_Z = [0.0, 0.30, 0.45, 0.55, 0.625, 0.70]
WING_GULL_DY = [0.013, 0.010, 0.0, -0.020, -0.045, -0.065]


def wing_solid(plan, h=0.0015, k=0.002):
    """Wing main plane: the side section, shifted up/down along the span by the gull line, cut to the plan."""
    from scipy import ndimage
    from scipy.interpolate import PchipInterpolator
    import open3d as o3d
    gull = PchipInterpolator(WING_GULL_Z, WING_GULL_DY)
    xs = np.arange(-2.172, -1.968, h); zs = np.arange(-0.64, 0.64 + 1e-9, h)
    ys = np.arange(1.215, 1.365, h)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    sec = poly_sdf(X, Y, WING_SECTION)                                       # [x, y] at dy = 0
    dy = gull(np.abs(zs))
    f = np.empty((len(xs), len(ys), len(zs)), np.float32)
    for kk, d in enumerate(dy):                                              # shift the section by d (bilinear in y)
        f[:, :, kk] = ndimage.shift(sec, (0, d / h), order=1, mode='nearest')
    Xp, Zp = np.meshgrid(xs, zs, indexing='ij')
    pl = poly_sdf(Xp, Zp, plan)[:, None, :]
    f = smax(f, pl, k).astype(np.float32)
    V, Fc, _, _ = measure.marching_cubes(f, 0.0)
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V.astype(np.float64)), o3d.utility.Vector3iVector(Fc.astype(np.int32)))
    m = m.simplify_quadric_decimation(max(len(Fc) // 8, 8000)); m.remove_degenerate_triangles(); m.remove_unreferenced_vertices()
    V, Fc = np.asarray(m.vertices), np.asarray(m.triangles)
    g = np.gradient(f)
    N = np.stack([ndimage.map_coordinates(gi, V.T, order=1, mode='nearest') for gi in g], 1)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    return V * h + [xs[0], ys[0], zs[0]], Fc, N


def wing_parts():
    ep_side = json.load(open(f'{HERE}/rear_endplate_profile.json'))
    py_side = json.load(open(f'{HERE}/rear_pylon_profile.json'))
    p32 = plan_poly([32])
    p7 = plan_poly([7])
    out = []
    V, F, N = wing_solid(p7)
    out.append(('Civic_Rear_Wing', 'gloss_black', 'Pivot_Hatch_Assembly', V, F, N, False))
    ep_front = [[0.695, 0.94], [0.765, 0.94], [0.600, 1.305], [0.562, 1.305]]
    V, F, N = solid(ep_side, ep_front, p32, ((-2.13, -1.64), (1.05, 1.30), (0.55, 0.78)), h=0.002, k=0.003)
    out.append(('Civic_Rear_Wing_Endplate', 'paint', 'Pivot_Hatch_Assembly', V, F, N, True))
    py_front = [[0.655, 0.94], [0.725, 0.94], [0.692, 1.10], [0.636, 1.10]]
    V, F, N = solid(py_side, py_front, p32, ((-2.00, -1.74), (0.96, 1.08), (0.62, 0.74)), h=0.002, k=0.003)
    out.append(('Civic_Rear_Wing_Pylon', 'gloss_black', 'Pivot_Hatch_Assembly', V, F, N, True))
    return out


def main():
    fld = Field(sys.argv[1], sigma=1.0)
    res, index = {}, []
    def add(name, mat, piv, V, F, N):
        res[name + '/V'] = V.astype(np.float32); res[name + '/F'] = F.astype(np.int32); res[name + '/N'] = N.astype(np.float32)
        index.append({'name': name, 'mat': mat, 'pivot': piv})
        print(f'{name:36s} {len(F):7d} faces  x[{V[:,0].min():.3f},{V[:,0].max():.3f}] y[{V[:,1].min():.3f},{V[:,1].max():.3f}] z[{V[:,2].min():.3f},{V[:,2].max():.3f}]')
    fldL = fld.mirrored()                       # driver-side parts are built on the mirrored body, then mirrored
    for nm, mat, piv, V, F, N in lamps(fld):
        add(nm + '_R', mat, piv, V, F, N)
    for nm, mat, piv, V, F, N in lamps(fldL):
        add(nm + '_L', mat, piv, *mirror(V, F, N))
    for nm, mat, piv, V, F, N in vents(fld):
        add(nm + '_R', mat, piv, V, F, N)
    for nm, mat, piv, V, F, N in vents(fldL):
        add(nm + '_L', mat, piv, *mirror(V, F, N))
    for nm, mat, piv, V, F, N in pinstripe(fld) + hatch_shutline(fld) + wiper(fld) + vortex_generators(fld, fldL) + hmsl():
        add(nm, mat, piv, V, F, N)
    for nm, mat, piv, V, F, N, mir in wing_parts():
        if mir:
            add(nm + '_R', mat, piv, V, F, N); add(nm + '_L', mat, piv, *mirror(V, F, N))
        else:
            add(nm, mat, piv, V, F, N)
    res['index'] = np.array(json.dumps(index))
    np.savez_compressed(sys.argv[2], **res)


if __name__ == '__main__':
    main()
