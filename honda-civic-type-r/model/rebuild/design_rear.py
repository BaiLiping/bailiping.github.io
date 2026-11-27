"""Rear-end shaping of the body field (owned by the rear worker). apply(F, grid) -> F.

Steps (all for x < -1.4):
  apply_hatch_top : the P24 side outline loses the hatch glass behind the wing endplate (region 20 is excluded),
                    which left a trough between the glass and the spoiler lip. A designed solid restores the
                    glass slope down to the black spoiler bar and its lip (glass ends x -1.97 at y 1.155, bar face x -2.105).
  apply_tail      : greenhouse rolled in from the belt crease to the C-pillar width (top-view region 33).
  apply_face_smooth: hatch panel replaced by a fitted smooth surface, bumper face smoothed (hull ridges removed).
  apply_lamp_face : the taillamp corners pushed in to the flatter lamp face triangulated from Honda's frames.
  apply_recesses  : openings carved forward from the local rear face: the lower hatch glass under the lip, the
                    licence-plate area and the exhaust opening around the kept tips; the bumper corner vents
                    carved along the vent-corner chart rays (apply_vents, rear_vent_gfx.json).
"""
import numpy as np
import cv2
import p24


def _axes(F, grid):
    x0, y0, z0, h = grid
    return (x0 + h * np.arange(F.shape[0]), y0 + h * np.arange(F.shape[1]), z0 + h * np.arange(F.shape[2]))


def _smax(a, b, k):
    return 0.5 * (a + b + np.sqrt((a - b) ** 2 + k * k))


# ---- hatch top: centre-line height of the glass / bar (x, y) and its crown across the car
# centre-line heights from Honda's 36-view hull at z = 0 (glass from (-1.97, 1.155) rising at ~0.3) and frame 28
# (lower glass y 0.995..1.10, black bar 1.10..1.15 on the hatch end)
HT_X = [-1.58, -1.65, -1.75, -1.85, -1.97, -2.04, -2.07, -2.10, -2.115]
HT_Y = [1.250, 1.236, 1.214, 1.188, 1.152, 1.150, 1.140, 1.128, 1.120]
HT_CROWN, HT_CROWN_Z = 0.050, 0.62     # drop of the top toward |z| = HT_CROWN_Z
LIP_X, LIP_Y, LIP_T = -2.105, 1.095, 1.125   # bar rear face x, its underside (lower-glass top) and its top


def apply_hatch_top(F, grid):
    """Takes the body field and grid; returns it united with the designed hatch-top solid."""
    xs, ys, zs = _axes(F, grid)
    out = F.copy()
    az = np.abs(zs)
    crown = HT_CROWN * np.clip(az / HT_CROWN_Z, 0, 1.5) ** 2.4                       # [z]
    for i in np.nonzero((xs > -2.12) & (xs < -1.56))[0]:
        x = xs[i]
        yc = np.interp(-x, [-v for v in HT_X], HT_Y)
        top = (yc - crown)[None, :] - ys[:, None]                                     # >0 below the top
        rear = (x - LIP_X) - np.maximum(0.0, (LIP_Y - ys)) * 2.0 - np.maximum(0.0, ys - LIP_T) * 2.2  # >0 ahead of the bar face
        g = -np.minimum(np.minimum(top, rear[:, None] * np.ones_like(top)), (ys[:, None] - 0.98) * np.ones_like(top))
        j98 = int(round((0.98 - ys[0]) / (ys[1] - ys[0])))
        foot = az[(F[i:i + 14, j98] < 0).any(0)]                     # stay on the body below (within 8 cm ahead)
        g = np.maximum(g, (az - min(0.75, foot.max() if len(foot) else 0.0))[None, :])
        w = np.clip((x + 1.56) / -0.06, 0, 1)                                         # fade in behind x -1.56
        g = g + (1 - w) * 0.05
        out[i] = np.minimum(out[i], g)
    return out


# ---- tail greenhouse: Honda's rear views include the wing endplates, so the hull is too wide above the
# quarters. The C-pillar outer edge is the outer boundary of top-view region 33 (measured), occluded by the
# endplate behind x = -1.68 (extrapolated); the greenhouse base is the side-view 8|15 line (0.955 m).
GH_X = [-1.12, -1.21, -1.25, -1.31, -1.40, -1.50, -1.61, -1.66, -1.80, -1.95, -2.30]
GH_W = [0.640, 0.600, 0.613, 0.629, 0.637, 0.646, 0.660, 0.668, 0.678, 0.682, 0.682]
GH_YX, GH_YV = [-2.30, -1.35, -1.27, -1.00], [0.955, 0.955, 1.045, 1.045]   # base follows the rising belt line
GH_TUMBLE = 0.22


GH_ROLL = 0.20


def apply_tail(F, grid):
    """Takes the body field and grid; returns it with the tail greenhouse rolled in to the C-pillar width.
    (Sides worker's version: the C-pillar starts at the body's width at the belt (the quarter's 8|15 crease, which
    Honda's fuel door spans) and rolls in to the C-pillar width wg over GH_ROLL; nothing below the belt is limited,
    so the quarter is one continuous surface creased at the belt.)"""
    xs, ys, zs = _axes(F, grid)
    out = F.copy()
    pos = zs > 0
    for i in np.nonzero(xs < -1.05)[0]:
        w = np.clip((-1.05 - xs[i]) / 0.12, 0, 1); w = w * w * (3 - 2 * w)
        wg = np.interp(-xs[i], [-v for v in GH_X], GH_W)
        gy = np.interp(xs[i], GH_YX, GH_YV)
        j = int(round((gy - ys[0]) / (ys[1] - ys[0]))); ins = np.nonzero(F[i, j][pos] < 0)[0]
        W0 = zs[pos][ins.max()] if len(ins) else wg                      # body width at the belt (crease)
        t = np.clip((ys - gy) / GH_ROLL, 0, 1); t = t * t * (3 - 2 * t)
        Wc = np.where(ys > gy, W0 - (W0 - wg) * t - GH_TUMBLE * np.maximum(ys - gy - GH_ROLL, 0), 9.0)
        Fd = np.abs(zs)[None, :] - Wc[:, None]
        out[i] = (1 - w) * F[i] + w * np.maximum(F[i], Fd)
    return out


# ---- rear recesses
def _rear_face(F, xs):
    """Rearmost inside x for every (y, z) column (nan where the column is empty)."""
    inside = F < 0
    hit = inside.any(0)
    k = np.argmax(inside, axis=0)
    k1 = np.maximum(k - 1, 0)
    f0 = np.take_along_axis(F, k1[None], 0)[0]; f1 = np.take_along_axis(F, k[None], 0)[0]
    t = np.clip(f0 / np.maximum(f0 - f1, 1e-9), 0, 1)
    h = xs[1] - xs[0]
    return np.where(hit, xs[k1] + t * h, np.nan)


def _rrect(Y, Z, y0, y1, zw, r):
    """Signed distance (m) of a rounded rectangle y in [y0, y1], |z| < zw with corner radius r."""
    cy, hy = 0.5 * (y0 + y1), 0.5 * (y1 - y0)
    qy = np.abs(Y - cy) - (hy - r); qz = np.abs(Z) - (zw - r)
    return np.hypot(np.maximum(qy, 0), np.maximum(qz, 0)) + np.minimum(np.maximum(qy, qz), 0) - r


# lower hatch recess, traced from Honda frame 28 resampled on x = -2.14 (contrast-enhanced): the character line runs
# as a flat arc from (|z| 0.43, y 0.850) through (0.25, 0.885) to (0, 0.895); below it the floor is set back, its
# sides running down and in to (|z| 0.33, y 0.705); the floor meets the hatch bottom lip flush
PLATE_ZY = [(-0.330, 0.700), (-0.380, 0.780), (-0.430, 0.850), (-0.350, 0.872), (-0.250, 0.885), (-0.120, 0.893),
            (0.0, 0.895), (0.120, 0.893), (0.250, 0.885), (0.350, 0.872), (0.430, 0.850), (0.380, 0.780), (0.330, 0.700)]
PLATE_D = 0.012


def _poly_sdf(Y, Z, poly, ys, zs):
    """Exact signed distance (m) on the (y, z) grid of a polygon given as [(z, y), ...] (straight, even edges)."""
    from rear_parts import poly_sdf
    return poly_sdf(Z, Y, [[z, y] for z, y in poly])


def apply_recesses(F, grid):
    """Takes the body field and grid; returns it with the rear openings carved forward from the rear face."""
    xs, ys, zs = _axes(F, grid)
    xf = _rear_face(F, xs)                                                  # [y, z]
    Y, Z = np.meshgrid(ys, zs, indexing='ij')
    ok = np.isfinite(xf)
    xfz = np.where(ok, xf, -9.0)
    # lower hatch glass, set in under the lip: depth ramps in from the hatch panel top to 25 mm
    r_glass = _rrect(Y, Z, 0.995, LIP_Y, 0.51, 0.03)
    d_glass = 0.025 * np.clip((Y - 0.995) / 0.05, 0, 1) ** 0.7
    # exhaust opening around the kept triple tips (x -2.284 .. -2.122, y 0.249 .. 0.365, |z| < 0.19)
    r_exh = _rrect(Y, Z, 0.236, 0.378, 0.205, 0.045)
    d_exh = np.clip(-2.150 - xfz, 0, None)
    # licence-plate recess of the lower hatch (frame 28): a trapezoid whose top edge (y 0.875, |z| < 0.20) is a
    # sharp step with a shadow line, the sides running down and out to the hatch bottom (|z| 0.50 at y 0.69)
    # the floor eases in from the outline over w: crisp (6 mm) under the character line, soft (46 mm) at the sides, so
    # the recess has no walls for the 6 mm field grid to stair-step
    r_plate = _poly_sdf(Y, Z, PLATE_ZY, ys, zs)
    w_plate = 0.006 + 0.040 * _smoothstep((0.85 - Y) / 0.04)
    d_plate = PLATE_D * _smoothstep((Y - 0.700) / 0.120) * _smoothstep(-r_plate / w_plate)
    r_plate = r_plate - 0.0005
    out = F.copy()
    for r2, depth in ((r_glass, d_glass), (r_exh, d_exh), (r_plate, d_plate)):
        r2 = np.where(ok, r2, 1.0)
        for i in np.nonzero(xs < -1.85)[0]:
            prism = np.maximum(r2, xs[i] - (xfz + depth))                  # inside: in region, behind the floor
            out[i] = _smax(out[i], -prism, 0.004)
    return apply_vents(out, grid)


VENT_DEPTH = 0.030


class _GridField:
    """Trilinear sampling of a field array (rear_chart.Field interface)."""
    def __init__(self, F, grid):
        from scipy import ndimage
        self.F = ndimage.gaussian_filter(F, 1.0); self.o = np.array(grid[:3]); self.h = grid[3]
    def __call__(self, P):
        from scipy import ndimage
        return ndimage.map_coordinates(self.F, ((np.asarray(P) - self.o) / self.h).T, order=1, mode='nearest')


def apply_vents(F, grid):
    """Takes the body field and grid; returns it with the bumper corner vents carved VENT_DEPTH deep along the
    vent-corner chart's rays (about the surface normal), so the opening reads the same from behind and from the
    side. The opening and the body-colour blade left standing in it are rear_vent_gfx.json polygons on that chart
    (traced from Honda frames 28/25/22/34)."""
    import json, os
    from rear_chart import Chart, VENT
    from rear_parts import poly_sdf
    gfx = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rear_vent_gfx.json')))
    xs, ys, zs = _axes(F, grid)
    fld = _GridField(F, grid)
    ch = Chart(fld, **VENT, dy=0.004)
    I = np.nonzero((xs > -2.35) & (xs < -1.70))[0]; J = np.nonzero((ys > 0.42) & (ys < 0.75))[0]
    out = F.copy()
    for sgn in (1, -1):
        K = np.nonzero((sgn * zs > 0.30) & (sgn * zs < 0.98))[0]
        X, Y, Z = np.meshgrid(xs[I], ys[J], zs[K], indexing='ij')
        P = np.stack([X.ravel(), Y.ravel(), sgn * Z.ravel()], 1)           # evaluate on the passenger side
        uv = ch.uv(P)
        r = np.maximum(poly_sdf(uv[:, 0], uv[:, 1], gfx['opening']), -poly_sdf(uv[:, 0], uv[:, 1], gfx['blade']))
        r = r.reshape(X.shape)
        sub = F[np.ix_(I, J, K)]
        carve = np.maximum(r, -(sub + VENT_DEPTH))                          # within VENT_DEPTH of the surface
        out[np.ix_(I, J, K)] = _smax(sub, -carve, 0.004)
    return out


def _smoothstep(t):
    t = np.clip(t, 0, 1); return t * t * (3 - 2 * t)


_SIL = None


def _p24_rear_x(y):
    """Rearmost x of the P24 side body silhouette at height y."""
    global _SIL
    if _SIL is None: _SIL = p24.body_mask('side', arches=False)
    r = int(round((1.65 - y) * 1000)); c = np.nonzero(_SIL[r])[0]
    return 2.5 - c.max() / 1000 if len(c) else np.nan


def apply_face_smooth(F, grid, sigma=0.060):
    """Takes the body field and grid; returns it with the hatch panel and bumper face (|z| < 0.6, y 0.42..0.99)
    moved onto a smoothed copy of itself: a Gaussian-smoothed face on the bumper, a fitted low-order surface on the
    hatch panel. The visual hull leaves cone-patch ridges there that read as dents in the paint reflections. Each (y, z) column is shifted along x by the depth correction, faded in over
    x -1.85 .. -1.93 so the body ahead is untouched."""
    xs, ys, zs = _axes(F, grid)
    h = xs[1] - xs[0]
    xf = _rear_face(F, xs)
    Y, Z = np.meshgrid(ys, zs, indexing='ij')
    M = (np.isfinite(xf) & (xf < -1.95)).astype(np.float32)
    X0 = np.where(M > 0, xf, 0.0).astype(np.float32)
    s = sigma / h
    S = cv2.GaussianBlur(X0 * M, (0, 0), s) / np.maximum(cv2.GaussianBlur(M, (0, 0), s), 1e-6)
    # hatch panel: a low-order surface x(y, z) (even in z) fitted to the face, so no hull ridges remain
    t = (Y - 0.83) / 0.17; q = (Z / 0.6) ** 2
    basis = [t ** i * q ** j for i in range(4) for j in range(4)]
    fit = (M > 0) & (np.abs(Z) < 0.54) & (Y > 0.68) & (Y < 0.99)
    A = np.stack([bb[fit] for bb in basis], 1)
    coef = np.linalg.lstsq(A, xf[fit], rcond=None)[0]
    T = sum(c * bb for c, bb in zip(coef, basis))
    # the fit sits 1-2 cm forward of Honda's side silhouette on the centre line: shift the panel (uniformly across)
    # so its centre runs on the P24 outline less the plate recess carved later (apply_recesses)
    k0 = int(np.argmin(np.abs(zs)))
    sil = np.array([_p24_rear_x(y) for y in ys])
    d_rec = PLATE_D * _smoothstep((ys - 0.700) / 0.120) * (ys < 0.895)
    corr = np.where((ys > 0.69) & (ys < 0.99), (sil - d_rec) - T[:, k0], 0.0)
    corr = np.clip(np.nan_to_num(corr), -0.03, 0.01)
    from scipy.ndimage import gaussian_filter1d
    corr = gaussian_filter1d(corr, 0.02 / h) * _smoothstep((ys - 0.69) / 0.03) * _smoothstep((0.99 - ys) / 0.03)
    T = T + corr[:, None]
    Wh = _smoothstep((0.60 - np.abs(Z)) / 0.06) * _smoothstep((Y - 0.67) / 0.02) * _smoothstep((1.0 - Y) / 0.03)
    S = Wh * T + (1 - Wh) * S
    W = _smoothstep((0.66 - np.abs(Z)) / 0.08) * _smoothstep((Y - 0.40) / 0.04) * \
        np.maximum(_smoothstep((0.955 - Y) / 0.03), Wh)
    delta = np.clip(np.where(M > 0, S - xf, 0.0) * W, -0.02, 0.02)
    out = F.copy()
    J, K = np.meshgrid(np.arange(F.shape[1]), np.arange(F.shape[2]), indexing='ij')
    for i in np.nonzero(xs < -1.85)[0]:
        b = _smoothstep((-1.85 - xs[i]) / 0.08)
        src = i - delta * b / h
        lo = np.clip(np.floor(src).astype(int), 0, F.shape[0] - 2); t = np.clip(src - lo, 0, 1)
        out[i] = (1 - t) * F[lo, J, K] + t * F[lo + 1, J, K]
    return out


# ---- lamp face: the visual hull is loose on the taillamp faces (they are flatter than the hull's convex corner).
# Triangulating the lamp's inner edge between Honda frames 28/22 and 28/25 (rear_tri.py) puts it 55-75 mm inside
# the hull surface at |z| 0.55-0.60 (y 0.80-0.98), the Type R badge edge (|z| 0.53) 13-18 mm inside, while the
# lamp's outer edge on the quarter (22/19) lies on it. The rear face is pushed forward by LF_DX(|z|) over the lamp
# heights, which turns the corner into the flatter lamp face Honda's frames show (the bumper top stays out as a
# ledge below it).
# The push follows the lamp's inner edge: it rises from 0 at 45 mm inboard of the edge line LF_EDGE(y) (the inner edge
# of the lamp housing, rear_lamp_gfx.json on this surface) to its plateau 90 mm outboard, so the hatch's turn into
# the flatter lamp face lies under the lamp instead of on the paint beside it (a push from a fixed |z| put that crease
# 2-5 cm inboard of the lamp: a dark band down the hatch in the 3/4 views); outboard it falls off to the quarter.
LF_EDGE_Y, LF_EDGE_Z = [0.80, 0.86, 0.92, 0.98], [0.551, 0.568, 0.596, 0.620]
LF_IN_D = [-0.045, -0.012, 0.010, 0.050, 0.090]           # offset from the edge line (m)
LF_IN_DX = [0.0, 0.006, 0.040, 0.068, 0.075]
LF_OUT_Z, LF_OUT_DX = [0.60, 0.73, 0.78, 0.81], [0.075, 0.062, 0.022, 0.0]
LF_Y = [0.69, 0.74, 1.03, 1.08]                         # fade in, full, full, fade out


def apply_lamp_face(F, grid):
    """Takes the body field and grid; returns it with the rear face over the lamp corners moved forward."""
    xs, ys, zs = _axes(F, grid)
    h = xs[1] - xs[0]
    from scipy.interpolate import PchipInterpolator
    inner = PchipInterpolator(LF_IN_D, LF_IN_DX, extrapolate=False)
    outer = PchipInterpolator(LF_OUT_Z, LF_OUT_DX, extrapolate=False)
    ze = np.interp(ys, LF_EDGE_Y, LF_EDGE_Z)                               # [y] lamp inner edge
    d = np.abs(zs)[None, :] - ze[:, None]
    din = np.where(d < LF_IN_D[0], 0.0, np.where(d > LF_IN_D[-1], LF_IN_DX[-1], np.nan_to_num(inner(d))))
    az = np.abs(zs)
    dout = np.where(az < LF_OUT_Z[0], LF_OUT_DX[0], np.where(az > LF_OUT_Z[-1], 0.0, np.nan_to_num(outer(az))))
    wy = _smoothstep((ys - LF_Y[0]) / (LF_Y[1] - LF_Y[0])) * _smoothstep((LF_Y[3] - ys) / (LF_Y[3] - LF_Y[2]))
    delta = wy[:, None] * np.minimum(din, dout[None, :])                   # [y, z] forward shift (m)
    out = F.copy()
    J, K = np.meshgrid(np.arange(F.shape[1]), np.arange(F.shape[2]), indexing='ij')
    for i in np.nonzero(xs < -1.55)[0]:
        b = _smoothstep((-1.55 - xs[i]) / 0.10)
        src = i - delta * b / h                                            # sample from behind: face moves forward
        lo = np.clip(np.floor(src).astype(int), 0, F.shape[0] - 2); t = np.clip(src - lo, 0, 1)
        out[i] = (1 - t) * F[lo, J, K] + t * F[lo + 1, J, K]
    return out


def apply(F, grid):
    """All rear shaping steps in order."""
    F = apply_hatch_top(F, grid)
    F = apply_tail(F, grid)
    F = apply_face_smooth(F, grid)                # smooth hatch / bumper face first, then the C1 lamp-face push
    F = apply_lamp_face(F, grid)
    return apply_recesses(F, grid)
