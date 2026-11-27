"""Designed half-width W(x, y) of the doors and sills, from the P24 side-view creases.

The visual hull and the plan outline bound the body but never see the door section (it is concave between
the wheel-arch flares), so the doors came out as flat vertical slabs. This module defines the section from
Honda's drawn lines:
  belt line (window base) .. shoulder crease : tumblehome from the hull's greenhouse width up to the crease
  shoulder crease                            : the widest point, = plan half-width (P24 top outline)
  shoulder .. lower crease                   : slight undercut
  lower crease .. scoop bottom (14 / 24)     : the FK8 lower-door scoop, stepped in at the crease
  scoop bottom .. door bottom (48 / 17)      : lower door band
  door bottom .. skirt top (28 + 23 + 107)   : body-colour rocker, convex, just outboard of the door bottom
  skirt top .. pinstripe (71 + 22)           : black side skirt blade, stepped out (incl. the kick before the
                                               rear wheel)
  pinstripe (72) .. underside                : rounded lower edge carrying the red pinstripe, then rolled under
Over the skirt band the designed section replaces the hull (which trims the blade's lower edge near the wheels).
apply(F, grid) intersects the body field with |z| <= W in the door span (x from -0.98 to 0.90, blended over
8 cm at each end), then designs the rear quarter (apply_quarter). Returns the new field.

Rear quarter (x -0.82 .. -1.70, faded 5 cm at both ends): the 36-view hull is faceted there (cone ridges read as
a diagonal crease across the fuel flap). Honda's shoulder crease is the rising stroke inside side region 2
(rear door) continuing as the horizontal 8|15 line (y 0.949, through the fuel door: 69|44) to the taillamp top
(59|51). The section replaces the hull from the flare / bumper line (15|61, 15|10) up to 5 cm above the crease:
  flare line .. crease : a plane flank from the hull's width just above the flare line to the crease width
                         (hull at the crease + Q_CREASE_OUT), so it reads clean and slightly convex-edged
  crease .. +5 cm      : the C-pillar base tumbling in at Q_SLOPE_UP (steeper than the flank: a crisp crease)
  +5 cm .. +12 cm      : the same line only carves (never wider than the hull), fading out into the hull; only
                         behind the quarter glass (x < -1.22). On the rear door the section stops 3 cm above.
"""
import numpy as np
import p24, lines as L

D_UNDERCUT = 0.012      # shoulder -> lower crease inward drift (m)
D_STEP = 0.020          # step in at the lower crease
D_SCOOP = 0.010         # extra concavity mid-scoop
D_SILL = 0.026          # lower door band inset (door bottom)
D_ROCK = (0.022, 0.011) # rocker inset at its upper edge / at its convex middle
D_ROCK_LOW = 0.028      # rocker inset at its lower edge, so the skirt blade (and its kick) steps out crisply
D_SKIRT = 0.005         # side skirt blade inset
SKIRT_X = (-0.962, 0.952)   # skirt blade span: there the designed blade overrides the hull (it trims the blade's
SKIRT_Y = (0.168, 0.176, 0.236, 0.262)   # bottom edge near the wheels); y ramps in / full / full / ramps out
D_EDGE = 0.010          # extra roll-in of the skirt's lower (pinstripe) edge
X_FRONT, X_REAR, BLEND = 0.90, -0.98, 0.08


def curve(P, xs):
    """Takes side-plate px points of a line and x stations; returns y(x) by interpolation (clamped)."""
    Q = L.side_xy(L.smooth_curve(P, 20))
    o = np.argsort(Q[:, 0])
    return np.interp(xs, Q[o, 0], Q[o, 1])


def plan_halfwidth(xs):
    """P24 top outline half-width at x (mirrors removed), mean of both sides. Returns array."""
    m = p24.body_mask('top')
    u = np.clip(np.round((2.5 - xs) * 1000).astype(int), 0, m.shape[1] - 1)
    out = []
    for c in u:
        rows = np.nonzero(m[:, c])[0]
        out.append(0.5 * ((1.15 - rows.min() / 1000) - (1.15 - rows.max() / 1000)) if len(rows) else 0.0)
    return np.array(out)


def lines_at(xs):
    """Returns dict of crease heights y(x) at stations xs."""
    sh = np.concatenate([L.boundary('side', 11, 1), L.inner_stroke('side', 2)])
    xy = L.side_xy(sh)
    sh = sh[(xy[:, 0] > -0.55) | (xy[:, 0] < -0.80)]           # the rising stroke at the rear door's end kept
    lc = np.concatenate([L.boundary('side', 1, 14), L.boundary('side', 2, 24), L.boundary('side', 2, 17)])
    return {
        'belt': curve(np.concatenate([L.boundary('side', 62, 11), L.boundary('side', 68, 2)]), xs),
        'shoulder': curve(sh, xs),
        'lower': curve(lc, xs),
        'scoop': curve(np.concatenate([L.boundary('side', 14, 48), L.boundary('side', 24, 17)]), xs),
        'sill': curve(np.concatenate([L.boundary('side', [48, 17, 24], [28]), L.boundary('side', 17, 107)]), xs),
        'skirt': curve(np.concatenate([L.boundary('side', 23, 71), L.boundary('side', [28, 107], 22)]), xs),
        'red': curve(L.boundary('side', 22, 72), xs),
        'redb': curve(L.boundary('side', 72, [3, 111, 125, 127, 170, 172]), xs),
    }


Q_X = (-0.82, -1.70)        # quarter design span (full weight), faded Q_FADE beyond
Q_FADE = 0.05
Q_SLOPE_UP = 0.80           # C-pillar base tumblehome above the crease (dz/dy)
Q_CREASE_OUT = 0.003        # crease proud of the hull (crisper edge)
Q_REPLACE_UP, Q_CARVE_UP = 0.05, 0.12
Q_LO_FADE = 0.015


def quarter_lines(xs):
    """Takes stations xs; returns (crease height yc, lower limit ylo) from the side plate: the rising stroke in
    region 2 then the 8|15 / 69|44 / 59|51 line; the flare line 15|61 and the bumper seam 15|10."""
    r = L.inner_stroke('side', 2); r = r[L.side_xy(r)[:, 0] < -0.80]
    Q = L.side_xy(L.smooth_curve(r, 20))
    Q = np.vstack([Q, L.side_xy(L.smooth_curve(np.concatenate([L.boundary('side', 8, 15), L.boundary('side', 69, 44),
                                                                 L.boundary('side', 59, 51)]), 20))])
    o = np.argsort(Q[:, 0]); yc = np.interp(xs, Q[o, 0], Q[o, 1])
    B = L.side_xy(np.concatenate([L.boundary('side', 15, 61), L.boundary('side', 15, 10, along='v')]))
    o = np.argsort(B[:, 0]); ylo = np.interp(xs, B[o, 0], B[o, 1])
    ylo = np.where(xs > -1.17, yc - 0.10, ylo)                # on the rear door: only the crease's neighbourhood
    return yc, np.maximum(ylo, yc - 0.20)


def column_halfwidth(Fi, ys, zs):
    """Takes one x-slice of the field [y, z]; returns the outer half-width per y on the +z side (nan if empty)."""
    pos = zs > 0; Fz = Fi[:, pos]; zp = zs[pos]
    inside = Fz < 0
    k = np.where(inside.any(1), inside.shape[1] - 1 - np.argmax(inside[:, ::-1], 1), -1)
    out = np.full(len(ys), np.nan)
    ok = (k >= 0) & (k < len(zp) - 1)
    j = np.nonzero(ok)[0]; kk = k[ok]
    a, b = Fz[j, kk], Fz[j, kk + 1]
    out[j] = zp[kk] + (zp[1] - zp[0]) * a / (a - b)
    return out


def apply_quarter(F, grid, Fh):
    """Takes the field, grid and the hull field (widths); returns the field with the rear-quarter section."""
    from scipy.ndimage import gaussian_filter1d
    x0, y0, z0, h = grid
    xs = x0 + h * np.arange(F.shape[0]); ys = y0 + h * np.arange(F.shape[1]); zs = z0 + h * np.arange(F.shape[2])
    sel = np.nonzero((xs > Q_X[1] - Q_FADE) & (xs < Q_X[0] + Q_FADE))[0]
    sm = lambda a: gaussian_filter1d(np.asarray(a, float), 0.04 / h, mode='nearest')
    yc, ylo = quarter_lines(xs[sel]); yc, ylo = sm(yc), sm(ylo)
    Wc, Wlo = np.zeros(len(sel)), np.zeros(len(sel))
    for n, i in enumerate(sel):
        Wh = column_halfwidth(Fh[i], ys, zs); ok = np.isfinite(Wh)
        Wc[n] = np.interp(yc[n], ys[ok], Wh[ok]); Wlo[n] = np.interp(ylo[n] + Q_LO_FADE, ys[ok], Wh[ok])
    Wc = sm(Wc) + Q_CREASE_OUT; Wlo = sm(Wlo)
    out = F.copy()
    ss = lambda t: (lambda c: c * c * (3 - 2 * c))(np.clip(t, 0, 1))
    for n, i in enumerate(sel):
        x = xs[i]
        wx = ss(min(Q_X[0] - x, x - Q_X[1]) / Q_FADE + 1.0)
        if wx <= 0: continue
        sb = max((Wlo[n] - Wc[n]) / max(yc[n] - ylo[n] - Q_LO_FADE, 0.02), 0.05)
        W = np.where(ys < yc[n], Wc[n] + sb * (yc[n] - ys), Wc[n] - Q_SLOPE_UP * (ys - yc[n]))
        Fd = np.abs(zs)[None, :] - W[:, None]
        up = Q_REPLACE_UP if x < -1.17 else 0.03                 # on the rear door: stay under the window frame
        wr = (ss((ys - ylo[n]) / Q_LO_FADE) * (1 - ss((ys - yc[n] - up) / 0.02)))[:, None] * wx
        wq = wx * ss((-1.22 - x) / 0.05)                          # carving above only behind the quarter glass
        wc = (ss((ys - yc[n] - up + 0.02) / 0.02) * (1 - ss((ys - yc[n] - Q_CARVE_UP) / 0.03)))[:, None] * wq
        out[i] = (1 - wr) * out[i] + wr * Fd
        out[i] = (1 - wc) * out[i] + wc * np.maximum(out[i], Fd)
    return out


def env_halfwidth(F, grid, xs, ys):
    """Takes the field and grid (x0, y0, z0, h); returns the hull/plan envelope half-width at (xs[i], ys[i])."""
    x0, y0, z0, h = grid
    i = np.clip(np.round((xs - x0) / h).astype(int), 0, F.shape[0] - 1)
    j = np.clip(np.round((ys - y0) / h).astype(int), 0, F.shape[1] - 1)
    zs = z0 + h * np.arange(F.shape[2])
    out = np.zeros(len(xs))
    for n, (a, b) in enumerate(zip(i, j)):
        col = F[a, b]; k = np.nonzero(col < 0)[0]
        if len(k):
            kk = min(k.max(), len(col) - 2)
            out[n] = zs[kk] + h * col[kk] / (col[kk] - col[kk + 1]) if col[kk + 1] != col[kk] else zs[kk]
    return out


def section(ys, ln, wplan, wbelt):
    """Takes heights and the crease heights / widths at one station; returns W(y)."""
    W = np.full_like(ys, np.nan)
    sh, lc, sc, sl, sk, bt = ln['shoulder'], ln['lower'], ln['scoop'], ln['sill'], ln['skirt'], ln['belt']
    rd, rb = ln['red'], ln['redb']
    sc = min(max(sc, sl + 0.01), lc - 0.005)
    t = np.clip((ys - sh) / max(bt - sh, 1e-3), 0, 1)
    W = np.where(ys >= sh, wplan - (wplan - wbelt) * t ** 1.5, W)
    t = np.clip((sh - ys) / max(sh - lc, 1e-3), 0, 1)
    W = np.where((ys < sh) & (ys >= lc), wplan - D_UNDERCUT * t ** 1.3, W)
    t = np.clip((lc - ys) / max(lc - sc, 1e-3), 0, 1)
    scoop = wplan - D_UNDERCUT - D_STEP * (1 - t) ** 0.6 - D_SCOOP * np.sin(np.pi * t) + (D_UNDERCUT + D_STEP - D_SILL) * t ** 2
    W = np.where((ys < lc) & (ys >= sc), scoop, W)
    W = np.where((ys < sc) & (ys >= sl), wplan - D_SILL, W)
    t = np.clip((sl - ys) / max(sl - sk, 1e-3), 0, 1)                     # rocker: convex body-colour band
    e = D_ROCK[0] + (D_ROCK_LOW - D_ROCK[0]) * t
    W = np.where((ys < sl) & (ys >= sk), wplan - e + (e - D_ROCK[1]) * np.sin(np.pi * t) ** 0.7, W)
    W = np.where((ys < sk) & (ys >= rd), wplan - D_SKIRT, W)               # black skirt blade
    t = np.clip((rd - ys) / max(rd - rb, 1e-3), 0, 1)                     # pinstripe edge rolls in
    W = np.where((ys < rd) & (ys >= rb), wplan - D_SKIRT - D_EDGE * t ** 2, W)
    t = np.clip((rb - ys) / 0.02, 0, 1)
    W = np.where(ys < rb, wplan - D_SKIRT - D_EDGE - 0.05 * t ** 2, W)
    return W


def apply(F, grid):
    """Takes the body field and grid; returns the field with the designed door/sill section imposed."""
    x0, y0, z0, h = grid
    nx, ny, nz = F.shape
    xs = x0 + h * np.arange(nx); ys = y0 + h * np.arange(ny); zs = z0 + h * np.arange(nz)
    sel = np.nonzero((xs > X_REAR - BLEND) & (xs < X_FRONT + BLEND))[0]
    from scipy.ndimage import gaussian_filter1d
    sm = lambda a: gaussian_filter1d(np.asarray(a, float), 0.04 / h, mode='nearest')   # 4 cm along the car
    ln = {k: sm(v) for k, v in lines_at(xs[sel]).items()}; wp = sm(plan_halfwidth(xs[sel]))
    wb = sm(env_halfwidth(F, grid, xs[sel], ln['belt'] + 0.012))
    out = F.copy()
    for n, i in enumerate(sel):
        x = xs[i]
        w = np.clip(min((X_FRONT + BLEND - x), (x - X_REAR + BLEND)) / BLEND, 0, 1)
        w = w * w * (3 - 2 * w)
        W = section(ys, {k: v[n] for k, v in ln.items()}, wp[n], wb[n])
        Wy = np.where(np.isnan(W), 10.0, W)                       # above the belt: no constraint
        Fd = np.abs(zs)[None, :] - Wy[:, None]
        Fd = np.where(np.isnan(W)[:, None], -10.0, Fd)
        blended = np.maximum(F[i], Fd)
        out[i] = (1 - w) * F[i] + w * blended
        # the skirt blade is the designed section even where the hull trims it (near the wheels)
        wx = np.clip(min(x - SKIRT_X[0], SKIRT_X[1] - x) / 0.015, 0, 1)
        if wx > 0:
            a, b, c, d = SKIRT_Y
            wy = np.clip(np.minimum((ys - a) / (b - a), (d - ys) / (d - c)), 0, 1)[:, None]
            wo = wx * wy * w
            out[i] = (1 - wo) * out[i] + wo * Fd
    return apply_quarter(out, grid, F)
