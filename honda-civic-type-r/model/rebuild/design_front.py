"""Front-end shaping of the body field: hood valley, front recesses, splitter shelf (owned by the front worker).
apply(F, grid) -> F. Only x > 1.4 is changed."""
import numpy as np
import cv2
from scipy import ndimage
import p24


# ---- hood: the FK8 bonnet is a valley between raised fender crests. The side outline only shows the crests,
# so the hood centre is set from the front view: leading edge 0.768 m (3|11), cowl 0.951 m (1|3); across the
# car it rises toward the crests (front view 20|15 reaches 0.93 m at |z| 0.74).
HOOD_X0, HOOD_X1 = 2.05, 1.10          # hood leading edge, cowl (x)
HOOD_Y0, HOOD_Y1 = 0.768, 0.951        # centre-line heights there
HOOD_ZC = 0.70                         # crest position |z|
HOOD_P = 2.2                           # centre profile exponent: 0.895 m at the scoop front (x 1.66, front 92|93)


def side_top(xs):
    """P24 side outline top height at x (scoop removed)."""
    m = p24.body_mask('side', arches=False)
    out = []
    for x in xs:
        c = int(np.clip(round((2.5 - x) * 1000), 0, m.shape[1] - 1)); r = np.nonzero(m[:, c])[0]
        out.append(1.65 - r.min() / 1000 if len(r) else 0.0)
    return np.array(out)


def apply_hood(F, grid):
    """Takes the body field and grid; returns it with the hood cut down to the designed valley surface."""
    x0, y0, z0, h = grid
    xs = x0 + h * np.arange(F.shape[0]); ys = y0 + h * np.arange(F.shape[1]); zs = z0 + h * np.arange(F.shape[2])
    sel = np.nonzero((xs > HOOD_X1 - 0.05) & (xs < HOOD_X0 + 0.10))[0]
    crest = side_top(xs[sel])
    out = F.copy()
    for n, i in enumerate(sel):
        x = xs[i]
        t = np.clip((HOOD_X0 - x) / (HOOD_X0 - HOOD_X1), 0, 1)
        hc = HOOD_Y0 + (HOOD_Y1 - HOOD_Y0) * (1 - (1 - t) ** HOOD_P)       # convex centre profile
        hc = min(hc, crest[n])
        a = np.clip(np.abs(zs) / HOOD_ZC, 0, 1)
        Hz = hc + (crest[n] - hc) * a ** 2.4                                # valley rising to the crests
        Hz = np.where(np.abs(zs) > HOOD_ZC, crest[n] + 1.0, Hz)             # outboard of the crest: no cut
        w = np.clip((x - (HOOD_X1 - 0.05)) / 0.08, 0, 1)                    # fade in behind the cowl
        Fd = ys[:, None] - Hz[None, :]
        out[i] = (1 - w) * F[i] + w * np.maximum(F[i], Fd)
    return out


# ---- hood scoop: a smooth hump on the bonnet (plan regions 24/25/106/123: plateau |z| < SCOOP_ZP, sides falling
# to 0 at SCOOP_ZE), rising from nothing at SCOOP_X0 to SCOOP_H above the bonnet just behind its front face at
# SCOOP_XF (front view: scoop top 0.941 over a bonnet at 0.895), then a steep front face carrying the mouth
# (front region 45, recessed in FRONT_RECESS). Only x > 1.4 is touched.
SCOOP_X0, SCOOP_XF, SCOOP_XE = 1.42, 1.655, 1.676
SCOOP_ZP, SCOOP_ZE, SCOOP_H = 0.150, 0.228, 0.048


def _smooth(t):
    t = np.clip(t, 0, 1); return t * t * (3 - 2 * t)


def press_hood_ridges(points, normals):
    """Stamp the paired hood ridges across every adjoining piece of skin.

    Apply the same displacement at a shared position, including inlays and
    fender edges. Its spatial derivatives tilt the existing shading normals.
    """
    x, y, z = points.T
    selected = (x > 1.125) & (x < 2.015) & (y > 0.68)
    if not selected.any():
        return points, normals
    x, y, z = points[selected].T

    def height(x, y, z):
        center = .445 + .205 * (1.96 - x)
        across = np.abs(z) - center
        width = np.where(across < 0, .068, .024)
        ends = np.clip((2.015 - x) / .17, 0, 1) * np.clip((x - 1.125) / .12, 0, 1)
        return .017 * _smooth(ends) * _smooth((y - .68) / .05) * np.exp(-.5 * (across / width)**2)

    step = .0005
    dx = (height(x + step, y, z) - height(x - step, y, z)) / (2 * step)
    dy = (height(x, y + step, z) - height(x, y - step, z)) / (2 * step)
    dz = (height(x, y, z + step) - height(x, y, z - step)) / (2 * step)
    result = points.copy()
    result[selected, 1] += height(x, y, z)
    shading = normals.copy()
    n = normals[selected]
    tilted = (1 + dy[:, None]) * n - n[:, 1:2] * np.column_stack([dx, dy, dz])
    shading[selected] = tilted / np.maximum(np.linalg.norm(tilted, axis=1, keepdims=True), 1e-12)
    return result, shading


def apply_scoop(F, grid):
    """Takes the body field and grid; returns it with the scoop hump added on top of the bonnet."""
    x0, y0, z0, h = grid
    xs = x0 + h * np.arange(F.shape[0]); ys = y0 + h * np.arange(F.shape[1]); zs = z0 + h * np.arange(F.shape[2])
    az = np.abs(zs)
    w = np.where(az <= SCOOP_ZP, 1.0, 0.5 * (1 + np.cos(np.pi * np.clip((az - SCOOP_ZP) / (SCOOP_ZE - SCOOP_ZP), 0, 1))))
    out = F.copy()
    for i in np.nonzero((xs > SCOOP_X0) & (xs < SCOOP_XE + 0.02))[0]:
        x = xs[i]
        Hx = SCOOP_H * _smooth((x - SCOOP_X0) / (SCOOP_XF - 0.03 - SCOOP_X0)) * (1 - _smooth((x - SCOOP_XF) / (SCOOP_XE - SCOOP_XF)))
        if Hx <= 1e-4: continue
        col = F[i]                                                     # [y, z]
        inside = col < 0
        k = (len(ys) - 1) - np.argmax(inside[::-1], axis=0)            # topmost inside sample per z
        top = np.where(inside.any(0), ys[k], -1.0)
        new_top = top + Hx * w
        hump = ys[:, None] - new_top[None, :]
        hump = np.where((w[None, :] > 0) & (ys[:, None] > top[None, :] - 0.03), hump, 1.0)
        kk = 0.006
        out[i] = 0.5 * (col + hump - np.sqrt((col - hump) ** 2 + kk * kk))   # smooth union
    return out


# ---- front recesses: openings of Honda's front drawing (image-left ids, mirrored) carved back from the local
# frontmost face by their depth (m). Each footprint is shrunk by INSET (more than half a drawn stroke) so the
# opening's side walls fall inside the panel that the opening's material claims (cut_front.json uses a permissive facing test there).
# Optional y limits restrict a footprint (the US car has no plate: only the lower part of box 6 is grille).
FRONT_RECESS = [
    # upper grille, nested so that no un-carved stroke ridges stand between its terraces
    # (the brow bar 47 with its slots stays at the surface: a smooth gloss-black bar under the hood lip)
    ([41, 19, 46, 10], 0.012, None),                       # wings level
    ([19, 46, 10], 0.030, None),                           # grille top band and the emblem seat
    ([10], 0.040, None),                                   # honeycomb floor
    ([4], 0.050, None),                                    # lower grille
    ([4, 6], 0.050, (0.0, 0.385)),                         # with the lower part of the plate box (US: no plate)
    ([12, 25, 54, 60, 75], 0.045, None),                   # fog pod (blade 54 and fog lamp 60 are parts)
    ([75], 0.065, None),                                   # pod inner slot
    ([63, 66], 0.015, None),                               # lower bumper strakes
    ([45], 0.030, None),                                   # hood scoop mouth
]
INSET = 0.006


def front_face(F, grid):
    """Frontmost surface x for every (y, z) field column, sub-voxel and lightly smoothed; -9 where missed."""
    x0, y0, z0, h = grid
    xs = x0 + h * np.arange(F.shape[0])
    A = F[::-1]
    inside = A < 0
    hit = inside.any(0)
    k = np.clip(np.argmax(inside, 0), 1, A.shape[0] - 1)
    a = np.take_along_axis(A, (k - 1)[None], 0)[0]; b = np.take_along_axis(A, k[None], 0)[0]
    t = a / np.maximum(a - b, 1e-9)
    xf = xs[-1] - h * ((k - 1) + np.clip(t, 0, 1))
    xf = np.where(hit, xf, np.nan)
    filled = np.where(hit, xf, np.nanmedian(xf[hit]))
    sm = ndimage.gaussian_filter(filled, 1.2)
    return np.where(hit, sm, -9.0)


def region_sdf(ids, ylim=None):
    """Front-plate signed distance (m) of merged image-left regions, mirrored; optional y limits."""
    import outlines
    m = outlines.region_mask('front', ids, sym=True)
    if ylim is not None:
        vv = np.arange(m.shape[0])[:, None]
        y = 1.65 - vv / 1000.0
        m = m & (y >= ylim[0]) & (y <= ylim[1])
    return p24.sdf2d(m) / 1000.0


def apply_front_recesses(F, grid):
    """Takes the body field and grid; returns it with the front openings recessed."""
    x0, y0, z0, h = grid
    xs = x0 + h * np.arange(F.shape[0]); ys = y0 + h * np.arange(F.shape[1]); zs = z0 + h * np.arange(F.shape[2])
    xface = front_face(F, grid)                                          # [y, z]
    sel = np.nonzero(xs > 1.55)[0]
    Y, Z = np.meshgrid(ys, zs, indexing='ij')
    u, v = p24.to_px('front', 0 * Y, Y, Z)
    out = F.copy()
    for ids, depth, ylim in FRONT_RECESS:
        r2 = p24.sample(region_sdf(ids, ylim), u, v, outside=1.0) + INSET   # [y, z] membership
        if (r2 < 0).sum() == 0: continue
        floor = xface - depth
        for i in sel:
            prism = np.maximum(r2, floor - xs[i])                          # inside: in region and in front of the floor
            out[i] = 0.5 * (out[i] - prism + np.sqrt((out[i] + prism) ** 2 + 0.004 ** 2))   # smooth max(F, -prism)
    return out


# ---- splitter shelf: a crisp black lip at the bottom of the nose. The rounded 3-view intersection of the
# envelope rounds the side-view tip (x 2.287 at y 0.20-0.22) back by ~3 cm; this adds the shelf as a slab
# inside the plan outline between SPLIT_Y.
SPLIT_Y = (0.203, 0.236)


def apply_splitter(F, grid):
    """Takes the body field and grid; returns it united with the splitter slab."""
    x0, y0, z0, h = grid
    xs = x0 + h * np.arange(F.shape[0]); ys = y0 + h * np.arange(F.shape[1]); zs = z0 + h * np.arange(F.shape[2])
    top = p24.sdf2d(p24.body_mask('top')) / 1000.0
    sel = np.nonzero(xs > 1.80)[0]
    yc, yh = 0.5 * (SPLIT_Y[0] + SPLIT_Y[1]), 0.5 * (SPLIT_Y[1] - SPLIT_Y[0])
    fy = np.abs(ys - yc) - yh                                              # [y]
    out = F.copy()
    for i in sel:
        u, v = p24.to_px('top', np.full_like(zs, xs[i]), 0 * zs, zs)
        pz = p24.sample(top, u, v, outside=1.0) + 0.002                   # [z] inside the plan outline
        slab = np.maximum(np.maximum(fy[:, None], pz[None, :]), 1.85 - xs[i])
        k = 0.004
        a = out[i]
        out[i] = 0.5 * (a + slab - np.sqrt((a - slab) ** 2 + k * k))        # smooth min (union)
    return out


def apply(F, grid):
    """All front shaping steps in order."""
    F = apply_hood(F, grid)
    F = apply_scoop(F, grid)
    F = apply_splitter(F, grid)
    return apply_front_recesses(F, grid)
