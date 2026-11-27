"""Lamp-corner chart: an unwrapping of the body's tail corner (passenger side, +Z) into (u, y).

Around a vertical axis at (XC, ZC), the angle phi = atan2(z - ZC, XC - x) sweeps from the hatch (phi < 0, rear
face) round the corner to the quarter (phi ~ 120 deg, side face). For every height y the body surface is found
along rays from the axis; u is the surface arc length from phi = 0 at that height (metres), so the chart is close
to isometric. Used by rear_parts.py (lamp graphics) and rear_atlas.py (Honda frames unwrapped for drawing).

chart(field) -> Chart with .uv(P) for points near the surface and .point(u, y) for chart samples.
"""
import numpy as np
from scipy import ndimage

# lamp corner: axis and heights (the vent corner uses its own, see VENT)
LAMP = dict(xc=-1.80, zc=0.48, y0=0.60, y1=1.16)
VENT = dict(xc=-1.86, zc=0.40, y0=0.22, y1=0.80)
PHI = np.radians(np.arange(-60.0, 150.01, 0.25))


class Field:
    """Body field (metres, negative inside) with trilinear sampling."""
    def __init__(self, path, sigma=1.0):
        d = np.load(path); F = d['F'].astype(np.float32)
        self.F = ndimage.gaussian_filter(F, sigma) if sigma else F
        self.h = float(d['h']); self.o = np.array([float(d['x0']), float(d['y0']), float(d['z0'])])

    def __call__(self, P):
        return ndimage.map_coordinates(self.F, ((np.asarray(P) - self.o) / self.h).T, order=1, mode='nearest')

    def mirrored(self):
        """The same field mirrored left/right (z -> -z), exactly: the 6 mm grid is not symmetric about z = 0, so the
        symmetrised body is symmetric about z = -2 mm and the driver side must be sampled on its own."""
        m = Field.__new__(Field)
        m.F = self.F[:, :, ::-1].copy(); m.h = self.h
        m.o = np.array([self.o[0], self.o[1], -(self.o[2] + self.h * (self.F.shape[2] - 1))])
        return m

    def grad(self, P, e=0.003):
        g = np.stack([(self(P + d) - self(P - d)) / (2 * e) for d in np.eye(3) * e], 1)
        return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)


class Chart:
    def __init__(self, fld, xc=LAMP['xc'], zc=LAMP['zc'], y0=LAMP['y0'], y1=LAMP['y1'], dy=0.004):
        self.fld = fld; self.xc, self.zc = xc, zc
        self.YS = YS = np.arange(y0, y1 + 1e-9, dy)
        R = np.arange(0.02, 0.75, 0.002)
        P = np.zeros((len(YS), len(PHI), 3))
        for j, y in enumerate(YS):
            # rays: x = XC - r cos(phi), z = ZC + r sin(phi); outermost inside sample -> zero crossing
            cx = xc - np.cos(PHI)[:, None] * R[None]; cz = zc + np.sin(PHI)[:, None] * R[None]
            Q = np.stack([cx.ravel(), np.full(cx.size, y), cz.ravel()], 1)
            f = fld(Q).reshape(cx.shape)
            ins = f < 0
            k = np.where(ins.any(1), len(R) - 1 - np.argmax(ins[:, ::-1], 1), 0)
            k1 = np.minimum(k + 1, len(R) - 1)
            f0 = f[np.arange(len(PHI)), k]; f1 = f[np.arange(len(PHI)), k1]
            t = np.clip(f0 / np.minimum(f0 - f1, -1e-9), 0, 1)
            r = R[k] + t * (R[k1] - R[k])
            r = np.where(ins.any(1), r, np.nan)
            P[j] = np.stack([xc - np.cos(PHI) * r, np.full(len(PHI), y), zc + np.sin(PHI) * r], 1)
        self.P = P                                                     # [y, phi, 3]
        seg = np.linalg.norm(np.diff(P, axis=1), axis=2)
        seg = np.where(np.isfinite(seg), seg, 0.0)
        S = np.concatenate([np.zeros((len(YS), 1)), np.cumsum(seg, 1)], 1)
        i0 = np.argmin(np.abs(PHI))
        self.S = S - S[:, i0:i0 + 1]                                   # arc length u(phi) per row, 0 at phi = 0

    def uv(self, P):
        """Chart coordinates (u, y) of points near the surface."""
        P = np.asarray(P)
        YS = self.YS
        phi = np.arctan2(P[:, 2] - self.zc, self.xc - P[:, 0])
        fi = (phi - PHI[0]) / (PHI[1] - PHI[0]); fj = (P[:, 1] - YS[0]) / (YS[1] - YS[0])
        u = ndimage.map_coordinates(self.S, [fj, fi], order=1, mode='nearest')
        return np.stack([u, P[:, 1]], 1)

    def grid(self, US):
        """Surface points P[y, u, 3] on the chart rows (self.YS) at arc lengths US."""
        P = np.full((len(self.YS), len(US), 3), np.nan)
        for j in range(len(self.YS)):
            ok = np.isfinite(self.P[j, :, 0])
            if ok.sum() < 2: continue
            for a in range(3):
                P[j, :, a] = np.interp(US, self.S[j][ok], self.P[j, ok, a])
        return P

    def point(self, u, y):
        """3D surface points for chart samples (1D arrays u, y): bilinear in (row, arc length)."""
        u = np.asarray(u, float); y = np.asarray(y, float); YS = self.YS
        fj = np.clip((y - YS[0]) / (YS[1] - YS[0]), 0, len(YS) - 1 - 1e-9)
        j0 = np.floor(fj).astype(int); tj = fj - j0
        out = np.zeros((len(u), 3))
        idx = np.arange(len(PHI), dtype=float)
        for jj, wj in ((j0, 1 - tj), (j0 + 1, tj)):
            fi = np.empty(len(u))
            for j in np.unique(jj):
                m = jj == j
                fi[m] = np.interp(u[m], self.S[j], idx)
            i0 = np.floor(fi).astype(int).clip(0, len(PHI) - 2); ti = fi - i0
            out += wj[:, None] * ((1 - ti)[:, None] * self.P[jj, i0] + ti[:, None] * self.P[jj, i0 + 1])
        return out
