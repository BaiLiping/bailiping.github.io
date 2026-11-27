"""Measure feature lines on the P24 plates: the stroke between two region sets, or a free stroke inside a region.

boundary(view, A, B, along='u'): stroke centreline between regions A and B. along='u' scans columns (A above,
  B below; for roughly horizontal lines), along='v' scans rows (A left, B right). Returns (N, 2) plate px.
inner_stroke(view, region, along='u'): centreline of a stroke whose two sides both belong to `region`
  (a line drawn inside a region that does not close it). Returns (N, 2) plate px.
side_xy(px): side-plate px -> glTF (X, Y) metres.  top_xz(px): top px -> (X, Z).  front_zy(px): front px -> (Z, Y).
"""
import numpy as np
import p24


def _scan(lab, A, B, along, maxgap=14):
    A, B = set(np.atleast_1d(A).tolist()), set(np.atleast_1d(B).tolist())
    L = lab if along == 'u' else lab.T
    out = []
    for u in range(L.shape[1]):
        col = L[:, u]
        nz = np.nonzero(col != 0)[0]
        if len(nz) < 2: continue
        a, b = nz[:-1], nz[1:]
        gap = b - a
        la, lb = col[a], col[b]
        ok = (gap > 1) & (gap <= maxgap) & np.isin(la, list(A)) & np.isin(lb, list(B))
        for k in np.nonzero(ok)[0]:
            out.append((u, 0.5 * (a[k] + b[k])))
    P = np.array(out, float).reshape(-1, 2)
    return P if along == 'u' else P[:, ::-1]


def boundary(view, A, B, along='u', maxgap=14):
    """Stroke centreline between region sets A and B (see module doc). Returns (N, 2) plate px (u, v)."""
    return _scan(p24.labels(view), A, B, along, maxgap)


def inner_stroke(view, region, along='u', maxgap=14):
    """Centreline of a dangling stroke with `region` on both sides. Returns (N, 2) plate px (u, v)."""
    return _scan(p24.labels(view), region, region, along, maxgap)


def smooth_curve(P, step=10.0, along=0):
    """Takes px points and the scan axis; returns a curve resampled every `step` px along that axis
    (median per bin, then light smoothing). Returns (M, 2)."""
    if len(P) == 0: return P
    k = np.round(P[:, along] / step).astype(int)
    keys = np.unique(k)
    Q = np.array([[np.median(P[k == q, 0]), np.median(P[k == q, 1])] for q in keys])
    if len(Q) > 4:
        o = 1 - along
        Q[1:-1, o] = (Q[:-2, o] + 2 * Q[1:-1, o] + Q[2:, o]) / 4
    return Q


def side_xy(P): return np.stack([2.5 - P[:, 0] / 1000, 1.65 - P[:, 1] / 1000], 1)
def top_xz(P): return np.stack([2.5 - P[:, 0] / 1000, 1.15 - P[:, 1] / 1000], 1)
def front_zy(P): return np.stack([1.15 - P[:, 0] / 1000, 1.65 - P[:, 1] / 1000], 1)
