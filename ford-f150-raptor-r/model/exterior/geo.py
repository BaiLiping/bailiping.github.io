"""Small numpy geometry toolkit for the Raptor R exterior (runs inside Blender or plain Python)."""
import numpy as np

def pchip(xp, yp):
    """Monotone cubic interpolant (Fritsch-Carlson). Returns f(x) (vectorised), clamped outside."""
    xp = np.asarray(xp, float); yp = np.asarray(yp, float)
    h = np.diff(xp); d = np.diff(yp) / h
    m = np.zeros_like(yp)
    m[1:-1] = np.where(d[:-1] * d[1:] > 0, 3 * (h[:-1] + h[1:]) / ((2 * h[1:] + h[:-1]) / d[:-1] + (h[1:] + 2 * h[:-1]) / d[1:]), 0.0)
    m[0] = d[0]; m[-1] = d[-1]
    def f(x):
        x = np.clip(np.asarray(x, float), xp[0], xp[-1])
        i = np.clip(np.searchsorted(xp, x) - 1, 0, len(h) - 1)
        t = (x - xp[i]) / h[i]
        h00 = (1 + 2 * t) * (1 - t) ** 2; h10 = t * (1 - t) ** 2; h01 = t * t * (3 - 2 * t); h11 = t * t * (t - 1)
        return h00 * yp[i] + h10 * h[i] * m[i] + h01 * yp[i + 1] + h11 * h[i] * m[i + 1]
    return f

def spline(pts, n, closed=False):
    """Catmull-Rom (centripetal) resampled to n points along a 3D/2D polyline of control points."""
    P = np.asarray(pts, float)
    if closed: P = np.vstack([P[-1], P, P[0], P[1]])
    else: P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    segs = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        def tj(ti, a, b): return ti + max(np.linalg.norm(b - a), 1e-9) ** 0.5
        t0 = 0; t1 = tj(t0, p0, p1); t2 = tj(t1, p1, p2); t3 = tj(t2, p2, p3)
        ts = np.linspace(t1, t2, 24, endpoint=False)[:, None]
        A1 = (t1 - ts) / (t1 - t0) * p0 + (ts - t0) / (t1 - t0) * p1
        A2 = (t2 - ts) / (t2 - t1) * p1 + (ts - t1) / (t2 - t1) * p2
        A3 = (t3 - ts) / (t3 - t2) * p2 + (ts - t2) / (t3 - t2) * p3
        B1 = (t2 - ts) / (t2 - t0) * A1 + (ts - t0) / (t2 - t0) * A2
        B2 = (t3 - ts) / (t3 - t1) * A2 + (ts - t1) / (t3 - t1) * A3
        segs.append((t2 - ts) / (t2 - t1) * B1 + (ts - t1) / (t2 - t1) * B2)
    dense = np.vstack(segs + ([P[-2][None]] if not closed else []))
    return resample(dense, n, closed)

def resample(poly, n, closed=False):
    P = np.asarray(poly, float)
    if closed: P = np.vstack([P, P[:1]])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    t = np.linspace(0, s[-1], n + (1 if closed else 0))
    out = np.stack([np.interp(t, s, P[:, k]) for k in range(P.shape[1])], 1)
    return out[:-1] if closed else out

def coons(c0, c1, d0, d1):
    """Coons patch. c0(u) bottom, c1(u) top (nu points); d0(v) left, d1(v) right (nv points).
    Corners must agree: c0[0]=d0[0], c0[-1]=d1[0], c1[0]=d0[-1], c1[-1]=d1[-1]."""
    c0, c1, d0, d1 = map(lambda a: np.asarray(a, float), (c0, c1, d0, d1))
    nu, nv = len(c0), len(d0)
    u = np.linspace(0, 1, nu)[:, None, None]; v = np.linspace(0, 1, nv)[None, :, None]
    Lc = (1 - v) * c0[:, None, :] + v * c1[:, None, :]
    Ld = (1 - u) * d0[None, :, :] + u * d1[None, :, :]
    B = ((1 - u) * (1 - v) * c0[0] + u * (1 - v) * c0[-1] + (1 - u) * v * c1[0] + u * v * c1[-1])
    return Lc + Ld - B

def grid_faces(nu, nv, closed_u=False, closed_v=False, offset=0, flip=False):
    F = []
    for i in range(nu - (0 if closed_u else 1)):
        for j in range(nv - (0 if closed_v else 1)):
            a = i * nv + j; b = ((i + 1) % nu) * nv + j; c = ((i + 1) % nu) * nv + (j + 1) % nv; d = i * nv + (j + 1) % nv
            q = (a + offset, b + offset, c + offset, d + offset)
            F.append(q[::-1] if flip else q)
    return F

def normals_grid(P):
    """Approximate vertex normals of a structured grid P[nu,nv,3] (orientation: du x dv)."""
    du = np.gradient(P, axis=0); dv = np.gradient(P, axis=1)
    n = np.cross(du, dv); l = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.maximum(l, 1e-12)

def hemmed_grid(P, depth=0.010, roll=0.004, edges=(True, True, True, True), flip=False):
    """Add rolled-under hems around a structured outer-skin grid P[nu,nv,3].
    Normals point to du x dv (outward). Returns (verts, faces).
    edges: (u=0, u=end, v=0, v=end)."""
    nu, nv, _ = P.shape
    N = normals_grid(P)
    if flip: N = -N
    # build extended grid by adding two rings on chosen edges
    rows = [P]
    G = P.copy()
    def ring(edge_pts, inward_dir, nrm):
        a = edge_pts + inward_dir * roll * 0.15 - nrm * roll * 0.55
        b = edge_pts + inward_dir * roll * 0.5 - nrm * depth
        return a, b
    verts = [G.reshape(-1, 3)]; faces = list(grid_faces(nu, nv, flip=flip)); base = nu * nv
    def add_strip(idx_line, pts, inward, nrm, flip_s):
        nonlocal base
        a, b = ring(pts, inward, nrm)
        m = len(pts)
        verts.append(a); verts.append(b)
        ia = base + np.arange(m); ib = base + m + np.arange(m); base += 2 * m
        for k in range(m - 1):
            q1 = (idx_line[k], idx_line[k + 1], ia[k + 1], ia[k]); q2 = (ia[k], ia[k + 1], ib[k + 1], ib[k])
            if flip_s: q1 = q1[::-1]; q2 = q2[::-1]
            faces.append(q1); faces.append(q2)
    idx = np.arange(nu * nv).reshape(nu, nv)
    def unit(v):
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
    if edges[0]:
        inward = unit(P[1, :] - P[0, :]); add_strip(idx[0, :], P[0, :], -inward * 0 + inward * 0 - inward * 0 + (-inward), N[0, :], not flip)
    if edges[1]:
        inward = unit(P[-2, :] - P[-1, :]); add_strip(idx[-1, :], P[-1, :], -inward, N[-1, :], flip)
    if edges[2]:
        inward = unit(P[:, 1] - P[:, 0]); add_strip(idx[:, 0], P[:, 0], -inward, N[:, 0], flip)
    if edges[3]:
        inward = unit(P[:, -2] - P[:, -1]); add_strip(idx[:, -1], P[:, -1], -inward, N[:, -1], not flip)
    return np.vstack(verts), faces
