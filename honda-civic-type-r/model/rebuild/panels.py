"""Blender-side panel builder: triangulate a drawn outline in its view plane, lift it onto the body surface
(height maps from depthmaps.py), add rolled hems along the shut lines. Geometry is built in glTF axes
(X forward, Y up, Z right) and converted to Blender axes (x, -z, y) when the object is made.
"""
import bpy, bmesh, mathutils
import numpy as np
from mathutils.geometry import delaunay_2d_cdt
from mathutils.kdtree import KDTree


class Maps:
    """Height maps: side Zs(x,y), top Yt(x,z), front Xf(z,y), rear Xr(z,y); bilinear sampling, NaN-safe."""
    def __init__(self, path):
        d = np.load(path)
        self.m = {k: d[k] for k in ('side', 'top', 'front', 'rear')}
        self.ax = {k: d[k + '_ax'] for k in ('side', 'top', 'front', 'rear')}

    def sample(self, view, a, b):
        M = self.m[view]; a0, da, b0, db = self.ax[view]
        fa = np.clip((np.asarray(a) - a0) / da, 0, M.shape[0] - 1.001); fb = np.clip((np.asarray(b) - b0) / db, 0, M.shape[1] - 1.001)
        i, j = fa.astype(int), fb.astype(int); ta, tb = fa - i, fb - j
        q = [M[i, j], M[i + 1, j], M[i, j + 1], M[i + 1, j + 1]]
        w = [(1 - ta) * (1 - tb), ta * (1 - tb), (1 - ta) * tb, ta * tb]
        num = sum(np.where(np.isfinite(qq), qq, 0) * ww for qq, ww in zip(q, w))
        den = sum(np.isfinite(qq) * ww for qq, ww in zip(q, w))
        return np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)

    def grad(self, view, a, b, e=0.003):
        return ((self.sample(view, a + e, b) - self.sample(view, a - e, b)) / (2 * e),
                (self.sample(view, a, b + e) - self.sample(view, a, b - e)) / (2 * e))


def surface(maps, view, side=+1):
    """Takes maps, view and side (+1 right / -1 left, side view only); returns (pos, nrm) functions of (a, b)
    giving glTF points and outward unit normals."""
    def unit(n): return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    if view == 'side':
        def pos(a, b): return np.stack([a, b, side * maps.sample('side', a, b)], -1)
        def nrm(a, b):
            gx, gy = maps.grad('side', a, b); return unit(np.stack([-gx, -gy, np.full_like(gx, float(side))], -1))
    elif view == 'top':
        def pos(a, b): return np.stack([a, maps.sample('top', a, b), b], -1)
        def nrm(a, b):
            gx, gz = maps.grad('top', a, b); return unit(np.stack([-gx, np.ones_like(gx), -gz], -1))
    elif view == 'front':
        def pos(a, b): return np.stack([maps.sample('front', a, b), b, a], -1)
        def nrm(a, b):
            gz, gy = maps.grad('front', a, b); return unit(np.stack([np.ones_like(gz), -gy, -gz], -1))
    else:
        def pos(a, b): return np.stack([maps.sample('rear', a, b), b, a], -1)
        def nrm(a, b):
            gz, gy = maps.grad('rear', a, b); return unit(np.stack([-np.ones_like(gz), gy, gz], -1))
    return pos, nrm


def resample_loop(P, step):
    P = np.asarray(P, float); Q = np.vstack([P, P[:1]])
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
    n = max(8, int(np.ceil(d[-1] / step)))
    t = np.linspace(0, d[-1], n, endpoint=False)
    return np.stack([np.interp(t, d, Q[:, 0]), np.interp(t, d, Q[:, 1])], 1)


def area(P):
    P = np.asarray(P); return 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])


def inside(P, pts):
    P = np.asarray(P); x, y = pts[:, 0], pts[:, 1]; res = np.zeros(len(pts), bool); j = len(P) - 1
    for i in range(len(P)):
        xi, yi = P[i]; xj, yj = P[j]
        res ^= ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi); j = i
    return res


def panel_geometry(maps, view, side, outer, holes=(), h=0.018, bnd=0.004, hem=0.010, roll=0.004, offset=0.0,
                   hem_outer=True):
    """Takes maps, view, side and outline loops (view coords); returns (V, F, N) in glTF axes:
    the surface patch plus rolled hems tucked under every boundary loop."""
    pos, nrm = surface(maps, view, side)
    outer = np.asarray(outer, float)
    if area(outer) < 0: outer = outer[::-1]
    loops = [resample_loop(outer, bnd)]
    for H in holes:
        H = np.asarray(H, float)
        if area(H) > 0: H = H[::-1]
        loops.append(resample_loop(H, bnd))
    lo, hi = outer.min(0), outer.max(0)
    G = np.stack(np.meshgrid(np.arange(lo[0] + h / 2, hi[0], h), np.arange(lo[1] + h / 2, hi[1], h), indexing='ij'), -1).reshape(-1, 2)
    keep = inside(outer, G)
    for L in loops[1:]: keep &= ~inside(L, G)
    G = G[keep]
    B = np.vstack(loops)
    if len(G):
        kd = KDTree(len(B))
        for i, p in enumerate(B): kd.insert((p[0], p[1], 0), i)
        kd.balance()
        G = np.array([g for g in G if kd.find((g[0], g[1], 0))[2] > h * 0.5]).reshape(-1, 2)
    pts = np.vstack(loops + ([G] if len(G) else []))
    faces_in = []; off = 0
    for L in loops: faces_in.append(list(range(off, off + len(L)))); off += len(L)
    res = delaunay_2d_cdt([mathutils.Vector(p) for p in pts], [], faces_in, 3, 1e-7)
    V2 = np.array([tuple(v) for v in res[0]]); F = [list(f) for f in res[2]]
    P = pos(V2[:, 0], V2[:, 1]); N = nrm(V2[:, 0], V2[:, 1])
    if offset: P = P + N * offset
    verts, normals, faces = [P], [N], list(F)
    nv = len(P)
    kd2 = KDTree(len(V2))
    for i, p in enumerate(V2): kd2.insert((p[0], p[1], 0), i)
    kd2.balance()
    if hem:
        for li, L in enumerate(loops):
            if li == 0 and not hem_outer: continue
            idx = [kd2.find((p[0], p[1], 0))[1] for p in L]
            p = P[idx]; n = N[idx]
            t = np.roll(p, -1, 0) - np.roll(p, 1, 0); t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
            out = np.cross(t, n)
            c = p.mean(0); sgn = np.sign(np.sum((p - c) * out)) or 1.0
            if li > 0: sgn = -sgn
            out *= sgn
            a = p + out * roll * 0.35 - n * roll * 0.6
            b = p + out * roll * 0.55 - n * hem
            ia = nv + np.arange(len(p)); ib = ia + len(p); nv += 2 * len(p)
            verts += [a, b]; normals += [out * 0.7 - n * 0.3, out]
            m = len(p)
            for k in range(m):
                k2 = (k + 1) % m
                faces.append([idx[k], idx[k2], ia[k2], ia[k]]); faces.append([ia[k], ia[k2], ib[k2], ib[k]])
    V = np.vstack(verts); Nn = np.vstack(normals)
    Nn /= np.maximum(np.linalg.norm(Nn, axis=1, keepdims=True), 1e-12)
    return V, faces, Nn


def to_blender(V):
    V = np.asarray(V); return np.stack([V[:, 0], -V[:, 2], V[:, 1]], 1)


def make_obj(name, V, F, N=None, mat=None, collection=None, parent=None):
    """Takes glTF-axis geometry; returns a Blender object (custom normals, faces wound to match them)."""
    Vb = to_blender(V); Nb = to_blender(N) if N is not None else None
    if Nb is not None:
        F2 = []
        for f in F:
            p = Vb[f]; g = np.zeros(3)
            for k in range(len(f)): g += np.cross(p[k], p[(k + 1) % len(f)])
            F2.append(f if np.dot(g, Nb[f].sum(0)) >= 0 else f[::-1])
        F = F2
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in Vb], [], [tuple(int(i) for i in f) for f in F])
    me.validate(clean_customdata=False)
    for p in me.polygons: p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(ob)
    if mat is not None:
        me.materials.append(mat)
    if Nb is not None and len(me.vertices) == len(Nb):
        me.normals_split_custom_set_from_vertices([tuple(n) for n in Nb])
    if parent is not None:
        mw = ob.matrix_world.copy(); ob.parent = parent; ob.matrix_parent_inverse = parent.matrix_world.inverted(); ob.matrix_world = mw
    return ob


def mirror_z(V, F, N):
    """glTF Z mirror (left <-> right): positions and normals mirrored, winding reversed."""
    V2 = np.asarray(V).copy(); V2[:, 2] *= -1
    N2 = np.asarray(N).copy(); N2[:, 2] *= -1
    return V2, [f[::-1] for f in F], N2
