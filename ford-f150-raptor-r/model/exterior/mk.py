"""Blender mesh construction helpers: outline panels on analytic surfaces, sweeps, mirroring."""
import bpy, bmesh, mathutils
import numpy as np
from mathutils.geometry import delaunay_2d_cdt
from geo import resample, spline

def poly_area(P):
    P = np.asarray(P); return 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])

def inside(P, pts):
    """Even-odd point in polygon test, vectorised."""
    P = np.asarray(P); x, y = pts[:, 0], pts[:, 1]; res = np.zeros(len(pts), bool)
    j = len(P) - 1
    for i in range(len(P)):
        xi, yi = P[i]; xj, yj = P[j]
        cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi)
        res ^= cond; j = i
    return res

class Surf:
    """Analytic surface: kind 'side' (u=x,v=z -> y=F), 'top' (u=x,v=y -> z=F), 'front' (u=y,v=z -> x=F).
    sign flips the outward normal direction."""
    def __init__(self, kind, F, sign=1.0):
        self.kind, self.F, self.sign = kind, F, sign
    def pos(self, u, v):
        w = self.F(u, v)
        if self.kind == 'side': return np.stack([u, w, v], -1)
        if self.kind == 'top': return np.stack([u, v, w], -1)
        return np.stack([w, u, v], -1)
    def nrm(self, u, v, e=1e-4):
        fu = (self.F(u + e, v) - self.F(u - e, v)) / (2 * e); fv = (self.F(u, v + e) - self.F(u, v - e)) / (2 * e)
        if self.kind == 'side': n = np.stack([-fu, np.ones_like(fu), -fv], -1)
        elif self.kind == 'top': n = np.stack([-fu, -fv, np.ones_like(fu)], -1)
        else: n = np.stack([np.ones_like(fu), -fu, -fv], -1)
        n = n * self.sign
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

def outline_panel(name, outer, surf, holes=(), h=0.025, bnd=0.008, hem=0.012, roll=0.004, hem_outer=True, hem_holes=True,
                  thickness=0.0, mat=None, collection=None, offset=0.0):
    """Triangulate a 2D outline (CCW, in surface uv) with optional holes, map to the surface.
    Adds rolled hems on boundary loops. Returns the Blender object."""
    outer = np.asarray(outer, float)
    if poly_area(outer) < 0: outer = outer[::-1]
    loops = [resample(outer, max(8, int(np.ceil(perim(outer) / bnd))), closed=True)]
    hole_loops = []
    for H in holes:
        H = np.asarray(H, float)
        if poly_area(H) > 0: H = H[::-1]
        hole_loops.append(resample(H, max(8, int(np.ceil(perim(H) / bnd))), closed=True))
    allloops = loops + hole_loops
    # interior grid
    lo = outer.min(0); hi = outer.max(0)
    gu = np.arange(lo[0] + h / 2, hi[0], h); gv = np.arange(lo[1] + h / 2, hi[1], h)
    G = np.stack(np.meshgrid(gu, gv, indexing='ij'), -1).reshape(-1, 2)
    keep = inside(outer, G)
    for H in hole_loops: keep &= ~inside(H, G)
    G = G[keep]
    # drop grid points too close to boundary
    B = np.vstack(allloops)
    if len(G):
        d = np.min(np.linalg.norm(G[:, None, :] - B[None, ::3, :], axis=2), axis=1) if len(G) * len(B) < 4e7 else None
        if d is not None: G = G[d > h * 0.45]
    pts = np.vstack(allloops + ([G] if len(G) else []))
    faces_in = []; off = 0
    for L in allloops:
        faces_in.append(list(range(off, off + len(L)))); off += len(L)
    res = delaunay_2d_cdt([mathutils.Vector(p) for p in pts], [], faces_in, 3, 1e-7)
    V2 = np.array([tuple(v) for v in res[0]]); F = [list(f) for f in res[2]]
    P3 = surf.pos(V2[:, 0], V2[:, 1]); N3 = surf.nrm(V2[:, 0], V2[:, 1])
    if offset: P3 = P3 + N3 * offset
    verts = [P3]; faces = [f if True else f for f in F]; normals = [N3]
    nv = len(P3)
    # map boundary loops to output vertex indices (nearest)
    from mathutils.kdtree import KDTree
    kd = KDTree(len(V2))
    for i, p in enumerate(V2): kd.insert((p[0], p[1], 0), i)
    kd.balance()
    if hem and (hem_outer or hem_holes):
        for li, L in enumerate(allloops):
            if li == 0 and not hem_outer: continue
            if li > 0 and not hem_holes: continue
            idx = [kd.find((p[0], p[1], 0))[1] for p in L]
            Puv = V2[idx]
            n = surf.nrm(Puv[:, 0], Puv[:, 1]); p = surf.pos(Puv[:, 0], Puv[:, 1]) + (n * offset if offset else 0)
            # in-surface outward direction: tangent of loop rotated
            t = np.roll(p, -1, 0) - np.roll(p, 1, 0); t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
            out = np.cross(t, n)  # for CCW outer loop in uv with correct handedness this points outward; fix by test
            # orientation test: outward should point away from loop centroid (in 3D) for outer, toward for holes
            c = p.mean(0); sgn = np.sign(np.sum((p - c) * out))
            if li > 0: sgn = -sgn
            out *= sgn if sgn != 0 else 1
            a = p + out * roll * 0.35 - n * roll * 0.6
            b = p + out * roll * 0.55 - n * hem
            ia = nv + np.arange(len(p)); ib = ia + len(p); nv += 2 * len(p)
            verts += [a, b]; normals += [out * 0.7 - n * 0.3, out]
            m = len(p)
            for k in range(m):
                k2 = (k + 1) % m
                faces.append([idx[k], idx[k2], ia[k2], ia[k]]); faces.append([ia[k], ia[k2], ib[k2], ib[k]])
    Vall = np.vstack(verts); Nall = np.vstack(normals)
    return make_obj(name, Vall, faces, mat=mat, normals=Nall, collection=collection, thickness=thickness)

def perim(P):
    P = np.asarray(P); return float(np.sum(np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)))

def make_obj(name, V, F, mat=None, normals=None, collection=None, thickness=0.0, smooth=True, recalc=True):
    me = bpy.data.meshes.new(name)
    V = np.asarray(V, float)
    if normals is not None and len(normals) == len(V):
        # orient each face so its winding agrees with the analytic outward normal
        N = np.asarray(normals, float); F2 = []
        for f in F:
            f = [int(i) for i in f]
            p = V[f]; g = np.zeros(3)
            for k in range(len(f)):
                g += np.cross(p[k], p[(k + 1) % len(f)])
            if np.dot(g, N[f].sum(0)) < 0: f = f[::-1]
            F2.append(f)
        F = F2
    me.from_pydata([tuple(v) for v in V], [], [tuple(int(i) for i in f) for f in F])
    me.validate(clean_customdata=False)
    if recalc:
        bm = bmesh.new(); bm.from_mesh(me)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
        bm.to_mesh(me); bm.free()
    for p in me.polygons: p.use_smooth = smooth
    ob = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(ob)
    if mat is not None:
        me.materials.append(mat if not isinstance(mat, str) else bpy.data.materials.get(mat))
    if normals is not None and len(me.vertices) == len(normals):
        try:
            me.normals_split_custom_set_from_vertices([tuple(n) for n in normals])
        except Exception:
            pass
    if thickness:
        mod = ob.modifiers.new('solid', 'SOLIDIFY'); mod.thickness = thickness; mod.offset = -1
    return ob

def outward_ref(p, mode='auto'):
    """reference outward direction for exterior parts (away from the vehicle core)"""
    p = np.asarray(p, float)
    if mode == 'auto':
        c = np.stack([np.clip(p[:, 0], -2.55, 2.25), np.zeros(len(p)), np.full(len(p), 1.0)], 1)
        return p - c
    if callable(mode):
        return mode(p)
    return np.tile(np.asarray(mode, float), (len(p), 1))

def grid_obj(name, P, mat=None, collection=None, closed_u=False, closed_v=False, flip=False, normals=None, thickness=0.0, outward='auto'):
    nu, nv, _ = P.shape
    F = []
    for i in range(nu - (0 if closed_u else 1)):
        for j in range(nv - (0 if closed_v else 1)):
            a = i * nv + j; b = ((i + 1) % nu) * nv + j; c = ((i + 1) % nu) * nv + (j + 1) % nv; d = i * nv + (j + 1) % nv
            F.append((a, d, c, b) if flip else (a, b, c, d))
    V = P.reshape(-1, 3)
    if outward is not None and normals is None and len(F):
        Fa = np.array(F); p0, p1, p2 = V[Fa[:, 0]], V[Fa[:, 1]], V[Fa[:, 2]]
        fn = np.cross(p1 - p0, p2 - p0); cen = V[Fa].mean(1)
        ref = outward_ref(cen, outward)
        if np.sum(np.einsum('ij,ij->i', fn, ref)) < 0:
            F = [f[::-1] for f in F]
    return make_obj(name, V, F, mat=mat, collection=collection, normals=None if normals is None else normals.reshape(-1, 3), thickness=thickness)

def vertex_normals(me):
    """per-vertex normals from the (custom) corner normals"""
    acc = np.zeros((len(me.vertices), 3))
    cn = me.corner_normals
    for li, loop in enumerate(me.loops):
        acc[loop.vertex_index] += cn[li].vector
    l = np.linalg.norm(acc, axis=1, keepdims=True); l[l == 0] = 1
    return acc / l

def mirror_y(ob, name):
    """Create a Y-mirrored copy: mirrored positions, reversed winding, mirrored custom normals."""
    src_n = vertex_normals(ob.data)
    me = ob.data.copy(); me.name = name
    for v in me.vertices: v.co.y = -v.co.y
    me.flip_normals()
    src_n[:, 1] *= -1
    try:
        me.normals_split_custom_set_from_vertices([tuple(n) for n in src_n])
    except Exception:
        pass
    ob2 = ob.copy(); ob2.data = me; ob2.name = name
    for c in ob.users_collection: c.objects.link(ob2)
    return ob2

def extrude_profile(name, outline, axis, depth, origin=(0, 0, 0), mat=None, bevel=0.0, collection=None, segments=2):
    """Prism from a 2D outline (in plane perpendicular to axis), with optional bevel modifier."""
    P = np.asarray(outline, float)
    if poly_area(P) < 0: P = P[::-1]
    n = len(P); V = []
    for d in (0.0, depth):
        for p in P:
            if axis == 'x': V.append((origin[0] + d, origin[1] + p[0], origin[2] + p[1]))
            elif axis == 'y': V.append((origin[0] + p[0], origin[1] + d, origin[2] + p[1]))
            else: V.append((origin[0] + p[0], origin[1] + p[1], origin[2] + d))
    F = [list(range(n))[::-1], list(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n; F.append([i, j, n + j, n + i])
    ob = make_obj(name, np.array(V), F, mat=mat, collection=collection, smooth=False)
    if axis == 'y':  # orientation for y-extrusion is mirrored
        ob.data.flip_normals()
    if bevel:
        m = ob.modifiers.new('bevel', 'BEVEL'); m.width = bevel; m.segments = segments; m.limit_method = 'ANGLE'
    return ob
