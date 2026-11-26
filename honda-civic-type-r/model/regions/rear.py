"""Rear region edits for the FK8 Civic Type R (hatch, wing, taillamps, bumper, exhaust).

Runs inside build.py (FFD "none": original body). Every position is taken from Honda CAD plate coordinates
(P24 side/top, 1 px = 1 mm) or from the current (post-FFD) geometry via ray casts, never from vertex indices.

Changes:
  1. Wing uprights: replaces the twisted winglet endplates with the FK8's large swept body-colour fins
     (side profile traced from plate-side). Round 2 (final cameras): A-frame cant, top inboard at |Z| ~0.6,
     foot outboard at 0.712 (rear frames 20/23/25 + plate-top foot footprint), bull-nosed rim with a fuller
     leading edge and crease, foot rim seated 4 mm into the shoulder surface instead of penetrating it.
  2. Wing main plane: drooped-nose, swept aerofoil (chord/section from plate-side, sweep from plate-top);
     round 2: span ends inside the canted upright tops, plane arcs down 60 mm towards the tips.
  3. Roof-edge vortex generators: 4 body-colour fins at Z +-0.085/+-0.215, X -1.27..-1.36 (plate-top,
     confirmed by the 28-mile car roof-edge close-up) instead of black fins out to +-0.348.
  4. Taillamps: adds the forward-pointing side wedge of the boomerang lamp on each rear quarter
     (tip at X -1.63 / Y 0.95 from plate-side): smoked border wedge, red lower arm and red lens zone up to
     the existing C light guide, conformed to the (post-FFD) body by ray casts. Parented to the body pivot.
Removes Civic_Rear_Wing_Buttresses and Civic_Rear_Wing_Mounting_Pads (now inside the new uprights).
"""
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree


def side_px(u, v):
    """Takes plate-side pixel (u, v); returns glTF (X, Y) in metres."""
    return 2.5 - u / 1000.0, 1.65 - v / 1000.0


def chaikin(P, n=2, closed=True):
    """Takes a polygon (k,2) array, iteration count and closed flag; returns the corner-cut polygon."""
    P = np.asarray(P, float)
    for _ in range(n):
        Q = []
        m = len(P) if closed else len(P) - 1
        if not closed:
            Q.append(P[0])
        for i in range(m):
            a, b = P[i], P[(i + 1) % len(P)]
            Q += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        if not closed:
            Q.append(P[-1])
        P = np.array(Q)
    return P


def world_tris(objs):
    """Takes mesh objects; returns (verts, tris) in Blender world coordinates for BVH building."""
    V, T, off = [], [], 0
    for ob in objs:
        me = ob.data
        me.calc_loop_triangles()
        co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', co)
        M = np.array(ob.matrix_world); co = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
        tri = np.empty(len(me.loop_triangles) * 3, np.int32); me.loop_triangles.foreach_get('vertices', tri)
        V.append(co); T.append(tri.reshape(-1, 3) + off); off += len(co)
    return np.concatenate(V), np.concatenate(T)


def make_bvh(objs):
    """Takes mesh objects; returns a BVHTree over their world-space triangles."""
    V, T = world_tris(objs)
    return BVHTree.FromPolygons([tuple(v) for v in V], [tuple(t) for t in T])


def set_mesh_world(ob, verts_gl, faces, mat_name=None, smooth=True, facing=None):
    """Takes an object, glTF-axis world vertices, faces, optional material and facing (None = closed solid,
    recalc outward; or a glTF direction the open patch should face); replaces its mesh. Returns ob."""
    Vb = np.stack([verts_gl[:, 0], -verts_gl[:, 2], verts_gl[:, 1]], 1)
    Minv = np.array(ob.matrix_world.inverted())
    Vl = Vb @ Minv[:3, :3].T + Minv[:3, 3]
    old = ob.data
    me = bpy.data.meshes.new(ob.name + '_mesh')
    me.from_pydata([tuple(v) for v in Vl], [], [tuple(f) for f in faces])
    me.validate()
    bm = bmesh.new(); bm.from_mesh(me)
    if facing is None:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    else:
        fb = Minv[:3, :3] @ np.array([facing[0], -facing[2], facing[1]])
        for f in bm.faces:
            if np.dot(np.array(f.normal), fb) < 0:
                f.normal_flip()
    bm.to_mesh(me); bm.free(); me.update()
    mat = bpy.data.materials.get(mat_name) if mat_name else (old.materials[0] if old and len(old.materials) else None)
    if mat:
        me.materials.append(mat)
    for p in me.polygons:
        p.use_smooth = smooth
    ob.data = me
    return ob


def new_object(name, parent, verts_gl, faces, mat_name, smooth=True, facing=None):
    """Takes a name, parent object, glTF world verts, faces, material; creates a mesh object. Returns it."""
    me = bpy.data.meshes.new(name + '_mesh')
    ob = bpy.data.objects.new(name, me)
    (parent.users_collection[0] if parent.users_collection else bpy.context.scene.collection).objects.link(ob)
    ob.parent = parent
    ob.matrix_world = Matrix.Identity(4)
    bpy.context.view_layer.update()
    return set_mesh_world(ob, verts_gl, faces, mat_name, smooth, facing)


def mirror_z(V, F):
    """Takes verts (n,3 glTF) and faces; returns the right-side mirror copy with flipped winding."""
    W = V.copy(); W[:, 2] *= -1
    return W, [tuple(reversed(f)) for f in F]


def merge(parts):
    """Takes a list of (V, F); returns one concatenated (V, F)."""
    V, F, off = [], [], 0
    for v, f in parts:
        V.append(v); F += [tuple(i + off for i in face) for face in f]; off += len(v)
    return np.concatenate(V), F


def loft_closed(rings, cap=True):
    """Takes a list of k rings, each (m,3) closed polygons; returns (V, F) of a lofted tube with end caps."""
    k, m = len(rings), len(rings[0])
    V = np.concatenate(rings)
    F = []
    for r in range(k - 1):
        for i in range(m):
            a, b = r * m + i, r * m + (i + 1) % m
            F.append((a, b, b + m, a + m))
    if cap:
        F.append(tuple(range(m))[::-1])
        F.append(tuple(range((k - 1) * m, k * m)))
    return V, F


# --------------------------------------------------------------------------------------------- wing uprights
UPRIGHT_PROFILE_PX = [  # plate-side (u, v), traced from P24 left elevation; front-low -> top -> trailing -> bottom
    (4140, 535), (4150, 529), (4380, 376), (4398, 368), (4520, 361), (4625, 353), (4634, 356), (4626, 366),
    (4600, 395), (4570, 428), (4540, 462), (4515, 495), (4500, 525), (4498, 550), (4503, 570), (4490, 574),
    (4400, 574), (4285, 574), (4200, 560), (4150, 541)]


def upright_mid(X, Y):
    """Takes X, Y arrays (glTF m); returns |Z| of the upright mid-surface. The fin is canted like an A-frame:
    top inboard at |Z| 0.585 (rear) .. 0.615 (front) where the wing joins, foot outboard at |Z| 0.712 on the
    shoulder (Clutch frame 20 / Honda 360 #28 with final cameras, plate-top foot footprint 0.665..0.75)."""
    ztop = np.interp(X, [-2.13, -1.86], [0.585, 0.615])
    s = np.clip((np.asarray(Y) - 1.10) / 0.19, 0.0, 1.15)
    return 0.712 + (ztop - 0.712) * s


def upright_half(X, Y, dle):
    """Takes X, Y arrays and distance to the leading edge dle (m); returns the fin half-thickness. 10 mm base,
    17 mm at the leading edge falling linearly to a crease 45 mm behind it, foot flared to 38 mm."""
    bump = np.clip(1.0 - dle / 0.045, 0.0, 1.0)
    foot = (1.0 - np.clip((np.asarray(Y) - 1.08) / 0.07, 0.0, 1.0)) ** 2
    return 0.010 + 0.007 * bump + 0.028 * foot


def seg_dist(P, L):
    """Takes points (n,2) and a polyline (k,2); returns each point's distance to the polyline."""
    d = np.full(len(P), np.inf)
    for a, b in zip(L[:-1], L[1:]):
        ab = b - a; t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
        d = np.minimum(d, np.linalg.norm(P - (a + t[:, None] * ab), axis=1))
    return d


def build_upright_left(bvh):
    """Takes a BVH of the body surfaces under the feet; builds the left wing upright as a closed solid with a
    bull-nosed rim (rim radius = local half-thickness, so the leading edge is full and round), a crease line
    45 mm behind the leading edge on both faces, and the foot rim seated 4 mm into the body surface.
    Returns (V, F) in glTF world axes."""
    from mathutils.geometry import delaunay_2d_cdt
    prof = chaikin(np.array([side_px(u, v) for u, v in UPRIGHT_PROFILE_PX]), 2)
    le = np.array([side_px(u, v) for u, v in UPRIGHT_PROFILE_PX[:4]])
    m = len(prof)
    t = np.roll(prof, -1, 0) - np.roll(prof, 1, 0)
    nrm = np.column_stack([-t[:, 1], t[:, 0]]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    if inside(prof, (prof + nrm * 1e-3)).sum() < m / 2:
        nrm = -nrm                                            # inward normals
    dle = seg_dist(prof, le)
    h = upright_half(prof[:, 0], prof[:, 1], dle); mid = upright_mid(prof[:, 0], prof[:, 1])
    inset = np.minimum(h, 0.012)
    th = np.linspace(0, np.pi, 9)
    rings = [np.column_stack([prof + nrm * (inset * (1 - np.sin(a)))[:, None], -(mid + h * np.cos(a))]) for a in th]
    V = [r[:, [0, 1, 2]] for r in rings]
    F = []
    for r in range(len(th) - 1):
        for i in range(m):
            a, b = r * m + i, r * m + (i + 1) % m
            F.append((a, b, b + m, a + m))
    nv = len(th) * m
    # caps: CDT over the inset outline with interior points, shared with rings 0 (outer) and last (inner)
    B = rings[0][:, :2]
    lo, hi = B.min(0), B.max(0); step = 0.012
    G = np.array([(x, y) for y in np.arange(lo[1], hi[1], step) for x in np.arange(lo[0], hi[0], step)])
    G = G[inside(B, G) & (seg_dist(G, np.vstack([B, B[:1]])) > step * 0.5)]
    vs, _, fs, ov, _, _ = delaunay_2d_cdt([Vector(p) for p in np.vstack([B, G])], [], [list(range(m))], 1, 1e-7)
    P2 = np.array([(v.x, v.y) for v in vs])
    dl = seg_dist(P2, le)
    zo = upright_mid(P2[:, 0], P2[:, 1]) + upright_half(P2[:, 0], P2[:, 1], dl)
    zi = upright_mid(P2[:, 0], P2[:, 1]) - upright_half(P2[:, 0], P2[:, 1], dl)
    extra = []
    io, ii = np.empty(len(vs), int), np.empty(len(vs), int)
    for k in range(len(vs)):
        src = [o for o in ov[k] if o < m]
        if src:
            io[k], ii[k] = src[0], (len(th) - 1) * m + src[0]
        else:
            io[k] = nv + len(extra); extra.append((P2[k, 0], P2[k, 1], -zo[k]))
            ii[k] = nv + len(extra); extra.append((P2[k, 0], P2[k, 1], -zi[k]))
    F += [tuple(io[i] for i in f) for f in fs] + [tuple(ii[i] for i in reversed(f)) for f in fs]
    V = np.concatenate(V + ([np.array(extra)] if extra else []))
    # seat the foot: bottom-edge rim points go 4 mm into the surface below; nothing else may sink deeper
    def surf(x, z):
        hit, _, _, _ = bvh.ray_cast(Vector((x, -z, 1.6)), Vector((0, 0, -1)), 1.0)
        return hit.z if hit else None
    bottom = (nrm[:, 1] > 0.35) & (prof[:, 1] < 1.17)
    for i in range(m):
        for r in range(len(th)):
            k = r * m + i
            y0 = surf(V[k, 0], V[k, 2])
            if y0 is None:
                continue
            V[k, 1] = y0 - 0.004 if bottom[i] else max(V[k, 1], y0 - 0.004)
    for k in range(nv, len(V)):
        y0 = surf(V[k, 0], V[k, 2])
        if y0 is not None:
            V[k, 1] = max(V[k, 1], y0 - 0.004)
    return V, F


# --------------------------------------------------------------------------------------------- wing main plane
WING_SECTION = np.array([  # centre section (X, Y), clockwise from LE over the top to TE and back underneath
    (-1.962, 1.297), (-1.972, 1.313), (-1.990, 1.327), (-2.020, 1.3355), (-2.070, 1.3380), (-2.120, 1.3385),
    (-2.160, 1.3380), (-2.170, 1.3345), (-2.150, 1.3300), (-2.100, 1.3250), (-2.050, 1.3180),
    (-2.010, 1.3080), (-1.980, 1.2995)])


def build_wing():
    """Builds the main plane; returns (V, F) in glTF world axes. Span ends inside the canted uprights' tops
    (|Z| = upright mid-surface per chord point); the plane arcs down 60 mm towards the tips (rear views with
    final cameras: top edge Y 1.326 centre, 1.313 at |Z| 0.4, 1.30 at 0.5, joins the upright at ~1.26)."""
    zs = np.concatenate([np.linspace(-0.585, -0.40, 9), np.linspace(-0.38, 0.38, 17), np.linspace(0.40, 0.585, 9)])
    rings = []
    for j, z in enumerate(zs):
        dx = 0.050 * (abs(z) / 0.712) ** 2                    # plan sweep: tips sit forward (plate-top)
        dy = -0.060 * (abs(z) / 0.60) ** 4.5
        X = WING_SECTION[:, 0] + dx; Y = WING_SECTION[:, 1] + dy
        Z = np.full(len(X), z)
        if j in (0, len(zs) - 1):
            Z = np.sign(z) * upright_mid(X, Y)               # end caps buried in the upright mid-surface
        rings.append(np.column_stack([X, Y, Z]))
    return loft_closed(rings)


# --------------------------------------------------------------------------------------------- taillamp side wedge
LAMP_WEDGE_PX = [  # plate-side outline of the quarter-mounted wedge: tip, top edge, inner diagonal, lower edge
    (4130, 700), (4230, 638), (4330, 578), (4448, 578), (4448, 624), (4362, 700), (4412, 864), (4350, 842), (4240, 771)]
LAMP_RED_PX = [(4190, 722), (4280, 719), (4398, 852), (4350, 836), (4240, 766), (4190, 735)]
LAMP_RED_MAIN_PX = [  # red lens between the smoked wedge's inner diagonal and the C light guide (plate-side)
    (4356, 700), (4444, 624), (4500, 584), (4535, 596), (4548, 660), (4545, 740), (4535, 815), (4552, 842),
    (4578, 858), (4530, 862), (4460, 860), (4404, 862)]   # lower corner follows the housing's lower edge


def inside(poly, P):
    """Takes polygon (k,2) and points (n,2); returns boolean mask of points inside (even-odd rule)."""
    x, y = P[:, 0], P[:, 1]; res = np.zeros(len(P), bool)
    for i in range(len(poly)):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % len(poly)]
        c = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1)
        res ^= c
    return res


def resample(poly, step):
    """Takes a closed polygon (k,2) and spacing; returns points spaced ~step along its perimeter."""
    out = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out += [a + (b - a) * k / n for k in range(n)]
    return np.array(out)


def conform_patch(poly_xy, bvh, lift, step=0.007, edge_lift=None, feather=0.012):
    """Takes a side-view polygon (X,Y), a BVH of the body and a lift (m); triangulates the polygon (CDT with
    interior grid points), projects it onto the left body side and lifts it along the surface normal; with
    edge_lift the lift eases from edge_lift at the outline to lift over `feather` m (no visible step).
    Returns (V, F) in glTF axes (left side)."""
    from mathutils.geometry import delaunay_2d_cdt
    B = resample(chaikin(poly_xy, 1), step * 0.7)
    lo, hi = B.min(0), B.max(0)
    G = np.array([(x, y) for y in np.arange(lo[1], hi[1], step) for x in np.arange(lo[0], hi[0], step)])
    d = np.min(np.linalg.norm(G[:, None] - B[None], axis=2), axis=1)
    G = G[inside(B, G) & (d > step * 0.5)]
    P = np.concatenate([B, G])
    vs, _, fs, _, _, _ = delaunay_2d_cdt([Vector(p) for p in P], [], [list(range(len(B)))], 1, 1e-6)
    Bc = np.vstack([B, B[:1]])
    V, ok = [], []
    for v in vs:
        lv = lift
        if edge_lift is not None:
            u = np.clip(seg_dist(np.array([[v.x, v.y]]), Bc)[0] / feather, 0, 1)
            lv = edge_lift + (lift - edge_lift) * u * u * (3 - 2 * u)
        o = Vector((v.x, 1.5, v.y))                           # Blender: +Y = glTF -Z (car left)
        hit, n, _, _ = bvh.ray_cast(o, Vector((0, -1, 0)), 3.0)
        if hit is None or n.y < 0.2:
            V.append((v.x, v.y, 0.0)); ok.append(False); continue
        p = hit + n * lv
        V.append((p.x, p.z, -p.y)); ok.append(True)
    F = [tuple(f) for f in fs if all(ok[i] for i in f)]
    return np.array(V), F


def build_lamp_wedges(bvh):
    """Takes a BVH of the left rear body side; returns [(V,F) smoked wedge, red arm, red main zone], both sides."""
    out = []
    for px, lift, edge in ((LAMP_WEDGE_PX, 0.0035, None), (LAMP_RED_PX, 0.0050, None), (LAMP_RED_MAIN_PX, 0.0030, 0.0008)):
        poly = np.array([side_px(u, v) for u, v in px])
        V, F = conform_patch(poly, bvh, lift, edge_lift=edge)
        out.append(merge([(V, F), mirror_z(V, F)]))
    return out


# --------------------------------------------------------------------------------------------- roof-edge VGs
VG_Z = (-0.215, -0.085, 0.085, 0.215)        # plate-top fin centres (4 fins, Honda P24 top view)
VG_X = (-1.270, -1.360)                      # plate-top fin front tip / blunt rear end


def build_vgs(bvh):
    """Takes a BVH of the roof/hatch top surfaces; returns (V, F) of the four body-colour roof-edge fins."""
    parts = []
    ang = np.linspace(0, np.pi, 9)
    for zc in VG_Z:
        rings = []
        for s in np.linspace(0, 1, 10):
            x = VG_X[0] + (VG_X[1] - VG_X[0]) * s
            hit, n, _, _ = bvh.ray_cast(Vector((x, -zc, 1.8)), Vector((0, 0, -1)), 1.0)
            y0 = hit.z if hit else 1.39
            w = 0.006 + 0.007 * s; h = 0.005 + 0.017 * s ** 0.7
            ring = [(x, y0 - 0.004, zc + w)] + [(x, y0 + h * np.sin(a), zc + w * np.cos(a)) for a in ang] + [(x, y0 - 0.004, zc - w)]
            rings.append(np.array(ring))
        parts.append(loft_closed(rings))
    return merge(parts)


def apply(ctx):
    """Takes the build context; applies all rear-region edits in place. Returns nothing."""
    body_pv = ctx.obj('Civic_Rear_Bumper').parent

    # body surfaces the lamp wedge conforms to (read only) - quarter, pillar shoulder, bumper, lamp housing
    surf = [bpy.data.objects[n] for n in ('Civic_Body_Shell_RearQuarter', 'Civic_Body_Shell_Pillars', 'Civic_Rear_Bumper',
                                          'Civic_Taillamp_Body_Housing_L', 'Civic_Taillamp_Body_C_Guide_L')]
    bvh = make_bvh(surf)

    # 1+2. wing assembly
    feet = make_bvh([bpy.data.objects[n] for n in ('Civic_Body_Shell_RearQuarter', 'Civic_Body_Shell_Pillars', 'Civic_Hatch_Panel')])
    V, F = build_upright_left(feet)
    V, F = merge([(V, F), mirror_z(V, F)])
    set_mesh_world(ctx.obj('Civic_Rear_Wing_Endplates'), V, F, 'Mat_Boost_Blue_B637P')
    V, F = build_wing()
    set_mesh_world(ctx.obj('Civic_Rear_Wing'), V, F, 'Mat_Gloss_Black_Trim')
    for n in ('Civic_Rear_Wing_Buttresses', 'Civic_Rear_Wing_Mounting_Pads'):
        bpy.data.objects.remove(ctx.obj(n), do_unlink=True)

    # 3. roof-edge vortex generators: 4 body-colour fins at the plate-top positions
    roof = make_bvh([bpy.data.objects[n] for n in ('Civic_Hatch_Panel', 'Civic_Body_Shell_Roof', 'Civic_Roof_Hatch_Header_Liner')])
    V, F = build_vgs(roof)
    set_mesh_world(ctx.obj('Civic_Roof_Hatch_Vortex_Generators'), V, F, 'Mat_Boost_Blue_B637P')

    # 4. taillamp side wedges
    (Vs, Fs), (Vr, Fr), (Vm, Fm) = build_lamp_wedges(bvh)
    for side, sgn in (('L', -1), ('R', 1)):
        for V, F, nm, mat in ((Vs, Fs, 'Smoke_Wedge', 'Mat_Headlamp_Black_Bowl'), (Vr, Fr, 'Red_Wedge', 'Mat_Taillight_Red_Lens'),
                              (Vm, Fm, 'Red_Lens', 'Mat_Taillight_Red_Lens')):
            sel = np.sign(V[:, 2]) == sgn
            idx = np.full(len(V), -1); idx[sel] = np.arange(sel.sum())
            Fk = [tuple(idx[i] for i in f) for f in F if all(sel[i] for i in f)]
            new_object(f'Civic_Taillamp_Side_{nm}_{side}', body_pv, V[sel], Fk, mat, facing=(0, 0, sgn))
