"""Front-region refinement for the 2020 FK8 Civic Type R (facelift front), run by build.py after the FFD fit.

All targets come from Honda's P24 CAD plates (plate-front/top/side.png, 1 px = 1 mm, glTF metres) cross-checked
against the 2020-21 facelift photos (BaT championship-white 02/08, Honda P09 studio front, Clutch 2020 frame 03).
Positions are derived from the current (post-FFD) geometry or from CAD coordinates, never from vertex indices.

Changes:
  1. Hood: adds the two bonnet character lines (CAD top plate line from the hood front corner X 2.00/|Z| .443
     back to |Z| .546 at X 1.37 and hooking into the cowl at X 1.19) as a crisp ridge on a locally refined mesh.
  2. Headlamps: lens/housing/DRL/projectors stretched upward so the lamp top follows the CAD outline (high and
     nearly level to |Z| .54, then dropping steeply to the inner corner) instead of a thin sliver; the gloss-black
     mask behind the new lens area is pushed back (a thin strip of it stays above the lens as the black brow).
  3. Upper grille: gloss-black frame so the honeycomb shows only inside the CAD trapezoid (top Y .628, sides
     |Z| .368 -> .335, bottom .540); the black bar above gets the CAD upper slot (Y .6925-.7175) with 6 dividers;
     the Type R script moves up onto the honeycomb (photos: just under the badge line).
  4. Lower bumper: fog-pod body-colour blades lifted 1-2.5 cm onto the CAD diagonal (outer-top corner down to the
     inner end); new gloss-black lower bar under the grille (CAD Y .468-.516) as on the 2020 car.
  5. Round 2: fog lamps shrunk (x0.79) and lifted 2 cm onto the CAD circle (|Z| .724, Y .372, r .036) with the bezel
     ring/insert warped to follow; hood scoop opening bottom raised from Y .874 to .891; separate gloss-black brow
     strips on the ledge face over each lamp; matte-black filler sheet closing front-view see-through gaps into the
     engine bay (under the lamps between mask and cheeks, and the ledge slit at |Z| .72-.77, Y .76).
"""
import bpy, bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

# ---------------------------------------------------------------- CAD data (glTF metres, |Z| = lateral distance)
HOOD_LINE = np.array([(2.030, 0.438), (2.000, 0.443), (1.800, 0.476), (1.600, 0.509), (1.450, 0.535),
                      (1.400, 0.543), (1.370, 0.546), (1.330, 0.541), (1.280, 0.526), (1.240, 0.508),
                      (1.190, 0.4875)])                                   # (X, |Z|), front -> rear
LAMP_TOP = np.array([(0.440, 0.640), (0.455, 0.675), (0.470, 0.700), (0.500, 0.725), (0.540, 0.740),
                     (0.650, 0.755), (0.750, 0.768), (0.835, 0.780), (0.900, 0.785)])   # (|Z|, Y) CAD lens top
HONEYCOMB = [(0.0, 0.568), (0.060, 0.568), (0.068, 0.600), (0.080, 0.622), (0.100, 0.628), (0.355, 0.628),
             (0.368, 0.621), (0.345, 0.560), (0.335, 0.543), (0.320, 0.540), (0.0, 0.540)]  # (|Z|, Y), +Z half
SLOT = [(0.0, 0.7175), (0.375, 0.7175), (0.368, 0.7075), (0.357, 0.6985), (0.345, 0.6925), (0.0, 0.6925)]
SLOT_DIVIDERS = (0.032, 0.163, 0.276)


# ---------------------------------------------------------------- generic helpers
def wco(ob):
    """Takes a mesh object; returns its vertex positions in world Blender coordinates (n,3)."""
    co = np.empty(len(ob.data.vertices) * 3); ob.data.vertices.foreach_get('co', co)
    M = np.array(ob.matrix_world); return co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]


def set_wco(ob, P):
    """Takes a mesh object and world Blender positions (n,3); writes them back as local coordinates. Returns nothing."""
    Mi = np.array(ob.matrix_world.inverted())
    ob.data.vertices.foreach_set('co', (P @ Mi[:3, :3].T + Mi[:3, 3]).ravel()); ob.data.update()


def clear_custom_normals(ob):
    """Takes a mesh object; drops its imported custom split normals so shading follows the edited shape. Returns nothing."""
    with bpy.context.temp_override(object=ob, active_object=ob, selected_objects=[ob], selected_editable_objects=[ob]):
        bpy.ops.mesh.customdata_custom_splitnormals_clear()


def bvh_of(obs):
    """Takes mesh objects; returns a BVHTree of their world-space polygons."""
    verts, polys, off = [], [], 0
    for ob in obs:
        P = wco(ob); verts.extend(map(Vector, P))
        polys.extend([[i + off for i in p.vertices] for p in ob.data.polygons]); off += len(P)
    return BVHTree.FromPolygons(verts, polys)


def front_x(bvh, z_gl, y_gl, x0=3.0):
    """Takes a BVH and glTF lateral/height arrays; returns the x of the first surface hit by a ray along -X (nan = miss)."""
    out = np.full(len(z_gl), np.nan)
    for i, (z, y) in enumerate(zip(z_gl, y_gl)):
        loc, _, _, _ = bvh.ray_cast(Vector((x0, -z, y)), Vector((-1, 0, 0)))
        if loc is not None:
            out[i] = loc.x
    return out


def smoothstep(t):
    """Takes an array; returns the cubic smoothstep of it clipped to [0, 1]."""
    t = np.clip(t, 0, 1); return t * t * (3 - 2 * t)


def polyline_sdist(q, P):
    """Takes 2D points q (n,2) and an ordered polyline P (m,2) in (X, |Z|); returns (signed distance, arclength param,
    total length). Negative = on the inboard (smaller |Z|) side when walking front -> rear."""
    seg = P[1:] - P[:-1]; L = np.linalg.norm(seg, axis=1); cum = np.concatenate([[0], np.cumsum(L)])
    d = q[:, None, :] - P[None, :-1, :]
    t = np.clip((d * seg[None]).sum(2) / (L ** 2)[None], 0, 1)
    proj = P[None, :-1, :] + t[..., None] * seg[None]
    dist = np.linalg.norm(q[:, None, :] - proj, axis=2); k = dist.argmin(1); r = np.arange(len(q))
    cr = seg[k, 0] * (q[:, 1] - proj[r, k, 1]) - seg[k, 1] * (q[:, 0] - proj[r, k, 0])
    return np.where(cr > 0, -1.0, 1.0) * dist[r, k], cum[k] + t[r, k] * L[k], cum[-1]


def point_in_poly(pts, poly):
    """Takes points (n,2) and a closed polygon (m,2); returns a boolean mask of points inside (even-odd rule)."""
    x, y = pts[:, 0], pts[:, 1]; inside = np.zeros(len(pts), bool); n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
        c = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1)
        inside ^= c
    return inside


def mirror_half(half):
    """Takes a +Z half outline [(|Z|, Y)...] running from the centreline out and back; returns the full closed polygon (Z, Y)."""
    h = np.array(half); return np.concatenate([h, h[::-1] * [-1, 1]])


def closest_on_poly(p, poly):
    """Takes a 2D point and a closed polygon (m,2); returns the nearest point on the polygon boundary."""
    a = poly; b = np.roll(poly, -1, 0); s = b - a
    t = np.clip(((p - a) * s).sum(1) / ((s ** 2).sum(1) + 1e-12), 0, 1); c = a + t[:, None] * s
    return c[np.linalg.norm(c - p, axis=1).argmin()]


def smooth_fit(z, y, x):
    """Takes glTF z, y and surface x samples; returns a smooth function f(z, y) -> x (least squares in y, z^2)."""
    def basis(z, y):
        """Takes z, y arrays; returns the polynomial design matrix."""
        z2 = z ** 2; return np.stack([np.ones_like(y), y, y ** 2, z2, y * z2, z2 ** 2, y * z2 ** 2, z2 ** 3], 1)
    ok = ~np.isnan(x); c = np.linalg.lstsq(basis(z[ok], y[ok]), x[ok], rcond=None)[0]
    return lambda zz, yy: basis(zz, yy) @ c


def surface_patch(name, poly, step, bvh, offset, mat, parent, hole=None, x_fn=None, fit=False):
    """Takes an outline polygon (Z, Y glTF), grid step, BVH (surface to sit on) or x_fn(z, y) -> x, offset along +X,
    material, parent pivot and an optional hole polygon. Builds a grid mesh covering poly minus hole, conformed to
    the surface, with boundary vertices snapped onto the outlines. Returns the new object."""
    z0, y0 = poly.min(0) - step; z1, y1 = poly.max(0) + step
    zs = np.arange(z0, z1 + step, step); ys = np.arange(y0, y1 + step, step)
    Zg, Yg = np.meshgrid(zs, ys); nz = len(zs)
    gpts = np.stack([Zg.ravel(), Yg.ravel()], 1)
    gin = point_in_poly(gpts, poly).reshape(Zg.shape)          # grid corners inside the outline / outside the hole
    gout = ~point_in_poly(gpts, hole).reshape(Zg.shape) if hole is not None else np.ones_like(gin)
    corners = lambda g: np.stack([g[:-1, :-1], g[:-1, 1:], g[1:, :-1], g[1:, 1:]])
    keep = corners(gin).any(0) & corners(gout).any(0)          # cells touching the region; overhang is snapped below
    cells = np.nonzero(keep)
    quads = [(r * nz + c, r * nz + c + 1, (r + 1) * nz + c + 1, (r + 1) * nz + c) for r, c in zip(*cells)]
    used = np.unique(np.array(quads).ravel()); remap = {v: i for i, v in enumerate(used)}
    V2 = np.stack([Zg.ravel()[used], Yg.ravel()[used]], 1)
    for i, p in enumerate(V2):                                   # snap overhanging corners onto the outlines
        if not point_in_poly(p[None], poly)[0]:
            V2[i] = closest_on_poly(p, poly)
        elif hole is not None and point_in_poly(p[None], hole)[0]:
            V2[i] = closest_on_poly(p, hole)
    X = x_fn(V2[:, 0], V2[:, 1]) if x_fn else front_x(bvh, V2[:, 0], V2[:, 1])
    if fit:                                                      # smooth surface that clears the samples
        f = smooth_fit(V2[:, 0], V2[:, 1], X); r = X - f(V2[:, 0], V2[:, 1])
        X = f(V2[:, 0], V2[:, 1]) + max(0.0, float(np.nanpercentile(r, 97)))
    X = np.where(np.isnan(X), np.nanmedian(X), X) + offset
    G = np.stack([X, V2[:, 1], V2[:, 0]], 1)
    B = np.stack([G[:, 0], -G[:, 2], G[:, 1]], 1)
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in B], [], [tuple(remap[v] for v in q) for q in quads])
    bm = bmesh.new(); bm.from_mesh(me)                           # drop cells collapsed by the snapping
    bmesh.ops.dissolve_degenerate(bm, dist=2e-4, edges=bm.edges[:]); bm.to_mesh(me); bm.free()
    me.materials.append(bpy.data.materials[mat])
    me.polygons.foreach_set('use_smooth', [True] * len(me.polygons)); me.update()
    ob = bpy.data.objects.new(name, me)
    (parent.users_collection[0] if parent.users_collection else bpy.context.scene.collection).objects.link(ob)
    ob.parent = parent; ob.matrix_parent_inverse = parent.matrix_world.inverted()
    return ob


# ---------------------------------------------------------------- 1. hood character lines
def hood_crease(ctx, height=0.007, w_in=0.08, w_out=0.012):
    """Takes ctx and ridge parameters (m); refines the hood mesh along the CAD character line and raises a crisp
    ridge there (inboard side fades over w_in, outboard side drops over w_out). Returns the number of moved vertices."""
    ob = ctx.obj('Civic_Hood_Panel'); M = ob.matrix_world
    bm = bmesh.new(); bm.from_mesh(ob.data)
    def near(v):
        """Takes a BMVert; returns True when it lies within 6 cm of the line (top view)."""
        w = M @ v.co; q = np.array([[w.x, abs(w.y)]])
        return abs(polyline_sdist(q, HOOD_LINE)[0][0]) < 0.06 and 1.12 < w.x < 2.10
    flag = {v.index: near(v) for v in bm.verts}
    edges = [e for e in bm.edges if flag[e.verts[0].index] or flag[e.verts[1].index]]
    bmesh.ops.subdivide_edges(bm, edges=edges, cuts=3, use_grid_fill=True)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
    bm.to_mesh(ob.data); bm.free(); ob.data.update()
    P = wco(ob)
    q = np.stack([P[:, 0], np.abs(P[:, 1])], 1)
    s, t, L = polyline_sdist(q, HOOD_LINE)
    taper = smoothstep(t / 0.04) * smoothstep((L - t) / 0.08)
    prof = np.where(s < 0, np.exp(s / w_in), np.exp(-s / w_out))
    lift = height * taper * prof * (np.abs(s) < 0.25)
    P[:, 2] += lift
    set_wco(ob, P); clear_custom_normals(ob)
    return int((lift > 1e-4).sum())


# ---------------------------------------------------------------- 2. headlamps
def binned(az, y, lo, hi, step, fn):
    """Takes lateral distances, heights, a bin range/step and np.max/np.min; returns (bin centres, per-bin values) with
    empty bins filled by interpolation."""
    edges = np.arange(lo, hi + step, step); c = (edges[:-1] + edges[1:]) / 2; v = np.full(len(c), np.nan)
    for i in range(len(c)):
        m = (az >= edges[i]) & (az < edges[i + 1])
        if m.any():
            v[i] = fn(y[m])
    ok = ~np.isnan(v); return c, np.interp(c, c[ok], v[ok])


def stretch_headlamp(ctx, side):
    """Takes ctx and 'L'/'R'; raises the lamp's top edge toward the CAD outline (capped under the brow), keeping the
    bottom edge, and conforms the moved parts to the front skin. Returns the maximum lift (m)."""
    sgn = -1.0 if side == 'L' else 1.0                         # glTF z sign of this lamp
    parts = [ctx.obj(f'Civic_Headlamp_{p}_{side}') for p in ('Lens', 'Housing', 'DRL', 'Projectors')]
    lens, mask, bup = parts[0], ctx.obj('Civic_Front_Headlamp_Mask'), ctx.obj('Civic_Front_Bumper_Upper')
    G = ctx.bl2gl(wco(lens)); az = np.abs(G[:, 2])
    lo, hi = az.min(), az.max()
    cz, T0 = binned(az, G[:, 1], lo, hi, 0.012, np.max)
    _, Bt = binned(az, G[:, 1], lo, hi, 0.012, np.min)
    T0 = np.polyval(np.polyfit(cz, T0, 3), cz); Bt = np.polyval(np.polyfit(cz, Bt, 3), cz)   # smooth outlines
    Gb = ctx.bl2gl(wco(bup)); mb = (Gb[:, 2] * sgn > 0) & (np.abs(Gb[:, 2]) > lo - 0.02)
    _, BU = binned(np.abs(Gb[mb, 2]), Gb[mb, 1], lo, hi, 0.012, np.min)
    T1 = np.minimum(np.interp(cz, LAMP_TOP[:, 0], LAMP_TOP[:, 1]), BU - 0.010)
    T1 = np.maximum(T1, T0)
    T1 = np.convolve(np.pad(T1, 2, mode='edge'), np.ones(5) / 5, 'valid')
    T1 = np.maximum(T1, T0)
    skin = bvh_of([lens, mask])
    for ob in parts:
        P = wco(ob); Gp = ctx.bl2gl(P); a = np.abs(Gp[:, 2]); y = Gp[:, 1]
        t0, t1, bt = np.interp(a, cz, T0), np.interp(a, cz, T1), np.interp(a, cz, Bt)
        k = (t1 - bt) / np.maximum(t0 - bt, 1e-3)
        yn = np.where(y <= bt, y, np.where(y <= t0, bt + (y - bt) * k, np.maximum(y, t1 + (y - t0) * 0.3)))
        moved = np.abs(yn - y) > 1e-5
        x_old = front_x(skin, Gp[moved, 2], y[moved]); x_new = front_x(skin, Gp[moved, 2], yn[moved])
        dx = np.where(np.isnan(x_old) | np.isnan(x_new), 0.0, x_new - x_old)
        P[moved, 0] += dx; P[moved, 2] = yn[moved]
        set_wco(ob, P); clear_custom_normals(ob)
    # mask: push the gloss-black skin behind the new lens area back so it does not z-fight with the lens
    P = wco(mask); Gm = ctx.bl2gl(P); a = np.abs(Gm[:, 2]); y = Gm[:, 1]
    sel = (Gm[:, 2] * sgn > 0) & (a > lo + 0.004) & (a < hi - 0.004)
    t0, t1 = np.interp(a, cz, T0), np.interp(a, cz, T1)
    sel &= (y > t0 - 0.004) & (y < t1 - 0.001)
    P[sel, 0] -= 0.008
    set_wco(mask, P)
    return float((T1 - T0).max())


def brow_black(ctx, z_in=0.45, depth=0.035):
    """Takes ctx, the inboard |Z| limit and the brow depth (m); turns the front edge of the body-colour ledge over
    the lamps into gloss-black brow (photos: thin black brow directly under the painted fender). Returns faces recoloured."""
    ob = ctx.obj('Civic_Front_Bumper_Upper'); me = ob.data
    if 'Mat_Gloss_Black_Trim' not in [m.name for m in me.materials if m]:
        me.materials.append(bpy.data.materials['Mat_Gloss_Black_Trim'])
    slot = [m.name if m else '' for m in me.materials].index('Mat_Gloss_Black_Trim')
    P = ctx.bl2gl(wco(ob)); az = np.abs(P[:, 2])
    cz, front = binned(az, P[:, 0], az.min(), az.max(), 0.02, np.max)
    n = 0
    for p in me.polygons:
        v = list(p.vertices); a = az[v].mean()
        if a > z_in and P[v, 0].mean() > np.interp(a, cz, front) - depth:
            p.material_index = slot; n += 1
    me.update(); return n


# ---------------------------------------------------------------- 3. upper grille frame + slot
def grille_frame(ctx):
    """Takes ctx; adds a gloss-black frame over the honeycomb outside the CAD trapezoid and the CAD upper slot with
    dividers on the black bar. Returns the names of the new objects."""
    grille, mask = ctx.obj('Civic_Front_Grille'), ctx.obj('Civic_Front_Headlamp_Mask')
    parent = grille.parent
    G = ctx.bl2gl(wco(grille)); m = (G[:, 1] > 0.52) & (np.abs(G[:, 2]) < 0.46)
    g = G[m]
    # front envelope of the honeycomb bars: per-cell max x, then a smooth fit x = f(y, z^2)
    key = np.round(g[:, 1] / 0.01) * 1000 + np.round(np.abs(g[:, 2]) / 0.02)
    order = np.argsort(key); uk, first = np.unique(key[order], return_index=True)
    env = np.maximum.reduceat(g[order, 0], first)
    ey = np.round(uk / 1000) * 0.01; ez = (uk % 1000) * 0.02
    A = np.stack([np.ones_like(ey), ey, ey ** 2, ez ** 2, ey * ez ** 2, ez ** 4], 1)
    coef = np.linalg.lstsq(A, env, rcond=None)[0]
    def x_fn(z, y):
        """Takes glTF z, y arrays; returns the fitted honeycomb front x."""
        z2 = z ** 2; return np.stack([np.ones_like(y), y, y ** 2, z2, y * z2, z2 ** 2], 1) @ coef
    top = float(np.percentile(g[:, 1], 99.5))
    bot = float(np.percentile(g[np.abs(g[:, 2]) < 0.3, 1], 0.5)) - 0.012    # outer box follows the (post-FFD) opening
    zw = float(np.abs(g[:, 2]).max()) + 0.01                                # and relies on the mask/lamps to clip it
    outer = np.array([(-zw, bot), (zw, bot), (zw, top + 0.004), (-zw, top + 0.004)])
    frame = surface_patch('Civic_Front_Grille_Frame', outer, 0.004, None, 0.004, 'Mat_Gloss_Black_Trim', parent,
                          hole=mirror_half(HONEYCOMB), x_fn=x_fn)
    mb = bvh_of([mask])
    slot = surface_patch('Civic_Front_Grille_Upper_Slot', mirror_half(SLOT), 0.004, mb, 0.0015, 'Mat_Black_Duct', parent)
    divs = []
    for zc in SLOT_DIVIDERS:
        for s in (-1, 1):
            poly = np.array([(s * zc - 0.004, 0.6915), (s * zc + 0.004, 0.6915), (s * zc + 0.004, 0.7185), (s * zc - 0.004, 0.7185)])
            divs.append(surface_patch(f'Civic_Front_Grille_Slot_Divider_{"L" if s < 0 else "R"}{int(zc * 1000)}', poly,
                                      0.004, mb, 0.0025, 'Mat_Gloss_Black_Trim', parent))
    # Type R script sits on the honeycomb just under the badge line (photos: Y .565-.597, |Z| ~.30), not on the frame
    tb = ctx.obj('Civic_Badge_Front_Type'); P = wco(tb); Gt = ctx.bl2gl(P)
    dy = 0.581 - (Gt[:, 1].min() + Gt[:, 1].max()) / 2
    P[:, 0] += x_fn(Gt[:, 2], Gt[:, 1] + dy) - x_fn(Gt[:, 2], Gt[:, 1]); P[:, 2] += dy; set_wco(tb, P)
    return [frame.name, slot.name] + [d.name for d in divs]


# ---------------------------------------------------------------- 4. lower bumper: fog-pod blades + black lower bar
BLADE_SHIFT = np.array([(0.43, 0.0), (0.46, 0.0), (0.52, 0.010), (0.60, 0.022), (0.67, 0.025), (0.72, 0.025),
                        (0.76, 0.014), (0.79, 0.0)])          # (|Z|, dY): model blade centre -> CAD blade centre
LOWER_BAR = [(0.0, 0.516), (0.405, 0.516), (0.405, 0.505), (0.385, 0.497), (0.345, 0.485), (0.300, 0.476),
             (0.230, 0.468), (0.0, 0.468)]                   # (|Z|, Y) CAD gloss-black lower bar under the grille


def raise_pod_blades(ctx):
    """Takes ctx; lifts the body-colour blade across each fog pod so it runs from the pod's outer-top corner down to
    the inner end like the CAD/photo blade (it sat 2-3 cm too low and too flat). The vent fins (|Z| > .79) stay.
    Returns the largest lift (m)."""
    ob = ctx.obj('Civic_Front_Bumper_Painted_Blades'); P = wco(ob); G = ctx.bl2gl(P)
    dy = np.interp(np.abs(G[:, 2]), BLADE_SHIFT[:, 0], BLADE_SHIFT[:, 1], left=0, right=0)
    dy *= (G[:, 1] > 0.39)                                    # only the blade, not the pod-bottom lip
    P[:, 2] += dy; set_wco(ob, P); clear_custom_normals(ob)
    return float(dy.max())


def lower_black_bar(ctx):
    """Takes ctx; adds the gloss-black lower bar under the upper grille (CAD band Y .468-.516, rising to the pods).
    Returns the new object's name."""
    cheeks, mask = ctx.obj('Civic_Front_Bumper_Cheeks'), ctx.obj('Civic_Front_Headlamp_Mask')
    ob = surface_patch('Civic_Front_Lower_Bar', mirror_half(LOWER_BAR), 0.004, bvh_of([cheeks, mask]), 0.0015,
                       'Mat_Gloss_Black_Trim', cheeks.parent, fit=True)
    return ob.name


# ---------------------------------------------------------------- 5. round 2: fog lamps, hood scoop, brow, gaps
FOG_CAD = (0.724, 0.372, 0.036)                               # CAD front plate fog circle (|Z|, Y, r)
FOG_LENS_R = 0.038                                            # target lens radius (outside of the CAD line)


def fog_lamps(ctx, r_full=0.064, r_zero=0.110):
    """Takes ctx and the bezel warp radii (m). Shrinks and lifts each fog lamp onto the CAD circle: the lamp body and
    optic move rigidly (similarity in the Z-Y plane), the gloss-black bezel ring and satin insert follow with a weight
    that is 1 inside r_full and fades to 0 at r_zero so the pod outline stays. Returns (scale, lift) applied."""
    lamp, optic = ctx.obj('Civic_Front_Fog_Lights'), ctx.obj('Civic_Front_Fog_Optics')
    ring, insert = ctx.obj('Civic_Front_Bumper_Fog_Surrounds'), ctx.obj('Civic_Front_Fog_Insert_Panels')
    Gl = ctx.bl2gl(wco(lamp)); a = np.abs(Gl[:, 2])
    c0 = np.array([(a.min() + a.max()) / 2, (Gl[:, 1].min() + Gl[:, 1].max()) / 2])   # model centre (|Z|, Y)
    s = FOG_LENS_R / ((Gl[:, 1].max() - Gl[:, 1].min()) / 2)
    c1 = np.array(FOG_CAD[:2])
    for ob in (lamp, optic, ring, insert):
        P = wco(ob); G = ctx.bl2gl(P); sg = np.sign(G[:, 2]); q = np.stack([np.abs(G[:, 2]), G[:, 1]], 1)
        d = np.linalg.norm(q - c0, axis=1)
        w = np.ones(len(q)) if ob in (lamp, optic) else 1 - smoothstep((d - r_full) / (r_zero - r_full))
        qn = q + w[:, None] * ((c1 + s * (q - c0)) - q)
        G[:, 2] = sg * qn[:, 0]; G[:, 1] = qn[:, 1]
        set_wco(ob, ctx.gl2bl(G)); clear_custom_normals(ob)
    return round(float(s), 3), round(float(c1[1] - c0[1]), 4)


def hood_scoop_opening(ctx, top=0.912, bottom=0.891):
    """Takes ctx, the fixed opening top Y and the new bottom Y. Lowers the opening height of the bonnet scoop by
    raising its floor/duct/mesh bottom to the CAD line (model bottom Y .874 vs CAD ~.89); the floor ramp keeps its
    front edge on the hood. Returns the old bottom Y."""
    duct, mesh, floor = ctx.obj('Civic_Hood_Duct'), ctx.obj('Civic_Hood_Inlet_Mesh'), ctx.obj('Civic_Hood_Scoop_Floor')
    y_old = float(ctx.bl2gl(wco(duct))[:, 1].min())
    k = (top - bottom) / (top - y_old)
    for ob in (duct, mesh):
        P = wco(ob); G = ctx.bl2gl(P); low = G[:, 1] < top
        G[low, 1] = top - (top - G[low, 1]) * k
        set_wco(ob, ctx.gl2bl(G)); clear_custom_normals(ob)
    P = wco(floor); G = ctx.bl2gl(P); x_front = G[:, 0].max()
    G[:, 1] += (bottom - y_old) * smoothstep((x_front - G[:, 0]) / 0.08)
    set_wco(floor, ctx.gl2bl(G)); clear_custom_normals(floor)
    return round(y_old, 4)


def visible_band(target, others, z, ys):
    """Takes a target BVH, a BVH of all front skins, a glTF z and candidate heights; returns the (min, max) height at
    which a ray along -X hits the target first (nan, nan when never)."""
    hit = []
    for y in ys:
        lt = target.ray_cast(Vector((3.0, -z, y)), Vector((-1, 0, 0)))[0]
        la = others.ray_cast(Vector((3.0, -z, y)), Vector((-1, 0, 0)))[0]
        if lt is not None and la is not None and lt.x >= la.x - 1e-4:
            hit.append(y)
    return (min(hit), max(hit)) if hit else (np.nan, np.nan)


def brow_strips(ctx, z_in=0.40, lip=0.004, under=0.004):
    """Takes ctx, the inboard |Z| start, the painted lip left at the top and the overlap onto the mask below (m).
    Adds a clean gloss-black brow strip on the front face of the painted ledge over each headlamp (photos: black brow
    directly under the painted fender/hood edge, continuous with the black bar over the grille). Returns names."""
    bup, mask, hood = ctx.obj('Civic_Front_Bumper_Upper'), ctx.obj('Civic_Front_Headlamp_Mask'), ctx.obj('Civic_Hood_Panel')
    lenses = [ctx.obj(f'Civic_Headlamp_Lens_{s}') for s in 'LR']
    tb = bvh_of([bup]); allb = bvh_of([bup, mask, hood] + lenses)
    names = []
    for side, sgn in (('L', -1.0), ('R', 1.0)):
        G = ctx.bl2gl(wco(ctx.obj(f'Civic_Headlamp_Lens_{side}'))); z_out = float(np.abs(G[:, 2]).max()) + 0.005
        zs = np.arange(z_in, z_out, 0.01); ys = np.arange(0.70, 0.80, 0.001)
        band = np.array([visible_band(tb, allb, sgn * z, ys) for z in zs]); ok = ~np.isnan(band[:, 0])
        zs, band = zs[ok], band[ok]
        lo = band[:, 0] - under; hi = np.maximum(band[:, 1] - lip, lo + 0.004)
        lo = np.convolve(np.pad(lo, 1, mode='edge'), np.ones(3) / 3, 'valid')
        hi = np.convolve(np.pad(hi, 1, mode='edge'), np.ones(3) / 3, 'valid')
        poly = np.concatenate([np.stack([sgn * zs, lo], 1), np.stack([sgn * zs[::-1], hi[::-1]], 1)])
        ob = surface_patch(f'Civic_Front_Brow_{side}', poly, 0.004, allb, 0.0015, 'Mat_Gloss_Black_Trim', bup.parent)
        names.append(ob.name)
    return names


SKIN_PREFIX = ('Civic_Front_', 'Civic_Headlamp_', 'Civic_Hood_Panel', 'Civic_Body_Shell_FrontFender', 'Civic_Fender_')


def gap_fillers(ctx, z_rng=(0.38, 0.95), y_rng=(0.54, 0.80), step=0.0025, back=0.006):
    """Takes ctx, the |Z| and Y window, grid step and the set-back behind the skins (m). Casts front-view rays over the
    window; cells that miss every exterior skin while skin lies above and below in the same column are see-through
    gaps into the engine bay (under the lamps between mask and cheeks, slits at the ledge). Fills them with one
    matte-black sheet set just behind the neighbouring skins. Returns the number of filled cells."""
    skins = [ob for ob in bpy.data.objects if ob.type == 'MESH' and ob.name.startswith(SKIN_PREFIX)]
    bvh = bvh_of(skins)
    bay = bvh_of([ob for ob in bpy.data.objects if ob.type == 'MESH' and ob.name.startswith('Civic_Eng_')])
    za = np.arange(z_rng[0], z_rng[1] + 1e-9, step); ya = np.arange(y_rng[0], y_rng[1] + 1e-9, step)
    verts, quads, vid = [], [], {}
    def vert(z, y, x, col):
        """Takes a grid corner (glTF z, y), x and its column key; returns its index in the filler mesh (corners
        merged within a column, so each column keeps its own depth)."""
        k = (col, round(z / step * 2), round(y / step * 2))
        if k not in vid:
            vid[k] = len(verts); verts.append((x, -z, y))
        return vid[k]
    n = 0
    for sgn in (-1.0, 1.0):
        for z in za:
            hx = np.array([(lambda l: np.nan if l is None else l.x)(bvh.ray_cast(Vector((3.0, -sgn * z, y)),
                                                                                  Vector((-1, 0, 0)))[0]) for y in ya])
            bx = np.array([(lambda l: -np.inf if l is None else l.x)(bay.ray_cast(Vector((3.0, -sgn * z, y)),
                                                                                    Vector((-1, 0, 0)))[0]) for y in ya])
            hx = np.where(bx > hx, np.nan, hx)                    # bay parts seen in front of a deep skin = gap too
            hit = ~np.isnan(hx)
            if hit.sum() < 2:
                continue
            i0, i1 = np.nonzero(hit)[0][[0, -1]]
            gap = ~hit & (np.arange(len(ya)) > i0) & (np.arange(len(ya)) < i1)
            if not gap.any():
                continue
            fill = gap | np.roll(gap, 1) | np.roll(gap, -1)          # one cell of overlap under the skins each side
            near = np.convolve(hit.astype(float), [1, 1, 1], 'same') > 0
            nb = np.where(near & hit & fill, hx, np.nan)
            x = float(np.nanmin(nb)) - back if (~np.isnan(nb)).any() else -np.inf
            x = max(x, float(np.max(np.where(gap, bx, -np.inf))) + 0.003)   # stay in front of the bay parts
            if not np.isfinite(x):
                continue
            for j in np.nonzero(fill)[0]:
                zc, yc = sgn * z, ya[j]; h = step                 # cells overlap their neighbours: no cracks
                c = (sgn, round(z / step))
                quads.append([vert(zc - h, yc - h, x, c), vert(zc + h, yc - h, x, c), vert(zc + h, yc + h, x, c),
                              vert(zc - h, yc + h, x, c)])
            n += int(gap.sum())
    if not quads:
        return 0
    cheeks = ctx.obj('Civic_Front_Bumper_Cheeks')
    me = bpy.data.meshes.new('Civic_Front_Bay_Gap_Filler'); me.from_pydata(verts, [], quads)
    me.materials.append(bpy.data.materials['Mat_Black_Duct']); me.update()
    ob = bpy.data.objects.new('Civic_Front_Bay_Gap_Filler', me)
    cheeks.users_collection[0].objects.link(ob)
    ob.parent = cheeks.parent; ob.matrix_parent_inverse = cheeks.parent.matrix_world.inverted()
    return n


# ---------------------------------------------------------------- entry point
def apply(ctx):
    """Takes the build context; applies every front-region change in a fixed order. Returns nothing."""
    print('front: hood crease verts', hood_crease(ctx), flush=True)
    print('front: grille', grille_frame(ctx), flush=True)          # before the lamp edit so it reads the original mask
    for side in ('L', 'R'):
        print('front: headlamp', side, 'max lift', round(stretch_headlamp(ctx, side), 4), flush=True)
    # brow_black(ctx) is kept but not applied: recolouring the triangulated ledge gives ragged black patches seen
    # from above; the mask strip left between the raised lens top and the ledge already reads as the black brow.
    print('front: pod blade lift', round(raise_pod_blades(ctx), 4), flush=True)
    print('front: lower bar', lower_black_bar(ctx), flush=True)
    print('front: fog lamps (scale, lift)', fog_lamps(ctx), flush=True)
    print('front: hood scoop old bottom', hood_scoop_opening(ctx), flush=True)
    print('front: brow strips', brow_strips(ctx), flush=True)
    print('front: bay gap cells filled', gap_fillers(ctx), flush=True)
