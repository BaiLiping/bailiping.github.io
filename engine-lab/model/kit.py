# kit.py: small part-builder kit for build.py (run inside Blender, no other dependencies).
#
# Everything is authored in millimetres and written to the scene in metres.  A part is a set of closed shells
# gathered in a Geo, then turned into one mesh object by make().  Repeated parts share one mesh through instance().
# The approach follows anymachine's hs.py (https://github.com/BaiLiping/anymachine, MIT): one builder per kind of
# solid, no modifiers, no booleans, normals fixed once at the end.
import bpy, bmesh, math
from mathutils import Vector, Matrix, Quaternion
from mathutils.geometry import delaunay_2d_cdt

MM = 0.001          # scene units per millimetre
V = Vector
TAU = 2.0 * math.pi


def frame(o=(0, 0, 0), z=(0, 0, 1), x=None):
    """Build a placement matrix.  Takes an origin, the direction local +Z should point along and an optional
    hint for local +X.  Returns a 4x4 Matrix (rotation and translation, no scale)."""
    z = V(z).normalized()
    if x is None:
        x = V((1, 0, 0)) if abs(z.x) < 0.9 else V((0, 1, 0))
    x = V(x)
    x = (x - z * x.dot(z)).normalized()
    y = z.cross(x)
    m = Matrix.Identity(4)
    for i, c in enumerate((x, y, z)):
        m[0][i], m[1][i], m[2][i] = c.x, c.y, c.z
    m[0][3], m[1][3], m[2][3] = o[0], o[1], o[2]
    return m


def basis(o, ex, ey):
    """Build a placement matrix from two axes.  Takes an origin and the directions local +X and +Y should point
    along (local +Z is their cross product).  Returns a 4x4 Matrix."""
    ex = V(ex).normalized()
    ey = V(ey)
    ey = (ey - ex * ey.dot(ex)).normalized()
    ez = ex.cross(ey)
    m = Matrix.Identity(4)
    for i, c in enumerate((ex, ey, ez)):
        m[0][i], m[1][i], m[2][i] = c.x, c.y, c.z
    m[0][3], m[1][3], m[2][3] = o[0], o[1], o[2]
    return m


def rot_x(deg):
    """Rotation about +X.  Takes degrees.  Returns a 4x4 Matrix."""
    return Matrix.Rotation(math.radians(deg), 4, 'X')


def circle(r, n, phase=0.0):
    """Regular polygon outline.  Takes radius, vertex count and a phase in turns.  Returns [(x, y), ...]."""
    return [(r * math.cos(TAU * (i + phase) / n), r * math.sin(TAU * (i + phase) / n)) for i in range(n)]


def rounded_rect(w, h, r, n=4):
    """Rounded rectangle outline centred on the origin.  Takes width, height, corner radius and the number of
    segments per corner.  Returns [(x, y), ...] counter-clockwise."""
    r = min(r, w / 2 - 1e-6, h / 2 - 1e-6)
    pts = []
    for cx, cy, a0 in ((w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90), (-w / 2 + r, -h / 2 + r, 180), (w / 2 - r, -h / 2 + r, 270)):
        for k in range(n + 1):
            a = math.radians(a0 + 90.0 * k / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def gear_outline(teeth, r_pitch, add, ded, land=0.28):
    """Spur gear outline with trapezoid teeth.  Takes the tooth count, pitch radius, addendum, dedendum and the
    tip land as a fraction of the tooth pitch.  Returns [(x, y), ...] counter-clockwise."""
    pts = []
    for i in range(teeth):
        for f, r in ((-0.25 - land / 2, r_pitch - ded), (-land / 2, r_pitch + add), (land / 2, r_pitch + add), (0.25 + land / 2, r_pitch - ded)):
            a = TAU * (i + f) / teeth
            pts.append((r * math.cos(a), r * math.sin(a)))
    return pts


def sprocket_outline(teeth, pitch, roller, tip=0.42):
    """Roller chain sprocket outline.  Takes the tooth count, chain pitch, roller diameter and tooth height as a
    fraction of the pitch.  Returns [(x, y), ...] counter-clockwise; a roller seat sits at angle 0."""
    rp = pitch / (2 * math.sin(math.pi / teeth))
    pts = []
    for i in range(teeth):
        a0 = TAU * i / teeth
        for da, r in ((-0.20, rp - roller * 0.30), (0.0, rp - roller * 0.5), (0.20, rp - roller * 0.30), (0.40, rp + pitch * tip * 0.6), (0.60, rp + pitch * tip * 0.6)):
            a = a0 + TAU * da / teeth
            pts.append((r * math.cos(a), r * math.sin(a)))
    return pts


def fillet_path(pts, radius, seg=6):
    """Round the corners of a polyline.  Takes points, a bend radius (one number or one per inner corner) and the
    segments per bend.  Returns a new list of Vectors that passes through the first and last point."""
    pts = [V(p) for p in pts]
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        r = radius[i - 1] if isinstance(radius, (list, tuple)) else radius
        u, w = (a - b), (c - b)
        lu, lw = u.length, w.length
        if lu < 1e-6 or lw < 1e-6:
            continue
        u /= lu; w /= lw
        ang = u.angle(w)
        if ang > math.pi - 1e-3 or r <= 0:
            out.append(b); continue
        t = min(r / math.tan(ang / 2), lu * 0.48, lw * 0.48)
        rr = t * math.tan(ang / 2)
        p0, p1 = b + u * t, b + w * t
        bis = (u + w).normalized()
        centre = b + bis * (rr / math.sin(ang / 2))
        v0, v1 = p0 - centre, p1 - centre
        sweep = v0.angle(v1)
        axis = v0.cross(v1).normalized()
        for k in range(seg + 1):
            q = Quaternion(axis, sweep * k / seg)
            out.append(centre + q @ v0)
    out.append(pts[-1])
    return out


def smooth_path(pts, sub=6):
    """Catmull-Rom spline through points.  Takes control points and the samples per span.  Returns Vectors."""
    pts = [V(p) for p in pts]
    P = [pts[0] * 2 - pts[1]] + pts + [pts[-1] * 2 - pts[-2]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for k in range(sub):
            t = k / sub
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(pts[-1])
    return out


def path_length(pts):
    """Length of a polyline.  Takes points.  Returns a float."""
    return sum((V(b) - V(a)).length for a, b in zip(pts, pts[1:]))


class Geo:
    """Closed shells gathered in one bmesh, in the part's own frame, in millimetres.  Every builder adds one
    watertight shell and returns self so calls can be chained.  at() sets where the next shells are placed."""

    def __init__(s):
        s.bm = bmesh.new()
        s.xf = Matrix.Identity(4)

    def at(s, m=None):
        """Set the placement for the shells that follow.  Takes a 4x4 Matrix or None for identity.  Returns self."""
        s.xf = Matrix.Identity(4) if m is None else m
        return s

    def _v(s, co):
        """Add one vertex.  Takes a local coordinate.  Returns the BMVert."""
        return s.bm.verts.new(s.xf @ V(co))

    def _cap(s, ring2d, verts, holes2d=(), hole_verts=()):
        """Fill a flat polygon, with optional holes, between existing verts.  Takes the 2D outline, the matching
        BMVerts, and the same for each hole.  Returns nothing."""
        pts = [V((p[0], p[1])) for p in ring2d]
        vv = list(verts)
        loops = [list(range(len(pts)))]
        for h2, hv in zip(holes2d, hole_verts):
            base = len(pts)
            pts += [V((p[0], p[1])) for p in h2]
            vv += list(hv)
            loops.append(list(range(base, base + len(h2))))
        edges = [(lp[i], lp[(i + 1) % len(lp)]) for lp in loops for i in range(len(lp))]
        out_v, _, out_f, orig_v, _, _ = delaunay_2d_cdt(pts, edges, [], 2 if len(loops) > 1 else 1, 1e-6)
        if len(loops) > 1:      # mode 2 keeps the region inside the outer edges; drop triangles inside holes
            import mathutils.geometry as mg
            keep = []
            for f in out_f:
                c = sum((out_v[i] for i in f), V((0, 0))) / len(f)
                inside_hole = any(_point_in_poly(c, [pts[i] for i in lp]) for lp in loops[1:])
                if not inside_hole and _point_in_poly(c, [pts[i] for i in loops[0]]):
                    keep.append(f)
            out_f = keep
        idx = []
        for i, o in enumerate(orig_v):
            idx.append(o[0] if o else None)
        for f in out_f:
            tri = [idx[i] for i in f]
            if None in tri or len(set(tri)) < 3:
                continue
            try:
                s.bm.faces.new([vv[i] for i in tri])
            except ValueError:
                pass

    def prism(s, outline, z0, z1, holes=()):
        """Extrude a flat outline along local Z.  Takes the outline [(x, y)], the two Z levels and optional hole
        outlines.  Returns self."""
        loops2d = [list(outline)] + [list(h) for h in holes]
        rings = []
        for lp in loops2d:
            a = [s._v((p[0], p[1], z0)) for p in lp]
            b = [s._v((p[0], p[1], z1)) for p in lp]
            n = len(lp)
            for i in range(n):
                j = (i + 1) % n
                s.bm.faces.new((a[i], a[j], b[j], b[i]))
            rings.append((a, b))
        s._cap(loops2d[0], rings[0][0], loops2d[1:], [r[0] for r in rings[1:]])
        s._cap(loops2d[0], rings[0][1], loops2d[1:], [r[1] for r in rings[1:]])
        return s

    def lathe(s, profile, n=32, closed=False, shape=None, phase=0.0):
        """Spin a profile about local Z.  Takes [(r, z), ...], the segment count, whether the profile is a closed
        loop (a ring-shaped solid) or runs from axis to axis, an optional shape(r, angle) -> (x, y) for
        non-round sections, and a phase in turns.  Returns self."""
        rings = []
        for r, z in profile:
            if r <= 1e-6:
                rings.append([s._v((0, 0, z))])
            else:
                ring = []
                for i in range(n):
                    a = TAU * (i + phase) / n
                    x, y = shape(r, a) if shape else (r * math.cos(a), r * math.sin(a))
                    ring.append(s._v((x, y, z)))
                rings.append(ring)
        pairs = list(zip(rings, rings[1:]))
        if closed:
            pairs.append((rings[-1], rings[0]))
        for a, b in pairs:
            if len(a) == 1 and len(b) == 1:
                continue
            for i in range(n):
                j = (i + 1) % n
                if len(a) == 1:
                    s.bm.faces.new((a[0], b[j], b[i]))
                elif len(b) == 1:
                    s.bm.faces.new((a[i], a[j], b[0]))
                else:
                    s.bm.faces.new((a[i], a[j], b[j], b[i]))
        if not closed:
            for ring in (rings[0], rings[-1]):
                if len(ring) > 1:
                    s._fan(ring)
        return s

    def _fan(s, ring):
        """Close a ring with a triangle fan about its centre.  Takes BMVerts.  Returns nothing."""
        c = s.bm.verts.new(sum((v.co for v in ring), V((0, 0, 0))) / len(ring))
        n = len(ring)
        for i in range(n):
            s.bm.faces.new((ring[i], ring[(i + 1) % n], c))

    def cyl(s, p0, p1, r, r1=None, n=24, phase=0.0):
        """Solid cylinder or cone frustum between two points.  Takes the end points, the radius at each end, the
        segment count and a phase in turns.  Returns self."""
        p0, p1 = V(p0), V(p1)
        keep = s.xf
        s.xf = keep @ frame(p0, p1 - p0)
        h = (p1 - p0).length
        s.lathe([(0, 0), (r, 0), (r if r1 is None else r1, h), (0, h)], n=n, phase=phase)
        s.xf = keep
        return s

    def tube(s, p0, p1, r_out, r_in, n=24):
        """Straight hollow tube between two points.  Takes the end points, outer and inner radius and the segment
        count.  Returns self."""
        p0, p1 = V(p0), V(p1)
        keep = s.xf
        s.xf = keep @ frame(p0, p1 - p0)
        h = (p1 - p0).length
        s.lathe([(r_in, 0), (r_out, 0), (r_out, h), (r_in, h)], n=n, closed=True)
        s.xf = keep
        return s

    def hexa(s, p, d, af, h):
        """Hexagonal prism, as for a bolt head or nut.  Takes the base centre, the axis direction, the width
        across flats and the height.  Returns self."""
        p, d = V(p), V(d).normalized()
        return s.cyl(p, p + d * h, af / math.sqrt(3.0), n=6)

    def sphere(s, c, r, n=16, m=8, squash=1.0):
        """Sphere or spheroid.  Takes centre, radius, segment counts round and along, and a Z squash factor.
        Returns self."""
        keep = s.xf
        s.xf = keep @ Matrix.Translation(V(c))
        s.lathe([(r * math.sin(math.pi * k / m), -r * squash * math.cos(math.pi * k / m)) for k in range(m + 1)], n=n)
        s.xf = keep
        return s

    def torus(s, c, axis, R, r, n=32, m=10):
        """Torus.  Takes centre, axis direction, ring radius, tube radius and the two segment counts.  Returns self."""
        keep = s.xf
        s.xf = keep @ frame(c, axis)
        s.lathe([(R + r * math.cos(TAU * k / m), r * math.sin(TAU * k / m)) for k in range(m)], n=n, closed=True)
        s.xf = keep
        return s

    def box(s, c, size, bevel=0.0, seg=1):
        """Axis-aligned box in the current placement.  Takes the centre, the three sizes, a chamfer width and the
        chamfer segments.  Returns self."""
        tmp = bmesh.new()
        bmesh.ops.create_cube(tmp, size=1.0)
        for v in tmp.verts:
            v.co = V((v.co.x * size[0] + c[0], v.co.y * size[1] + c[1], v.co.z * size[2] + c[2]))
        if bevel > 0:
            bmesh.ops.bevel(tmp, geom=list(tmp.edges), offset=bevel, offset_type='OFFSET', segments=seg, profile=0.5, affect='EDGES', clamp_overlap=True)
        s._merge(tmp)
        return s

    def _merge(s, tmp):
        """Copy a temporary bmesh into this one through the current placement, then free it.  Returns nothing."""
        m = {}
        for v in tmp.verts:
            m[v.index] = s.bm.verts.new(s.xf @ v.co)
        tmp.verts.index_update()
        for f in tmp.faces:
            try:
                s.bm.faces.new([m[v.index] for v in f.verts])
            except ValueError:
                pass
        tmp.free()

    def sweep(s, path, section, scales=None, closed_path=False, up=None, twist=0.0):
        """Sweep a closed flat section along a path with rotation-minimising frames.  Takes the path points, the
        section [(u, v)], an optional scale per path point (a number or a (su, sv) pair), whether the path is a
        loop, an optional up hint for the first frame and a total twist in degrees.  Returns self."""
        P = [V(p) for p in path]
        n = len(P)
        tang = []
        for i in range(n):
            a = P[i - 1] if (i > 0 or closed_path) else P[i]
            b = P[(i + 1) % n] if (i < n - 1 or closed_path) else P[i]
            t = (b - a)
            tang.append(t.normalized() if t.length > 1e-9 else V((0, 0, 1)))
        t0 = tang[0]
        u = V(up) if up is not None else (V((0, 0, 1)) if abs(t0.z) < 0.9 else V((1, 0, 0)))
        u = (u - t0 * u.dot(t0)).normalized()
        frames = []
        for i in range(n):
            t = tang[i]
            if i > 0:
                q = tang[i - 1].rotation_difference(t)
                u = (q @ u)
                u = (u - t * u.dot(t)).normalized()
            frames.append((u.copy(), t.cross(u)))
        rings = []
        m = len(section)
        for i in range(n):
            uu, vv = frames[i]
            if twist:
                a = math.radians(twist) * i / max(1, n - 1)
                uu, vv = uu * math.cos(a) + vv * math.sin(a), vv * math.cos(a) - uu * math.sin(a)
            sc = 1.0 if scales is None else scales[i]
            su, sv = (sc, sc) if not isinstance(sc, (tuple, list)) else sc
            rings.append([s._v(P[i] + uu * (p[0] * su) + vv * (p[1] * sv)) for p in section])
        last = n if closed_path else n - 1
        for i in range(last):
            a, b = rings[i], rings[(i + 1) % n]
            for k in range(m):
                j = (k + 1) % m
                s.bm.faces.new((a[k], a[j], b[j], b[k]))
        if not closed_path:
            s._cap(section, rings[0])
            s._cap(section, rings[-1])
        return s

    def pipe(s, path, r_out, r_in=0.0, n=16, scales=None):
        """Round pipe along a path, hollow when r_in > 0.  Takes the path points, outer and inner radius, the
        segment count and an optional scale per path point.  Returns self."""
        P = [V(p) for p in path]
        if r_in <= 0:
            return s.sweep(P, circle(r_out, n), scales=scales)
        tang = [(P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]).normalized() for i in range(len(P))]
        t0 = tang[0]
        u = V((0, 0, 1)) if abs(t0.z) < 0.9 else V((1, 0, 0))
        u = (u - t0 * u.dot(t0)).normalized()
        ro, ri = [], []
        co, ci = circle(r_out, n), circle(r_in, n)
        for i in range(len(P)):
            t = tang[i]
            if i > 0:
                u = tang[i - 1].rotation_difference(t) @ u
                u = (u - t * u.dot(t)).normalized()
            w = t.cross(u)
            sc = 1.0 if scales is None else scales[i]
            ro.append([s._v(P[i] + (u * p[0] + w * p[1]) * sc) for p in co])
            ri.append([s._v(P[i] + (u * p[0] + w * p[1]) * sc) for p in ci])
        for i in range(len(P) - 1):
            for k in range(n):
                j = (k + 1) % n
                s.bm.faces.new((ro[i][k], ro[i][j], ro[i + 1][j], ro[i + 1][k]))
                s.bm.faces.new((ri[i][j], ri[i][k], ri[i + 1][k], ri[i + 1][j]))
        for i in (0, len(P) - 1):
            for k in range(n):
                j = (k + 1) % n
                s.bm.faces.new((ro[i][k], ro[i][j], ri[i][j], ri[i][k]))
        return s

    def helix(s, p0, p1, R, r, turns, n=14, m=6):
        """Coil spring between two points.  Takes the end points, coil radius, wire radius, the number of turns,
        segments per turn and segments round the wire.  Returns self."""
        p0, p1 = V(p0), V(p1)
        L = (p1 - p0).length
        keep = s.xf
        s.xf = keep @ frame(p0, p1 - p0)
        k = int(turns * n)
        pts = []
        for i in range(k + 1):
            f = i / k
            a = TAU * turns * f
            pts.append((R * math.cos(a), R * math.sin(a), r + (L - 2 * r) * f))
        s.sweep(pts, circle(r, m))
        s.xf = keep
        return s

    def bolt(s, p, d, size=8.0, length=0.0, head=None, washer=True):
        """Hex bolt head with washer and an optional shank, the head sitting on point p.  Takes the seat point,
        the outward axis, the thread size, the shank length below the seat and an optional head height.
        Returns self."""
        p, d = V(p), V(d).normalized()
        af = size * 1.6
        hh = size * 0.68 if head is None else head
        if washer:
            s.cyl(p, p + d * (size * 0.14), size * 1.05, n=16)
            p = p + d * (size * 0.14)
        s.hexa(p, d, af, hh)
        if length > 0:
            s.cyl(p - d * length, p, size * 0.5, n=10)
        return s

    def nut(s, p, d, size=8.0, stud=4.0, washer=True):
        """Hex nut on a stud end, sitting on point p.  Takes the seat point, the outward axis, the thread size,
        how far the stud stands proud and whether there is a washer.  Returns self."""
        p, d = V(p), V(d).normalized()
        af = size * 1.6
        if washer:
            s.cyl(p, p + d * (size * 0.14), size * 1.05, n=16)
            p = p + d * (size * 0.14)
        s.hexa(p, d, af, size * 0.8)
        s.cyl(p, p + d * (size * 0.8 + stud), size * 0.46, n=10)
        return s

    def loft(s, rings, closed_ring=True, cap=True):
        """Skin a series of rings that all have the same vertex count.  Takes the rings as lists of points, whether
        each ring is a closed loop and whether the two end rings are filled.  Returns self."""
        R = [[s._v(p) for p in ring] for ring in rings]
        m = len(R[0])
        for a, b in zip(R, R[1:]):
            for k in range(m if closed_ring else m - 1):
                j = (k + 1) % m
                s.bm.faces.new((a[k], a[j], b[j], b[k]))
        if cap:
            for ring in (R[0], R[-1]):
                s._fan(ring)
        return s

    def shell(s, outer, inner):
        """Thin open shell lofted through stations, such as a duct cover.  Takes the outer skin as a list of
        open rings (one per station, equal counts) and the inner skin the same way.  Returns self."""
        O = [[s._v(p) for p in ring] for ring in outer]
        I = [[s._v(p) for p in ring] for ring in inner]
        m = len(O[0])
        for i in range(len(O) - 1):
            for k in range(m - 1):
                s.bm.faces.new((O[i][k], O[i][k + 1], O[i + 1][k + 1], O[i + 1][k]))
                s.bm.faces.new((I[i][k + 1], I[i][k], I[i + 1][k], I[i + 1][k + 1]))
            s.bm.faces.new((O[i][0], O[i + 1][0], I[i + 1][0], I[i][0]))
            s.bm.faces.new((O[i + 1][m - 1], O[i][m - 1], I[i][m - 1], I[i + 1][m - 1]))
        for i in (0, len(O) - 1):
            for k in range(m - 1):
                s.bm.faces.new((O[i][k], O[i][k + 1], I[i][k + 1], I[i][k]))
        return s

    def tris(s):
        """Count triangles.  Returns an int."""
        return sum(len(f.verts) - 2 for f in s.bm.faces)


def _point_in_poly(p, poly):
    """Even-odd point in polygon test.  Takes a 2D point and a list of 2D points.  Returns bool."""
    inside = False
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if (a[1] > p[1]) != (b[1] > p[1]):
            if p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]:
                inside = not inside
    return inside


# ---------------------------------------------------------------------------------------------------------------
# scene side: materials, objects, instances
# ---------------------------------------------------------------------------------------------------------------
_MATS = {}


def material(name, rgb, metal=0.0, rough=0.5):
    """Create or fetch a Principled material.  Takes a name, an sRGB colour triple (0 to 1), metallic and
    roughness.  Returns the material."""
    if name in _MATS:
        return _MATS[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes.get('Principled BSDF')
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    b.inputs['Base Color'].default_value = (lin[0], lin[1], lin[2], 1.0)
    b.inputs['Metallic'].default_value = metal
    b.inputs['Roughness'].default_value = rough
    m.diffuse_color = (lin[0], lin[1], lin[2], 1.0)
    m.metallic, m.roughness = metal, rough
    _MATS[name] = m
    return m


def make(name, geo, mat, parent=None, loc=(0, 0, 0), rot=None, props=None, sharp=38.0, weld=0.0):
    """Turn a Geo into a mesh object.  Takes the object name, the Geo, a material name, an optional parent object,
    the location in millimetres, an optional Quaternion, a dict of custom properties (exported as glTF extras),
    the auto-smooth angle in degrees and an optional weld distance in millimetres.  Returns the object."""
    bm = geo.bm
    if weld > 0:
        bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=weld)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    for v in bm.verts:
        v.co *= MM
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    me.set_sharp_from_angle(angle=math.radians(sharp))
    me.materials.append(_MATS[mat])
    return _object(name, me, parent, loc, rot, props)


def instance(src, name, parent=None, loc=(0, 0, 0), rot=None, props=None):
    """Place another object that shares src's mesh.  Takes the source object, the new name, an optional parent,
    the location in millimetres, an optional Quaternion and custom properties.  Returns the object."""
    return _object(name, src.data, parent, loc, rot, props)


def empty(name, parent=None, loc=(0, 0, 0), rot=None, props=None):
    """Create an empty used as a pivot or group.  Takes a name, optional parent, location in millimetres,
    optional Quaternion and custom properties.  Returns the object."""
    return _object(name, None, parent, loc, rot, props)


def _object(name, data, parent, loc, rot, props):
    """Create and link one object.  Takes the name, mesh data or None, the parent, a location in millimetres, a
    Quaternion or None and a property dict.  Returns the object."""
    o = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(o)
    o.rotation_mode = 'QUATERNION'
    o.location = V(loc) * MM
    if rot is not None:
        o.rotation_quaternion = rot
    if parent is not None:
        o.parent = parent
    for k, v in (props or {}).items():
        o[k] = v
    return o


def quat_from_matrix(m):
    """Rotation part of a placement matrix.  Takes a 4x4 Matrix.  Returns a Quaternion."""
    return m.to_3x3().to_quaternion()
