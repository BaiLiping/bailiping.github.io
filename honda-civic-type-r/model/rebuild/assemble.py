"""Assemble the rebuilt Civic exterior with the kept running gear, engine, drivetrain and cabin.

Run: blender -b --python assemble.py -- <base.glb> <panels.npz> <out.glb> [--parts a.npz[,b.npz,...]]
Each npz holds <name>/V, /F, /N (and optionally /UV: float32 N x 2 per vertex, glTF convention) plus an index;
a material key not in MATS is looked up as an existing material name of the base GLB (e.g. Mat_Civic_Script).
Every exterior mesh of the base model is deleted (the body, glass, lamps, trim, mirrors, handles, badges);
the rig pivots, wheels, brakes, suspension, engine bay, drivetrain, underbody, exhaust and cabin are kept.
The new panels (cutpanels.py) and parts are added under the pivots the viewer animates (doors, hood, hatch,
fuel flap) using the site's material names. Derived parts: the mirror's body-colour cap (from the black
housing, mirror_cap), a dark filler pocket behind the fuel flap (fuel_pocket) and body B-pillars inside the door
appliques (b_pillar, so the roof is carried when both doors are open). Kept cabin, door-card and engine-bay
meshes are pulled back inside the new shell (fit_inside: subdivided first where the skin is concave across a face,
so no face spans through it between fitted vertices; deeper inside the clearance footprints a parts npz may carry,
e.g. the door backings / hood inner frame / hatch trim from sides_openings.py; door cards clamped into their
door outline). A parts npz may also carry 'cutout' (cut-piece faces moved to a pivot group as inlays). The
kept wheels are set to Honda's frames (fit_wheels: loaded height, rim lip radius, caliper clock position).
fit_inside and b_pillar use the designed field body-<tag>.npz next to the panels npz. Exports a GLB.
Returns nothing.
"""
import bpy, sys, re, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import panels
from finishes import BODY_PAINT, POCKET_PAINT, WHEEL_PAINT, FINISHES, set_blender_finish

argv = sys.argv[sys.argv.index('--') + 1:]
base, panels_npz, out = argv[:3]
parts_npz = argv[argv.index('--parts') + 1].split(',') if '--parts' in argv else []   # one or more, comma separated

REMOVE = re.compile(r'^Civic_(Badge_|Body_Shell_|Fender_|Front_|Fuel_|Glass_|Headlamp_|Pillar_Trim|Rear_(?!Seat)|Roof_|'
                    r'Side_Skirt|Taillamp_|Wheelhouse_|Windshield_Cowl|Door_Frame_|Door_Handle_|Door_Shell_|Mirror_|Hatch_|Hood_)')

MATS = {  # cutpanels material key -> site material name (created if missing) and fallback colour
    'paint': (BODY_PAINT, FINISHES[BODY_PAINT].color),
    'pocket': (POCKET_PAINT, FINISHES[POCKET_PAINT].color),    # door-handle recesses: paint in shadow
    'glass': ('Mat_Automotive_Glass', (0.02, 0.03, 0.04)),
    'privacy': ('Mat_Rear_Privacy_Glass', (0.01, 0.01, 0.012)),
    'gloss_black': ('Mat_Gloss_Black_Trim', (0.01, 0.01, 0.012)),
    'black': ('Mat_Aero_Matte_Black', (0.025, 0.025, 0.027)),
    'headlamp': ('Mat_FK8_Headlamp_Glass', (0.05, 0.055, 0.06)),
    'taillamp': ('Mat_Taillight_Red_Lens', (0.45, 0.01, 0.015)),
    'grille': ('Mat_FK8_Grille_Graphite', (0.03, 0.03, 0.035)),
    'fog': ('Mat_FK8_Lightguide_Opal', (0.8, 0.82, 0.85)),
    'red': ('Mat_Brake_Caliper_Red', (0.6, 0.01, 0.01)),
    'liner': ('Mat_Wheelhouse_Liner', (0.02, 0.02, 0.02)),
    'seam': ('Mat_Black_Duct', (0.01, 0.01, 0.01)),
    'chrome': ('Mat_Exhaust_Chrome', (0.8, 0.8, 0.8)),
    'mirror': ('Mat_Mirror_Chrome', (0.8, 0.8, 0.8)),
    'carbon': ('Mat_Carbon_Fiber', (0.03, 0.03, 0.03)),
}


def material(key):
    """Takes a material key; returns the site material (existing one reused, else a new principled one)."""
    name, col = MATS.get(key, (key, (0.5, 0.5, 0.5)))
    m = bpy.data.materials.get(name)
    if m is None:
        m = bpy.data.materials.new(name); m.use_nodes = True
        b = m.node_tree.nodes['Principled BSDF']; b.inputs['Base Color'].default_value = (*col, 1)
        b.inputs['Roughness'].default_value = 0.35 if key != 'black' else 0.7
        if key == 'headlamp':
            b.inputs['Metallic'].default_value = 0.6; b.inputs['Roughness'].default_value = 0.12
    # The base GLB can supply an older metallic paint. Set both the reused
    # and newly built materials to the same pigment and protective clear coat.
    if key in ('paint', 'pocket'):
        set_blender_finish(m)
    return m


# FK8 door mirror: one black housing (parts_mid.json, the whole head outline) under a body-colour cap. The cap is
# the housing skin above Honda's cap line (side regions 79 | 101, 75), lifted 2.5 mm off it.
CAP_LINE = ((0.482, 1.020), (0.716, 0.991))     # side view (x, y) of the 79 | 101 stroke, ends
CAP_LIFT = 0.0025


def mirror_cap(V, F, N):
    """Takes the housing mesh (glTF axes); returns the cap mesh (V, F, N): faces above the cap line, offset
    outward along the normals."""
    (x0, y0), (x1, y1) = CAP_LINE
    ys = y0 + (np.clip(V[:, 0], x0, x1) - x0) * (y1 - y0) / (x1 - x0)
    up = V[:, 1] > ys + 0.001
    keep = up[F].all(1)
    F2 = F[keep]; used = np.unique(F2)
    remap = -np.ones(len(V), int); remap[used] = np.arange(len(used))
    return V[used] + N[used] * CAP_LIFT, remap[F2], N[used]


POCKET_DEPTH = 0.045


def fuel_pocket(V, F, N):
    """Takes the fuel flap mesh; returns a dark filler pocket behind it (V, F, N): the flap skin pushed
    POCKET_DEPTH inward as the back wall plus side walls from the flap's outline, so the shut line around the
    closed flap reads dark and the open flap shows a recess instead of the cargo bay."""
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    u, cnt = np.unique(E, axis=0, return_counts=True)
    B = u[cnt == 1]                                              # outline edges
    n = len(V); c = V.mean(0)
    rad = V - c; rad -= N * (rad * N).sum(1, keepdims=True); rad /= np.maximum(np.linalg.norm(rad, axis=1, keepdims=True), 1e-9)
    Vr = V + rad * 0.004 - N * 0.0015                             # rim: under the shut line around the flap
    Vb = V - N * POCKET_DEPTH                                     # back wall
    walls = np.concatenate([np.stack([B[:, 0], B[:, 1], B[:, 1] + n], 1), np.stack([B[:, 0], B[:, 1] + n, B[:, 0] + n], 1)])
    Nr = -rad * 0.9 + N * 0.1; Nr /= np.linalg.norm(Nr, axis=1, keepdims=True)
    return np.vstack([Vr, Vb]), np.vstack([F + n, walls]), np.vstack([Nr, N])


# Body B-pillar behind the door appliques (side regions 31 | 33) and the doors' shared shut line, 30 mm inside the
# skin: hidden with the doors shut, it carries the roof when both doors on a side are open.
BPILLAR = [(-0.28, 1.335), (-0.41, 1.335), (-0.37, 0.98), (-0.30, 0.95), (-0.30, 0.30), (-0.17, 0.30), (-0.17, 0.95),
           (-0.19, 0.98)]
BPILLAR_INSET, BPILLAR_STEP = 0.030, 0.012


def _inside(poly, P):
    poly = np.asarray(poly); x, y = P[:, 0], P[:, 1]; res = np.zeros(len(P), bool); j = len(poly) - 1
    for i in range(len(poly)):
        (xi, yi), (xj, yj) = poly[i], poly[j]
        res ^= ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi); j = i
    return res


def b_pillar(field, side):
    """Takes the body field and side (-1 left, +1 right); returns the B-pillar mesh (V, F, N): a grid over the
    BPILLAR outline (side view) placed BPILLAR_INSET inside the body surface found along Z."""
    P = np.asarray(BPILLAR); xs = np.arange(P[:, 0].min(), P[:, 0].max() + 1e-9, BPILLAR_STEP)
    ys = np.arange(P[:, 1].min(), P[:, 1].max() + 1e-9, BPILLAR_STEP)
    G = np.stack(np.meshgrid(xs, ys, indexing='ij'), -1).reshape(-1, 2); nx, ny = len(xs), len(ys)
    lo, hi = np.zeros(len(G)), np.full(len(G), 1.2)                # |z| of the skin: last inside -> outside
    for _ in range(24):
        m = 0.5 * (lo + hi)
        f = field(np.stack([G[:, 0], G[:, 1], side * m], 1))
        lo = np.where(f < 0, m, lo); hi = np.where(f < 0, hi, m)
    Z = side * np.maximum(lo - BPILLAR_INSET, 0.3)
    V = np.stack([G[:, 0], G[:, 1], Z], 1)
    ok = _inside(P, G).reshape(nx, ny) & (lo > 0.5).reshape(nx, ny)
    F = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            if ok[i, j] and ok[i + 1, j] and ok[i, j + 1] and ok[i + 1, j + 1]:
                a, b, c, d = i * ny + j, (i + 1) * ny + j, (i + 1) * ny + j + 1, i * ny + j + 1
                F += [(a, b, c), (a, c, d)]
    F = np.array(F); used = np.unique(F); remap = -np.ones(len(V), int); remap[used] = np.arange(len(used))
    N = np.tile([0.0, 0.0, float(side)], (len(used), 1))
    return V[used], remap[F], N


def add_npz(path, report, drop=None):
    d = np.load(path); index = json.loads(str(d['index']))
    geo = {e['name']: (d[e['name'] + '/V'], d[e['name'] + '/F'], d[e['name'] + '/N']) for e in index}
    for n, faces in (drop or {}).items():                          # faces moved to a pivot group as inlays
        if n in geo:
            V, F, N = geo[n]; keep = np.ones(len(F), bool); keep[[i for i in faces if i < len(F)]] = False
            geo[n] = (V, F[keep], N)
    if 'Civic_Fuel_Flap' in geo and 'Civic_Fuel_Pocket' not in geo:
        geo['Civic_Fuel_Pocket'] = fuel_pocket(*geo['Civic_Fuel_Flap'])
        index.append({'name': 'Civic_Fuel_Pocket', 'mat': 'seam', 'pivot': ''})
    for side in ('L', 'R'):
        h = 'Civic_Mirror_' + side
        if h in geo and h + '_Painted_Cap' not in geo:
            e = next(e for e in index if e['name'] == h)
            geo[h + '_Painted_Cap'] = mirror_cap(*geo[h])
            index.append({'name': h + '_Painted_Cap', 'mat': 'paint', 'pivot': e.get('pivot', '')})
    for e in index:
        n = e['name']; V, F, N = geo[n]
        if len(F) == 0: continue
        piv = bpy.data.objects.get(e['pivot']) if e.get('pivot') else bpy.data.objects.get('Pivot_Body_Assembly')
        ob = panels.make_obj(n, V, F.tolist(), N, material(e['mat']), parent=piv)
        if n + '/UV' in d: set_uv(ob, d[n + '/UV'])
        report.append((n, len(ob.data.polygons), piv.name if piv else '-'))


def set_uv(ob, UV):
    """Takes an object and per-vertex UVs (N x 2, glTF convention: v down from the image top); sets them as the
    mesh's UV map (Blender stores v up, the glTF exporter flips it back)."""
    UV = np.asarray(UV, np.float32)
    me = ob.data
    if len(UV) != len(me.vertices):
        print('UV size mismatch for', ob.name, len(UV), len(me.vertices)); return
    lay = me.uv_layers.new(name='UVMap')
    vi = np.zeros(len(me.loops), np.int64); me.loops.foreach_get('vertex_index', vi)
    uv = UV[vi].copy(); uv[:, 1] = 1.0 - uv[:, 1]
    lay.data.foreach_set('uv', uv.ravel())


class Field:
    """The designed body field (approximate signed distance, metres, negative inside) with trilinear sampling
    and a central-difference gradient, in glTF axes."""
    def __init__(self, path):
        d = np.load(path); self.F = d['F'].astype(np.float32); self.h = float(d['h'])
        self.o = np.array([float(d['x0']), float(d['y0']), float(d['z0'])])

    def __call__(self, P):
        g = (P - self.o) / self.h; n = np.array(self.F.shape) - 1
        g = np.clip(g, 0, n - 1e-6); i = np.floor(g).astype(int); t = g - i
        out = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (t[:, 0] if dx else 1 - t[:, 0]) * (t[:, 1] if dy else 1 - t[:, 1]) * (t[:, 2] if dz else 1 - t[:, 2])
                    out = out + w * self.F[i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz]
        inside = np.all(((P - self.o) / self.h >= 0) & ((P - self.o) / self.h <= n), axis=1)
        return np.where(inside, out, 1.0)

    def grad(self, P, e=0.004):
        return np.stack([(self(P + d) - self(P - d)) / (2 * e) for d in np.eye(3) * e], 1)


# kept meshes that live inside the shell: they must stay MARGIN inside the new body / glass / roof / wheelhouses
FIT_PIVOTS = re.compile(r'^Pivot_(Interior_Cabin|Door_(L|R|RL|RR)$|Engine_)')
MARGIN, SOFT, Y_MIN, REACH = 0.010, 0.008, 0.25, 0.35
WHEEL_X, WHEEL_Y = (1.3484, -1.3484), 0.3195
DRY = '--fit-dry' in argv


def cutouts(paths):
    """Takes parts npz paths; returns {cut piece name: set of face indices} to drop (moved to a pivot group as
    inlays by sides_openings.py)."""
    out = {}
    for p in paths:
        if not p: continue
        d = np.load(p)
        if 'cutout' in d.files:
            for k, v in json.loads(str(d['cutout'])).items(): out.setdefault(k, set()).update(v)
    return out


# The base model's rear door cards end in a lower-rear wedge (x < -0.80, y < 0.56) shaped for its old, longer
# door; the new rear doors are cut back around the arch there, so the wedge sticks out of the opened door.
DOORCARD_DROP = (re.compile(r'^Civic_DoorCard_(Pod_|Trim_|Stitch_)?R[LR]$'), -0.80, 0.56)


def clamp_into(ob, c, field=None):
    """Takes a kept mesh under a door pivot and the door's clearance entry; moves its vertices that lie outside
    the door outline (side view) to just inside it (c['clamp'] metres), so door cards never hang out of the
    opened door; drops faces left outside and deep inboard flanges at the door edge. Returns (moved vertices,
    dropped faces)."""
    poly = np.asarray(c['poly'], float); e1, e2 = np.array(c['e1']), np.array(c['e2'])
    mw = np.array(ob.matrix_world); co = np.array([v.co for v in ob.data.vertices])
    Wb = co @ mw[:3, :3].T + mw[:3, 3]; W = np.stack([Wb[:, 0], Wb[:, 2], -Wb[:, 1]], 1)
    uv = np.stack([W @ e1, W @ e2], 1)
    out = ~_in_poly(poly, uv)
    A = poly; B = np.roll(poly, -1, 0); AB = B - A
    if 0.5 * np.sum(A[:, 0] * B[:, 1] - B[:, 0] * A[:, 1]) < 0: A, B = B[::-1], A[::-1]; AB = B - A
    Q = uv[out] if out.any() else np.zeros((0, 2))
    t = np.clip(((Q[:, None, :] - A[None]) * AB[None]).sum(-1) / np.maximum((AB * AB).sum(-1), 1e-12)[None], 0, 1)
    Pn = A[None] + t[..., None] * AB[None]; dist = np.linalg.norm(Pn - Q[:, None, :], axis=-1); k = dist.argmin(1)
    near = Pn[np.arange(len(Q)), k]; tg = AB[k] / np.maximum(np.linalg.norm(AB[k], axis=1, keepdims=True), 1e-12)
    inward = np.stack([-tg[:, 1], tg[:, 0]], 1)                       # left normal of a counter-clockwise loop
    uv2 = near + inward * c['clamp']
    W2 = W[out] + (uv2 - Q)[:, :1] * e1 + (uv2 - Q)[:, 1:2] * e2
    W[out] = W2
    Wb = np.stack([W[:, 0], -W[:, 2], W[:, 1]], 1); inv = np.linalg.inv(mw)
    if ob.data.users > 1: ob.data = ob.data.copy()
    ob.data.vertices.foreach_set('co', (Wb @ inv[:3, :3].T + inv[:3, 3]).astype(np.float32).ravel()); ob.data.update()
    # big faces can still span a concave notch of the outline (rear door's arch cut): drop faces whose centre
    # stays outside
    import bmesh
    bm = bmesh.new(); bm.from_mesh(ob.data); bm.faces.ensure_lookup_table()
    uvv = np.stack([W @ e1, W @ e2], 1)
    cen = np.array([uvv[[v.index for v in f.verts]].mean(0) for f in bm.faces]) if len(bm.faces) else np.zeros((0, 2))
    drop = ~_in_poly(poly, cen)
    if field is not None and len(cen):
        # deep inboard flanges right at the door's edge (a card wedge left over where the outline was cut back
        # around the wheel arch) stick out of the opened door: drop faces within 6 cm of the outline and more
        # than 16 cm inside the skin
        C3 = np.array([W[[v.index for v in f.verts]].mean(0) for f in bm.faces])
        A2 = poly; B2 = np.roll(poly, -1, 0); AB2 = B2 - A2
        tt = np.clip(((cen[:, None, :] - A2[None]) * AB2[None]).sum(-1) / np.maximum((AB2 * AB2).sum(-1), 1e-12)[None], 0, 1)
        de = np.linalg.norm(A2[None] + tt[..., None] * AB2[None] - cen[:, None, :], axis=-1).min(1)
        drop |= (de < 0.06) & (-field(C3) > 0.16)
    if DOORCARD_DROP[0].match(ob.name) and len(cen):
        C3 = np.array([W[[v.index for v in f.verts]].mean(0) for f in bm.faces])
        drop |= (C3[:, 0] < DOORCARD_DROP[1]) & (C3[:, 1] < DOORCARD_DROP[2])
    gone = [f for f, o in zip(bm.faces, drop) if o]
    if gone: bmesh.ops.delete(bm, geom=gone, context='FACES')
    bm.to_mesh(ob.data); bm.free(); ob.data.update()
    return int(out.sum()), len(gone)


def clearances(paths):
    """Takes parts npz paths; returns the clearance footprints they carry ('clear': [{a, e1, e2, poly, margin}],
    written by sides_openings.py): kept meshes inside a footprint must stay `margin` deep in the body there."""
    out = []
    for p in paths:
        if not p: continue
        d = np.load(p)
        if 'clear' in d.files: out += json.loads(str(d['clear']))
    return out


def _in_poly(poly, P):
    poly = np.asarray(poly); x, y = P[:, 0], P[:, 1]; res = np.zeros(len(P), bool); j = len(poly) - 1
    for i in range(len(poly)):
        (xi, yi), (xj, yj) = poly[i], poly[j]
        res ^= ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi); j = i
    return res


def margins(W, clear):
    """Per-vertex required depth: MARGIN, raised inside the clearance footprints (door inner frames, B-pillars,
    hood inner frame, hatch trim)."""
    m = np.full(len(W), MARGIN)
    for c in clear:
        e1, e2 = np.array(c['e1']), np.array(c['e2'])
        inside = _in_poly(c['poly'], np.stack([W @ e1, W @ e2], 1))
        m = np.where(inside, np.maximum(m, c['margin']), m)
    return m


def fit_inside(field, clear=()):
    for c in clear:                                                 # door cards into their door outlines first
        if not c.get('clamp'): continue
        for ob in [o for o in bpy.data.objects if o.type == 'MESH' and not REMOVE.match(o.name)]:
            p = ob.parent
            while p is not None and not p.name.startswith('Pivot_'): p = p.parent
            if p is not None and p.name == c.get('pivot'):
                n, g = clamp_into(ob, c, field)
                if n or g: print(f'  clamp {ob.name:36s} {n:5d} verts into the door outline, {g} edge faces dropped')
    return _fit_inside(field, clear)


SUBDIV_EDGE, SUBDIV_BAND, SUBDIV_ITERS = 0.015, 0.010, 7


def _world(ob):
    mw = np.array(ob.matrix_world)
    return lambda co: (lambda Wb: np.stack([Wb[:, 0], Wb[:, 2], -Wb[:, 1]], 1))(co @ mw[:3, :3].T + mw[:3, 3])


def subdivide_near_skin(ob, field, clear, edge=SUBDIV_EDGE):
    """Takes a kept mesh; splits its edges longer than `edge` where an edge middle or face centre lies within
    SUBDIV_BAND of the required depth and nearer the skin than its corners (the skin is concave across it), so
    that after the vertex fit no face spans across the new skin between fitted vertices (low-poly cargo bay walls
    under the taillamp inner corners).
    Returns the number of added vertices."""
    import bmesh
    if ob.data.users > 1: ob.data = ob.data.copy()
    me = ob.data; n0 = len(me.vertices); to_w = _world(ob)
    bm = bmesh.new(); bm.from_mesh(me)
    for _ in range(SUBDIV_ITERS):
        bm.verts.ensure_lookup_table(); bm.edges.ensure_lookup_table(); bm.faces.ensure_lookup_table()
        W = to_w(np.array([v.co for v in bm.verts]))
        lim = lambda P: -margins(P, clear) - SOFT - SUBDIV_BAND
        Ei = np.array([[e.verts[0].index, e.verts[1].index] for e in bm.edges])
        if not len(Ei): break
        L = np.linalg.norm(W[Ei[:, 0]] - W[Ei[:, 1]], axis=1)
        M = 0.5 * (W[Ei[:, 0]] + W[Ei[:, 1]])
        fv = field(W); fm = field(M)
        # only where the skin is concave across the edge / face (its middle nearer the skin than its corners) and
        # near the required depth: a flat card face parallel to the skin is left alone (the vertex fit moves it)
        hot = (fm > lim(M)) & (fm > np.maximum(fv[Ei[:, 0]], fv[Ei[:, 1]]) + 0.002)
        fhot = set()
        if len(bm.faces):
            Fv = [[v.index for v in f.verts] for f in bm.faces]
            C = np.array([W[v].mean(0) for v in Fv]); fc = field(C)
            fmax = np.array([fv[v].max() for v in Fv])
            for k in np.nonzero((fc > lim(C)) & (fc > fmax + 0.002))[0]:
                fhot.update(e.index for e in bm.faces[int(k)].edges)
        hot[list(fhot)] = True
        sel = [bm.edges[int(i)] for i in np.nonzero(hot & (L > edge) & (M[:, 1] > Y_MIN))[0]]
        if not sel: break
        bmesh.ops.subdivide_edges(bm, edges=sel, cuts=1, use_grid_fill=True)
        bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
    bm.to_mesh(me); bm.free(); me.update()
    return len(me.vertices) - n0


def _fit_inside(field, clear=()):
    """Takes the body field; pulls every vertex of the kept cabin, door-card and engine-bay meshes (above Y_MIN,
    the open underbody stays) that lies outside, or within its margin, of the new body surface back inside it: it
    walks along the inward field gradient (straight inboard inside a wheel arch; fixed directions when the
    gradient is blocked; bisection, up to REACH) to the margin + a soft tanh roll-off, so nothing pokes through
    the glass, roof, panels or wheelhouse liners and nothing gets flattened. Meshes are first subdivided near
    the skin (subdivide_near_skin); a second, finer pass runs where face centres still cross.
    Returns [(name, poking verts, moved verts, max excess mm, max after mm)]."""
    out = []
    for ob in [o for o in bpy.data.objects if o.type == 'MESH']:
        p = ob.parent
        while p is not None and not p.name.startswith('Pivot_'): p = p.parent
        if p is None or not FIT_PIVOTS.match(p.name) or REMOVE.match(ob.name): continue
        r = _fit_one(ob, field, clear, SUBDIV_EDGE)
        if r and not DRY and _crossing(ob, field, clear):
            r2 = _fit_one(ob, field, clear, SUBDIV_EDGE / 2)
            if r2: r = (r[0], r[1], r[2] + r2[2], r[3], r2[4])
        if r: out.append(r)
    return out


def _crossing(ob, field, clear):
    if not len(ob.data.polygons): return False
    W = _world(ob)(np.array([v.co for v in ob.data.vertices]))
    C = W[np.array([p.vertices[:3] for p in ob.data.polygons])].mean(1)
    return bool(((field(C) > -0.5 * margins(C, clear)) & (C[:, 1] > Y_MIN)).any())


def _fit_one(ob, field, clear, edge):
    if True:
        mw = np.array(ob.matrix_world); co = np.array([v.co for v in ob.data.vertices])
        if len(co) == 0: return None
        Wb = co @ mw[:3, :3].T + mw[:3, 3]
        W = np.stack([Wb[:, 0], Wb[:, 2], -Wb[:, 1]], 1)            # Blender -> glTF
        f = field(W); f0 = -margins(W, clear) - SOFT
        bad = (f > f0) & (W[:, 1] > Y_MIN)
        if not DRY and len(ob.data.polygons):                          # faces may cross between vertices
            C = W[np.array([p.vertices[:3] for p in ob.data.polygons])].mean(1)
            if bad.any() or ((field(C) > -margins(C, clear) - SOFT) & (C[:, 1] > Y_MIN)).any():
                added = subdivide_near_skin(ob, field, clear, edge)
                if added:
                    print(f'  subdivide {ob.name:32s} +{added} verts near the skin')
                    co = np.array([v.co for v in ob.data.vertices]); Wb = co @ mw[:3, :3].T + mw[:3, 3]
                    W = np.stack([Wb[:, 0], Wb[:, 2], -Wb[:, 1]], 1)
                    f = field(W); f0 = -margins(W, clear) - SOFT
                    bad = (f > f0) & (W[:, 1] > Y_MIN)
        if not bad.any(): return None
        n_poke = int(((f > 0) & bad).sum()); exc = float(f[bad].max())
        if DRY:
            q = W[(f > 0) & bad]
            if len(q): print('   poke', ob.name, np.round(q.min(0), 3), np.round(q.max(0), 3))
            return (ob.name, n_poke, int(bad.sum()), exc * 1000, exc * 1000)
        if ob.data.users > 1: ob.data = ob.data.copy()
        P = W[bad]; f0 = f0[bad]; target = f0 + SOFT * np.tanh((f[bad] - f0) / SOFT)
        g = field.grad(P); d = -g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
        well = np.zeros(len(P), bool)                                # inside a wheel arch: straight inboard
        for wx in WHEEL_X:
            well |= (np.hypot(P[:, 0] - wx, P[:, 1] - WHEEL_Y) < 0.47) & (np.abs(P[:, 2]) > 0.45)
        d[well] = np.stack([np.zeros(well.sum()), np.zeros(well.sum()), -np.sign(P[well, 2])], 1)
        lo, hi = np.zeros(len(P)), np.full(len(P), REACH)
        ok = field(P + d * REACH) <= target
        for alt in ([1.0, 0, 0], [0, 0, 0], [0, -1.0, 0], [0.7, -0.7, 0]):   # gradient blocked: try fixed ways in
            if ok.all(): break
            da = np.tile(alt, (len(P), 1)) if any(alt) else np.stack([np.zeros(len(P)), np.zeros(len(P)), -np.sign(P[:, 2])], 1)
            fix = ~ok & (field(P + da * REACH) <= target)
            d[fix] = da[fix]; ok |= fix
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            inside = field(P + d * mid[:, None]) <= target
            hi = np.where(inside, mid, hi); lo = np.where(inside, lo, mid)
        P2 = np.where(ok[:, None], P + d * hi[:, None], P)
        W[bad] = P2
        Wb = np.stack([W[:, 0], -W[:, 2], W[:, 1]], 1)
        inv = np.linalg.inv(mw)
        co2 = Wb @ inv[:3, :3].T + inv[:3, 3]
        ob.data.vertices.foreach_set('co', co2.astype(np.float32).ravel())
        ob.data.update()
        return (ob.name, n_poke, int(ok.sum()), exc * 1000, float(field(W[bad]).max()) * 1000)



# Kept wheels against Honda's frames 01/19 (red rim-stripe circle fit and tyre top edge; the body registers to
# 1 px): Honda's rim centres sit ~2 px (9 mm) below P24's loaded 0.3195 and 18 mm below the base's 0.3275, the red
# rim lip is 270 mm in radius (base stripe 259) and the tread top matches at y 0.652, so the tyre's outer radius
# is ~343 mm (base 327.5): the loaded tyre then sits ~3 cm into the studio floor, as Honda's contact does.
# Calipers: Honda's sit at ~30 deg (front) and ~134 deg (rear) from the forward axis toward the top (base 4 / 9).
WHEEL_DROP, RIM_SCALE, TYRE_OUT_SCALE = 0.018, 1.045, 1.046
CALIPER_TURN = {'F': 26.0, 'R': 125.0}                             # degrees, forward -> up


def fit_wheels():
    """Drops the four corner pivots to the loaded wheel-centre height and enlarges the rims (and the tyres'
    beads with them) radially about each hub, keeping the tread radius close to Honda's. Returns a report."""
    # The spokes and barrel share the wheel's black coating. The brake disc
    # and machined fasteners retain their metal surfaces.
    rim = bpy.data.materials.get(WHEEL_PAINT)
    if rim:
        set_blender_finish(rim)
    rep = []
    for c in ('FL', 'FR', 'RL', 'RR'):
        corner = bpy.data.objects.get('Pivot_Corner_' + c); spin = bpy.data.objects.get('Pivot_WheelSpin_' + c)
        if corner is None or spin is None: continue
        mw = corner.matrix_world.copy(); mw.translation.z -= WHEEL_DROP; corner.matrix_world = mw   # Blender z = up
        bpy.context.view_layer.update()
        ctr = spin.matrix_world.translation                         # Blender (x, -z_gl, y_gl)
        cx, cy = ctr.x, ctr.z
        for ob in [o for o in bpy.data.objects if o.type == 'MESH' and o.name.startswith('Civic_Wheel_' + c + '_')]:
            kind = 'rim' if re.search(r'_Rim(_Pinstripe)?$', ob.name) else 'tyre' if re.search(r'_(Tire|Tread_Shoulder)$', ob.name) else None
            if kind is None: continue
            if ob.data.users > 1: ob.data = ob.data.copy()
            mw = np.array(ob.matrix_world); co = np.array([v.co for v in ob.data.vertices])
            Wb = co @ mw[:3, :3].T + mw[:3, 3]
            dx, dy = Wb[:, 0] - cx, Wb[:, 2] - cy; r = np.hypot(dx, dy)
            if kind == 'rim':
                k = np.full(len(r), RIM_SCALE)
            else:
                r_in, r_out = r.min(), r.max()
                r2 = r_in * RIM_SCALE + (r - r_in) * (r_out * TYRE_OUT_SCALE - r_in * RIM_SCALE) / (r_out - r_in)
                k = r2 / np.maximum(r, 1e-9)
            Wb[:, 0] = cx + dx * k; Wb[:, 2] = cy + dy * k
            inv = np.linalg.inv(mw)
            ob.data.vertices.foreach_set('co', (Wb @ inv[:3, :3].T + inv[:3, 3]).astype(np.float32).ravel()); ob.data.update()
            rep.append((ob.name, kind, round(float(r.max()), 4), round(float((r * k).max()), 4)))
        from mathutils import Matrix
        a = np.radians(CALIPER_TURN[c[0]])
        T = Matrix.Translation(corner.matrix_world.translation)
        M = T @ Matrix.Rotation(-a, 4, 'Y') @ T.inverted()        # glTF x->y turn = Blender rotation about -Y
        for ob in [o for o in bpy.data.objects if o.type == 'MESH' and re.match(r'Civic_Brake_(Caliper|Caliper_Fasteners|Caliper_Logo|Bracket)_' + c + '$', o.name)]:
            ob.matrix_world = M @ ob.matrix_world
            rep.append((ob.name, 'turn', 0.0, float(CALIPER_TURN[c[0]])))
    return rep


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=base)
    gone = [o.name for o in bpy.data.objects if o.type == 'MESH' and REMOVE.match(o.name)]
    for n in gone: bpy.data.objects.remove(bpy.data.objects[n], do_unlink=True)
    for o in [o for o in bpy.data.objects if o.type == 'EMPTY' and REMOVE.match(o.name) and not o.children]:
        bpy.data.objects.remove(o, do_unlink=True)
    print('removed', len(gone), 'old exterior meshes')
    fpath = Path(panels_npz).with_name(Path(panels_npz).name.replace('cut-', 'body-', 1))
    if fpath.exists() and fpath != Path(panels_npz):
        for n, poke, moved, exc, after in fit_inside(Field(str(fpath)), clearances(parts_npz)):
            print(f'  fit {n:40s} poking {poke:5d} moved {moved:5d} verts, max {exc:6.1f} mm -> {after:6.1f} mm')
    for n, kind, r0, r1 in fit_wheels():
        print(f'  wheel {n:36s} {kind:5s} r {r0:.4f} -> {r1:.4f}')
    report = []
    has_pillar = any(e['name'].startswith('Civic_Body_Shell_BPillar_') for p in parts_npz if p
                     for e in json.loads(str(np.load(p)['index'])))
    if fpath.exists() and fpath != Path(panels_npz) and not has_pillar:     # sides_openings.py brings its own
        fld = Field(str(fpath)); body = bpy.data.objects.get('Pivot_Body_Assembly')
        for side, sfx in ((-1, 'L'), (1, 'R')):
            V, F, N = b_pillar(fld, side)
            ob = panels.make_obj('Civic_Body_Shell_BPillar_' + sfx, V, F.tolist(), N, material('black'), parent=body)
            report.append((ob.name, len(ob.data.polygons), body.name if body else '-'))
    add_npz(panels_npz, report, cutouts(parts_npz))
    for p in parts_npz:
        if p: add_npz(p, report)
    for n, f, p in report: print(f'  {n:40s} {f:7d} faces  under {p}')
    bpy.ops.export_scene.gltf(filepath=out, export_format='GLB', export_extras=True, export_yup=True,
                              export_apply=False, export_animations=False, export_draco_mesh_compression_enable=False)
    print('wrote', out)


main()
