"""Cut the body mesh into panels along Honda's drawn lines.

Run: venv/bin/python cutpanels.py <body-mesh.npz> <field.npz> <specs.json> <out.npz>
specs.json: [{"names": [left, right] | "name": both, "terms": [[view, ids, neg?], ...], "face": side|top|front|rear|none,
              "mat": ..., "pivot": ...}, ...] in priority order.
A panel is the part of the body surface where every term holds: the projection into `view` lies inside the
merged drawing regions `ids` (or outside, when neg). Top and front masks are made left/right symmetric. For
named left/right panels the surface is further limited to that side; `face` keeps only surface facing that
view, which drops e.g. the underbody under a roof outline. Membership is a smooth signed distance (metres,
from the region masks), so triangles are split exactly where it crosses zero and the panel edges follow the
drawn outlines. Unclaimed exterior surface is paint except at the specified closure perimeters. Normals of new vertices come
from the field gradient.
Optional spec keys: nsmooth (voxels, default 2.5) smooths the field the panel's normals come from; nflat: true
interpolates the panel's normals harmonically from its outline (a flush flap without the hull's bumps); grow (m, default 0) moves the panel edge outward into Honda's stroke (narrower, even shut
lines); close (px) closes the merged drawing regions further (swallows small inner strokes, e.g. around the door
handles). Extra term: ["hp", a, b, c] = side-view half-plane a*X + b*Y + c < 0 (straight cuts across a region).
A view term may carry a 4th element, an offset in metres that dilates the merged regions before the test:
[view, ids, "neg", 0.01] keeps only surface more than 1 cm outside those regions (their shut lines included).
["face", dir, thr, "neg"] inside a group keeps surface facing dir LESS than thr (e.g. the inner wheelhouse lip).
Triangles are partitioned exactly in priority order (clip_all), so the pieces tile the body with no holes.
Panels marked "mesh": "fine" receive an extra subdivision of the common surface before any cuts.
Their neighbours share the same edge vertices, including the transition to the coarse triangles.
Despeckle: islands of seams / catch-all under 1.5 cm2 go to the panel bordering them most (env CUT_DESPECKLE=<cm2>,
0 disables).
Writes one npz with per-panel V, F, N (glTF axes) plus a JSON index of names/materials/pivots.
"""
import sys, json
import numpy as np
import cv2
from scipy import ndimage
import p24, outlines, tailgate
import hood_pressing as hood
from design_front import press_hood_ridges

fd = np.load(sys.argv[2]); FLD = ndimage.gaussian_filter(fd['F'], 2.5)   # normals from a smoother field: even reflections
GRID = np.array([float(fd['x0']), float(fd['y0']), float(fd['z0'])]); H = float(fd['h'])
GX, GY, GZ = np.gradient(FLD)
SDF = {}


GRADS = {}


def field_normals(P, sigma=None):
    """Unit outward normals from the smoothed field gradient at points P (N,3); sigma (voxels) other than the
    default 2.5 uses a field smoothed that much more (spec key nsmooth: even shading over a lumpy hull)."""
    idx = ((P - GRID) / H).T
    if sigma is None or sigma == 2.5:
        G = (GX, GY, GZ)
    else:
        if sigma not in GRADS: GRADS[sigma] = np.gradient(ndimage.gaussian_filter(fd['F'], sigma))
        G = GRADS[sigma]
    g = np.stack([ndimage.map_coordinates(A, idx, order=1, mode='nearest') for A in G], 1)
    return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)


def region_sdf(view, ids, close=0):
    """Signed distance image (metres, negative inside) of merged regions; top/front symmetric. close: extra
    morphological closing (px)."""
    key = (view, tuple(sorted(ids)), close)
    if key not in SDF:
        m = outlines.region_mask(view, ids, sym=view in ('top', 'front'))
        if close:
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))) > 0
        SDF[key] = p24.sdf2d(m) / 1000.0
    return SDF[key]


CAM28 = None


def h28_value(P, classes):
    """Membership in Honda rear frame 28 colour classes, via the calibrated camera (metres at depth)."""
    global CAM28
    if CAM28 is None:
        cams, w, h = p24.rig(); K, R, C = cams[28]; sc = 1920.0 / w
        K = K.copy(); K[:2] *= sc
        CAM28 = (K, R, C)
    key = ('h28', tuple(classes))
    if key not in SDF:
        m = np.zeros((1080, 1920), bool)
        for c in classes:
            m |= cv2.imread(str(p24.P24 / f'h28-{c}.png'), cv2.IMREAD_GRAYSCALE) > 127
        SDF[key] = p24.sdf2d(m)
    K, R, C = CAM28
    u, v, d = p24.project(K, R, C, P)
    return p24.sample(SDF[key], u, v, outside=500.0) * d / K[0, 0]


def term_value(P, view, ids, neg=False, close=0, off=0.0):
    """Signed membership of points P for one term (negative = inside)."""
    if view == 'h28':
        s = h28_value(P, ids); return -s if neg else s
    if view == 'box':                          # ids = [axis, lo, hi]; axis x|y|z|az (|z|)
        ax, lo, hi = ids
        c = np.abs(P[:, 2]) if ax == 'az' else P[:, 'xyz'.index(ax)]
        s = np.maximum(lo - c, c - hi); return -s if neg else s
    if view == 'disc':                         # ids = [x0, y0, r] in the side view
        s = np.hypot(P[:, 0] - ids[0], P[:, 1] - ids[1]) - ids[2]; return -s if neg else s
    if view == 'hp':                           # ids = [a, b, c]: side-view half-plane a*X + b*Y + c < 0
        s = ids[0] * P[:, 0] + ids[1] * P[:, 1] + ids[2]; return -s if neg else s
    u, v = p24.to_px(view, P[:, 0], P[:, 1], P[:, 2])   # side ignores Z (the left-side plate serves both sides)
    s = p24.sample(region_sdf(view, ids, close), u, v, outside=1.0) - off
    return -s if neg else s


FACE = {'side': None, 'top': np.array([0, 1, 0]), 'front': np.array([1, 0, 0]), 'rear': np.array([-1, 0, 0])}


def membership(spec, P, N, side):
    """Panel membership value at points (negative = in the panel)."""
    s = np.full(len(P), 1.0)
    for group in spec.get('any', [spec.get('terms', [])]):          # union of groups, each an intersection
        g = np.full(len(P), -1.0)
        for t in group:
            if t[0] == 'face':                                         # facing constraint inside a group
                d = {'front': np.array([1, 0, 0]), 'rear': np.array([-1, 0, 0]), 'top': np.array([0, 1, 0])}.get(t[1])
                fv = (N @ d) if d is not None else (side if side else np.sign(P[:, 2])) * N[:, 2]
                thr = float(t[2]) if len(t) > 2 else 0.15
                g = np.maximum(g, (fv - thr) * 0.02 if len(t) > 3 and t[3] == 'neg' else (thr - fv) * 0.02)
                continue
            g = np.maximum(g, term_value(P, t[0], t[1], len(t) > 2 and t[2] == 'neg', spec.get('close', 0), float(t[3]) if len(t) > 3 else 0.0)
                           if t[0] not in ('box', 'disc', 'hp') else term_value(P, t[0], t[1:], False))
        s = np.minimum(s, g)
    s = s - spec.get('grow', 0.0)
    if side:
        s = np.maximum(s, -side * P[:, 2] - 0.0)                      # keep this side of the centre line
    face = spec.get('face', 'none')
    if face == 'side':
        side_n = side if side else np.sign(P[:, 2])
        s = np.maximum(s, (0.15 - side_n * N[:, 2]) * 0.02)
    elif False:
        s = np.maximum(s, (0.15 - side * N[:, 2]) * 0.02)
    elif face in FACE and FACE[face] is not None:
        s = np.maximum(s, (0.15 - N @ FACE[face]) * 0.02)
    return s


def cut(V, F, s):
    """Keep the part of a triangle mesh where s < 0, splitting crossed triangles exactly.
    Returns (V2, F2, new-vertex mask)."""
    inside = s < 0
    cnt = inside[F].sum(1)
    keep = F[cnt == 3]
    mixed = F[(cnt > 0) & (cnt < 3)]
    newv = {}; extra = []
    def edge_point(a, b):
        k = (a, b) if a < b else (b, a)
        if k not in newv:
            t = s[a] / (s[a] - s[b])
            extra.append(V[a] + t * (V[b] - V[a])); newv[k] = len(V) + len(extra) - 1
        return newv[k]
    tris = []
    for f in mixed:
        poly = []
        for i in range(3):
            a, b = f[i], f[(i + 1) % 3]
            if inside[a]: poly.append(a)
            if inside[a] != inside[b]: poly.append(edge_point(a, b))
        for k in range(1, len(poly) - 1):
            tris.append((poly[0], poly[k], poly[k + 1]))
    V2 = np.vstack([V] + ([np.array(extra)] if extra else []))
    F2 = np.vstack([keep] + ([np.array(tris)] if tris else [])) if len(keep) or tris else np.zeros((0, 3), int)
    used = np.unique(F2)
    remap = -np.ones(len(V2), int); remap[used] = np.arange(len(used))
    isnew = np.zeros(len(V2), bool); isnew[len(V):] = True
    return V2[used], remap[F2], isnew[used]


def clip_all(V, F, S):
    """Exact priority partition of a triangle mesh. Takes vertices, faces and per-vertex values S (nv, K), one
    column per piece in priority order (negative = wants the surface; the last column must be negative
    everywhere). Every triangle is clipped by each column's linear interpolant in turn: the part where column k
    is negative goes to piece k, the rest stays for the later columns. The pieces tile the mesh exactly (no
    holes, however thin the band between two panels). Returns a list of (V_k, F_k) per column."""
    nf, K = len(F), S.shape[1]
    anyneg = np.zeros((nf, K), bool); allneg = np.zeros((nf, K), bool)
    for a in range(0, nf, 200000):                   # (chunk, 3, K) sign tests, chunked for the fine mesh
        neg = S[F[a:a + 200000]] < 0
        anyneg[a:a + 200000] = neg.any(1); allneg[a:a + 200000] = neg.all(1)
    del neg
    kf = anyneg.argmax(1)
    fast = allneg[np.arange(nf), kf]
    fast_tris = [[] for _ in range(K)]
    for k in np.unique(kf[fast]):
        fast_tris[k] = F[fast & (kf == k)]
    slow_tris = [[] for _ in range(K)]
    E = np.eye(3)
    for i in np.nonzero(~fast)[0]:
        f = F[i]; P3 = V[f]
        poly = [E[0], E[1], E[2]]
        for k in np.nonzero(anyneg[i])[0]:
            sv = S[f, k]
            vals = [w @ sv for w in poly]
            if all(v < 0 for v in vals):
                ins, poly = poly, []
            elif all(v >= 0 for v in vals):
                continue
            else:
                ins, outp = [], []
                n = len(poly)
                for j in range(n):
                    p, q, sp, sq = poly[j], poly[(j + 1) % n], vals[j], vals[(j + 1) % n]
                    (ins if sp < 0 else outp).append(p)
                    if (sp < 0) != (sq < 0):
                        r = p + (sp / (sp - sq)) * (q - p); ins.append(r); outp.append(r)
                poly = outp
            W = np.array(ins) @ P3
            for j in range(1, len(W) - 1):
                slow_tris[k].append((W[0], W[j], W[j + 1]))
            if len(poly) < 3: break
    out = []
    for k in range(K):
        parts = []
        if len(fast_tris[k]): parts.append(V[fast_tris[k]].reshape(-1, 3))
        if slow_tris[k]: parts.append(np.array(slow_tris[k]).reshape(-1, 3))
        if not parts:
            out.append((np.zeros((0, 3)), np.zeros((0, 3), int))); continue
        P = np.vstack(parts)
        key = np.round(P * 1e6).astype(np.int64)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        Vk = P[first]; Fk = inv.reshape(-1, 3)
        ok = (Fk[:, 0] != Fk[:, 1]) & (Fk[:, 1] != Fk[:, 2]) & (Fk[:, 0] != Fk[:, 2])
        a = np.linalg.norm(np.cross(Vk[Fk[:, 1]] - Vk[Fk[:, 0]], Vk[Fk[:, 2]] - Vk[Fk[:, 0]]), axis=1)
        Fk = Fk[ok & (a > 1e-11)]
        used = np.unique(Fk); remap = -np.ones(len(Vk), int); remap[used] = np.arange(len(used))
        out.append((Vk[used], remap[Fk]))
    return out


def catch_all(points, normals, joint_zone, claimed):
    """Paint the rest of the outer skin. Reserve dark strips at real panel
    openings, and leave the underside and inner wheelhouses dark."""
    x, y, z = points.T
    # The elevation drawing also depicts stamped creases, roof hardware and
    # parts seen through the glass. Only the closure outlines describe gaps.
    side_facing = np.sign(z) * normals[:, 2] > 0.35
    ink = cv2.dilate((p24.labels('side') == 0).astype(np.uint8), np.ones((2, 2), np.uint8))
    u, v = p24.to_px('side', x, y, z)
    stroke = p24.sample(p24.sdf2d(ink == 0), u, v) / 1000 + 0.0008
    # Clipping can leave a narrow gap even when every corner of a triangle
    # belongs to a neighbouring panel. Keep those gaps dark at the openings.
    seam = np.minimum(np.maximum(stroke, -claimed), joint_zone)
    paint = np.where(side_facing, seam, -0.01)
    paint = np.maximum(paint, (-0.5 - normals[:, 1]) * 0.02)
    for axle in p24.WHEEL_X:
        wheelhouse = 0.40 - np.hypot(x - axle, y - p24.WHEEL_Y)
        paint = np.maximum(paint, np.where(np.abs(z) < 0.9, wheelhouse, -1.0))
    return paint


DESPECKLE_CM2 = float(__import__('os').environ.get('CUT_DESPECKLE', '1.5'))   # 0 disables


def components(V, F):
    """Connected components of a triangle mesh (shared vertices). Returns per-face labels and per-label areas."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(V)
    A = coo_matrix((np.ones(3 * len(F)), (np.concatenate([F[:, 0], F[:, 1], F[:, 2]]), np.concatenate([F[:, 1], F[:, 2], F[:, 0]]))), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    fl = lab[F[:, 0]]
    ar = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    return fl, np.bincount(fl, ar, lab.max() + 1)


def despeckle(pieces, index, amin):
    """Small islands of the seams and the catch-all (area < amin m2, e.g. at triple junctions of strokes) are
    handed to the neighbouring panel that borders them most (vertices within 2 mm), so they do not show as dark
    or blue specks. Long shut-line strips are much larger and stay. Returns the new pieces list."""
    from scipy.spatial import cKDTree
    names = [e['name'] for e in index]
    targets = [k for k, n in enumerate(names) if n in ('Civic_Body_Seams', 'Civic_Body_Shell_Misc')]
    trees = {k: cKDTree(pieces[k][0]) for k in range(len(pieces)) if len(pieces[k][1])}
    add = {k: [] for k in range(len(pieces))}; moved = 0
    for t in targets:
        V, F = pieces[t]
        if len(F) == 0: continue
        fl, area = components(V, F)
        small = np.nonzero(area < amin)[0]
        keep = np.ones(len(F), bool)
        for c in small:
            fm = fl == c; vs = V[np.unique(F[fm])]
            best, bk = 0, None
            for k, tr in trees.items():
                if k == t: continue
                hit = int((tr.query(vs, distance_upper_bound=0.002)[0] < 0.002).sum())
                if hit > best: best, bk = hit, k
            if bk is None: continue
            add[bk].append(V[F[fm]]); keep[fm] = False; moved += 1
        pieces[t] = (V, F[keep])
    for k, tris in add.items():
        if not tris: continue
        V, F = pieces[k]; T = np.vstack(tris).reshape(-1, 3)
        pieces[k] = (np.vstack([V, T]), np.vstack([F, len(V) + np.arange(len(T)).reshape(-1, 3)]))
    print(f'despeckle: {moved} islands < {amin * 1e4:.1f} cm2 handed to neighbours')
    return pieces


def flat_normals(V, F, N):
    """Harmonic interpolation of the normals over a panel from its outline (spec key nflat): the panel reads as
    a smooth flush skin matching its surroundings at the shut line, without the hull's local bumps."""
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import spsolve
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    u, cnt = np.unique(E, axis=0, return_counts=True)
    bnd = np.zeros(len(V), bool); bnd[u[cnt == 1].ravel()] = True
    if bnd.all() or not bnd.any(): return N
    n = len(V)
    W = coo_matrix((np.ones(2 * len(u)), (np.concatenate([u[:, 0], u[:, 1]]), np.concatenate([u[:, 1], u[:, 0]]))), shape=(n, n)).tocsr()
    L = (diags(np.asarray(W.sum(1)).ravel()) - W).tocsr()
    I = np.nonzero(~bnd)[0]; B = np.nonzero(bnd)[0]
    A = L[I][:, I]; R = -L[I][:, B] @ N[B]
    out = N.copy(); out[I] = np.stack([spsolve(A.tocsc(), R[:, i]) for i in range(3)], 1)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def refine_detail_regions(vertices, faces, detailed):
    """Bisect edges near fine features once for both incident triangles.

    All children stay on the same source triangle. A triangle at the edge
    of a refined region may have just one or two of its edges bisected.
    """
    if not detailed.any():
        return vertices, faces
    edges = np.stack([faces, np.roll(faces, -1, axis=1)], axis=2)
    unique, inverse = np.unique(np.sort(edges, axis=2).reshape(-1, 2), axis=0, return_inverse=True)
    face_edges = inverse.reshape(-1, 3)
    split = np.zeros(len(unique), dtype=bool)
    split[face_edges[detailed[faces].any(axis=1)]] = True
    midpoints = np.full(len(unique), -1)
    midpoints[split] = np.arange(split.sum()) + len(vertices)
    mids = midpoints[face_edges]
    count = (mids >= 0).sum(axis=1)
    pieces = [faces[count == 0]]
    full_vertices, full_mids = faces[count == 3], mids[count == 3]
    # Rotate partially refined triangles to put the two split edges first
    # (or the single split edge first). Preserve the original winding.
    for edge in range(3):
        a, b, c = edge, (edge + 1) % 3, (edge + 2) % 3
        one = (count == 1) & (mids[:, a] >= 0)
        two = (count == 2) & (mids[:, c] < 0)
        v, m = faces[one], mids[one]
        pieces.extend([np.column_stack([v[:, a], m[:, a], v[:, c]]),
                       np.column_stack([m[:, a], v[:, b], v[:, c]])])
        v, m = faces[two], mids[two]
        pieces.extend([np.column_stack([v[:, a], m[:, a], v[:, c]]),
                       np.column_stack([m[:, a], v[:, b], m[:, b]]),
                       np.column_stack([m[:, a], m[:, b], v[:, c]])])
        pieces.append(np.column_stack([full_vertices[:, a], full_mids[:, a], full_mids[:, c]]))
    pieces.append(full_mids)
    return np.vstack([vertices, vertices[unique[split]].mean(axis=1)]), np.vstack(pieces)


def main():
    specs = [s for p in sys.argv[3].split(",") for s in json.load(open(p))]
    mesh = np.load(sys.argv[1])
    vertices, faces = mesh['V'].astype(float), mesh['F']
    normals = field_normals(vertices)
    detailed = np.zeros(len(vertices), dtype=bool)
    entries, index = [], []
    for spec in specs:
        names = spec.get('names') or [spec['name']]
        sides = [-1, +1] if len(names) == 2 else [spec.get('side', 0)]
        for name, side in zip(names, sides):
            entries.append((spec, side))
            if spec.get('mesh') == 'fine':
                detailed |= membership(spec, vertices, normals, side) < 0.014
            index.append({'name': name, 'mat': spec.get('mat', 'paint'),
                          'pivot': spec.get('pivot', '').replace('{S}', 'L' if side < 0 else 'R')})
    vertices, faces = refine_detail_regions(vertices, faces, detailed)
    normals = field_normals(vertices)
    cols = np.empty((len(vertices), len(entries) + 5), dtype=np.float32)
    bonnet = hood.signed_distance(vertices)
    hatch = tailgate.signed_distance(vertices)
    joint = np.full(len(vertices), -0.01)
    for k, (spec, side) in enumerate(entries):
        if spec.get('surface') == 'hood':
            cols[:, k] = bonnet + hood.HALF_GAP
        else:
            local = membership(spec, vertices, normals, side)
            moving_hood = index[k]['pivot'] == hood.PIVOT
            cols[:, k] = np.maximum(local, (bonnet if moving_hood else -bonnet) + hood.HALF_GAP)
        moving = index[k]['pivot'] == tailgate.PIVOT
        cols[:, k] = np.maximum(cols[:, k], (hatch if moving else -hatch) + tailgate.HALF_GAP)
        if spec.get('shut_line'):
            joint = np.maximum(joint, 0.025 - np.abs(cols[:, k]))
    # Every paint, glass and trim boundary must agree on which side of the
    # physical hem it occupies, including the strips between the drawing regions.
    cols[:, -5] = bonnet - hood.HALF_GAP
    index.append({'name': 'Civic_Hood_Shutline', 'mat': 'seam', 'pivot': ''})
    cols[:, -4] = hatch + tailgate.HALF_GAP
    # The preceding column claimed the inner half-space. Clip the other
    # side of the reveal independently even when a triangle spans both edges.
    cols[:, -3] = hatch - tailgate.HALF_GAP
    index.extend([{'name': 'Civic_Hatch_Surround', 'mat': 'paint', 'pivot': tailgate.PIVOT},
                  {'name': 'Civic_Hatch_Reveal', 'mat': 'seam', 'pivot': ''}])
    cols[:, -2] = catch_all(vertices, normals, joint, cols[:, :-2].min(axis=1))
    cols[:, -1] = -1
    index.extend([{'name': 'Civic_Body_Shell_Misc', 'mat': 'paint', 'pivot': ''},
                  {'name': 'Civic_Body_Seams', 'mat': 'seam', 'pivot': ''}])
    pieces = clip_all(vertices, faces, cols)
    if DESPECKLE_CM2 > 0: pieces = despeckle(pieces, index, DESPECKLE_CM2 * 1e-4)
    out, idx2 = {}, []
    nsm = [spec.get('nsmooth') for spec, _ in entries] + [3., 3., 3., None, None]
    nfl = [spec.get('nflat', False) for spec, _ in entries] + [False] * 5
    for e, (V, F), ns, nf in zip(index, pieces, nsm, nfl):
        nm = e['name']
        if len(F) == 0:
            print('EMPTY', nm); continue
        N = field_normals(V, ns)
        if nf: N = flat_normals(V, F, N)
        V, N = press_hood_ridges(V, N)
        out[nm + '/V'] = V.astype(np.float32); out[nm + '/F'] = F.astype(np.int32); out[nm + '/N'] = N.astype(np.float32)
        idx2.append(e)
        print(f'{nm:36s} {len(F):7d} faces')
    out['index'] = np.array(json.dumps(idx2))
    np.savez_compressed(sys.argv[4], **out)
    print('seams', len(pieces[-1][1]), 'faces; panels', len(idx2) - 1)


if __name__ == '__main__':
    main()
