"""Opened-state geometry for the doors, hood, hatch: what a real car shows when a panel is open.

Run: venv/bin/python sides_openings.py <cut.npz> <body field.npz> <out.npz>
For every opening panel group (the cut pieces under one viewer pivot) the outline is found by projecting the
pieces along the group's opening axis a (doors: outboard, hood: up, hatch: rear-up) to a 2 mm mask (closed,
holes filled). From that outline and the designed body field (outer skin found by bisection along a):
  panel side (moves with the pivot): the shut face (the panel's edge return, D deep along -a) and the inner
      panel at depth D (doors: dark trim backing behind the door card below the window, black belt weatherstrip;
      hood: body-colour inner frame; hatch: dark trim panel around the glass);
  body side: the jamb around the opening (outline + 5 mm gap, Dj deep), a dark seal strip on it, black sill /
      cargo scuff plates on the bottom run, and the body B-pillar between each side's doors (it carries the
      seals and sits just inboard of the doors' inner panels).
Body pieces lying inside an outline (strokes and catch-all between the group's own panels) are moved to the
group as '<group>_Inlay_<mat>' meshes, so nothing floats in the opening. The trim is recessed along the local skin normal and then along the opening axis. Writes the meshes like parts3v.py (V/F/N per name + index with mat /
pivot) plus 'clear' (kept meshes inside each footprint must stay that deep inside the body, door-pivot meshes
are clamped into their door's outline; assemble.py fit_inside reads it) and 'cutout' ({cut piece: face indices
to drop}, applied by assemble.py).
"""
import sys, json
import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial import Delaunay
import outlines

RES = 0.002                 # mask resolution (m)
GAP = 0.005                 # shut-line gap between panel and body jamb
SKIN = 0.002                # extra inset along the opening axis, in metres
DOOR = dict(d_body=0.070, d_frame=0.035, dj_body=0.110, dj_frame=0.050)
HOOD = dict(d=0.035, dj=0.060)
HATCH = dict(d=0.040, dj=0.090)
SEAL = (0.012, 0.030, 0.003)   # seal depth range along the jamb and its stand-off toward the opening
TRIM = 'Mat_Interior_Black_Suede'   # cabin-side trim (the kept door cards' material)


class Field:
    """The designed body field with trilinear sampling (approximate signed distance, metres, < 0 inside)."""
    def __init__(self, path):
        d = np.load(path); self.F = d['F'].astype(np.float32); self.h = float(d['h'])
        self.o = np.array([float(d['x0']), float(d['y0']), float(d['z0'])])

    def __call__(self, P):
        g = ((np.asarray(P) - self.o) / self.h).T
        return ndimage.map_coordinates(self.F, g, order=1, mode='nearest')


def basis(a):
    a = np.asarray(a, float); a /= np.linalg.norm(a)
    t = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0, 0])
    e1 = t - a * (t @ a); e1 /= np.linalg.norm(e1); e2 = np.cross(a, e1)
    return a, e1, e2


def skin(field, a, e1, e2, uv, t_in, t_out=1.8):
    """Outer skin points along +a above plane coordinates uv (N,2): bisection between t_in (inside the body,
    scalar or per point) and t_out. Returns (N,3) points and a validity mask."""
    base = uv[:, :1] * e1 + uv[:, 1:2] * e2
    lo = np.broadcast_to(np.asarray(t_in, float), (len(uv),)).copy(); hi = np.full(len(uv), t_out)
    ok = (field(base + lo[:, None] * a) < 0) & (field(base + hi[:, None] * a) > 0)
    for _ in range(26):
        m = 0.5 * (lo + hi); ins = field(base + m[:, None] * a) < 0
        lo = np.where(ins, m, lo); hi = np.where(ins, hi, m)
    points = base + lo[:, None] * a
    # A lateral offset barely clears a horizontal roof edge. Recess the
    # structure along the surface normal too, including at grazing angles.
    step = np.eye(3) * field.h
    normal = np.column_stack([field(points + dx) - field(points - dx) for dx in step])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    return points - 0.006 * normal, ok


class Frame2D:
    """A raster frame over plane coordinates (u, v) at RES."""
    def __init__(self, uv, pad=0.06):
        self.lo = uv.min(0) - pad; self.shape = tuple((np.ceil((uv.max(0) + pad - self.lo) / RES)).astype(int)[::-1])

    def px(self, uv): return (uv - self.lo) / RES
    def uv(self, px): return np.asarray(px, float) * RES + self.lo

    def mask(self, V2, F):
        m = np.zeros(self.shape, np.uint8)
        P = np.round(self.px(V2)).astype(np.int32)
        cv2.fillPoly(m, [P[f].reshape(-1, 1, 2) for f in F], 1)
        return m > 0


def disk(r_m):
    r = max(1, int(round(r_m / RES))); return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def closed(m, r_close=0.008, fill=True):
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, disk(r_close))
    if fill: m = ndimage.binary_fill_holes(m).astype(np.uint8)
    return m > 0


def grow(m, r): return cv2.dilate(m.astype(np.uint8), disk(r)) > 0 if r > 0 else cv2.erode(m.astype(np.uint8), disk(-r)) > 0


def contour(fr, m, step=0.006):
    """Largest outer contour of a mask as a smoothed closed loop in (u, v), counter-clockwise, with outward
    2D normals. Returns (P (N,2), Nout (N,2))."""
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)[:, 0].astype(float)
    P = fr.uv(outlines.smooth_loop(c, sigma=3.0, step=step / RES))
    if 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]) < 0: P = P[::-1]
    t = np.roll(P, -1, 0) - np.roll(P, 1, 0); t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    return P, np.stack([t[:, 1], -t[:, 0]], 1)


def strip(outer, depth, a, n3, side_off=0.0):
    """Quad strip from a closed loop of skin points outer (N,3) running depth (N,) along -a; n3 (N,3) the
    strip normal. side_off moves the strip along n3. Returns V, F, N."""
    o = outer - a * SKIN + n3 * side_off; i = o - a * depth[:, None]
    n = len(o); V = np.vstack([o, i]); k = np.arange(n); k2 = (k + 1) % n
    F = np.concatenate([np.stack([k, k2, n + k2], 1), np.stack([k, n + k2, n + k], 1)])
    return V, F, np.vstack([n3, n3])


def band(outer, d0, d1, a, n3, side_off, sel):
    """A strip between depths d0 and d1 (arrays) on the loop points where sel (open runs kept)."""
    o = outer - a * (SKIN + d0)[:, None] + n3 * side_off; i = outer - a * (SKIN + d1)[:, None] + n3 * side_off
    n = len(o); k = np.arange(n); k2 = (k + 1) % n
    keep = sel & sel[k2]
    F = np.concatenate([np.stack([k, k2, n + k2], 1)[keep], np.stack([k, n + k2, n + k], 1)[keep]])
    return np.vstack([o, i]), F, np.vstack([n3, n3])


def cap(fr, m, a, e1, e2, field, t_in, depth_fn, normal, grid=0.025, t_out=1.8):
    """Surface over a mask region (constrained by its contour), placed depth_fn(uv) under the skin along -a.
    Returns V, F, N."""
    P, _ = contour(fr, m, step=0.008)
    lo, hi = P.min(0), P.max(0)
    G = np.stack(np.meshgrid(np.arange(lo[0], hi[0], grid), np.arange(lo[1], hi[1], grid), indexing='ij'), -1).reshape(-1, 2)
    px = np.round(fr.px(G)).astype(int)
    okg = (px[:, 0] >= 0) & (px[:, 1] >= 0) & (px[:, 0] < m.shape[1]) & (px[:, 1] < m.shape[0])
    G = G[okg]; px = px[okg]
    inner = grow(m, -0.006)[px[:, 1], px[:, 0]]
    UV = np.vstack([P, G[inner]])
    tri = Delaunay(UV).simplices
    c = UV[tri].mean(1); cp = np.round(fr.px(c)).astype(int)
    cp = np.clip(cp, 0, [m.shape[1] - 1, m.shape[0] - 1])
    tri = tri[m[cp[:, 1], cp[:, 0]]]
    S, ok = skin(field, a, e1, e2, UV, t_in(UV), t_out)
    V = S - a * (SKIN + depth_fn(UV))[:, None]
    tri = tri[ok[tri].all(1)]
    return V, tri, np.tile(normal, (len(V), 1))


def to3(n2, e1, e2): return n2[:, :1] * e1 + n2[:, 1:2] * e2


def inlays(d, index, fr, m, a, e1, e2, keep_fn, tag, pivot, out_cut, add):
    """Body pieces that lie inside an opening's outline (internal strokes and catch-all between the group's own
    panels) must move with the panel, or they float in the opening when it is open. Takes the group's frame /
    mask / axis, a face filter keep_fn(centroids, normals) and the group tag; records the faces to drop from
    the cut pieces (out_cut[name] += indices) and adds them, per material, as '<tag>_Inlay_<mat>' meshes under
    the group's pivot."""
    me = grow(m, -0.001)
    per_mat = {}
    for e in index:
        if e.get('pivot') or e['name'] not in ('Civic_Body_Seams', 'Civic_Body_Shell_Misc'): continue   # leftovers only
        n = e['name']; V, F, N = d[n + '/V'], d[n + '/F'], d[n + '/N']
        C = V[F].mean(1); Nf = N[F].mean(1)
        px = np.round(fr.px(np.stack([C @ e1, C @ e2], 1))).astype(int)
        ok = (px[:, 0] >= 0) & (px[:, 1] >= 0) & (px[:, 0] < me.shape[1]) & (px[:, 1] < me.shape[0])
        sel = np.zeros(len(F), bool)
        sel[ok] = me[px[ok, 1], px[ok, 0]]
        sel &= (Nf @ a > 0.2) & keep_fn(C, Nf)
        if not sel.any(): continue
        out_cut.setdefault(n, []).extend(np.nonzero(sel)[0].tolist())
        Fs = F[sel]; used = np.unique(Fs); remap = -np.ones(len(V), int); remap[used] = np.arange(len(used))
        L = per_mat.setdefault(e['mat'], [[], [], [], 0])
        L[0].append(V[used]); L[2].append(N[used]); L[1].append(remap[Fs] + L[3]); L[3] += len(used)
        print(f'   inlay {tag}: {sel.sum():5d} faces of {n}')
    for mat, (Vs, Fs, Ns, _) in per_mat.items():
        add(f'{tag}_Inlay_{mat}', mat, pivot, (np.vstack(Vs), np.vstack(Fs), np.vstack(Ns)))


def group(d, index, pivot, exclude=()):
    Vs, Fs, off = [], [], 0
    for e in index:
        if e.get('pivot') != pivot or any(x in e['name'] for x in exclude): continue
        V, F = d[e['name'] + '/V'], d[e['name'] + '/F']
        Vs.append(V); Fs.append(F + off); off += len(V)
    return (np.vstack(Vs), np.vstack(Fs)) if Vs else (None, None)


def main():
    d = np.load(sys.argv[1]); index = json.loads(str(d['index'])); field = Field(sys.argv[2])
    out, idx, clear, cutout = {}, [], [], {}
    def add(name, mat, pivot, VFN):
        V, F, N = VFN
        if F is None or len(F) == 0: print('EMPTY', name); return
        N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
        out[name + '/V'] = V.astype(np.float32); out[name + '/F'] = np.asarray(F, np.int32); out[name + '/N'] = N.astype(np.float32)
        idx.append({'name': name, 'mat': mat, 'pivot': pivot}); print(f'{name:40s} {len(F):6d} faces  {pivot}')

    # ---------------- doors (per side: front + rear) ----------------
    for sfx, s in (('L', -1.0), ('R', 1.0)):
        a = np.array([0, 0, s]); e1 = np.array([1.0, 0, 0]); e2 = np.array([0, 1.0, 0])   # plane coords (x, y)
        groups = {}
        for door in ('', 'R'):
            V, F = group(d, index, 'Pivot_Door_' + door + sfx)
            groups[door] = (V, F)
        allV = np.vstack([groups[''][0], groups['R'][0]])
        fr = Frame2D(allV[:, :2])
        masks, belts = {}, {}
        for door, (V, F) in groups.items():
            masks[door] = closed(fr.mask(V[:, :2], F))
            Vg, Fg = group(d, index, 'Pivot_Door_' + door + sfx, exclude=('Shell', 'Frame', 'Sash', 'Sail', 'Pocket'))
            gm = fr.mask(Vg[:, :2], Fg)                                  # the door glass
            cols = np.nonzero(gm.any(0))[0]
            bottom = np.array([np.nonzero(gm[:, c])[0].min() for c in cols])   # lowest glass row (rows run up in y)
            xb = fr.uv(np.stack([cols, np.zeros_like(cols)], 1))[:, 0]
            yb = fr.uv(np.stack([np.zeros_like(bottom), bottom], 1))[:, 1]
            belts[door] = (xb, ndimage.uniform_filter1d(yb, 15))
        def belt_y(x):
            xs = np.concatenate([belts[''][0], belts['R'][0]]); ys = np.concatenate([belts[''][1], belts['R'][1]])
            o = np.argsort(xs); return np.interp(x, xs[o], ys[o])
        t_in = lambda UV: np.full(len(UV), 0.30)
        ss = lambda t: (lambda c: c * c * (3 - 2 * c))(np.clip(t, 0, 1))
        frame_w = lambda P: ss((P[:, 1] - belt_y(P[:, 0]) + 0.02) / 0.04)    # 0 below the belt, 1 in the frame
        def front_pillar_weight(P):
            # The jamb turns behind the narrow A-pillar here. Extending the
            # full lower-door return inward shows through the windshield.
            return ss((P[:, 0] - 0.18) / 0.14) * ss((P[:, 1] - 0.92) / 0.06)

        for door, piv in (('', 'Pivot_Door_' + sfx), ('R', 'Pivot_Door_R' + sfx)):
            m = masks[door]; tag = door + sfx
            P, N2 = contour(fr, grow(m, -0.001))
            S, ok = skin(field, a, e1, e2, P, t_in(P))
            w = frame_w(P); D = DOOR['d_body'] * (1 - w) + DOOR['d_frame'] * w
            D += front_pillar_weight(P) * (0.022 - D)
            V, F, N = strip(S, D, a, to3(N2, e1, e2)); keep = ok & np.roll(ok, -1)
            F = F[np.concatenate([keep, keep])]
            add(f'Civic_Door_Shell_Jamb_{tag}', 'paint', piv, (V, F, N))
            # inner panel below the window (dark trim like the door card, so the cabin reads dark through the
            # glass), just outboard of the door card
            low = m.copy(); rows = np.arange(m.shape[0])
            ub = belt_y(fr.uv(np.stack([np.arange(m.shape[1]), np.zeros(m.shape[1])], 1))[:, 0])
            vb = np.round((ub - fr.lo[1]) / RES).astype(int)
            low &= rows[:, None] < vb[None, :] - int(0.005 / RES)
            low = grow(low, 0.003) & grow(m, 0.002)
            add(f'Civic_DoorCard_Backing_{tag}', TRIM, piv,
                cap(fr, low, a, e1, e2, field, t_in, lambda UV: np.full(len(UV), DOOR['d_body']), -a))
            # belt weatherstrip: closes the door top below the glass, from the skin to the inner frame
            xs = np.nonzero(m.any(0))[0]
            xb = fr.uv(np.stack([xs, np.zeros_like(xs)], 1))[:, 0]
            gx = belts[door][0]; xb = xb[(xb > gx.min() + 0.01) & (xb < gx.max() - 0.01)][::3]
            Pb = np.stack([xb, belt_y(xb) - 0.006], 1)
            Sb, okb = skin(field, a, e1, e2, Pb, t_in(Pb))
            n = len(Pb); o = Sb - a * SKIN; i = Sb - a * (SKIN + DOOR['d_body'])
            k = np.arange(n - 1); keep = okb[:-1] & okb[1:]
            Fb = np.concatenate([np.stack([k, k + 1, n + k + 1], 1)[keep], np.stack([k, n + k + 1, n + k], 1)[keep]])
            add(f'Civic_Door_Weatherstrip_Belt_{tag}', 'black', piv, (np.vstack([o, i]), Fb, np.tile([0, 1.0, 0], (2 * n, 1))))
            # kept door cards stay outboard-clear of the inner frame, and inside the door's outline
            Pc, _ = contour(fr, m)
            clear.append({'a': a.tolist(), 'e1': e1.tolist(), 'e2': e2.tolist(), 'poly': Pc[::4].tolist(), 'margin': DOOR['d_body'] + 0.006,
                          'pivot': piv, 'clamp': 0.012})
            inlays(d, index, fr, m, a, e1, e2, lambda C, N: np.sign(C[:, 2]) == s, f'Civic_Door_Shell_{tag}', piv, cutout, add)
        # body side: one opening for both doors, B-pillar between them
        mu = closed(masks[''] | masks['R'], r_close=0.010)
        P, N2 = contour(fr, grow(mu, GAP))
        S, ok = skin(field, a, e1, e2, P, t_in(P))
        w = frame_w(P); Dj = DOOR['dj_body'] * (1 - w) + DOOR['dj_frame'] * w
        Dj += front_pillar_weight(P) * (0.032 - Dj)
        n3 = -to3(N2, e1, e2)                                             # faces into the opening
        V, F, N = strip(S, Dj, a, n3); keep = ok & np.roll(ok, -1)
        add(f'Civic_Body_Shell_Door_Jamb_{sfx}', 'paint', '', (V, F[np.concatenate([keep, keep])], N))
        add(f'Civic_Door_Weatherstrip_Body_{sfx}', 'black', '',
            band(S, np.full(len(P), SEAL[0]), np.full(len(P), SEAL[1]), a, n3, SEAL[2], ok))
        sill = ok & (P[:, 1] < 0.45) & (np.abs(N2[:, 1]) > 0.7)
        add(f'Civic_Body_Shell_Sill_Plate_{sfx}', 'black', '',
            band(S, np.full(len(P), 0.020), np.full(len(P), 0.100), a, n3, 0.002, sill))
        # B-pillar: behind the doors' shared shut line
        mf, mr = masks[''], masks['R']
        rows = np.nonzero(mf.any(1) & mr.any(1))[0]
        xs_f = np.array([np.nonzero(mf[r])[0].min() for r in rows]); xs_r = np.array([np.nonzero(mr[r])[0].max() for r in rows])
        xsh = fr.uv(np.stack([0.5 * (xs_f + xs_r), rows], 1)); xsh = xsh[np.argsort(xsh[:, 1])]
        yb = np.linspace(xsh[:, 1].min() + 0.01, xsh[:, 1].max() - 0.01, 60)
        xb = np.interp(yb, xsh[:, 1], ndimage.uniform_filter1d(xsh[:, 0], 9))
        us = np.array([0.025, 0.0, -0.05])
        G = np.stack([np.repeat(xb, 3) + np.tile(us, len(yb)), np.repeat(yb, 3)], 1)
        Sg, okg = skin(field, a, e1, e2, G, t_in(G))
        face = Sg - a * 0.075; back = Sg - a * 0.120
        nb = len(G); Vp, Fq, Np = [face], [], [np.tile(a, (nb, 1))]
        for r in range(len(yb) - 1):
            for c in range(2):
                i0 = 3 * r + c; q = [i0, i0 + 1, i0 + 4, i0 + 3]
                if okg[q].all(): Fq += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
        for c, nx in ((0, 1.0), (2, -1.0)):                               # front / rear side faces
            ic = np.arange(len(yb)) * 3 + c; base = sum(len(v) for v in Vp)
            Vp += [face[ic], back[ic]]; Np += [np.tile([nx, 0, 0], (2 * len(ic), 1))]
            n = len(ic)
            for r in range(n - 1):
                if okg[ic[r]] and okg[ic[r + 1]]:
                    Fq += [[base + r, base + r + 1, base + n + r + 1], [base + r, base + n + r + 1, base + n + r]]
        add(f'Civic_Body_Shell_BPillar_{sfx}', 'paint', '', (np.vstack(Vp), np.array(Fq), np.vstack(Np)))
        add(f'Civic_BPillar_Trim_{sfx}', TRIM, '', (back, np.array([[i for i in f] for f in Fq if max(f) < nb]), np.tile(-a, (nb, 1))))
        # B-pillar seals: along its front and rear edges on the outer face
        seal = []
        for c, du in ((0, -0.012), (2, 0.012)):
            idx_c = np.arange(len(yb)) * 3 + c
            o = Sg[idx_c] - a * 0.073; i = o + np.array([du, 0, 0])
            seal.append((o, i))
        Vs = np.vstack([p for o, i in seal for p in (o, i)]); Fs = []; base = 0
        for o, i in seal:
            n = len(o)
            for r in range(n - 1):
                if okg[3 * r] and okg[3 * r + 3]: Fs += [[base + r, base + r + 1, base + n + r + 1], [base + r, base + n + r + 1, base + n + r]]
            base += 2 * n
        add(f'Civic_Door_Weatherstrip_BPillar_{sfx}', 'black', '', (Vs, np.array(Fs), np.tile(a, (len(Vs), 1))))
        Pb2 = np.stack([np.concatenate([xb + 0.035, (xb - 0.06)[::-1]]), np.concatenate([yb, yb[::-1]])], 1)
        clear.append({'a': a.tolist(), 'e1': e1.tolist(), 'e2': e2.tolist(), 'poly': Pb2.tolist(), 'margin': 0.125})

    # ---------------- hood ----------------
    V, F = group(d, index, 'Pivot_Hood_Assembly')
    if V is not None:
        a = np.array([0, 1.0, 0]); e1 = np.array([1.0, 0, 0]); e2 = np.array([0, 0, -1.0])   # (x, -z), right-handed with a
        uvf = lambda P3: np.stack([P3 @ e1, P3 @ e2], 1)
        fr = Frame2D(uvf(V)); m = closed(fr.mask(uvf(V), F), r_close=0.010)
        t_in = lambda UV: np.full(len(UV), 0.45)
        P, N2 = contour(fr, grow(m, -0.001))
        S, ok = skin(field, a, e1, e2, P, t_in(P))
        Vs, Fs, Ns = strip(S, np.full(len(P), HOOD['d']), a, to3(N2, e1, e2)); keep = ok & np.roll(ok, -1)
        add('Civic_Hood_Jamb', 'paint', 'Pivot_Hood_Assembly', (Vs, Fs[np.concatenate([keep, keep])], Ns))
        add('Civic_Hood_Inner_Frame', 'paint', 'Pivot_Hood_Assembly',
            cap(fr, grow(m, 0.002), a, e1, e2, field, t_in, lambda UV: np.full(len(UV), HOOD['d']), -a))
        P, N2 = contour(fr, grow(m, GAP))
        S, ok = skin(field, a, e1, e2, P, t_in(P)); n3 = -to3(N2, e1, e2)
        Vs, Fs, Ns = strip(S, np.full(len(P), HOOD['dj']), a, n3); keep = ok & np.roll(ok, -1)
        add('Civic_Body_Shell_Hood_Jamb', 'paint', '', (Vs, Fs[np.concatenate([keep, keep])], Ns))
        rear = ok & (P[:, 0] < P[:, 0].min() + 0.12)                     # cowl seal along the rear edge
        add('Civic_Hood_Weatherstrip_Cowl', 'black', '', band(S, np.full(len(P), 0.004), np.full(len(P), 0.018), a, n3, 0.002, rear))
        Pc, _ = contour(fr, m)
        clear.append({'a': a.tolist(), 'e1': e1.tolist(), 'e2': e2.tolist(), 'poly': Pc[::4].tolist(), 'margin': HOOD['d'] + 0.008})
        inlays(d, index, fr, m, a, e1, e2, lambda C, N: C[:, 1] > 0.6, 'Civic_Hood', 'Pivot_Hood_Assembly', cutout, add)

    # ---------------- hatch ----------------
    V, F = group(d, index, 'Pivot_Hatch_Assembly')
    if V is not None:
        a = np.array([-1.0, 0.85, 0]); a /= np.linalg.norm(a)
        e1 = np.array([0, 0, 1.0]); e2 = np.cross(a, e1)
        uvf = lambda P3: np.stack([P3 @ e1, P3 @ e2], 1)
        fr = Frame2D(uvf(V)); m = closed(fr.mask(uvf(V), F), r_close=0.010)
        c_in = np.array([-1.55, 0.80, 0.0])
        t_in = lambda UV: np.full(len(UV), c_in @ a)
        P, N2 = contour(fr, grow(m, -0.001))
        S, ok = skin(field, a, e1, e2, P, t_in(P), t_out=4.0)
        Vs, Fs, Ns = strip(S, np.full(len(P), HATCH['d']), a, to3(N2, e1, e2)); keep = ok & np.roll(ok, -1)
        add('Civic_Hatch_Jamb', 'paint', 'Pivot_Hatch_Assembly', (Vs, Fs[np.concatenate([keep, keep])], Ns))
        Vg, Fg = group(d, index, 'Pivot_Hatch_Assembly', exclude=('Panel', 'Deck'))
        gm = grow(closed(fr.mask(uvf(Vg), Fg), r_close=0.006), 0.010)
        trim = grow(m, 0.002) & ~gm
        add('Civic_Hatch_Inner_Trim', 'black', 'Pivot_Hatch_Assembly',
            cap(fr, trim, a, e1, e2, field, lambda UV: np.full(len(UV), c_in @ a), lambda UV: np.full(len(UV), HATCH['d']), -a, t_out=4.0))
        P, N2 = contour(fr, grow(m, GAP))
        S, ok = skin(field, a, e1, e2, P, t_in(P), t_out=4.0); n3 = -to3(N2, e1, e2)
        Vs, Fs, Ns = strip(S, np.full(len(P), HATCH['dj']), a, n3); keep = ok & np.roll(ok, -1)
        add('Civic_Body_Shell_Cargo_Jamb', 'paint', '', (Vs, Fs[np.concatenate([keep, keep])], Ns))
        add('Civic_Hatch_Weatherstrip_Body', 'black', '', band(S, np.full(len(P), SEAL[0]), np.full(len(P), SEAL[1]), a, n3, SEAL[2], ok))
        low = ok & (S[:, 1] < S[ok, 1].min() + 0.06)
        add('Civic_Body_Shell_Cargo_Sill_Plate', 'black', '', band(S, np.full(len(P), 0.015), np.full(len(P), 0.085), a, n3, 0.002, low))
        Pc, _ = contour(fr, m)
        clear.append({'a': a.tolist(), 'e1': e1.tolist(), 'e2': e2.tolist(), 'poly': Pc[::4].tolist(), 'margin': HATCH['d'] + 0.010})
        inlays(d, index, fr, m, a, e1, e2, lambda C, N: C[:, 0] < -1.0, 'Civic_Hatch', 'Pivot_Hatch_Assembly', cutout, add)

    out['index'] = np.array(json.dumps(idx)); out['clear'] = np.array(json.dumps(clear))
    out['cutout'] = np.array(json.dumps({k: sorted(set(v)) for k, v in cutout.items()}))
    np.savez_compressed(sys.argv[3], **out)
    print('openings', len(idx), 'meshes;', len(clear), 'clearance footprints')


if __name__ == '__main__':
    main()
