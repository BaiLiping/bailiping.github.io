"""Small exterior parts as the rounded intersection of their outlines in Honda's three P24 views.

Run: venv/bin/python parts3v.py <specs.json> <body field.npz> <out.npz>
Spec: {name, mat, pivot?, side|top|front: region ids (top/front ids on the passenger half: top-plate upper
half, front-plate left half), profile?: extra side-view polygon [[x, y], ...] used instead of side ids,
yband?: [y0, y1] slab, zband?: [z0, z1] slab on the passenger side, follow?: offset in metres to hug the body
surface (door handles), k?: rounding (m), h?: grid (m), mirror?: also build the driver-side copy,
names?: [right name, left name]}.
Each given view contributes its outline extruded across the car; the part is their smooth intersection,
meshed by marching cubes on an h grid inside the outlines' box, with normals from the field gradient.
Built on the passenger side (+Z) and mirrored. Writes per-part V/F/N plus an index (like cutpanels.py).
"""
import sys, json
import numpy as np
import cv2
from scipy import ndimage
from skimage import measure
import p24, outlines

body = np.load(sys.argv[2])
BF = ndimage.gaussian_filter(body['F'], 1.0); BG = np.array([float(body['x0']), float(body['y0']), float(body['z0'])]); BH = float(body['h'])


def body_field(P):
    return ndimage.map_coordinates(BF, ((P - BG) / BH).T, order=1, mode='nearest')


def mask_sdf(view, ids, profile=None):
    """Signed distance (m) image of the merged regions; side view may use a polygon profile instead."""
    if profile is not None and view == 'front':           # polygon [[z, y], ...] on the passenger side
        m = np.zeros((1700, 2300), np.uint8)
        pts = np.array([[(1.15 - z) * 1000, (1.65 - y) * 1000] for z, y in profile], np.int32)
        cv2.fillPoly(m, [pts], 1); m = m > 0
    elif profile is not None:
        m = np.zeros((1700, 5000), np.uint8)
        pts = np.array([[(2.5 - x) * 1000, (1.65 - y) * 1000] for x, y in profile], np.int32)
        cv2.fillPoly(m, [pts], 1); m = m > 0
    else:
        m = outlines.region_mask(view, ids, sym=False)
    return p24.sdf2d(m) / 1000.0, m


def smax(a, b, k):
    return 0.5 * (a + b + np.sqrt((a - b) ** 2 + k * k))


def build(spec):
    k = spec.get('k', 0.006); h = spec.get('h', 0.003)
    terms, boxes = [], []
    for view in ('side', 'top', 'front'):
        prof = spec.get('profile') if view == 'side' else spec.get('front_profile') if view == 'front' else None
        if view in spec or prof is not None:
            sd, m = mask_sdf(view, spec.get(view, []), prof)
            terms.append((view, sd))
            vv, uu = np.nonzero(m)
            if view == 'side': boxes.append(('x', 2.5 - uu.max() / 1000, 2.5 - uu.min() / 1000)); boxes.append(('y', 1.65 - vv.max() / 1000, 1.65 - vv.min() / 1000))
            if view == 'top': boxes.append(('x', 2.5 - uu.max() / 1000, 2.5 - uu.min() / 1000)); boxes.append(('z', 1.15 - vv.max() / 1000, 1.15 - vv.min() / 1000))
            if view == 'front': boxes.append(('z', 1.15 - uu.max() / 1000, 1.15 - uu.min() / 1000)); boxes.append(('y', 1.65 - vv.max() / 1000, 1.65 - vv.min() / 1000))
    lo = {a: max(v[1] for v in boxes if v[0] == a) for a in 'xyz' if any(v[0] == a for v in boxes)}
    lo = {a: min(v[1] for v in boxes if v[0] == a) for a in lo}
    hi = {a: max(v[2] for v in boxes if v[0] == a) for a in lo}
    if 'yband' in spec: lo['y'], hi['y'] = spec['yband']
    if 'zband' in spec: lo['z'], hi['z'] = spec['zband']
    if 'z' not in lo: lo['z'], hi['z'] = 0.0, 1.05
    pad = 3 * h + k
    ax = {a: np.arange(lo[a] - pad, hi[a] + pad, h) for a in 'xyz'}
    PX, PY, PZ = np.meshgrid(ax['x'], ax['y'], ax['z'], indexing='ij')
    P = np.stack([PX.ravel(), PY.ravel(), PZ.ravel()], 1)
    f = np.full(len(P), -1.0)
    for view, sd in terms:
        u, v = p24.to_px(view, P[:, 0], P[:, 1], P[:, 2])
        f = smax(f, p24.sample(sd, u, v, outside=1.0), k)
    if 'yband' in spec: f = smax(f, np.maximum(spec['yband'][0] - P[:, 1], P[:, 1] - spec['yband'][1]), k)
    if 'zband' in spec: f = smax(f, np.maximum(spec['zband'][0] - P[:, 2], P[:, 2] - spec['zband'][1]), k)
    if 'follow' in spec:              # a skin following the body surface, `follow` thick, outboard of it
        b = body_field(P)
        f = smax(f, np.maximum(-b - 0.002, b - spec['follow']), k * 0.5)
    F = f.reshape(PX.shape)
    if F.min() >= 0: raise ValueError('empty part ' + str(spec.get('name', spec.get('names'))))
    Fs = ndimage.gaussian_filter(F, 0.6)
    V, Fc, _, _ = measure.marching_cubes(Fs, 0.0)
    g = np.stack(np.gradient(Fs), 0)
    N = np.stack([ndimage.map_coordinates(g[i], V.T, order=1) for i in range(3)], 1)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    P3 = np.array([lo['x'] - pad, lo['y'] - pad, lo['z'] - pad]) + V * h
    return P3, Fc, N


def main():
    specs = [s for p in sys.argv[1].split(",") for s in json.load(open(p))]; out = {}; index = []
    for s in specs:
        V, F, N = build(s)
        names = s['names'] if 'names' in s else [s['name']]
        out[names[0] + '/V'] = V.astype(np.float32); out[names[0] + '/F'] = F.astype(np.int32); out[names[0] + '/N'] = N.astype(np.float32)
        index.append({'name': names[0], 'mat': s.get('mat', 'paint'), 'pivot': s.get('pivot', '').replace('{S}', 'R')})
        if s.get('mirror', True) and len(names) > 1:
            V2 = V * [1, 1, -1]; N2 = N * [1, 1, -1]
            out[names[1] + '/V'] = V2.astype(np.float32); out[names[1] + '/F'] = F[:, ::-1].astype(np.int32); out[names[1] + '/N'] = N2.astype(np.float32)
            index.append({'name': names[1], 'mat': s.get('mat', 'paint'), 'pivot': s.get('pivot', '').replace('{S}', 'L')})
        print(f"{names[0]:34s} {len(F):6d} faces  box x[{V[:,0].min():.2f},{V[:,0].max():.2f}] y[{V[:,1].min():.2f},{V[:,1].max():.2f}] z[{V[:,2].min():.2f},{V[:,2].max():.2f}]")
    out['index'] = np.array(json.dumps(index))
    np.savez_compressed(sys.argv[3], **out)


if __name__ == '__main__':
    main()
