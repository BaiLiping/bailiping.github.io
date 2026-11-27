"""Base body field: Honda 36-view visual hull intersected with the P24 side, top and front body outlines.

Run: venv/bin/python field.py <out.npz> [h=0.006]
Field value is an approximate signed distance in metres (negative inside). Each Honda view contributes the
image-space signed distance to its alpha mask, scaled by depth / focal length; the hull is the max over the
36 views. The P24 outlines are extruded across the car. The parts are joined by a smooth max (6 mm), then the
field is made left/right symmetric. Writes {F (float32, [nx, ny, nz]), x0, y0, z0, h}. Returns nothing.
"""
import sys, time
from multiprocessing import Pool
import numpy as np
import cv2
import p24

H = float(sys.argv[2]) if len(sys.argv) > 2 else 0.006
X = np.arange(-2.34, 2.34 + 1e-9, H)
Y = np.arange(0.06, 1.47 + 1e-9, H)
Z = np.arange(-0.98, 0.98 + 1e-9, H)
G = {}


def init():
    """Pool initialiser: loads the plates' and Honda masks' signed distance images once per worker."""
    G['views'] = {v: p24.sdf2d(p24.body_mask(v, arches=False)) / 1000.0 for v in ('side', 'top', 'front')}
    cams, w, h = p24.rig()
    G['cams'] = []
    for k, (K, R, C) in sorted(cams.items()):
        m = cv2.imread(str(p24.MASKS / f'{k:02d}.png'), cv2.IMREAD_GRAYSCALE) > 127
        G['cams'].append((K, R, C, p24.sdf2d(m)))


def smax(a, b, k=0.006):
    """Smooth maximum (intersection) with blend width k."""
    return 0.5 * (a + b + np.sqrt((a - b) ** 2 + k * k))


def slab(i0):
    """Takes a start X index; returns (i0, field block for 12 X planes)."""
    xs = X[i0:i0 + 12]
    PX, PY, PZ = np.meshgrid(xs, Y, Z, indexing='ij')
    P = np.stack([PX.ravel(), PY.ravel(), PZ.ravel()], 1)
    hull = np.full(len(P), -1.0)
    for K, R, C, sd in G['cams']:
        u, v, d = p24.project(K, R, C, P)
        hull = np.maximum(hull, p24.sample(sd, u, v, outside=500.0) * d / K[0, 0])
    f = np.full(len(P), -1.0)
    sides = [p24.sample(G['views'][view], *p24.to_px(view, P[:, 0], P[:, 1], P[:, 2]), outside=1.0)
             for view in ('side', 'top', 'front')]
    sharp = smax(smax(sides[0], sides[1]), sides[2])
    # nose and tail: a rounded intersection of the three outlines (bumper corners), no hull
    rnd = smax(smax(sides[0], sides[1], 0.035), sides[2], 0.035)
    w = np.clip(np.maximum((P[:, 0] - 1.62) / 0.15, (-P[:, 0] - 1.82) / 0.15), 0, 1)
    f = (1 - w) * sharp + w * rnd
    # wheel arches: the side-view disc, cut only outboard of the wheelhouse inner wall
    for wx, zw in zip(p24.WHEEL_X, p24.WHEELHOUSE_Z):
        r = np.hypot(P[:, 0] - wx, P[:, 1] - p24.WHEEL_Y)
        arch = smax(r - p24.ARCH_R, zw - np.abs(P[:, 2]), 0.004)
        f = smax(f, -arch, 0.004)
    return i0, f.reshape(PX.shape).astype(np.float32), hull.reshape(PX.shape).astype(np.float32)


def main():
    t = time.time()
    E = np.empty((len(X), len(Y), len(Z)), np.float32); Hh = np.empty_like(E)
    with Pool(20, initializer=init) as pool:
        for i0, blk, hb in pool.imap_unordered(slab, range(0, len(X), 12)):
            E[i0:i0 + blk.shape[0]] = blk; Hh[i0:i0 + hb.shape[0]] = hb
    E = 0.5 * (E + E[:, :, ::-1]); Hh = 0.5 * (Hh + Hh[:, :, ::-1])      # left/right symmetric
    # the hull wraps protruding details (splitter, fog pods, vents, wing endplates): smooth it hard at the
    # ends of the car, lightly elsewhere; the P24 outlines (E) stay sharp
    from scipy import ndimage
    # hull: tight on the greenhouse, smoothed hard on the lower body, unused at the nose and tail
    wy = np.clip((Y - 0.92) / 0.10, 0, 1)[None, :, None]
    Hs = wy * ndimage.gaussian_filter(Hh, 1.0) + (1 - wy) * ndimage.gaussian_filter(Hh, 0.03 / H)
    F = smax(Hs, E)
    np.savez_compressed(sys.argv[1], F=F.astype(np.float32), x0=X[0], y0=Y[0], z0=Z[0], h=H)
    print(f'field {F.shape} h={H} in {time.time() - t:.0f}s; inside voxels {(F < 0).sum()}')


if __name__ == '__main__':
    main()
