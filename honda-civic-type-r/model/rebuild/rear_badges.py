"""Rear badges (CIVIC script, Type R script, red H) and the licence plate as thin textured plates following the
hatch surface, plus the plate's black frame (untextured).

Run: venv/bin/python rear_badges.py <body field.npz> <out.npz>
Writes the parts format (<name>/V, /F, /N, plus /UV float32 N x 2 planar UVs in the glTF convention: u across the
texture, v down from its top) and an index with the base GLB's textured materials (alpha-blended images), on the
hatch pivot. Each plate is a 2.5 mm grid over its (z, y) rectangle, placed on the body's rear face (first surface
crossing along +x from behind), lifted 1 mm along the normal.

Plate rectangles are sized so the opaque part of each texture (alpha bounding box: CIVIC u 0.158..0.838 v 0.246..0.731,
Type R u 0.127..0.760 v 0.250..0.850, H u 0.047..0.955 v 0.109..0.893) covers the lettering measured in Honda's rear
frame 28 resampled on the plane x = -2.12: CIVIC z -0.564..-0.421 y 0.919..0.941 (driver side), Type R z 0.422..0.542
y 0.920..0.949, H z -0.053..0.049 y 0.910..0.991.
"""
import sys, json
import numpy as np
from rear_chart import Field

BADGES = [  # name, material, z0, z1, y0, y1 (texture rectangle; u grows with z, i.e. left to right seen from behind)
    ('Civic_Badge_Rear_Civic', 'Mat_Civic_Script', -0.5975, -0.3875, 0.9090, 0.9510),
    ('Civic_Badge_Rear_TypeR', 'Mat_Type_R_Script', 0.3977, 0.5877, 0.9130, 0.9610),
    ('Civic_Badge_Rear_H', 'Mat_Badge_Honda_Red', -0.0560, 0.0560, 0.8990, 1.0020),
]
# the plates only cover the opaque part of each texture (+ 0.03 margin): a transparent margin would otherwise bend
# round the lamp corner. Texture-space crop (u0, u1, v0, v1) per badge
CROP = {'Civic_Badge_Rear_Civic': (0.128, 0.868, 0.216, 0.761), 'Civic_Badge_Rear_TypeR': (0.097, 0.790, 0.220, 0.880),
        'Civic_Badge_Rear_H': (0.0, 1.0, 0.0, 1.0)}
STEP, LIFT = 0.0025, 0.0010           # grid step, lift off the body (badges)


def rear_face(fld, Y, Z):
    """x of the first surface crossing along +x from behind, for points (Y, Z) (1D arrays)."""
    xs = np.arange(-2.40, -1.80, 0.001)
    P = np.stack([np.repeat(xs[None], len(Y), 0).ravel(), np.repeat(Y, len(xs)), np.repeat(Z, len(xs))], 1)
    f = fld(P).reshape(len(Y), len(xs))
    k = np.argmax(f < 0, 1); k0 = np.maximum(k - 1, 0)
    f0 = f[np.arange(len(Y)), k0]; f1 = f[np.arange(len(Y)), k]
    t = np.clip(f0 / np.maximum(f0 - f1, 1e-9), 0, 1)
    return xs[k0] + t * 0.001


def plate(fld, z0, z1, y0, y1):
    nz = int(np.ceil((z1 - z0) / STEP)) + 1; ny = int(np.ceil((y1 - y0) / STEP)) + 1
    zs = np.linspace(z0, z1, nz); ys = np.linspace(y0, y1, ny)
    Z, Y = np.meshgrid(zs, ys)
    X = rear_face(fld, Y.ravel(), Z.ravel())
    P = np.stack([X, Y.ravel(), Z.ravel()], 1)
    for _ in range(3):
        n = fld.grad(P); P = P - fld(P)[:, None] * n
    N = fld.grad(P); P = P + N * LIFT
    q = np.arange(nz * ny).reshape(ny, nz)
    a, b, c, d = q[:-1, :-1].ravel(), q[:-1, 1:].ravel(), q[1:, :-1].ravel(), q[1:, 1:].ravel()
    F = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])
    UV = np.stack([(Z.ravel() - z0) / (z1 - z0), (y1 - Y.ravel()) / (y1 - y0)], 1)
    return P, F, N, UV


# licence plate in the hatch recess (Honda frame 28 on x = -2.14: z -0.150..0.150, y 0.716..0.862), the site's
# display plate graphic (Mat_American_Honda_Display_Graphic, 1024 x 512 = the plate's 2:1) on a black frame
PLATE = (-0.150, 0.150, 0.716, 0.862)
PLATE_LIFT, FRAME_W, FRAME_LIFT = 0.0045, 0.009, 0.0035


def ring(fld, z0, z1, y0, y1, w, lift):
    """Flat ring (frame) of width w around the rectangle, on the surface, lifted. Returns V, F, N."""
    V, F, N, UV = plate(fld, z0 - w, z1 + w, y0 - w, y1 + w)
    zz = UV[:, 0] * (z1 - z0 + 2 * w) + z0 - w; yy = y1 + w - UV[:, 1] * (y1 - y0 + 2 * w)
    inner = (zz > z0 + 0.002) & (zz < z1 - 0.002) & (yy > y0 + 0.002) & (yy < y1 - 0.002)
    keep = ~inner[F].all(1)
    F = F[keep]; used = np.unique(F); remap = -np.ones(len(V), int); remap[used] = np.arange(len(used))
    return V[used] + N[used] * (lift - LIFT), remap[F], N[used]


def main():
    fld = Field(sys.argv[1], sigma=1.0)
    out, index = {}, []
    BAD = list(BADGES) + [('Civic_Rear_LicensePlate', 'Mat_American_Honda_Display_Graphic') + PLATE]
    for name, mat, z0, z1, y0, y1 in BAD:
        cu0, cu1, cv0, cv1 = CROP.get(name, (0.0, 1.0, 0.0, 1.0))
        V, F, N, UV = plate(fld, z0 + cu0 * (z1 - z0), z0 + cu1 * (z1 - z0), y1 - cv1 * (y1 - y0), y1 - cv0 * (y1 - y0))
        UV = np.stack([cu0 + UV[:, 0] * (cu1 - cu0), cv0 + UV[:, 1] * (cv1 - cv0)], 1)
        if 'LicensePlate' in name: V = V + N * (PLATE_LIFT - LIFT)
        out[name + '/V'] = V.astype(np.float32); out[name + '/F'] = F.astype(np.int32)
        out[name + '/N'] = N.astype(np.float32); out[name + '/UV'] = UV.astype(np.float32)
        index.append({'name': name, 'mat': mat, 'pivot': 'Pivot_Hatch_Assembly'})
        print(f'{name:28s} {len(F):5d} faces  x[{V[:,0].min():.3f},{V[:,0].max():.3f}]')
    V, F, N = ring(fld, *PLATE, FRAME_W, FRAME_LIFT)
    out['Civic_Rear_LicensePlate_Frame/V'] = V.astype(np.float32); out['Civic_Rear_LicensePlate_Frame/F'] = F.astype(np.int32)
    out['Civic_Rear_LicensePlate_Frame/N'] = N.astype(np.float32)
    index.append({'name': 'Civic_Rear_LicensePlate_Frame', 'mat': 'black', 'pivot': 'Pivot_Hatch_Assembly'})
    print(f'Civic_Rear_LicensePlate_Frame {len(F):5d} faces')
    out['index'] = np.array(json.dumps(index))
    np.savez_compressed(sys.argv[2], **out)


if __name__ == '__main__':
    main()
