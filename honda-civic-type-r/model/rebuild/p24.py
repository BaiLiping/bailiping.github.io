"""Honda P24 factory 3-view plates (1 px = 1 mm) and Honda 360 rig: masks, 2D signed distances, sampling.

All geometry is in glTF axes (metres): X forward (nose +), Y up (ground 0), Z right (passenger +, driver -).
Plate pixel maps (plates-meta.json):
  side  (left elevation):  u = (2.5 - X) * 1000, v = (1.65 - Y) * 1000
  top   (plan):            u = (2.5 - X) * 1000, v = (1.15 - Z) * 1000
  front (front elevation): u = (1.15 - Z) * 1000, v = (1.65 - Y) * 1000
"""
import json
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage

ROOT = Path('/home/bailiping/Desktop/civic-fk8-refine')
REB = ROOT / 'work/rebuild'
P24 = REB / 'p24'
RIG = ROOT / 'work/calib_honda/rig_body.json'
MASKS = ROOT / 'work/calib_honda/masks'

WHEEL_X = (1.3484, -1.3484)          # axle X (P24)
WHEEL_Y = 0.3195                      # wheel centre height (static loaded radius)
ARCH_R = 0.385                        # arch opening radius in the side view (wheelhouse gap outer edge)
WHEELHOUSE_Z = (0.60, 0.62)           # inner wheelhouse wall |Z|, front and rear

# regions that are not body shell (ids from regions.py on each plate)
EXCLUDE = {
    'side': [12, 3, 9,                       # under-car areas closed by dimension lines
             64, 67, 77, 98,                 # arch openings below the body line
             20, 93, 99, 129,                # rear wing and its lower pylons
             86, 145,                        # shark-fin antenna, hood scoop
             105],                           # exhaust tip
    'top': [41, 121, 40, 124],                                    # door mirror heads (their feet are on the doors)
    'front': [37, 106, 50, 57, 49, 95, 36, 105, 51, 56, 48, 94,   # door mirrors
              2, 23, 24, 32, 33, 58, 67],                          # ground area, tyres and their outer strips
}


def labels(view):
    """Takes 'side'|'top'|'front'; returns the region label image (int32; -1 outside, 0 stroke)."""
    return np.load(P24 / f'{view}-labels.npy')


def to_px(view, X, Y, Z):
    """Takes glTF coordinates (arrays); returns plate pixel (u, v) for the view."""
    if view == 'side': return (2.5 - X) * 1000, (1.65 - Y) * 1000
    if view == 'top': return (2.5 - X) * 1000, (1.15 - Z) * 1000
    return (1.15 - Z) * 1000, (1.65 - Y) * 1000


def body_mask(view, arches=True):
    """Takes a view; returns the body-shell silhouette (bool image): inside the car outline, minus excluded
    regions. In the side view the wheel-arch discs are cut out (arches=True) or filled down to 0.20 m
    (arches=False, for the car's core between the wheelhouses). Strokes between kept regions stay inside."""
    lab = labels(view)
    m = lab != -1
    for k in EXCLUDE[view]:
        m &= lab != k
    if view == 'side':
        H, W = lab.shape
        vv, uu = np.mgrid[0:H, 0:W]
        for wx in WHEEL_X:
            cu, cv = (2.5 - wx) * 1000, (1.65 - WHEEL_Y) * 1000
            disc = (uu - cu) ** 2 + (vv - cv) ** 2 <= (ARCH_R * 1000) ** 2
            m = m & ~disc if arches else (m | disc) & ~(disc & (vv > (1.65 - 0.20) * 1000))
    # strokes that now border excluded areas: drop thin leftovers
    disk = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, disk(9))           # fill slits left by thin regions
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, disk(11)) > 0                         # drop strokes of removed parts
    # the mirror heads hide the body outline behind them: bridge it across the mirror span
    if view == 'top':
        m = _bridge(m, (2.5 - 1.10) * 1000, (2.5 - 0.35) * 1000, axis=1)
    if view == 'front':
        m = _bridge(m, (1.65 - 1.12) * 1000, (1.65 - 0.86) * 1000, axis=0)
    # keep the largest component (the car)
    n, lab2, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n > 2:
        big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        m = lab2 == big
    return m


def _bridge(m, a, b, axis):
    """Takes a mask and an index span [a, b] along axis (1 = columns, 0 = rows); returns the mask with the
    outer outline across that span replaced by linear interpolation of the outline at the span ends."""
    m = m.copy(); a, b = int(a), int(b)
    M = m if axis == 1 else m.T
    def edges(c):
        r = np.nonzero(M[:, c])[0]
        return (r.min(), r.max()) if len(r) else None
    ea, eb = edges(a), edges(b)
    if ea is None or eb is None: return m
    for c in range(a + 1, b):
        t = (c - a) / (b - a)
        lo = int(round((1 - t) * ea[0] + t * eb[0])); hi = int(round((1 - t) * ea[1] + t * eb[1]))
        col = np.zeros(M.shape[0], bool); col[lo:hi + 1] = True
        M[:, c] = col
    return m


def sdf2d(mask):
    """Takes a bool image; returns a float32 signed distance in pixels (negative inside), sub-pixel smoothed."""
    m = mask.astype(np.uint8)
    din = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    d = np.where(mask, -din + 0.5, dout - 0.5).astype(np.float32)
    return ndimage.gaussian_filter(d, 0.7)


def sample(img, u, v, outside=1e3):
    """Bilinear sample of a 2D float image at pixel coords (u along columns, v along rows); outside -> value."""
    H, W = img.shape
    ok = (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    out = ndimage.map_coordinates(img, [np.clip(v, 0, H - 1), np.clip(u, 0, W - 1)], order=1, mode='nearest')
    return np.where(ok, out, outside)


def rig():
    """Returns (cameras dict {frame: (K, R, C)}, width, height) of the body-referenced Honda 360 rig."""
    r = json.load(open(RIG))
    cams = {int(k): (np.array(c['K']), np.array(c['R']), np.array(c['C'])) for k, c in r['cameras'].items()}
    return cams, r['width'], r['height']


def project(K, R, C, P):
    """Takes camera K, R, C and points P (N,3) in glTF axes; returns pixel u, v and depth."""
    q = (P - C) @ R.T
    d = q[:, 2]
    uv = q @ K.T
    return uv[:, 0] / d, uv[:, 1] / d, d
