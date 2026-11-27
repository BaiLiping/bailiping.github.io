"""Panel outlines from P24 drawing regions, in view coordinates (metres).

Run: venv/bin/python outlines.py <panels.json> <out outlines.json>
panels.json: [{name, view: side|top|front, ids: [...], sym: bool}, ...]
  side : the plate is the car's left side; outline in (x, y). The builder mirrors it for the right side.
  top  : outline in (x, z); with sym the passenger-half mask is mirrored across the centre line.
  front: outline in (z, y); with sym likewise.
Regions are merged across the strokes between them (closing), so a panel's edge sits on the outer edge of
Honda's shut-line stroke and neighbouring panels keep a stroke-wide gap. Writes [{name, view, outer, holes}].
"""
import sys, json
import numpy as np
import cv2
import p24


def region_mask(view, ids, sym):
    """Takes a view, region ids and symmetry flag; returns the merged panel mask (bool image)."""
    lab = p24.labels(view)
    m = np.isin(lab, ids).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    if sym:
        if view == 'top':
            c = 1150; m[c:] = 0; m[c:] = m[:c][::-1][:m.shape[0] - c]
        else:
            c = 1150; m[:, c:] = 0; m[:, c:] = m[:, :c][:, ::-1][:, :m.shape[1] - c]
    return m > 0


def smooth_loop(C, sigma=2.5, step=3.0):
    """Takes a closed pixel contour; returns it Gaussian-smoothed and resampled every `step` px."""
    C = C.astype(float)
    n = len(C)
    if n < 10: return C
    k = int(3 * sigma); w = np.exp(-0.5 * (np.arange(-k, k + 1) / sigma) ** 2); w /= w.sum()
    S = np.stack([np.convolve(np.concatenate([C[-k:, i], C[:, i], C[:k, i]]), w, 'valid') for i in range(2)], 1)
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(np.vstack([S, S[:1]]), axis=0), axis=1))])
    t = np.arange(0, d[-1], step)
    P = np.vstack([S, S[:1]])
    return np.stack([np.interp(t, d, P[:, 0]), np.interp(t, d, P[:, 1])], 1)


def to_view(view, P):
    """Plate px -> view coordinates (m)."""
    u, v = P[:, 0], P[:, 1]
    if view == 'side': return np.stack([2.5 - u / 1000, 1.65 - v / 1000], 1)
    if view == 'top': return np.stack([2.5 - u / 1000, 1.15 - v / 1000], 1)
    return np.stack([1.15 - u / 1000, 1.65 - v / 1000], 1)


def outline(spec):
    """Takes a panel spec; returns {name, view, outer, holes} with polygons as lists of [a, b]."""
    m = region_mask(spec['view'], spec['ids'], spec.get('sym', False)).astype(np.uint8)
    cs, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not cs: raise ValueError('empty panel ' + spec['name'])
    areas = [cv2.contourArea(c) for c in cs]
    top = int(np.argmax(areas))
    outer = to_view(spec['view'], smooth_loop(cs[top][:, 0]))
    holes = []
    for i, c in enumerate(cs):
        if hier[0][i][3] == top and cv2.contourArea(c) > 150:
            holes.append(to_view(spec['view'], smooth_loop(c[:, 0])).round(5).tolist())
    return {'name': spec['name'], 'view': spec['view'], 'outer': outer.round(5).tolist(), 'holes': holes}


if __name__ == '__main__':
    specs = json.load(open(sys.argv[1]))
    out = []
    for s in specs:
        o = outline(s); o.update({k: v for k, v in s.items() if k not in ('ids',)})
        out.append(o)
        print(f"{s['name']:28s} {s['view']:5s} outer {len(o['outer']):4d} pts, holes {len(o['holes'])}")
    json.dump(out, open(sys.argv[2], 'w'))
