"""Split a P24 factory-drawing plate into its closed regions and the car silhouette.

Run: venv/bin/python regions.py <plate.png> <out prefix>
Line pixels are the dark strokes. The car silhouette is every pixel not reachable from the image border
without crossing a stroke (strokes themselves count as inside when they border the car). Every closed
non-stroke area inside the silhouette is a region with an id.
Writes <prefix>-labels.npy (int32, 0 = stroke/outside, -1 = outside), <prefix>-sil.png (silhouette mask),
<prefix>-regions.json (id, area, bbox, centroid in plate pixels) and <prefix>-ids.png (colour map with ids,
for picking regions by eye). Returns nothing.
"""
import sys, json, numpy as np, cv2

src, prefix = sys.argv[1], sys.argv[2]
img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
H, W = img.shape
stroke = img < 190
stroke = cv2.dilate(stroke.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0     # close hairline gaps
free = (~stroke).astype(np.uint8)
n, lab = cv2.connectedComponents(free, connectivity=4)
border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
outside = np.isin(lab, border[border > 0])
# silhouette: not outside; strokes adjacent to outside are the outline and count as inside
sil = ~outside
sil = cv2.morphologyEx(sil.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
labels = np.where(outside, -1, np.where(stroke, 0, lab)).astype(np.int32)
# renumber inside regions by area, drop specks
ids, counts = np.unique(labels[labels > 0], return_counts=True)
keep = ids[counts >= 40]
order = keep[np.argsort(-counts[np.isin(ids, keep)])]
remap = np.zeros(lab.max() + 1, np.int32)
remap[order] = np.arange(1, len(order) + 1)
labels = np.where(labels > 0, remap[np.clip(labels, 0, None)], labels)
rows = []
for k in range(1, len(order) + 1):
    ys, xs = np.nonzero(labels == k)
    rows.append({'id': k, 'area': int(len(xs)), 'bbox': [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                 'centroid': [float(xs.mean()), float(ys.mean())]})
np.save(prefix + '-labels.npy', labels)
cv2.imwrite(prefix + '-sil.png', (sil * 255).astype(np.uint8))
json.dump(rows, open(prefix + '-regions.json', 'w'))
rng = np.random.default_rng(3)
pal = (rng.integers(90, 250, (len(order) + 1, 3))).astype(np.uint8)
vis = np.full((H, W, 3), 255, np.uint8)
inside = labels > 0
vis[inside] = pal[labels[inside]]
vis[labels == 0] = 0
for r in rows:
    x, y = r['centroid']
    # put the label inside the region: nearest region pixel to the centroid
    if labels[int(y), int(x)] != r['id']:
        ys, xs = np.nonzero(labels == r['id']); j = np.argmin((xs - x) ** 2 + (ys - y) ** 2); x, y = xs[j], ys[j]
    s = 0.9 if r['area'] > 20000 else 0.6 if r['area'] > 2000 else 0.45
    cv2.putText(vis, str(r['id']), (int(x) - 8, int(y) + 5), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, str(r['id']), (int(x) - 8, int(y) + 5), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 1, cv2.LINE_AA)
cv2.imwrite(prefix + '-ids.png', vis)
print(src, 'regions', len(rows), 'silhouette px', int(sil.sum()))
