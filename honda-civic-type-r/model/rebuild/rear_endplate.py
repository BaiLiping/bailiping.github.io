"""Side-view outlines of the wing endplate parts, from P24 side regions, smeared down by DROP so the part
overlaps what it stands on.

Run: venv/bin/python rear_endplate.py <ids comma list> <DROP m> > <profile.json>
  endplate fin (body colour): ids 20, small drop (overlaps the pylon strip below it)
  pylon foot (black):         ids 99,129 (the "lower pylons" under the fin), drop into the quarter shoulder
Frame 19 shows region 59 under the pylons is the black top of the taillamp housing on the body, not the fin.
Prints a simplified polygon [[x, y], ...] (glTF metres) for parts_rear.json 'profile'.
"""
import sys, json
import numpy as np
import cv2
import outlines

ids = [int(v) for v in sys.argv[1].split(',')]
drop = float(sys.argv[2])
m = outlines.region_mask('side', ids, sym=False).astype(np.uint8)
d = int(round(drop * 1000))
ext = m.copy()
for k in range(1, d + 1):
    ext[k:] |= m[:-k]                     # smear the mask downward (v grows downward)
ext = cv2.morphologyEx(ext, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
cs, _ = cv2.findContours(ext, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
c = max(cs, key=cv2.contourArea)
c = cv2.approxPolyDP(c, 2.0, True)[:, 0]
pts = [[round(2.5 - u / 1000, 4), round(1.65 - v / 1000, 4)] for u, v in c]
print(json.dumps(pts))
print(len(pts), 'points; x', min(p[0] for p in pts), max(p[0] for p in pts), 'y', min(p[1] for p in pts),
      max(p[1] for p in pts), file=sys.stderr)
