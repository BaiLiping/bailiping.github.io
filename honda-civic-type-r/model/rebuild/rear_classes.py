"""Colour classes of Honda's rear frame 28 (Boost Blue 360 render), for cutting the rear face.

Run: venv/bin/python rear_classes.py <28.jpg> <out prefix>
Classes (cleaned with small openings/closings): paint (blue), red (lamp lenses, bumper strips, pinstripe),
clear (reverse-lamp lenses, chrome, badges: bright and unsaturated), dark (black trim, vents, diffuser, wing,
tyres). Writes <prefix>-<class>.png masks at the frame's resolution and <prefix>-classes.png for checking.
"""
import sys
import numpy as np
import cv2

img = cv2.imread(sys.argv[1]); prefix = sys.argv[2]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(float)
H, S, V = hsv[..., 0] * 2, hsv[..., 1] / 255, hsv[..., 2] / 255
car = ~((V > 0.93) & (S < 0.06))                        # white studio background
paint = car & (H > 190) & (H < 230) & (S > 0.45) & (V > 0.25)
red = car & ((H < 15) | (H > 340)) & (S > 0.45) & (V > 0.25)
clear = car & (S < 0.18) & (V > 0.55) & ~paint
dark = car & (V < 0.30) & ~red & ~paint
k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
def clean(m, o=3, c=5):
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k(c))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, k(o)) > 0
out = {'paint': clean(paint), 'red': clean(red, 3, 3), 'clear': clean(clear, 3, 3), 'dark': clean(dark, 3, 7)}
vis = np.full(img.shape, 255, np.uint8)
for name, col in (('dark', (40, 40, 40)), ('paint', (200, 120, 30)), ('red', (30, 30, 220)), ('clear', (200, 200, 200))):
    vis[out[name]] = col
    cv2.imwrite(f'{prefix}-{name}.png', out[name].astype(np.uint8) * 255)
cv2.imwrite(f'{prefix}-classes.png', vis)
print({k: int(v.sum()) for k, v in out.items()})
