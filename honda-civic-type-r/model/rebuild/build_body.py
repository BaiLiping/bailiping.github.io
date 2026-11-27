"""Apply the designed shaping to the envelope field. Run: venv/bin/python build_body.py <in.npz> <out.npz>.
Order: doors/sills (design_side), front end (design_front), rear end (design_rear)."""
import sys
import numpy as np
import design_side, design_front, design_rear

d = np.load(sys.argv[1])
F = d['F']; grid = (float(d['x0']), float(d['y0']), float(d['z0']), float(d['h']))
F = design_side.apply(F, grid)
F = design_front.apply(F, grid)
F = design_rear.apply(F, grid)
np.savez_compressed(sys.argv[2], F=F.astype(np.float32), x0=d['x0'], y0=d['y0'], z0=d['z0'], h=d['h'])
print('designed field written', sys.argv[2])
