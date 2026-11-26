"""Calibrated cameras for Ford's 36 F-150 Raptor 360-degree renders (2160x926 frames, numbered 01-36).

setup(k) places a Blender camera that reproduces frame k; project(k, X) maps model-frame points to pixels.
Ford renders the body 62 mm lower on its wheels than the published ride height, so the model frame is the
render frame raised by DZ."""
import json, numpy as np
try:
    import bpy, mathutils
except ImportError:
    bpy = mathutils = None
from pathlib import Path
CAMS = str(Path(__file__).resolve().parent / 'ford360_cameras.json')
DZ = 0.062
def setup(k, scene=None, cams=CAMS):
    scene = scene or bpy.context.scene
    cam = json.load(open(cams)); f = cam['f']; W, H = cam['width'], cam['height']
    fr = cam['frames'][str(k)]
    R = np.array(fr['R']); t = np.array(fr['t']); C = -R.T @ t + np.array([0, 0, DZ])
    du, dv, th, s = fr.get('warp', [0, 0, 0, 1])
    cx, cy = fr.get('cx', cam['cx']), fr.get('cy', cam['cy'])
    # warp acts around image centre: principal point moves, focal scales, image rotates by th
    c, sn = np.cos(th), np.sin(th)
    pcx = s * (c * (cx - W / 2) - sn * (cy - H / 2)) + W / 2 + du
    pcy = s * (sn * (cx - W / 2) + c * (cy - H / 2)) + H / 2 + dv
    # camera axes: rows of R are (right, down, forward) in world; apply image rotation th about optical axis
    Rr = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
    Rw2c = Rr @ R
    right, down, fwd = Rw2c
    M = np.eye(4); M[:3, 0] = right; M[:3, 1] = -down; M[:3, 2] = -fwd; M[:3, 3] = C
    name = f'FordCam_{k:02d}'
    ob = bpy.data.objects.get(name)
    if ob is None:
        cd = bpy.data.cameras.new(name); ob = bpy.data.objects.new(name, cd); scene.collection.objects.link(ob)
    ob.matrix_world = mathutils.Matrix(M.tolist())
    cd = ob.data; cd.sensor_fit = 'HORIZONTAL'; cd.sensor_width = 36.0
    cd.lens = f * s * 36.0 / W
    cd.shift_x = -(pcx - W / 2) / W
    cd.shift_y = (pcy - H / 2) / W
    cd.clip_start = 0.5; cd.clip_end = 80
    scene.render.resolution_x = W; scene.render.resolution_y = H; scene.render.resolution_percentage = 100
    scene.camera = ob
    return ob
def project(k, X, cams=CAMS):
    cam = json.load(open(cams)); f = cam['f']; W, H = cam['width'], cam['height']; fr = cam['frames'][str(k)]
    R = np.array(fr['R']); t = np.array(fr['t']); P = np.asarray(X, float) - np.array([0, 0, DZ])
    Xc = P @ R.T + t; u = f * Xc[:, 0] / Xc[:, 2] + fr.get('cx', cam['cx']); v = f * Xc[:, 1] / Xc[:, 2] + fr.get('cy', cam['cy'])
    du, dv, th, s = fr.get('warp', [0, 0, 0, 1]); c, sn = np.cos(th), np.sin(th)
    return np.stack([s * (c * (u - W / 2) - sn * (v - H / 2)) + W / 2 + du, s * (sn * (u - W / 2) + c * (v - H / 2)) + H / 2 + dv], 1)
