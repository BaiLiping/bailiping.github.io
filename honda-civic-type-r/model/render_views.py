"""Render a Civic GLB from calibrated photo cameras in Blender, for overlay against the photos.

Run: blender -b --python render_views.py -- <model.glb> <cameras.json> <frames comma list|all> <out_dir> [--mode M] [--scale S]
Takes a GLB, a cameras json (turntable_calib.py output, or any json with the same 'cameras' / 'width' /
'height' layout; camera K/R/C are OpenCV convention in glTF axes), frames, and an output dir.
Modes: 'parts' (flat random colour per object -> panel boundaries), 'shaded' (studio-lit matcap),
'sil' (white silhouette). --scale multiplies the camera resolution (default 0.5).
Writes <out_dir>/<frame>-<mode>.png. Returns nothing.
"""
import bpy, sys, json, math
import numpy as np
from mathutils import Matrix

argv = sys.argv[sys.argv.index('--') + 1:]
glb, cams_path, frames, out = argv[:4]
mode = argv[argv.index('--mode') + 1] if '--mode' in argv else 'parts'
scale = float(argv[argv.index('--scale') + 1]) if '--scale' in argv else 0.5
cams = json.load(open(cams_path))
W, H = cams['width'], cams['height']
frames = list(cams['cameras']) if frames == 'all' else frames.split(',')

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
sc = bpy.context.scene
sc.render.engine = 'BLENDER_WORKBENCH'
sh = sc.display.shading
if mode == 'parts':
    sh.light = 'FLAT'; sh.color_type = 'RANDOM'
elif mode == 'sil':
    sh.light = 'FLAT'; sh.color_type = 'SINGLE'; sh.single_color = (1, 1, 1)
else:
    sh.light = 'STUDIO'; sh.color_type = 'MATERIAL'; sh.show_cavity = True; sh.cavity_type = 'BOTH'
sh.background_type = 'VIEWPORT'; sh.background_color = (0, 0, 0)
sc.view_settings.view_transform = 'Standard'
sc.render.film_transparent = mode != 'sil'
sc.render.resolution_x = int(round(W * scale)); sc.render.resolution_y = int(round(H * scale))
sc.render.resolution_percentage = 100

cd = bpy.data.cameras.new('cam'); cam = bpy.data.objects.new('cam', cd); sc.collection.objects.link(cam); sc.camera = cam
for k in frames:
    c = cams['cameras'][str(k)]
    K = np.array(c['K']); R = np.array(c['R']); C = np.array(c['C'])
    gl2bl = lambda v: np.array([v[0], -v[2], v[1]])
    right, down, fwd = R
    M = np.eye(4)
    M[:3, 0] = gl2bl(right); M[:3, 1] = gl2bl(-down); M[:3, 2] = gl2bl(-fwd); M[:3, 3] = gl2bl(C)
    cam.matrix_world = Matrix(M.tolist())
    cd.sensor_fit = 'HORIZONTAL'; cd.sensor_width = 36.0
    cd.lens = K[0, 0] * 36.0 / W
    cd.shift_x = -(K[0, 2] - W / 2) / W
    cd.shift_y = (K[1, 2] - H / 2) / W
    cd.clip_start = 0.3; cd.clip_end = 60
    sc.render.filepath = f'{out}/{int(k):02d}-{mode}.png'
    bpy.ops.render.render(write_still=True)
