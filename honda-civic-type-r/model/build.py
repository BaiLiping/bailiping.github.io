"""Build the Civic Type R GLB: base model -> body shape fit (FFD) -> region edit scripts -> export.

Run: blender -b --python build.py -- <base.glb> <ffd.npz|none> <regions comma list|none> <out.glb> [--blend out.blend]
Takes the base GLB, an optional FFD displacement field (from work/tools/ffd_fit.py), the region scripts
to run (work/pipeline/regions/<name>.py, each defining apply(ctx)), and the output path.
Writes the uncompressed GLB (and optionally a .blend). Returns nothing.

Region scripts may only modify objects their region owns (see ownership.json); build.py checks this by
fingerprinting every other object's geometry and transform before and after each region runs.
"""
import bpy, sys, json, importlib.util, hashlib
from pathlib import Path
import numpy as np
from mathutils import Matrix, Vector

HERE = Path(__file__).resolve().parent
OWN = json.load(open(HERE / 'ownership.json'))
FROZEN_PIVOTS = ('Pivot_Corner_', 'Pivot_WheelSpin_', 'Pivot_Engine_', 'Pivot_Drivetrain', 'Pivot_Spin_',
                 'Pivot_Trans_', 'Pivot_Conrod_', 'Pivot_Piston_', 'Pivot_Interior_Cabin')
HINGE_PIVOTS = ('Pivot_Door_L', 'Pivot_Door_R', 'Pivot_Door_RL', 'Pivot_Door_RR', 'Pivot_Hood_Assembly',
                'Pivot_Hatch_Assembly', 'Pivot_Fuel_Flap')


def gl2bl(p):
    """Takes glTF-axis points (n,3); returns Blender-axis points (x, -z, y)."""
    return np.stack([p[:, 0], -p[:, 2], p[:, 1]], 1)


def bl2gl(p):
    """Takes Blender-axis points (n,3); returns glTF-axis points (x, z, -y)."""
    return np.stack([p[:, 0], p[:, 2], -p[:, 1]], 1)


class FFD:
    """Trilinear, mirror-symmetric displacement field from ffd_fit.py (glTF axes, grid over z<=0)."""

    def __init__(self, path):
        d = np.load(path); self.lo = d['lo']; self.step = float(d['step']); self.D = d['D']
        self.n = np.array(self.D.shape[:3])

    def disp_gl(self, P):
        """Takes glTF points (n,3); returns glTF displacement (n,3) by numpy trilinear lookup (border clamp).

        Matches torch grid_sample(align_corners=True, padding_mode='border') used in ffd_fit.py; numpy-only
        because Blender's bundled Python has no scipy.
        """
        sgn = np.where(P[:, 2] > 0, -1.0, 1.0)
        Q = P.copy(); Q[:, 2] = -np.abs(P[:, 2])
        g = np.clip((Q - self.lo) / self.step, 0, self.n - 1)
        i0 = np.minimum(np.floor(g).astype(int), self.n - 2); t = g - i0
        d = np.zeros((len(P), 3))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (t[:, 0] if dx else 1 - t[:, 0]) * (t[:, 1] if dy else 1 - t[:, 1]) * (t[:, 2] if dz else 1 - t[:, 2])
                    d += w[:, None] * self.D[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
        d[:, 2] *= sgn
        return d

    def disp_bl(self, Pb):
        """Takes Blender points (n,3); returns Blender displacement (n,3)."""
        return gl2bl(self.disp_gl(bl2gl(Pb)))


def under_frozen(ob):
    """Takes an object; returns True if it hangs under a mechanical/interior pivot that must not deform."""
    p = ob.parent
    while p is not None:
        if p.name.startswith(FROZEN_PIVOTS):
            return True
        p = p.parent
    return False


def apply_ffd(ffd):
    """Takes an FFD; deforms every non-frozen mesh and moves hinge pivots with the field. Returns count."""
    meshes = [ob for ob in bpy.data.objects if ob.type == 'MESH' and not under_frozen(ob)]
    world = {}
    for ob in meshes:
        co = np.empty(len(ob.data.vertices) * 3); ob.data.vertices.foreach_get('co', co)
        M = np.array(ob.matrix_world); co = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
        world[ob.name] = co
    for name in HINGE_PIVOTS:                                   # hinges ride with the body around them
        pv = bpy.data.objects.get(name)
        if pv is None:
            continue
        loc = np.array(pv.matrix_world.translation)[None]
        d = ffd.disp_bl(loc)[0]
        pv.matrix_world = Matrix.Translation(Vector(d)) @ pv.matrix_world
    bpy.context.view_layer.update()
    for ob in meshes:
        me = ob.data
        if me.users > 1:
            ob.data = me = me.copy()
        new = world[ob.name] + ffd.disp_bl(world[ob.name])
        Minv = np.array(ob.matrix_world.inverted())
        loc = new @ Minv[:3, :3].T + Minv[:3, 3]
        me.vertices.foreach_set('co', loc.ravel()); me.update()
    return len(meshes)


def fingerprint(ob):
    """Takes an object; returns a hash of its world matrix and (for meshes) vertex positions."""
    h = hashlib.sha1(np.round(np.array(ob.matrix_world), 6).tobytes())
    if ob.type == 'MESH':
        co = np.empty(len(ob.data.vertices) * 3); ob.data.vertices.foreach_get('co', co)
        h.update(np.round(co, 6).tobytes()); h.update(str(len(ob.data.polygons)).encode())
    return h.hexdigest()


def owns(region, name):
    """Takes a region and object name; returns True if the region may modify that object."""
    rules = OWN[region]
    return name in rules.get('names', []) or any(name.startswith(p) for p in rules.get('prefixes', []))


class Ctx:
    """Context handed to region scripts: owned-object lookup, helpers and paths."""

    def __init__(self, region, ffd):
        self.region = region; self.ffd = ffd; self.root = HERE.parent.parent
        self.gl2bl, self.bl2gl = gl2bl, bl2gl

    def obj(self, name):
        """Takes an owned object name; returns the Blender object (raises if not owned or missing)."""
        if not owns(self.region, name):
            raise PermissionError(f'{self.region} does not own {name}')
        return bpy.data.objects[name]

    def owned(self):
        """Returns all objects the region owns."""
        return [ob for ob in bpy.data.objects if owns(self.region, ob.name)]


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    base, ffd_path, regions, out = argv[:4]
    blend = argv[argv.index('--blend') + 1] if '--blend' in argv else None
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=base)
    ffd = FFD(ffd_path) if ffd_path != 'none' else None
    if ffd:
        print('FFD deformed', apply_ffd(ffd), 'meshes', flush=True)
    for region in ([] if regions == 'none' else regions.split(',')):
        before = {ob.name: fingerprint(ob) for ob in bpy.data.objects}
        spec = importlib.util.spec_from_file_location(region, HERE / 'regions' / f'{region}.py')
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        mod.apply(Ctx(region, ffd)); bpy.context.view_layer.update()
        after = {ob.name: fingerprint(ob) for ob in bpy.data.objects}
        bad = [n for n in before if n in after and before[n] != after[n] and not owns(region, n)]
        bad += [n for n in before if n not in after and not owns(region, n)]
        if bad:
            raise RuntimeError(f'region {region} modified objects it does not own: {bad[:10]}')
        print('region', region, 'ok', flush=True)
    bpy.ops.export_scene.gltf(filepath=out, export_format='GLB', export_extras=True, export_yup=True,
                              export_apply=False, export_animations=False, export_draco_mesh_compression_enable=False)
    if blend:
        bpy.ops.wm.save_as_mainfile(filepath=blend)
    print('wrote', out, flush=True)


main()
