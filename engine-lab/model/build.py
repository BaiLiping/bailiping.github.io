# build.py: builds the Porsche 911 Turbo 3.3 (930) Type 930/60 engine model from nothing.
#
#   blender -b --factory-startup --python build.py -- [--renders] [--blend] [--only core] [--theta 90]
#
# Writes engine930.glb and engine930.json beside the page (one directory up), prints triangles per group and the
# bounding box, and fails if a mechanism rule is broken.  --renders also writes check renders into _renders/.
# The model is original work built from published dimensions and photographs studied for reference; see
# PARTS.md for every number and NOTES.md for what is simplified.  Unofficial, not affiliated with Porsche.
import bpy, sys, os, math, json, time
from mathutils import Vector as V, Matrix, Quaternion

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kit
import spec as S
from kit import Geo, make, instance, empty, material, MM

ARGS = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
OUT = os.path.dirname(HERE)
STEP = 4                    # crank degrees between baked keys
TRI_BUDGET = 420000         # unique triangles in the file

MATERIALS = {
    # name: (sRGB colour, metallic, roughness)
    'alu_cast': ((0.74, 0.745, 0.73), 1.0, 0.62),
    'alu_cover': ((0.80, 0.80, 0.785), 1.0, 0.50),
    'alu_mach': ((0.86, 0.865, 0.87), 1.0, 0.30),
    'alu_piston': ((0.82, 0.815, 0.80), 1.0, 0.36),
    'steel': ((0.60, 0.615, 0.64), 1.0, 0.32),
    'steel_hot': ((0.50, 0.47, 0.46), 1.0, 0.40),
    'steel_dark': ((0.23, 0.235, 0.25), 1.0, 0.48),
    'bearing': ((0.72, 0.66, 0.50), 1.0, 0.35),
    'zinc': ((0.80, 0.72, 0.42), 1.0, 0.42),
    'iron_cast': ((0.40, 0.31, 0.26), 0.85, 0.85),
    'exhaust': ((0.36, 0.345, 0.33), 0.9, 0.62),
    'black_paint': ((0.075, 0.078, 0.082), 0.0, 0.55),
    'black_grp': ((0.10, 0.10, 0.105), 0.0, 0.72),
    'rubber': ((0.045, 0.045, 0.048), 0.0, 0.92),
    'plastic_black': ((0.06, 0.06, 0.06), 0.0, 0.6),
    'plastic_brown': ((0.36, 0.21, 0.11), 0.0, 0.6),
    'brass': ((0.78, 0.58, 0.26), 1.0, 0.38),
    'copper': ((0.76, 0.45, 0.30), 1.0, 0.42),
    'ceramic': ((0.93, 0.93, 0.90), 0.0, 0.35),
    'cap_orange': ((0.70, 0.27, 0.12), 0.0, 0.50),
    'filter_paper': ((0.86, 0.66, 0.36), 0.0, 0.9),
    'label_yellow': ((0.88, 0.78, 0.34), 0.0, 0.6),
    'lead_black': ((0.07, 0.07, 0.08), 0.0, 0.7),
    'fuel_line': ((0.70, 0.71, 0.72), 1.0, 0.38),
}


# The mechanism is never sliced by the cutaway planes; every other part is.
MECHANISM_GROUPS = {'rods', 'pistons', 'valves', 'rockers', 'cams', 'timing', 'chain'}
MECHANISM_PARTS = {'Crankshaft', 'Flywheel', 'Clutch', 'Crank_Pulley', 'Turbo_Rotor', 'Wastegate_Valve', 'Throttle_Plate', 'Sensor_Plate',
                   'Bypass_Valve', 'Distributor_Rotor', 'Oil_Pump_Shaft'}


class Engine:
    """Everything the builders share: the root, the part register, the motion laws and the exported data."""

    def __init__(s):
        s.root = empty('Engine930')
        s.parts = {}        # name -> dict(group, label, cut, ex, tris, inst)
        s.anim = {}         # name -> function(theta) -> (location mm or None, Quaternion or None, scale or None)
        s.chains = {}       # tag -> chain path for the viewer
        s.gas = []          # gas paths for the viewer
        s.drive = {}        # name -> description of viewer-driven motion
        s.unique_tris = 0

    def add(s, name, geo, mat, group, label=None, cut=True, ex=(0, 0, 0), parent=None, loc=(0, 0, 0), rot=None, sharp=38.0,
            spin=None, param=None):
        """Create a part from a Geo.  Takes its name, geometry, material, group, display label, whether cutaway
        planes clip it, its exploded-view offset in millimetres, an optional parent, its location, rotation and
        auto-smooth angle, an optional spin (axis x, y, z and ratio to crank speed) and an optional viewer
        parameter.  Returns the object."""
        tris = geo.tris()
        o = make(name, geo, mat, parent or s.root, loc, rot, None, sharp)
        s.unique_tris += tris
        s._register(o, group, label or name, cut, ex, tris, False, spin, param)
        return o

    def inst(s, src, name, group=None, label=None, cut=None, ex=None, parent=None, loc=(0, 0, 0), rot=None, spin=None, param=None):
        """Create another part that shares src's mesh.  Arguments default to the source's.  Returns the object."""
        m = s.parts[src.name]
        o = instance(src, name, parent or s.root, loc, rot)
        s._register(o, group or m['group'], label or m['label'], m['cut'] if cut is None else cut, m['ex'] if ex is None else ex,
                    m['tris'], True, spin, param)
        return o

    def fastener(s, src, name, geo, loc, rot=None, mat='zinc', label='Fastener', group='fasteners', ex=(0, 0, 0), cut=False, parent=None):
        """Place a repeated small part: the first call builds the mesh, later calls instance it.  Takes the
        shared source object or None, the name, the Geo used on the first call, and placement.  Returns the
        source object to pass to the next call."""
        if src is None:
            return s.add(name, geo, mat, group, label, cut=cut, ex=ex, loc=loc, rot=rot, parent=parent)
        s.inst(src, name, group=group, label=label, cut=cut, ex=ex, loc=loc, rot=rot, parent=parent)
        return src

    def _register(s, o, group, label, cut, ex, tris, inst, spin, param):
        """Record a part and write its viewer data as custom properties (exported as glTF extras)."""
        if o.name in s.parts:
            raise RuntimeError('duplicate part name ' + o.name)
        cut = not (group in MECHANISM_GROUPS or o.name in MECHANISM_PARTS)
        s.parts[o.name] = dict(group=group, label=label, cut=bool(cut), ex=[float(v) for v in ex], tris=tris, inst=inst)
        o['g'] = group
        o['label'] = label
        o['cut'] = 1 if cut else 0
        o['ex'] = [float(v) for v in ex]
        if spin is not None:
            o['spin'] = [float(v) for v in spin]
        if param is not None:
            o['param'] = json.dumps(param)

    def pose(s, theta):
        """Put every moving part at a crank angle.  Takes degrees.  Returns nothing."""
        for name, fn in s.anim.items():
            o = bpy.data.objects[name]
            loc, rot, scl = fn(theta)
            if loc is not None:
                o.location = V(loc) * MM
            if rot is not None:
                o.rotation_quaternion = rot
            if scl is not None:
                o.scale = scl


# ---------------------------------------------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------------------------------------------
def check_mechanism():
    """Assert the mechanism rules over two crank turns: the valves stay clear of the pistons and of each other,
    and every valve is seated at its cylinder's firing TDC.  Returns a dict of the margins found."""
    worst_pv = 1e9
    for c in S.CYLS:
        for th in range(0, 720, 2):
            crown = S.piston_s(S.beta(c, th)) + S.COMP_H
            for kind in ('in', 'ex'):
                v = S.VALVE[kind]
                au, aw = S.valve_axis(kind)
                lift = S.valve_lift(kind, c, th)
                # lowest point of the valve head: its rim on the side toward the piston
                u_face = v['seat'][0] - au * lift - 1.2 * au - (v['head_d'] / 2) * abs(aw)
                worst_pv = min(worst_pv, u_face - crown)
        assert S.valve_lift('in', c, c['fire']) == 0.0 and S.valve_lift('ex', c, c['fire']) == 0.0, 'valve open at firing TDC'
    assert worst_pv > 0.5, 'piston meets valve: margin %.2f mm' % worst_pv
    rg_i, rg_e = S.rocker_geom('in'), S.rocker_geom('ex')
    lobe_max = S.CAM_BASE_R + S.VALVE['in']['lift'] / S.ROCKER_RATIO
    clear = min(math.hypot(g['shaft'][0] - S.CAM_U, g['shaft'][1]) for g in (rg_i, rg_e)) - S.ROCKER_SHAFT_D / 2 - lobe_max
    assert clear > 0.5, 'cam lobe meets a rocker shaft: %.2f mm' % clear
    assert lobe_max < S.CAM_JOURNAL_R, 'cam lobes must pass through the bearing bores'
    return dict(piston_to_valve_mm=round(worst_pv, 2), lobe_to_rocker_shaft_mm=round(clear, 2),
                overlap_tdc_intake_lift_mm=round(S.valve_lift('in', S.CYLS[0], 360.0), 2))


def bounds():
    """World bounding box of all meshes.  Returns (min, max) in millimetres."""
    lo, hi = V((1e9,) * 3), V((-1e9,) * 3)
    dg = bpy.context.evaluated_depsgraph_get()
    for o in bpy.data.objects:
        if o.type != 'MESH':
            continue
        mw = o.matrix_world
        for c in o.bound_box:
            p = mw @ V(c)
            lo = V((min(lo.x, p.x), min(lo.y, p.y), min(lo.z, p.z)))
            hi = V((max(hi.x, p.x), max(hi.y, p.y), max(hi.z, p.z)))
    return lo / MM, hi / MM


# ---------------------------------------------------------------------------------------------------------------
# check renders
# ---------------------------------------------------------------------------------------------------------------
VIEWS = {
    # name: (camera position mm, target mm, orthographic scale mm)
    'rear': ((-2400, 0, 120), (0, 0, 60), 1250),
    'front': ((2400, 0, 120), (0, 0, 60), 1250),
    'left': ((-80, 2400, 120), (-80, 0, 60), 1250),
    'right': ((-80, -2400, 120), (-80, 0, 60), 1250),
    'top': ((-80, 0, 2400), (-80, 0, 0), 1250),
    'bottom': ((-80, 0, -2400), (-80, 0, 0), 1250),
    'q_rear_left': ((-1700, 1500, 1000), (-60, 0, 60), 1350),
    'q_front_right': ((1700, -1500, 1000), (-60, 0, 60), 1350),
}


def render_views(outdir, names=None, res=(1500, 1100), hide_groups=(), prefix=''):
    """Render check images with the Workbench engine.  Takes the output directory, the view names, the
    resolution, groups to hide and a file name prefix.  Returns the list of files written."""
    scn = bpy.context.scene
    scn.render.engine = 'BLENDER_WORKBENCH'
    sh = scn.display.shading
    sh.light = 'MATCAP'
    sh.studio_light = 'basic_bright.exr'
    sh.color_type = 'MATERIAL'
    sh.show_cavity = True
    sh.cavity_type = 'BOTH'
    sh.show_object_outline = True
    sh.show_shadows = False
    scn.display.render_aa = '8'
    scn.render.resolution_x, scn.render.resolution_y = res
    scn.render.film_transparent = False
    scn.world = scn.world or bpy.data.worlds.new('W')
    scn.world.color = (0.9, 0.9, 0.88)
    scn.view_settings.view_transform = 'Standard'
    cam = bpy.data.objects.get('CheckCam')
    if cam is None:
        cd = bpy.data.cameras.new('CheckCam')
        cd.type = 'ORTHO'
        cd.clip_start, cd.clip_end = 0.01, 100
        cam = bpy.data.objects.new('CheckCam', cd)
        scn.collection.objects.link(cam)
    scn.camera = cam
    for o in bpy.data.objects:
        if o.type == 'MESH':
            o.hide_render = o.get('g') in hide_groups
    os.makedirs(outdir, exist_ok=True)
    files = []
    for name in (names or VIEWS):
        pos, tgt, scale = VIEWS[name]
        cam.location = V(pos) * MM
        d = (V(tgt) - V(pos)).normalized()
        cam.rotation_mode = 'QUATERNION'
        cam.rotation_quaternion = d.to_track_quat('-Z', 'Y')
        cam.data.ortho_scale = scale * MM
        scn.render.filepath = os.path.join(outdir, prefix + name + '.png')
        bpy.ops.render.render(write_still=True)
        files.append(scn.render.filepath)
    return files


# ---------------------------------------------------------------------------------------------------------------
# bake and export
# ---------------------------------------------------------------------------------------------------------------
def bake(E):
    """Key every crank-synchronous part through two crank turns, linear, STEP degrees apart.  Takes the Engine.
    Returns the number of keys per part."""
    scn = bpy.context.scene
    scn.render.fps = 60
    n = 720 // STEP
    scn.frame_start, scn.frame_end = 0, n
    for k in range(n + 1):
        th = k * STEP
        E.pose(th)
        for name, fn in E.anim.items():
            o = bpy.data.objects[name]
            loc, rot, scl = fn(th)
            if loc is not None:
                o.keyframe_insert('location', frame=k)
            if rot is not None:
                o.keyframe_insert('rotation_quaternion', frame=k)
            if scl is not None:
                o.keyframe_insert('scale', frame=k)
    for act in bpy.data.actions:
        for fc in _fcurves(act):
            for kp in fc.keyframe_points:
                kp.interpolation = 'LINEAR'
    scn.frame_set(0)
    E.pose(0.0)
    return n + 1


def _fcurves(act):
    """All F-curves of an action, across Blender's layered and legacy layouts.  Returns a list."""
    out = []
    for layer in getattr(act, 'layers', []):
        for strip in layer.strips:
            for bag in getattr(strip, 'channelbags', []):
                out.extend(bag.fcurves)
    if not out and hasattr(act, 'fcurves'):
        out = list(act.fcurves)
    return out


def export(E, info):
    """Write the GLB and the JSON the page loads.  Takes the Engine and the info dict.  Returns the GLB path."""
    glb = os.path.join(OUT, 'engine930.glb')
    cam = bpy.data.objects.get('CheckCam')
    if cam:
        bpy.data.objects.remove(cam)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.export_scene.gltf(filepath=glb, export_format='GLB', use_selection=False, export_yup=True, export_apply=False,
                              export_extras=True, export_animations=True, export_animation_mode='SCENE',
                              export_force_sampling=True, export_optimize_animation_size=True, export_anim_scene_split_object=False,
                              export_optimize_animation_keep_anim_object=False, export_bake_animation=False,
                              export_texcoords=False, export_tangents=False, export_cameras=False, export_lights=False,
                              export_materials='EXPORT', export_skins=False, export_morph=False)
    with open(os.path.join(OUT, 'engine930.json'), 'w') as fh:
        json.dump(info, fh, separators=(',', ':'))
    return glb


def main():
    """Build, check, optionally render, export.  Returns nothing."""
    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for name, (rgb, metal, rough) in MATERIALS.items():
        material(name, rgb, metal, rough)
    E = Engine()
    import engine_core
    engine_core.build_all(E)
    only = ARGS[ARGS.index('--only') + 1] if '--only' in ARGS else ''
    if only != 'core':
        import engine_dress
        engine_dress.build_all(E)
    margins = check_mechanism()
    E.pose(float(ARGS[ARGS.index('--theta') + 1]) if '--theta' in ARGS else 0.0)
    bpy.context.view_layer.update()
    lo, hi = bounds()
    groups = {}
    for name, m in E.parts.items():
        gg = groups.setdefault(m['group'], [0, 0, 0])
        gg[0] += 1
        gg[1] += 0 if m['inst'] else m['tris']
        gg[2] += m['tris']
    print('\n%-14s %6s %9s %9s' % ('group', 'parts', 'unique', 'drawn'))
    for k in sorted(groups):
        print('%-14s %6d %9d %9d' % (k, *groups[k]))
    drawn = sum(v[2] for v in groups.values())
    print('%-14s %6d %9d %9d' % ('TOTAL', len(E.parts), E.unique_tris, drawn))
    print('bounds mm  x %.0f..%.0f  y %.0f..%.0f  z %.0f..%.0f' % (lo.x, hi.x, lo.y, hi.y, lo.z, hi.z))
    print('margins', margins)
    assert E.unique_tris <= TRI_BUDGET, 'triangle budget exceeded: %d' % E.unique_tris
    if '--renders' in ARGS:
        rd = os.path.join(HERE, '_renders')
        render_views(rd)
        render_views(rd, ['q_rear_left', 'top', 'rear', 'left'], hide_groups=('cooling', 'induction', 'covers', 'camhousing', 'chainhousing', 'crankcase', 'heads', 'cylinders', 'mounts', 'exhaust', 'turbo', 'ignition', 'fuel'), prefix='inside_')
    if '--noexport' not in ARGS:
        keys = bake(E)
        info = dict(
            name='Porsche 911 Turbo 3.3 (930), engine Type 930/60', unofficial=True,
            units='millimetres; +X front of car, +Y left, +Z up; origin on the crank axis at main bearing 4',
            step_deg=STEP, keys=keys, fps=60,
            spec=dict(bore=S.BORE, stroke=S.STROKE, rod=S.ROD, displacement_cc=round(math.pi / 4 * S.BORE ** 2 * S.STROKE * 6 / 1000.0, 1),
                      compression=7.0, firing_order=list(S.FIRING_ORDER), power_ps=S.POWER_PS, power_rpm=S.POWER_RPM,
                      torque_nm=S.TORQUE_NM, torque_rpm=S.TORQUE_RPM, boost_bar=S.BOOST_BAR, boost_rpm=S.BOOST_RPM,
                      redline=S.REDLINE, idle=S.IDLE, fan_ratio=S.FAN_RATIO),
            cylinders=[dict(n=c['n'], bank=c['bank'], x=c['x'], fire=c['fire'],
                            intake=[c['fire'] + S._valve_window('in')[0] - S._full_half('in'), c['fire'] + S._valve_window('in')[0] + S._full_half('in')],
                            exhaust=[c['fire'] + S._valve_window('ex')[0] - S._full_half('ex'), c['fire'] + S._valve_window('ex')[0] + S._full_half('ex')])
                       for c in S.CYLS],
            chains=E.chains, gas=E.gas, drive=E.drive, margins=margins,
            groups={k: dict(parts=v[0], unique_tris=v[1], drawn_tris=v[2]) for k, v in groups.items()},
            bounds=dict(min=[round(v, 1) for v in lo], max=[round(v, 1) for v in hi]), unique_tris=E.unique_tris, drawn_tris=drawn)
        glb = export(E, info)
        print('wrote %s  %.2f MB' % (glb, os.path.getsize(glb) / 1e6))
    if '--blend' in ARGS:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, 'engine930.blend'))
    print('done in %.1f s' % (time.time() - t0))


main()
