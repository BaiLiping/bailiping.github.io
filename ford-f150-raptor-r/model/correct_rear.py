"""Local rear-body corrections for the verified 2024 Raptor R base mesh.

Call ``apply()`` after importing the pre-fit GLB. Coordinates are Blender world
coordinates in metres: X forwards, Y left, Z up. The named articulation pivots
are retained. Styling dimensions below are photo-derived, not factory CAD data.
The Ford 2024 Raptor R dimensions remain the authority for the outer envelope.
"""

import math

import bpy
import bmesh
from mathutils import Matrix, Vector


def _object(name):
    obj = bpy.data.objects.get(name)
    if obj is None or obj.type != 'MESH':
        raise ValueError('Missing base-model mesh: ' + name)
    return obj


def _bounds(obj):
    vertices = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    return [min(v[i] for v in vertices) for i in range(3)], [max(v[i] for v in vertices) for i in range(3)]


def _edit_world(obj, transform):
    world = obj.matrix_world.copy()
    inverse = world.inverted()
    for vertex in obj.data.vertices:
        vertex.co = inverse @ Vector(transform(world @ vertex.co))
    obj.data.update()
    # The original split normals correspond to the old shape. Regenerate them.
    if obj.data.has_custom_normals:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        try:
            bpy.ops.mesh.customdata_custom_splitnormals_clear()
        except RuntimeError:
            pass
        obj.select_set(False)


def _scale_world(obj, center, scales):
    _edit_world(obj, lambda p: [center[i] + (p[i] - center[i]) * scales[i] for i in range(3)])


def _material(name):
    material = bpy.data.materials.get(name)
    if material is None:
        raise ValueError('Missing base-model material: ' + name)
    return material


def _mesh(name, vertices, faces, material, parent, bevel=0.0):
    old = bpy.data.objects.get(name)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)
    data = bpy.data.meshes.new(name + '_Mesh')
    data.from_pydata(vertices, [], faces)
    data.materials.append(material)
    data.update()
    bm = bmesh.new()
    bm.from_mesh(data)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(data)
    bm.free()
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.parent = parent
    obj.matrix_parent_inverse = parent.matrix_world.inverted()
    obj.matrix_world = Matrix.Identity(4)
    if bevel:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        modifier = obj.modifiers.new('Soft stamped edge', 'BEVEL')
        modifier.width = bevel
        modifier.segments = 3
        modifier.limit_method = 'ANGLE'
        modifier.harden_normals = True
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        modifier = obj.modifiers.new('Panel normals', 'WEIGHTED_NORMAL')
        modifier.keep_sharp = True
        modifier.weight = 50
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        obj.select_set(False)
    return obj


def _prism(name, outline, rear_x, front_x, material, parent, bevel=0.0):
    """Extrude a closed Y/Z outline. X ends may be functions of Y/Z."""
    vertices = []
    for depth in (rear_x, front_x):
        for y, z in outline:
            vertices.append((depth(y, z) if callable(depth) else depth, y, z))
    n = len(outline)
    faces = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    faces += [(i, (i + 1) % n, (i + 1) % n + n, i + n) for i in range(n)]
    return _mesh(name, vertices, faces, material, parent, bevel)


def _box(name, lower, upper, material, parent, bevel=0.0):
    return _prism(name, [(lower[1], lower[2]), (upper[1], lower[2]),
                         (upper[1], upper[2]), (lower[1], upper[2])],
                  lower[0], upper[0], material, parent, bevel)


def _tube(name, points, radius, material, parent, sides=10):
    """Small rounded edging on a bumper pressing, lying on a rear face."""
    vertices, faces = [], []
    for x, y, z in points:
        for step in range(sides):
            angle = 2 * math.pi * step / sides
            vertices.append((x + radius * math.cos(angle), y, z + radius * math.sin(angle)))
    for row in range(len(points) - 1):
        for step in range(sides):
            a = row * sides + step
            b = row * sides + (step + 1) % sides
            faces.append((a, b, b + sides, a + sides))
    return _mesh(name, vertices, faces, material, parent)


def _disc(name, center, radius, material, parent):
    x, y, z = center
    count = 20
    vertices = [(x, y, z)] + [(x, y + radius * math.cos(i * 2 * math.pi / count),
                              z + radius * math.sin(i * 2 * math.pi / count)) for i in range(count)]
    return _mesh(name, vertices, [(0, (i + 1) % count + 1, i + 1) for i in range(count)], material, parent)


def _correct_letters(log):
    letters = _object('RaptorR_Tailgate_FORD_Letters')
    low, high = _bounds(letters)
    center = [(a + b) / 2 for a, b in zip(low, high)]
    width, height = high[1] - low[1], high[2] - low[2]
    scales = (1.0, 1.32 / width, 0.100 / height)
    for name in ('RaptorR_Tailgate_FORD_Letters', 'RaptorR_Tailgate_FORD_Inlays'):
        _scale_world(_object(name), center, scales)
    log['tailgate_letter_width_height_m'] = {'before': [width, height], 'after': [1.32, 0.100]}

    # The pre-fit applique is already much closer than the warped live asset.
    # Only reduce its height by 10 mm, retaining the actual horizontal emphasis.
    applique = _object('RaptorR_Tailgate_Applique')
    low, high = _bounds(applique)
    panel_center = [(a + b) / 2 for a, b in zip(low, high)]
    panel_scale = 0.280 / (high[2] - low[2])
    for name in ('RaptorR_Tailgate_Applique', 'RaptorR_Tailgate_AppliqueGroove_0',
                 'RaptorR_Tailgate_AppliqueGroove_1'):
        _scale_world(_object(name), panel_center, (1, 1, panel_scale))
    log['tailgate_applique_height_m'] = {'before': high[2] - low[2], 'after': 0.280}

    # Factory small wordmarks are squat too, but keep their positions and widths.
    for name, amount in [('RaptorR_Tailgate_Badge_F150', 0.78),
                         ('RaptorR_Tailgate_Badge_RAPTOR', 0.65),
                         ('RaptorR_Tailgate_Badge_CodeOrangeR', 0.65)]:
        obj = _object(name)
        low, high = _bounds(obj)
        center = [(a + b) / 2 for a, b in zip(low, high)]
        _scale_world(obj, center, (1, 1, amount))


def _correct_lamps(parent, log):
    paint = _material('Mat_Rapid_Red_Paint')
    for side, sign in [('L', 1), ('R', -1)]:
        housing = _object('RaptorR_Taillight_Housing_' + side)
        low, high = _bounds(housing)
        bottom, old_top = low[2], high[2]
        new_top = bottom + 0.390
        ratio = 0.390 / (old_top - bottom)
        for prefix in ('RaptorR_Taillight_Housing_', 'RaptorR_Taillight_Lens_', 'RaptorR_Taillight_Signature_'):
            _edit_world(_object(prefix + side), lambda p: (p[0], p[1], bottom + (p[2] - bottom) * ratio))

        # Close the former upper lamp opening with painted metal. This joins
        # the existing bedside, inner lamp jamb and rail; it is not a floating
        # rectangle hiding an open hole. The outer corner follows the bed taper.
        vertices = []
        for z in (new_top - .003, 1.428):
            vertices += [(-2.996, sign * .815, z), (-2.996, sign * 1.005, z),
                         (-2.802, sign * 1.027, z), (-2.802, sign * .815, z)]
        faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5),
                 (2, 3, 7, 6), (3, 0, 4, 7)]
        _mesh('RaptorR_Body_BedLampUpperCorner_' + side, vertices, faces, paint, parent, .006)
        log['taillight_' + side + '_z_m'] = {'before': [bottom, old_top], 'after': [bottom, new_top]}


def _correct_bumper(parent, log):
    steel = _material('Mat_Bumper_Steel_Charcoal')
    trim = _material('Mat_Trim_Matte_Black')
    original = _object('RaptorR_Rear_Bumper')
    old_bounds = _bounds(original)
    # Both old objects contain the straight bar/exhaust-hole construction.
    for name in ('RaptorR_Rear_Bumper', 'RaptorR_Rear_Bumper_StepPad'):
        bpy.data.objects.remove(_object(name), do_unlink=True)

    # The rear-most original step-pad point, -3.108 m, is preserved. All other
    # rear components stay forward of it, keeping the overall length unchanged.
    def rear(y, z):
        t = max(0.0, min(1.0, (abs(y) - .82) / .212))
        return -3.102 + .079 * t * t

    wing = [(.278, .849), (.992, .849), (1.032, .817), (1.032, .751),
            (.985, .727), (.450, .727), (.351, .633), (.278, .633)]
    for side, sign in [('L', 1), ('R', -1)]:
        profile = [(sign * y, z) for y, z in wing]
        _prism('RaptorR_Rear_Bumper_' + side + 'Wing', profile, rear,
               lambda y, z: rear(y, z) + .205, steel, parent, .008)

        # Dark shaped pad follows the wing and slopes into the plate recess.
        inner = [(sign * y, z) for y, z in [(.314, .824), (.973, .824),
                    (1.005, .804), (1.005, .765), (.969, .748), (.438, .748),
                    (.340, .653), (.314, .653)]]
        _prism('RaptorR_Rear_Bumper_' + side + 'Pressing', inner,
               lambda y, z: rear(y, z) - .0015, lambda y, z: rear(y, z) + .009,
               trim, parent, .003)

        # Rubber tread is on the horizontal upper step, not across the bumper face.
        _box('RaptorR_Rear_Bumper_StepPad_' + side,
             (-3.090, min(sign * .36, sign * .99), .851),
             (-2.835, max(sign * .36, sign * .99), .865), trim, parent, .005)
        for index in range(13):
            y = sign * (.380 + index * .046)
            _box('RaptorR_Rear_Bumper_StepRib_' + side + str(index),
                 (-3.070, y - .009, .865), (-2.855, y + .009, .868), trim, parent, .001)

        for index, y in enumerate((.43, .96)):
            yy = sign * y
            xx = rear(yy, .790) - .0020
            _disc('RaptorR_Rear_Bumper_SensorBezel_' + side + str(index),
                  (xx, yy, .790), .0135, steel, parent)
            _disc('RaptorR_Rear_Bumper_Sensor_' + side + str(index),
                  (xx - .0002, yy, .790), .009, trim, parent)

    # An inset centre well surrounds the retained plate and sits above the hitch.
    _prism('RaptorR_Rear_Bumper', [(-.278, .633), (.278, .633), (.278, .844), (-.278, .844)],
           -3.035, -2.875, steel, parent, .006)
    _box('RaptorR_Rear_Bumper_CenterLip', (-3.108, -.348, .615),
         (-2.955, .348, .638), steel, parent, .006)
    _box('RaptorR_Rear_Bumper_PlateWellTop', (-3.047, -.281, .821),
         (-2.870, .281, .849), trim, parent, .004)

    # Route the final tailpipe segments to outlets beneath the wings, instead
    # of penetrating the previous bumper bar. Blend the short terminal bend.
    def terminal_route(p):
        t = max(0.0, min(1.0, (-p[0] - 2.35) / .46))
        t = t * t * (3.0 - 2.0 * t)
        sign = 1 if p[1] > 0 else -1
        return (p[0], p[1] + sign * .130 * t, p[2] - .040 * t)

    _edit_world(_object('RaptorR_Exhaust_Pipes'), terminal_route)
    for side in ('L', 'R'):
        _edit_world(_object('RaptorR_Exhaust_Tip_' + side), terminal_route)
    log['rear_bumper'] = {'previous_bounds_m': old_bounds, 'rear_extent_m': -3.108,
                         'wing_bottom_m': .727, 'center_bottom_m': .615,
                         'exhaust_center_m': {'abs_y': .820, 'z': .650}}


def apply():
    """Correct rear details once and return a small dimension/change record."""
    if bpy.data.objects.get('RaptorR_Body_BedLampUpperCorner_L'):
        raise RuntimeError('Rear corrections already applied. Reimport the baseline before rebuilding.')
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.update()
    parent = bpy.data.objects.get('Pivot_Body_Assembly')
    if parent is None:
        raise ValueError('Missing Pivot_Body_Assembly')
    log = {}
    _correct_letters(log)
    _correct_lamps(parent, log)
    _correct_bumper(parent, log)
    bpy.context.view_layer.update()
    log['reference'] = '2024 Raptor R matched rear and rear-three-quarter photographs'
    log['scope'] = 'Photo-derived component proportions; articulation pivots and rear envelope retained'
    return log
