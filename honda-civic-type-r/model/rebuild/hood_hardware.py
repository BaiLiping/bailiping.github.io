"""Cowl hinges and the removable-tip hood stay on the FK8 passenger apron.

All dimensions are in metres in the assembled, Y-up car. The two open-car
photographs locate the hinge and stay. The side photograph sets the rotation.
"""
import copy

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial.transform import Rotation

HINGE = np.array([1.055, 0.899, 0.0])
ANGLE = 52.0
STAY_BASE = np.array([1.522, 0.828, 0.744])
STAY_SOCKET = np.array([1.633, 0.825, 0.688])
STORED_DIRECTION = np.array([0.91, -0.20, -0.36])
BODY = 'Pivot_Body_Assembly'
HOOD = 'Pivot_Hood_Assembly'


def normalized(vec):
    return vec / np.linalg.norm(vec)


def from_up(direction):
    direction = normalized(direction)
    axis = np.cross([0.0, 1.0, 0.0], direction)
    return Rotation.from_rotvec(normalized(axis) * np.arctan2(np.linalg.norm(axis), direction[1]))


def plated(outline, z, width):
    """A folded hinge link seen in profile, extruded across the pin axis."""
    shape = np.array(outline)
    n = len(shape)
    verts = np.vstack([np.column_stack([shape, np.full(n, z + dz)]) for dz in (-width / 2, width / 2)])
    # The hinge blanks are convex. Use a separate center on each face.
    verts = np.vstack([verts, verts[:n].mean(0), verts[n:].mean(0)])
    i = np.arange(n)
    j = (i + 1) % n
    faces = np.vstack(
        [
            np.column_stack([i, j, j + n]),
            np.column_stack([i, j + n, i + n]),
            np.column_stack([j, i, np.full(n, 2 * n)]),
            np.column_stack([i + n, j + n, np.full(n, 2 * n + 1)]),
        ]
    )
    return solid_normals(verts, faces)


def solid_normals(points, faces):
    tri = points[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if np.sum(normal * (tri.mean(1) - points.mean(0))) < 0:
        faces = faces[:, ::-1]
        normal *= -1
    normals = np.zeros_like(points)
    for col in faces.T:
        np.add.at(normals, col, normal)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return points, normals, faces


def turned(profile, hollow=False):
    n = 20
    angle = np.arange(n) * 2 * np.pi / n
    points = np.vstack(
        [np.column_stack([r * np.cos(angle), np.full(n, y), r * np.sin(angle)]) for y, r in profile]
    )
    rings = np.arange(len(points)).reshape(-1, n)
    lower = rings if hollow else rings[:-1]
    upper = np.roll(rings, -1, axis=0) if hollow else rings[1:]
    a, b = lower.ravel(), upper.ravel()
    c, d = np.roll(upper, -1, axis=1).ravel(), np.roll(lower, -1, axis=1).ravel()
    faces = np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    if not hollow:
        for cap in (rings[0], rings[-1, ::-1]):
            faces = np.vstack([faces, np.column_stack([np.full(n - 2, cap[0]), cap[1:-1], cap[2:]])])
    return solid_normals(points, faces)


def install_hardware(model):
    indices = {n['name']: i for i, n in enumerate(model.document['nodes'])}
    mats = {m['name']: i for i, m in enumerate(model.document['materials'])}
    paint = model.document['materials'][model.primitive('Civic_Hood_Panel')['material']]['name']
    hardware = 'Mat_Engine_Machined_Aluminum'
    black = 'Mat_Gloss_Black_Trim'
    identity = Rotation.identity()

    def add(name, parent, material, position, geometry, rotation=identity):
        origin = np.asarray(model.nodes[parent].get('translation', [0.0, 0.0, 0.0]))
        model.add_mesh(
            name,
            indices[parent],
            mats[material],
            np.asarray(position) - origin,
            rotation.as_quat(),
            *geometry
        )

    # The exposed inner pressing has a softer, less glossy finish than the
    # exterior skin. Both use the same Boost Blue base color.
    pressing = copy.deepcopy(model.document['materials'][mats[paint]])
    pressing['name'] = 'Mat_Boost_Blue_Underhood'
    pressing['pbrMetallicRoughness']['roughnessFactor'] = 0.49
    coat = pressing.setdefault('extensions', {}).setdefault('KHR_materials_clearcoat', {})
    coat.update(clearcoatFactor=0.25, clearcoatRoughnessFactor=0.33)
    model.primitive('Civic_Hood_Inner_Frame')['material'] = len(model.document['materials'])
    model.document['materials'].append(pressing)

    # Rebase the inherited rig without moving any of the assembled skin.
    hood = model.nodes[HOOD]
    offset = np.array(hood['translation']) - HINGE
    for idx in hood['children']:
        child = model.document['nodes'][idx]
        child['translation'] = (np.array(child.get('translation', [0, 0, 0])) + offset).tolist()
    hood['translation'] = HINGE.tolist()
    hood.setdefault('extras', {}).update(openAngleDegrees=ANGLE, stayClearanceDegrees=2.0)

    pivot_x, pivot_y, _ = HINGE
    base = [
        (-0.056, -0.034),
        (0.044, -0.033),
        (0.048, -0.026),
        (0.008, 0.010),
        (-0.005, 0.012),
        (-0.047, -0.016),
    ]
    arm = [
        (-0.008, -0.004),
        (0.006, -0.008),
        (0.111, 0.002),
        (0.158, 0.009),
        (0.158, 0.016),
        (0.109, 0.015),
        (0.038, 0.011),
        (-0.009, 0.007),
    ]
    for side, sign in [('L', -1), ('R', 1)]:
        z = sign * 0.698
        anchor = np.array([pivot_x, pivot_y, z])
        add('Civic_Body_Shell_Hood_Hinge_' + side, BODY, paint, anchor, plated(base, 0.0, 0.016))
        add('Civic_Hood_Hinge_Arm_' + side, HOOD, paint, anchor, plated(arm, -sign * 0.010, 0.012))
        pin_dir = Rotation.from_euler('x', sign * 90, degrees=True)
        pin = turned([(-0.018, 0.005), (-0.015, 0.009), (0.010, 0.009), (0.013, 0.011), (0.015, 0.010)])
        add('Civic_Body_Shell_Hood_Hinge_Pin_' + side, BODY, hardware, anchor, pin, pin_dir)

    open_rotation = Rotation.from_euler('z', ANGLE, degrees=True)
    opened = open_rotation.apply(STAY_SOCKET - HINGE) + HINGE
    reach = np.linalg.norm(opened - STAY_BASE)
    rod = turned(
        [
            (0, 0.006),
            (0.014, 0.006),
            (0.018, 0.004),
            (reach * 0.63, 0.0038),
            (reach * 0.64, 0.009),
            (reach * 0.79, 0.009),
            (reach * 0.80, 0.004),
            (reach - 0.014, 0.004),
            (reach, 0.0035),
        ]
    )
    name = 'Civic_Body_Shell_Hood_Stay'
    add(name, BODY, black, STAY_BASE, rod, from_up(STORED_DIRECTION))
    # Both the viewer and the visibility bake use these same, rigid endpoints.
    model.nodes[name]['extras'] = {'raisedRotation': from_up(opened - STAY_BASE).as_quat().tolist()}
    add(
        'Civic_Body_Shell_Hood_Stay_Foot',
        BODY,
        black,
        STAY_BASE,
        turned([(-0.018, 0.020), (-0.004, 0.020), (0, 0.012)]),
    )
    add(
        'Civic_Hood_Stay_Socket',
        HOOD,
        hardware,
        STAY_SOCKET,
        turned([(-0.003, 0.008), (-0.002, 0.014), (0.002, 0.014), (0.012, 0.009),
                (0.012, 0.006), (0, 0.006)], hollow=True),
        from_up(open_rotation.inv().apply(opened - STAY_BASE)),
    )

    # Two rubber stops and the black center seal are visible along the
    # leading edge in the straight-on underside photo (BaT lot 163254).
    from hood_pressing import mesh_normals, outside_loop, sweep

    skin = model.read_geometry('Civic_Hood_Inner_Frame')[0]
    inside = LinearNDInterpolator(skin[:, [0, 2]], skin[:, 1])
    inverted = Rotation.from_euler('z', 180, degrees=True)
    for side, z in [('L', -0.358), ('R', 0.358)]:
        x = 1.873
        position = [x, float(inside(x, z)), z]
        bumper = turned([(-0.002, 0.013), (0, 0.013), (0.003, 0.011), (0.009, 0.010), (0.011, 0.009)])
        add('Civic_Hood_Buffer_' + side, HOOD, black, position, bumper, inverted)
    spans = np.linspace(-0.224, 0.224, 35)
    shape = np.array(
        [
            [-0.009, -0.001],
            [-0.008, 0.005],
            [-0.004, 0.009],
            [0.006, 0.009],
            [0.010, 0.004],
            [0.010, -0.001],
        ]
    )
    rings = []
    for z in spans:
        x = 2.056 - 0.24 * z * z + shape[:, 0]
        side = np.full_like(x, z)
        depth = shape[:, 1] * np.interp(abs(z), [0.198, 0.224], [1.0, 0.32])
        rings.append(np.column_stack([x, inside(x, side) - depth, side]))
    points, faces, normals = sweep(rings)
    add('Civic_Hood_Front_Seal', HOOD, black, [0, 0, 0], (points, normals, faces))

    # The upper fender aprons run downhill from the cowl to the lamp backs.
    # Their shelves carry the stay foot and turn down into the wheelhouses.
    outer, _, faces = model.read_geometry('Civic_Hood_Panel')
    rim = outside_loop(outer, faces)
    surface = LinearNDInterpolator(outer[:, [0, 2]], outer[:, 1])
    stations = np.linspace(1.13, 1.965, 76)
    for side, sign in [('L', -1), ('R', 1)]:
        # Intersect each transverse section with the actual hem, so even
        # where the fender tapers over a headlamp the apron meets its flange.
        crossing = []
        for station in stations:
            end = np.roll(rim, -1, axis=0)
            a = rim[(rim[:, 0] > station) != (end[:, 0] > station)]
            b = end[(rim[:, 0] > station) != (end[:, 0] > station)]
            section = a + (b - a) * ((station - a[:, 0]) / (b[:, 0] - a[:, 0]))[:, None]
            crossing.append(section[np.argmax(sign * section[:, 2])])
        edge = np.array(crossing)
        width = 0.116 + 0.035 * np.exp(-(((stations - 1.51) / 0.23) ** 2))
        rings = []
        for fraction, depth in [(0, 0.035), (0.18, 0.043), (0.47, 0.045), (0.85, 0.048), (1, 0.058)]:
            z = edge[:, 2] - sign * (0.004 + fraction * width)
            top = surface(stations, z)
            top = np.where(np.isfinite(top), top, edge[:, 1])
            rings.append(np.column_stack([stations, top - depth, z]))

        def strip(rings, normal):
            vertices = np.vstack(rings)
            ids = np.arange(len(vertices)).reshape(len(rings), -1)
            a, b, c, d = (
                ids[:-1, :-1].ravel(),
                ids[:-1, 1:].ravel(),
                ids[1:, 1:].ravel(),
                ids[1:, :-1].ravel(),
            )
            faces = np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
            pts, tris, normals = mesh_normals(vertices, faces, normal)
            return pts, normals, tris

        add(
            'Civic_Body_Shell_Upper_Apron_' + side,
            BODY,
            'Mat_Boost_Blue_Pocket',
            [0, 0, 0],
            strip(rings, np.array([0.0, 1.0, 0.0])),
        )
        upper = rings[-1]
        wall = [upper, upper + [0, -0.016, -sign * 0.014], upper + [0, -0.141, sign * 0.010]]
        add(
            'Civic_Body_Shell_Inner_Apron_' + side,
            BODY,
            'Mat_Aero_Matte_Black',
            [0, 0, 0],
            strip(wall, np.array([0.0, 0.0, -sign])),
        )
