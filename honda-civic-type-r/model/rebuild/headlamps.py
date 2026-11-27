"""Fit the FK8 reflector cells and upper running-light row to the traced lamp shell."""

import numpy as np
from scipy.spatial import cKDTree


# The clear aperture in Honda colour-selector frame 07, traced on the original lamp
# skin. Stations run from grille to fender in u = |z| - x. Heights are in metres.
STATIONS = [-1.609, -1.585, -1.540, -1.460, -1.360, -1.260, -1.160, -1.105, -1.075, -1.056]
LOWER = [0.649, 0.630, 0.631, 0.636, 0.650, 0.666, 0.687, 0.702, 0.715, 0.735]
UPPER = [0.663, 0.679, 0.692, 0.703, 0.718, 0.731, 0.742, 0.748, 0.750, 0.747]


def recess(points):
    """Sink the backing behind the new cells, blending to the original bezel."""
    x, y, z = points.T
    u = abs(z) - x
    margin = np.minimum.reduce([u - STATIONS[0], STATIONS[-1] - u,
                                y - np.interp(u, STATIONS, LOWER),
                                np.interp(u, STATIONS, UPPER) - y])
    depth = 0.020 * np.clip((margin + 0.006) / 0.012, 0, 1)
    result = points.copy()
    result[:, 0] -= depth / 2
    result[:, 2] -= np.sign(z) * depth / 2
    return result


def corner_surface(points, side):
    """Map (diagonal station, height) to the original swept corner surface."""
    x, y, z = points.T
    uv = np.column_stack([abs(z) - x, y])
    width = abs(z) + x
    tree = cKDTree(uv)

    def sample(u, height, lift):
        # Interpolate the swept lens surface before adding optical depth. The
        # original panels are sampled at different densities across their seams.
        target = [u, height]
        _, idx = tree.query(target, k=18)
        delta = uv[idx] - target
        plane = np.column_stack([delta, np.ones(len(idx))])
        fitted = np.linalg.lstsq(plane, width[idx], rcond=None)[0][2] + lift
        return [(fitted - u) / 2, height, side * (fitted + u) / 2]

    return sample


def make_emitters(sample, side, kind):
    triangles = []
    # The chamfered perimeter of one reflector, in cell-local coordinates.
    corners = np.array([[-0.78, -1], [0.78, -1], [1, -0.73], [1, 0.70],
                        [0.76, 1], [-0.78, 1], [-1, 0.70], [-1, -0.73]])
    lower_row = kind == 'reflectors'
    stations = np.linspace(STATIONS[0], STATIONS[-1], 9) if lower_row else np.linspace(-1.567, -1.065, 15)
    # The lower row has open reflector cups. The narrower upper row has convex
    # fluted light guides with a fine black break between adjacent lenses.
    low, high = (0.03, 0.68) if lower_row else (0.74, 0.99)
    for a, b in zip(stations[:-1], stations[1:]):
        sill = np.interp([a, b], STATIONS, LOWER)
        top = np.interp([a, b], STATIONS, UPPER)

        def ring(scale, lift, offset=0):
            vertices = []
            for s, t in corners * scale:
                u = (a + b) / 2 + (b - a) * (0.451 * s + offset + 0.054 * t)
                # Each individual stamped cup has a straight sill. Following a
                # station kink across a narrow chamfer would fold its triangles.
                floor = np.interp(u, [a, b], sill)
                span = np.interp(u, [a, b], top) - floor
                h = floor + span * ((high + low) / 2 + (high - low) * t / 2)
                vertices.append(sample(u, h, lift))
            return vertices

        if lower_row:
            rings = [ring(1, 0.0015), ring(0.90, 0.003), ring(0.28, -0.008, -0.11)]
        else:
            rings = [ring(1, 0.0015), ring(0.69, 0.005), ring(0.23, 0.007)]
        for outer, inner in zip(rings, rings[1:]):
            for j in range(len(corners)):
                k = (j + 1) % len(corners)
                triangles.extend([[outer[j], outer[k], inner[k]], [outer[j], inner[k], inner[j]]])
        center = np.mean(rings[-1], axis=0)
        for j, vertex in enumerate(rings[-1]):
            triangles.append([vertex, rings[-1][(j + 1) % len(corners)], center])

    triangles = np.array(triangles)
    # Give each polished reflector facet its own normal so it catches light
    # independently as the viewer orbits. Fix winding on the mirrored assembly.
    if side > 0:
        triangles = triangles[:, ::-1]
    edges = triangles[:, 1:] - triangles[:, :1]
    normals = np.cross(edges[:, 0], edges[:, 1])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    points = triangles.reshape(-1, 3)
    return points, np.repeat(normals, 3, axis=0), np.arange(len(points)).reshape(-1, 3)


def refine_headlamps(model):
    document = model.document
    reflective = len(document['materials'])
    document['materials'].append({
        'name': 'Mat_FK8_Headlamp_Reflector_Aluminum',
        'pbrMetallicRoughness': {'baseColorFactor': [0.72, 0.75, 0.77, 1],
                                 'metallicFactor': 0.83, 'roughnessFactor': 0.19},
    })
    for label, side in [('L', -1), ('R', 1)]:
        names = {part: f'Civic_Headlamp_{part}_{label}'
                 for part in ('Housing', 'Reflector', 'LED', 'Optics', 'Optics_Cells')}
        parts = {part: model.read_geometry(name) for part, name in names.items()}
        sample = corner_surface(np.vstack([mesh[0] for mesh in parts.values()]), side)

        # Keep the lamp's five traced shell sections as a continuous black bowl.
        # Reuse the original optic and LED nodes for their respective inserts.
        for part in ('Housing', 'Reflector'):
            points, normals, faces = parts[part]
            model.replace_geometry(names[part], recess(points), normals, faces)
        points, normals, faces = [], [], []
        offset = 0
        for part in ('LED', 'Optics', 'Optics_Cells'):
            p, n, f = parts[part]
            points.extend(recess(p))
            normals.extend(n)
            faces.extend(f + offset)
            offset += len(p)
        model.replace_geometry(names['Optics_Cells'], np.array(points), np.array(normals), np.array(faces))
        model.replace_geometry(names['Optics'], *make_emitters(sample, side, 'reflectors'))
        model.replace_geometry(names['LED'], *make_emitters(sample, side, 'running lights'))
        model.primitive(names['Optics'])['material'] = reflective
