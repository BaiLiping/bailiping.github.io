"""Form the four openings beside each fog lamp using Honda frame 07's vent outline."""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from headlamps import corner_surface


# A cross section through the slanted slot in corner coordinates (|z|-x, y).
# The three narrow painted vanes follow the row heights visible in the Honda frame.
BOTTOM, TOP = 0.348, 0.574
HALF_WIDTH = 0.061
VANE_HEIGHTS = [0.406, 0.462, 0.517]


def center(y):
    return -1.088 - 0.56 * (y - 0.36)


def lower_skin(points, side):
    x, y, z = points.T
    s = (abs(z) - x - center(y)) / HALF_WIDTH
    t = (2 * y - TOP - BOTTOM) / (TOP - BOTTOM)
    radius = (s**4 + t**4)**0.25
    # Clear the inside of the pocket in all adjoining shell sections, including
    # the coarse bumper triangles crossing beneath the new mouth.
    depth = 0.064 * np.clip((1.26 - radius) / 0.38, 0, 1)
    depth[(z * side < 0.70) | (x < 1.82)] = 0
    result = points.copy()
    result[:, 0] -= depth / 2
    result[:, 2] -= side * depth / 2
    return result


def pocket(sample, side):
    paint, dark = [], []
    # Sampling each side of the rounded rectangle avoids long
    # diagonal triangles where the end caps join the upright side walls.
    path = []
    for u in np.linspace(-1, 1, 19, endpoint=False):
        path.append((u, -1))
    for v in np.linspace(-1, 1, 40, endpoint=False):
        path.append((1, v))
    for u in np.linspace(1, -1, 19, endpoint=False):
        path.append((u, 1))
    for v in np.linspace(1, -1, 40, endpoint=False):
        path.append((-1, v))
    outline = np.asarray(path)
    outline /= np.sum(outline**4, axis=1, keepdims=True)**0.25

    def vertex(s, t, lift):
        y = (TOP + BOTTOM + (TOP - BOTTOM) * t) / 2
        return sample(center(y) + HALF_WIDTH * s, y, lift)

    def ring(sx, sy, lift):
        return [vertex(s * sx, t * sy, lift) for s, t in outline]

    def bridge(a, b, triangles):
        for j in range(len(a)):
            k = (j+1) % len(a)
            triangles.extend([[a[j], a[k], b[k]], [a[j], b[k], b[j]]])

    lip = ring(1.06, 1.03, 0.0015)
    mouth = ring(0.93, 0.975, -0.002)
    bevel = ring(0.80, 0.94, -0.011)
    back = ring(0.72, 0.90, -0.038)
    bridge(lip, mouth, paint)
    bridge(mouth, bevel, paint)
    bridge(bevel, back, dark)
    middle = vertex(0, 0, -0.041)
    for j in range(len(back)):
        dark.append([back[j], back[(j+1) % len(back)], middle])

    for y in VANE_HEIGHTS:
        # Aerofoil cross section: a rounded painted leading edge with its broad
        # surface falling back into the channel. The ends tuck into the sidewalls.
        sections = []
        for s in np.linspace(-0.91, 0.91, 13):
            front = y + 0.004 * s
            sections.append([sample(center(h) + HALF_WIDTH * s, h, lift)
                             for h, lift in ((front - 0.0025, -0.019),
                                             (front - 0.0020, -0.006),
                                             (front + 0.0005, -0.003),
                                             (front + 0.0034, -0.005),
                                             (front + 0.0055, -0.030))])
        for a, b in zip(sections, sections[1:]):
            for j in range(len(a)-1):
                paint.extend([[a[j], b[j], b[j+1]], [a[j], b[j+1], a[j+1]]])

    def mesh(triangles):
        triangles = np.array(triangles)
        if side > 0:
            triangles = triangles[:, ::-1]
        normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        points = triangles.reshape(-1, 3)
        return points, np.repeat(normals, 3, axis=0), np.arange(len(points)).reshape(-1, 3)

    return mesh(paint), mesh(dark)


def fog_blade(points, faces, side):
    """Sweep a solid rail through the original P24 blade footprint (front region 54)."""
    # This diagonal coordinate keeps sections perpendicular to the blade as its outer
    # tip turns upward. Intersect the actual mesh so the new lens-shaped section
    # has the same front outline on both sides of the bend.
    u = abs(points[:, 2]) + .65 * (points[:, 1] - .43)
    edges = np.unique(np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0)
    line = points[edges]
    ua, ub = u[edges].T
    samples = np.linspace(u.min()+.00015, u.max()-.00015, 96)
    sections = []
    for s in samples:
        hit = (np.minimum(ua, ub) <= s) & (s < np.maximum(ua, ub))
        t = (s-ua[hit])/(ub[hit]-ua[hit])
        ps = line[hit, 0] + t[:, None] * (line[hit, 1]-line[hit, 0])
        sections.append([ps[:, 1].min(), ps[:, 1].max(), ps[:, 0].max()])
    low, high, front = gaussian_filter1d(sections, 1.15, axis=0, mode='nearest').T
    # The photographed lip is proud of the black pod. Give it a rounded nose and
    # bury its return in the backing, so the two surfaces cannot flicker together.
    angle = np.arange(28)*2*np.pi/28
    nose = np.cos(angle)
    rings = []
    for s, bottom, top, x in zip(samples, low, high, front):
        y = (bottom+top)/2 + (top-bottom)/2*np.sin(angle)
        z = side * (s-.65*(y-.43))
        rings.append(np.column_stack([x-.013+.025*nose, y, z]))
    p = np.vstack(rings)
    grid = np.arange(len(p)).reshape(len(rings), -1)
    a, b = grid[:-1].ravel(), grid[1:].ravel()
    c, d = np.roll(grid[1:], -1, axis=1).ravel(), np.roll(grid[:-1], -1, axis=1).ravel()
    tri = [np.column_stack([a, d, c]), np.column_stack([a, c, b])]
    for cap in (grid[0, ::-1], grid[-1]):
        tri.append(np.column_stack([np.full(len(cap)-2, cap[0]), cap[1:-1], cap[2:]]))
    triangles = np.vstack(tri)
    if side < 0:
        triangles = triangles[:, ::-1]
    normals = np.zeros_like(p)
    v = p[triangles]
    weights = np.cross(v[:, 1]-v[:, 0], v[:, 2]-v[:, 0])
    for col in triangles.T:
        np.add.at(normals, col, weights)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return p, normals, triangles


def refine_bumper_vents(model):
    fill = 'Civic_Front_Fascia_Trim_Fill'
    fill_points, fill_normals, fill_faces = model.read_geometry(fill)
    for suffix, side in [('L', -1), ('R', 1)]:
        blade = 'Civic_Front_Bumper_Painted_Blades_' + suffix
        rail, _, rail_faces = model.read_geometry(blade)
        model.replace_geometry(blade, *fog_blade(rail, rail_faces, side))
        slots = 'Civic_Front_Bumper_Vents_' + suffix
        bumper = 'Civic_Front_Bumper_Upper_' + suffix
        points, normals, faces = model.read_geometry(bumper)
        # Two disconnected islands of the original bumper cut fill the fog pod
        # immediately under the rail. Keep their shared edges and put these
        # faces in the satin-black pod with the surrounding faces.
        x, y, z = points.T
        in_pod = ((abs(z) > .548) & (abs(z) < .645) & (y > .420) & (y < .454)
                  & (x > 2.13) & (x < 2.214))
        pod_faces = in_pod[faces].all(axis=1)
        if np.any(pod_faces):
            pod = 'Civic_Front_Bumper_Fog_Surrounds_' + suffix
            shell, shell_normals, shell_faces = model.read_geometry(pod)
            ids, moved = np.unique(faces[pod_faces], return_inverse=True)
            model.replace_geometry(pod, np.vstack([shell, points[ids]]),
                                   np.vstack([shell_normals, normals[ids]]),
                                   np.vstack([shell_faces, len(shell)+moved.reshape(-1, 3)]))
            faces = faces[~pod_faces]
        old_slots, slot_normals, slot_faces = model.read_geometry(slots)
        zone = np.vstack([points, old_slots])
        mask = (zone[:, 0] > 1.85) & (zone[:, 1] < 0.62) & (abs(zone[:, 2]) > 0.735)
        sample = corner_surface(zone[mask], side)
        (lip, lip_normals, lip_faces), channel = pocket(sample, side)
        model.replace_geometry(slots, *channel)
        model.replace_geometry(bumper,
            np.vstack([lower_skin(points, side), lower_skin(old_slots, side), lip]),
            np.vstack([normals, slot_normals, lip_normals]),
            np.vstack([faces, slot_faces + len(points), lip_faces + len(points) + len(old_slots)]))
        fill_points = lower_skin(fill_points, side)
    model.replace_geometry(fill, fill_points, fill_normals, fill_faces)
