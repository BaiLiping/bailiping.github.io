"""FK8 bonnet reinforcement and air guide, fitted to the factory panel underside.

Coordinates follow American Honda bulletin AII11806-113 and the bare OEM panel
photograph (Redline TR105, image 960731). Heights are inferred from those views.
"""
from functools import lru_cache

import cv2
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.ndimage import gaussian_filter1d

import outlines
import p24


PIVOT = 'Pivot_Hood_Assembly'
HALF_GAP = 0.0018


@lru_cache(maxsize=1)
def plan_distance():
    # Follow the outer shut line in Honda's P24 plan. The creases and scoop
    # all lie within it. Region 23 ahead of the seam is the fixed grille trim.
    mask = outlines.region_mask('top', [3, 9, 13, 24, 25, 106, 123], sym=True)
    contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    panel = np.zeros_like(mask, dtype='uint8')
    cv2.drawContours(panel, [max(contours, key=cv2.contourArea)], -1, 1, cv2.FILLED)
    return p24.sdf2d(panel != 0) * 0.001 - 0.002


def signed_distance(points):
    x, y, z = points.T
    u, v = p24.to_px('top', x, y, z)
    plan = p24.sample(plan_distance(), u, v)
    # Exclude wheel arches and floor on the same vertical rays.
    return np.maximum(plan, 0.860 - 0.18 * (x - 1.10) - y)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def stamp(x, z):
    """Draw depth below the inset outer sheet, including four pressed pockets."""
    side = abs(z)
    depth = np.full_like(x, 0.008)
    # Hemmed reinforcement surrounding the shallow pockets either side of the scoop.
    for lateral, center_x, half_x, half_width, rake in [
        (0.212, 1.548, 0.151, 0.058, -0.24),
        (0.417, 1.498, 0.145, 0.051, 0.12),
    ]:
        dx = (x - center_x) / half_x
        dz = (side - lateral - rake * (x - center_x)) / half_width
        d = (dx**4 + dz**4) ** 0.25
        depth += 0.022 * (1.0 - smooth((d - 1.12) / 0.57))
        depth -= 0.021 * (1.0 - smooth((d - 0.78) / 0.25))
    depth += 0.016 * np.exp(-(((x - (1.746 - 0.100 * z * z)) / 0.044) ** 4))
    # Paired diagonal beads across the crown, then the longitudinal side rails.
    for offset in (0.0, 0.072):
        distance = x - (1.54 + 0.58 * (side - 0.42) + offset)
        depth += (
            0.012
            * np.exp(-((distance / 0.021) ** 2))
            * smooth((side - 0.45) / 0.07)
            * smooth((0.72 - side) / 0.05)
        )
    depth += (
        0.013
        * np.exp(-(((side - (0.617 + 0.045 * (x - 1.5))) / 0.037) ** 4))
        * smooth((x - 1.13) / 0.09)
        * smooth((1.92 - x) / 0.11)
    )
    depth += 0.011 * np.exp(-(((x - 1.91) / 0.085) ** 4) - (z / 0.20) ** 4)
    return depth


def mesh_normals(points, faces, expected):
    tri = points[faces]
    area = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flipped = area @ expected < 0
    faces = faces.copy()
    faces[flipped] = faces[flipped, ::-1]
    area[flipped] *= -1
    normals = np.zeros_like(points)
    for col in faces.T:
        np.add.at(normals, col, area)
    return points, faces, unit(normals)


def outside_loop(points, faces):
    """Walk the skin's outermost edge, excluding the separate scoop opening."""
    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    adjacent = {}
    for a, b in edges[counts == 1]:
        adjacent.setdefault(a, []).append(b)
        adjacent.setdefault(b, []).append(a)
    start = max(adjacent, key=lambda i: points[i, 2])
    ids = [start]
    previous, current = start, adjacent[start][0]
    while current != start:
        ids.append(current)
        a, b = adjacent[current]
        previous, current = current, b if a == previous else a
    loop = points[ids]
    x, z = loop[:, 0], loop[:, 2]
    if np.sum(x * np.roll(z, -1) - z * np.roll(x, -1)) < 0:
        loop = loop[::-1]
    return loop


def sweep(rings, rows=None):
    """Join consecutive cross sections along an ordered, closed perimeter."""
    rings = np.array(rings)
    count = rings.shape[1]
    i = np.arange(count)
    if rows is not None:
        i = i[rows & np.roll(rows, -1)]
    j = (i + 1) % count
    faces = []
    for level in range(len(rings) - 1):
        offset = level * count
        faces.extend(
            [
                offset + np.column_stack([i, j, i + count]),
                offset + np.column_stack([j, j + count, i + count]),
            ]
        )
    faces = np.vstack(faces)
    points = rings.reshape(-1, 3)
    normals = np.zeros_like(points)
    cross = np.cross(
        points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]]
    )
    for indices in faces.T:
        np.add.at(normals, indices, cross)
    return points, faces, unit(normals)


def form_hood(skin, faces, add, clear):
    """Return the outer skin around the full rim, including the areas above both wheels."""
    edge = outside_loop(skin, faces)
    plan = edge[:, [0, 2]]
    center = np.array([1.59, 0.0])
    outward = unit(plan - center)

    def ring(spread, depth):
        row = edge.copy()
        row[:, [0, 2]] += spread * outward
        row[:, 1] -= depth
        return row

    # The first ring uses the actual cut vertices, including their stamped
    # heights. The hem folds underneath to meet the pressed inner sheet.
    hem = [edge, ring(-0.0015, 0.003), ring(-0.006, 0.005), ring(-0.014, 0.0065)]
    add('Civic_Hood_Jamb', 'paint', PIVOT, sweep(hem))
    # The body flange is press-formed. Sample it at even distances before
    # smoothing so small triangles from the outer panel cut do not corrugate it.
    closed = np.vstack([edge, edge[0]])
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    stations = np.arange(0.0, arc[-1], 0.005)
    pressed = np.column_stack([np.interp(stations, arc, axis) for axis in closed.T])
    pressed = gaussian_filter1d(pressed, 1.35, axis=0, mode='wrap')
    radial = unit(pressed[:, [0, 2]] - center)

    def body_ring(spread, depth):
        row = pressed.copy()
        row[:, [0, 2]] += spread * radial
        row[:, 1] -= depth
        return row

    flange = [
        body_ring(0.004, 0.0009),
        body_ring(0.003, 0.012),
        body_ring(-0.009, 0.022),
        body_ring(-0.036, 0.025),
        body_ring(-0.046, 0.042),
        body_ring(-0.046, 0.056),
    ]
    add('Civic_Body_Shell_Hood_Jamb', 'paint', '', sweep([r[::-1] for r in flange]))
    rear = (pressed[:, 0] < 1.26) & (abs(pressed[:, 2]) < 0.70)
    cowl = [
        body_ring(-0.013, 0.022),
        body_ring(-0.016, 0.015),
        body_ring(-0.021, 0.013),
        body_ring(-0.026, 0.016),
        body_ring(-0.028, 0.025),
    ]
    add('Civic_Hood_Weatherstrip_Cowl', 'black', '', sweep([r[::-1] for r in cowl], rear[::-1]))
    clear.append(
        {'a': [0, 1, 0], 'e1': [1, 0, 0], 'e2': [0, 0, 1], 'poly': plan[::4].tolist(), 'margin': 0.050}
    )

    x, y, z = skin.T

    def poly(x, z):
        x = x - 1.55
        return np.column_stack([np.ones_like(x), x, x * x, x**3, z * z, x * z * z, z**4])

    band = (abs(z) > 0.29) | (x > 1.78) | (x < 1.29)
    coeff = np.linalg.lstsq(poly(x[band], z[band]), y[band], rcond=None)[0]

    def sheet(x, z):
        return poly(x, z) @ coeff

    def inner_height(x, z):
        return sheet(x, z) - 0.012 - stamp(x, z)

    lip = hem[-1]
    boundary = lip[:, [0, 2]]
    polygon = np.ascontiguousarray(boundary, dtype=np.float32)
    axes = [np.arange(low, high, 0.009) for low, high in zip(boundary.min(0), boundary.max(0))]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 2)

    def inside(points):
        return np.array([cv2.pointPolygonTest(polygon, tuple(p), False) >= 0 for p in points])

    tree = cKDTree(boundary)
    grid = grid[inside(grid) & (tree.query(grid)[0] > 0.004)]
    gap, near = tree.query(grid)
    x, z = grid.T
    height = inner_height(x, z)
    fade = smooth(gap / 0.038)
    height = (1 - fade) * lip[near, 1] + fade * height
    points = np.vstack([lip, np.column_stack([x, height, z])])
    coords = points[:, [0, 2]]
    triangles = Delaunay(coords).simplices
    triangles = triangles[inside(coords[triangles].mean(axis=1))]
    down = np.array([0.0, -1.0, 0.0])
    add('Civic_Hood_Inner_Frame', 'paint', PIVOT, mesh_normals(points, triangles, down))

    # The pressed aluminum air guide has a folded section, capped ends and
    # upper and lower returns that bed into the adjacent inner pressing.
    spans = np.linspace(-0.585, 0.585, 99)
    mid = 1.391 - 0.35 * spans**2
    section = [
        (0, 0.016),
        (0.003, 0.036),
        (0.015, 0.051),
        (0.056, 0.054),
        (0.067, 0.046),
        (0.068, 0.028),
        (0.062, 0.015),
    ]
    rings = []
    for z, x in zip(spans, mid):
        xx = x + np.array([s[0] for s in section])
        zz = np.full_like(xx, z)
        # The guide bridges the reinforcing pockets as a smooth, independent
        # stamping. The lip returns into the sheet across its mounting flange.
        rings.append(np.column_stack([xx, sheet(xx, zz) - [s[1] for s in section], zz]))
    verts, tris, _ = sweep(rings)
    n = len(section)
    ends = np.array([[0, j, j + 1] for j in range(1, n - 1)])
    tris = np.vstack([tris, ends[:, ::-1], ends + (len(rings) - 1) * n])
    tri = verts[tris]
    normals = np.zeros_like(verts)
    weights = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for ids in tris.T:
        np.add.at(normals, ids, weights)
    add('Civic_Hood_Air_Guide', 'Mat_Cabin_Brushed_Aluminum', PIVOT, (verts, tris, unit(normals)))

    x = 1.973
    height = inner_height(np.array([x]), np.array([0.0]))[0]
    path = np.array(
        [
            [x - 0.014, height + 0.004, 0],
            [x - 0.014, height - 0.023, 0],
            [x - 0.009, height - 0.030, 0],
            [x + 0.010, height - 0.030, 0],
            [x + 0.015, height - 0.023, 0],
            [x + 0.015, height + 0.004, 0],
        ]
    )
    angles = np.arange(12) * 2 * np.pi / 12.0
    tangents = unit(np.gradient(path, axis=0))
    across = np.cross(tangents, [0.0, 0.0, 1.0])
    normals = (
        across[:, None, :] * np.cos(angles)[None, :, None]
        + np.array([0.0, 0.0, 1.0]) * np.sin(angles)[None, :, None]
    )
    verts, tris, _ = sweep(path[:, None, :] + 0.0035 * normals)
    add('Civic_Hood_Latch_Loop', 'chrome', PIVOT, (verts, tris, normals.reshape(-1, 3)))

    radiator_cover(pressed[:, [0, 2]], pressed, add)
    cowl_cover(pressed, add)


def radiator_cover(plan, edge, add):
    """Sight shield directly behind the fixed garnish, with an opening for the hood latch."""
    nose = plan[:, 0] > 1.75
    order = np.argsort(plan[nose, 1])
    x, y, z = edge[nose][order].T
    across = np.r_[np.linspace(-0.642, -0.043, 27), 0.0, np.linspace(0.043, 0.642, 27)]
    front = np.interp(across, z, x) - 0.028
    top = np.interp(across, z, y) - 0.028
    rows = np.array([0.0, 0.025, 0.055, 0.084, 0.124, 0.159, 0.185])
    xx = front[None, :] - rows[:, None]
    zz = np.broadcast_to(across, xx.shape)
    yy = (
        top[None, :]
        + 0.070 * rows[:, None]
        - 0.011 * (rows[:, None] / 0.185) ** 4
        - 0.014 * (abs(zz) / 0.642) ** 6
    )
    points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    ids = np.arange(len(points)).reshape(xx.shape)
    a, b, c, d = ids[:-1, :-1], ids[:-1, 1:], ids[1:, 1:], ids[1:, :-1]
    open_latch = (
        (rows[:-1, None] >= 0.055)
        & (rows[:-1, None] < 0.124)
        & (abs(0.5 * (across[:-1] + across[1:])) < 0.043)
    )
    a, b, c, d = [corners[~open_latch] for corners in (a, b, c, d)]
    tris = np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    add(
        'Civic_Eng_Radiator_Sight_Shield',
        'Mat_FK8_Fog_Satin',
        '',
        mesh_normals(points, tris, np.array([0.0, 1.0, 0.0])),
    )


def cowl_cover(edge, add):
    """Black cowl molding under the concave trailing edge, ahead of the windscreen."""
    z = np.linspace(-0.693, 0.693, 111)
    row = edge[edge[:, 0] < 1.26]
    row = row[np.argsort(row[:, 2])]
    x = np.interp(z, row[:, 2], row[:, 0])
    y = np.interp(z, row[:, 2], row[:, 1])
    rings = []
    for dx, drop in [
        (-0.009, 0.012),
        (0.007, 0.010),
        (0.028, 0.021),
        (0.081, 0.026),
        (0.106, 0.032),
        (0.111, 0.050),
    ]:
        rings.append(np.column_stack([x + dx, y - drop, z]))
    points = np.vstack(rings)
    ids = np.arange(len(points)).reshape(-1, len(z))
    a, b, c, d = ids[:-1, :-1].ravel(), ids[:-1, 1:].ravel(), ids[1:, 1:].ravel(), ids[1:, :-1].ravel()
    tris = np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    add(
        'Civic_Windshield_Cowl_Tray', 'black', '', mesh_normals(points, tris, np.array([0.0, 1.0, 0.0]))
    )
