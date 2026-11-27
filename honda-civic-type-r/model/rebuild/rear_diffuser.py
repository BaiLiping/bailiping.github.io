"""Fit the FK8 diffuser section and triple outlet to the low rear photo and Honda frame 28."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def smooth(value):
    t = np.clip(value, 0, 1)
    return t*t*(3 - 2*t)


def fit_couplings(points, normals, faces):
    """Fit the three unpolished branch pipes behind the sleeves to their bores."""
    locations, vertex_id = np.unique(points, axis=0, return_inverse=True)
    ids = vertex_id[faces]
    edges = coo_matrix((np.ones(ids.size), (ids.ravel(), ids[:, [1, 2, 0]].ravel())),
                       shape=(len(locations), len(locations)))
    count, labels = connected_components(edges, directed=False)
    for part in range(count):
        mask = labels[vertex_id] == part
        branch = points[mask].copy()
        if branch[:, 0].min() >= -2.1:
            continue
        # These three swept pipe segments have one ring at each end. Keep the
        # upstream joint inserted into the muffler and seat the terminal ring
        # concentrically inside the polished sleeve, clear of the visible lip.
        outlet = branch[:, 0] < -2.05
        for ring, scale, shift in [(outlet, .62, [-.006, 0]), (~outlet, .77, [0, 0])]:
            yz = branch[ring, 1:]
            center = (yz.max(axis=0) + yz.min(axis=0))/2
            branch[ring, 1:] = center + (yz-center)*scale + shift
        points[mask] = branch
        normals[mask, 1:] /= np.where(outlet, .62, .77)[:, None]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)


def fit_outlets(model):
    # The outer ring envelope is 77 px across Honda's frame-28 close-up, versus
    # 92 px on the old mesh. Raise the mouths by 25 mm and keep the taller projected bore.
    # Blend out at the concealed muffler connection so the pipes stay attached.
    for part in ('Tips', 'Bores', 'Inner_Sleeves', 'Muffler'):
        name = 'Civic_Exhaust_' + part
        points, normals, faces = model.read_geometry(name)
        x, y, z = points.T
        t = np.clip((-1.926 - x) / 0.145, 0, 1)
        weight = smooth(t)
        derivative = -6*t*(1-t)/0.145
        scale = 1 - 0.17*weight
        dy = 0.025 - 0.10*(y - 0.086)
        result = np.column_stack([x, y + dy*weight, z*scale])
        # Transform the authored normals by the inverse-transpose Jacobian.
        normal = normals.copy()
        normal[:, 1] /= 1 - 0.10*weight
        normal[:, 2] /= scale
        normal[:, 0] -= derivative * (normal[:, 1]*dy - normal[:, 2]*0.17*z)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        if part == 'Muffler':
            fit_couplings(result, normal, faces)
        model.replace_geometry(name, result, normal, faces, uv=model.read_uv(name))


def strake(z, outer):
    # Horizontal slices through one tapered moulded fin, from the undertray to
    # the lip. The outer two support the upper shelf as in the low-angle photo.
    # Each station gives height, rear edge, forward attachment and half thickness.
    aft = -2.242 if outer else -2.249
    stations = [(0.259, aft+0.044, -1.98, 0.005),
                (0.278, aft-0.007, -2.12, 0.007),
                (0.314, aft, -2.13, 0.0068),
                (0.408, aft+0.002, -2.14, 0.0055),
                (0.454, aft+0.006, -2.17, 0.005)]
    if outer:
        stations += [(0.481, aft+0.017, -2.16, 0.0055), (0.519, aft+0.023, -2.16, 0.005)]
    rings = []
    for y, rear, front, thickness in stations:
        rings.append(np.array([[rear, y, z-thickness*0.4], [rear, y, z+thickness*0.4],
                               [rear+0.004, y, z+thickness], [front, y, z+thickness],
                               [front, y, z-thickness], [rear+0.004, y, z-thickness]]))
    triangles = []
    for lower, upper in zip(rings, rings[1:]):
        for j in range(len(lower)):
            k = (j+1) % len(lower)
            triangles.extend([[lower[j], lower[k], upper[k]], [lower[j], upper[k], upper[j]]])
    # The winding above follows the cross section from the rear towards +z.
    for ring in (rings[0][::-1], rings[-1]):
        for j in range(1, len(ring)-1):
            triangles.append([ring[0], ring[j], ring[j+1]])
    triangles = np.array(triangles)
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    points = triangles.reshape(-1, 3)
    return points, np.repeat(normals, 3, axis=0), np.arange(len(points)).reshape(-1, 3)


def roll_diffuser_panel(model, name):
    points, normals, faces = model.read_geometry(name)
    # The corner trims are a second fitted skin 3 mm above the main panel.
    # All three need exactly the same section at their overlapping boundaries.
    def offset(p):
        x, y, z = p.T
        lip = np.interp(abs(z), [0, .30, .32, .34, .36, .38], [.452, .452, .449, .442, .431, .418])
        lower = smooth((lip-.007-y)/.028) * smooth((y-.25)/.038)
        upper = smooth((y-lip-.007)/.021) * smooth((.53-y)/.037)
        inside = smooth((.362-abs(z))/.038) * smooth((-2.12-x)/.085)
        return inside * (.044*lower + .029*upper)

    result = points.copy()
    result[:, 0] += offset(points)
    gradients = []
    for direction in np.eye(3)*.0005:
        gradients.append((offset(points+direction)-offset(points-direction))/.001)
    dx, dy, dz = gradients
    normal = normals.copy()
    normal[:, 0] /= 1 + dx
    normal[:, 1] -= dy*normal[:, 0]
    normal[:, 2] -= dz*normal[:, 0]
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    return result, normal, faces


def form_diffuser(model):
    for name in ('Civic_Rear_Bumper_Corner_Trim_L', 'Civic_Rear_Bumper_Corner_Trim_R',
                 'Civic_Rear_Bumper_Red_Strips', 'Civic_Rear_Bumper'):
        model.replace_geometry(name, *roll_diffuser_panel(model, name))
    name = 'Civic_Rear_Diffuser'
    result, normal, faces = roll_diffuser_panel(model, name)
    vertex_blocks, normal_blocks, index_blocks = [result], [normal], [faces]
    length = len(result)
    for z in [-0.309, -0.207, 0.207, 0.309]:
        p, n, f = strake(z, abs(z) > 0.3)
        vertex_blocks.append(p)
        normal_blocks.append(n)
        index_blocks.append(f + length)
        length += len(p)
    model.replace_geometry(name, np.vstack(vertex_blocks), np.vstack(normal_blocks), np.vstack(index_blocks))


def refine_rear_diffuser(model):
    fit_outlets(model)
    form_diffuser(model)
