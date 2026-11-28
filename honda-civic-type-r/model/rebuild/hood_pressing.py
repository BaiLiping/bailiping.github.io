"""FK8 bonnet reinforcement and air guide, fitted to the factory panel underside.

Coordinates follow American Honda bulletin AII11806-113 and the bare OEM panel
photograph (Redline TR105, image 960731). Heights are inferred from those views.
"""
import numpy as np
from scipy.spatial import cKDTree


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def smooth(t):
    t = np.clip(t, 0., 1.)
    return t*t*(3.-2.*t)


def stamp(x, z):
    """Draw depth below the inset outer sheet, including four pressed pockets."""
    side = abs(z)
    depth = np.full_like(x, .008)
    # Hemmed reinforcement surrounding the shallow pockets either side of the scoop.
    for lateral, center_x, half_x, half_width, rake in [
        (.212, 1.548, .151, .058, -.24), (.417, 1.498, .145, .051, .12),
    ]:
        dx = (x-center_x) / half_x
        dz = (side-lateral-rake*(x-center_x)) / half_width
        d = (dx**4 + dz**4)**.25
        depth += .022*(1.-smooth((d-1.12)/.57))
        depth -= .021*(1.-smooth((d-.78)/.25))
    depth += .016*np.exp(-((x-(1.746-.100*z*z))/.044)**4)
    # Paired diagonal beads across the crown, then the longitudinal side rails.
    for offset in (0., .072):
        distance = x - (1.54 + .58*(side-.42) + offset)
        depth += .012*np.exp(-(distance/.021)**2)*smooth((side-.45)/.07)*smooth((.72-side)/.05)
    depth += .013*np.exp(-((side-(.617+.045*(x-1.5)))/.037)**4)*smooth((x-1.13)/.09)*smooth((1.92-x)/.11)
    depth += .011*np.exp(-((x-1.91)/.085)**4-(z/.20)**4)
    return depth


def mesh_normals(points, faces, expected):
    tri = points[faces]
    area = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    flipped = area @ expected < 0
    faces = faces.copy()
    faces[flipped] = faces[flipped, ::-1]
    area[flipped] *= -1
    normals = np.zeros_like(points)
    for col in faces.T:
        np.add.at(normals, col, area)
    return points, faces, unit(normals)


def form_hood(base, add):
    points, faces, _ = base
    used = np.unique(faces)
    # The original cap follows the protruding scoop on the outside. Fit the
    # sheet under that scoop from the surrounding metal before stamping it.
    x, y, z = points[used].T
    def poly(x, z):
        x = x-1.55
        return np.column_stack([np.ones_like(x), x, x*x, x**3, z*z, x*z*z, z**4])
    band = (abs(z) > .30) | (x > 1.78) | (x < 1.27)
    coeff = np.linalg.lstsq(poly(x[band],z[band]), y[band], rcond=None)[0]
    def sheet(x, z):
        return poly(x,z) @ coeff

    # Subdivide the existing footprint, keeping the stamped panel joined to
    # the hood's moving hem on all four sides.
    edges = np.sort(faces[:, [[0,1],[1,2],[2,0]]], axis=2)
    unique, inverse, count = np.unique(edges.reshape(-1,2), axis=0, return_counts=True, return_inverse=True)
    boundary = cKDTree(points[np.unique(unique[count==1])][:,[0,2]])
    for _ in range(2):
        mid = inverse.reshape(-1,3) + len(points)
        a,b,c = faces.T
        ab,bc,ca = mid.T
        faces = np.vstack([np.column_stack(section) for section in [(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)]])
        points = np.vstack([points, points[unique].mean(axis=1)])
        edges = np.sort(faces[:, [[0,1],[1,2],[2,0]]], axis=2)
        unique,inverse = np.unique(edges.reshape(-1,2),axis=0,return_inverse=True)

    x, y, z = points.T
    inner = sheet(x,z) + .012 - stamp(x,z)
    fade = smooth(boundary.query(points[:,[0,2]])[0]/.042)
    points[:,1] += fade*(inner-y)
    pivot = 'Pivot_Hood_Assembly'
    down = np.array([0., -1., 0.])
    add('Civic_Hood_Inner_Frame', 'paint', pivot, mesh_normals(points,faces,down))

    # The metal air guide arches across the rear of the scoop opening.
    spans = np.linspace(-.55,.55,91)
    cross = [(0,0),(.004,.006),(.017,.012),(.032,.012),(.043,.007),(.045,.002)]
    rows = []
    for dx, thickness in cross:
        xs = 1.365 - .275*spans**2 + dx
        rows.append(np.column_stack([xs, sheet(xs,spans)-.026-thickness, spans]))
    width = len(spans)
    grid = np.arange(len(cross)*width).reshape(-1,width)
    a,b,c,d = grid[:-1,:-1].ravel(),grid[:-1,1:].ravel(),grid[1:,1:].ravel(),grid[1:,:-1].ravel()
    tris = np.vstack([np.column_stack([a,b,c]),np.column_stack([a,c,d])])
    add('Civic_Hood_Air_Guide', 'Mat_Cabin_Brushed_Aluminum', pivot, mesh_normals(np.vstack(rows),tris,down))

    # Steel latch loop on the front inner cross-member.
    height = float(sheet(np.array([1.978]), np.array([0.]))[0])-.015
    path = np.array([[1.965,height+.004,0],[1.965,height-.021,0],
                     [1.969,height-.027,0],[1.991,height-.027,0],
                     [1.995,height-.021,0],[1.995,height+.004,0]])
    angle = np.arange(12)*2*np.pi/12.
    tangents = unit(np.gradient(path,axis=0))
    across = np.cross(tangents, [0.,0.,1.])
    normals = across[:,None,:]*np.cos(angle)[None,:,None] + np.array([0.,0.,1.])*np.sin(angle)[None,:,None]
    verts = (path[:,None,:]+.0035*normals).reshape(-1,3)
    ids = np.arange(len(verts)).reshape(-1,len(angle))
    a,b,c,d = ids[:-1].ravel(),np.roll(ids[:-1],-1,axis=1).ravel(),np.roll(ids[1:],-1,axis=1).ravel(),ids[1:].ravel()
    add('Civic_Hood_Latch_Loop','chrome',pivot,
        (verts,np.vstack([np.column_stack([a,b,c]),np.column_stack([a,c,d])]),normals.reshape(-1,3)))
