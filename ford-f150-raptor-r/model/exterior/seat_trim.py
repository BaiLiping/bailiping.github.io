"""Front seat upholstery based on the 2024 truck, BaT lot 181611, Recaro close-up."""
import bpy
import bmesh
import numpy as np
from geo import pchip
from mathutils.kdtree import KDTree

LEATHER = 'Mat_Interior_Black_Leather'
SUEDE = 'Mat_Int_AlcantaraPerf'


def stitch_tracks(name, rows, columns):
    """Two rows of short contrast stitches follow the outer bolster seams."""
    vertices, faces = [], []
    rows = np.asarray(rows)
    for column in columns:
        track = rows[:, column, :]
        distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(track, axis=0), axis=1))]
        for start in np.arange(.012, distance[-1] - .013, .010):
            ends = np.column_stack([np.interp([start, start + .0052], distance, track[:, i])
                                    for i in range(3)])
            direction = ends[1] - ends[0]
            direction /= np.linalg.norm(direction)
            normal = np.cross(direction, [0, 1, 0])
            normal /= np.linalg.norm(normal)
            side = np.cross(direction, normal)
            r = .00125
            corners = [r * normal, r * side, -r * normal, -r * side]
            offset = len(vertices)
            vertices.extend(point + corner for point in ends for corner in corners)
            faces.extend([(offset + i, offset + (i + 1) % 4, offset + (i + 1) % 4 + 4, offset + i + 4)
                          for i in range(4)])
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.materials.append(bpy.data.materials['Mat_Int_StitchOrange'])
    ob = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def upholstery(name, rows, materials, uv):
    """Stitched surface strips share their boundaries across changes of fabric."""
    rows = np.asarray(rows)
    nr, nc, _ = rows.shape
    faces = [(j * nc + i, j * nc + (i + 1) % nc, (j + 1) * nc + (i + 1) % nc, (j + 1) * nc + i)
             for j in range(nr - 1) for i in range(nc)]
    faces += [tuple(reversed(range(nc))), tuple((nr - 1) * nc + i for i in range(nc))]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(rows.reshape(-1, 3).tolist(), [], faces)
    # Set consistent outward winding across the closed upholstery envelope.
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    for mat in (LEATHER, SUEDE):
        mesh.materials.append(bpy.data.materials[mat])
    mesh.update()
    tex = mesh.uv_layers.new(name='UVMap')
    for face in mesh.polygons:
        face.use_smooth = True
        if face.index < (nr - 1) * nc:
            j, i = divmod(face.index, nc)
            face.material_index = materials[j][i]
        for loop in face.loop_indices:
            point = mesh.vertices[mesh.loops[loop].vertex_index].co
            tex.data[loop].uv = uv(point)
    ob = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def cushion(yc):
    # The central perforated squab is one continuous saddle. Both bolsters
    # taper together into a rounded nose, with a leather tongue set into it.
    outline_width = pchip([0, .1, .42, .75, .94, 1], [.224, .250, .261, .264, .250, .229])
    top = pchip([0, .15, .42, .72, .90, 1], [1.008, .990, .995, 1.016, 1.025, .997])
    bolster = pchip([0, .12, .38, .70, .90, 1], [.020, .064, .083, .071, .041, .012])
    rows, mats = [], []
    for t in np.linspace(0, 1, 50):
        half = float(outline_width(t))
        tongue = .001 + .094 * np.sqrt(np.clip((t - .59) / .24, 0, 1))
        inside = .137 + .015 * t
        rails = [-half, -half * .90, -inside, -tongue, 0, tongue, inside, half * .90, half]
        widths = np.concatenate([np.linspace(a, b, 5, endpoint=False) for a, b in zip(rails[:-1], rails[1:])] + [[half]])
        row = []
        for y in widths:
            u = abs(y / half)
            x = -.219 + .506 * t - .067 * (t ** 3) * u ** 3 + .025 * (1 - t) ** 3 * u ** 3
            mound = np.exp(-((u - .79) / .22) ** 2)
            seam = .0038 * np.exp(-((abs(y) - inside) / .009) ** 2)
            z = float(top(t)) + float(bolster(t)) * mound + .005 * u * u - seam
            row.append((x, yc + y, z))
        # Round each welt down into the valance. The final return closes the underside.
        x, y, z = row[-1]
        left = row[0]
        bottom = .917 + .015 * t
        row.extend([(x, y + .006, z - .010), (x - .004, y + .005, bottom + .018),
                    (x - .004, y - .010, bottom), (left[0] - .004, left[1] + .010, bottom),
                    (left[0] - .004, left[1] - .005, bottom + .018), (left[0], left[1] - .006, left[2] - .010)])
        row_mats = [int(abs((a + b) / 2) < inside and (abs((a + b) / 2) > tongue or t < .59))
                    for a, b in zip(widths[:-1], widths[1:])]
        rows.append(row)
        mats.append(row_mats + [0] * (len(row) - len(row_mats)))
    return [upholstery('Recaro cushion', rows, mats, lambda p: (p.x * 5.5, p.y * 5.5)),
            stitch_tracks('Cushion welt stitch', rows, [5, 35])]


def backrest(yc):
    # In front view the perforated chest insert tapers to a low V, while
    # the leather yoke under the headrest has its own smaller rounded point.
    outer = pchip([1.01, 1.12, 1.32, 1.48, 1.56, 1.64, 1.685],
                  [.226, .245, .269, .271, .257, .217, .168])
    suede_width = pchip([1.01, 1.15, 1.24, 1.37, 1.50, 1.61, 1.685],
                        [.001, .001, .061, .121, .171, .187, .155])
    shield_width = pchip([1.01, 1.387, 1.414, 1.449, 1.55, 1.685],
                         [.001, .001, .044, .070, .097, .103])
    rows, mats = [], []
    for z in np.linspace(1.01, 1.685, 68):
        half = float(outer(z))
        wing = float(suede_width(z))
        shield = min(float(shield_width(z)), wing * .92)
        rails = [-half, -wing, -shield, 0, shield, wing, half]
        widths = np.concatenate([np.linspace(a, b, 9, endpoint=False) for a, b in zip(rails[:-1], rails[1:])] + [[half]])
        base = -.172 - (z - 1.02) * .305
        row = []
        for y in widths:
            u = abs(y / half)
            mid = np.clip((z - 1.01) / .675, 0, 1)
            shell = .040 + .020 * np.sin(np.pi * mid)
            ridge = (.060 + .036 * np.sin(np.pi * mid)) * np.exp(-((u - .77) / .21) ** 2)
            trim = -.004 * np.exp(-((abs(y) - wing) / .011) ** 2) if z > 1.16 else 0
            x = base + shell + ridge + trim - .029 * u ** 6
            # The existing badge follows a leather patch that is nearly flat.
            if z > 1.50:
                mix = np.clip((shield - abs(y)) / .018, 0, 1) * np.clip((z - 1.50) / .046, 0, 1)
                x = x * (1 - mix) + (base + .049) * mix
            row.append((x, yc + y, z))
        # Wrap each bolster backwards before blending into the seat back.
        row.extend([(base + .012, yc + half + .005, z), (base - .047, yc + half - .005, z),
                    (base - .073, yc + half * .78, z), (base - .088, yc, z),
                    (base - .073, yc - half * .78, z), (base - .047, yc - half + .005, z),
                    (base + .012, yc - half - .005, z)])
        row_mats = [int(shield < abs((a + b) / 2) < wing and z > 1.15) for a, b in zip(widths[:-1], widths[1:])]
        mats.append(row_mats + [0] * (len(row) - len(row_mats)))
        rows.append(row)
    return [upholstery('Recaro backrest', rows, mats, lambda p: (p.y * 5.5, p.z * 5.5)),
            stitch_tracks('Backrest welt stitch', rows, [2, 52])]


def apply():
    for side, center in (('FL', .48), ('FR', -.48)):
        seat = bpy.data.objects['RaptorR_Int_Seat_' + side]
        # Keep the embroidered patch, headrest, switchgear, belt latch and rails.
        # The old cushion/backrest and their floating stitch meshes are replaced together.
        mesh = seat.data
        vis = mesh.attributes['_STUDIO_VISIBILITY']
        exposure = [tuple(v.vector) for v in vis.data]
        tree = KDTree(len(mesh.vertices))
        for v in mesh.vertices:
            tree.insert(v.co, v.index)
        tree.balance()
        bm = bmesh.new()
        bm.from_mesh(mesh)
        fabrics = {i for i, m in enumerate(mesh.materials) if m.name in (LEATHER, SUEDE, 'Mat_Int_StitchOrange')}
        obsolete = [f for f in bm.faces if f.material_index in fabrics and max(v.co.z for v in f.verts) < 1.695]
        bmesh.ops.delete(bm, geom=obsolete, context='FACES')
        bm.to_mesh(mesh)
        bm.free()
        additions = cushion(center) + backrest(center)
        for ob in additions:
            attr = ob.data.attributes.new('_STUDIO_VISIBILITY', 'FLOAT2', 'POINT')
            for i, v in enumerate(ob.data.vertices):
                attr.data[i].vector = exposure[tree.find(v.co)[1]]
        with bpy.context.temp_override(active_object=seat, object=seat, selected_editable_objects=[seat, *additions]):
            bpy.ops.object.join()
