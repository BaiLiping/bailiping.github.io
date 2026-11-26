"""Bounded cabin refinements for the structurally intact 2024 Raptor R mesh.

Call apply() after importing raptor-prefit-decoded.glb into Blender.  Coordinates
are Blender X forward, Y left, Z up.  Mesh coordinates are edited in world space
and written through each object's existing inverse matrix, preserving every
door pivot, hierarchy, and object transform.

The modest handle/mirror changes and lower rear-window corner radius are visual
estimates from frame-by-frame/frames/side-profile-left-photo.webp.  They are not
claimed as factory part dimensions.  The cab envelope, windshield, roof height,
door perimeter, and mirror width are retained.
"""

import math

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


REVISION = "raptor_cabin_2024_v1"


def _world_vertices(obj):
    return np.array([tuple(obj.matrix_world @ v.co) for v in obj.data.vertices])


def _write_world_vertices(obj, coordinates):
    # Some imported components share meshes.  Preserve every untargeted instance.
    if obj.data.users > 1:
        obj.data = obj.data.copy()
    inverse = obj.matrix_world.inverted()
    for vertex, coordinate in zip(obj.data.vertices, coordinates):
        vertex.co = inverse @ Vector(coordinate)
    obj.data.update()


def _replace_mesh(obj, coordinates, faces, closed=False):
    original = obj.data
    materials = list(original.materials)
    inverse = obj.matrix_world.inverted()
    mesh = bpy.data.meshes.new(obj.data.name + "_CabinCorrection")
    mesh.from_pydata([tuple(inverse @ Vector(v)) for v in coordinates], [], faces)
    for material in materials:
        mesh.materials.append(material)
    mesh.update()
    if closed:
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
        bm.to_mesh(mesh)
        bm.free()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    if closed and hasattr(mesh, "set_sharp_from_angle"):
        mesh.set_sharp_from_angle(angle=math.radians(45))
    _transfer_studio_visibility(original, mesh)
    obj.data = mesh


def _transfer_studio_visibility(original, replacement):
    """Resample both studio-occlusion channels onto the rebuilt surface.

    Glass vertices remain on the old surface, where barycentric interpolation
    preserves the original varying values.  The expanded painted corner takes
    values from the closest old surface point at its boundary.  This avoids
    resetting newly subdivided frames/glass to unoccluded shader defaults.
    Both meshes use the same unchanged object coordinate system.
    """
    source = original.attributes.get("_STUDIO_VISIBILITY")
    if source is None:
        return
    if source.domain != "POINT" or source.data_type != "FLOAT2":
        raise ValueError("Expected POINT/FLOAT2 studio visibility on " + original.name)
    values = np.empty(len(original.vertices) * 2, dtype=np.float32)
    source.data.foreach_get("vector", values)
    values = values.reshape(-1, 2)
    original.calc_loop_triangles()
    triangles = [tuple(t.vertices) for t in original.loop_triangles]
    points = [v.co.copy() for v in original.vertices]
    tree = BVHTree.FromPolygons(points, triangles, all_triangles=True)
    mapped = np.empty((len(replacement.vertices), 2), dtype=np.float32)
    for index, vertex in enumerate(replacement.vertices):
        location, _normal, triangle_index, _distance = tree.find_nearest(vertex.co)
        if triangle_index is None:
            raise ValueError("Cannot map studio visibility on " + original.name)
        ids = triangles[triangle_index]
        a, b, c = (points[i] for i in ids)
        ab, ac, ap = b - a, c - a, location - a
        aa, bb, cc = ab.dot(ab), ab.dot(ac), ac.dot(ac)
        denominator = aa * cc - bb * bb
        if abs(denominator) < 1e-20:
            nearest = min(ids, key=lambda i: (points[i] - location).length_squared)
            mapped[index] = values[nearest]
            continue
        wb = (cc * ab.dot(ap) - bb * ac.dot(ap)) / denominator
        wc = (aa * ac.dot(ap) - bb * ab.dot(ap)) / denominator
        weights = np.clip(np.array((1.0 - wb - wc, wb, wc)), 0.0, 1.0)
        weights /= weights.sum()
        mapped[index] = weights @ values[list(ids)]
    if not np.isfinite(mapped).all():
        raise ValueError("Non-finite studio visibility on " + original.name)
    target = replacement.attributes.new("_STUDIO_VISIBILITY", "FLOAT2", "POINT")
    target.data.foreach_set("vector", mapped.ravel())


def _stretch_group(objects, factor):
    arrays = {obj.name: _world_vertices(obj) for obj in objects}
    all_vertices = np.concatenate(list(arrays.values()))
    center_x = float((all_vertices[:, 0].min() + all_vertices[:, 0].max()) / 2)
    before = float(np.ptp(all_vertices[:, 0]))
    for obj in objects:
        coordinates = arrays[obj.name]
        coordinates[:, 0] = center_x + factor * (coordinates[:, 0] - center_x)
        _write_world_vertices(obj, coordinates)
    return {"before_length_m": before, "after_length_m": before * factor}


def _rows(vertices):
    """Return surface section bounds without relying on glTF vertex order."""
    groups = {}
    for point in vertices:
        groups.setdefault(round(float(point[2]), 4), []).append(point)
    rows = []
    for key in sorted(groups):
        row = np.asarray(groups[key])
        rows.append((float(row[:, 2].mean()), float(row[:, 0].min()),
                     float(row[:, 0].max()), float(np.abs(row[:, 1]).min()),
                     float(np.abs(row[:, 1]).max())))
    return np.asarray(rows)


def _round_rear_window(side):
    glass = bpy.data.objects.get("RaptorR_Glass_Door_" + side)
    frame = bpy.data.objects.get("RaptorR_Door_FrameRear_" + side)
    if glass is None or frame is None:
        raise RuntimeError("Expected intact rear-door glass and frame: " + side)
    glass_rows = _rows(_world_vertices(glass))
    frame_rows = _rows(_world_vertices(frame))
    z_min, z_max = glass_rows[0, 0], glass_rows[-1, 0]
    if len(glass_rows) < 20 or not 0.45 < z_max - z_min < 0.65:
        raise RuntimeError("Unexpected rear-window topology: " + side)
    sign = 1 if side.endswith("L") else -1

    # Fit only the unrounded, lower/middle rear glass edge.  Extrapolate this
    # smooth rake over the original small lower corner, then apply a 75 mm
    # circular corner.  The original upper corner and front edge are retained.
    radius = 0.075
    straight = glass_rows[(glass_rows[:, 0] >= z_min + 0.08)
                          & (glass_rows[:, 0] <= z_min + 0.30)]
    baseline = np.polyfit(straight[:, 0] - z_min, straight[:, 1], 2)

    def original_rear(z):
        return float(np.interp(z, glass_rows[:, 0], glass_rows[:, 1]))

    def rear(z):
        dz = float(z - z_min)
        old = original_rear(z)
        if not 0 <= dz < radius:
            return old
        inset = radius - math.sqrt(max(0.0, radius * radius - (radius - dz) ** 2))
        return max(old, float(np.polyval(baseline, dz)) + inset)

    # Keep every original section and add short sections through the new curve.
    # The denser rows avoid a visibly chamfered lower rear window corner.
    z_levels = sorted(set(glass_rows[:, 0].tolist()
                          + np.linspace(z_min, z_min + radius, 31).tolist()))
    columns = 10
    coordinates = []
    for z in z_levels:
        x_rear = rear(z)
        x_front = float(np.interp(z, glass_rows[:, 0], glass_rows[:, 2]))
        y = sign * float(np.interp(z, glass_rows[:, 0], glass_rows[:, 4]))
        for t in np.linspace(0, 1, columns):
            coordinates.append((x_rear + t * (x_front - x_rear), y, z))
    faces = []
    for row in range(len(z_levels) - 1):
        for column in range(columns - 1):
            a = row * columns + column
            face = (a, a + columns, a + columns + 1, a + 1)
            faces.append(face if sign > 0 else tuple(reversed(face)))
    _replace_mesh(glass, coordinates, faces)

    # Rebuild only the existing rear painted frame strip, matching the glass
    # curve.  Its outer rear edge, top/bottom and thickness stay fixed.  This
    # fills the corner cutout with body paint instead of leaving a dark hole.
    frame_levels = sorted(set(frame_rows[:, 0].tolist() + z_levels))
    coordinates = []
    for z in frame_levels:
        outer_x = float(np.interp(z, frame_rows[:, 0], frame_rows[:, 1]))
        inner_x = float(np.interp(z, frame_rows[:, 0], frame_rows[:, 2]))
        if z_min <= z <= z_min + radius:
            inner_x = max(inner_x, rear(z) + 0.004)
        elif z < z_min:
            # Preserve overlap with the door shell and belt seal at the base.
            inner_x = max(inner_x, rear(z_min) + 0.004)
        y_in = sign * float(np.interp(z, frame_rows[:, 0], frame_rows[:, 3]))
        y_out = sign * float(np.interp(z, frame_rows[:, 0], frame_rows[:, 4]))
        coordinates.extend(((outer_x, y_out, z), (inner_x, y_out, z),
                            (inner_x, y_in, z), (outer_x, y_in, z)))
    faces = [(3, 2, 1, 0)]
    for row in range(len(frame_levels) - 1):
        for edge in range(4):
            a = row * 4 + edge
            b = row * 4 + (edge + 1) % 4
            faces.append((a, b, b + 4, a + 4))
    top = (len(frame_levels) - 1) * 4
    faces.append((top, top + 1, top + 2, top + 3))
    _replace_mesh(frame, coordinates, faces, closed=True)
    return {"lower_corner_radius_m": radius,
            "lower_rear_glass_edge_shift_m": rear(z_min) - original_rear(z_min),
            "original_glass_height_m": float(z_max - z_min),
            "glass_rows": len(z_levels)}


def apply():
    """Apply cabin changes once, retaining all existing parent pivots."""
    if bpy.context.scene.get(REVISION):
        return {"revision": REVISION, "already_applied": True}
    bpy.context.view_layer.update()
    report = {"revision": REVISION, "handles": {}, "mirrors": {}, "windows": {}}
    for side in ("FL", "FR", "RL", "RR"):
        objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"
                   and obj.name.startswith("RaptorR_Door_Handle_" + side + "_")]
        if not objects:
            raise RuntimeError("Missing complete handle assembly: " + side)
        report["handles"][side] = _stretch_group(objects, 1.25)
    for side in ("FL", "FR"):
        objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"
                   and obj.name.startswith("RaptorR_Mirror_")
                   and (obj.name.endswith("_" + side) or ("_" + side + "_") in obj.name)
                   and not obj.name.startswith(("RaptorR_Mirror_Arm_", "RaptorR_Mirror_Sail_"))]
        if not objects:
            raise RuntimeError("Missing exterior mirror assembly: " + side)
        report["mirrors"][side] = _stretch_group(objects, 1.35)
    for side in ("RL", "RR"):
        report["windows"][side] = _round_rear_window(side)
    bpy.context.scene[REVISION] = True
    bpy.context.view_layer.update()
    return report
