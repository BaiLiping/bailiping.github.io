"""Replace the Raptor's rectangular tire blocks with interlocking tread.

Run ``apply()`` after importing the verified, pre-deformation GLB in Blender.
Only the four named tire meshes and their shared rubber materials are edited.
The original sidewall mesh, UVs, embossed normal texture, wheel pivots, rim,
brakes, wheel centers, and maximum 0.4635 m radius are retained.

The staggered, chamfered blocks and extended shoulder lugs are a visual
approximation informed by ``../frame-by-frame/frames/
wheel-and-37-ko2-tire-photo.webp``. They are not a tire-manufacturer mold model.
No image, registry, network, or optional Python package is needed.
"""

import math

import bpy


RADIUS = 0.4635
PITCHES = 48
TIRE_NAMES = tuple("RaptorR_Tire_" + corner for corner in ("FL", "FR", "RL", "RR"))


def _rubber_materials():
    """Black molded lettering gets its contrast from the retained normal map."""
    for material in bpy.data.materials:
        if not material.name.startswith(("Mat_Underbody_KO2_Sidewall", "Mat_Tire_KO2_Tread")):
            continue
        sidewall = material.name.startswith("Mat_Underbody_KO2_Sidewall")
        color = (0.021, 0.022, 0.023, 1) if sidewall else (0.026, 0.027, 0.028, 1)
        material.diffuse_color = color
        if not material.use_nodes:
            continue
        for node in material.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                # The old albedo painted the lettering pale gray. KO2 black
                # sidewalls have molded rubber letters, retained in the normal
                # texture, not this high-contrast printed paint.
                for link in list(node.inputs["Base Color"].links):
                    material.node_tree.links.remove(link)
                node.inputs["Base Color"].default_value = color
                node.inputs["Roughness"].default_value = 0.83 if sidewall else 0.87
                node.inputs["Metallic"].default_value = 0.0
                if "Specular IOR Level" in node.inputs:
                    node.inputs["Specular IOR Level"].default_value = 0.30
                if "Coat Weight" in node.inputs:
                    node.inputs["Coat Weight"].default_value = 0.0
            elif node.type == "NORMAL_MAP" and sidewall:
                node.inputs["Strength"].default_value = 0.08


def _surface_radius(axial):
    """Rounded crown and shoulder following the original sidewall section."""
    a = abs(axial)
    if a <= 0.095:
        return RADIUS - 0.0030 * (a / 0.095) ** 2
    stations = (
        (0.095, RADIUS - 0.0030),
        (0.127, 0.4590),
        (0.135, 0.4510),
        (0.145, 0.4290),
        (0.1515, 0.4130),
        (0.156, 0.3970),
    )
    for (a0, r0), (a1, r1) in zip(stations, stations[1:]):
        if a <= a1:
            return r0 + (r1 - r0) * (a - a0) / (a1 - a0)
    return stations[-1][1]


def _build_tread():
    vertices, faces, top_faces = [], [], set()
    # Coordinates are axial / circumferential in a tire-unwrapped plane.
    # Concave waist/notches break up the former rows of rectangular paddles.
    center_shape = (
        (-0.48, -0.43), (0.04, -0.50), (0.49, -0.26),
        (0.29, -0.02), (0.48, 0.29), (0.00, 0.49),
        (-0.49, 0.25), (-0.30, 0.01),
    )
    shoulder_shape = (
        (-0.49, -0.35), (-0.12, -0.48), (0.48, -0.32),
        (0.37, -0.05), (0.49, 0.25), (0.13, 0.48),
        (-0.48, 0.33), (-0.32, 0.02),
    )
    circumference_pitch = 2 * math.pi * RADIUS / PITCHES

    def point(axial, tangent, radial_offset):
        theta = tangent / RADIUS
        radius = _surface_radius(axial) + radial_offset
        return (radius * math.sin(theta), axial, radius * math.cos(theta))

    def lug(footprint, center, length, width, mirror):
        base = len(vertices)
        n = len(footprint)
        axial_center, tangent_center = center
        # The sidewall intersects the base of each shoulder lug. The 1.3 mm
        # chamfer softens its crown without a modifier or extra object.
        for inset, radial_offset in ((1.0, -0.0135), (1.0, -0.0013), (0.90, 0.0)):
            for a, t in footprint:
                vertices.append(point(
                    axial_center + a * width * inset,
                    tangent_center + t * length * inset * mirror,
                    radial_offset,
                ))
        faces.append(tuple(base + k for k in range(n - 1, -1, -1)))
        for ring in range(2):
            for k in range(n):
                other = (k + 1) % n
                faces.append((base + ring * n + k, base + ring * n + other,
                              base + (ring + 1) * n + other, base + (ring + 1) * n + k))
        center_index = len(vertices)
        vertices.append(point(axial_center, tangent_center, 0.0))
        for k in range(n):
            top_faces.add(len(faces))
            faces.append((center_index, base + 2 * n + k, base + 2 * n + (k + 1) % n))
        if mirror > 0:
            # Axial/circumferential coordinates map to an inward-facing
            # clockwise basis on the cylinder. Reverse the unmirrored faces.
            start = len(faces) - (3 * n + 1)
            for i in range(start, len(faces)):
                faces[i] = tuple(reversed(faces[i]))

    # Adjacent rows are out of phase. Shoulder blocks extend over the curved
    # sidewall instead of ending as a square row at the tread's outside edge.
    rows = ((-0.120, 0.065, 0.25), (-0.058, 0.055, 0.50),
            (0.000, 0.055, 0.00), (0.058, 0.055, 0.50), (0.120, 0.065, 0.25))
    for row, (axial, width, phase) in enumerate(rows):
        for pitch in range(PITCHES):
            shoulder = row in (0, 4)
            shape = shoulder_shape if shoulder else center_shape
            # Alternate the shoulder's short/long lugs like the reference.
            length = circumference_pitch * (0.88 if shoulder and pitch % 2 else 0.94)
            mirror = -1 if axial < 0 else 1
            lug(shape, (axial, (pitch + phase) * circumference_pitch), length, width, mirror)
    return vertices, faces, top_faces


def _replace_tread(obj, vertices, faces, smooth_faces):
    old = obj.data
    tread_slots = {i for i, mat in enumerate(old.materials)
                   if mat and mat.name.startswith("Mat_Tire_KO2_Tread")}
    if len(tread_slots) != 1:
        raise ValueError(f"Unexpected tread materials on {obj.name}: {tread_slots}")
    tread_slot = next(iter(tread_slots))
    kept = [polygon for polygon in old.polygons if polygon.material_index not in tread_slots]
    old_indices = sorted({index for polygon in kept for index in polygon.vertices})
    remap = {old_index: new_index for new_index, old_index in enumerate(old_indices)}
    sidewall_vertices = [tuple(old.vertices[index].co) for index in old_indices]
    sidewall_faces = [tuple(remap[index] for index in polygon.vertices) for polygon in kept]
    sidewall_normals = [tuple(old.corner_normals[loop].vector)
                        for polygon in kept for loop in polygon.loop_indices]
    uv_values = {
        layer.name: [tuple(layer.data[loop].uv) for polygon in kept for loop in polygon.loop_indices]
        for layer in old.uv_layers
    }
    offset = len(sidewall_vertices)
    combined_faces = sidewall_faces + [tuple(offset + index for index in face) for face in faces]
    new = bpy.data.meshes.new(old.name + "_InterlockingTread")
    new.from_pydata(sidewall_vertices + vertices, [], combined_faces)
    for material in old.materials:
        new.materials.append(material)
    for i, polygon in enumerate(kept):
        new.polygons[i].material_index = polygon.material_index
        new.polygons[i].use_smooth = polygon.use_smooth
    for i, polygon in enumerate(new.polygons[len(kept):]):
        polygon.material_index = tread_slot
        polygon.use_smooth = i in smooth_faces
    for name, values in uv_values.items():
        layer = new.uv_layers.new(name=name)
        for loop, value in zip(layer.data, values):
            loop.uv = value
    new.update()
    # Preserve the original sidewall's imported split normals and its normal
    # map. A zero custom normal asks Blender to calculate the new lug normals.
    normals = sidewall_normals + [(0.0, 0.0, 0.0)] * (len(new.loops) - len(sidewall_normals))
    new.normals_split_custom_set(normals)
    obj.data = new
    if old.users == 0:
        bpy.data.meshes.remove(old)
    new.calc_loop_triangles()
    return {
        "name": obj.name,
        "parent": obj.parent.name if obj.parent else None,
        "center_m": list(obj.matrix_world.translation),
        "outer_radius_m": max(math.hypot(v.co.x, v.co.z) for v in new.vertices),
        "width_m": max(v.co.y for v in new.vertices) - min(v.co.y for v in new.vertices),
        "triangles": len(new.loop_triangles),
        "sidewall_faces_preserved": len(kept),
    }


def apply():
    """Apply once to the imported prefit model and return measurement records."""
    objects = [bpy.data.objects.get(name) for name in TIRE_NAMES]
    if any(obj is None or obj.type != "MESH" for obj in objects):
        raise ValueError("Import the Raptor GLB before calling correct_wheels.apply()")
    vertices, faces, smooth_faces = _build_tread()
    _rubber_materials()
    result = [_replace_tread(obj, vertices, faces, smooth_faces) for obj in objects]
    for item in result:
        if abs(item["outer_radius_m"] - RADIUS) > 0.00001:
            raise AssertionError(f"Tire radius changed: {item}")
        if not 0.3178 <= item["width_m"] <= 0.3182:
            raise AssertionError(f"Tire width changed: {item}")
    return result
