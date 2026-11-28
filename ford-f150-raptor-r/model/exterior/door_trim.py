"""Front inner-door mouldings traced from the 2024 truck's open driver-door photograph.

Station coordinates are metres in the assembled Blender master (+X forward, +Z up).
The upper upholstery, carbon insert, inner handle and speaker are retained.
"""
import bmesh
import bpy
from mathutils import Vector
from mathutils.kdtree import KDTree


def loft(tag, name, donor, sections, bevel):
    """Skin successive YZ profiles along X, preserving the donor hinge and lighting."""
    prefix = 'RaptorR_DoorCard_'
    source = bpy.data.objects[prefix + donor + '_' + tag]
    matrix = source.matrix_world.copy()
    local = matrix.inverted()
    side = 1 if tag == 'FL' else -1
    old_mesh = source.data
    old_weights = old_mesh.attributes.get('_STUDIO_VISIBILITY')
    if old_weights:
        neighbors = KDTree(len(old_mesh.vertices))
        for v in old_mesh.vertices:
            neighbors.insert(matrix @ v.co, v.index)
        neighbors.balance()

    verts = [Vector((x, y * side, z)) for x, profile in sections for y, z in profile]
    count = len(sections[0][1])
    faces = [list(reversed(range(count))), list(range(len(verts) - count, len(verts)))]
    for station in range(len(sections) - 1):
        for j in range(count):
            a = station * count + j
            b = station * count + (j + 1) % count
            faces.append((a, b, b + count, a + count))

    mesh = bpy.data.meshes.new(prefix + name + '_' + tag)
    mesh.from_pydata([local @ v for v in verts], [], faces)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(old_mesh.materials[0])
    if old_weights:
        weights = mesh.attributes.new('_STUDIO_VISIBILITY', 'FLOAT2', 'POINT')
        for vertex in mesh.vertices:
            nearest = neighbors.find(matrix @ vertex.co)[1]
            weights.data[vertex.index].vector = old_weights.data[nearest].vector

    obj = source if name == donor else source.copy()
    if obj != source:
        obj.name = prefix + name + '_' + tag
        source.users_collection[0].objects.link(obj)
    obj.data = mesh
    obj.modifiers.clear()
    if bevel > 0:
        rounding = obj.modifiers.new('Moulded edge radius', 'BEVEL')
        rounding.width = bevel
        rounding.segments = 3
        rounding.harden_normals = True
        normal = obj.modifiers.new('Face-weighted normals', 'WEIGHTED_NORMAL')
        normal.keep_sharp = True
    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)


def apply():
    for tag in ('FL', 'FR'):
        # The open bin has moulded end returns, a low centre lip and a continuous
        # floor. The inner face of the door panel forms the back of the cavity.
        sections = []
        for x, nose, top, low in [
            (.055, .870, .878, .736), (.080, .846, .823, .685),
            (.137, .817, .786, .653), (.465, .810, .792, .646),
            (.520, .807, .778, .646), (.620, .803, .765, .647),
            (.845, .809, .776, .659), (.949, .823, .789, .677),
            (.991, .853, .810, .705), (1.006, .875, .852, .749),
        ]:
            sections.append((x, [
                (.913, low), (nose + .020, low), (nose, low + .028),
                (nose, top - .009), (nose + .005, top), (nose + .014, top),
                (nose + .019, low + .035), (.903, low + .035),
                (.903, top + .018), (.913, top + .018),
            ]))
        loft(tag, 'MapPocket', 'MapPocket', sections, .004)

        for i, x in enumerate((.496, .840)):
            # Short raised ribs inside the bin support bottles without closing its mouth.
            profile = [(.823, .673), (.888, .673), (.888, .759),
                       (.876, .764), (.845, .714), (.823, .703)]
            loft(tag, 'PocketDivider' + str(i + 1), 'MapPocket',
                 [(x, profile), (x + .013, profile)], .004)

        # The side bolster sweeps inward below the elbow pad, then falls away
        # under the black switch bridge and rises beside the forward speaker.
        sections = []
        for x, front, high, low in [
            (.046, .869, 1.143, .912), (.068, .836, 1.140, .894),
            (.130, .828, 1.096, .895), (.437, .833, 1.093, .900),
            (.510, .846, 1.087, .905), (.566, .857, 1.023, .911),
            (.630, .855, 1.010, .918), (.745, .852, 1.018, .927),
            (.824, .866, 1.055, .969), (.866, .881, 1.099, 1.021),
        ]:
            sections.append((x, [(.913, low), (front + .013, low),
                                (front, low + .022), (front, high - .016),
                                (front + .013, high), (.913, high)]))
        loft(tag, 'SculptedBolster', 'Base', sections, .008)

        sections = []
        for x, y, top, low in [
            (.041, .832, 1.158, 1.105), (.060, .807, 1.172, 1.093),
            (.115, .796, 1.175, 1.075), (.405, .796, 1.168, 1.077),
            (.487, .797, 1.167, 1.081), (.517, .804, 1.160, 1.088),
        ]:
            sections.append((x, [(.911,low), (y+.018,low), (y,low+.016),
                                (y,top-.016), (y+.016,top), (.908,top-.002)]))
        loft(tag, 'Armrest', 'Armrest', sections, .006)
        seam = []
        for x, profile in sections[1:]:
            y, z = profile[2][0], profile[3][1] - .014
            seam.append((x, [(y-.0008,z-.0007), (y+.0006,z-.0007),
                             (y+.0006,z+.0007), (y-.0008,z+.0007)]))
        loft(tag, 'ArmStitch', 'ArmStitch', seam, 0.)

        sections = []
        for x, y, top, low in [
            (.519,.805,1.157,1.091), (.565,.808,1.155,1.079),
            (.730,.814,1.157,1.079), (.838,.829,1.168,1.096),
            (.906,.846,1.190,1.132), (.938,.868,1.207,1.165),
        ]:
            sections.append((x, [(.903,low), (y+.019,low), (y,low+.012),
                                 (y,top-.006), (y+.007,top), (.903,top)]))
        loft(tag, 'SwitchPanel', 'SwitchPanel', sections, .004)
        switches = [(x, y) for x in (.707,.755) for y in (.830,.859)] if tag == 'FL' else [(.746,.847)]
        for i, (x,y) in enumerate(switches):
            profile = [(y,1.156), (y+.022,1.156), (y+.022,1.167), (y,1.167)]
            loft(tag, 'WindowSwitch' + str(i + 1), 'Base',
                 [(x, profile), (x+.031,profile)], .002)
