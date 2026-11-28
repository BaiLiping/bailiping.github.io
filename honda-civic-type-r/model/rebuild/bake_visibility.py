"""Bake how much of the studio each vertex can see with the FK8 closed and open.

python bake_visibility.py car.glb studio-visibility.bin.gz [--variant cuda_ad_rgb]
Requires NumPy, SciPy, DracoPy and Mitsuba 3. Geometry stays in its original GLB.
"""
import argparse
import gzip
import hashlib
import json
import math
import struct
from pathlib import Path

import DracoPy
import drjit as dr
import mitsuba as mi
import numpy as np
from scipy.spatial.transform import Rotation


# Match the inspection poses in the viewer, in the rig's glTF coordinates.
HINGES = {
    'Pivot_Door_L': ('y', -62), 'Pivot_Door_R': ('y', 62),
    'Pivot_Door_RL': ('y', -60), 'Pivot_Door_RR': ('y', 60),
    'Pivot_Hood_Assembly': ('z', 52), 'Pivot_Hatch_Assembly': ('z', -88),
    'Pivot_Fuel_Flap': ('y', -75),
}


def primitive_transforms(doc, opened):
    transforms = {}
    lift_meshes = {}

    def visit(index, parent):
        node = doc['nodes'][index]
        local = np.eye(4)
        rotation = Rotation.from_quat(node.get('rotation', [0, 0, 0, 1]))
        if opened and node.get('name') in HINGES:
            axis, angle = HINGES[node['name']]
            rotation = rotation * Rotation.from_euler(axis, angle, degrees=True)
        local[:3, :3] = rotation.as_matrix() @ np.diag(node.get('scale', [1, 1, 1]))
        local[:3, 3] = node.get('translation', [0, 0, 0])
        world = parent @ local
        if 'mesh' in node:
            if node['mesh'] in transforms:
                raise ValueError('Instanced meshes need independent visibility values')
            transforms[node['mesh']] = world
            if node.get('name', '').startswith(('Civic_Body_Shell_Lift_Support_', 'Civic_Hatch_Lift_Rod_')):
                lift_meshes[node['name']] = node['mesh']
        for child in node.get('children', []):
            visit(child, world)

    for root in doc['scenes'][doc['scene']]['nodes']:
        visit(root, np.eye(4))
    if opened:
        for side in ('L', 'R'):
            tube = lift_meshes.get('Civic_Body_Shell_Lift_Support_'+side)
            rod = lift_meshes.get('Civic_Hatch_Lift_Rod_'+side)
            if tube is None or rod is None:
                continue
            for part, target in ((tube, rod), (rod, tube)):
                world = transforms[part]
                direction = transforms[target][:3, 3] - world[:3, 3]
                direction /= np.linalg.norm(direction)
                axis = world[:3, 1] / np.linalg.norm(world[:3, 1])
                cross = np.cross(axis, direction)
                sine = np.linalg.norm(cross)
                if sine > 1e-8:
                    align = Rotation.from_rotvec(cross / sine * np.arctan2(sine, axis @ direction))
                    world[:3, :3] = align.as_matrix() @ world[:3, :3]
    return transforms


def read_primitives(doc, binary):
    result = []
    for mesh_index, mesh in enumerate(doc['meshes']):
        for primitive in mesh['primitives']:
            draco = primitive['extensions']['KHR_draco_mesh_compression']
            view = doc['bufferViews'][draco['bufferView']]
            start = view.get('byteOffset', 0)
            geom = DracoPy.decode(bytes(binary[start:start + view['byteLength']]))
            mat = doc['materials'][primitive['material']]
            # Light enters the cabin through the glazing. Exclude transparent
            # graphics too, so the opaque surface underneath determines shadow.
            opaque = mat.get('alphaMode', 'OPAQUE') == 'OPAQUE' and 'Clear_Lens' not in mat['name']
            result.append((mesh_index, geom.points, geom.normals, geom.faces if opaque else None))
    return result


def trace(primitives, transforms, samples):
    positions, normals, triangles = [], [], []
    start = 0
    for mesh_index, points, local_normals, faces in primitives:
        matrix = transforms[mesh_index]
        positions.append(points @ matrix[:3, :3].T + matrix[:3, 3])
        normals.append(local_normals @ np.linalg.inv(matrix[:3, :3]))
        if faces is not None:
            triangles.append(faces + start)
        start += len(points)
    points = np.concatenate(positions).astype('f')
    normals = np.concatenate(normals).astype('f')
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    faces = np.concatenate(triangles).astype('I')
    mesh = mi.Mesh('opaque panels and assemblies', len(points), len(faces))
    params = mi.traverse(mesh)
    params['vertex_positions'] = points.ravel()
    params['faces'] = faces.ravel()
    params.update()
    scene = mi.load_dict({'type': 'scene', 'vehicle': mesh})
    result = []
    for start in range(0, len(points), 131072):
        stop = min(len(points), start + 131072)
        normal = mi.Normal3f(normals[start:stop].T.copy())
        frame = mi.Frame3f(normal)
        origin = mi.Point3f(points[start:stop].T.copy()) + .0015 * normal
        visible = mi.Float(0)
        # A deterministic, cosine-weighted hemisphere. The 1.5 mm origin
        # offset avoids false hits from quantisation and interpolated normals.
        for i in range(samples):
            radius = math.sqrt((i + .5) / samples)
            angle = i * math.pi * (3 - math.sqrt(5))
            local = mi.Vector3f(radius * math.cos(angle), radius * math.sin(angle), math.sqrt(1 - radius**2))
            ray = mi.Ray3f(origin, frame.to_world(local))
            visible += dr.select(scene.ray_test(ray), 0., 1.)
            if i % 8 == 7:
                dr.eval(visible)
        result.append(np.rint(np.asarray(visible) * (255 / samples)).astype('B'))
    return np.concatenate(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--variant', choices=['llvm_ad_rgb', 'cuda_ad_rgb'], default='llvm_ad_rgb')
    args = parser.parse_args()
    mi.set_variant(args.variant)
    raw = args.model.read_bytes()
    json_size = struct.unpack_from('<I', raw, 12)[0]
    doc = json.loads(raw[20:20 + json_size])
    primitives = read_primitives(doc, memoryview(raw)[28 + json_size:])
    counts = np.array([len(points) for _, points, _, _ in primitives], dtype='<u4')
    states = []
    for opened in (False, True):
        print('Tracing with panels', 'open' if opened else 'closed', flush=True)
        states.append(trace(primitives, primitive_transforms(doc, opened), 96))
    payload = np.column_stack(states)
    header = struct.pack('<4sI', b'FKV1', len(counts)) + hashlib.sha256(raw).digest()
    args.output.write_bytes(gzip.compress(header + counts.tobytes() + payload.tobytes(), mtime=0))
    print(len(payload), 'visibility pairs;', args.output.stat().st_size, 'compressed bytes')


if __name__ == '__main__':
    main()
