"""Apply local, reference-led corrections to the verified articulated Raptor R.

Run in Blender:
  blender -b --python-exit-code 1 --python model/build_model.py -- decoded-input.glb output.glb

Input is the decoded asset at site commit 007d1fc7ff1f939016f3ddcc9330afb3a02dfca5.
That revision preserves rigid running gear, the closed bed cover, and panel gaps.
The later whole-vehicle silhouette deformation must not be used as input.
"""

from pathlib import Path
import importlib
import json
import struct
import sys

import bpy
import numpy as np
from mathutils.kdtree import KDTree


def glb_arrays(path):
    data = Path(path).read_bytes()
    length = struct.unpack_from('<I', data, 12)[0]
    doc = json.loads(data[20:20 + length])
    binary = memoryview(data)[28 + length:]
    types = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
             5123: np.uint16, 5125: np.uint32, 5126: np.float32}
    dims = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}

    def read(index):
        acc = doc['accessors'][index]
        view = doc['bufferViews'][acc['bufferView']]
        dtype = np.dtype(types[acc['componentType']])
        size = dims[acc['type']]
        offset = view.get('byteOffset', 0) + acc.get('byteOffset', 0)
        return np.ndarray((acc['count'], size), dtype=dtype, buffer=binary,
                          offset=offset,
                          strides=(view.get('byteStride', dtype.itemsize * size), dtype.itemsize)).copy()
    return doc, read


def preserve_studio_visibility(path):
    """Restore the custom two-component light-occlusion data the importer omits.

    Do this before changing geometry, so values follow their own vertices when
    a handle, mirror or window is reshaped. This preserves cabin/engine shading.
    """
    doc, read = glb_arrays(path)
    restored = 0
    for node in doc['nodes']:
        if 'mesh' not in node:
            continue
        obj = bpy.data.objects.get(node.get('name', ''))
        if obj is None or obj.type != 'MESH':
            continue
        mesh = doc['meshes'][node['mesh']]
        points, values = [], []
        for primitive in mesh['primitives']:
            attrs = primitive['attributes']
            p = read(attrs['POSITION'])
            # glTF Y-up to Blender Z-up, before the named articulation pivots.
            points.append(p[:, [0, 2, 1]] * np.array([1, -1, 1]))
            values.append(read(attrs['_STUDIO_VISIBILITY']) if '_STUDIO_VISIBILITY' in attrs
                          else np.ones((len(p), 2), dtype=np.float32))
        if not any('_STUDIO_VISIBILITY' in p['attributes'] for p in mesh['primitives']):
            continue
        points, values = np.concatenate(points), np.concatenate(values)
        tree = KDTree(len(points))
        for i, point in enumerate(points):
            tree.insert(point, i)
        tree.balance()
        mapped = np.empty((len(obj.data.vertices), 2), dtype=np.float32)
        maximum_error = 0.0
        for i, vertex in enumerate(obj.data.vertices):
            _, index, distance = tree.find(vertex.co)
            maximum_error = max(maximum_error, distance)
            mapped[i] = values[index]
        if maximum_error > 0.0001:
            raise ValueError(f'Custom-attribute mapping failed for {obj.name}: {maximum_error} m')
        attribute = obj.data.attributes.get('_STUDIO_VISIBILITY')
        if attribute is None:
            attribute = obj.data.attributes.new('_STUDIO_VISIBILITY', 'FLOAT2', 'POINT')
        attribute.data.foreach_set('vector', mapped.ravel())
        restored += 1
    return restored


def validate_rig():
    """Guard the relationships the discarded silhouette fit broke."""
    def points(name):
        o = bpy.data.objects[name]
        return np.array([o.matrix_world @ v.co for v in o.data.vertices])

    hubs = {k: bpy.data.objects['Pivot_Corner_' + k].matrix_world.translation.copy()
            for k in ('FL', 'FR', 'RL', 'RR')}
    wheelbase = (hubs['FL'].x - hubs['RL'].x + hubs['FR'].x - hubs['RR'].x) / 2
    assert abs(wheelbase - 3.6932) < 0.0002, wheelbase
    gap = points('RaptorR_Body_Cab_RearWall')[:, 0].min() - max(
        points('RaptorR_Body_BedSide_' + side)[:, 0].max() for side in ('L', 'R'))
    assert gap > 0.02, gap
    rail = points('RaptorR_Bed_RailCap_L')[:, 2].max()
    cover = [points('RaptorR_Tonneau_' + part + '_Panel')[:, 2].max()
             for part in ('Front', 'Middle', 'Rear')]
    assert max(abs(v - rail) for v in cover) < 0.001, (rail, cover)
    assert points('RaptorR_Drivetrain_RearDiffCase')[:, 2].min() >= 0.332, 'Differential moved'
    return {'wheelbase_m': wheelbase, 'cab_to_bed_gap_m': gap,
            'cover_to_rail_m': [v - rail for v in cover]}


def main():
    source, output = map(Path, sys.argv[sys.argv.index('--') + 1:][:2])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source.resolve()))
    bpy.context.view_layer.update()
    original_pivots = {o.name: o.matrix_world.copy() for o in bpy.data.objects
                       if o.name.startswith('Pivot_')}
    report = {'studio_visibility_meshes_restored': preserve_studio_visibility(source)}
    visibility_meshes = {o.name for o in bpy.data.objects if o.type == 'MESH'
                         and o.data.attributes.get('_STUDIO_VISIBILITY') is not None}
    sys.path.insert(0, str(Path(__file__).parent))
    for module in ('correct_cabin', 'correct_rear', 'correct_wheels'):
        report[module] = importlib.import_module(module).apply()
    bpy.context.view_layer.update()
    for name, matrix in original_pivots.items():
        assert np.allclose(np.array(bpy.data.objects[name].matrix_world), np.array(matrix), atol=1e-6), name
    for name in visibility_meshes:
        assert bpy.data.objects[name].data.attributes.get('_STUDIO_VISIBILITY') is not None, name
    # The inherited unwrapped auxiliary plumbing cannot sample a normal map.
    # The separate UV-mapped foil-shield material remains textured.
    foil = bpy.data.materials.get('Mat_Eng_Foil_Shield')
    if foil and foil.use_nodes:
        users = [o for o in bpy.data.objects if o.type == 'MESH'
                 and any(s.material == foil for s in o.material_slots)]
        if users and all(not o.data.uv_layers for o in users):
            for node in foil.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    for link in list(node.inputs['Normal'].links):
                        foil.node_tree.links.remove(link)
            report['unmapped_foil_normal_removed'] = True
    report['assembly'] = validate_rig()
    bpy.ops.export_scene.gltf(
        filepath=str(output.resolve()), export_format='GLB',
        export_animations=False, export_cameras=False, export_lights=False,
        export_apply=True, export_attributes=True, export_extras=True,
        export_yup=True, export_image_format='AUTO')
    output.with_suffix('.build.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print('RAPTOR_BUILD ' + json.dumps(report, default=str))


if __name__ == '__main__':
    main()
