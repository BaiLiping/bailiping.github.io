"""Save the complete editable model with the photo cameras from its web viewer.

blender -b editable.blend --python-exit-code 1 --python save_reference_master.py -- master.blend
"""
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector


def create_reference_cameras(scene, page):
    views = json.loads((page / 'raptor-views.json').read_text())
    poses = json.loads((page / 'exterior-photo-cameras.json').read_text())
    collection = bpy.data.collections.new('Photo reference cameras')
    scene.collection.children.link(collection)
    cameras = []
    for view in views:
        frame = view['id']
        pose = poses.get(frame)
        if pose is None and view.get('referenceOnly'):
            continue
        if pose is None:
            raise ValueError(f'Missing viewer camera for {frame}')
        theta, phi = pose['theta'], pose['phi']
        cx, cy, cz = pose['center']
        target = Vector((cx, -cz, cy))
        # The GLB uses Y up and the Blender model uses Z up.
        outward = Vector((math.sin(phi) * math.cos(theta),
                          -math.sin(phi) * math.sin(theta), math.cos(phi)))
        camera = bpy.data.cameras.new(frame)
        camera.sensor_fit = 'VERTICAL'
        camera.sensor_height = 24
        camera.lens = camera.sensor_height / (2 * math.tan(math.radians(pose['fov']) / 2))
        camera.clip_start = 0.02
        camera.clip_end = max(160, pose['distance'] + 80)
        camera.display_size = 0.075
        obj = bpy.data.objects.new(frame, camera)
        collection.objects.link(obj)
        obj.matrix_world = (Matrix.Translation(target + pose['distance'] * outward)
                            @ (-outward).to_track_quat('-Z', 'Y').to_matrix().to_4x4()
                            @ Matrix.Rotation(pose.get('roll', 0), 4, 'Z'))
        obj['label'] = view['label']
        obj['photo_url'] = view.get('original', view['image'])
        obj['source'] = view['source']
        obj['reference_width'] = view['width']
        obj['reference_height'] = view['height']
        obj['panel_state'] = view.get('panelState', 'closed')
        obj['viewer_pose'] = json.dumps(pose)
        obj.hide_set(True)
        cameras.append(obj)
    return cameras


def main():
    destination = Path(sys.argv[sys.argv.index('--') + 1]).resolve()
    scene = bpy.context.scene
    cameras = create_reference_cameras(scene, Path(__file__).resolve().parents[1])
    scene.camera = cameras[0]
    scene.render.resolution_x = scene.camera['reference_width']
    scene.render.resolution_y = scene.camera['reference_height']
    scene.render.resolution_percentage = 100
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1.0
    bpy.ops.file.pack_all()
    bpy.context.preferences.filepaths.save_version = 0
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            area.spaces.active.region_3d.view_location = (0.0, 0.0, 1.0)
            area.spaces.active.region_3d.view_distance = 8.8
            area.spaces.active.region_3d.view_rotation = Vector((-5.8, -4.6, -2.0)).to_track_quat('-Z', 'Y')
            area.spaces.active.overlay.show_extras = False
            area.spaces.active.shading.color_type = 'MATERIAL'
    bpy.ops.wm.save_as_mainfile(filepath=str(destination), compress=True)
    print(f'REFERENCE_MASTER {len(cameras)} cameras and {len(bpy.data.meshes)} meshes')


if __name__ == '__main__':
    main()
