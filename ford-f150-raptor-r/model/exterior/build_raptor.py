"""Rebuild the 2024 F-150 Raptor R exterior and merge it with the kept running gear, engine and cabin.

blender -b --python-exit-code 1 --python build_raptor.py -- <decoded-input.glb> <output.glb> [--blend out.blend]
"""
import bpy, sys, re, json, struct, importlib
from pathlib import Path
import numpy as np
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree
B = Path(__file__).resolve().parent
sys.path.insert(0, str(B))
import geo, mk, design, parts_body, parts_front, parts_rear, parts_misc, parts_extra, parts_detail
for m in (geo, mk, design, parts_body, parts_front, parts_rear, parts_misc, parts_extra, parts_detail): importlib.reload(m)
D = design

def glb_arrays(path):
    data = Path(path).read_bytes(); L = struct.unpack_from('<I', data, 12)[0]
    doc = json.loads(data[20:20 + L]); binary = memoryview(data)[28 + L:]
    types = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
    dims = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}
    def read(i):
        acc = doc['accessors'][i]; view = doc['bufferViews'][acc['bufferView']]; dt = np.dtype(types[acc['componentType']]); n = dims[acc['type']]
        off = view.get('byteOffset', 0) + acc.get('byteOffset', 0)
        return np.ndarray((acc['count'], n), dtype=dt, buffer=binary, offset=off, strides=(view.get('byteStride', dt.itemsize * n), dt.itemsize)).copy()
    return doc, read

def preserve_studio_visibility(path):
    """restore the custom two-channel light-occlusion attribute the importer drops (kept meshes only)"""
    doc, read = glb_arrays(path); restored = 0
    for node in doc['nodes']:
        if 'mesh' not in node: continue
        ob = bpy.data.objects.get(node.get('name', ''))
        if ob is None or ob.type != 'MESH': continue
        mesh = doc['meshes'][node['mesh']]
        if not any('_STUDIO_VISIBILITY' in p['attributes'] for p in mesh['primitives']): continue
        pts, vals = [], []
        for p in mesh['primitives']:
            P = read(p['attributes']['POSITION']); pts.append(P[:, [0, 2, 1]] * np.array([1, -1, 1]))
            vals.append(read(p['attributes']['_STUDIO_VISIBILITY']) if '_STUDIO_VISIBILITY' in p['attributes'] else np.ones((len(P), 2), np.float32))
        pts, vals = np.concatenate(pts), np.concatenate(vals)
        tree = KDTree(len(pts))
        for i, q in enumerate(pts): tree.insert(q, i)
        tree.balance()
        mapped = np.empty((len(ob.data.vertices), 2), np.float32)
        for i, v in enumerate(ob.data.vertices): mapped[i] = vals[tree.find(v.co)[1]]
        att = ob.data.attributes.get('_STUDIO_VISIBILITY') or ob.data.attributes.new('_STUDIO_VISIBILITY', 'FLOAT2', 'POINT')
        att.data.foreach_set('vector', mapped.ravel()); restored += 1
    return restored

REMOVE = re.compile(r'^RaptorR_(Bed_|Body_(?!Cab_FloorPan|Cab_Firewall|UnderbodyFloor|PinchWeld)|Cowl_|Door_|FenderFlare|Fender_|FlareMarker|'
                    r'FogPocket|Front_|Fuel_|Glass_|Graphic_|Headlight|Hood_|Mirror_(?!Cabin)|RearCab_|Rear_|Roof_|RunningBoard|'
                    r'Tailgate|Taillight|Tonneau|Wheelhouse|Windshield)')

def move_pivot(name, loc):
    piv = bpy.data.objects[name]
    kids = [(c, c.matrix_world.copy()) for c in piv.children]
    piv.matrix_world = Matrix.Translation(Vector(loc)) @ piv.matrix_world.to_3x3().to_4x4()
    bpy.context.view_layer.update()
    for c, mw in kids: c.matrix_world = mw
    bpy.context.view_layer.update()

def parent_keep(ob, piv):
    mw = ob.matrix_world.copy(); ob.parent = piv; ob.matrix_parent_inverse = piv.matrix_world.inverted(); ob.matrix_world = mw

def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    src, out = Path(argv[0]), Path(argv[1])
    blend = Path(argv[argv.index('--blend') + 1]) if '--blend' in argv else None
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(src.resolve()))
    bpy.context.view_layer.update()
    report = {'studio_visibility_restored': preserve_studio_visibility(src)}
    removed = [o.name for o in bpy.data.objects if o.type == 'MESH' and REMOVE.match(o.name)]
    for n in removed: bpy.data.objects.remove(bpy.data.objects[n], do_unlink=True)
    report['removed_meshes'] = len(removed)
    # ---- new exterior
    col = bpy.data.collections.new('NewExterior'); bpy.context.scene.collection.children.link(col)
    parts_body.side_panels(col); parts_body.greenhouse(col); parts_body.hood(col)
    parts_front.grille(col); parts_detail.headlamps_v3(col); parts_detail.front_cover(col); parts_front.bumper(col); parts_detail.wheel_aprons(col); parts_detail.fender_nose(col)
    parts_rear.tailgate(col); parts_detail.tailgate_letters(col, parts_rear.XT); parts_detail.taillamps(col); parts_detail.bed_rear_closure(col); parts_rear.rear_bumper(col); parts_rear.bed_box(col)
    parts_misc.flare(col, 'RaptorR_FenderFlare_F', D.FLARE_F_OUT, D.FLARE_F_IN, D.WB2,
                     yout_fn=geo.pchip([1.2, 1.9, 2.05, 2.2, 2.3], [1.105, 1.105, 1.080, 1.058, 1.048]))
    parts_misc.flare(col, 'RaptorR_FenderFlare_Rr', D.FLARE_R_OUT, D.FLARE_R_IN, -D.WB2)
    parts_misc.liner(col, 'RaptorR_Wheelhouse_F', D.FLARE_F_IN, D.WB2); parts_misc.liner(col, 'RaptorR_Wheelhouse_Rr', D.FLARE_R_IN, -D.WB2)
    parts_misc.mirrors(col); parts_misc.running_boards(col); parts_misc.handles(col); fuel = parts_misc.fuel_door(col)
    parts_misc.fender_vents(col); parts_detail.hood_extractor(col); parts_misc.cowl_and_wipers(col); parts_misc.decals(col); parts_misc.hood_decal(col)
    parts_detail.hood_nose(col)
    old = bpy.data.objects.get('RaptorR_Body_CHMSL_Housing')
    if old: bpy.data.objects.remove(old, do_unlink=True)
    parts_detail.chmsl(col); parts_detail.roof_markers(col); parts_detail.cpillar_appliques(col)
    for tag, outline in (('F', D.FRONT_DOOR), ('R', D.REAR_DOOR)):
        band, _ = parts_extra.door_shell(col, f'RaptorR_Door_Shell_{tag}_L', outline)
        mk.mirror_y(band, f'RaptorR_Door_Shell_{tag}_R_Edge')
    ton = parts_extra.tonneau(col); parts_extra.bed_details(col); parts_extra.fuel_pocket(col); parts_extra.plate(col)
    parts_extra.badges(col); parts_extra.rear_bumper_wings(col); parts_extra.hood_underside(col)
    bpy.context.view_layer.update()
    # ---- fit kept parts inside the new shell
    def Yg(x, z):
        t = np.clip((z - D.Z_BELT) / (D.Z_DRIP - D.Z_BELT), 0, 1.2); yb, yd = D.Y_BELT(x), D.Y_DRIP(x)
        return yb - (yb - yd) * t ** D.P_TUMBLE(x)
    def smooth(a, b, x): t = np.clip((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)
    def deform(ob, fn):
        M = np.array(ob.matrix_world); Mi = np.linalg.inv(M)
        V = np.array([v.co for v in ob.data.vertices]); W = V @ M[:3, :3].T + M[:3, 3]
        W2 = fn(W.copy()); L = (W2 - M[:3, 3]) @ Mi[:3, :3].T
        for v, p in zip(ob.data.vertices, L): v.co = p
    fitted = {}
    def fit_cabin(W):
        x, y, z = W[:, 0], W[:, 1], W[:, 2]
        w = smooth(1.40, 1.50, z) * (x < 1.25)
        ylim = Yg(np.clip(x, -1.16, 1.2), np.maximum(z, D.Z_BELT)) - 0.022
        ay = np.abs(y); k = 0.80 * ylim; L = ylim - k
        ay2 = np.where(ay > k, k + L * np.tanh((ay - k) / np.maximum(L, 1e-4)), ay)
        y2 = np.sign(y) * (ay + (ay2 - ay) * w)
        # stay under the windshield and the roof
        zws = 1.505 + (1.272 - x) * (1.978 - 1.505) / (1.272 - 0.562) - 0.014
        zr = D.Z_ROOF(np.clip(x, D.ROOF_REAR_X, D.ROOF_FRONT_X), np.clip(y2, -0.69, 0.69)) - 0.030
        zlim = np.where(x > 0.56, zws, zr)
        z2 = np.where((z > zlim) & (x > -1.10), zlim - 0.004 * np.tanh((z - zlim) / 0.01), z)
        return np.stack([x, y2, z2], 1)
    for ob in [o for o in bpy.data.objects if o.type == 'MESH' and o.name.startswith('RaptorR_Int_') and o.name not in ('RaptorR_Int_Floor',)]:
        deform(ob, fit_cabin); fitted[ob.name] = 'cabin'
    # Maintain room between the cabin trim and the lower door pressing.
    # Shift each complete card with its switches and pull, on its existing hinge.
    for ob in [o for o in bpy.data.objects if o.type == 'MESH' and o.name.startswith('RaptorR_DoorCard_')]:
        def move_card(W):
            W[:, 1] -= np.sign(W[:, 1].mean()) * 0.016
            return W
        deform(ob, move_card)
    import seat_trim, door_trim
    seat_trim.apply()
    door_trim.apply()
    def fit_engine(W):
        x, y, z = W[:, 0], W[:, 1], W[:, 2]
        inside = (x > D.HOOD_REAR_X) & (x < 2.64) & (np.abs(y) < 0.84)
        zl = D.Z_HOOD(np.clip(x, 1.27, 2.64), np.clip(y, -0.84, 0.84)) - 0.040
        z2 = np.where(inside & (z > zl - 0.03), np.minimum(z, zl - 0.03 + 0.03 * np.tanh((z - zl + 0.03) / 0.03)), z)
        return np.stack([x, y, z2], 1)
    for ob in [o for o in bpy.data.objects if o.type == 'MESH' and o.name.startswith('RaptorR_Eng_')]:
        deform(ob, fit_engine); fitted[ob.name] = 'engine'
    # exhaust tips to the BaT rear photo (plate-scaled): centres at |y| 0.758, z 0.577, ends flush with the bumper face
    TIP = (0.040, -0.062, -0.073)
    def fit_tip(W, side):
        W[:, 0] += TIP[0]; W[:, 1] += side * TIP[1]; W[:, 2] += TIP[2]; return W
    deform(bpy.data.objects['RaptorR_Exhaust_Tip_L'], lambda W: fit_tip(W, 1)); deform(bpy.data.objects['RaptorR_Exhaust_Tip_R'], lambda W: fit_tip(W, -1))
    def fit_pipes(W):
        x, y, z = W[:, 0], W[:, 1], W[:, 2]
        w = smooth(-2.45, -2.78, x) * (np.abs(y) > 0.55)
        return np.stack([x + TIP[0] * w, y + np.sign(y) * TIP[1] * w, z + TIP[2] * w], 1)
    deform(bpy.data.objects['RaptorR_Exhaust_Pipes'], fit_pipes)
    report['fitted_kept_meshes'] = len(fitted)
    # ---- names the viewer's labels and exploded view key on (fog lamps travel with the bumper, markers with the grille)
    renames = [(r'^RaptorR_Front_(FogHousing|FogLights|FogPocket|ParkSensor)_', r'RaptorR_Front_Bumper_\1_'),
               (r'^RaptorR_Front_AmberMarker', 'RaptorR_Front_Grille_Amber_Marker'),
               (r'^RaptorR_Body_RoofMarker_', 'RaptorR_Body_Roof_Antennas_'),
               (r'^RaptorR_FenderFlare_F_([LR])$', r'RaptorR_FenderFlare_F\1'), (r'^RaptorR_FenderFlare_Rr_([LR])$', r'RaptorR_FenderFlare_R\1')]
    # the hood skin and its downturned nose are one stamping: join them (and keep the published mesh name)
    hood_ob, nose_ob = bpy.data.objects['RaptorR_Hood_Panel'], bpy.data.objects['RaptorR_Hood_Nose']
    with bpy.context.temp_override(active_object=hood_ob, selected_editable_objects=[hood_ob, nose_ob], object=hood_ob):
        bpy.ops.object.join()
    hood_ob.name = hood_ob.data.name = 'RaptorR_Hood_PowerDome'
    for ob in list(col.objects):
        for pat, rep in renames:
            if re.match(pat, ob.name):
                ob.name = re.sub(pat, rep, ob.name); ob.data.name = ob.name; break
    # ---- pivots to the new hinge lines
    move_pivot('Pivot_Door_FL', (1.152, 0.968, 0.98)); move_pivot('Pivot_Door_FR', (1.152, -0.968, 0.98))
    move_pivot('Pivot_Door_RL', (-0.060, 0.968, 0.98)); move_pivot('Pivot_Door_RR', (-0.060, -0.968, 0.98))
    move_pivot('Pivot_Hood', (D.HOOD_REAR_X - 0.025, 0.0, 1.452))
    # hinge axis just behind the bed floor's rear edge so the lowered tailgate's liner face meets the floor
    report['old_tailgate_pivot'] = list(bpy.data.objects['Pivot_Tailgate'].matrix_world.translation)
    move_pivot('Pivot_Tailgate', (-2.935, 0.0, 0.870))
    move_pivot('Pivot_Fuel_Door', (-1.297, D.Y_SIDE(np.array([-1.297]), np.array([1.205]))[0] + 0.004, 1.205))
    move_pivot('Pivot_Tonneau_Middle', ton['pivots']['Pivot_Tonneau_Middle'])
    move_pivot('Pivot_Tonneau_Rear', ton['pivots']['Pivot_Tonneau_Rear'])
    # ---- parenting
    P = {n: bpy.data.objects[n] for n in ('Pivot_Body_Assembly', 'Pivot_Door_FL', 'Pivot_Door_FR', 'Pivot_Door_RL', 'Pivot_Door_RR', 'Pivot_Hood',
                                          'Pivot_Tailgate', 'Pivot_Fuel_Door', 'Pivot_Tonneau_Middle', 'Pivot_Tonneau_Rear', 'Pivot_Chassis_Frame')}
    def pivot_for(n):
        m = re.search(r'_(F|R)_(L|R)(_|$)', n)
        if re.match(r'RaptorR_(Door_(Skin|Frame|Handle|HandleCup|Shell)|Glass_Door)_', n) and m:
            return 'Pivot_Door_' + m.group(1) + m.group(2)
        if n.startswith('RaptorR_Mirror_'): return 'Pivot_Door_FL' if n.endswith('_FL') else 'Pivot_Door_FR'
        if n.startswith(('RaptorR_Hood_', 'RaptorR_Graphic_Hood')): return 'Pivot_Hood'
        if n.startswith('RaptorR_Tailgate_'): return 'Pivot_Tailgate'
        if n == 'RaptorR_Fuel_Door': return 'Pivot_Fuel_Door'
        if n.startswith(('RaptorR_Tonneau_Middle_',)): return 'Pivot_Tonneau_Middle'
        if n.startswith(('RaptorR_Tonneau_Rear_',)): return 'Pivot_Tonneau_Rear'
        if n.startswith(('RaptorR_Front_SkidPlate', 'RaptorR_Front_TowHooks')): return 'Pivot_Chassis_Frame'
        return 'Pivot_Body_Assembly'
    counts = {}
    for ob in list(col.objects):
        pv = pivot_for(ob.name); parent_keep(ob, P[pv]); counts[pv] = counts.get(pv, 0) + 1
        # move into the scene collection (keep the importer's hierarchy tidy)
        for c in list(ob.users_collection): c.objects.unlink(ob)
        bpy.context.scene.collection.objects.link(ob)
    report['new_meshes_by_pivot'] = counts
    bpy.context.view_layer.update()
    # ---- validation
    hubs = {k: bpy.data.objects['Pivot_Corner_' + k].matrix_world.translation.copy() for k in ('FL', 'FR', 'RL', 'RR')}
    wb = (hubs['FL'].x - hubs['RL'].x + hubs['FR'].x - hubs['RR'].x) / 2
    assert abs(wb - 3.6932) < 0.0005, wb
    def bbox(prefix):
        pts = np.array([o.matrix_world @ v.co for o in bpy.data.objects if o.type == 'MESH' and o.name.startswith(prefix) for v in o.data.vertices])
        return pts.min(0).round(4).tolist(), pts.max(0).round(4).tolist()
    body = np.array([o.matrix_world @ v.co for o in bpy.data.objects if o.type == 'MESH' and o.name.startswith('RaptorR_')
                     and not re.match(r'RaptorR_(Int_|Eng_|Drivetrain|Susp|FOX|Tire|Rim|Brake|WheelCap|Spare|Chassis|Exhaust|DoorCard|Underbody|Axle|Hub|Steer|Diff|Driveshaft|Rear_Hitch)', o.name)
                     for v in o.data.vertices])
    mirrors = np.array([o.matrix_world @ v.co for o in bpy.data.objects if o.type == 'MESH' and o.name.startswith('RaptorR_Mirror_') and 'Cabin' not in o.name for v in o.data.vertices])
    nomirror = body[np.abs(body[:, 1]) < 1.12]
    report['dimensions_mm'] = {
        'length': round(1000 * (body[:, 0].max() - body[:, 0].min()), 1),
        'width_without_mirrors': round(1000 * (nomirror[:, 1].max() - nomirror[:, 1].min()), 1),
        'width_with_mirrors': round(1000 * (mirrors[:, 1].max() - mirrors[:, 1].min()), 1),
        'height': round(1000 * body[:, 2].max(), 1), 'wheelbase': round(1000 * wb, 1)}
    report['doorcard_bbox'] = bbox('RaptorR_DoorCard_'); report['interior_bbox'] = bbox('RaptorR_Int_'); report['engine_bbox'] = bbox('RaptorR_Eng_')
    letter_material = bpy.data.materials.get('Mat_FORD_Letter_Face')
    if letter_material and letter_material.use_nodes:
        for node in letter_material.node_tree.nodes:
            if node.type == 'NORMAL_MAP':
                node.inputs['Strength'].default_value = 0.15
    # ---- export
    out.parent.mkdir(parents=True, exist_ok=True)
    if blend: bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    bpy.ops.export_scene.gltf(filepath=str(out.resolve()), export_format='GLB', export_animations=False, export_cameras=False, export_lights=False,
                              export_apply=True, export_attributes=True, export_extras=True, export_yup=True, export_image_format='AUTO')
    out.with_suffix('.build.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print('RAPTOR_BUILD ' + json.dumps(report, default=str))

if __name__ == '__main__':
    main()
