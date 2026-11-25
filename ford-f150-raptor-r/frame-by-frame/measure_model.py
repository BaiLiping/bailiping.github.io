"""Measure the assembled GLB in world space using Blender's glTF/Draco importer.

blender -b --python-exit-code 1 --python measure_model.py -- truck.glb measurements.json
Add a previous GLB as the third argument to include its measurements.
"""
import hashlib
import json
from pathlib import Path
import sys

import bpy
from mathutils.bvhtree import BVHTree
import numpy as np


FORD = {
    "length": 232.6,
    "wheelbase": 145.4,
    "width_without_mirrors": 87.0,
    "width_with_mirrors": 96.0,
    "height": 80.6,
    "front_track": 74.3,
    "rear_track": 73.9,
    "ground_clearance": 13.1,
}
SOURCE = "https://www.fromtheroad.ford.com/content/dam/fordmediasite/us/en/library/2024/specs/2024-Ford-Raptor-and-Raptor-R-Tech-Specs.pdf"


def sheet_clearance(meshes, vertices, first, second):
    def surface(name):
        polygons = [tuple(p.vertices) for p in meshes[name].data.polygons]
        return BVHTree.FromPolygons(vertices[name].tolist(), polygons)

    def edges(name):
        indices = [tuple(e.vertices) for e in meshes[name].data.edges]
        return vertices[name][indices]

    a, b = surface(first), surface(second)
    if a.overlap(b):
        return 0.0

    # For disjoint triangles the closest features are vertex/face or edge/edge.
    best = min(
        min(b.find_nearest(tuple(v))[3] for v in vertices[first]),
        min(a.find_nearest(tuple(v))[3] for v in vertices[second]),
    )
    other = edges(second)
    lower = other.min(axis=1)
    upper = other.max(axis=1)
    for start, end in edges(first):
        nearby = (
            (upper >= np.minimum(start, end) - best)
            & (lower <= np.maximum(start, end) + best)
        ).all(axis=1)
        candidates = other[nearby]
        if not len(candidates):
            continue
        direction = end - start
        directions = candidates[:, 1] - candidates[:, 0]
        offsets = start - candidates[:, 0]
        aa = direction @ direction
        bb = directions @ direction
        cc = np.einsum("ij,ij->i", directions, directions)
        dd = offsets @ direction
        ee = np.einsum("ij,ij->i", directions, offsets)
        determinant = aa * cc - bb * bb
        nonparallel = determinant > 1e-12 * aa * cc
        s = np.divide(
            bb * ee - cc * dd, determinant, out=np.zeros_like(bb), where=nonparallel
        )
        t = np.divide(
            aa * ee - bb * dd, determinant, out=np.zeros_like(bb), where=nonparallel
        )
        interior = nonparallel & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
        if interior.any():
            delta = (
                offsets[interior]
                + s[interior, None] * direction
                - t[interior, None] * directions[interior]
            )
            best = min(best, np.linalg.norm(delta, axis=1).min())
    return best


def measure(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(path.resolve()))
    bpy.context.view_layer.update()
    meshes = {o.name: o for o in bpy.context.scene.objects if o.type == "MESH"}
    vertices = {}
    for name, obj in meshes.items():
        coords = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
        obj.data.vertices.foreach_get("co", coords)
        transform = np.asarray(obj.matrix_world)
        vertices[name] = coords.reshape(-1, 3) @ transform[:3, :3].T + transform[:3, 3]

    points = np.vstack(list(vertices.values()))
    body = np.vstack(
        [p for name, p in vertices.items() if not name.startswith("RaptorR_Mirror_")]
    )
    running_gear = (
        "RaptorR_Tire_",
        "RaptorR_Rim_",
        "RaptorR_WheelCap_",
        "RaptorR_Brake",
    )
    clearance = min(
        p[:, 2].min()
        for name, p in vertices.items()
        if not name.startswith(running_gear)
    )
    contact = min(
        vertices["RaptorR_Tire_" + k][:, 2].min() for k in ("FL", "FR", "RL", "RR")
    )
    hubs = {
        k: np.asarray(bpy.data.objects["Pivot_Corner_" + k].matrix_world.translation)
        for k in ("FL", "FR", "RL", "RR")
    }
    measured = {
        "length": np.ptp(points[:, 0]),
        "width_without_mirrors": np.ptp(body[:, 1]),
        "width_with_mirrors": np.ptp(points[:, 1]),
        "height": points[:, 2].max() - contact,
        "wheelbase": np.mean([hubs["F" + s][0] - hubs["R" + s][0] for s in ("L", "R")]),
        "front_track": hubs["FL"][1] - hubs["FR"][1],
        "rear_track": hubs["RL"][1] - hubs["RR"][1],
        "ground_clearance": clearance - contact,
    }
    mm = lambda m: round(float(m) * 1000, 2)
    rail_face = vertices["RaptorR_Bed_RailCap_L"][:, 0].max()
    bedside = max(vertices["RaptorR_Body_BedSide_" + s][:, 0].max() for s in ("L", "R"))
    guard_gap = sheet_clearance(
        meshes, vertices, "RaptorR_Front_SkidPlate", "RaptorR_Chassis_MidSkid"
    )
    geometry = {
        "bedside_forward_of_rail_mm": mm(bedside - rail_face),
        "cab_wall_to_bedside_longitudinal_gap_mm": mm(
            vertices["RaptorR_Body_Cab_RearWall"][:, 0].min() - bedside
        ),
        "front_to_middle_skid_surface_gap_mm": mm(guard_gap),
    }
    if "RaptorR_Tonneau_Front_Panel" in vertices:
        rail_top = vertices["RaptorR_Bed_RailCap_L"][:, 2].max()
        geometry["tonneau_panel_tops_above_rail_mm"] = [
            mm(vertices["RaptorR_Tonneau_" + section + "_Panel"][:, 2].max() - rail_top)
            for section in ("Front", "Middle", "Rear")
        ]
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "measured_mm": {key: mm(measured[key]) for key in FORD},
        "geometry": geometry,
    }


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1 :]
    asset, output = map(Path, args[:2])
    report = {"source": SOURCE, "ford_inches": FORD, "current": measure(asset)}
    if len(args) > 2:
        report["previous"] = measure(Path(args[2]))
    output.write_text(json.dumps(report, indent=2) + "\n")
