"""Refine the assembled FK8 GLB, preserving its rig and untouched compressed meshes.

Usage: python refine_exterior.py assembled.draco.glb output.glb
Requires numpy, scipy and DracoPy >= 2.2.
"""
import argparse
import json
import re
import struct
from pathlib import Path

import DracoPy
import numpy as np
from scipy.spatial import cKDTree

from bumper_vents import refine_bumper_vents
from finishes import set_gltf_finishes
from headlamps import refine_headlamps
from rear_diffuser import refine_rear_diffuser


class CivicModel:
    """Read and replace the single-primitive Draco meshes in the FK8 export."""

    def __init__(self, path):
        data = path.read_bytes()
        json_size = struct.unpack_from('<I', data, 12)[0]
        self.document = json.loads(data[20:20 + json_size])
        binary = memoryview(data)[28 + json_size:]
        self.payloads = []
        for view in self.document['bufferViews']:
            start = view.get('byteOffset', 0)
            self.payloads.append(binary[start:start + view['byteLength']])
        self.nodes = {node['name']: node for node in self.document['nodes']}

    def primitive(self, name):
        primitive, = self.document['meshes'][self.nodes[name]['mesh']]['primitives']
        return primitive

    def _decode(self, name):
        ext = self.primitive(name)['extensions']['KHR_draco_mesh_compression']
        return DracoPy.decode(bytes(self.payloads[ext['bufferView']]))

    def read_geometry(self, name):
        mesh = self._decode(name)
        return mesh.points.astype(float), mesh.normals.astype(float), mesh.faces

    def read_uv(self, name):
        coordinates = self._decode(name).tex_coord
        return None if coordinates is None else np.ascontiguousarray(coordinates, dtype=float)

    def replace_geometry(self, name, positions, normals, faces, *, uv=None):
        primitive = self.primitive(name)
        ext = primitive['extensions']['KHR_draco_mesh_compression']
        expected = {'POSITION', 'NORMAL'} if uv is None else {'POSITION', 'NORMAL', 'TEXCOORD_0'}
        if expected != set(primitive['attributes']):
            raise ValueError(f'{name}: missing or unexpected vertex attributes')
        if not (np.isfinite(positions).all() and np.isfinite(normals).all()):
            raise ValueError(f'{name}: non-finite geometry')
        encoded = DracoPy.encode(np.asarray(positions, dtype=np.float64),
                                 np.ascontiguousarray(faces, dtype=np.uint32),
                                 normals=np.asarray(normals, dtype=np.float64),
                                 tex_coord=uv, quantization_bits=16,
                                 normal_quantization_bits=12, compression_level=7)
        mesh = DracoPy.decode(encoded)
        semantic = {0: 'POSITION', 1: 'NORMAL', 3: 'TEXCOORD_0'}
        ext['attributes'] = {semantic[a['attribute_type']]: a['unique_id'] for a in mesh.attributes}
        self.payloads[ext['bufferView']] = encoded
        accessors = self.document['accessors']
        for key, index in primitive['attributes'].items():
            accessor = accessors[index]
            accessor['count'] = len(mesh.points)
            if key == 'POSITION':
                accessor.update(min=mesh.points.min(axis=0).tolist(), max=mesh.points.max(axis=0).tolist())
        indices = accessors[primitive['indices']]
        indices['count'] = mesh.faces.size
        indices['componentType'] = 5125 if len(mesh.points) > 65535 else 5123

    def write(self, path):
        binary = bytearray()
        for view, payload in zip(self.document['bufferViews'], self.payloads):
            binary.extend(b'\0' * (-len(binary) % 4))
            view.update(byteOffset=len(binary), byteLength=len(payload))
            binary.extend(payload)
        self.document['buffers'][0]['byteLength'] = len(binary)
        binary.extend(b'\0' * (-len(binary) % 4))
        metadata = json.dumps(self.document, separators=(',', ':')).encode()
        metadata += b' ' * (-len(metadata) % 4)
        path.write_bytes(struct.pack('<4sII', b'glTF', 2, 28 + len(metadata) + len(binary))
                         + struct.pack('<I4s', len(metadata), b'JSON') + metadata
                         + struct.pack('<I4s', len(binary), b'BIN\0') + binary)


def unit(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def panel_edges(faces):
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges, count = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    return edges[count == 1]


def edge_fade(points, faces, width):
    """Hold the shut lines in place and blend into the corrected panel interior."""
    border = np.unique(panel_edges(faces))
    if not len(border):
        return np.ones(len(points))
    distance = cKDTree(points[border]).query(points)[0]
    t = np.clip(distance / width, 0, 1)
    return t * t * (3 - 2 * t)


def smooth_pressing(model, name):
    """Fair small hull ridges without averaging across the stamped character lines."""
    points, normals, faces = model.read_geometry(name)
    tree = cKDTree(points)
    fade = edge_fade(points, faces, .030)
    result = points.copy()
    result_normals = normals.copy()
    # Fit a local tangent plane using aligned neighbours. Limit the normal
    # displacement to 3 mm so the panel still meets its seals and adjacent trim.
    for chunk in np.array_split(np.arange(len(points)), max(1, len(points) // 256)):
        distances, ids = tree.query(points[chunk], k=min(180, len(points)))
        alignment = np.einsum('ij,ikj->ik', normals[chunk], normals[ids])
        weight = np.exp(-0.5 * (distances / .043)**2 - 0.5 * ((1 - alignment) / .035)**2)
        weight /= weight.sum(axis=1, keepdims=True)
        normal = unit(np.einsum('ik,ikj->ij', weight, normals[ids]))
        offset = np.einsum('ij,ij->i', np.einsum('ik,ikj->ij', weight, points[ids]) - points[chunk], normal)
        result[chunk] += (fade[chunk] * np.clip(offset, -.003, .003))[:, None] * normal
        result_normals[chunk] = normal
    if name == 'Civic_Hood_Panel':
        result, result_normals = hood_character_lines(result, result_normals)
    model.replace_geometry(name, result, result_normals, faces)


def hood_character_lines(points, normals):
    """Form the two pressed ridges which widen from the nose towards the cowl.

    Their centres follow the visible breaks in Honda's frame 07 projected on
    the hood. The inboard face rolls up more gradually than the outboard face.
    """
    def relief(x, z):
        center = .445 + .205 * (1.96 - x)
        across = np.abs(z) - center
        width = np.where(across < 0, .068, .024)
        fade = np.clip((2.015 - x) / .17, 0, 1) * np.clip((x - 1.125) / .12, 0, 1)
        return .017 * fade**2 * (3 - 2 * fade) * np.exp(-.5 * (across / width)**2)

    x, _, z = points.T
    step = .0005
    dx = (relief(x + step, z) - relief(x - step, z)) / (2 * step)
    dz = (relief(x, z + step) - relief(x, z - step)) / (2 * step)
    corrected = points.copy()
    corrected[:, 1] += relief(x, z)
    adjusted = normals.copy()
    adjusted[:, 0] -= normals[:, 1] * dx
    adjusted[:, 2] -= normals[:, 1] * dz
    return corrected, unit(adjusted)


def form_glass(model, name):
    """Fit a continuous bent pane while retaining its cut perimeter against the seals."""
    points, normals, faces = model.read_geometry(name)
    center = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points - center, full_matrices=False)
    if np.dot(axes[2], normals.mean(axis=0)) < 0:
        axes[2] *= -1
    u, v, height = ((points - center) @ axes.T).T
    powers = [(i, j) for i in range(4) for j in range(4 - i)]
    basis = np.column_stack([u**i * v**j for i, j in powers])
    coeffs = np.linalg.lstsq(basis, height, rcond=None)[0]
    du = sum(c * i * u**(i - 1) * v**j for c, (i, j) in zip(coeffs, powers) if i)
    dv = sum(c * j * u**i * v**(j - 1) for c, (i, j) in zip(coeffs, powers) if j)
    smooth_normals = unit(axes[2] - du[:, None] * axes[0] - dv[:, None] * axes[1])
    delta = np.clip(basis @ coeffs - height, -.008, .008) * edge_fade(points, faces, .050)
    model.replace_geometry(name, points + delta[:, None] * axes[2], smooth_normals, faces)


def form_headliner(model):
    """Form the fabric insert from the roof pressing inside the cabin pivot."""
    roof, normals, faces = model.read_geometry('Civic_Body_Shell_Roof')
    points = roof - .024 * normals
    points[:, 2] *= 1.045  # tuck the edge under the roof-side trim
    points -= model.nodes['Pivot_Interior_Cabin']['translation']
    uv = np.ascontiguousarray(points[:, [0, 2]])
    model.replace_geometry('Civic_Interior_Headliner', points, -normals, faces[:, ::-1], uv=uv)


def refine_body(model):
    pressing = re.compile(r'Civic_(Body_Shell_(Roof|FrontFender(_Top)?_[LR]|RearQuarter_[LR])|'
                          r'Hood_Panel(_Front)?|Door_Shell_(L|R|RL|RR)|Hatch_Panel)')
    for name in model.nodes:
        if pressing.fullmatch(name):
            smooth_pressing(model, name)
        elif name.startswith('Civic_Glass_'):
            form_glass(model, name)
    form_headliner(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    car = CivicModel(args.source)
    if any(m['name'] == 'Mat_FK8_Headlamp_Reflector_Aluminum' for m in car.document['materials']):
        parser.error('The input already has the FK8 refinements. Supply the original assembly.')
    set_gltf_finishes(car.document)
    refine_body(car)
    refine_headlamps(car)
    refine_bumper_vents(car)
    refine_rear_diffuser(car)
    car.write(args.output)
