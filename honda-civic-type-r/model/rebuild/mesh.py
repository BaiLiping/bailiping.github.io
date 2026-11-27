"""Mesh a body field: marching cubes, vertices projected onto the zero level, normals from the field gradient.

Run: venv/bin/python mesh.py <field.npz> <out.glb> [smooth_sigma_voxels=1.0] [step=1]
Writes a GLB (one mesh, smooth custom normals) for viewing and silhouette checks, and <out>-mesh.npz with
V (glTF metres), F (faces), N (normals). Returns nothing.
"""
import sys
import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure


def trilinear(A, idx):
    """Takes a 3D array and fractional indices (N,3); returns trilinear samples."""
    return ndimage.map_coordinates(A, idx.T, order=1, mode='nearest')


def main():
    d = np.load(sys.argv[1]); F = d['F']; h = float(d['h']); o = np.array([d['x0'], d['y0'], d['z0']])
    sig = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    Fs = ndimage.gaussian_filter(F, sig) if sig > 0 else F
    step = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    V, Fc, _, _ = measure.marching_cubes(Fs, 0.0, step_size=step)
    gx, gy, gz = np.gradient(Fs)
    for _ in range(3):                                # Newton steps onto the smoothed zero level
        f = trilinear(Fs, V)
        g = np.stack([trilinear(gx, V), trilinear(gy, V), trilinear(gz, V)], 1)
        V = V - (f / np.maximum((g * g).sum(1), 1e-9))[:, None] * g
    g = np.stack([trilinear(gx, V), trilinear(gy, V), trilinear(gz, V)], 1)
    N = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
    P = o + V * h
    m = trimesh.Trimesh(P, Fc, vertex_normals=N, process=False)
    # marching cubes winding: make faces agree with the outward field gradient
    fn = np.cross(P[Fc[:, 1]] - P[Fc[:, 0]], P[Fc[:, 2]] - P[Fc[:, 0]])
    if (fn * N[Fc].mean(1)).sum() < 0:
        Fc = Fc[:, ::-1]; m = trimesh.Trimesh(P, Fc, vertex_normals=N, process=False)
    m.export(sys.argv[2])
    np.savez_compressed(sys.argv[2].replace('.glb', '-mesh.npz'), V=P.astype(np.float32), F=Fc.astype(np.int32), N=N.astype(np.float32))
    print(f'mesh {len(P)} verts {len(Fc)} faces; bbox {P.min(0).round(3)} {P.max(0).round(3)}')


if __name__ == '__main__':
    main()
