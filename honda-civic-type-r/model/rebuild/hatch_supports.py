"""Named FK8 lift supports. Each local +Y axis points from its socket to the other socket."""
import numpy as np
from scipy.spatial.transform import Rotation


def build_lift_supports(model):
    doc = model.document
    indices = {node['name']: i for i,node in enumerate(doc['nodes'])}
    materials = {m['name']: i for i,m in enumerate(doc['materials'])}

    def add(name, parent, material, start, end, profile):
        # Sweep the socket, the tube and its cap as one closed turned mesh.
        angle = np.arange(16)*2*np.pi/16
        rings = [np.column_stack([r*np.cos(angle), np.full(16,y), r*np.sin(angle)]) for y,r in profile]
        points = np.vstack(rings)
        idx = np.arange(len(points)).reshape(-1,16)
        a,b,c,d = idx[:-1].ravel(),idx[1:].ravel(),np.roll(idx[1:],-1,axis=1).ravel(),np.roll(idx[:-1],-1,axis=1).ravel()
        faces = np.vstack([np.column_stack([a,b,c]),np.column_stack([a,c,d])])
        for cap in (idx[0], idx[-1, ::-1]):
            triangles = np.column_stack([np.full(len(cap)-2,cap[0]),cap[1:-1],cap[2:]])
            faces = np.vstack([faces,triangles])
        normals = np.zeros_like(points)
        tri = points[faces]
        face_n = np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        for j in faces.T:
            np.add.at(normals,j,face_n)
        normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        direction = (end-start)/np.linalg.norm(end-start)
        axis = np.cross([0.,1.,0.],direction)
        rotation = Rotation.from_rotvec(axis/np.linalg.norm(axis)*np.arccos(direction[1]))
        offset = np.asarray(model.nodes[parent].get('translation',[0.,0.,0.]))
        model.add_mesh(name,indices[parent],materials[material],start-offset,rotation.as_quat(),points,normals,faces)

    for suffix, sign in [('L',-1.),('R',1.)]:
        body = np.array([-1.397,1.162,sign*.574])
        hatch = np.array([-1.80,1.143,sign*.607])
        add('Civic_Body_Shell_Lift_Support_'+suffix,'Pivot_Body_Assembly','Mat_Gloss_Black_Trim',body,hatch,
            [(-.008,.001),(-.006,.011),(.005,.014),(.019,.010),(.032,.012),(.385,.012),(.394,.009),(.395,.001)])
        add('Civic_Hatch_Lift_Rod_'+suffix,'Pivot_Hatch_Assembly','Mat_Engine_Machined_Aluminum',hatch,body,
            [(-.007,.001),(-.004,.011),(.007,.013),(.021,.006),(.401,.005),(.402,.001)])
