"""Approach, departure and breakover angles of an exported Raptor R GLB.

blender -b --python-exit-code 1 --python check_offroad_angles.py -- truck.glb

Each angle is the steepest ground line tangent to a tire (radius 0.4635 m) that clears the lowest body points.
Tires, wheels, brakes, cabin, door cards and suspension links are excluded. Ford: 33.1, 24.9 and 24.4 degrees.
"""
import bpy, sys, re, numpy as np
glb=sys.argv[sys.argv.index('--')+1]
bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.gltf(filepath=glb); bpy.context.view_layer.update()
R=0.4635; hub={k: np.array(bpy.data.objects['Pivot_Corner_'+k].matrix_world.translation) for k in ('FL','RL')}
P=[]; N=[]
for o in bpy.data.objects:
    if o.type!='MESH' or re.match(r'RaptorR_(Tire|Rim|WheelCap|Brake|Int_|DoorCard|Susp|FOX)', o.name): continue
    M=np.array(o.matrix_world); V=np.array([v.co for v in o.data.vertices])@M[:3,:3].T+M[:3,3]; V=V[V[:,2]<0.95]
    if len(V): P.append(V); N+= [o.name]*len(V)
P=np.vstack(P); N=np.array(N)
def end_angle(front):
    C=hub['FL' if front else 'RL']; s=(P[:,0]>C[0]+0.2) if front else (P[:,0]<C[0]-0.2); Q=P[s]
    dx=np.abs(Q[:,0]-C[0]); dz=Q[:,2]-C[2]; d=np.hypot(dx,dz); a=np.degrees(np.arctan2(dz,dx)+np.arcsin(np.clip(R/d,0,1)))
    i=np.argmin(a); return round(float(a[i]),2), N[s][i]
def tangent(px,pz,xc):
    dx=abs(xc-px); a=np.linspace(0.001,1.2,4000); v=np.abs((R-pz)+np.tan(a)*dx)/np.sqrt(1+np.tan(a)**2)-R
    i=np.where(np.diff(np.sign(v)))[0]; return a[i[0]] if len(i) else np.nan
xf, xr = hub['FL'][0], hub['RL'][0]
s=(P[:,0]>xr+0.55)&(P[:,0]<xf-0.55)&(P[:,2]<0.6)&(np.abs(P[:,1])<0.75); Q=P[s][::5]
b=min(np.degrees(tangent(p[0],p[2],xf)+tangent(p[0],p[2],xr)) for p in Q)
(ap, an), (dp, dn) = end_angle(True), end_angle(False)
print(f'approach {ap} deg ({an}) | departure {dp} deg ({dn}) | breakover {round(float(b), 2)} deg')
