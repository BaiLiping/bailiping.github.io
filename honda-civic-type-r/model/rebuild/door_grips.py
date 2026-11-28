"""Fit the four painted grips and their finger cups to the actual FK8 doors."""
import numpy as np
from scipy.spatial import cKDTree


def refine_grips(model):
    for side, sign in [('L', -1), ('R', 1), ('RL', -1), ('RR', 1)]:
        shell = 'Civic_Door_Shell_'+side
        pocket = 'Civic_Door_Handle_Pocket_'+side
        grip = 'Civic_Door_Handle_'+side
        cup, _, cf = model.read_geometry(pocket)
        old, _, _ = model.read_geometry(grip)
        skin, sn, _ = model.read_geometry(shell)
        x0, x1 = old[:,0].min(), old[:,0].max()
        cx, cy = cup[:,:2].mean(axis=0)
        sel = (abs(skin[:,0]-cx)<.22)&(abs(skin[:,1]-cy)<.084)&(sn[:,2]*sign>.55)
        def basis(x,y):
            x,y = x-cx,y-cy
            return np.column_stack([np.ones_like(x),x,y,x*x,x*y,y*y])
        coeff = np.linalg.lstsq(basis(*skin[sel,:2].T), skin[sel,2]*sign,rcond=None)[0]
        def side_skin(x,y):
            return basis(x,y)@coeff

        # Follow the original cut right up to its shared perimeter. Shape
        # the bowl from the projected edge, where the door and cup meet.
        edges = np.sort(cf[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)
        ed, count = np.unique(edges,axis=0,return_counts=True)
        outline = cup[np.unique(ed[count==1]),:2]
        tree = cKDTree(outline)
        x,y = cup[:,:2].T
        def inner(x,y):
            dist = tree.query(np.column_stack([x,y]))[0]
            t = np.clip(dist/.036,0,1)
            return side_skin(x,y)-.020*t*t*(3-2*t)
        dist = tree.query(cup[:,:2])[0]
        w = np.clip(dist/.009,0,1); w=w*w*(3-2*w)
        cup[:,2] += w*(sign*inner(x,y)-cup[:,2])
        h=.001
        vn=np.column_stack([-(inner(x+h,y)-inner(x-h,y))/(2*h),
                            -(inner(x,y+h)-inner(x,y-h))/(2*h),np.full(len(cup),sign)])
        vn/=np.linalg.norm(vn,axis=1,keepdims=True)
        model.replace_geometry(pocket,cup,vn,cf)
        model.primitive(pocket)['material']=model.primitive(shell)['material']

        # Longitudinal rings give the grip a rounded nose and a continuous
        # painted crown. The two feet enter the door on either side of the cup.
        rear = len(side)==2
        center_y=.905 if rear else .832
        theta=np.arange(36)*2*np.pi/36
        stations=np.r_[.001,.004,.012,np.arange(.025,.981,.012),.989,.997,.999]
        rings=[]
        for t in stations:
            x=x0+(x1-x0)*t
            end=min(t/.093,(1-t)/.105,1)
            rounding=np.sqrt(max(0,1-(1-end)**2))
            mid=np.sin(np.pi*t)
            y=center_y+(.036 if rear else .018)*(.5-t)
            ys=y+(.014+.004*(1-t))*rounding*np.sin(theta)
            raised=.004+.018*mid**.72
            zs=side_skin(np.full(len(theta),x),ys)+raised+.0125*rounding*np.cos(theta)
            rings.append(np.column_stack([np.full(len(theta),x),ys,sign*zs]))
        p=np.vstack(rings)
        grid=np.arange(len(p)).reshape(len(rings),-1)
        a,b,c,d=grid[:-1].ravel(),grid[1:].ravel(),np.roll(grid[1:],-1,axis=1).ravel(),np.roll(grid[:-1],-1,axis=1).ravel()
        faces=np.vstack([np.column_stack([a,b,c]),np.column_stack([a,c,d])])
        for ring in (grid[0],grid[-1,::-1]):
            faces=np.vstack([faces,np.column_stack([np.full(len(ring)-2,ring[0]),ring[1:-1],ring[2:]])])
        if sign<0: faces=faces[:,::-1]
        normal=np.zeros_like(p)
        tri=p[faces];area=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        for col in faces.T:np.add.at(normal,col,area)
        normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
        model.replace_geometry(grip,p,normal,faces)
