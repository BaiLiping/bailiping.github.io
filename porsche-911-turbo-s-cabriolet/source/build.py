"""Original metric reconstruction of Porsche 992.2 Turbo S Cabriolet, model 992652.
Run Blender in background: blender -b -P source/build.py. Coordinates below are
X driver-left, Y up, Z forward; convert to Blender Z-up at the mesh boundary.
"""
import bpy, math, sys
from math import sin,cos,pi,sqrt,exp
from pathlib import Path
from mathutils import Vector
from collections import defaultdict
R=Path(__file__).resolve().parent.parent
bpy.ops.wm.read_factory_settings(use_empty=True)
def mat(name,col,metal=0,rough=.35,alpha=1):
 m=bpy.data.materials.new(name);m.diffuse_color=(*col,alpha);m.use_nodes=True
 b=m.node_tree.nodes.get('Principled BSDF');b.inputs['Base Color'].default_value=(*col,1);b.inputs['Metallic'].default_value=metal;b.inputs['Roughness'].default_value=rough;b.inputs['Alpha'].default_value=alpha
 if alpha<1 and hasattr(m,'surface_render_method'):m.surface_render_method='DITHERED'
 if metal>.45:b.inputs['Coat Weight'].default_value=.38
 m.use_backface_culling=False;return m
blue=mat('Gentian Blue Metallic 1A',(.018,.041,.148),.57,.27)
black=mat('Satin black intakes',(.013,.015,.018),.22,.36)
rubber=mat('Tire tread',(.021,.023,.025),.04,.75)
well=mat('Wheel housings and recesses',(.006,.007,.009),0,.86)
seal=mat('EPDM glazing and joint seals',(.012,.014,.017),.03,.65)
grey=mat('Turbonite wheels',(.210,.202,.176),.72,.30)
brushed=mat('Diamond cut faces',(.40,.41,.42),.70,.31)
carbon=mat('Ceramic disc',(.047,.054,.060),.34,.56)
yellow=mat('PCCB yellow',(.94,.54,.012),.28,.28)
glass=mat('Clear tinted glazing',(.038,.057,.068),.18,.16,.72)
lens=mat('Dark chrome headlight housing',(.037,.047,.054),.62,.22)
reflector=mat('LED optics',(.65,.77,.85),.64,.16)
red=mat('Rear light lenses',(.48,.009,.013),.35,.22)
cloth=mat('Black convertible fabric',(.018,.019,.023),.02,.86)
leather=mat('Black leather',(.035,.036,.042),.01,.58)
screen=mat('Display glass',(.014,.022,.027),.37,.16)
lit=mat('Instrument markings',(.57,.66,.68),.06,.35)
outputs={};groups={};parent='Bodywork'
def cv(p):return (p[0],-p[2],p[1])
def group(n):
 if n not in groups:
  ob=bpy.data.objects.new(n,None);bpy.context.scene.collection.objects.link(ob);groups[n]=ob
 return groups[n]
def mesh(name,vs,fs,m,smooth=True):
 me=bpy.data.meshes.new(name);me.from_pydata([cv(p) for p in vs],[],fs);me.materials.append(m);me.update()
 ob=bpy.data.objects.new(name,me);bpy.context.scene.collection.objects.link(ob);ob.parent=group(parent)
 if smooth:
  for p in me.polygons:p.use_smooth=True
 return ob
def patch(n,nu,nv,fn,m):
 vs=[fn(i/nu,j/nv) for j in range(nv+1) for i in range(nu+1)]
 fs=[(j*(nu+1)+i,j*(nu+1)+i+1,(j+1)*(nu+1)+i+1,(j+1)*(nu+1)+i) for j in range(nv) for i in range(nu)]
 return mesh(n,vs,fs,m)
def linterp(k,t):
 if t<=k[0][0]:return k[0][1]
 if t>=k[-1][0]:return k[-1][1]
 for i,((a,y),(b,v)) in enumerate(zip(k,k[1:])):
  if t<=b:
   h=(t-a)/(b-a);pl=(v-k[max(0,i-1)][1])/(b-k[max(0,i-1)][0]);pr=(k[min(i+2,len(k)-1)][1]-y)/(k[min(i+2,len(k)-1)][0]-a)
   if i>0 and (k[i-1][1]-y)*(v-y)>=0:pl=0
   if i+2<len(k) and (k[i+2][1]-v)*(y-v)>=0:pr=0
   return (2*h**3-3*h**2+1)*y+(h**3-2*h**2+h)*(b-a)*pl+(-2*h**3+3*h**2)*v+(h**3-h**2)*(b-a)*pr
def blend(a,b,t):return tuple((1-t)*x+t*y for x,y in zip(a,b))
def tube(n,points,r,m,sides=8,closed=False):
 P=[Vector(v) for v in points];vs=[];fs=[]
 for i,p in enumerate(P):
  j=(i-1)%len(P) if closed else max(0,i-1);k=(i+1)%len(P) if closed else min(len(P)-1,i+1)
  tangent=(P[k]-P[j]).normalized();ref=Vector((0,1,0)) if abs(tangent.y)<.94 else Vector((0,0,1))
  a=tangent.cross(ref).normalized();b=tangent.cross(a)
  for h in range(sides):vs.append(p+r*(a*cos(2*pi*h/sides)+b*sin(2*pi*h/sides)))
 for i in range(len(P) if closed else len(P)-1):
  for h in range(sides):
   j=(i+1)%len(P);k=(h+1)%sides;fs.append((i*sides+h,i*sides+k,j*sides+k,j*sides+h))
 if not closed:fs.extend([tuple(range(sides-1,-1,-1)),tuple((len(P)-1)*sides+h for h in range(sides))])
 return mesh(n,vs,fs,m)
def ellipse(n,c,r,m,nu=32,nv=16):
 return patch(n,nu,nv,lambda a,b:(c[0]+r[0]*cos(2*pi*a)*sin(pi*b),c[1]+r[1]*cos(pi*b),c[2]+r[2]*sin(2*pi*a)*sin(pi*b)),m)
def roundrect(cx,cy,rx,ry,p=4,N=64):
 return [(cx+rx*math.copysign(abs(cos(2*pi*i/N))**(2/p),cos(2*pi*i/N)),cy+ry*math.copysign(abs(sin(2*pi*i/N))**(2/p),sin(2*pi*i/N))) for i in range(N)]
def path(n,pts,r,m):return tube(n,pts,r,m,sides=8)
def filled(n,pts,m):
 mid=tuple(sum(p[k] for p in pts)/len(pts) for k in range(3));return mesh(n,[mid]+pts,[(0,i+1,(i+1)%len(pts)+1) for i in range(len(pts))],m)
def plate(n,pts,project,m,rim=None,th=.006):
 loop=[project(x,y) for x,y in pts];filled(n,loop,m)
 if rim:tube(n+' surround',loop,th,rim,9,True)
def shaft(n,a,b,r,m):return tube(n,[a,b],r,m,24)
def sidew(z):return linterp([(-2.31,.851),(-1.76,.930),(-1.21,.949),(-.90,.928),(-.25,.863),(.47,.859),(1.18,.916),(1.53,.909),(1.90,.878),(2.241,.834)],z)
def centerheight(z):return linterp([(-2.31,.785),(-1.98,.916),(-1.62,.976),(-1.15,.979),(-.91,.950),(.59,.853),(.83,.876),(1.23,.853),(1.57,.801),(1.89,.710),(2.10,.604),(2.241,.49)],z)
def shoulder(z):return linterp([(-2.31,.714),(-2.08,.831),(-1.65,.906),(-1.27,.966),(-.95,.960),(-.55,.901),(.17,.855),(.65,.854),(1.17,.859),(1.43,.868),(1.66,.844),(1.88,.739),(2.241,.512)],z)
def displacement(z,u):
 # Elliptical plan corners have a side-facing tangent at the fender junction.
 corner=1-sqrt(max(0,1-min(1,abs(u))**2.5))
 return max(0,(z-1.0)/1.241)**2*(-.402)*corner+max(0,(-z-1.05)/1.26)**2*.368*corner
def top(t,z):
 u=abs(t);h=centerheight(z)+(shoulder(z)-centerheight(z))*u**1.8
 # Arched fender peaks rise above the depressed center of the front bonnet.
 h+=.023*exp(-((z-1.52)/.51)**2)*exp(-((u-.79)/.19)**2)
 h+=.025*exp(-((z+1.32)/.55)**2)*sin(pi*u)**3
 h-=.048*u**6
 h-=.049*exp(-((z-1.84)/.28)**4)*exp(-((u-.78)/.22)**4)
 x=sidew(z)*t*.962;zz=z+displacement(z,u)
 # Stamp each headlamp bowl into the front fender so the true round frontal
 # outline is independent of the outer fender's local shoulder slope.
 lu=(abs(x)-.695)/.118;lv=(zz-1.762)/.16;rho=sqrt(lu*lu+lv*lv)
 fade=max(0,min(1,(1.43-rho)/.43));fade=fade*fade*(3-2*fade)
 lensplane=.710-.117*lv-.005*lu
 h=h*(1-fade)+(lensplane-.007)*fade
 return (x,h,zz)
def bottom(z):
 return linterp([(-2.31,.218),(-1.88,.168),(-1.2,.125),(-.30,.12),(.75,.12),(1.62,.128),(2.04,.162),(2.241,.188)],z)
def arch(z):
 h=bottom(z)
 for ax,rad,zz in [(1.225,.367,.34325),(-1.225,.389,.36420)]:
  d=abs(z-ax)
  if d<rad:h=max(h,zz+sqrt(max(0,rad*rad-d*d)))
  elif d<rad+.029:h=max(h,zz-(zz-bottom(z))*(d-rad)/.029)
 return h
def side(s,z,t):
 # Cross section from the arch lip or sill to the shoulder.
 x0,ht,realz=top(s,z);hb=arch(realz);y=hb+(ht-hb)*t
 bulge=sin(pi*t)**.70*.033+(.012*exp(-((t-.80)/.14)**2))
 return (s*(sidew(z)*(.969-.007*t)+bulge-.032*(1-t)**3*max(0,(abs(realz)-1.6)/.65)),y,realz)
def sxyz(s,z,y):
 # Invert the effective side coordinate at the rounded ends.
 zz=z
 for _ in range(5):zz=z-displacement(zz,1)
 topy=top(s,zz)[1];t=(y-arch(z))/max(.005,topy-arch(z))
 p=side(s,zz,max(0,min(1,t)));return (p[0]+s*.004,y,z)
def topside(x,z,offset=.002):
 zz=z
 for i in range(4):
  u=min(1,abs(x)/(.962*sidew(zz)));zz=z-displacement(zz,u)
 p=top(math.copysign(u,x),zz);return (x,p[1]+offset,z)
def frontz(x,y):
 u=abs(x)/(.962*sidew(2.241))
 return 2.222+displacement(2.241,u)-.012*((y-.39)/.22)**2+.026*exp(-((y-.454)/.068)**2)+.017*exp(-((y-.196)/.042)**2)-.016*exp(-((y-.31)/.070)**2)
def rearz(x,y):
 u=abs(x)/(.962*sidew(-2.31));return -2.31+displacement(-2.31,u)+.026*((y-.47)/.31)**2
def end(s,x,y,o=0):return (x,y,frontz(x,y)+o if s==1 else rearz(x,y)-o)
def side_marker(s):
 # Wrap the lamp just above the forward corner of the wheel arch.
 plate('Front side marker',roundrect(1.648,.588,.069,.010,3),lambda z,y:sxyz(s,z,y),black,seal,.003)
def endskin(s):
 z=2.241 if s==1 else -2.31
 def fun(u,v):
  a=2*u-1;x,h,edgez=top(a,z);hb=bottom(z)+(arch(edgez)-bottom(z))*abs(a)**6;y=hb+(h-hb)*v
  # A continuous shoulder rounds each outer corner into the side of the car.
  xs=side(1,z,v)[0];f=abs(a)**5
  xx=x+math.copysign(f*(xs-abs(top(1,z)[0])),a)
  p=end(s,x,y)
  return (xx,y,p[2]+(top(a,z)[2]-p[2])*((abs(a)**12)+v**12*(1-abs(a)**12)))
 nu,nv=152,66
 vs=[fun(i/nu,j/nv) for j in range(nv+1) for i in range(nu+1)];fs=[]
 for j in range(nv):
  for i in range(nu):
   x,y,_=fun((i+.5)/nu,(j+.5)/nv)
   def aperture(cx,cy,w,h,expn=4):
    return abs((x-cx)/w)**expn+abs((y-cy)/h)**expn<1
   if s==1 and (aperture(.635,.342,.162,.120) or aperture(-.635,.342,.162,.120) or aperture(0,.269,.36,.055,10)):continue
   if s==-1 and (aperture(.622,.323,.190,.104) or aperture(-.622,.323,.190,.104) or aperture(0,.271,.435,.093)):continue
   n=j*(nu+1)+i;fs.append((n,n+1,n+nu+2,n+nu+1))
 mesh('Front fascia' if s==1 else 'Rear bumper skin',vs,fs,blue)

def body():
 global parent
 parent='Bodywork'
 # An open cockpit is cut directly from the deck grid. Its trim follows the aperture.
 for name,a,b in [('Engine lid and rear shoulders',-2.31,-1.25),('Bonnet and front fenders',.65,2.241)]:
  patch(name,72,80,lambda u,v:top(2*u-1,a+(b-a)*v),blue)
 for s in (-1,1):
  patch('Upper door rail',22,113,lambda u,v,s=s:top(s*(.745+.255*u),-1.25+1.90*v),blue)
  patch('Rear-seat shelf',30,14,lambda u,v,s=s:top(s*.745*u,-1.25+.17*v),leather)
  patch('Outer quarter and door skin',199,32,lambda u,v,s=s:side(s,-2.31+4.551*u,v),blue)
  patch('Sill return and inner arches',180,5,lambda u,v,s=s:tuple(Vector(side(s,-2.31+4.551*u,0))+Vector((-s*(.15*v),-.009*v,0))),well)
  side_marker(s)
  tx,ty,tz=sxyz(s,.495,.254);text_side('t-hybrid',(tx+.003*s,ty,tz),.032,grey,s)
  door=[(.64,.845),(.67,.71),(.69,.49),(.69,.26),(.66,.214),(.48,.199),(-.34,.219),(-.43,.246),(-.49,.31),(-.55,.42),(-.61,.59),(-.62,.75),(-.57,.88)]
  path('Door perimeter', [sxyz(s,z,y) for z,y in door],.0034,seal)
  plate('Flush door pull',roundrect(-.407,.708,.103,.019,3.2),lambda z,y:sxyz(s,z,y),blue,seal,.0028)
  intake=[(-.758,.598),(-.725,.613),(-.722,.636),(-.786,.751),(-.828,.808),(-.867,.823),(-.947,.824),(-.954,.809),(-.884,.698),(-.806,.612)]
  plate('Turbo rear fender induction duct',intake,lambda z,y:sxyz(s,z,y),well,black,.010)
  path('Intake inner lip',[sxyz(s,z,y) for z,y in [(-.744,.616),(-.785,.637),(-.843,.718),(-.90,.803)]],.005,black)
  pts=[sxyz(s,z,.15+.015*abs(z)) for z in [-.79,-.57,-.20,.3,.72,.80]]
  path('Side skirt leading edge',pts,.021,black)
 endskin(1);endskin(-1)
 patch('Covered undertray',38,16,lambda u,v:((2*u-1)*(.68-.14*abs(2*v-1)**3),.135,-1.93+3.86*v),well)
 # Narrow hood gaps along the raised outer contour.
 for s in [-1,1]:
  path('Bonnet side seam',[topside(s*(.35+.224*v),2.117-1.325*v,.002) for v in [j/55 for j in range(56)]],.0028,seal)
 path('Bonnet nose joint',[topside(x,.018*(x/.38)**2+2.117,.002) for x in [j/100 for j in range(-36,37)]],.0024,seal)
 # Hood crest, divided into warm and dark enamel zones.
 cp=[(-.024,.614),(.024,.614),(.021,.598),(0,.579),(-.021,.598)]
 plate('Hood shield',cp,lambda x,y:topside(x,2.098+(.60-y)*1.05,.003),grey,brushed,.0018)
 # End openings are registered in the fascia, with a raised lip, dark return and set-back blades.
 def aperture(name,outline,sgn):
  # Deep, contoured liner behind the rim and modeled reveal faces.
  cen=[sum(p[k] for p in outline)/len(outline) for k in range(2)]
  loop=[end(sgn,x,y,.002) for x,y in outline]
  inset=[end(sgn,cen[0]+.96*(x-cen[0]),cen[1]+.94*(y-cen[1]),-.040) for x,y in outline]
  mesh(name+' reveal',loop+inset,[(i,(i+1)%len(loop),(i+1)%len(loop)+len(loop),i+len(loop)) for i in range(len(loop))],black)
  filled(name+' liner',inset,well)
  tube(name+' edge',[end(sgn,x,y,.004) for x,y in outline],.008,black,10,True)
 for s in (-1,1):
  outline=roundrect(s*.635,.342,.166,.123,4.2)
  aperture('Active front aero duct',outline,1)
  for k in range(5):
   x=s*(.504+.063*k);ylo=.222+.015*(k==4);yhi=.459-.022*(k==4)
   patch('Variable cooling blade',7,8,lambda u,v,x=x,ylo=ylo,yhi=yhi:end(1,x+(u-.5)*.036+.009*(v-.5),ylo+(yhi-ylo)*v,.014+.020*sin(pi*u)),black)
  # Lens ellipses follow the fender dome, with an inset dark pocket and optical modules.
  def lamp(u,v,s=s):
   # u,v in unit disk. The long direction follows the swept fender.
   x=s*(.695+.118*u);z=1.762+.160*v
   y=.710-.117*v-.005*u+.014*(1-u*u-v*v)
   return (x,y,z)
  ring=[lamp(cos(2*pi*i/72),sin(2*pi*i/72)) for i in range(72)]
  patch('Oval headlight optical carrier',72,16,lambda u,v:lamp(v*cos(2*pi*u),v*sin(2*pi*u)),lens)
  tube('Headlamp rubber gasket',ring,.009,seal,9,True);tube('Lens polished perimeter',ring,.004,brushed,8,True)
  for uu in [-.42,.42]:
   for vv in [-.35,.35]:
    outline=[tuple(Vector(lamp(uu+.24*cos(pi*i/20),vv+.18*sin(pi*i/20)))+Vector((0,.003,.002))) for i in range(40)]
    tube('Projector optical bezel',outline,.0022,brushed,8,True)
  for level in [-.65,.62]:
   path('Optical lens horizontal break',[tuple(Vector(lamp(a,level+.042*a*a))+Vector((0,.003,.001))) for a in [(-1+i/15)*sqrt(1-level*level)*.86 for i in range(31)]],.002,black)
  for ax in (-.40,.40):
   for az in (-.37,.32):
    ellipse('Matrix projector',tuple(Vector(lamp(ax,az))+Vector((0,.007,.002))),(.025,.008,.027),lens,20,10)
    path('Four point HD Matrix daytime lamp',[tuple(Vector(lamp(ax+(k-.5)*.62,az+.04*cos(pi*k)))+Vector((0,.012,.006))) for k in [0,.25,.5,.75,1]],.0037,reflector)
  plate('Rear valance',roundrect(s*.622,.323,.194,.107,4.5),lambda x,y:end(-1,x,y,.006),black,seal,.008)
  exhaust=roundrect(s*.600,.324,.111,.050,4.1)
  plate('Twin titanium exhaust tips',exhaust,lambda x,y:end(-1,x,y,.022),well,brushed,.0085)
  path('Exhaust tip bridge',[end(-1,s*.60,.278,.023),end(-1,s*.598,.324,.024),end(-1,s*.604,.37,.023)],.005,brushed)
  patch('Wraparound red rear reflector',27,3,lambda u,v,s=s:end(-1,s*(.61+.175*u),.462+.018*v,.026),red)
 outline=[(-.375,.209),(.375,.209),(.358,.318),(.343,.326),(-.343,.326),(-.358,.318)]
 aperture('Central radiator',outline,1)
 for i in range(10):
  y=.218+.0105*i;path('Radiator lamella',[end(1,x,y,-.005) for x in [-.355,-.2,0,.2,.355]],.0024,black)
 for i in range(-17,18):path('Radiator vertical fin',[end(1,i*.020,.213,-.012),end(1,i*.020,.322,-.012)],.0015,black)
 for face,z in [(1,2.241),(-1,-2.31)]:
  points=[]
  for k in range(-65,66):
   u=k/65;x,_,actualz=top(u,z)
   y=bottom(z)+(arch(actualz)-bottom(z))*abs(u)**6+.004
   points.append(end(face,x,y,.001))
  for sign in (-1,1):
   sidepts=[]
   for k in range(1,28):
    zz=z-face*.43*(k/27);x,y,zp=side(sign,zz,0)
    sidepts.append((x,y+.008,zp))
   if sign==-1:points=sidepts[::-1]+points
   else:points+=sidepts
  path('Continuous lower bumper lip',points,.012,black)
 for y,r,m in [( .755,.029,black),(.772,.008,red)]:
  p=[end(-1,x,y-.026*(abs(x)/.80)**3,.038 if m==red else .012) for x in [i*.01 for i in range(-80,81)]];path('Continuous rear light strip',p,r,m)
 plate('Rear registration recess',roundrect(0,.543,.261,.055,6),lambda x,y:end(-1,x,y,.0045),blue,seal,.004)
 plate('Lower diffuser',roundrect(0,.271,.438,.095,4),lambda x,y:end(-1,x,y,.006),well,black,.009)
 for x in [-.36,-.22,0,.22,.36]:
  path('Diffuser strake',[end(-1,x,.20,.032),end(-1,x,.30,.016),end(-1,x,.345,.005)],.009,black)
def text_side(label,position,size,m,s,vertical=False):
 from mathutils import Matrix
 cu=bpy.data.curves.new(label,'FONT');cu.body=label;cu.align_x='CENTER';cu.align_y='CENTER';cu.size=size;cu.extrude=.0003;cu.materials.append(m)
 ob=bpy.data.objects.new(label,cu);bpy.context.scene.collection.objects.link(ob);ob.parent=group(parent)
 # Blender text local +Z faces the viewer; use local +Y for the letter height.
 ax=Vector((0,-s,0));ay=Vector((0,0,1));az=ax.cross(ay)
 if az.x*s<0:ax=-ax;az=ax.cross(ay)
 if vertical:ax,ay=ay,-ax
 ob.matrix_world=Matrix.Translation(Vector(cv(position)))@Matrix((ax,ay,az)).transposed().to_4x4()
 bpy.context.view_layer.objects.active=ob;ob.select_set(True);bpy.ops.object.convert(target='MESH');ob.select_set(False)
def wheels():
 global parent
 for s in [-1,1]:
  for tag,z,x,width,tr,r in [('Front',1.225,.7915,.255,.34325,.254),('Rear',-1.225,.794,.325,.3642,.2667)]:
   parent=tag+('_Left_Wheel' if s==1 else '_Right_Wheel');h=tr
   def polar(rad,a,off):return (s*(x+off),h+rad*sin(a),z+rad*cos(a))
   def rev(n,profile,m):
    vs=[polar(rr,2*pi*j/112,xx) for rr,xx in profile for j in range(112)];fs=[]
    for i in range(len(profile)-1):
     for j in range(112):j2=(j+1)%112;fs.append((i*112+j,i*112+j2,(i+1)*112+j2,(i+1)*112+j))
    mesh(n,vs,fs,m)
   w=width/2
   ts=[(r*1.022,-w*.90),(tr-.055,-w),(tr-.028,-w*1.015),(tr-.010,-w*.95),(tr,-w*.78)]
   for a in [-.55,-.17,.2,.54]:ts += [(tr,w*(a-.035)),(tr-.004,w*(a-.018)),(tr-.004,w*(a+.014)),(tr,w*(a+.03))]
   ts += [(tr,w*.78),(tr-.009,w*.95),(tr-.03,w*1.015),(tr-.056,w),(r*1.058,w*.974),(r*1.022,-w*.90)]
   rev('Performance tire with four circumferential channels',ts,rubber)
   rev('Inner rim barrel',[(r*.91,-w*.78),(r*1.06,-w*.78),(r*1.06,w*.89),(r*1.012,w*.92),(r*.958,w*.85),(r*.91,-w*.78)],well)
   rev('Barrel interior',[(0,-w*.70),(r*.953,-w*.70),(r*.953,w*.82)],well)
   rev('Diamond cut wheel flange',[(r*.988,w*.979),(r*1.012,w*1.016),(r*1.073,w*1.014),(r*1.075,w*.988)],brushed)
   off=w*.81;dr=.210 if tag=='Front' else .205
   rev('Cross-drilled PCCB rotor',[(.065,off-.093),(dr,off-.093),(dr,off-.079),(.065,off-.079),(.065,off-.093)],carbon)
   dots=[];faces=[]
   for k in range(33):
    for j in range(3):
     a=2*pi*(k/33 + .007*j);rr=dr-.012-j*.022
     base=len(dots);dots += [polar(rr+.0023*sin(2*pi*i/8),a+.0023*cos(2*pi*i/8)/rr,off-.0785) for i in range(8)];faces.append(tuple(base+i for i in range(8)))
   mesh('Rotor cooling perforations',dots,faces,well,False)
   def cal(u,v):
    a=pi+(-.59+1.18*v) if tag=='Front' else -.55+1.10*v
    bulge=sin(pi*v)**.45
    rr=dr-.045-.024*bulge+(.042+.030*bulge)*u
    return polar(rr,a,off-.035+.012*bulge*sin(pi*u))
   patch('Yellow ceramic brake caliper',12,36,cal,yellow)
   for uu in [0,1]:
    patch('Caliper casting side',5,32,lambda a,b,uu=uu:(cal(uu,b)[0]-s*.02*a,*cal(uu,b)[1:]),yellow)
   for v in [.10,.32,.68,.90]:
    c=cal(.74,v);ellipse('Caliper bridge bolt',(c[0]+s*.002,c[1],c[2]),(.004,.006,.006),grey,12,6)
   # Caliper logo and phone-legible Turbo S powertrain script.
   c=cal(.5,.50)
   text_side('P O R S C H E',(c[0]+s*.004,c[1],c[2]),.009,black,s,True)
   for ia in range(10):
    a=2*pi*ia/10+.023
    for j in [-1,1]:
     def spoke(u,v):
      rr=.057+v*(r*1.013-.057)
      half=linterp([(0,.114),(.23,.085),(.60,.036),(.9,.024),(1,.049)],v)
      ang=a+j*(.021+.106*v**.70)+(2*u-1)*half
      return polar(rr,ang,w*(.57+.45*v**.70)+.002*sin(pi*u))
     patch('Forked forged spoke face',6,22,spoke,grey)
     for edge in (0,1):
      def edgept(u,v):
       pos=spoke(edge,v);return (pos[0]-s*.021*u,pos[1],pos[2])
      patch('Spoke side wall',2,22,edgept,grey)
      path('Diamond cut spoke edge',[spoke(edge,k/24) for k in range(25)],.00165,brushed)
     patch('Machined spoke land',2,22,lambda u,v:tuple(Vector(spoke(.30+.32*u,v))+Vector((s*.0015,0,0))),brushed)
   rev('Central locking boss',[(0,w*.62),(.059,w*.6),(.061,w*.53),(.060,w*.465),(.052,w*.46)],grey)
   rev('Center-lock nut',[(.027,w*.69),(.044,w*.695),(.046,w*.63),(.045,w*.61)],black)
   rev('Center-lock cap',[(0,w*.722),(.027,w*.728),(.028,w*.69)],grey)
   capcrest=[(-.009,.012),(.009,.012),(.008,0),(0,-.011),(-.008,0)]
   filled('Turbonite center cap shield',[(s*(x+w*.736),h+yy,z+xx) for xx,yy in capcrest],brushed)
   for k in range(15):
    a=2*pi*k/15
    path('Centerlock knurl',[polar(.038,a, w*.69),polar(.045,a-.02,w*.675)],.0019,brushed)

def wsurface(u,v):
 # Curved laminated windscreen. The transverse crown and fore/aft bow are separate.
 a=abs(u);wb=.713*(1-v)+.568*v
 base=(wb*u,.857+.035*(1-a**2),.625+.210*(1-a**2))
 head=(wb*u,1.204+.074*(1-a**2),.118+.084*(1-a**2))
 p=blend(base,head,v);return (p[0],p[1]+.012*sin(pi*v),p[2]+.025*sin(pi*v))
def roof(u,v):
 z=.118-1.738*v
 w=linterp([(-1.62,.62),(-1.44,.71),(-1.12,.72),(-.72,.656),(-.31,.596),(.118,.568)],z)
 center=linterp([(-1.62,.960),(-1.47,1.031),(-1.14,1.164),(-.84,1.252),(-.49,1.296),(-.20,1.304),(.118,1.278)],z)
 edge=linterp([(-1.62,.945),(-1.47,.951),(-1.12,.953),(-.91,.950),(-.78,1.034),(-.57,1.138),(-.38,1.187),(-.12,1.215),(.118,1.204)],z)
 a=abs(u);zp=z+.084*(1-a*a)*max(0,1-2*v)
 return (w*u,center-(center-edge)*a**2.85,zp)
def cabinexterior():
 global parent
 parent='Bodywork'
 patch('Laminated windshield',64,36,lambda u,v:wsurface(2*u-1,v),glass)
 for s in [-1,1]:
  path('Body-colour A-pillar',[wsurface(s,.012+.97*v/40) for v in range(41)],.021,blue)
 path('Upper windscreen rail',[wsurface(k/36,.988) for k in range(-36,37)],.015,black)
 path('Cowl and glass weatherseal',[wsurface(k/36,0) for k in range(-36,37)],.012,seal)
 for s in [-1,1]:
  path('Parked wiper blade',[wsurface(s*u,.043+.015*sin(pi*u)) for u in [.08,.22,.4,.6,.75,.92]],.005,black)
  shaft('Mirror cantilever',(s*.746,.857,.425),(s*.868,.921,.388),.028,black)
  ellipse('Gentian mirror housing',(s*.913,.928,.385),(.1035,.054,.106),blue,36,18)
  ellipse('Mirror glass',(s*.916,.930,.318),(.090,.044,.018),brushed,32,16)
 parent='Soft_Top'
 patch('Black fabric top and sail panels',72,80,lambda u,v:roof(2*u-1,v),cloth)
 # Flexible rear screen inset projects onto the actual cloth surface.
 def windowpt(u,v):
  # fitted from the high rear studio image, rounded corners and slightly narrower lower edge
  up=.72*(1-.11*v)*u;vp=.555+.30*v+.018*abs(u)**6*(1-2*v)
  x,y,z=roof(up,vp);return (x,y+.002,z-.001)
 patch('Heated rear window',46,26,lambda u,v:windowpt(2*u-1,v),glass)
 # Cut open the dark roof surface below the rear glass.
 ob=bpy.data.objects['Black fabric top and sail panels'];me=ob.data
 import bmesh
 bm=bmesh.new();bm.from_mesh(me);bm.faces.ensure_lookup_table();kill=[]
 for j in range(80):
  v=(j+.5)/80
  for i in range(72):
   u=2*(i+.5)/72-1;d=(v-.555)/.30
   if 0<d<1 and abs(u)<.72*(1-.11*d)-.012 and .555+.018*(abs(u)/.72)**6<v<.855-.018*(abs(u)/.66)**6:kill.append(bm.faces[j*72+i])
 bmesh.ops.delete(bm,geom=kill,context='FACES');bm.to_mesh(me);bm.free()
 rr=[windowpt(i/28,0) for i in range(-28,29)]+[windowpt(1,j/14) for j in range(1,15)]+[windowpt(i/28,1) for i in range(27,-29,-1)]+[windowpt(-1,j/14) for j in range(13,0,-1)]
 tube('Glass binding',rr,.0045,cloth,8,True)
 for s in [-1,1]:
  for side,U in [('door',0),('quarter',1)]:
   def window(u,v):
    # Along the outer roof silhouette. The front part terminates at the A-pillar.
    if side=='door':z=.625-1.175*u
    else:z=-.55-.366*u
    by=.857+.069*(.625-z)/1.541;bx=.72-.027*(.625-z)/1.541
    if z>=.118:
     f=(.625-z)/(.625-.118);q=wsurface(s,f)
    else:q=roof(s,(.118-z)/1.738)
    return blend((s*bx,by,z),q,v)
   patch('Frameless '+side+' glazing',38,18,window,glass)
   if side=='door':path('Side glass trailing edge',[window(1,k/10) for k in range(11)],.0044,seal)
  left=[roof(s,j/100) for j in range(101)];path('Convertible fabric binding',left,.006,cloth)
 parent='Bodywork'
 # Tonneau surface below folded roof; extends back from the twin rear seats.
 patch('Rear soft-top storage lid',40,35,lambda u,v:top((2*u-1)*.85,-1.75+.67*v),blue)
 for s in [-1,1]:
  path('Tonneau seam',[topside(s*(.64+.038*sin(pi*v)), -1.53+.51*v,.003) for v in [i/28 for i in range(29)]],.0033,seal)
 # Engine air inlet between fabric and adjustable rear spoiler.
 def grille(xf,t):
  z=-1.99+.331*t+.026*xf*xf;return topside(xf*.603,z,.005)
 patch('Engine deck grille pocket',44,14,lambda u,v:grille(2*u-1,v),well)
 for i in range(-21,22):
  path('Longitudinal deck slat',[tuple(Vector(grille(i/22,t))+Vector((0,.005,0))) for t in [.025,.3,.7,.97]],.005,black)
 parent='Rear_Spoiler'
 for s in [-1,1]:
  shaft('Adjustable wing support',(s*.47,.827,-2.00),(s*.47,.936,-2.035),.035,black)
 def wing(u,v):
  x=.729*(2*u-1);a=abs(2*u-1);return (x,.940+.020*sin(pi*v)+.010*a*a,-1.993-.166*v+.094*a*a)
 patch('Active rear spoiler upper surface',60,15,wing,blue)
 patch('Active rear spoiler underside',60,15,lambda u,v:(wing(u,v)[0],wing(u,v)[1]-.013-.025*sin(pi*v),wing(u,v)[2]),black)
 for edge in (0,1):
  patch('Spoiler rounded tip',10,18,lambda u,v,edge=edge:(wing(edge,v)[0]+(.009 if edge==1 else -.009)*sin(pi*u),wing(edge,v)[1]-(.013+.025*sin(pi*v))*u,wing(edge,v)[2]),blue)
 path('Rear spoiler leading radius',[wing(i/70,0) for i in range(71)],.009,blue)
 patch('Sliding rear spoiler base',35,12,lambda u,v:((2*u-1)*.527,.836+.073*sin(pi*v),-1.973-.141*v),blue)
 path('Rear spoiler trailing edge',[wing(i/70,1) for i in range(71)],.007,black)
def cabin():
 global parent
 parent='Cabin'
 patch('Cabin carpet',30,26,lambda u,v:((2*u-1)*.687,.199,-1.13+1.78*v),well)
 # The PDK projects ahead of the rear axle under the raised divider between the
 # two rear seats. The dark upholstered bridge flows into the front armrest.
 carpet=mat('Upholstered rear tunnel',(.023,.024,.026),.01,.79)
 knots=[(-1.27,.294,.681),(-1.19,.307,.632),(-1.13,.302,.625),(-1.04,.231,.589),(-.86,.197,.552),(-.72,.174,.526),(-.588,.147,.505),(-.52,.124,.475),(-.42,.112,.422),(-.36,.102,.408)]
 def rear_tunnel(u,v):
  t=2*u-1;z=-1.27+.91*v;w=linterp([(a,b) for a,b,c in knots],z);h=linterp([(a,c) for a,b,c in knots],z)
  a=t*pi/2
  return (w*math.copysign(abs(sin(a))**.61,sin(a)),.195+(h-.195)*max(0,cos(a))**.35,z)
 patch('Rear seat transmission tunnel',48,45,rear_tunnel,carpet)
 for v in (0,1):
  patch('Transmission tunnel end',36,6,lambda u,q,v=v:(rear_tunnel(u,v)[0],.195+q*(rear_tunnel(u,v)[1]-.195),rear_tunnel(u,v)[2]),carpet)
 for side in [.29,.71]:
  path('Transmission tunnel piping',[tuple(Vector(rear_tunnel(side,j/42))+Vector((0,.001,0))) for j in range(2,40)],.0015,seal)

 for s in [-1,1]:
  patch('Interior door card',26,20,lambda u,v:(s*(.705-.018*cos(pi*v)),.206+.646*v,-1.09+1.77*u),leather)
  path('Door pull rail',[(s*.672,.65,-.43),(s*.65,.647,-.28),(s*.645,.671,.06),(s*.67,.713,.21)],.016,black)
  c=s*.371
  ellipse('Front seat cushion',(c,.340,-.249),(.234,.098,.268),leather)
  for j in [-1,1]:
   ellipse('Seat cushion bolster',(c+j*.182,.370,-.26),(.057,.074,.25),leather)
  def seat(u,v):
   h=.36+.748*v;w=linterp([(0,.20),(.32,.224),(.67,.211),(.83,.13),(1,.100)],v)
   return (c+w*(u*2-1),h,-.421-.185*v+.077*abs(2*u-1)**2+.019*cos(3*pi*v))
  patch('Adaptive sport seat face',28,42,seat,leather)
  patch('Sport seat rear shell',28,36,lambda u,v:(seat(u,v)[0],seat(u,v)[1],seat(u,v)[2]-.06-.019*sin(pi*v)),leather)
  patch('Padded headrest crown',28,10,lambda u,v:(seat(u,1)[0],seat(u,1)[1]+.022*sin(pi*v)*(1-(2*u-1)**2),seat(u,1)[2]-.06*v),leather)
  for s2 in [-1,1]:
   patch('Stitched seat bolsters',6,30,lambda u,v,s2=s2:tuple(Vector(seat(1 if s2==1 else 0,v))+Vector((-s2*.035*sin(pi*u),0,-.063*u))),leather)
  for t in [.21,.36,.52]:
   path('Seat insert seam',[tuple(Vector(seat(i/18,t))+Vector((0,.001,.002))) for i in range(3,16)],.0018,seal)
  def cushion(u,v):
   a=2*u-1;return (c+a*.218,.422+.039*a*a+.026*sin(pi*v),-1.09+.340*v)
  patch('Rear seating cushion',25,19,cushion,leather)
  def back(u,v):
   a=2*u-1;return (c+a*(.218-.025*v),.43+.418*v,-1.067-.107*v+.061*a*a)
  patch('Rear backrest upholstery',25,22,back,leather)
  for v in [.23,.47,.71]:
   path('Rear insert seam',[tuple(Vector(back(i/22,v))+Vector((0,0,.002))) for i in range(2,21)],.002,seal)
  patch('Rear lap and shoulder belt',4,25,lambda u,v:(s*(.458+.12*v)+.042*u,.475+.364*v,-1.06-.102*v+.024),seal)
  # Forward and rear bulkheads close the cabin footwells below the belt line.
  patch('Rear wheelhouse trim',22,25,lambda u,v:(s*(.587+.113*v),.205+.692*v,-1.23+.55*u),leather)
  patch('Rear firewall trim',22,27,lambda u,v:(s*.704*u,.20+.723*v,-1.19-.030*v+.038*u*u),leather)
  patch('Front footwell toe board',17,18,lambda u,v:(s*.703*u,.19+.589*v,.568-.063*v),well)
  path('Seatbelt buckle',[(c-s*.205,.327,-.382),(c-s*.205,.410,-.348)],.017,black)
  path('Red buckle release',[(c-s*.210,.412,-.345),(c-s*.199,.412,-.345)],.009,red)
  for vx in [-.60,.60]:
   path('Sport seat inner shoulder seam',[tuple(Vector(seat((1+vx*(.77+.22*sin(pi*v)))/2,v))+Vector((0,.001,.003))) for v in [i/26 for i in range(4,24)]],.0022,seal)
  # A leather center insert covers the sport seat's sculpted cushion.
  patch('Seat cushion insert',24,22,lambda u,v:(c+(2*u-1)*.14,.385+.017*(2*u-1)**2+.019*sin(pi*v),-.438+.383*v),leather)
 def dash(u,v):
  x=(2*u-1)*.711;y=.707+.140*sin(v*pi*.72)-.030*(abs(x)/.72)**2
  return (x,y,.347+v*.34+.035*(x/.72)**2)
 patch('Sculpted horizontal dashboard',70,33,dash,leather)
 patch('Dashboard fascia',60,15,lambda u,v:((2*u-1)*.715,.565+.143*v,.357+.033*(2*u-1)**2),black)
 ellipse('Instrument binnacle',(.361,.735,.372),(.222,.107,.098),black)
 def display(name,cx,h,w,height):
  plate(name,roundrect(cx,h,w,height,5),lambda x,y:(x,y,.302-.012*((x-cx)/w)**2),screen,black,.006)
 display('Digital driver display',.36,.740,.187,.066)
 display('PCM touchscreen',-.042,.679,.13,.053)
 for i in range(5):
  c=.356+(i-2)*.060
  path('Instrument scale',[(c+.030*cos(t*pi/25),.732+.030*sin(t*pi/25),.292) for t in range(-1,28)],.0017,lit)
  path('Gauge needle',[(c,.732,.29),(c+.010,.75,.29)],.0014,red)
 for i in range(3):
  for j in range(3):ellipse('PCM shortcut',(-.12+.078*i,.668+.020*j,.297),(.006,.004,.002),lit,9,6)
 path('Dashboard lower accent',[(x,.60,.342) for x in [-.68,-.46,-.2,0,.24,.51,.68]],.0055,grey)
 for i in range(13):
  x=-.575+i*.035
  shaft('Passenger vent vanes',(x,.713,.347),(x,.765,.363),.003,black)
 ellipse('Sport Chrono clock',(0,.875,.550),(.048,.047,.053),black)
 ellipse('Clock dial',(0,.876,.502),(.036,.034,.005),black)
 for i in range(12):
  a=2*pi*i/12
  path('Chronometer face index',[(r*cos(a),.876+r*sin(a),.497) for r in [.025,.031]],.0013,lit)
 path('Chronometer hand',[(0,.876,.494),(.014,.890,.494)],.0015,lit)
 def console_h(z):return linterp([(-.46,.41),(-.29,.437),(-.13,.415),(.12,.49),(.32,.575)],z)
 def island(u,v):
  z=-.46+.78*v;w=.094+.035*sin(pi*v)**.5;return ((2*u-1)*w,console_h(z)-.010*abs(2*u-1)**3,z)
 patch('PDK console piano black inset',26,46,island,black)
 for side in [-1,1]:
  patch('Raised console side rail',12,46,lambda u,v,side=side:tuple(Vector(island(1 if side==1 else 0,v))+Vector((side*.021*u,-.045*u*u,0))),leather)
  path('Console brushed trim',[island(1 if side==1 else 0,i/40) for i in range(41)],.004,grey)
 def control(n,xc,zc,wx,wz,m,e=0):
  plate(n,roundrect(xc,zc,wx,wz,4),lambda x,z:(x,console_h(z)+.005+e,z),m)
 control('Gear selector module',0,.116,.049,.078,grey,.004)
 control('Selector gate',0,.123,.028,.058,black,.011)
 ellipse('PDK toggle',(0,console_h(.135)+.043,.135),(.023,.036,.028),black)
 path('Brushed selector toggle cap',[(-.019,console_h(.135)+.064,.140),(0,console_h(.135)+.074,.138),(.019,console_h(.135)+.064,.140)],.005,grey)
 for x in [-.084,.084]:
  for z in [-.04,.033,.104,.175,.246]:
   control('Console switch',x,z,.012,.020,leather,.003)
   path('Switch index',[(x-.005,console_h(z)+.012,z),(x+.005,console_h(z)+.012,z)],.0012,lit)
 control('Rear console storage well',0,-.18,.072,.066,well)
 ellipse('Padded console armrest',(0,.440,-.359),(.112,.048,.117),leather)
 shaft('Steering column',(.36,.694,.365),(.36,.674,.193),.047,black)
 def sw(a,r):return (.361+r*cos(a),.683+r*sin(a),.178+.044*sin(a))
 tube('GT Sport steering wheel',[sw(2*pi*i/96,.174) for i in range(96)],.0165,leather,12,True)
 for a in [.03,pi-.03,1.5*pi]:
  path('Steering wheel spoke',[sw(a,.047),sw(a,.105),sw(a,.171)],.019,grey)
 ellipse('Driver airbag',(.361,.683,.177),(.070,.057,.033),leather)
 ellipse('Wheel center crest',(.361,.687,.145),(.013,.016,.003),grey,16,8)
 # Relocated windshield flange bounds the open cabin when the fabric assembly is hidden.
 for s in [-1,1]:path('Inner belt molding',[sxyz(s*.81,z,.846+.08*(-z+.65)/1.65) for z in [-1.1,-.8,-.4,.0,.33,.59]],.008,seal)
def fit_common_cockpit():
 # The neighboring studio contains this site's authored 992.2 steering and dash.
 # Its GT3-only shifter, cluster artwork, upholstery, shell and aero are excluded.
 from mathutils import Matrix
 donor=R.parent/'porsche-911-gt3/porsche_911_gt3_992_2.glb'
 keep=set(bpy.data.objects);bpy.ops.import_scene.gltf(filepath=str(donor))
 imported=set(bpy.data.objects)-keep
 M=Matrix(((0,.018,0,.014),(-.018,0,0,-.062),(0,0,.018,-.025),(0,0,0,1)))
 for src in imported:
  name=src.name
  if src.type!='MESH' or not name.startswith(('GT3_SteeringWheel_','GT3_Steering_DriveModeDial_','GT3_Dash_','GT3_PCMScreen_','GT3_DigitalCluster_')):continue
  if 'Lugano' in name or 'Rearview' in name:continue
  me=src.data.copy();me.transform(M@src.matrix_world)
  if name.startswith('GT3_Dash_'):
   # Isolate real dashboard surfaces from the coupe's integrated overhead fittings.
   import bmesh
   bm=bmesh.new();bm.from_mesh(me);seen=set();drop=[]
   for vert in bm.verts:
    if vert in seen:continue
    stack=[vert];component=[];seen.add(vert)
    while stack:
     v=stack.pop();component.append(v)
     for edge in v.link_edges:
      vv=edge.other_vert(v)
      if vv not in seen:stack.append(vv);seen.add(vv)
    if max(v.co.z for v in component)>.989 or min(v.co.z for v in component)<.40 or max(v.co.y for v in component)>.36:drop.extend(component)
   bmesh.ops.delete(bm,geom=drop,context='VERTS');bm.to_mesh(me);bm.free()
   if not len(me.polygons):continue
  ob=bpy.data.objects.new(name.removeprefix('GT3_').replace('_',' '),me);bpy.context.scene.collection.objects.link(ob);ob.parent=group('Cabin')
  if 'Stitch' in name or 'MicroSuede' in name:
   for idx in range(len(me.materials)):me.materials[idx]=leather
  if 'DigitalCluster_Cabin_Instrument' in name:
   m=mat('Turbo S instrument display',(.03,.04,.06),.1,.21)
   nodes=m.node_tree.nodes;tex=nodes.new('ShaderNodeTexImage');tex.image=bpy.data.images.load(str(R/'source/instruments.png'));tex.image.pack();shader=nodes.get('Principled BSDF')
   m.node_tree.links.new(tex.outputs['Color'],shader.inputs['Base Color']);m.node_tree.links.new(tex.outputs['Color'],shader.inputs['Emission Color']);shader.inputs['Emission Strength'].default_value=.65
   uv=me.uv_layers.active or me.uv_layers.new();xs=[v.co.x for v in me.vertices];zs=[v.co.z for v in me.vertices]
   for l in me.loops:
    v=me.vertices[l.vertex_index].co;uv.data[l.index].uv=(1-(v.x-min(xs))/(max(xs)-min(xs)),(v.z-min(zs))/(max(zs)-min(zs)))
   for idx in range(len(me.materials)):me.materials[idx]=m
 for ob in imported:bpy.data.objects.remove(ob,do_unlink=True)
 # Replace the blockout fascia, wheel and controls with the common-generation parts.
 for ob in list(group('Cabin').children):
  if any(ob.name.startswith(x) for x in ['Sculpted horizontal','Dashboard fascia','Instrument binnacle','Digital driver display','PCM touchscreen','Instrument scale','Gauge needle','PCM shortcut','Dashboard lower','Passenger vent','GT Sport','Steering wheel spoke','Driver airbag','Wheel center crest','Steering column','removed blockout clock']):
   bpy.data.objects.remove(ob,do_unlink=True)
 # Fit the cabriolet's mirror to its own laminated screen header.
 shaft('Interior mirror stalk',(0,1.224,.196),(0,1.173,.144),.010,black)
 ellipse('Auto-dimming mirror frame',(0,1.154,.136),(.116,.039,.016),black)
 ellipse('Auto-dimming glass',(0,1.154,.124),(.109,.033,.006),brushed)
 bpy.ops.outliner.orphans_purge(do_recursive=True)
def lettering(text,location,size,material):
 cu=bpy.data.curves.new(text,'FONT');cu.body=text;cu.align_x='CENTER';cu.align_y='CENTER';cu.size=size;cu.extrude=.0007;cu.materials.append(material);ob=bpy.data.objects.new(text,cu);bpy.context.scene.collection.objects.link(ob)
 ob.location=cv(location);ob.rotation_euler=(pi/2,0,pi);ob.parent=group('Bodywork')
 bpy.context.view_layer.objects.active=ob;ob.select_set(True);bpy.ops.object.convert(target='MESH');ob.select_set(False)
exec(compile((R/'source/powertrain.py').read_text(),str(R/'source/powertrain.py'),'exec'))
body();wheels();cabinexterior();cabin();fit_common_cockpit();build_hybrid()
lettering('P O R S C H E',(0,.728,-2.322),.031,grey);lettering('911',(0,.541,-2.314),.073,brushed)
# Merge repeated elements of a named assembly by mesh name root. Preserve top-level
# groups and component names. Each tire, disc, seat, panel and glazing stays editable.
import re
sets=defaultdict(list)
for ob in list(bpy.context.scene.objects):
 if ob.type=='MESH':sets[(ob.parent.name,re.sub(r'\.\d+$','',ob.name))].append(ob)
for (_,base),obs in sets.items():
 if len(obs)>1:
  bpy.ops.object.select_all(action='DESELECT')
  for ob in obs:ob.select_set(True)
  bpy.context.view_layer.objects.active=obs[0];bpy.ops.object.join();obs[0].name=base
bpy.ops.object.select_all(action='DESELECT')
# Use the neighboring panel directions at section boundaries to carry reflections
# smoothly across the nose and rear corners. Retain all editable patch topology.
from mathutils.kdtree import KDTree
skin=[ob for ob in group('Bodywork').children if ob.name.startswith(('Front fascia','Rear bumper skin','Engine lid and rear shoulders','Bonnet and front fenders','Upper door rail','Outer quarter and door skin'))]
count=sum(len(o.data.vertices) for o in skin);kd=KDTree(count);vn=[]
for ob in skin:
 for v in ob.data.vertices:
  kd.insert(v.co,len(vn));vn.append(v.normal.copy())
kd.balance()
for ob in skin:
 me=ob.data;norm=[]
 for v in me.vertices:
  p=v.co;orig=v.normal.copy();rad=.038
  acc=orig*1.2
  for _,idx,dist in kd.find_range(p,rad):
   n=vn[idx];dot=n.dot(orig)
   if abs(dot)>.18:acc+=n*(1 if dot>0 else -1)*(1-dist/rad)**2
  norm.append(acc.normalized())
 me.normals_split_custom_set_from_vertices(norm)
# Anchor the rear overhang at 1,085 mm including lip and rear lenses.
meshes=[ob for ob in bpy.context.scene.objects if ob.type=='MESH']
bpy.context.view_layer.update()
rearmost=max((ob.matrix_world@v.co).y for ob in meshes for v in ob.data.vertices)
factor=1.085/(rearmost-1.225)
for ob in meshes:
 if ob.parent and ob.parent.name in {"Boxer_Engine","Left_eTurbo","Right_eTurbo","PDK_Hybrid","PDK_Motor_Envelope"}: continue
 mw=ob.matrix_world;inv=mw.inverted()
 for v in ob.data.vertices:
  co=mw@v.co
  if co.y>1.225:co.y=1.225+(co.y-1.225)*factor;v.co=inv@co
# Set tire pivots at the physical hubs, preserving children in world space.
for s in [-1,1]:
 for tag,z,x,y in [('Front',1.225,.7915,.34325),('Rear',-1.225,.794,.3642)]:
  groupob=groups[tag+('_Left_Wheel' if s==1 else '_Right_Wheel')]
  pivot=Vector(cv((s*x,y,z)));groupob.location=pivot
  for child in groupob.children:child.location-=pivot
sc=bpy.context.scene;sc.unit_settings.system='METRIC';sc.unit_settings.scale_length=1
sc.render.fps=24
bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(R/'source/turbo-s-cabriolet.blend'),compress=True)
bpy.ops.export_scene.gltf(filepath=str(R/'porsche.glb'),export_format='GLB',export_yup=True,export_animations=False,export_cameras=False,export_lights=False,
 export_draco_mesh_compression_enable=True,export_draco_mesh_compression_level=7,
 export_draco_position_quantization=16,export_draco_normal_quantization=12,export_draco_texcoord_quantization=14)
