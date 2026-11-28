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
blue=mat('Gentian Blue Metallic 1A',(.030,.062,.235),.55,.255)
black=mat('Satin black intakes',(.013,.015,.018),.22,.36)
rubber=mat('Tire tread',(.021,.023,.025),.04,.75)
well=mat('Wheel housings and recesses',(.006,.007,.009),0,.86)
seal=mat('EPDM glazing and joint seals',(.012,.014,.017),.03,.65)
grey=mat('Turbonite wheels',(.210,.202,.176),.72,.30)
brushed=mat('Diamond cut faces',(.40,.41,.42),.70,.31)
carbon=mat('Ceramic disc',(.047,.054,.060),.34,.56)
yellow=mat('PCCB yellow',(.94,.54,.012),.28,.28)
glass=mat('Neutral laminated glazing',(.022,.034,.041),.15,.08,.66)
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
# Panel boundaries and body station measurements use the same coordinate system.
exec(compile((R/'source/exterior.py').read_text(),str(R/'source/exterior.py'),'exec'))

def text_side(label,position,size,m,s,vertical=False):
 from mathutils import Matrix
 cu=bpy.data.curves.new(label,'FONT');cu.body=label;cu.align_x='CENTER';cu.align_y='CENTER';cu.size=size;cu.extrude=.0003;cu.materials.append(m)
 ob=bpy.data.objects.new(label,cu);bpy.context.scene.collection.objects.link(ob);ob.parent=group(parent)
 # Blender text local +Z faces the viewer; use local +Y for the letter height.
 ax=Vector((0,-s,0));ay=Vector((0,0,1));az=ax.cross(ay)
 if az.x*s<0:ax=-ax;az=ax.cross(ay)
 if vertical:ax,ay=ay,-ax
 ob.matrix_world=Matrix.Translation(Vector(cv(position)))@Matrix((ax,ay,az)).transposed().to_4x4()
 if label=='t-hybrid':cu.shear=.32;cu.space_character=1.09
 bpy.context.view_layer.objects.active=ob;ob.select_set(True);bpy.ops.object.convert(target='MESH');ob.select_set(False)
 if label=='t-hybrid':
  matrix=ob.matrix_world.copy();inverse=matrix.inverted()
  for v in ob.data.vertices:
   co=matrix@v.co
   z=position[2]+(-co.y-position[2])*1.50
   co.y=-z;co.x=sxyz(s,z,co.z)[0]+s*.005
   v.co=inverse@co

exec(compile((R/'source/wheels.py').read_text(),str(R/'source/wheels.py'),'exec'))

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
 a=abs(u);zp=z+.084*(1-a*a)*max(0,1-2*v)+.164*a**5*smooth(.81,1,v)
 return (w*u,center-(center-edge)*a**2.85,zp)
def cabinexterior():
 global parent
 parent='Windscreen'
 patch('Laminated windshield',64,36,lambda u,v:wsurface(2*u-1,v),glass)
 for s in [-1,1]:
  def pillar(u,v):
   # A wide, tapered header frame rolls from the body-coloured outside to the padded inside.
   x,y,z=wsurface(s,.024+.963*v);q=2*pi*u
   tip=1-.81*smooth(.925,1,v)
   w=(.023+.021*(1-v)**2)*tip
   return (x+s*w*cos(q),y+.017*tip*sin(q),z+.020*tip*sin(q)+.006*tip*cos(q))
  patch('Painted A-pillar exterior',18,52,lambda u,v:pillar(u*.50,v),blue)
  patch('Upholstered A-pillar inner trim',18,52,lambda u,v:pillar(.50+.50*u,v),leather)
  path('Windscreen edge seal',[wsurface(s*.978,v/40) for v in range(41)],.0048,seal)
 def header(u,v):
  t=2*u-1;x,y,z=wsurface(t,.988-.022*abs(t)**6);a=2*pi*v
  cap=1-.38*abs(t)**8
  return (x,y+.009*cap*cos(a),z+.020*cap*sin(a))
 patch('Upper windscreen rail',72,18,header,black)
 patch('Windscreen upper ceramic border',72,6,lambda u,v:tuple(Vector(wsurface(2*u-1,.939+.056*v-.012*abs(2*u-1)**5))+Vector((0,.0006,.0008))),seal)
 patch('Windscreen lower frit band',72,6,lambda u,v:tuple(Vector(wsurface(2*u-1,.003+.026*v))+Vector((0,.0006,.0008))),seal)
 parent='Soft_Top'
 for s in [-1,1]:
  patch('Roof latch shoulder trim',18,12,lambda u,v:tuple(Vector(roof(s*(.92+.08*u),.048*v))+Vector((0,-.006,0))),cloth)
 parent='Windscreen'
 path('Cowl and glass weatherseal',[wsurface(k/36,0) for k in range(-36,37)],.012,seal)
 for s in [-1,1]:
  path('Parked wiper blade',[wsurface(s*u,.043+.015*sin(pi*u)) for u in [.08,.22,.4,.6,.75,.92]],.005,black)
 parent='Soft_Top'
 patch('Black fabric top and sail panels',112,118,lambda u,v:roof(2*u-1,v),cloth)
 # The rear bow carries a rubber welt around the tonneau seam.
 tube('Soft top rear welt',[roof(i/60,1) for i in range(-60,61)],.0055,seal,9)
 # Four rounded glass corners are shared by the glazing and the fabric cutout.
 def rear_glass(u,v):
  return roof(u,v)
 def in_glass(u,v):
  d=(v-.699)/.145
  return (abs(u)/(.704-.024*d))**4.8+abs(d)**4.8 < 1
 ob=bpy.data.objects['Black fabric top and sail panels'];me=ob.data
 import bmesh
 bm=bmesh.new();bm.from_mesh(me);bm.faces.ensure_lookup_table()
 kill=[bm.faces[j*112+i] for j in range(118) for i in range(112) if in_glass(2*(i+.5)/112-1,(j+.5)/118)]
 bmesh.ops.delete(bm,geom=kill,context='FACES');bm.to_mesh(me);bm.free()
 def rear_window_ring(angle,r):
  cu=math.copysign(abs(cos(angle))**(2/4.8),cos(angle));sv=math.copysign(abs(sin(angle))**(2/4.8),sin(angle))
  u=(.709-.024*sv*r)*cu*r;v=.699+.147*sv*r
  x,y,z=rear_glass(u,v);return (x,y+.0015,z-.001)
 patch('Heated rear window',96,22,lambda u,v:rear_window_ring(2*pi*u,v),glass)
 tube('Glass binding',[rear_window_ring(2*pi*i/120,1) for i in range(120)],.004,cloth,9,True)
 for side in [-1,1]:
  path('Convertible top seam',[tuple(Vector(roof(side*(.78+.051*sin(pi*v)),v))+Vector((0,.0014,0))) for v in [j/100 for j in range(97)]],.0013,cloth)
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
 # The flexible rear skirt fits between the bows and the stowage compartment.
 def roof_skirt(u,v,back=False,side=1):
  point=roof(2*u-1,1) if back else roof(side,.61+.39*u)
  x,y,z=point;base=topside(x,z,.0005)
  return (x,y+(base[1]-y)*v,z)
 patch('Canvas rear skirt',82,8,lambda u,v:roof_skirt(u,v,back=True),cloth)
 for side in [-1,1]:
  patch('Canvas side skirt',34,8,lambda u,v,side=side:roof_skirt(u,v,side=side),cloth)
 parent='Bodywork'
 # Tonneau surface below folded roof; extends back from the twin rear seats.
 patch('Rear soft-top storage lid',40,35,lambda u,v:top((2*u-1)*.85,-1.75+.67*v),blue)
 for s in [-1,1]:
  path('Tonneau seam',[topside(s*(.64+.038*sin(pi*v)), -1.53+.51*v,.003) for v in [i/28 for i in range(29)]],.0033,seal)
 # Upholstered folded-top compartment between the rear backrests and painted deck.
 # Its skin sits under the raised top and is revealed when the fabric is stowed.
 def tonneau(a,r):
  x=.692*math.copysign(abs(cos(a))**.43,cos(a))*r
  z=-1.308+.221*math.copysign(abs(sin(a))**.42,sin(a))*r+.032*(abs(x)/.692)**4
  return topside(x,z,.008+.007*(1-r*r))
 patch('Cabriolet stowage cover upholstery',96,26,lambda u,v:tonneau(2*pi*u,v),leather)
 tube('Stowage cover edge seal',[tonneau(2*pi*i/120,1) for i in range(120)],.004,seal,9,True)
 # The padded front wall of the stowage compartment closes the cabin above
 # the rear backrests. Its folded upper edge joins the shelf across both seats.
 def compartment_wall(u,v):
  a=2*u-1
  x=.720*a
  z=-1.159+.081*v+.021*a*a
  top_h=topside(x,-1.078+.021*a*a,.010)[1]
  y=.728+(top_h-.728)*v
  return (x,y,z+.006*sin(pi*v))
 patch('Padded top compartment bulkhead',82,22,compartment_wall,leather)
 path('Top compartment upholstery welt',[compartment_wall(i/80,.985) for i in range(81)],.004,seal)
 for side in [0,1]:
  patch('Top compartment side closure',16,16,lambda u,v,side=side:tuple(Vector(compartment_wall(side,v))+Vector((((-1 if side==0 else 1)*.028*u),0,-.23*u))),leather)
 # Engine air inlet between fabric and adjustable rear spoiler.
 def grille(xf,t,depth=0):
  z=-2.008+.353*t+.052*xf*xf
  return topside(xf*.615,z,depth)
 # Cut the intake in the engine cover, then roll its opening into a recessed tray.
 def grille_cut(x,back,y):
  z=-back-.052*(x/.615)**2
  return abs(x/.614)**9+abs((z+1.8315)/.176)**9-1
 for ob in list(group('Bodywork').children):
  if ob.name.startswith('Engine lid and rear shoulders'):trim_mesh(ob,grille_cut)
 uv=roundrect(0,.5,1,.5,9,108)
 lip=[grille(u,t,.001) for u,t in uv]
 inside=[grille(.972*u,.5+.91*(t-.5),-.030) for u,t in uv]
 mesh('Engine intake rolled return',lip+inside,[(i,(i+1)%len(lip),(i+1)%len(lip)+len(lip),i+len(lip)) for i in range(len(lip))],black)
 tube('Rear grille top surround',lip,.0038,black,9,True)
 patch('Engine deck grille pocket',60,24,lambda u,v:grille((2*u-1)*.997,v,-.037),well)
 slat_face=mat('Engine cover satin charcoal fins',(.055,.062,.068),.38,.40)
 for i in range(-13,14):
  center=i*.071
  def slat(u,v):
   start=.019+.105*abs(center)**7;end=.87-.075*abs(center)**6
   t=start+(end-start)*v
   return grille(center+.013*(2*u-1),t,.0035+.0036*sin(pi*u)-.012*(1-v)**5)
  patch('Engine lid cooling fin',6,22,slat,slat_face)
  for side in [0,1]:
   patch('Engine vent fin flank',3,22,lambda u,v,side=side:tuple(Vector(slat(side,v))+Vector((-.002*u,-.025*u,0))),black)
 parent='Rear_Spoiler'
 # The deployed blade has a crowned upper skin, a thinner trailing edge,
 # and a chord that collapses continuously to rounded outer tips.
 def wing(u,v):
  a=pi*(u-.5);t=sin(a);tip=max(0,cos(a))
  chord=.244*tip**.62
  mid=-2.091+.103*t*t
  base=.891+.038*(1-t*t)
  return (.698*t,base+.023*sin(pi*v)*tip+.005*v*tip,mid+chord*(.5-v))
 for underside in [False,True]:
  def surface(u,v):
   x,y,z=wing(u,v);tip=max(0,sin(pi*u))
   return x,y-(.005+.022*sin(pi*v)**.66)*tip*underside,z
  patch('Active rear spoiler underside' if underside else 'Active rear spoiler upper surface',82,27,surface,black if underside else blue)
 for v in [0,1]:
  path('Spoiler rolled edge',[wing(i/96,v) for i in range(97)],.0035,blue)
 # A broad painted telescoping cassette beneath the blade frames the dark slot.
 def cassette_ring(w,h,z):
  return [(x,y,z+.032*(x/.515)**2) for x,y in roundrect(0,.861,w,h,6.0,112)]
 rings=[cassette_ring(.518,.050,-2.167),cassette_ring(.514,.048,-2.183),
        cassette_ring(.489,.035,-2.190),cassette_ring(.477,.028,-2.150),
        cassette_ring(.469,.024,-2.106)]
 for k,m in enumerate([blue,blue,black,black]):
  n=len(rings[k])
  mesh('Spoiler cassette painted surround' if k<2 else 'Spoiler cassette recessed wall',rings[k]+rings[k+1],
       [(i,(i+1)%n,(i+1)%n+n,i+n) for i in range(n)],m)
 filled('Spoiler cassette deep aperture',rings[-1],well)
 tube('Spoiler cassette inner rolled edge',rings[2],.0038,blue,10,True)
 for side in [-1,1]:
  patch('Spoiler telescopic upright',30,14,lambda u,v:(side*(.445+.043*cos(2*pi*u)),.840+.071*v,-2.085+.027*sin(2*pi*u)),black)
 patch('Spoiler cassette lower plinth',58,16,lambda u,v:((2*u-1)*(.528+.030*v),.816+.008*sin(pi*u)+.014*(1-v),
       -2.136-.061*v+.033*(2*u-1)**2),blue)
exec(compile((R/'source/seats.py').read_text(),str(R/'source/seats.py'),'exec'))
dark_seat_trim=mat('Charcoal carpet and seat rail trim',(.022,.024,.026),.01,.81)
def cabin():
 global parent
 parent='Cabin'
 cabin_floor()
 seat_insert=perforated_seat_material()
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
  patch('Rear quarter interior trim',26,20,lambda u,v:(s*(.705-.018*cos(pi*v)),.206+.646*v,-1.09+.541*u),leather)
  c=s*.367
  build_sport_seat(s,seat_insert)
  rear_bench(s,seat_insert)
  patch('Rear firewall trim',36,24,lambda u,v:(s*(.562+.098*v)*u,.471+.403*v,-1.184-.04*v+.03*u*u),leather)
  path('Seatbelt buckle',[(c-s*.205,.327,-.382),(c-s*.205,.410,-.348)],.017,black)
  path('Red buckle release',[(c-s*.210,.412,-.345),(c-s*.199,.412,-.345)],.009,red)
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
 # The door cards and mirror housings also share the generation's architecture.
 from mathutils import Matrix
 donor=R.parent/'porsche-911-gt3/porsche_911_gt3_992_2.glb'
 keep=set(bpy.data.objects);bpy.ops.import_scene.gltf(filepath=str(donor))
 imported=set(bpy.data.objects)-keep
 M=Matrix(((0,.018,0,.014),(-.018,0,0,-.062),(0,0,.018,-.025),(0,0,0,1)))
 for src in imported:
  name=src.name
  if src.type=='MESH' and name.startswith('GT3_DoorCard_'):
   install_door_card(src)
   continue
  if src.type=='MESH' and name.startswith(('GT3_Mirror_L_','GT3_Mirror_R_')):
   fit_mirror(src)
   continue
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
body();wheels();cabinexterior();cabin();fit_common_cockpit();build_matrix_headlamps();cut_apertures();finish_outer_skin();fabric_surface();build_hybrid()
exec(compile((R/'source/chassis.py').read_text(),str(R/'source/chassis.py'),'exec'))
build_chassis()
rear_badging()
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
# Register the finished painted envelope (including the fascia trim) to the
# 4,551 x 1,900 mm body specification. Keep tyres, axle centres and cabin fixed.
meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
bpy.context.view_layer.update()
roots={'Bodywork','Rear_Bumper','Front_Bumper','Rear_Light_Bar','Rear_Spoiler','Soft_Top','Windscreen',
       'Bonnet','Left_Door','Right_Door','Left_Headlight','Right_Headlight'}
panels=[o for o in meshes if o.parent.name in roots and not o.name.startswith('Mirror ')]
coords=[o.matrix_world@v.co for o in panels for v in o.data.vertices]
rear=max(p.y for p in coords);nose=min(p.y for p in coords)
width=max(abs(p.x) for p in coords)
for ob in panels:
 mw=ob.matrix_world;inv=mw.inverted()
 for v in ob.data.vertices:
  co=mw@v.co
  # Blend correction beyond the tyre, retaining the circular arch openings.
  if co.y>1.64:co.y=1.64+(co.y-1.64)*(2.310-1.64)/(rear-1.64)
  if co.y<-1.64:co.y=-1.64+(co.y+1.64)*(2.241-1.64)/(-nose-1.64)
  co.x+=math.copysign((.950-width)*smooth(.71,width,abs(co.x)),co.x)
  v.co=inv@co
mirror_meshes=[o for o in meshes if o.parent.name in {'Left_Door','Right_Door'} and o.name.startswith('Mirror ')]
mirror_edge=max(abs((o.matrix_world@v.co).x) for o in mirror_meshes for v in o.data.vertices)
for ob in mirror_meshes:
 for v in ob.data.vertices:v.co.x+=math.copysign(1.0165-mirror_edge,v.co.x)
for ob in meshes:
 if ob.parent.name.endswith('_Wheel'):
  for v in ob.data.vertices:v.co.z=max(v.co.z,0)
# Set tire pivots at the physical hubs, preserving children in world space.
for s in [-1,1]:
 for tag,z,x,y in [('Front',1.225,.7915,.34325),('Rear',-1.225,.794,.3642)]:
  pivot=Vector(cv((s*x,y,z)))
  for assembly in ('Wheel','Brake'):
   groupob=groups[tag+('_Left_' if s==1 else '_Right_')+assembly]
   groupob.location=pivot
   for child in groupob.children:child.location-=pivot
sc=bpy.context.scene;sc.unit_settings.system='METRIC';sc.unit_settings.scale_length=1
sc.render.fps=24
bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(R/'source/turbo-s-cabriolet.blend'),compress=True)
bpy.ops.export_scene.gltf(filepath=str(R/'porsche.glb'),export_format='GLB',export_yup=True,export_animations=False,export_cameras=False,export_lights=False,
 export_draco_mesh_compression_enable=True,export_draco_mesh_compression_level=7,
 export_draco_position_quantization=16,export_draco_normal_quantization=12,export_draco_texcoord_quantization=14)
