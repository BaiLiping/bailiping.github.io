"""Contoured adaptive sport seats and Cabriolet rear bench in the metric cabin frame."""

def perforated_seat_material():
    material = mat('Perforated black leather inserts',(.051,.053,.058),.02,.63)
    tex = material.node_tree.nodes.new('ShaderNodeTexImage')
    tex.image = bpy.data.images.load(str(R/'source/leather-perforation.png'),check_existing=True)
    tex.image.pack()
    material.node_tree.links.new(tex.outputs['Color'],material.node_tree.nodes['Principled BSDF'].inputs['Base Color'])
    return material


def seat_uv(obj, back=False):
    uv=obj.data.uv_layers.new(name='Leather grain')
    for loop in obj.data.loops:
        x,y,z=obj.data.vertices[loop.vertex_index].co
        uv.data[loop.index].uv=(x/.042,(z if back else -y)/.042)
    return obj


def build_sport_seat(side, insert):
    global parent
    parent = 'Driver_Seat' if side == 1 else 'Passenger_Seat'
    cx=side*.367
    widths=[(0,.174),(.07,.196),(.24,.209),(.45,.222),(.57,.237),(.66,.225),
            (.745,.183),(.794,.139),(.897,.113),(.962,.099),(1,.077)]
    def width(v): return linterp(widths,v)
    def back(u,v):
        w=width(v); edge=abs(u)
        bulge=exp(-((edge-.81)/.24)**2)*(.059+.048*exp(-((v-.58)/.17)**2))*sin(pi*v)**.48
        z=-.427-.181*v+.021*sin(pi*v*1.4)+bulge-(.033+.014*v)*edge**8
        y=.375+.722*v+.027*(1-edge**3)*v**9
        return (cx+u*w,y,z)
    def shell(u,v):
        x,y,z=back(u,v)
        return (x,y,z-.031-(.042+.029*sin(pi*v))*(1-u*u)**.68)
    patch('Adaptive sport seat backrest',56,65,lambda u,v:back(2*u-1,v),leather)
    patch('Upholstered sport seat rear shell',42,52,lambda u,v:shell(2*u-1,v),leather)
    for hand in (-1,1):
        def padside(t,v):
            a,b=back(hand,v),shell(hand,v)
            x,y,z=blend(a,b,t)
            return (x+hand*.010*sin(pi*t),y,z)
        patch('Rolled backrest side bolster',12,52,padside,leather)
    for end in (0,1):
        def close(t,v):
            u=2*t-1
            x,y,z=blend(back(u,end),shell(u,end),v)
            return (x,y+(.013 if end else -.007)*sin(pi*v)*(1-u**4),z)
        patch('Headrest crown' if end else 'Backrest base',44,12,close,leather)
    def inset_width(v):
        return linterp([(0,.23),(.055,.49),(.18,.59),(.49,.56),(.76,.53),(.932,.43),(1,.20)],v)
    def inset(u,v):
        xx=(2*u-1)*inset_width(v)
        x,y,z=back(xx,.071+.656*v)
        return (x,y,z+.003+.0032*sin(pi*u)*sin(pi*v))
    seat_uv(patch('Perforated seat backrest insert',32,46,inset,insert),True)
    for hand in (-1,1):
        path('Inset welt seam',[inset(0 if hand<0 else 1,i/64) for i in range(65)],.00145,seal)
        pts=[]
        for i in range(66):
            v=.04+.91*i/65
            u=linterp([(.04,.835),(.42,.795),(.61,.915),(.69,.853),(.772,.698),(.825,.842),(.95,.90)],v)
            x,y,z=back(hand*u,v);pts.append((x,y,z+.0007))
        path('Shoulder bolster topstitch',pts,.00135,seal)
    for v in (0,1):
        path('Insert end binding',[inset(i/28,v) for i in range(29)],.0014,seal)
    for v in (.24,.415,.592,.762):
        pts=[inset(i/36,v-.009*cos(pi*(i/18-1))) for i in range(1,36)]
        path('Backrest transverse embossed seam',pts,.0013,seal)
        path('Backrest stitch return',[(x,y+.0035,z) for x,y,z in pts],.00065,grey)
    # Headrest emboss and stitch follow the face of the cushion.
    def headrest_face(u,v,d):
        t=.876+v/.76
        x,y,z=back(u/width(t),t)
        return (x,y,z+.0011+d)
    badge_relief('Headrest turbo S emboss','turbo-s',.102,.0152,headrest_face,
                 seal,.0007)

    # A single seat pan wraps around its cushion. Raised bolsters taper as they
    # approach the front of the thighs, exposing the perforated centre insert.
    def pad(u,v):
        w=.180+.032*sin(pi*v)**.55+.011*(1-v)
        xx=abs(u)
        yy=.340+.039*v+.017*sin(pi*v)+.071*exp(-((xx-.79)/.205)**2)*sin(pi*(.09+.84*v))**.6
        return (cx+u*w,yy,-.469+.498*v-.024*(abs(u)**4)*v**5)
    patch('Contoured sport seat cushion',48,46,lambda u,v:pad(2*u-1,v),leather)
    def center(u,v):
        x,y,z=pad((2*u-1)*(.53+.08*v),.07+.86*v);return (x,y+.002,z)
    seat_uv(patch('Perforated seat cushion insert',30,32,center,insert))
    for edge in (0,1):
        path('Seat cushion seam',[center(edge,i/45) for i in range(46)],.00145,seal)
    for pos in (.34,.54,.74):
        path('Seat cushion cross seam',[center(i/26,pos) for i in range(27)],.0012,seal)
    for hand in (-1,1):
        patch('Cushion side valance',12,40,lambda t,v:tuple(Vector(pad(hand,v))+
              Vector((hand*(.003*sin(pi*t)-.018*t*t),-.060*t,0))),leather)
        shaft('Seat mounting rail',(cx+hand*.144,.234,-.448),(cx+hand*.144,.234,.005),.015,black)
        for zz in (-.413,-.022):
            ellipse('Seat rail foot',(cx+hand*.144,.222,zz),(.031,.016,.031),dark_seat_trim,16,8)
    for edge in (0,1):
        patch('Seat cushion end cap',40,10,lambda u,t:tuple(Vector(pad(2*u-1,edge))+
              Vector((-(2*u-1)*.018*t*t,-.060*t,(1 if edge else -1)*.008*sin(pi*t)))),leather)
    for z,ry,rz in [(-.31,.018,.040),(-.24,.014,.022)]:
        ellipse('Seat position switch',(cx+side*.213,.312,z),(.010,ry,rz),dark_seat_trim,22,12)
    parent='Cabin'


def cabin_floor():
    def tub(u,v):
        a=2*u-1;z=-1.19+1.83*v
        width=linterp([(-1.19,.54),(-.97,.596),(-.60,.661),(-.37,.688),(.21,.681),(.64,.59)],z)
        rear=linterp([(-1.19,.506),(-.99,.502),(-.832,.457),(-.77,.373),(-.705,.241),(-.58,.209),(.35,.208),(.64,.254)],z)
        # The tunnel fully encloses the prop shaft down the centre of the cabin.
        tunnel=.238*max(0,1-(abs(a)*width/.148)**4)**1.45
        h=max(rear,tunnel+.202)+.077*abs(a)**8
        return (a*width,h,z)
    patch('Moulded cabin carpet and rear seat pan',60,90,tub,dark_seat_trim)
    for s in (-1,1):
        patch('Rear quarter inner trim',32,30,lambda u,v:(s*(.568+.112*v-.024*cos(pi*u)),.439+.452*v,
              -1.237+.55*u+.024*v),leather)
        patch('Front footwell toe board',32,30,lambda u,v:(s*(.586+.023*sin(pi*v))*u,
              .254+.492*v+.077*u**8,.64-.116*v+.024*u*u),dark_seat_trim)


def rear_bench(side, insert):
    c=side*.368
    def base(u,v):
        a=2*u-1;return (c+a*(.188+.031*v),.470+.035*a*a+.030*(1-v)+.023*sin(pi*v),-1.111+.366*v)
    def back(u,v):
        a=2*u-1
        return (c+a*(.198+.014*sin(pi*v)-.012*v),.491+.370*v-.016*a*a*v**4,
                -1.076-.106*v+.061*a*a+.013*sin(pi*v))
    seat_uv(patch('Rear seating cushion',32,30,base,insert))
    patch('Rear cushion upholstered apron',32,18,lambda u,t:tuple(Vector(base(u,1))+
          Vector((0,-.188*t,.011*sin(pi*t)))),leather)
    for edge in (0,1):
        patch('Rear cushion side trim',26,16,lambda v,t:tuple(Vector(base(edge,v))+Vector((0,-.16*t,0))),leather)
    patch('Rear backrest upholstery',35,32,back,leather)
    for level in (.24,.48,.71):
        path('Rear backrest seam',[tuple(Vector(back(i/26,level))+Vector((0,0,.0012))) for i in range(3,24)],.0015,seal)
    for u in (.23,.77):
        path('Rear cushion seam',[tuple(Vector(base(u,i/25))+Vector((0,.0015,0))) for i in range(2,24)],.0014,seal)
    patch('Rear shoulder belt',4,30,lambda u,v:(side*(.416+.18*v)+.039*u,.507+.374*v,-1.043-.093*v),seal)
