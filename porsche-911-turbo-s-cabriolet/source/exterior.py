"""992.2 Turbo S Cabriolet outer panels, expressed in metres.
The seams bound independent surfaces so dismantling a panel leaves its opening.
"""
def smooth(a,b,t):
    v=max(0,min(1,(t-a)/(b-a)))
    return v*v*(3-2*v)

def sidew(z):
    return linterp([(-2.31,.880),(-1.77,.934),(-1.22,.952),(-.86,.931),
                    (-.40,.884),(.21,.869),(.70,.888),(1.18,.924),
                    (1.53,.922),(1.91,.893),(2.241,.850)],z)

def centerheight(z):
    return linterp([(-2.31,.797),(-2.06,.884),(-1.75,.950),(-1.32,.975),
                    (-1.04,.960),(.61,.877),(.83,.881),(1.15,.819),
                    (1.56,.751),(1.88,.650),(2.10,.550),(2.241,.485)],z)

def shoulder(z):
    return linterp([(-2.31,.821),(-2.08,.837),(-1.72,.910),(-1.27,.962),
                    (-.94,.959),(-.54,.892),(.18,.866),(.73,.856),(1.17,.861),
                    (1.48,.855),(1.68,.807),(1.91,.709),(2.241,.516)],z)

def displacement(z,u):
    k=1-sqrt(max(0,1-min(abs(u),1)**2.45))
    return k*(max(0,(-z-1.05)/1.26)**2*.381-max(0,(z-1.0)/1.241)**2*.412)

def top(t,z):
    u=abs(t)
    x=.966*sidew(z)*t
    zz=z+displacement(z,u)
    dome=exp(-((z-1.44)/.69)**4)
    # The bonnet falls between two raised lamp pods; the crowns roll down into the arches.
    h=centerheight(z)+(shoulder(z)-centerheight(z))*u**1.45
    h+=dome*(.043*exp(-((u-.771)/.191)**2)-.010*exp(-((u-.51)/.10)**2))
    h-=.012*exp(-((z-1.46)/.54)**4)*exp(-((u-.476)/.103)**2)
    # The front wing rolls toward the upright bumper from a gently inset crown.
    h-=.014*exp(-((z-1.55)/.63)**4)*u**6
    h+=.019*exp(-((z+1.31)/.73)**4)*exp(-((u-.71)/.21)**2)
    h-=(.049+.019*exp(-((z+.05)/.87)**4))*u**8
    # The rim of the reflector and the stamped fender share a tangent plane.
    lx=(abs(x)-.694)/.121
    lz=(zz-1.757+.010*lx)/.162
    rad=sqrt(lx*lx+lz*lz)
    plane=.706-.117*lz+.008*lx
    weight=1-smooth(.98,1.85,rad)
    h+=weight*(plane-.003-h)
    # The shallow pocket below the raised spoiler cassette forms a continuous painted deck.
    h-=.023*exp(-((z+2.14)/.078)**4)*exp(-(x/.55)**8)
    return (x,h,zz)

def bottom(z):
    return linterp([(-2.31,.215),(-1.91,.165),(-1.28,.129),(-.35,.120),
                    (.76,.124),(1.60,.133),(2.02,.158),(2.241,.182)],z)

def arch(z):
    for axle,r,cy in [(1.225,.368,.34325),(-1.225,.389,.3642)]:
        d=abs(z-axle)
        if d<r:return cy+sqrt(max(0,r*r-d*d))
        if d<r+.023:return cy+(bottom(z)-cy)*smooth(0,.023,d-r)
    return bottom(z)

def side(s,z,t):
    # z is the station before plan taper. Both the fascia and deck use this same edge.
    x,h,zz=top(s,z); lo=arch(zz); y=lo+(h-lo)*t
    # Interpolate at physical height, so a wheelarch cuts the same flank as the door.
    q=(y-bottom(zz))/max(.01,h-bottom(zz)); q=max(0,min(1,q))
    waist=linterp([(0,-.056),(.12,-.015),(.22,-.039),(.43,-.028),
                   (.62,-.005),(.75,.007),(.86,.018),(1,0)],q)
    door=smooth(-.93,-.56,zz)*(1-smooth(.57,.99,zz))
    crown=.013*sin(pi*q)-.014*(1-q)**3
    # Flatten the ridge immediately around the wheel opening into its own arch lip.
    radius=min(sqrt((zz-a)**2+(y-c)**2)-r for a,r,c in
               [(1.225,.368,.34325),(-1.225,.389,.3642)])
    lip=.009*(1-smooth(.017,.049,radius))*smooth(.18,.32,y)*(1-(q**8))
    inlet=.026*exp(-((zz+.814)/.14)**2-((y-.68)/.17)**2)
    return (s*(abs(x)+waist*door+crown*(1-door)+lip-inlet),y,zz)

def side_station(z):
    a,b=-2.31,2.241
    for i in range(16):
        m=(a+b)/2
        if m+displacement(m,1)<z:a=m
        else:b=m
    return (a+b)/2

def sxyz(s,z,y):
    st=side_station(z); h=top(s,st)[1]
    f=(y-arch(z))/max(.001,h-arch(z))
    x,_,_=side(s,st,max(0,min(1,f)))
    return (x+s*.0017,y,z)

def topside(x,z,offset=.002):
    st=z
    for i in range(9):
        u=max(-1,min(1,x/(.966*sidew(st))))
        st=z-displacement(st,u)
    return (x,top(u,st)[1]+offset,z)

def frontz(x,y):
    u=min(1,abs(x)/(.966*sidew(2.241)))
    return (2.231+displacement(2.241,u)-.012*((y-.39)/.24)**2
            +.021*exp(-((y-(.463+.034*u**2))/.061)**2)
            -.033*exp(-((y-.33)/.086)**2)+.022*exp(-((y-.197)/.038)**2))

def rearz(x,y):
    u=min(1,abs(x)/(.966*sidew(-2.31)))
    return (-2.304+displacement(-2.31,u)+.032*((y-.53)/.28)**2
            +.011*exp(-((y-.669)/.063)**2)-.009*exp(-((y-.605)/.066)**2)
            +.018*exp(-((y-.515)/.095)**2)*exp(-(x/.43)**8))

def end(s,x,y,o=0):return (x,y,frontz(x,y)+o if s==1 else rearz(x,y)-o)

def side_tag(s):return 'Left' if s==1 else 'Right'

def dark_return(name,p,vector,m,depth=1):
    inner=[tuple(Vector(v)+Vector(vector)*depth) for v in p];n=len(p)
    return mesh(name,p+inner,[(i,(i+1)%n,(i+1)%n+n,i+n) for i in range(n)],m)

def endskin(s):
    z=2.241 if s==1 else -2.31
    def fun(u,v):
        t=2*u-1; x,h,zedge=top(t,z);lo=bottom(z)+(arch(zedge)-bottom(z))*abs(t)**8
        y=lo+(h-lo)*v
        at_edge=abs(t)**7
        xx=x+t**9*(side(1,z,v)[0]-abs(top(1,z)[0]))
        zs=end(s,x,y)[2]
        return (xx,y,zs+(zedge-zs)*(at_edge+v**10*(1-at_edge)))
    patch('Front fascia' if s==1 else 'Rear bumper skin',128,54,fun,blue)

def face_duct(name,profile,face,depth,thick):
    # A raised perimeter and dark throat grow outward from the painted bumper skin.
    cen=[sum(p[k] for p in profile)/len(profile) for k in range(2)]
    rings=[]
    for scale,off in [(1.105,.001),(1.047,depth*.67),(1,depth),(.94,-.047)]:
        rings.append([end(face,cen[0]+scale*(x-cen[0]),cen[1]+scale*(y-cen[1]),off) for x,y in profile])
    n=len(profile)
    for j,m in enumerate([blue,blue,black]):
        mesh(name+' rolled surround',rings[j]+rings[j+1],[(i,(i+1)%n,(i+1)%n+n,i+n) for i in range(n)],m)
    # The back follows bumper curvature too; a planar n-gon would cut through the bumper.
    patch(name+' recessed grille',n,16,lambda u,v:end(face,cen[0]+(profile[min(n-1,round(u*n)%n)][0]-cen[0])*v,
        cen[1]+(profile[min(n-1,round(u*n)%n)][1]-cen[1])*v,-.045),well)
    tube(name+' outer gasket',rings[2],thick,black,8,True)

def panel_edges(n,points,s):
    dark_return(n,points,(-s*.018,0,0),seal)

def body():
    global parent
    # The front end: a distinct bonnet, left/right front fender skins, and a bumper cap.
    for s in [-1,1]:
        a,b=.65,2.241
        def hood_t(z):
            return linterp([(.65,.690),(.88,.644),(1.60,.553),(2.125,.452)],z)
        def hood_z(t):
            return 2.129-.010*t*t
        parent='Bonnet'
        patch('Pressed bonnet',52,65,lambda u,v:top(s*u*hood_t(a+(hood_z(u)-a)*v),
                                                   a+(hood_z(u)-a)*v),blue)
        rim=[top(s*hood_t(a+(hood_z(1)-a)*v/70),a+(hood_z(1)-a)*v/70) for v in range(71)]
        dark_return('Bonnet folded side flange',rim,(0,-.013,0),seal)
        path('Bonnet shut line',rim,.0018,seal)
        parent='Bodywork'
        # Split the fender into the portion next to the hood and the nose strip forward of it.
        def fender(u,v):
            z=a+(hood_z(1)-a)*v
            return top(s*(hood_t(z)+(1-hood_t(z))*u),z-.139*u**3*v**2)
        patch('Front fender crown',44,69,fender,blue)
        parent='Front_Bumper'
        def bumper_cap(u,v):
            t=u
            if t<=hood_t(2.12):z=hood_z(t/hood_t(2.12))
            else:z=hood_z(1)-.139*((t-hood_t(2.12))/(1-hood_t(2.12)))**3
            return top(s*t,z+(b-z)*v)
        patch('Bumper upper return',68,22,bumper_cap,blue)
        # The two cabin sills and the door cap are split at the actual top of the door.
        parent='Bodywork'
        patch('Engine lid and rear shoulders',56,64,lambda u,v:top(s*u,-2.31+1.06*v),blue)
        patch('Rear-seat shelf',22,14,lambda u,v:top(s*.745*u,-1.25+.17*v),leather)
        def rear_edge(t):
            return linterp([(0,-.340),(.12,-.425),(.33,-.517),(.65,-.600),(1,-.590)],t)
        def front_edge(t):
            return linterp([(0,.637),(.11,.680),(.46,.693),(.78,.674),(1,.623)],t)
        for tag,lo,hi in [('fixed',-1.25,rear_edge(1)),('door',rear_edge(1),front_edge(1)),('cowl',front_edge(1),.65)]:
            parent=side_tag(s)+'_Door' if tag=='door' else 'Bodywork'
            patch('Upper door rail' if tag=='door' else 'Upper quarter return',18,38,
                  lambda u,v:top(s*(.745+.255*u),lo+(hi-lo)*v),blue)
        # Interior sill tub sits inboard of the door aperture.
        parent='Bodywork'
        patch('Tub side member',12,60,lambda u,v:(s*(.691+.112*u),.16+.017*sin(pi*u),-.78+1.59*v),well)
        def for_side(z,t):
            return side(s,side_station(z),t)
        def sill_frac(z):
            h=top(s,side_station(z))[1]
            return max(0,min(.99,(.211+.008*sin((z+.67)*2)-arch(z))/max(.010,h-arch(z))))
        def upper_door(z,t):
            f=sill_frac(z)
            return for_side(z,f+(1-f)*t)
        def door(u,t):
            z=rear_edge(t)+(front_edge(t)-rear_edge(t))*u
            return upper_door(z,t)
        parent=side_tag(s)+'_Door'
        patch('Door outer panel',74,40,door,blue)
        outline=[door(i/50,0) for i in range(51)]+[door(1,i/40) for i in range(1,41)]+[door(i/50,1) for i in range(49,-1,-1)]+[door(0,i/40) for i in range(39,0,-1)]
        panel_edges('Door hem and seal',outline,s)
        seam=[door(1,i/30) for i in range(30,-1,-1)]+[door(i/50,0) for i in range(49,-1,-1)]+[door(0,i/30) for i in range(1,31)]
        path('Door aperture joint',seam,.00215,seal)
        plate('Flush door pull',roundrect(-.405,.709,.102,.0185,3.4),lambda z,y:sxyz(s,z,y),blue,seal,.0024)
        parent='Bodywork'
        patch('Painted rocker',70,12,lambda u,t:for_side(rear_edge(0)+(front_edge(0)-rear_edge(0))*u,
              sill_frac(rear_edge(0)+(front_edge(0)-rear_edge(0))*u)*t),blue)
        for end_index,is_front in [(0,False),(1,True)]:
            boundary=1.642 if is_front else -1.742
            def flank(u,t):
                edge=front_edge(t) if is_front else rear_edge(t)
                z=edge+(boundary-edge)*u
                # Blend lower edge from door height down over the first 12 cm.
                f=sill_frac(edge)*(1-smooth(0,.13,abs(z-edge)))
                return for_side(z,f+(1-f)*t)
            patch('Front arch and fender' if is_front else 'Rear haunch and wheel arch',96,35,flank,blue)
            # Triangular rocker returns beneath the curved shut line.
            def return_panel(u,t):
                edge=front_edge(0) if is_front else rear_edge(0)
                z=edge+(.13 if is_front else -.13)*u
                return for_side(z,sill_frac(edge)*(1-smooth(0,.13,abs(z-edge)))*t)
            patch('Rocker end stamping',20,10,return_panel,blue)
            parent='Front_Bumper' if is_front else 'Rear_Bumper'
            tip=top(s,b if is_front else -2.31)[2]
            patch('Bumper wing',48,30,lambda u,t:for_side(boundary+(tip-boundary)*u,t),blue)
            parent='Bodywork'
        for axle,r,cy in [(1.225,.368,.34325),(-1.225,.389,.3642)]:
            def flange(u,v):
                ang=pi*u;z=axle+r*cos(ang);y=cy+r*sin(ang)
                x=sxyz(s,z,y)[0]
                return (x-s*(.002+.082*v),y-.004*v,z)
            patch('Wheel arch folded flange',72,6,flange,black)
        for axle,opening,hub,wheel_in in [(1.225,.368,.34325,.655),(-1.225,.389,.3642,.615)]:
            # Deep moulded wheelhouse. The arch lip covers its outboard edge;
            # vertical mud shields close off daylight ahead of and behind each tyre.
            def arch_liner(u,v):
                a=pi*u
                radius=opening+.005+(.028*sin(pi*v))
                z=axle+radius*cos(a)
                y=hub+radius*sin(a)
                out=abs(sxyz(s,axle+opening*cos(a),hub+opening*sin(a))[0])-.003
                return (s*(out+(wheel_in-out)*v),y,z)
            patch('Moulded arch liner',100,15,arch_liner,well)
            patch('Inner wheelhouse splash shield',86,16,
                  lambda u,v:(s*wheel_in,.128+v*(hub+opening*sin(pi*u)-.128),axle+opening*cos(pi*u)),well)
            for edge in [0,1]:
                def apron(u,v):
                    xx,yy,zz=arch_liner(edge,u)
                    return (xx,.123+(yy-.123)*v,zz+(.017 if edge==0 else -.017)*(1-v))
                patch('Wheelhouse end apron',20,20,apron,well)
            for k in [.16,.26,.74,.84]:
                path('Wheelhouse moulding rib',[tuple(Vector(arch_liner(k,v/18))+Vector((0,-.003,0))) for v in range(2,17)],.0028,black)
        parent='Front_Bumper'
        plate('Front side marker',roundrect(1.586,.586,.095,.010,3),lambda z,y:sxyz(s,z,y),black,seal,.002)
        parent='Bodywork'
        # Tapered induction pocket and the offset, angled internal guide.
        def scoop(a):
            v=(1+sin(a))/2;u=math.copysign(abs(cos(a))**.67,cos(a))
            z=-.748-.122*v+u*(.035+.042*v)*max(0,sin(pi*v))**.22
            return (z,.588+.226*v)
        rim=[scoop(2*pi*i/72) for i in range(72)]
        p=[sxyz(s,z,y) for z,y in rim]
        inside=[(x-s*.049,.693+(y-.693)*.86,-.825+(z+.825)*.82) for x,y,z in p]
        filled('Quarter intake throat',inside,well)
        mesh('Quarter intake return',p+inside,[(i,(i+1)%len(p),(i+1)%len(p)+len(p),i+len(p)) for i in range(len(p))],black)
        tube('Quarter intake painted radius',p,.007,blue,10,True)
        path('Induction inner vane',[tuple(Vector(sxyz(s,-.760-.101*v,.601+.186*v))+Vector((-s*.014,0,0))) for v in [i/20 for i in range(21)]],.0075,black)
        tx,ty,tz=sxyz(s,.495,.249);text_side('t-hybrid',(tx+.003*s,ty,tz),.031,grey,s)
        patch('Black sill blade',70,8,lambda u,v:(s*(.795+.068*sin(v*pi*.74)+.009*sin(pi*u)),.126+.023*v-.009*sin(pi*u),-.805+1.59*u),black)
    parent='Bodywork'
    patch('Covered undertray',38,16,lambda u,v:((2*u-1)*(.68-.14*abs(2*v-1)**3),.135,-1.93+3.86*v),well)
    parent='Bonnet'
    p=[]
    for i in range(-75,76):
        u=i/75
        z=2.129-.010*u*u
        t=linterp([(.65,.690),(.88,.644),(1.60,.553),(2.125,.452)],z)*u
        x,y,z=top(t,z)
        p.append((x,y+.0006,z))
    path('Bonnet leading edge',p,.0018,seal)
    plate('Hood shield',[(-.022,-.017),(.022,-.017),(.020,.006),(0,.025),(-.020,.006)],lambda x,z:topside(x,2.097+z,.003),grey,brushed,.001)
    for face in [1,-1]:
        parent='Front_Bumper' if face==1 else 'Rear_Bumper'
        endskin(face)
        z=2.241 if face==1 else -2.31
        path('Painted lower bumper radius',[end(face,top(i/64,z)[0],bottom(z)+.004,.005) for i in range(-62,63)],.010,blue)
    front_details()
    rear_details()

def front_outline(s):
    return [(s*(.628+x+.010*y/.12),.339+y+.013*x/.165) for x,y in roundrect(0,0,.166,.115,4.3,84)]

def central_outline():
    return [(x*(1+.06*y/.060),.265+y-.005*(x/.364)**2) for x,y in roundrect(0,0,.364,.060,6,100)]

def front_details():
    global parent
    parent='Front_Bumper'
    for s in [-1,1]:
        face_duct('Active front cooling intake',front_outline(s),1,.019,.0035)
        for k in range(5):
            x=.488+k*.065
            yl=.230+.009*(k==4);yh=.442+.014*(k/4)-.015*(k in [0,4])
            def blade(u,v):
                xx=s*(x+.001*v+.007*sin(pi*v)+.059*u)
                return end(1,xx,yl+(yh-yl)*v,.003+.017*sin(pi*u)-.016*u)
            patch('Active cooling vane',9,18,blade,black)
            path('Cooling vane front corner',[blade(.21,i/18) for i in range(19)],.0023,black)
        for k in range(12):
            y=.237+.016*k
            path('Auxiliary intake core',[end(1,s*(.478+i*.032),y,-.016) for i in range(10)],.0018,carbon)
        path('Bumper eyebrow',[end(1,s*(.44+.336*t),.453+.044*t-.009*t*t,.009) for t in [j/40 for j in range(41)]],.0038,blue)
        for x,y in [(.336,.412),(.755,.438)]:
            tube('Parking sensor flush ring',[end(1,s*x+.0094*cos(2*pi*i/24),y+.0094*sin(2*pi*i/24),.001) for i in range(24)],.0011,blue,8,True)
    face_duct('Central radiator intake',central_outline(),1,.019,.003)
    for i in range(16):
        y=.212+.0069*i
        path('Radiator horizontal lamella',[end(1,x,y,-.014) for x in [-.342,-.18,0,.18,.342]],.0016,carbon)
    for i in range(-27,28):
        path('Radiator fin core',[end(1,i*.0124,.212,-.018),end(1,i*.0124,.320,-.018)],.00105,grey)
    for s in [-1,1]:
        patch('Radiator outer turning vane',7,8,lambda u,v:end(1,s*(.290+.027*u),.206+.107*v,.009+.002*sin(pi*u)),black)
    patch('Flexible front spoiler',120,12,lambda u,v:end(1,(2*u-1)*.813,.152+.036*v+.010*abs(2*u-1)**3,.004+.010*sin(pi*v)),black)

def rear_details():
    global parent
    parent='Rear_Bumper'
    # The lower cladding spans the full width. Its upper edge turns up around each corner.
    def valance(t,h):
        x=t*.832;y=.213+h*(.187+.012*t*t)-.014*abs(t)**3
        return end(-1,x,y,.013+.014*sin(pi*h))
    patch('Rear lower bumper cladding',130,20,lambda u,v:valance(2*u-1,v),black)
    path('Rear valance roll',[valance(i/64,1) for i in range(-64,65)],.005,black)
    for s in [-1,1]:
        profile=[(s*(.602+x+.013*y/.051),.312+y) for x,y in roundrect(0,0,.108,.052,3.7,80)]
        p=[end(-1,x,y,.047) for x,y in profile]
        # Taper each pair toward the dark tailpipe; the bridge is integral to the tip.
        mesh('Tailpipe internal wall',p+[end(-1,x*.995,y,.021) for x,y in profile],
             [(i,(i+1)%len(p),(i+1)%len(p)+len(p),i+len(p)) for i in range(len(p))],carbon)
        filled('Tailpipe shadow',[end(-1,x,y,.034) for x,y in profile],well)
        tube('Titanium sport exhaust lip',p,.0052,brushed,10,True)
        path('Exhaust tip bridge',[end(-1,s*.602,.263,.047),end(-1,s*.597,.279,.047),end(-1,s*.596,.349,.047),end(-1,s*.605,.362,.047)],.0048,brushed)
        patch('Rear marker housing',50,8,lambda u,v:end(-1,s*(.498+.308*u),.428+.009*u+.029*v-.009*abs(2*u-1)**6,.016),black)
        patch('Rear bumper reflector',50,8,lambda u,v:end(-1,s*(.514+.286*u),.431+.009*u+(.020-.008*abs(2*u-1)**6)*v,.020),red)
        for k in range(3):
            path('Outer valance flute',[end(-1,s*(.709+.018*k+.014*v),.263+.080*v,.030) for v in [j/12 for j in range(13)]],.003,black)
        def wrap(u,v):
            # The black valance rounds the corner beneath the sloping red reflector.
            start=valance(s*.98,v)
            edge_z=top(s,-2.31)[2]
            if u<.41:
                t=u/.41
                y=start[1]+.009*t*v
                edge=sxyz(s,edge_z,y)
                x=start[0]+(edge[0]-start[0])*t
                z=rearz(x,y)-.017
                z+=smooth(.45,1,t)*(edge_z-z)
                return (x+s*.008*sin(pi*t/2),y,z)
            t=(u-.41)/.59
            y=.213+v*(.199+.012*t)-.014+.009*t
            z=edge_z+(-1.814-.058*(1-v)-edge_z)*t
            x=sxyz(s,z,y)[0]
            return (x+s*(.008+.001*sin(pi*t)),y,z)
        patch('Outer rear valance return',36,15,wrap,black)
        path('Valance side molding',[wrap(u/36,1) for u in range(37)],.0048,black)
        def marker(u,v):
            return tuple(Vector(wrap(u,1))+Vector((s*.0015,.024+.013*v+.004*u,0)))
        patch('Side return reflector',28,6,marker,red)
        path('Rear air outlet upper crease',[tuple(Vector(sxyz(s,-1.81-.20*t,.426+.106*t))+Vector((s*.002,0,0))) for t in [i/18 for i in range(19)]],.0043,blue)
    # Pair of trapezoidal diffuser tunnels divided by one thick central keel.
    def tunnel(s,u,v):
        x=s*(.012+.413*u-.043*v*u+.014*v)
        y=.214+.145*v-.013*(2*u-1)**4
        return end(-1,x,y,.040-.005*u)
    for s in [-1,1]:
        p=[tunnel(s,i/30,0) for i in range(31)]+[tunnel(s,1,i/10) for i in range(1,11)]+[tunnel(s,i/30,1) for i in range(29,-1,-1)]+[tunnel(s,0,i/10) for i in range(9,0,-1)]
        tube('Diffuser channel rolled edge',p,.009,black,10,True)
        patch('Diffuser recessed ramp',30,13,lambda u,v:tuple(Vector(tunnel(s,u,v))+Vector((0,0,.002+.010*sin(pi*v)*sin(pi*u)))),well)
        for k in range(3):
            path('Diffuser guide rib',[tunnel(s,.23+.23*k,v) for v in [0,.06,.12,.20]],.0032,black)
    plate('Registration plate plinth',roundrect(0,.535,.259,.058,8),lambda x,y:end(-1,x,y,.008),blue,seal,.0023)
    parent='Rear_Light_Bar'
    # Project the rounded horizontal cartridge into the rear face. Porsche's
    # main red guide stops short of the outer ends of the smoke-coloured optic.
    def carrier(x,y,depth):
        return end(-1,x,y,depth)
    profile=roundrect(0,.722,.810,.043,11,128)
    border=[carrier(x,y,.003) for x,y in profile]
    inside=[carrier(.986*x,.722+(y-.722)*.91,-.013) for x,y in profile]
    n=len(profile)
    mesh('Recessed rear lamp pocket',border+inside,[(i,(i+1)%n,(i+1)%n+n,i+n) for i in range(n)],black)
    tube('Rear lamp aperture gasket',border,.0038,black,9,True)
    def rear_field(x,back,y):
        return abs(x/.809)**11+abs((y-.722)/.0425)**11-1
    for ob in list(group('Rear_Bumper').children):
        if ob.name.startswith('Rear bumper skin'):trim_mesh(ob,rear_field)
    def oval_surface(name,cx,cy,w,h,p,material,depth):
        outline=roundrect(cx,cy,w,h,p,144)
        patch(name,144,12,lambda u,v:carrier(cx+(outline[round(u*144)%144][0]-cx)*v,
              cy+(outline[round(u*144)%144][1]-cy)*v,depth),material)
    oval_surface('Smoke rear lamp housing',0,.722,.804,.041,11,black,-.010)
    oval_surface('Full width rear ruby carrier',0,.746,.753,.011,11,lens,.000)
    rear_red=mat('Continuous ruby light pipe',(.46,.006,.012),.20,.23)
    oval_surface('Continuous rear red light guide',0,.7475,.738,.0063,12,rear_red,.005)
    for side in [-1,1]:
        oval_surface('Lower rear lamp clear prism',side*.522,.698,.255,.008,8,lens,.0005)
        oval_surface('Rear lamp silver inner optic',side*.527,.700,.191,.0021,9,grey,.001)
    # The badge occupies the dark center of the lamp, below the red guide.
def rear_badging():
    for txt,y,size,offset in [('P O R S C H E',.706,.047,.008),('turbo S',.652,.031,.014),('911',.535,.071,.020)]:
        prev=set(bpy.data.objects)
        lettering(txt,(0,y,rearz(0,y)-offset),size,brushed)
        ob=next(o for o in set(bpy.data.objects)-prev if o.type=='MESH')
        ob.parent=group('Rear_Light_Bar' if txt=='P O R S C H E' else 'Rear_Bumper')

def fit_mirror(src):
    # Register the common 992.2 mirror from the neighbouring editable model.
    name=src.name
    if name.startswith('GT3_Mirror'):
        s=1 if '_L_' in name else -1
        me=src.data.copy();me.transform(src.matrix_world)
        for vert in me.vertices:
            z,x,y=vert.co
            vert.co=cv((s*(.753+(abs(x)-43.67)*.0205),.846+(y-43.78)*.0129,.313+(z-10.85)*.0146))
        ob=bpy.data.objects.new('Mirror '+name.split('_',3)[3],me);bpy.context.scene.collection.objects.link(ob);ob.parent=group(side_tag(s)+'_Door')
        if 'Lugano' in name:
            for i in range(len(me.materials)):me.materials[i]=blue
        return
def trim_mesh(ob,field):
    """Clip a panel against an opening in its own plane and retain shared edges."""
    old=ob.data
    verts=[tuple(v.co) for v in old.vertices]
    vals=[field(*p) for p in verts]
    extra={}
    def crossing(i,j):
        key=(min(i,j),max(i,j))
        if key not in extra:
            a,b=verts[i],verts[j]; va=vals[i]
            lo,hi=0,1
            for _ in range(12):
                t=(lo+hi)/2
                p=blend(a,b,t)
                if (field(*p)>=0)==(va>=0):lo=t
                else:hi=t
            extra[key]=len(verts);verts.append(blend(a,b,(lo+hi)/2))
        return extra[key]
    fs=[]
    for f in old.polygons:
        face=list(f.vertices)
        inside=[vals[i]>=0 for i in face]
        if all(inside):fs.append(face);continue
        if not any(inside):continue
        polygon=[]
        for i,j in zip(face,[*face[1:],face[0]]):
            if vals[i]>=0:polygon.append(i)
            if (vals[i]>=0)!=(vals[j]>=0):polygon.append(crossing(i,j))
        if len(polygon)>2:fs.append(polygon)
    new=bpy.data.meshes.new(old.name+' stamped')
    new.from_pydata(verts,[],fs);new.update();new.materials.append(old.materials[0])
    for f in new.polygons:f.use_smooth=True
    ob.data=new;bpy.data.meshes.remove(old)

def cut_apertures():
    global parent
    for ob in list(group('Front_Bumper').children):
        if ob.name.startswith('Front fascia'):
            def front_field(x,back,y):
                # The inner molding covers the trimmed edge and conceals the panel thickness.
                cy=(y-.339-.013*(abs(x)-.628)/.165)
                dx=(abs(x)-.628-.010*cy/.12)/.165
                outer=abs(dx)**4.3+abs(cy/.115)**4.3-1
                h=(y-.265+.005*(x/.364)**2)
                inner=abs(x/(.364*(1+.06*h/.060)))**6+abs(h/.06)**6-1
                return min(inner,outer)
            trim_mesh(ob,front_field)
    def lamp_field(x,back,y):
        lx=(abs(x)-.694)/.121
        lz=(-back-1.757+.008*lx)/.162
        return lx*lx+lz*lz-.996**2
    for ob in list(group('Bodywork').children):
        if ob.name.startswith('Front fender crown'):trim_mesh(ob,lamp_field)
        elif ob.name.startswith('Rear haunch and wheel arch'):
            def intake_field(x,back,y):
                v=(y-.588)/.226
                if v<=0 or v>=1:return 1
                half=(.035+.042*v)*(max(0,1-(2*v-1)**2)**.335)*max(0,sin(pi*v))**.22
                return abs(-back+.748+.122*v)-.97*half
            trim_mesh(ob,intake_field)

def build_matrix_headlamps():
    """Mount two tiers of projectors behind the swept elliptical HD-Matrix covers."""
    global parent
    chrome=mat('Matrix optic metallised reflector',(.094,.100,.105),.77,.27)
    bezel=mat('Black chrome Matrix shrouds',(.031,.037,.039),.64,.24)
    crystal=mat('Projector convex optical glass',(.022,.027,.032),.28,.23)
    clear=mat('Matrix outer polycarbonate cover',(.14,.18,.20),.18,.08,.20)
    ivory=mat('Matrix guide opal face',(.71,.78,.81),.20,.26)
    shader=ivory.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Emission Color'].default_value=(.84,.92,1,1)
    shader.inputs['Emission Strength'].default_value=.72
    for s in [-1,1]:
        parent=side_tag(s)+'_Headlight'
        def point(u,v,depth):
            return (s*(.694+.121*u),.706-.117*v+.008*u+.810*depth,
                    1.757+.162*v-.008*u+.585*depth)
        def lensring(r,d):
            return [point(r*cos(2*pi*i/96),r*sin(2*pi*i/96),d) for i in range(96)]
        tube('Headlamp aperture seal',lensring(1.006,-.001),.0048,seal,10,True)
        def annulus(name,rings,m):
            vs=[]
            for radius,depth in rings:vs.extend(lensring(radius,depth))
            mesh(name,vs,[(j*96+i,j*96+(i+1)%96,(j+1)*96+(i+1)%96,(j+1)*96+i)
                 for j in range(len(rings)-1) for i in range(96)],m)
        annulus('Headlamp black reflector bowl',[(1.00,.002),(.975,-.005),(.928,-.013),(.82,-.028),(.68,-.044),(.17,-.055),(.001,-.055)],bezel)
        annulus('Lens edge retainer',[(.978,-.002),(.967,-.004),(.953,-.0048)],lens)
        for row in [-.32,.32]:
            for col in [-.36,.36]:
                # The two optical barrels sit under each shelf, behind a squared cowl.
                def projector(u,v):
                    a=2*pi*u;uu=cos(a);vv=sin(a)
                    return point(col+.206*uu*v,row+.158*vv*v-.032,-.017-.012*v*v)
                patch('HD Matrix projector glass',40,10,projector,crystal)
                def barrel(u,v):
                    a=2*pi*u
                    return point(col+(.210+.022*v)*cos(a),row+(.165+.021*v)*sin(a)-.032,-.028-.012*v)
                patch('Projector metallised cup',36,5,barrel,chrome)
                cutout=roundrect(col,row-.034,.164,.113,3.4,44)
                shroud=roundrect(col,row-.038,.273,.172,4.7,44)
                rim=[point(x,v,-.0135) for x,v in shroud]
                opening=[point(x,v,-.0125) for x,v in cutout]
                n=len(rim)
                mesh('Rectangular projector cowl',rim+opening,
                     [(i,(i+1)%n,(i+1)%n+n,i+n) for i in range(n)],bezel)
                a=col-.053
                patch('Matrix projection element',24,6,
                      lambda u,v:point(a+.099*cos(2*pi*u)*v,row+.107*sin(2*pi*u)*v-.040,-.011-.003*v*v),lens)
                def shelf(u,v):
                    t=2*u-1;vv=row-.186+.045*t*col+.072*v
                    return point(col+.298*t,vv,-.001-.011*v-.004*t*t)
                patch('Light module upper shelf',24,9,shelf,bezel)
                patch('Four-point daytime light conductor',24,5,
                    lambda u,v:point(col+.28*(2*u-1),row-.199+.036*(2*u-1)*col+.044*v-.008*abs(2*u-1)**4,.0045),ivory)
        # Complete the lower housing with the short black shelf visible through the lens.
        patch('Headlight lower ledge',36,7,lambda u,v:point((2*u-1)*.44,.734+.052*v-.053*(2*u-1)**2,-.019+.005*v),bezel)
        patch('Convex outer headlamp cover',96,20,lambda u,v:point(v*cos(2*pi*u),v*sin(2*pi*u),
             .004+.015*(1-v*v)),clear)

def install_door_card(src):
    s=1 if '_L_' in src.name else -1
    me=src.data.copy();me.transform(src.matrix_world)
    for vert in me.vertices:
        z,x,y=vert.co
        vert.co=cv((s*(.713+(abs(x)-41.0)*.014),.180+(y-12.60)*.0187,-.584+(z+32.34)*.0175))
    ob=bpy.data.objects.new('Door trim '+src.name.split('_',3)[3].replace('_',' '),me)
    bpy.context.scene.collection.objects.link(ob);ob.parent=group(side_tag(s)+'_Door')
    if 'MicroSuede' in src.name or 'Stitch' in src.name:
        for i in range(len(me.materials)):me.materials[i]=leather

def finish_outer_skin():
    """Carry normals through mating body sections without welding separate panels."""
    from mathutils.kdtree import KDTree
    skin_roots={'Bodywork','Left_Door','Right_Door','Bonnet','Front_Bumper','Rear_Bumper'}
    skin=[]
    for ob in bpy.context.scene.objects:
        if ob.type!='MESH' or not ob.parent or ob.parent.name not in skin_roots:continue
        if (ob.name.startswith(('Front fascia','Rear bumper skin','Pressed bonnet','Front fender crown',
                                'Bumper upper return','Engine lid and rear shoulders','Door outer panel',
                                'Upper door rail','Upper quarter return','Painted rocker','Front arch and fender',
                                'Rear haunch and wheel arch','Bumper wing','Rocker end stamping'))):
            skin.append(ob)
    cache=[];tree=KDTree(sum(len(ob.data.vertices) for ob in skin))
    for ob in skin:
        for v in ob.data.vertices:
            tree.insert(v.co,len(cache));cache.append((ob,v.normal.copy()))
    tree.balance()
    for ob in skin:
        normals=[]
        for v in ob.data.vertices:
            initial=v.normal.copy();out=initial.copy()
            for _,index,distance in tree.find_range(v.co,.042):
                other,n=cache[index]
                if other==ob:continue
                dot=n.dot(initial)
                if abs(dot)<.14:continue
                out+=n*((1 if dot>0 else -1)*(1-distance/.042)**2)
            normals.append(out.normalized())
        ob.data.normals_split_custom_set_from_vertices(normals)

def fabric_surface():
    """A tileable woven normal map on the folding roof, with metric texture coordinates."""
    import numpy as np
    size=128
    yy,xx=np.mgrid[0:size,0:size]
    ax,ay=2*pi*xx/16,2*pi*yy/16
    # Weft rises above warp on alternating crossings.
    crossover=np.cos(ax/2)*np.cos(ay/2)
    height=(.55+.27*crossover)*np.cos(ax)+(.55-.27*crossover)*np.cos(ay)
    dx=(np.roll(height,-1,axis=1)-np.roll(height,1,axis=1))*.55
    dy=(np.roll(height,-1,axis=0)-np.roll(height,1,axis=0))*.55
    normal=np.stack([-dx,-dy,np.ones_like(dx)],axis=-1)
    normal/=np.linalg.norm(normal,axis=-1,keepdims=True)
    pixels=np.concatenate([.5+.5*normal,np.ones((size,size,1))],axis=-1).astype('f')
    im=bpy.data.images.new('Convertible canvas weave',width=size,height=size,alpha=True)
    im.colorspace_settings.name='Non-Color';im.pixels.foreach_set(pixels.ravel());im.pack()
    nodes=cloth.node_tree.nodes;texture=nodes.new('ShaderNodeTexImage');texture.image=im
    bump=nodes.new('ShaderNodeNormalMap');bump.inputs['Strength'].default_value=.53
    cloth.node_tree.links.new(texture.outputs['Color'],bump.inputs['Color'])
    cloth.node_tree.links.new(bump.outputs['Normal'],nodes.get('Principled BSDF').inputs['Normal'])
    for ob in group('Soft_Top').children:
        if ob.type!='MESH' or not any(m==cloth for m in ob.data.materials):continue
        me=ob.data
        uv=me.uv_layers.new(name='Canvas grain')
        for loop in me.loops:
            p=me.vertices[loop.vertex_index].co;uv.data[loop.index].uv=(p.x/.013,p.y/.013)
