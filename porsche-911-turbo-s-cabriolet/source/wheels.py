"""Five forked webs with auxiliary branches, center-lock hardware and separate PCCB brakes.

Outer diameters follow the specified 255/35 ZR20 and 325/30 ZR21 tires.
The left-front configurator close-up supplies the spoke openings, center lock,
caliper bridge profile and surface appearance. Tire shoulder sipes and the
carbon-ceramic drilling sequence are representative. The rectified locking face
comes from the factory exterior detail:
https://newsroom.porsche.com/dam/jcr:072c8358-f4dd-4320-93fd-5bbd0e28bf85/911%20Turbo%20S%20Cabrio%20Ice%20Grey%20Metallic%2027.jpg
"""

def wheel_material(name, filename, metal, roughness):
    material = mat(name, (.15,.15,.15), metal, roughness)
    texture = material.node_tree.nodes.new('ShaderNodeTexImage')
    texture.image = bpy.data.images.load(str(R/'source'/filename), check_existing=True)
    texture.image.pack()
    material.node_tree.links.new(texture.outputs['Color'],
                                 material.node_tree.nodes['Principled BSDF'].inputs['Base Color'])
    return material


def wheels():
    global parent
    face = mat('Turbonite turned spoke face', (.215,.208,.184), .76,.34)
    recess = mat('Dark satin wheel pockets', (.014,.015,.015), .31,.46)
    hat = mat('PCCB rotor bell', (.046,.048,.052), .49,.43)
    nut = mat('Anthracite center lock', (.048,.049,.046), .59,.39)
    # Ice Grey Metallic 27 supplies the stamped shield, dual-language locking
    # legend and torque scale, all upright in the photographed centre.
    cap = wheel_material('Centerlock III Turbonite crest and legend','wheel-cap.png',.34,.40)
    ceramic = wheel_material('PCCB friction surface','friction-surface.jpg',.32,.55)
    sidewalls = [wheel_material('20-inch embossed tire sidewall','front-sidewall.webp',.01,.70),
                 wheel_material('21-inch embossed tire sidewall','rear-sidewall.webp',.01,.70)]

    for axle, z, x, width, radius, rim, disc in [
            ('Front',1.225,.7915,.255,.34325,.254,.210),
            ('Rear',-1.225,.794,.325,.3642,.2667,.205)]:
        for side in (1,-1):
            position = axle + ('_Left' if side == 1 else '_Right')
            parent = position + '_Wheel'
            w = width/2
            def polar(r, a, depth):
                return (side*(x+depth), radius+r*sin(a), z+r*cos(a))

            def lathe(name, layers, material, segments=120, texture=None, smooth=True, emboss=None):
                points = [polar(r,2*pi*i/segments,depth+(emboss(r,2*pi*i/segments) if emboss else 0))
                          for r,depth in layers for i in range(segments)]
                faces = [(j*segments+i,j*segments+(i+1)%segments,(j+1)*segments+(i+1)%segments,(j+1)*segments+i)
                         for j in range(len(layers)-1) for i in range(segments)]
                obj = mesh(name, points, faces, material, smooth)
                # The turned annular faces are flat. A grazing reflection should
                # stop at the shoulder and run concentrically around each face.
                for n, polygon in enumerate(obj.data.polygons):
                    row = n // segments
                    if abs(layers[row+1][1]-layers[row][1]) < .16*abs(layers[row+1][0]-layers[row][0]):
                        polygon.use_smooth = False
                if texture:
                    uv = obj.data.uv_layers.new(name='Wheel UV')
                    for row in range(len(layers)-1):
                        for i in range(segments):
                            poly = obj.data.polygons[row*segments+i]
                            for index, (n, k) in zip(poly.loop_indices,[(row,i),(row,i+1),(row+1,i+1),(row+1,i)]):
                                r = layers[n][0]
                                angle = 2*pi*k/segments
                                if texture == 'tire':
                                    coord = (1-side*k/segments, 1-(r-layers[0][0])/(layers[-1][0]-layers[0][0]))
                                else:
                                    coord = (.5-side*.5*r*cos(angle)/layers[-1][0],
                                             .5+.5*r*sin(angle)/layers[-1][0])
                                uv.data[index].uv = coord
                return obj

            # Rounded asymmetric outer shoulder, bead and four open tread channels.
            tread = [(rim*.996,-w*.88),(radius-.045,-w),(radius-.023,-w*1.016),
                     (radius-.008,-w*.94),(radius,-w*.76)]
            for channel in (-.52,-.16,.21,.53):
                tread += [(radius,w*(channel-.034)),(radius-.006,w*(channel-.020)),
                          (radius-.006,w*(channel+.018)),(radius,w*(channel+.031))]
            tread += [(radius,w*.76),(radius-.004,w*.865),(radius-.0085,w*.934)]
            lathe('Performance tire tread and grooves', tread, rubber)
            lathe('Tire sidewall lettering',[(rim*1.025,w*.912),(rim*1.058,w*.985),
                  (radius-.041,w*1.014),(radius-.025,w*1.016),(radius-.014,w*.976),
                  (radius-.0085,w*.934)],sidewalls[axle=='Rear'],texture='tire')
            # The thin diagonals are inset in the outer tread shoulders.
            for sign in (-1,1):
                vertices, faces = [], []
                for repeat in range(58):
                    angle=2*pi*repeat/58+(.01 if sign==1 else -.01)
                    for fraction,r in [(.959,radius-.010),(.92,radius-.0064),
                                       (.85,radius-.0031),(.77,radius+.0002),(.64,radius+.0003)]:
                        a=angle+sign*(.959-fraction)*.145
                        vertices.extend([polar(r+.0002,a+off,sign*w*fraction) for off in [-.0024,.0024]])
                    base=repeat*10
                    faces.extend([(base+j,base+j+1,base+j+3,base+j+2) for j in range(0,8,2)])
                mesh('Tire shoulder slots',vertices,faces,well,False)
            lathe('Wheel rim and inner barrel',[(rim*1.035,-w*.93),(rim*1.04,-w*.88),(rim*.955,-w*.76),
                  (rim*.94,w*.59),(rim*.957,w*.83),(rim*1.00,w*.96),(rim*1.042,w*.983),
                  (rim*1.06,w*.964),(rim*1.046,w*.827)],recess)
            lathe('Turned rim lip',[(rim*.976,w*.946),(rim*.981,w*.980),(rim*.986,w*.990),
                  (rim*1.041,w*.990),(rim*1.048,w*.984),(rim*1.052,w*.965)],face)

            # The CNC face is a broad, flat land with a bevel above darker forged
            # sidewalls. The second branch leaves the web well outside the lock ring.
            def blade(name, axis, stations):
                ring_points = []
                for rr, angle, halfspan, offset, thickness in stations:
                    a = axis+angle
                    width_rad = halfspan/max(rr,.001)
                    depth = w*offset
                    ring_points += [polar(rr,a+width_rad*k,depth+dz) for k,dz in [
                        (-.81,0),(.81,0),(1,-.0013),(1.13,-thickness),(-1.13,-thickness),(-1,-.0013)]]
                polys=[tuple(range(5,-1,-1))]
                for row in range(len(stations)-1):
                    for k in range(6):
                        start=row*6
                        polys.append((start+k,start+(k+1)%6,start+6+(k+1)%6,start+6+k))
                polys.append(tuple(range(len(ring_points)-6,len(ring_points))))
                obj=mesh(name,ring_points,polys,recess,False)
                obj.data.materials.append(face)
                for row in range(len(stations)-1):
                    for k in (0,1,5):
                        obj.data.polygons[1+row*6+k].material_index=1

            for repeat in range(5):
                axis=-pi/2+2*pi*repeat/5
                blade('Center spider web',axis,[
                    (.054,0,.024,.65,.031),(.062,0,.027,.66,.032),(.070,0,.029,.69,.031),
                    (.077,0,.028,.72,.027),(.082,0,.027,.74,.024),(.086,0,.025,.75,.023)])
                for hand in (-1,1):
                    blade('Primary forked spoke',axis,[(rr,hand*ang,hw*1.42,depth,th*1.45) for rr,ang,hw,depth,th in [
                        (.074,.142,.010,.709,.021),(.082,.180,.0081,.744,.021),
                        (.090,.204,.0063,.760,.021),(.111,.198,.0049,.781,.021),
                        (.153,.164,.0040,.826,.020),(.209,.126,.0037,.906,.018),
                        (rim*.94,.110,.0038,.960,.014),(rim*.970,.108,.0040,.985,.014),
                        (rim*.981,.108,.0045,.99,.013),(rim*1.01,.107,.010,.99,.013)]])
                    blade('Flying outer spoke',axis,[(rr,hand*ang,hw*1.42,depth,th*1.45) for rr,ang,hw,depth,th in [
                        (.103,.190,.0062,.737,.022),(.117,.212,.0056,.752,.022),
                        (.137,.256,.0044,.775,.024),(.164,.308,.0037,.819,.021),
                        (.216,.387,.0035,.902,.017),(rim*.941,.404,.0037,.960,.013),(rim*.972,.410,.0040,.987,.013),
                        (rim*.982,.411,.0045,.99,.012),(rim*1.012,.413,.009,.99,.012)]])
            lathe('Forged wheel hub',[(0,w*.40),(.069,w*.46),(.071,w*.60),(.066,w*.677),
                                      (.058,w*.690),(.049,w*.68)],recess)
            lathe('Machined center shoulder',[(.045,w*.682),(.051,w*.711),(.061,w*.711),
                  (.064,w*.699),(.066,w*.676)],face)
            lock_radius=.185*rim
            def lock_relief(r,a):
                # The stamped gripping collar stands proud of the recessed legend.
                radial=max(0,1-abs(r/lock_radius-.62)/.13)
                return .0019*radial*((1+cos(14*a))*.5)**.6
            profile=[(.0001,.787),(.32,.787),(.466,.782),(.50,.792),(.53,.766),
                     (.58,.782),(.62,.794),(.69,.788),(.72,.753),(.744,.741),
                     (.765,.742),(.90,.742),(.971,.745),(.985,.751),(1,.743)]
            lathe('Center lock Turbonite crest and locking legend',
                  [(r*lock_radius,w*d) for r,d in profile],cap,168,texture='radial',emboss=lock_relief)
            a=-1.61
            shaft('Tire inflation valve',polar(rim*.91,a,w*.65),polar(rim*.952,a,w*.73),.004,seal)
            shaft('Tire valve cap',polar(rim*.901,a,w*.637),polar(rim*.919,a,w*.674),.005,black)

            parent = position + '_Brake'
            depth = w*.79-.059
            thickness = .040 if axle=='Front' else .032
            lathe('Vented ceramic rotor edge',[(disc-.014,depth-thickness),(disc,depth-thickness),
                  (disc,depth-thickness*.78),(disc-.003,depth-thickness*.70),
                  (disc-.003,depth-thickness*.26),(disc,depth-thickness*.20),(disc,depth)],carbon)
            lathe('Rotor friction ring back',[(.094,depth-thickness),(disc,depth-thickness)],ceramic,texture='radial')
            # Tessellate each annular cell around an open bore. The depth and the
            # dark bore walls remain visible on close approach, including wheel-off.
            points, quads, sides = [], [], []
            for row in range(4):
                r0=.092+(disc-.092)*row/4
                r1=.092+(disc-.092)*(row+1)/4
                for i in range(31):
                    a=2*pi*i/31+.034*row
                    corners=[(r0,a),(r0,a+2*pi/31),(r1,a+2*pi/31+.034),(r1,a+.034)]
                    midr=(r0+r1)/2; mida=a+pi/31+.017
                    ring=[]
                    for j in range(4):
                        rA,aA=corners[j];rB,aB=corners[(j+1)%4]
                        for k in range(4):
                            f=k/4; ring.append((rA+(rB-rA)*f,aA+(aB-aA)*f))
                    start=len(points)
                    for level in range(3):
                        for rr,aa in ring:
                            if level:
                                dr,da=rr-midr,(aa-mida)*midr
                                scale=.00225/math.hypot(dr,da)
                                rr,aa=midr+dr*scale,mida+da*scale/midr
                            points.append(polar(rr,aa,depth-(.006 if level==2 else 0)))
                    for k in range(16):
                        kk=(k+1)%16
                        quads.append((start+k,start+kk,start+16+kk,start+16+k))
                        sides.append((start+16+k,start+16+kk,start+32+kk,start+32+k))
            rotor=mesh('Drilled ceramic friction ring',points,quads+sides,ceramic,False)
            rotor.data.materials.append(carbon);uv=rotor.data.uv_layers.new(name='Friction surface UV')
            for n,poly in enumerate(rotor.data.polygons):
                if n>=len(quads):poly.material_index=1
                for li in poly.loop_indices:
                    point=points[rotor.data.loops[li].vertex_index]
                    uv.data[li].uv=((point[2]-z)*8,(point[1]-radius)*8)
            lathe('Rotor bell and retaining flange',[(.046,depth+.009),(.050,depth+.011),(.074,depth+.012),
                  (.085,depth+.004),(.101,depth+.002),(.106,depth-.004)],hat)
            for i in range(10):
                a=2*pi*i/10+.12
                shaft('Floating rotor bobbin',polar(.094,a,depth+.0005),polar(.094,a,depth+.007),.0065,brushed)
                tube('Rotor bobbin recess',[polar(.094,a,depth+.007),polar(.094,a,depth+.008)],.0030,nut,6)
            # Piston lobes and outer bridge give the caliper real thickness.
            center_angle=pi if axle=='Front' else 0
            rz = .207 if axle=='Front' else .201
            def caliper_section(n, start, end, inner, outer, deep, material):
                sections=[]
                for k in range(27):
                    f=k/26; angle=center_angle+start+(end-start)*f
                    roll=sin(pi*f)**.42
                    ri=inner+.009*(1-roll);ro=outer-.013*(1-roll)
                    sections.append([polar(r,angle,d) for r,d in [
                        (ri+.002,deep-.018),(ri,deep+.001),(ri+.004,deep+.009*roll),
                        (ro-.005,deep+.012*roll),(ro,deep+.004*roll),(ro,deep-.026),
                        (ro-.004,deep-.035),(ri+.004,deep-.031)]])
                faces=[tuple(reversed(range(8)))]
                for k in range(26):
                    for j in range(8):faces.append((k*8+j,k*8+(j+1)%8,(k+1)*8+(j+1)%8,(k+1)*8+j))
                faces.append(tuple(range(26*8,27*8)))
                return mesh(n,[p for ring in sections for p in ring],faces,material)
            front_depth=depth+.027
            caliper_section('Monobloc caliper casting',-.71,.71,rz-.062,rz+.007,front_depth,yellow)
            caliper_section('Caliper pad bridge',-.47,.47,rz-.068,rz-.011,front_depth+.008,yellow)
            for a in [-.57,-.30,0,.30,.57]:
                p=polar(rz-.030,center_angle+a,front_depth+.002)
                ellipse('Caliper piston housing',p,(.011,.026,.017),yellow,20,12)
            for a in [-.51,.51]:
                shaft('Caliper bridge pin',polar(rz-.050,center_angle+a,front_depth+.005),
                      polar(rz-.050,center_angle+a,front_depth+.020),.006,brushed)
                shaft('Caliper pin bore',polar(rz-.050,center_angle+a,front_depth+.020),
                      polar(rz-.050,center_angle+a,front_depth+.0205),.003,well)
            bx,by,bz=polar(rz-.038,center_angle,front_depth+.023)
            badge_relief('PORSCHE caliper lettering','caliper',.086,.0082,
                         lambda u,v,d:(bx+side*d,by-u,bz-side*v),black,.00025)
    parent='Bodywork'
