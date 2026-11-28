"""External 992.2 Turbo S power unit reconstruction in the car's metric frame.

Porsche's released head-on engine photograph sets the two separated low eTurbos,
opposed banks, upper induction and rear crossmember.  The 992.2 Turbo S whole-car
section constrains the aft engine and forward-facing PDK package.  Port diameters,
casting outlines, cable paths and ribs are approximate.  No hidden gear set.
"""
def build_hybrid():
    global parent
    alloy = mat('Aluminium engine castings', (.48,.49,.49), .48, .52)
    flange = mat('Machined flange edges', (.48,.49,.49), .76, .30)
    graphite = mat('Low gloss cast alloy', (.16,.17,.17), .47, .5)
    polymer = mat('Moulded black induction', (.025,.028,.030), .03, .43)
    hose = mat('Elastomer cooling hoses', (.025,.027,.029), .01, .68)
    hot = mat('Heat marked exhaust steel', (.196,.155,.119), .66, .45)
    orange = mat('High voltage orange sheath', (.82,.145,.014), .03, .42)
    bolt = mat('Zinc plated fasteners', (.41,.425,.43), .76,.28)

    def section(name, stations, material, exponent=3.0):
        # Each longitudinal station gives z, center x/y and both half-dimensions.
        rings=[]
        for z,cx,cy,w,h in stations:
            rings.append([(x,y,z) for x,y in roundrect(cx,cy,w,h,exponent,56)])
        return rings_mesh(name,rings,material)
    def rings_mesh(name,rings,material,cap=True):
        n=len(rings[0]);fs=[tuple(range(n-1,-1,-1))] if cap else []
        for i in range(len(rings)-1):
            for j in range(n):
                k=(j+1)%n;fs.append((i*n+j,i*n+k,(i+1)*n+k,(i+1)*n+j))
        if cap: fs.append(tuple((len(rings)-1)*n+j for j in range(n)))
        ob=mesh(name,[p for ring in rings for p in ring],fs,material)
        if cap:
            ob.data.polygons[0].use_smooth=False
            ob.data.polygons[-1].use_smooth=False
        return ob
    def hexbolt(name,pos,axis,r=.007):
        p=Vector(pos);d=Vector(axis).normalized()
        tube(name,[p,p+d*.006],r,bolt,6)
    def ringz(name,x,y,z,rad,section_rad,m):
        return tube(name,[(x+rad*cos(i*2*pi/56),y+rad*sin(i*2*pi/56),z) for i in range(56)],section_rad,m,9,True)
    def bend(name,controls,r,m):
        from mathutils.geometry import interpolate_bezier
        pts=[Vector(v) for v in controls];samples=[]
        for i,(p,q) in enumerate(zip(pts,pts[1:])):
            a=pts[max(i-1,0)];b=pts[min(i+2,len(pts)-1)]
            seg=interpolate_bezier(p,p+(q-a)/6,q-(b-p)/6,q,9)
            samples.extend(seg if not i else seg[1:])
        return tube(name,samples,r,m,11)
    def raised_rim(name,z,cx,cy,w,h,m,th=.004):
        return tube(name,[(x,y,z) for x,y in roundrect(cx,cy,w,h,3.3,56)],th,m,8,True)

    def facecasting(name,profile,zmin,zmax,material):
        cx=sum(p[0] for p in profile)/len(profile);cy=sum(p[1] for p in profile)/len(profile)
        rings=[[(cx+(x-cx)*k,cy+(y-cy)*k,z) for x,y in profile] for z,k in [(zmin,.97),(zmin+.005,1),(zmax-.006,1),(zmax,.965)]]
        return rings_mesh(name,rings,material)
    def open_pipe(name,a,b,r,material):
        a,b=Vector(a),Vector(b);axis=(b-a).normalized();t=axis.cross(Vector((0,1,0))).normalized();u=axis.cross(t)
        def ring(c,rad):return [c+rad*(t*cos(2*pi*j/36)+u*sin(2*pi*j/36)) for j in range(36)]
        rings=[ring(a,r*.89),ring(a+axis*.005,r),ring(b-axis*.003,r),ring(b,r*.90),ring(b-axis*.022,r*.84)]
        rings_mesh(name,rings,material,False)
        mesh(name+' dark bore',ring(b-axis*.023,r*.845),[tuple(range(36))],well,False)

    parent='Boxer_Engine'
    section('Longitudinal split crankcase',[
        (-1.98,0,.391,.112,.132),(-1.94,0,.401,.159,.158),
        (-1.80,0,.401,.170,.159),(-1.46,0,.406,.170,.167),
        (-1.362,0,.416,.162,.156),(-1.325,0,.410,.151,.140)],alloy)
    section('Lower oil sump casting',[
        (-1.97,0,.272,.104,.052),(-1.93,0,.246,.142,.065),
        (-1.77,0,.238,.149,.065),(-1.43,0,.250,.150,.062),(-1.385,0,.277,.135,.051)],graphite)
    for x in [-.100,-.060,-.020,.020,.060,.10]:
        bend('Sump longitudinal ribs',[(x,.194,-1.90),(x,.178,-1.73),(x,.199,-1.44)],.005,alloy)
    for s in [-1,1]:
        q=.021*s
        section('Three-cylinder horizontal bank',[
            (-1.923+q,s*.228,.408,.138,.113),(-1.875+q,s*.237,.410,.148,.130),
            (-1.475+q,s*.242,.409,.146,.128),(-1.416+q,s*.231,.409,.130,.111)],alloy)
        # Outboard timing cover and cam-cover perimeter enclose the three chambers.
        rings=[]
        for x,rad in [(s*.335,.93),(s*.351,1),(s*.389,1.01),(s*.409,.96)]:
            yz=roundrect(-1.666+q,.421,.267*rad,.124*rad,3.4,64)
            rings.append([(x,y,z) for z,y in yz])
        rings_mesh('Camshaft carrier cover',rings,graphite)
        for rr,x,m in [(1,.395,alloy),(.955,.410,flange)]:
            pts=[(s*x,y,z) for z,y in roundrect(-1.666+q,.421,.25*rr,.120*rr,3.4,64)]
            tube('Cam cover seam',pts,.004,m,8,True)
        for z in [-1.84+q,-1.67+q,-1.50+q]:
            ellipse('Ignition coil housing',(s*.410,.428,z),(.017,.034,.047),polymer,20,10)
            bend('Ignition harness branch',[(s*.422,.438,z),(s*.428,.477,z+.016),(s*.411,.493,z+.055)],.0045,hose)
            for y in [.328,.514]:hexbolt('Cam cover fastener',(s*.404,y,z+.044),(s,0,0))
            # The six separate runners descend from a pair of black intake plenums.
            bend('Boxer induction runner',[(s*.138,.621,z),(s*.190,.641,z-.019),(s*.251,.589,z-.018),(s*.275,.523,z)],.029,polymer)
            # Exhaust ports below each horizontal bank converge toward the hot housing.
            bend('Exhaust primary',[(s*.338,.325,z),(s*.373,.278,z),(s*.411,.265,(z-1.71)/2),(s*.492,.319,-1.653)],.022,hot)
        bend('Coil cable rail',[(s*.416,.492,-1.90+q),(s*.417,.494,-1.67+q),(s*.417,.49,-1.44+q)],.008,polymer)
        section('Intake distribution plenum',[
            (-1.895,s*.210,.626,.071,.064),(-1.85,s*.212,.631,.082,.070),(-1.49,s*.213,.627,.082,.073),(-1.45,s*.207,.628,.066,.061)],polymer)
        for i in range(9):
            z=-1.842+.039*i
            raised_rim('Moulded induction rib',z,s*.211,.629,.081,.070,polymer,.003)
        # Upper throttle connections and cable/pipe bridge visible on the factory photograph.
        bend('Throttle body casting',[(s*.19,.633,-1.904),(s*.159,.643,-1.925),(s*.113,.642,-1.937)],.042,alloy)
        bend('Charge air crossover',[(s*.512,.456,-1.766),(s*.442,.528,-1.699),(s*.34,.633,-1.553),(s*.237,.644,-1.480)],.035,hose)
        bend('Upper coolant line',[(s*.328,.504,-1.920),(s*.370,.585,-1.863),(s*.342,.710,-1.608),(s*.222,.710,-1.510),(s*.135,.681,-1.408)],.012,hose)
        for z in [-1.822,-1.685,-1.548]:
            for y in [.367,.459]:
                bend('Rear facing cast reinforcement',[(s*.166,y,z),(s*.253,y+.018,z+.028),(s*.346,y,z)],.005,alloy)
        # Separate ends of the structural crossmember double as a visual locator.
        section('Rear engine carrier wing',[
            (-2.005,s*.233,.315,.126,.038),(-1.986,s*.234,.316,.137,.045),(-1.938,s*.235,.313,.130,.035)],alloy)
        for xx in [.137,.316]:hexbolt('Rear carrier fastener',(s*xx,.319,-2.007),(0,0,-1))
    section('Rear engine carrier center',[
        (-2.011,0,.337,.101,.067),(-1.987,0,.334,.127,.063),(-1.958,0,.338,.119,.071)],alloy)
    for x in [-.070,.070]:
        for y in [.306,.371]:hexbolt('Carrier mounting bolt',(x,y,-2.012),(0,0,-1),.010)
    for side in [-1,1]:
        bend('Crankcase division rib',[(side*.11,.321,-1.961),(side*.13,.405,-1.982),(side*.11,.49,-1.962)],.006,flange)
    # Flat, finned electronics enclosure above the engine as in the installed cutaway.
    section('Upper power electronics housing',[
        (-1.805,-.043,.682,.107,.029),(-1.79,-.043,.682,.118,.034),
        (-1.49,-.043,.682,.119,.034),(-1.477,-.043,.682,.108,.027)],alloy,5)
    for x in [-.135,-.112,-.089,-.066,-.043,-.020,.003,.026,.049]:
        path('Power module cooling ridge',[(x,.718,-1.766),(x,.718,-1.512)],.003,flange)

    # The engine photo shows exposed, shallow timing-case faces, with their
    # reinforcement webs visible above the carrier rather than on top of the banks.
    for side in [-1,1]:
        pxy=[(.080,.347),(.116,.323),(.323,.319),(.365,.330),(.403,.344),(.407,.395),(.397,.438),(.358,.492),(.345,.517),(.297,.538),(.228,.523),(.178,.517),(.111,.507),(.081,.488)]
        facecasting('Rear timing case end plate',[(side*x,y) for x,y in pxy],-1.985,-1.921,alloy)
        outline=[(side*x,y,-1.987) for x,y in pxy];tube('Timing cover machined perimeter',outline,.0045,flange,8,True)
        for y in [.359,.384,.433,.461]:
            bend('Timing plate cast web',[(side*.088,y,-1.987),(side*.19,y+.002,-1.990),(side*.287,y+.008,-1.988),(side*.355,y+.015,-1.987)],.004,alloy)
        for x in [.189,.296]:
            path('Timing plate transverse web',[(side*x,.331,-1.987),(side*(x+.004),.462,-1.989),(side*(x-.005),.515,-1.986)],.004,alloy)
        for x,y in [( .113,.490),(.189,.437),(.218,.354),(.275,.498),(.338,.383),(.359,.469),(.392,.362),(.326,.331)]:
            shaft('Timing plate cast boss',(side*x,y,-1.986),(side*x,y,-1.996),.013,alloy)
            hexbolt('Timing cover stud',(side*x,y,-1.998),(0,0,-1),.0064)
        facecasting('Cam sensor and connector',[(side*(x+.343),y+.450) for x,y in [(-.019,-.012),(-.020,.009),(-.008,.018),(.015,.015),(.023,.002),(.016,-.016)]],-2.002,-1.994,polymer)
        bend('Timing cover sensor wire',[(side*.342,.465,-2.004),(side*.326,.484,-2.008),(side*.218,.516,-2.005),(side*.193,.540,-1.985)],.0032,hose)
        bend('Front bank harness tray',[(side*.002,.524,-1.978),(side*.099,.524,-1.982),(side*.218,.527,-1.977),(side*.291,.556,-1.960),(side*.351,.562,-1.929)],.0145,polymer)
        bend('Front bank coolant hoses',[(side*.020,.54,-1.966),(side*.147,.538,-1.976),(side*.261,.560,-1.950),(side*.290,.601,-1.925),(side*.274,.651,-1.903)],.0095,hose)
        bend('Upper induction hoses',[(side*.381,.519,-1.777),(side*.354,.597,-1.784),(side*.323,.674,-1.767),(side*.268,.712,-1.758),(side*.237,.719,-1.705)],.015,hose)
        bend('Upper induction cross hose',[(side*.265,.663,-1.692),(side*.201,.696,-1.718),(side*.118,.693,-1.756),(side*.066,.673,-1.816),(side*.032,.670,-1.833)],.013,hose)
        for x,y,z in [(.356,.581,-1.811),(.315,.638,-1.791),(.228,.684,-1.720)]:
            bend('Small bore vacuum line',[(side*x,y,z),(side*(x-.053),y+.020,z-.027),(side*(x-.118),y+.014,z-.034)],.004,hose)
        # Each isolated factory compressor has a separate dark rear-facing duct above it.
        open_pipe('Dark charge pipe opening',(side*.455,.477,-1.774),(side*.505,.502,-1.873),.039,polymer)
        open_pipe('Throttle body mouth',(side*.185,.621,-1.868),(side*.112,.627,-1.931),.039,alloy)
        path('Fuel rail section',[(side*.337,.554,-1.894),(side*.342,.554,-1.525)],.009,alloy)
    carrier=[(-.419,.313),(-.406,.328),(-.343,.336),(-.235,.332),(-.154,.345),(-.109,.381),(-.084,.394),(.084,.394),(.109,.381),(.154,.345),(.235,.332),(.343,.336),(.406,.328),(.419,.313),(.405,.262),(.382,.245),(.161,.249),(.136,.232),(.102,.224),(.097,.199),(-.097,.199),(-.102,.224),(-.136,.232),(-.161,.249),(-.382,.245),(-.405,.262)]
    facecasting('Contoured rear engine support',carrier,-2.039,-1.995,alloy)
    for side in [-1,1]:
        path('Engine support pressed upper edge',[(side*.012,.388,-2.040),(side*.090,.386,-2.038),(side*.157,.343,-2.038),(side*.250,.331,-2.039),(side*.394,.327,-2.039)],.005,flange)
        for x,y,r in [(.045,.346,.013),(.054,.270,.010),(.176,.327,.009),(.300,.320,.008),(.390,.316,.008)]:
            shaft('Carrier countersink',(side*x,y,-2.038),(side*x,y,-2.044),r,graphite)
            hexbolt('Support bracket bolt',(side*x,y,-2.046),(0,0,-1),r*.69)
    facecasting('Upper carrier pedestal',[(-.095,.388),(-.092,.475),(-.074,.487),(.083,.474),(.096,.462),(.091,.390)],-2.013,-1.974,alloy)
    for x,y in [(-.072,.461),(.073,.451)]:shaft('Carrier pedestal stud',(x,y,-2.008),(x,y,-2.018),.012,graphite)

    # Visible intake and harness hardware between the two upper pipes.
    for side in [-1,1]:
        section('Upper intake collector', [(-1.884,side*.190,.614,.048,.052),(-1.868,side*.192,.617,.058,.062),(-1.716,side*.196,.617,.059,.063),(-1.686,side*.197,.619,.046,.055)],polymer,3.4)
        facecasting('Throttle actuator cover',[(side*(x+.204),y+.614) for x,y in [(-.026,-.040),(.012,-.044),(.026,-.021),(.025,.020),(.004,.042),(-.023,.030)]],-1.924,-1.881,graphite)
        for k in range(5):
            z=-1.865+k*.029
            raised_rim('Intake casting ribs',z,side*.192,.617,.058,.062,polymer,.0036)
        bend('Central air duct',[(side*.118,.624,-1.899),(side*.078,.622,-1.865),(side*.058,.587,-1.852),(side*.036,.569,-1.851)],.031,polymer)
        bend('Central hose curve',[(side*.179,.654,-1.813),(side*.138,.664,-1.829),(side*.086,.669,-1.805),(side*.055,.681,-1.772)],.0085,hose)
        bend('Upper wiring band',[(side*.281,.644,-1.817),(side*.249,.671,-1.805),(side*.183,.693,-1.822),(side*.122,.710,-1.789),(side*.084,.718,-1.740)],.007,polymer)
        open_pipe('Intake junction lip',(side*.147,.622,-1.899),(side*.121,.630,-1.922),.039,alloy)
        for x,z in [(.305,-1.783),(.263,-1.801),(.120,-1.815)]:
            shaft('Harness connector',(side*x,.686,z),(side*(x-.013),.697,z-.017),.013,graphite)
        for x in [.292,.266,.241]:
            path('Clip ribs',[(side*x,.653,-1.834),(side*x,.660,-1.808),(side*x,.659,-1.785)],.003,alloy)
    facecasting('Lift eye mounting base',[(-.153,.681),(-.139,.719),(-.114,.734),(-.079,.719),(-.064,.674)],-1.730,-1.701,alloy)
    ringz('Engine lifting eye',-.114,.719,-1.731,.013,.0045,graphite)
    section('Electrical connector manifold',[(-1.879,.033,.604,.044,.036),(-1.863,.036,.606,.055,.043),(-1.780,.036,.608,.057,.044)],graphite,3.5)
    for i in range(3):
        bend('Electrical loom',[(.012+.017*i,.639,-1.843),(.016+.017*i,.668,-1.832),(.015+.012*i,.681,-1.795),(.001+.015*i,.679,-1.766)],.0055,hose)
    # Register the upper induction height and central support lip against the
    # photograph while preserving turbo centre coordinates and the PDK interface.
    for ob in group('Boxer_Engine').children:
        for vert in ob.data.vertices:
            v=vert.co
            if v.z>.51: v.z+=.127*min(1,(v.z-.51)/.199)
            elif v.z<.35: v.z+=.045*min(1,(.35-v.z)/.14)

    for s,assembly in [(1,'Left_eTurbo'),(-1,'Right_eTurbo')]:
        parent=assembly;cx=s*.510;cy=.346
        # Longitudinal spindle: compressor at the outboard rear, e-machine in
        # the center cartridge, then the exhaust turbine towards the heads.
        shaft('Electric turbo center cartridge',(cx,cy,-1.864),(cx,cy,-1.720),.048,graphite)
        for z in [-1.855,-1.837,-1.742,-1.729]:ringz('Center cartridge collar',cx,cy,z,.052,.006,alloy)
        for z,r,m in [(-1.904,.079,alloy),(-1.683,.080,hot)]:
            # Taper the scroll cross section towards the small end of its tongue.
            rings=[]
            for i in range(62):
                a=2*pi*i/61+.17;rad=r*(.78+.21*i/61);rr=.019+.010*i/61
                ring=[]
                for j in range(12):
                    b=2*pi*j/12
                    ring.append((cx+s*(rad+rr*cos(b))*cos(a),cy+(rad+rr*cos(b))*sin(a),z+rr*sin(b)))
                rings.append(ring)
            rings_mesh('Compressor scroll' if m==alloy else 'Exhaust turbine scroll',rings,m)
            section('Turbocharger backplate',[(z+.008,cx,cy,r*.88,r*.88),(z+.020,cx,cy,r*.87,r*.87)],m,2)
        # The compressor inlet, swept vanes and spindle face are exposed in the isolated engine photograph.
        shell=[]
        for z,r in [(-1.970,.067),(-1.964,.070),(-1.935,.069),(-1.914,.077),(-1.889,.079),(-1.895,.044),(-1.960,.043),(-1.972,.046)]:
            shell.append([(cx+r*cos(2*pi*i/64),cy+r*sin(2*pi*i/64),z) for i in range(64)])
        rings_mesh('Compressor inlet and volute neck',shell+[shell[0]],alloy,False)
        ringz('Compressor inlet machined lip',cx,cy,-1.971,.047,.003,flange)
        # Interior bore is dark so that the actual vane surfaces read in depth.
        shaft('Compressor bore shadow',(cx,cy,-1.929),(cx,cy,-1.933),.045,well)
        for k in range(9):
            a=k*2*pi/9
            def impeller(u,v):
                r=.009+.0275*v;ang=a+.48*v+.35*u*(1-.38*v)
                return (cx+r*cos(ang),cy+r*sin(ang),-1.957+.020*v+.006*u)
            patch('Swept compressor wheel vane',3,9,impeller,flange)
        ellipse('Compressor shaft nose',(cx,cy,-1.955),(.011,.011,.012),flange,18,9)
        for k in range(4):
            a=pi/4+k*pi/2;hexbolt('Compressor cover bolt',(cx+.076*cos(a),cy+.076*sin(a),-1.943),(0,0,-1),.006)
        bend('Compressor tangential outlet',[(cx+s*.061,cy+.017,-1.902),(cx+s*.089,cy+.044,-1.900),(cx+s*.090,cy+.077,-1.88),(cx+s*.078,cy+.119,-1.865)],.026,alloy)
        bend('Compressor silicone connection',[(cx+s*.077,cy+.123,-1.864),(cx+s*.052,cy+.154,-1.837),(cx+s*.008,cy+.145,-1.778),(cx,cy+.116,-1.758)],.030,hose)
        open_pipe('Upturned compressor outlet',(cx+s*.031,cy+.065,-1.885),(cx+s*.064,cy+.125,-1.930),.0225,alloy)
        bend('ETurbo motor electrical connector',[(cx,cy+.043,-1.793),(cx-s*.014,cy+.070,-1.782),(cx-s*.040,cy+.084,-1.765)],.012,orange)
        bend('ETurbo high-voltage lead',[(cx-s*.038,cy+.083,-1.765),(s*.447,.465,-1.695),(s*.387,.577,-1.535),(s*.283,.687,-1.484),(s*.092,.706,-1.510)],.0068,orange)

    parent='PDK_Hybrid'
    # Joined bellhousing encloses the permanent-magnet motor at the engine/PDK interface.
    shp=[(-1.354,.412,.156,.154),(-1.321,.414,.192,.178),(-1.266,.416,.217,.189),
         (-1.195,.416,.222,.188),(-1.110,.407,.192,.164),(-.997,.386,.162,.149),
         (-.845,.384,.148,.137),(-.731,.388,.125,.123),(-.595,.392,.100,.088),
         (-.531,.395,.070,.061)]
    section('Eight-speed PDK and motor housing',[(z,0,y,x,h) for z,y,x,h in shp],alloy,3.35)
    for z,y,w,h in [(-1.325,.414,.192,.178),(-1.275,.414,.216,.184),(-1.190,.416,.217,.184),(-1.105,.407,.190,.164),(-.849,.384,.146,.135),(-.627,.393,.106,.095)]:
        raised_rim('Machined casing split flange',z,0,y,w,h,flange,.006)
    def profile(z,i):return linterp([(row[0],row[i]) for row in shp],z)
    for side in [-1,1]:
        for i in range(6):
            z=-1.19+.080*i
            pts=[]
            for k in range(13):
                t=k/12;zz=z+.071*t
                y=profile(zz,1);w=profile(zz,2);h=profile(zz,3)
                a=-.64+1.42*t
                pts.append((side*(w+.002)*abs(cos(a))**.73,y+h*math.copysign(abs(sin(a))**.73,sin(a)),zz))
            path('Diagonal PDK casting web',pts,.006,alloy)
        # Rear wheel half shafts attach at the transaxle differential next to the engine.
        shaft('Differential output boss',(side*.159,.397,-1.13),(side*.249,.397,-1.13),.071,graphite)
        shaft('Axle flange',(side*.24,.397,-1.13),(side*.264,.397,-1.13),.078,alloy)
        for i in range(6):
            a=2*pi*i/6
            hexbolt('Output flange bolt',(side*.265,.397+.057*cos(a),-1.13+.057*sin(a)),(side,0,0))
    for k in range(12):
        a=2*pi*k/12
        hexbolt('Motor bellhousing bolt',(.197*cos(a),.414+.170*sin(a),-1.322),(0,0,-1))
    section('PDK lower oil pan',[
        (-1.106,0,.268,.105,.031),(-1.071,0,.250,.130,.037),
        (-.755,0,.258,.113,.034),(-.710,0,.274,.091,.026)],graphite,4)
    shaft('PDK electric terminal feed',(.061,.559,-1.270),(.066,.606,-1.260),.021,orange)
    bend('High-voltage line from gearbox',[(.065,.606,-1.259),(.079,.651,-1.270),(.091,.683,-1.342),(.065,.704,-1.488)],.010,orange)

    # The published motor dimensions are 286 mm diameter x 55 mm in axial length.
    # This independent solid occupies that space inside the PDK. Its central
    # opening is illustrative; the factory cutaway does not dimension the shaft.
    parent='PDK_Motor_Envelope'
    rings=[]
    for z,r in [(-1.297,.040),(-1.297,.139),(-1.294,.143),(-1.245,.143),(-1.242,.139),(-1.242,.040),(-1.297,.040)]:
        rings.append([(r*cos(2*pi*i/72),.414+r*sin(2*pi*i/72),z) for i in range(72)])
    rings_mesh('Permanent magnet motor dimensional envelope',rings,graphite,False)
