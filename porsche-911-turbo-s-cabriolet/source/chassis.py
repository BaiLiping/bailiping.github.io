"""Visible driveline and suspension reconstruction in metres (Y up, +Z forward).

The Porsche Turbo S transparent-body illustration (Porsche AG, image 323526)
locates the battery and driveline. The 992.2 Cabriolet technical data identifies
the front strut and five-link rear axle. Cast shapes, pivot locations and cable
routing are visual estimates for the assembly illustration.
"""

def build_chassis():
    global parent
    aluminum = mat('Suspension • cast aluminium', (.31, .34, .35), .65, .39)
    machined = mat('Suspension • machined lands', (.49, .50, .49), .73, .29)
    dark = mat('Axle carrier graphite', (.057, .063, .07), .52, .43)
    damper = mat('Damper satin steel', (.095, .094, .080), .58, .34)
    orange = mat('Battery HV connector', (.88, .17, .012), .05, .39)

    def ring_solid(name, rings, material):
        count = len(rings[0])
        faces = [tuple(reversed(range(count)))]
        for row in range(len(rings) - 1):
            for j in range(count):
                k = (j + 1) % count
                faces.append((row*count+j, row*count+k, (row+1)*count+k, (row+1)*count+j))
        faces.append(tuple(range((len(rings)-1)*count, len(rings)*count)))
        obj = mesh(name, [p for ring in rings for p in ring], faces, material)
        for cap in (obj.data.polygons[0], obj.data.polygons[-1]):
            cap.use_smooth = False
        return obj

    def casting(name, sections, material, power=3.4):
        rings = [[(x, y, z) for x, y in roundrect(cx, cy, w, h, power, 28)]
                 for z, cx, cy, w, h in sections]
        return ring_solid(name, rings, material)

    def sleeve(name, a, b, stations, material, sides=20):
        first, last = Vector(a), Vector(b)
        axis = (last-first).normalized()
        reference = Vector((0, 0, 1)) if abs(axis.z) < .9 else Vector((1, 0, 0))
        u = axis.cross(reference).normalized()
        v = axis.cross(u)
        rings = [[first.lerp(last, t) + radius*(u*cos(2*pi*i/sides)+v*sin(2*pi*i/sides))
                  for i in range(sides)] for t, radius in stations]
        return ring_solid(name, rings, material)

    def pivot(c):
        p = Vector(c)
        half = Vector((0, 0, .024))
        sleeve('Suspension pivot bushing', p-half, p+half,
               [(0,.028),(.09,.032),(.90,.032),(1,.028)], dark)
        tube('Suspension pivot fastener', [p+half, p+1.28*half], .016, machined, 6)

    def link(start, end):
        sleeve('Forged suspension link', start, end,
               [(0,.022),(.06,.026),(.18,.014),(.78,.011),(.92,.023),(1,.022)], aluminum)
        pivot(start)
        pivot(end)

    def half_shaft(start, end):
        a, b = Vector(start), Vector(end)
        shaft('Drive half shaft', a, b, .014, dark)
        for t, direction in [(0, 1), (1, -1)]:
            p, q = a.lerp(b,t), a.lerp(b,t + direction*.19)
            sleeve('CV joint housing', p, q,
                   [(0,.047),(.16,.049),(.30,.047),(.40,.039),(1,.022)], dark)
            stations = []
            for i, radius in enumerate([.047,.044,.040,.034]):
                stations.extend([(i*.21+.10, radius*.79),(i*.21+.17, radius),(i*.21+.255,radius*.87)])
            sleeve('Articulated CV boot', p, q, stations, seal)
            sleeve('CV boot clamp', p, q, [(.06,.047),(.10,.047)], machined)

    def spring_strut(a, b):
        lo, hi = Vector(a), Vector(b)
        up = (hi-lo).normalized()
        across = up.cross(Vector((0,0,1))).normalized()
        fore = up.cross(across)
        shaft('Damper piston rod', lo, hi, .012, machined)
        sleeve('Damper body', lo, hi, [(0,.025),(.03,.029),(.58,.029),(.61,.024)], damper)
        sleeve('Lower spring seat', lo, hi, [(.38,.029),(.42,.064),(.44,.065)], dark)
        sleeve('Strut top mount', lo, hi, [(.88,.046),(.92,.065),(.97,.065),(1,.030)], aluminum)
        helix = []
        for i in range(133):
            t = i/132
            angle = 11*pi*t
            helix.append(lo.lerp(hi,.442+.455*t) + (.048+.006*sin(pi*t))*(across*cos(angle)+fore*sin(angle)))
        tube('Suspension coil spring', helix, .0065, dark, 8)
        sleeve('Strut dust bellows', lo, hi, [(i/100, .021+(.003 if i%2 else 0)) for i in range(62,83)], seal)

    for front in (True, False):
        parent = 'Front_Axle' if front else 'Rear_Axle'
        z, y = (1.225,.34325) if front else (-1.225,.3642)
        # Twin rails leave clearance for the transaxle in the rear cradle.
        for sign in (-1, 1):
            casting('Axle carrier side casting',
                    [(z+dz, sign*x, cy, w, h) for dz,x,cy,w,h in [
                        (-.30,.38,.268,.045,.026),(-.25,.38,.264,.061,.038),
                        (-.13,.35,.235,.043,.043),(.06,.33,.233,.042,.041),
                        (.20,.38,.245,.055,.036),(.26,.42,.268,.051,.026)]], aluminum)
        for dz in (-.232,.189):
            shaft('Axle carrier crossmember', (-.39,.234,z+dz), (.39,.234,z+dz), .031, aluminum)
        if front:
            casting('Water-cooled front differential',
                    [(z+dz,0,cy,w,h) for dz,cy,w,h in [
                        (-.22,.337,.052,.050),(-.16,.343,.068,.060),(-.08,.352,.095,.073),
                        (-.03,.356,.12,.092),(.06,.352,.124,.088),(.134,.356,.073,.060)]], dark)
            for dz in (-.10,-.053,-.028,.058):
                tube('Differential cooling rib', [(x, yy, z+dz) for x,yy in roundrect(0,.351,.108,.080,3,32)],
                     .0038, aluminum, 8, True)
            shaft('Steering rack housing', (-.45,.314,z-.18), (.45,.314,z-.18), .026, aluminum)
        for sign in (-1,1):
            def local(x, h, dz): return (sign*x,h,z+dz)
            hub = .79 if front else .794
            shaft('Wheel bearing carrier', local(hub-.095,y,0), local(hub+.005,y,0), .055, dark)
            sleeve('Wheel mounting flange', local(hub-.02,y,0), local(hub+.009,y,0),
                   [(0,.053),(.23,.074),(.73,.074),(1,.047)], aluminum)
            casting('Hub upright', [(z+dz,sign*.706,cy,w,h) for dz,cy,w,h in [
                (-.040,y+.018,.025,.070),(-.014,y+.025,.029,.117),(.018,y+.018,.033,.105),
                (.053,y+.005,.021,.059)]], aluminum)
            half_shaft(local(.123,y+.007,0) if front else local(.265,.397,.095),local(hub-.043,y,0))
            if front:
                link(local(.36,.238,.175),local(.705,.252,.017))
                link(local(.375,.261,-.244),local(.682,.254,.017))
                link(local(.43,.314,-.18),local(.710,.325,-.095))
                spring_strut(local(.706,y+.031,-.018),local(.543,.736,-.055))
            else:
                for inner, outer in [
                    ((.35,.246,.165),(.701,.257,.050)), ((.382,.261,-.239),(.706,.267,-.048)),
                    ((.350,.438,-.178),(.700,.443,-.038)), ((.348,.452,.195),(.695,.432,.048)),
                    ((.397,.328,.252),(.694,.340,.098))]:
                    link(local(*inner),local(*outer))
                spring_strut(local(.678,.252,-.082),local(.559,.737,-.150))
        points = [(-.672,.278,z-.048),(-.60,.264,z-.201),(-.43,.274,z-.205),
                  (.43,.274,z-.205),(.60,.264,z-.201),(.672,.278,z-.048)]
        tube('Anti-roll bar', points, .014, dark, 14)
    parent = 'Propeller_Shaft'
    lo, hi = (0,.394,-.534),(0,.337,1.011)
    sleeve('Front drive propeller shaft',lo,hi,
           [(0,.035),(.027,.035),(.044,.025),(.966,.025),(.981,.035),(1,.035)], dark)
    for t in (.02,.49,.977):
        sleeve('Driveshaft coupling',lo,hi,[(t-.012,.029),(t-.01,.042),(t+.009,.042),(t+.012,.029)], aluminum)

    parent = 'HV_Battery'
    def pack_profile(name, heights, material):
        rings = [[(x,h,z) for x,z in roundrect(0,1.045,w,d,8,48)] for h,w,d in heights]
        return ring_solid(name, rings, material)
    pack_profile('400 V battery housing',[(.536,.223,.087),(.552,.240,.10),(.711,.240,.10),(.718,.244,.103)], aluminum)
    pack_profile('Pack lid seal',[(.717,.245,.104),(.725,.245,.104)], seal)
    pack_profile('Battery enclosure lid',[(.726,.247,.104),(.735,.246,.105),(.744,.233,.097)], machined)
    for x in (-.208,-.130,.130,.208):
        shaft('Battery case lid bolt',(x,.740,1.129),(x,.748,1.129),.006,dark)
        shaft('Battery case lid bolt',(x,.740,.962),(x,.748,.962),.006,dark)
    for side in (-1,1):
        x = side*.253
        casting('HV battery mounting foot',
                [(z, x,.548,.028,.010) for z in (.991,1.104)], dark)
        shaft('Battery tray isolator',(x,.528,1.043),(x,.542,1.043),.014,seal)
    for i in range(2):
        x = .118 + i*.049
        sleeve('HV connector backshell',(x,.635,.949),(x,.635,.893),
               [(0,.020),(.14,.024),(.68,.024),(.72,.019),(1,.018)], orange)
        tube('Battery cable tail',[(x,.635,.896),(x,.632,.876),(x+.009,.605,.861),
                                  (x+.028,.578,.867)],.010,orange,12)
    parent = 'Bodywork'
