# engine_core.py: the mechanism.  Crank train, crankcase, cylinders, heads, valve gear and timing drive.
#
# Part list, in build order (what each rides on in brackets):
#   Crankshaft [crank]            7 mains, 6 pins, webs with counterweights, timing gear, distributor gear, nose
#   Flywheel, Clutch, Crank_Pulley [crank]
#   Rod_1..6 [rod_n]              one mesh, six instances
#   Piston_1..6 [piston_n]        one mesh, six instances
#   Case_L, Case_R [static]       built from plates: walls, bearing webs, end walls, spigot holes
#   Cylinder_1..6 [static]        finned barrel, one mesh
#   Head_1..6 [static]            domed chamber, ports, fins, one mesh
#   IntakeValve_n, ExhaustValve_n [valve]   ValveSpring_* [scaled with the valve]
#   Rocker_In_n, Rocker_Ex_n [rocker]       RockerShaft_* [static]
#   CamHousing_L/R, ValveCover_Upper/Lower_L/R [static]
#   Camshaft_L/R [cam]  CamSprocket_L/R [cam]
#   Intermediate_Shaft [spins], Oil_Pump, Oil_Pump_Shaft
#   Chain_L/R links [moved by the viewer along the exported path], Idler_L/R, IdlerArm_L/R, Tensioner_L/R,
#   ChainRamp_*, ChainHousing_L/R, ChainCover_L/R
import math
from mathutils import Vector as V, Matrix, Quaternion
from kit import Geo, frame, basis, circle, rounded_rect, gear_outline, sprocket_outline, TAU
import spec as S

QZ180 = Quaternion((0, 0, 1), math.pi)


def bank_q(bank):
    """Rotation that carries the head frame of a bank into the world.  Takes the bank sign.  Returns a Quaternion."""
    return Quaternion() if bank > 0 else QZ180.copy()


def bank_m(bank, x=0.0):
    """Placement of a bank's head frame.  Takes the bank sign and a position along X.  Returns a 4x4 Matrix."""
    m = Matrix.Translation((x, 0, 0))
    return m if bank > 0 else m @ Matrix.Rotation(math.pi, 4, 'Z')


def XF(x0=0.0):
    """Frame for a (u, w) cross-section extruded along +x from x0.  Takes x0.  Returns a 4x4 Matrix."""
    return frame((x0, 0, 0), z=(1, 0, 0), x=(0, 1, 0))


def UF(u0=0.0, x=0.0, w=0.0):
    """Frame for a lathe about the cylinder axis, with local Z along +u starting at u0.  Returns a 4x4 Matrix."""
    return frame((x, u0, w), z=(0, 1, 0), x=(1, 0, 0))


def arc(c, r, a0, a1, n):
    """Points on an arc.  Takes the centre, radius, start and end angle in degrees and the segment count.
    Returns [(u, w), ...] including both ends."""
    return [(c[0] + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)), c[1] + r * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]


def hull_of(circles, n=40):
    """Convex outline round a set of circles.  Takes [((u, w), r), ...] and samples per circle.  Returns
    [(u, w), ...] counter-clockwise."""
    pts = []
    for (cu, cw), r in circles:
        for i in range(n):
            t = TAU * i / n
            pts.append((round(cu + r * math.cos(t), 5), round(cw + r * math.sin(t), 5)))
    return S._hull(pts)


def clip_min_u(poly, umin):
    """Clip a convex polygon to u >= umin.  Takes [(u, w), ...] and the limit.  Returns the clipped polygon."""
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ina, inb = a[0] >= umin, b[0] >= umin
        if ina:
            out.append(a)
        if ina != inb:
            t = (umin - a[0]) / (b[0] - a[0])
            out.append((umin, a[1] + (b[1] - a[1]) * t))
    return out


# ---------------------------------------------------------------------------------------------------------------
# crank train
# ---------------------------------------------------------------------------------------------------------------
def build_crank(E):
    """Crankshaft with its gears, plus flywheel, clutch and pulley riding on it.  Takes the Engine.  Returns
    nothing."""
    g = Geo()
    rm, rp = S.MAIN_D / 2, S.PIN_J_D / 2
    for i, x in enumerate(S.MAIN_X):
        w = 28.0 if i == 0 else S.MAIN_W
        g.cyl((x - w / 2, 0, 0), (x + w / 2, 0, 0), rm, n=32)
    for c in S.CYLS:
        a = math.radians(c['alpha'])
        py, pz = S.R_CRANK * math.cos(a), S.R_CRANK * math.sin(a)
        g.cyl((c['x'] - S.PIN_J_W / 2, py, pz), (c['x'] + S.PIN_J_W / 2, py, pz), rp, n=28)
        # two webs, one each side of the pin, each with a counterweight lobe opposite the pin
        cw = [(-math.cos(a + math.radians(t)) * 50.0, -math.sin(a + math.radians(t)) * 50.0) for t in range(-52, 53, 8)]
        outline = S._hull([(round(p[0], 4), round(p[1], 4)) for p in
                           [(34.0 * math.cos(TAU * k / 28), 34.0 * math.sin(TAU * k / 28)) for k in range(28)] +
                           [(py + 31.0 * math.cos(TAU * k / 28), pz + 31.0 * math.sin(TAU * k / 28)) for k in range(28)] + cw])
        for sgn in (-1, 1):
            x0 = c['x'] + sgn * (S.PIN_J_W / 2 + S.WEB_T / 2) - S.WEB_T / 2
            g.at(XF(x0)).prism(outline, 0, S.WEB_T)
        g.at()
    # flywheel flange and rear main seal journal
    g.cyl((S.MAIN_X[0] + 14, 0, 0), (S.X_FLANGE - 14, 0, 0), 37.0, n=32)
    g.cyl((S.X_FLANGE - 14, 0, 0), (S.X_FLANGE, 0, 0), 46.0, n=36)
    # timing gear: 28 teeth, module from the 84 mm centre distance
    mod = 2.0 * S.IMS_DROP / (S.Z_CRANK_GEAR + S.Z_IMS_GEAR)
    g.at(XF(S.X_TIMING_GEAR - 8)).prism(gear_outline(S.Z_CRANK_GEAR, mod * S.Z_CRANK_GEAR / 2, mod, mod * 1.2), 0, 16)
    g.at(XF(S.X_DIST_GEAR - 7)).prism(gear_outline(22, 21.0, 1.3, 1.6), 0, 14)
    g.at()
    g.cyl((S.MAIN_X[6] - 11, 0, 0), (S.X_DIST_GEAR - 7, 0, 0), 24.0, n=28)
    g.cyl((S.X_DIST_GEAR - 7, 0, 0), (S.X_MAIN8 - 12, 0, 0), 20.0, n=28)      # nose journal
    g.cyl((S.X_MAIN8 - 12, 0, 0), (S.X_PULLEY - 12, 0, 0), 15.0, n=24)
    crank = E.add('Crankshaft', g, 'steel', 'crank', 'Crankshaft', cut=False)

    # flywheel with its ring gear, bolted to the flange with nine bolts
    g = Geo()
    g.at(frame((S.X_FLANGE, 0, 0), z=(1, 0, 0), x=(0, 1, 0)))
    g.lathe([(0, 0), (126, 0), (126, 9), (132, 9), (132, 22), (118, 22), (118, 12), (52, 12), (52, 20), (0, 20)], n=64)
    g.prism(gear_outline(135, 138.6, 1.7, 2.3), 4, 16, holes=[circle(126.0, 64)])
    for k in range(9):
        a = TAU * k / 9
        g.bolt((38 * math.cos(a), 38 * math.sin(a), 20), (0, 0, 1), 8.0, washer=False)
    E.add('Flywheel', g, 'steel_dark', 'crank', 'Flywheel and ring gear', cut=False, ex=(140, 0, 0), parent=crank)
    g = Geo()
    g.at(frame((S.X_FLANGE + 22, 0, 0), z=(1, 0, 0), x=(0, 1, 0)))
    g.lathe([(122, 0), (124, 0), (124, 14), (104, 34), (58, 40), (58, 36), (100, 30), (119, 12), (122, 12)], n=56, closed=True)
    for k in range(18):         # diaphragm spring fingers
        a = TAU * k / 18
        g.at(frame((S.X_FLANGE + 22, 0, 0), z=(1, 0, 0), x=(0, 1, 0)) @ Matrix.Rotation(a, 4, 'Z'))
        g.box((42, 0, 36), (36, 9, 2.4))
    g.at(frame((S.X_FLANGE + 22, 0, 0), z=(1, 0, 0), x=(0, 1, 0)))
    for k in range(9):
        a = TAU * (k + 0.5) / 9
        g.bolt((116 * math.cos(a), 116 * math.sin(a), 13), (0, 0, 1), 8.0)
    E.add('Clutch', g, 'steel_dark', 'crank', 'Clutch pressure plate', cut=False, ex=(230, 0, 0), parent=crank)

    # crank pulley with the Z1 timing notch
    g = Geo()
    R = S.CRANK_PULLEY_D / 2
    g.at(frame((S.X_PULLEY, 0, 0), z=(-1, 0, 0), x=(0, 1, 0)))
    g.lathe([(0, -9), (R, -9), (R, -5.5), (R - 9, -1.5), (R - 9, 1.5), (R, 5.5), (R, 9), (24, 9), (22, 16), (0, 16)], n=56)
    g.box((R - 3, 0, 9.5), (6, 2.2, 1.2))
    g.bolt((0, 0, 16), (0, 0, 1), 12.0)
    E.add('Crank_Pulley', g, 'alu_mach', 'crank', 'Crank pulley (Z1 mark)', cut=False, ex=(-90, 0, 0), parent=crank)
    E.anim['Crankshaft'] = lambda th: (None, Quaternion((1, 0, 0), math.radians(th)), None)


def build_rods_pistons(E):
    """Connecting rods and pistons.  Takes the Engine.  Returns nothing."""
    # rod: local origin at the big-end centre, +Y toward the small end, pin axes along X
    g = Geo()
    hw = S.PIN_J_W / 2 - 0.1
    g.tube((-hw, 0, 0), (hw, 0, 0), 35.5, S.ROD_BIG_D / 2, n=36)
    g.tube((-12, S.ROD, 0), (12, S.ROD, 0), 17.5, S.PIN_D / 2 + 1.5, n=28)
    beam = [(-8, -3), (-8, -8.5), (8, -8.5), (8, -3), (3, -3), (3, 3), (8, 3), (8, 8.5), (-8, 8.5), (-8, 3), (-3, 3), (-3, -3)]
    n = 6
    path = [(0, 33 + (S.ROD - 49) * i / n, 0) for i in range(n + 1)]
    g.sweep(path, [(p[1] * 1.9, p[0]) for p in beam], scales=[(1.0 - 0.36 * i / n, 1.0) for i in range(n + 1)], up=(0, 0, 1))
    for sz in (-1, 1):
        g.box((0, 2, sz * 39.5), (hw * 2, 34, 13), bevel=1.5)
        g.cyl((0, -15, sz * 40.5), (0, 19, sz * 40.5), 4.8, n=10)
        g.hexa((0, 19, sz * 40.5), (0, 1, 0), 14.3, 8)
        g.cyl((0, -15, sz * 40.5), (0, -19, sz * 40.5), 7.5, n=12)
    rod = None
    for c in S.CYLS:
        name = 'Rod_%d' % c['n']
        if rod is None:
            rod = E.add(name, g, 'steel', 'rods', 'Connecting rod %d' % c['n'], cut=False)
        else:
            E.inst(rod, name, label='Connecting rod %d' % c['n'])
        E.anim[name] = (lambda th, c=c: _rod_pose(c, th))

    # piston: local origin at the pin centre, +Y toward the crown
    g = Geo()
    R = S.BORE / 2 - 0.25
    top = S.COMP_H
    prof = [(radius, top + height) for radius, height in S.CROWN_PROFILE] + [(R, top), (R, top - 5.2)]
    y = top - 5.2
    for wdt, gap in ((1.2, 4.2), (1.2, 4.2), (3.0, 4.0)):       # ring grooves 1.2 / 1.2 / 3.0 mm [S]
        prof += [(R - 3.2, y), (R - 3.2, y - wdt), (R, y - wdt), (R, y - wdt - gap)]
        y -= wdt + gap
    prof += [(R, -30), (R - 4.2, -30), (R - 4.2, 9), (R - 16, top - 15), (0, top - 15)]
    g.at(frame((0, 0, 0), z=(0, 1, 0), x=(1, 0, 0))).lathe(prof, n=48, closed=True)
    g.at()
    g.tube((-39, 0, 0), (39, 0, 0), S.PIN_D / 2, 7.0, n=24)
    for sx in (-1, 1):
        g.tube((sx * 17, 0, 0), (sx * 42.5, 0, 0), 17.0, S.PIN_D / 2, n=24)
    pis = None
    for c in S.CYLS:
        name = 'Piston_%d' % c['n']
        if pis is None:
            pis = E.add(name, g, 'alu_piston', 'pistons', 'Piston %d' % c['n'], cut=False)
        else:
            E.inst(pis, name, label='Piston %d' % c['n'])
        E.anim[name] = (lambda th, c=c: (V((c['x'], c['bank'] * S.piston_s(S.beta(c, th)), 0)), bank_q(c['bank']), None))


def _rod_pose(c, th):
    """Pose of one rod.  Takes a CYLS row and the crank angle.  Returns (location, rotation, scale)."""
    a = math.radians(c['alpha'] + th)
    py, pz = S.R_CRANK * math.cos(a), S.R_CRANK * math.sin(a)
    s = S.piston_s(S.beta(c, th))
    swing = math.atan2(-pz, abs(c['bank'] * s - py))
    psi = swing if c['bank'] > 0 else math.pi - swing
    return V((c['x'], py, pz)), Quaternion((1, 0, 0), psi), None


# ---------------------------------------------------------------------------------------------------------------
# crankcase
# ---------------------------------------------------------------------------------------------------------------
X_FRONT, X_BODY, X_REAR = 205.0, -168.0, -274.0
TOP_OUT = [(0, 92), (48, 92), (112.5, 64)]
TOP_IN = [(100, 62), (46, 84), (0, 84)]
BOT_OUT = [(112.5, -64), (80, -106), (63, -160), (0, -160)]
BOT_IN = [(0, -152), (55, -152), (72, -100), (100, -62)]
REAR_TOP_OUT = [(0, 46), (62, 46), (112.5, 30)]
REAR_TOP_IN = [(100, 24), (58, 38), (0, 38)]
CHAIN_X = {+1: (-243.0, -150.5), -1: (-270.0, -209.5)}      # each chain housing's extent along X (rear, front)


def _case_half(E, bank):
    """One crankcase half, assembled from plates so that it is hollow and shows its webs when cut.  Takes the
    Engine and the bank sign.  Returns the object."""
    g = Geo()
    sx = float(bank)        # head frame x = sx * world X

    def span(xa, xb):
        """World X range to an extrusion start and length in the bank's frame."""
        lo, hi = sorted((sx * xa, sx * xb))
        return lo, hi - lo

    x0, L = span(X_BODY, X_FRONT)
    g.at(XF(x0)).prism(TOP_OUT + TOP_IN, 0, L)
    g.at(XF(x0)).prism([(112.5, -64)] + BOT_OUT[1:] + BOT_IN, 0, L)
    x0, L = span(X_REAR, X_BODY)
    g.at(XF(x0)).prism(REAR_TOP_OUT + REAR_TOP_IN, 0, L)
    g.at(XF(x0)).prism([(112.5, -64)] + BOT_OUT[1:] + BOT_IN, 0, L)
    # side wall with the three cylinder spigot bores; plate lies in (w, x) and is 12.5 thick along u
    holes = [[(p[0], sx * c['x'] + p[1]) for p in circle(52.6, 40)] for c in S.CYLS if c['bank'] == bank]
    xa, xb = sorted((sx * X_BODY, sx * X_FRONT))
    side = basis((0, 100, 0), (0, 0, 1), (1, 0, 0))
    g.at(side).prism([(-64, xa), (64, xa), (64, xb), (-64, xb)], 0, 12.5, holes=holes)
    for c in S.CYLS:
        if c['bank'] == bank:
            g.at(UF(112.5 - 4, sx * c['x'])).lathe([(52.6, 0), (59.5, 0), (59.5, 4), (52.6, 4)], n=40, closed=True)
    # rear section side wall, only where the chain housing does not open into the case
    ca, cb = CHAIN_X[bank]
    for wa, wb in ((X_REAR, min(ca, X_BODY)), (max(cb, X_REAR), X_BODY)):
        if wb - wa > 1.0:
            a, b = sorted((sx * wa, sx * wb))
            g.at(side).prism([(-64, a), (30, a), (30, b), (-64, b)], 0, 12.5)
    # main bearing webs: plates with a saddle for the crank and a passage for the oil pump shaft
    web = [(0, 84), (46, 84), (58, 78), (58, -128), (55, -152), (0, -152), (0, -92)] + arc((0, -84), 8, -90, 90, 8)[1:-1] + \
          [(0, -76), (0, -32.5)] + arc((0, 0), 32.5, -90, 90, 16)[1:-1] + [(0, 32.5)]
    for i, xm in enumerate(S.MAIN_X):
        x0, L = span(xm - 9, xm + 9)
        if xm > X_BODY:
            g.at(XF(x0)).prism(web, 0, L)
        else:               # main 7 sits in the lower rear section
            low = [(0, 38), (58, 38), (58, -128), (55, -152), (0, -152), (0, -138)] + arc((0, -84), 54, -90, 90, 16)[1:-1] + \
                  [(0, -30)] + arc((0, 0), 32.5, -90, 90, 16)[1:-1] + [(0, 32.5)]
            g.at(XF(x0)).prism(low, 0, L)
    # end walls
    front = TOP_OUT + BOT_OUT + [(0, -46)] + arc((0, 0), 46, -90, 90, 16)[1:-1] + [(0, 46)]
    x0, L = span(X_FRONT - 12, X_FRONT)
    g.at(XF(x0)).prism(front, 0, L)
    rear = REAR_TOP_OUT + BOT_OUT + [(0, -104)] + arc((0, -84), 20, -90, 90, 10)[1:-1] + [(0, -64), (0, -22)] + \
        arc((0, 0), 22, -90, 90, 12)[1:-1] + [(0, 22)]
    x0, L = span(X_REAR, X_REAR + 12)
    g.at(XF(x0)).prism(rear, 0, L)
    step = [(0, 92), (48, 92), (112.5, 64), (112.5, 30), (62, 46), (0, 46)]     # where the tall body steps down
    x0, L = span(X_BODY - 4, X_BODY + 4)
    g.at(XF(x0)).prism(step, 0, L)
    # joint flanges along the top and bottom of the split line, and a boss ring round the nose bearing
    xa, L = span(X_BODY, X_FRONT)
    g.at(XF(xa)).prism([(0, 92), (9, 92), (9, 104), (0, 104)], 0, L)
    xa, L = span(X_REAR, X_FRONT)
    g.at(XF(xa)).prism([(0, -160), (0, -172), (9, -172), (9, -160)], 0, L)
    g.at()
    name = 'Case_L' if bank > 0 else 'Case_R'
    return E.add(name, g, 'alu_cast', 'crankcase', 'Crankcase half, %s' % ('left' if bank > 0 else 'right'),
                 ex=(0, bank * 40, -150), rot=bank_q(bank), sharp=30)


def build_case(E):
    """Both crankcase halves with their joint nuts, through-bolts and the main bearing shells.  Takes the
    Engine.  Returns nothing."""
    for bank in (+1, -1):
        _case_half(E, bank)
    # perimeter nuts on the split line: M8 on studs, top and bottom
    g = Geo(); g.nut((0, 0, 0), (0, 1, 0), 8.0, stud=5)
    src = None
    k = 0
    for x in range(-150, 201, 44):
        for z in (98.0, -166.0):
            src = E.fastener(src, 'Case_Nut_%02d' % k, g, (x, 9.0, z), label='Case joint nut M8', group='crankcase', ex=(0, 70, -150 if z < 0 else 0)); k += 1
    # eleven M10 through-bolts [S]: heads on the right, nuts on the left, across the bearing webs
    gh = Geo(); gh.bolt((0, 0, 0), (0, -1, 0), 10.0)
    gn = Geo(); gn.nut((0, 0, 0), (0, 1, 0), 10.0, stud=6)
    hs = ns = None
    spots = [(xm, 74.0) for xm in S.MAIN_X[:6]] + [(xm, -74.0) for xm in S.MAIN_X[1:6]]
    for i, (x, z) in enumerate(spots):
        y = 112.5 - (abs(z) - 64) * 0.0
        hs = E.fastener(hs, 'Case_ThroughBolt_%02d' % i, gh, (x, -112.5, z * 0.82), label='Case through-bolt M10', group='crankcase', ex=(0, -90, 0))
        ns = E.fastener(ns, 'Case_ThroughNut_%02d' % i, gn, (x, 112.5, z * 0.82), label='Case through-bolt nut M10', group='crankcase', ex=(0, 90, 0))
    # main bearing shells, 1 to 7 on the 60 mm journals and the nose bearing
    src = None
    for i, xm in enumerate(S.MAIN_X):
        g = Geo(); g.tube((-9, 0, 0), (9, 0, 0), 32.5, S.MAIN_D / 2 + 0.05, n=32)
        src = E.fastener(src, 'MainBearing_%d' % (i + 1), g, (xm, 0, 0), mat='bearing', label='Main bearing %d' % (i + 1), group='crank', cut=True)
    g = Geo(); g.tube((-12, 0, 0), (12, 0, 0), 22.0, 20.05, n=28)
    E.add('MainBearing_8', g, 'bearing', 'crank', 'Nose bearing (main 8)', loc=(S.X_MAIN8, 0, 0))


# ---------------------------------------------------------------------------------------------------------------
# cylinders and heads
# ---------------------------------------------------------------------------------------------------------------
def _fin_shape(r, a):
    """Outline of the cooling fins: round near the barrel, flattened where neighbouring cylinders sit 118 mm
    apart and taller above and below.  Takes a nominal radius and an angle.  Returns (x, y) in the lathe frame."""
    c, s = math.cos(a), math.sin(a)
    if r <= 57.0:
        return r * c, r * s
    t = min(1.0, (r - 57.0) / 15.0)
    e = 2.0 / 3.4
    sx = 57.0 + (57.5 - 57.0) * t
    sz = 57.0 + (75.0 - 57.0) * t
    ex = sx * (abs(c) ** e) * (1 if c >= 0 else -1)
    ez = sz * (abs(s) ** e) * (1 if s >= 0 else -1)
    return (r * c) * (1 - t) + ex * t, (r * s) * (1 - t) + ez * t


def build_cylinders(E):
    """Six finned Nikasil barrels.  Takes the Engine.  Returns nothing."""
    g = Geo()
    u0 = S.DECK
    prof = [(S.BORE / 2, -20.5), (52.4, -20.5), (52.4, 0), (59.0, 0), (59.0, 6.0), (54.6, 6.0)]
    u = 8.0
    while u < S.FIN_LIMIT:             # turbo barrels carry their fins on the lower part [S]
        prof += [(54.6, u), (72.0, u), (72.0, u + 2.2), (54.6, u + 2.2)]
        u += 5.6
    prof += [(54.6, S.FIN_LIMIT + 2.5), (60.5, S.FIN_LIMIT + 2.5), (60.5, S.CYL_H), (S.BORE / 2, S.CYL_H)]
    g.at(UF(u0)).lathe(prof, n=48, closed=True, shape=_fin_shape)
    src = None
    for c in S.CYLS:
        name = 'Cylinder_%d' % c['n']
        kw = dict(loc=(c['x'], 0, 0), rot=bank_q(c['bank']), ex=(0, c['bank'] * 115, 0), label='Cylinder %d' % c['n'])
        if src is None:
            src = E.add(name, g, 'alu_cast', 'cylinders', sharp=35, **kw)
        else:
            E.inst(src, name, **kw)
    # 24 head studs [S], four per cylinder, from the case through the fins into the head
    g = Geo(); g.cyl((0, S.DECK - 6, 0), (0, S.HEAD_FACE + S.HEAD_H + 6, 0), 5.0, n=10)
    g.hexa((0, S.HEAD_FACE + S.HEAD_H, 0), (0, 1, 0), 15.0, 11.0)
    g.cyl((0, S.HEAD_FACE + S.HEAD_H - 1.6, 0), (0, S.HEAD_FACE + S.HEAD_H, 0), 10.5, n=14)
    src = None
    for c in S.CYLS:
        for k, (dx, dz) in enumerate(((44, 44), (-44, 44), (44, -44), (-44, -44))):
            p = bank_m(c['bank'], c['x']) @ V((dx, 0, dz))
            src = E.fastener(src, 'HeadStud_%d%d' % (c['n'], k + 1), g, p, rot=bank_q(c['bank']), mat='steel_dark', label='Head stud', group='cylinders', ex=(0, c['bank'] * 60, 0))


def build_heads(E):
    """Six separate heads: domed chamber, fins, intake port on top, exhaust port underneath, plug boss.  Takes
    the Engine.  Returns nothing."""
    g = Geo()
    hf, top = S.HEAD_FACE, S.HEAD_FACE + S.HEAD_H
    # chamber end: a ring with a domed roof, so a cut shows the chamber
    g.at(UF(hf)).lathe([(S.BORE / 2, 0), (61, 0), (61, 30), (0, 30), (0, 24), (26, 21.5), (42, 12)], n=48, closed=True)
    # upper body up to the cam housing face
    g.at(XF(-52)).prism([(hf + 30, -64), (hf + 46, -84), (top, -84), (top, 84), (hf + 46, 84), (hf + 30, 64)], 0, 104)
    # fins round the chamber end, square to the cylinder axis like the barrel's
    fin = [(p[1], p[0]) for p in rounded_rect(114, 176, 16, 4)]
    for k in range(5):
        g.at(basis((0, hf + 3.5 + k * 6.0, 0), (0, 0, 1), (1, 0, 0))).prism(fin, 0, 2.3, holes=[circle(58, 40)])
    # fins along the sides of the upper body
    for sgn in (-1, 1):
        for k in range(4):
            g.at(XF(sgn * 52 - (2.2 if sgn < 0 else 0)) @ Matrix.Translation((0, 0, 0))).prism(
                [(hf + 34 + k * 12, -78), (hf + 36.4 + k * 12, -78), (hf + 36.4 + k * 12, 78), (hf + 34 + k * 12, 78)], 0, 2.2)
    g.at()
    # ports: intake stub on top, exhaust underneath, each with its flange
    g.tube((0, 236, 70), (0, 243, 100), 23.0, 16.0, n=28)
    g.at(basis((0, 243, 100), (1, 0, 0), (0, 1, 0))).prism(rounded_rect(76, 56, 12, 4), -2, 6, holes=[circle(16.0, 28)])
    g.at()
    g.tube((0, 232, -70), (0, 238, -102), 25.0, 17.5, n=28)
    g.at(basis((0, 238, -102), (1, 0, 0), (0, -1, 0))).prism(rounded_rect(80, 58, 12, 4), -2, 7, holes=[circle(17.5, 28)])
    g.at()
    # valve guides and spring seats, and the plug boss between the valves on the upper side
    for kind in ('in', 'ex'):
        v = S.VALVE[kind]
        au, aw = S.valve_axis(kind)
        s0 = v['length'] - S.SPRING_FROM_TIP - S.SPRING_H
        p = lambda s: (0, v['seat'][0] + au * s, v['seat'][1] + aw * s)
        g.cyl(p(40), p(s0 + 10), 7.6, n=14)
        g.cyl(p(s0 - 4), p(s0), 19.0, n=24)
    pd = V((0.52, 0.55, 0.66)).normalized()
    pt = V((14, hf + 22, 6))
    g.cyl(pt + pd * 16, pt + pd * 62, 14.5, n=18)
    src = None
    for c in S.CYLS:
        name = 'Head_%d' % c['n']
        kw = dict(loc=(c['x'], 0, 0), rot=bank_q(c['bank']), ex=(0, c['bank'] * 215, 0), label='Cylinder head %d' % c['n'])
        if src is None:
            src = E.add(name, g, 'alu_cast', 'heads', sharp=35, **kw)
        else:
            E.inst(src, name, **kw)
    # spark plugs: one per cylinder [S]
    g = Geo()
    g.cyl(pt + pd * 8, pt + pd * 60, 6.8, n=12)
    g.hexa(pt + pd * 60, pd, 21.0, 12)
    E.tmp_plug = (pt, pd)
    gi = Geo()
    gi.cyl(pt + pd * 72, pt + pd * 100, 5.6, n=12)
    gi.cyl(pt + pd * 100, pt + pd * 108, 3.2, n=10)
    src = srci = None
    for c in S.CYLS:
        kw = dict(loc=(c['x'], 0, 0), rot=bank_q(c['bank']), ex=(0, c['bank'] * 215, 60))
        if src is None:
            src = E.add('SparkPlug_%d' % c['n'], g, 'steel', 'ignition', label='Spark plug %d' % c['n'], **kw)
            srci = E.add('SparkPlug_Insulator_%d' % c['n'], gi, 'ceramic', 'ignition', label='Spark plug %d insulator' % c['n'], **kw)
        else:
            E.inst(src, 'SparkPlug_%d' % c['n'], label='Spark plug %d' % c['n'], **kw)
            E.inst(srci, 'SparkPlug_Insulator_%d' % c['n'], label='Spark plug %d insulator' % c['n'], **kw)


# ---------------------------------------------------------------------------------------------------------------
# valves, springs, rockers
# ---------------------------------------------------------------------------------------------------------------
ROCKER_X = {'in': 14.0, 'ex': -14.0}        # the two lobes sit side by side; the valves share the centre plane


def stem_frame(kind, s=0.0):
    """Placement with local Z along a valve stem.  Takes 'in' or 'ex' and a distance along the stem from the
    seat point.  Returns a 4x4 Matrix in the head frame."""
    v = S.VALVE[kind]
    au, aw = S.valve_axis(kind)
    return frame((0, v['seat'][0] + au * s, v['seat'][1] + aw * s), z=(0, au, aw), x=(1, 0, 0))


def build_valves(E):
    """Valves and their double springs.  Takes the Engine.  Returns nothing."""
    gs = Geo()
    gs.helix((0, 0, 0), (0, 0, S.SPRING_H), 14.6, 2.1, 5.0, n=14, m=6)
    gs.helix((0, 0, 0), (0, 0, S.SPRING_H), 9.6, 1.5, 6.5, n=12, m=5)
    spring = None
    for kind, nm, mat in (('in', 'IntakeValve', 'steel'), ('ex', 'ExhaustValve', 'steel_hot')):
        v = S.VALVE[kind]
        R, L = v['head_d'] / 2, v['length']
        rt = L - S.SPRING_FROM_TIP
        g = Geo()
        g.lathe([(0, -1.2), (R - 1.5, -1.2), (R, 0.6), (R, 2.0), (11.0, 9.0), (v['stem_d'] / 2, 20.0), (v['stem_d'] / 2, rt - 5),
                 (10.5, rt - 5), (17.0, rt - 2), (17.0, rt), (7.5, rt), (7.5, rt + 2.2), (v['stem_d'] / 2, rt + 2.2), (v['stem_d'] / 2, L), (0, L)], n=28)
        src = None
        for c in S.CYLS:
            name = '%s_%d' % (nm, c['n'])
            lab = '%s valve %d' % ('Intake' if kind == 'in' else 'Exhaust', c['n'])
            if src is None:
                src = E.add(name, g, mat, 'valves', lab, cut=False)
            else:
                E.inst(src, name, label=lab)
            E.anim[name] = (lambda th, c=c, kind=kind: _valve_pose(kind, c, th))
            sname = 'ValveSpring_%s_%d' % ('In' if kind == 'in' else 'Ex', c['n'])
            if spring is None:
                spring = E.add(sname, gs, 'steel_dark', 'valves', 'Valve springs', cut=False)
            else:
                E.inst(spring, sname, label='Valve springs')
            E.anim[sname] = (lambda th, c=c, kind=kind: _spring_pose(kind, c, th))


def _valve_pose(kind, c, th):
    """Pose of one valve.  Takes the kind, a CYLS row and the crank angle.  Returns (location, rotation, scale)."""
    m = bank_m(c['bank'], c['x']) @ stem_frame(kind, -S.valve_lift(kind, c, th))
    return m.translation.copy(), m.to_quaternion(), None


def _spring_pose(kind, c, th):
    """Pose of one spring pair, squeezed by the valve's lift.  Returns (location, rotation, scale)."""
    v = S.VALVE[kind]
    m = bank_m(c['bank'], c['x']) @ stem_frame(kind, v['length'] - S.SPRING_FROM_TIP - S.SPRING_H)
    return m.translation.copy(), m.to_quaternion(), V((1, 1, (S.SPRING_H - S.valve_lift(kind, c, th)) / S.SPRING_H))


def build_rockers(E):
    """Twelve rocker arms on their own shafts.  Takes the Engine.  Returns nothing."""
    gsh = Geo(); gsh.cyl((-21, 0, 0), (21, 0, 0), S.ROCKER_SHAFT_D / 2, n=20)
    gsh.hexa((21, 0, 0), (1, 0, 0), 8.0, 3.0)
    shaft = None
    for kind in ('in', 'ex'):
        rg = S.rocker_geom(kind)
        su, sw = rg['shaft']
        pu, pw = rg['pad'][0] - su, rg['pad'][1] - sw
        tu, tw = S.valve_tip(kind)
        au, aw = S.valve_axis(kind)
        tu, tw = tu + au * 7.0 - su, tw + aw * 7.0 - sw          # the arm passes over the stem tip
        xr = ROCKER_X[kind]
        g = Geo()
        g.tube((-9, 0, 0), (xr + (9 if xr > 0 else -9), 0, 0), 15.0, S.ROCKER_SHAFT_D / 2, n=24)
        pad_c = (pu + (pu / math.hypot(pu, pw)) * 0.0 + 4.0 * math.cos(math.radians(rg['pad_angle'])), pw + 4.0 * math.sin(math.radians(rg['pad_angle'])))
        g.at(XF(xr - 6.5)).prism(hull_of([((0, 0), 13.0), (pad_c, 5.0)]), 0, 13.0)
        g.at(XF(-6.5)).prism(hull_of([((0, 0), 13.0), ((tu, tw), 8.5)]), 0, 13.0)
        g.at()
        # adjusting screw and lock nut at the valve end
        g.cyl((0, tu - au * 7.0, tw - aw * 7.0), (0, tu + au * 14, tw + aw * 14), 3.6, n=10)
        g.hexa((0, tu + au * 8.5, tw + aw * 8.5), (0, au, aw), 12.0, 5.0)
        src = None
        for c in S.CYLS:
            name = 'Rocker_%s_%d' % ('In' if kind == 'in' else 'Ex', c['n'])
            lab = '%s rocker %d' % ('Intake' if kind == 'in' else 'Exhaust', c['n'])
            if src is None:
                src = E.add(name, g, 'steel', 'rockers', lab, cut=False)
            else:
                E.inst(src, name, label=lab)
            E.anim[name] = (lambda th, c=c, kind=kind, su=su, sw=sw: (
                bank_m(c['bank'], c['x']) @ V((0, su, sw)),
                bank_q(c['bank']) @ Quaternion((1, 0, 0), math.radians(S.rocker_angle(kind, c, th))), None))
            p = bank_m(c['bank'], c['x']) @ V((xr * 0.5, su, sw))
            shaft = E.fastener(shaft, 'RockerShaft_%s_%d' % ('In' if kind == 'in' else 'Ex', c['n']), gsh, p, rot=bank_q(c['bank']),
                               mat='steel', label='Rocker shaft', group='rockers', cut=False)


# ---------------------------------------------------------------------------------------------------------------
# cam housings, covers, camshafts
# ---------------------------------------------------------------------------------------------------------------
HOUSING_OUT = [(283, -92), (290, -104), (350, -86), (368, -46), (368, 42), (350, 84), (290, 100), (283, 92)]


def bank_span(bank):
    """Extent of a bank's cam housing along its own frame's x.  Takes the bank sign.  Returns (x_front, x_chain):
    the free end and the end that meets the chain housing."""
    xs = [bank * c['x'] for c in S.CYLS if c['bank'] == bank]
    front = max(xs) + 62.0 if bank > 0 else min(xs) - 62.0
    chain = bank * CHAIN_X[bank][1]
    return front, chain


def build_cam_housings(E):
    """Cam housings with their bearing webs, the four valve covers and the cover nuts.  Takes the Engine.
    Returns nothing."""
    nut = Geo(); nut.nut((0, 0, 0), (0, 0, 1), 8.0, stud=4)
    nsrc = None
    for bank in (+1, -1):
        tag = 'L' if bank > 0 else 'R'
        xf, xc = bank_span(bank)
        x0, x1 = min(xf, xc), max(xf, xc)
        L = x1 - x0
        cx = [bank * c['x'] for c in S.CYLS if c['bank'] == bank]
        g = Geo()
        # outboard wall and its two chamfers
        g.at(XF(x0)).prism([(360, -43), (368, -46), (368, 42), (360, 39)], 0, L)
        g.at(XF(x0)).prism([(346, -78), (350, -86), (368, -46), (360, -43)], 0, L)
        g.at(XF(x0)).prism([(360, 39), (368, 42), (350, 84), (346, 76)], 0, L)
        # inboard flange against the heads, pierced where the valve springs come through
        holes = []
        for xcyl in cx:
            for kind in ('in', 'ex'):
                v = S.VALVE[kind]
                au, aw = S.valve_axis(kind)
                wv = v['seat'][1] + (283.0 - v['seat'][0]) * aw / au
                holes.append([(wv + p[0], xcyl + p[1]) for p in circle(20.5, 24)])
        g.at(basis((0, 283, 0), (0, 0, 1), (1, 0, 0))).prism([(-92, x0), (92, x0), (92, x1), (-92, x1)], 0, 8, holes=holes)
        g.at(XF(x0)).prism([(283, 92), (291, 92), (296, 99.5), (290, 100)], 0, L)
        g.at(XF(x0)).prism([(283, -92), (290, -104), (296, -103), (291, -92)], 0, L)
        # end walls; the chain end is bored for the camshaft
        for xe, bore in ((xf, 0.0), (xc, 27.0)):
            xa = xe - 8 if xe == x1 else xe
            g.at(XF(xa)).prism(HOUSING_OUT, 0, 8, holes=[[(S.CAM_U + p[0], p[1]) for p in circle(bore, 28)]] if bore else ())
        # four cam bearing webs [S], between and outside the cylinders
        for xj in sorted(set([x + 59 for x in cx] + [min(cx) - 59])):
            g.at(XF(xj - 9)).prism([(298, -46), (360, -43), (360, 39), (298, 46)], 0, 18,
                                   holes=[[(S.CAM_U + p[0], p[1]) for p in circle(S.CAM_JOURNAL_R + 0.05, 32)]])
        g.at()
        E.add('CamHousing_' + tag, g, 'alu_cast', 'camhousing', 'Camshaft housing, %s' % ('left' if bank > 0 else 'right'),
              rot=bank_q(bank), ex=(0, bank * 330, 0), sharp=30)
        # valve covers: upper over the intake rockers, lower over the exhaust rockers
        for nm, poly, rib_n, exz in (('Upper', [(290, 100), (350, 84), (353, 91), (322, 109), (292, 109)], (0.26, 0.97), 90),
                                     ('Lower', [(290, -104), (292, -113), (322, -114), (353, -93), (350, -86)], (0.33, -0.94), -90)):
            g = Geo()
            g.at(XF(x0 + 3)).prism(poly, 0, L - 6)
            rib = [(300, 108), (344, 96), (346, 99.5), (322, 112.5), (300, 112)] if exz > 0 else \
                  [(300, -113), (300, -117), (322, -117.5), (346, -102.5), (344, -99)]
            for k in range(1, 6):       # stiffening ribs across the cover
                g.at(XF(x0 + 3 + (L - 6) * k / 6.0 - 1.5)).prism(rib, 0, 3)
            g.at()
            E.add('ValveCover_%s_%s' % (nm, tag), g, 'alu_cover', 'covers', '%s valve cover, %s' % (nm, 'left' if bank > 0 else 'right'),
                          rot=bank_q(bank), ex=(0, bank * 430, exz), sharp=30)
            # nuts along both long edges
            for k in range(4):
                xn = x0 + 14 + (L - 28) * k / 3.0
                for (u, w), d in (((293, 109 if exz > 0 else -113.5), (0, 0, 1 if exz > 0 else -1)), ((349.5, 90 if exz > 0 else -92), (rib_n[0], 0, rib_n[1]))):
                    p = bank_m(bank) @ V((xn, u, w))
                    q = bank_q(bank) @ V((0, 0, 1)).rotation_difference(V((0, d[0], d[2])).normalized())
                    nsrc = E.fastener(nsrc, 'CoverNut_%s%s_%d%d' % (nm[0], tag, k, 0 if u < 300 else 1), nut, p, rot=q,
                                      label='Valve cover nut M8', group='covers', ex=(0, bank * 470, exz * 1.3))


def build_cams(E):
    """Two camshafts with four journals and six lobes each, phased from the firing order, and their sprockets.
    Takes the Engine.  Returns nothing."""
    for bank in (+1, -1):
        tag = 'L' if bank > 0 else 'R'
        xf, xc = bank_span(bank)
        xchain = bank * S.X_CHAIN[bank]
        ends = sorted((xf - 6 if xf > xc else xf + 6, xchain + (14 if xchain > xf else -14)))
        g = Geo()
        g.cyl((ends[0], 0, 0), (ends[1], 0, 0), 13.0, n=24)
        cyls = [c for c in S.CYLS if c['bank'] == bank]
        cx = [bank * c['x'] for c in cyls]
        for xj in sorted(set([x + 59 for x in cx] + [min(cx) - 59])):
            g.cyl((xj - 10, 0, 0), (xj + 10, 0, 0), S.CAM_JOURNAL_R, n=36)
        for c in cyls:
            for kind in ('in', 'ex'):
                nose = S.lobe_nose_angle(kind, c)
                out = []
                for k in range(72):
                    a = 5.0 * k
                    r = S.lobe_radius(kind, a - nose)
                    out.append((r * math.cos(math.radians(a)), r * math.sin(math.radians(a))))
                g.at(XF(bank * c['x'] + ROCKER_X[kind] - 6.5)).prism(out, 0, 13.0)
        g.at()
        fl = xchain + (6 if xchain < xf else -6)
        g.cyl((min(fl, xchain), 0, 0), (max(fl, xchain), 0, 0), 22.0, n=28)      # sprocket flange
        cam = E.add('Camshaft_' + tag, g, 'steel', 'cams', 'Camshaft, %s' % ('left' if bank > 0 else 'right'),
                    cut=False, loc=(0, bank * S.CAM_U, 0), rot=bank_q(bank), sharp=28)
        E.anim['Camshaft_' + tag] = (lambda th, bank=bank: (None, Quaternion((1, 0, 0), math.radians(-0.5 * th)) @ bank_q(bank), None))
        # cam sprocket, 28 teeth, two rows, a seat turned to meet a chain pin at theta = 0
        pins = S.chain_pins(bank, 0.0)
        best = max(pins, key=lambda p: bank * p[0])
        seat = math.degrees(math.atan2(best[1], bank * (best[0] - bank * S.CAM_U)))
        g = Geo()
        m = XF(xchain) @ Matrix.Rotation(math.radians(seat), 4, 'Z')
        out = sprocket_outline(S.Z_CAM_SPROCKET, S.CHAIN_PITCH, S.CHAIN_ROLLER)
        for row in (-1, 1):
            g.at(m).prism(out, row * S.CHAIN_ROW / 2 - 2.6, row * S.CHAIN_ROW / 2 + 2.6, holes=[circle(15.0, 24)])
        g.at(m).prism(circle(30.0, 32), -S.CHAIN_ROW / 2 - 2.6, S.CHAIN_ROW / 2 + 2.6, holes=[circle(15.0, 24)])
        g.at(m)
        for k in range(3):
            a = TAU * k / 3
            g.bolt((21 * math.cos(a), 21 * math.sin(a), (S.CHAIN_ROW / 2 + 2.6) * (1 if xchain > xf else -1)), (0, 0, 1 if xchain > xf else -1), 6.0)
        g.at()
        E.add('CamSprocket_' + tag, g, 'steel', 'timing', 'Cam sprocket, 28 teeth, %s' % ('left' if bank > 0 else 'right'), cut=False, parent=cam)


# ---------------------------------------------------------------------------------------------------------------
# timing drive
# ---------------------------------------------------------------------------------------------------------------
def build_timing(E):
    """Intermediate shaft, oil pump, both chains with idlers, tensioners and ramps, chain housings and covers.
    Takes the Engine.  Returns nothing."""
    mod = 2.0 * S.IMS_DROP / (S.Z_CRANK_GEAR + S.Z_IMS_GEAR)
    ratio = -float(S.Z_CRANK_GEAR) / S.Z_IMS_GEAR
    g = Geo()
    g.cyl((-262, 0, 0), (-180, 0, 0), 15.0, n=24)
    gear = gear_outline(S.Z_IMS_GEAR, mod * S.Z_IMS_GEAR / 2, mod, mod * 1.2)
    g.at(XF(S.X_TIMING_GEAR - 7) @ Matrix.Rotation(math.radians(90 + 180.0 / S.Z_IMS_GEAR), 4, 'Z')).prism(gear, 0, 14, holes=[circle(15.0, 24)] + [
        [(34 * math.cos(TAU * k / 6) + p[0], 34 * math.sin(TAU * k / 6) + p[1]) for p in circle(9.0, 12)] for k in range(6)])
    out = sprocket_outline(S.Z_IMS_SPROCKET, S.CHAIN_PITCH, S.CHAIN_ROLLER)
    for bank in (+1, -1):
        pins = S.chain_pins(bank, 0.0)
        low = min(pins, key=lambda p: p[1])
        seat = math.degrees(math.atan2(low[1] + S.IMS_DROP, low[0]))
        for row in (-1, 1):
            g.at(XF(S.X_CHAIN[bank]) @ Matrix.Rotation(math.radians(seat), 4, 'Z')).prism(out, row * S.CHAIN_ROW / 2 - 2.6, row * S.CHAIN_ROW / 2 + 2.6, holes=[circle(15.0, 24)])
    g.at()
    E.add('Intermediate_Shaft', g, 'steel', 'timing', 'Intermediate shaft: 48-tooth gear, two 24-tooth sprockets', cut=False,
          loc=(0, 0, -S.IMS_DROP), spin=(1, 0, 0, ratio))
    # oil pump in the sump, driven from the front of the intermediate shaft
    g = Geo(); g.cyl((-180, 0, 0), (-62, 0, 0), 5.5, n=12)
    E.add('Oil_Pump_Shaft', g, 'steel', 'oil', 'Oil pump drive shaft', cut=False, loc=(0, 0, -S.IMS_DROP), spin=(1, 0, 0, ratio))
    g = Geo()
    g.cyl((-62, 0, -118), (58, 0, -118), 31.0, n=28)
    g.box((-2, 0, -132), (112, 74, 26), bevel=4)
    for x in (-40, 0, 40):
        g.cyl((x, 0, -118), (x, 0, -86), 9.0, n=12)
    g.cyl((-62, 0, -84), (-50, 0, -84), 13.0, n=16)
    g.cyl((-56, 0, -118), (-56, 0, -84), 12.0, n=14)
    E.add('Oil_Pump', g, 'alu_cast', 'oil', 'Oil pump: one pressure stage, scavenge stages', ex=(0, 0, -230))

    # chains: one inner and one outer link mesh, instanced along the path
    def link(inner):
        """One chain link: both rows' rollers on this pin and the plates reaching to the next pin."""
        gl = Geo()
        P = S.CHAIN_PITCH
        plate = hull_of([((0, 0), 4.1), ((P, 0), 4.1)], n=14)
        offs = (1.9, 3.9) if inner else (4.4, 6.0)
        for row in (-1, 1):
            xr = row * S.CHAIN_ROW / 2
            if inner:
                gl.cyl((xr - 2.6, 0, 0), (xr + 2.6, 0, 0), S.CHAIN_ROLLER / 2, n=10)
                gl.cyl((xr - 2.6, P, 0), (xr + 2.6, P, 0), S.CHAIN_ROLLER / 2, n=10)
            for side in (-1, 1):
                xa = xr + side * offs[0]
                gl.at(XF(min(xa, xa + side * 1.3))).prism(plate, 0, 1.3)
                gl.at()
        if not inner:
            gl.cyl((-S.CHAIN_ROW / 2 - 7.2, 0, 0), (S.CHAIN_ROW / 2 + 7.2, 0, 0), 1.9, n=8)
            gl.cyl((-S.CHAIN_ROW / 2 - 7.2, P, 0), (S.CHAIN_ROW / 2 + 7.2, P, 0), 1.9, n=8)
        return gl
    srcs = {True: None, False: None}
    for bank in (+1, -1):
        tag = 'L' if bank > 0 else 'R'
        pins = S.chain_pins(bank, 0.0)
        path = S.chain_path(bank)
        E.chains[tag] = dict(x=S.X_CHAIN[bank], pts=[[round(p[0], 3), round(p[1], 3)] for p in path['pts']], links=S.CHAIN_LINKS,
                             pitch=S.CHAIN_PITCH, pitches_per_rev=S.Z_IMS_SPROCKET * S.Z_CRANK_GEAR / S.Z_IMS_GEAR)
        for k, p in enumerate(pins):
            q = pins[(k + 1) % len(pins)]
            a = math.atan2(q[1] - p[1], q[0] - p[0])
            inner = (k % 2 == 0)
            name = 'Chain_%s_%03d' % (tag, k)
            kw = dict(loc=(S.X_CHAIN[bank], p[0], p[1]), rot=Quaternion((1, 0, 0), a), label='Timing chain, %s' % ('left' if bank > 0 else 'right'))
            if srcs[inner] is None:
                srcs[inner] = E.add(name, link(inner), 'steel_dark', 'chain', cut=False, **kw)
            else:
                E.inst(srcs[inner], name, **kw)
        # idler sprocket on its arm, and the tensioner that pushes the arm
        iy, iz = path['idler']
        near = min(pins, key=lambda p: math.hypot(p[0] - iy, p[1] - iz) if (p[1] - iz) * bank > 0 or True else 1e9)
        seat = math.degrees(math.atan2(near[1] - iz, near[0] - iy))
        g = Geo()
        out19 = sprocket_outline(S.Z_IDLER, S.CHAIN_PITCH, S.CHAIN_ROLLER)
        for row in (-1, 1):
            g.at(XF(0) @ Matrix.Rotation(math.radians(seat), 4, 'Z')).prism(out19, row * S.CHAIN_ROW / 2 - 2.6, row * S.CHAIN_ROW / 2 + 2.6, holes=[circle(8.0, 16)])
        g.at(XF(0)).prism(circle(17.0, 24), -S.CHAIN_ROW / 2 - 2.6, S.CHAIN_ROW / 2 + 2.6, holes=[circle(8.0, 16)])
        g.at()
        E.add('Idler_' + tag, g, 'steel', 'timing', 'Tensioner idler sprocket, 19 teeth', cut=False, loc=(S.X_CHAIN[bank], iy, iz),
              spin=(1, 0, 0, ratio * S.Z_IMS_SPROCKET / S.Z_IDLER))
        up = 1.0 if bank > 0 else -1.0          # the idler pushes up on the left bank, down on the right
        piv = (iy + bank * 62.0, iz - up * 24.0)
        g = Geo()
        for side in (-1, 1):
            g.at(XF(S.X_CHAIN[bank] + side * 13.5 - 1.5)).prism(hull_of([((iy, iz), 11.0), (piv, 9.0)], n=16), 0, 3)
        g.at()
        g.cyl((S.X_CHAIN[bank] - 15, iy, iz), (S.X_CHAIN[bank] + 15, iy, iz), 7.9, n=14)
        g.cyl((S.X_CHAIN[bank] - 16, piv[0], piv[1]), (S.X_CHAIN[bank] + 16, piv[0], piv[1]), 6.5, n=14)
        E.add('IdlerArm_' + tag, g, 'steel', 'timing', 'Idler arm', cut=False)
        g = Geo()
        tb = (iy + bank * 8.0, iz - up * 58.0)
        g.cyl((S.X_CHAIN[bank], tb[0], tb[1]), (S.X_CHAIN[bank], tb[0], tb[1] + up * 30), 13.0, n=18)
        g.cyl((S.X_CHAIN[bank], tb[0], tb[1] + up * 30), (S.X_CHAIN[bank], tb[0], tb[1] + up * 46), 6.0, n=12)
        g.box((S.X_CHAIN[bank], tb[0], tb[1] - up * 4), (26, 34, 10), bevel=2)
        E.add('Tensioner_' + tag, g, 'alu_mach', 'timing', 'Chain tensioner', cut=False)
        # guide ramps on the runs; five black and one brown across the engine [S]
        a_pt, b_pt = (0.0, -S.IMS_DROP), (bank * S.CAM_U, 0.0)
        dy, dz = b_pt[0] - a_pt[0], b_pt[1] - a_pt[1]
        Lc = math.hypot(dy, dz)
        ty, tz = dy / Lc, dz / Lc
        ny, nz = (-tz, ty) if (-tz * 0 + ty) > 0 else (tz, -ty)       # normal with +z
        for k, (f, side) in enumerate(((0.30, -up), (0.68, -up), (0.80, up))):
            cy, cz = a_pt[0] + dy * f, a_pt[1] + dz * f
            off = (S.pitch_radius(S.Z_IMS_SPROCKET) * (1 - f) + S.pitch_radius(S.Z_CAM_SPROCKET) * f + 7.5) * side
            cy, cz = cy + ny * off, cz + nz * off
            g = Geo()
            g.at(basis((S.X_CHAIN[bank], cy, cz), (0, ty, tz), (1, 0, 0))).box((0, 0, 0), (54, 22, 7), bevel=2)
            g.at()
            brown = (bank < 0 and k == 0)
            E.add('ChainRamp_%s_%d' % (tag, k + 1), g, 'plastic_brown' if brown else 'plastic_black', 'timing', 'Chain guide ramp', cut=False)

        # housing: a wall round the chain, a front wall toward the cam housing and a bolted cover on the rear
        xa, xb = CHAIN_X[bank]          # rear, front in world X

        def outline(pad):
            """Chain outline grown by pad, in the bank frame, counter-clockwise, clipped at the case wall and
            rotated so it starts and ends on the clip line."""
            poly = [(bank * p[0], p[1]) for p in S.chain_hull(bank, pad)]
            if bank < 0:
                poly = poly[::-1]
            poly = clip_min_u(poly, 112.5)
            n = len(poly)
            k = next(i for i in range(n) if poly[i][0] < 112.6 and poly[(i + 1) % n][0] < 112.6)
            return poly[k + 1:] + poly[:k + 1]
        outer, inner = outline(24.0), outline(17.0)
        lo, hi = sorted((bank * xa, bank * xb))
        rear_lo = (bank * xa == lo)
        g = Geo()
        g.at(XF(lo + 5)).prism(outer + inner[::-1], 0, hi - lo - 10)                    # wall, open toward the case
        g.at(XF(hi - 5 if rear_lo else lo)).prism(outer, 0, 5, holes=[[(S.CAM_U + p[0], p[1]) for p in circle(27.0, 28)]])
        if xa < X_REAR:         # close the part that stands behind the crankcase
            a, b = sorted((bank * xa, bank * X_REAR))
            g.at(XF(a)).prism([(112.5, inner[0][1]), (119.5, inner[0][1]), (119.5, inner[-1][1]), (112.5, inner[-1][1])][::-1 if inner[0][1] > inner[-1][1] else 1], 0, b - a)
        g.at()
        E.add('ChainHousing_' + tag, g, 'alu_cast', 'chainhousing', 'Chain housing, %s' % ('left' if bank > 0 else 'right'),
              rot=bank_q(bank), ex=(-60, bank * 250, 0), sharp=30)
        g = Geo()
        g.at(XF(lo if rear_lo else hi - 5)).prism(outer, 0, 5)
        g.at()
        E.add('ChainCover_' + tag, g, 'alu_cast', 'chainhousing', 'Chain housing cover, %s' % ('left' if bank > 0 else 'right'),
              rot=bank_q(bank), ex=(-170, bank * 250, 0), sharp=30)
        bolt = Geo(); bolt.bolt((0, 0, 0), (-1, 0, 0), 6.0)
        m = min(len(outer), len(inner))
        for k, i in enumerate(range(1, m - 1, max(1, m // 12))):
            u, w = (outer[i][0] + inner[i][0]) / 2, (outer[i][1] + inner[i][1]) / 2
            E._cover_bolt = E.fastener(getattr(E, '_cover_bolt', None), 'ChainCoverBolt_%s_%02d' % (tag, k), bolt, V((xa, bank * u, w)),
                                       label='Chain cover bolt M6', group='chainhousing', ex=(-230, bank * 250, 0))


def build_all(E):
    """Build the mechanism in list order.  Takes the Engine.  Returns nothing."""
    build_crank(E)
    build_rods_pistons(E)
    build_case(E)
    build_cylinders(E)
    build_heads(E)
    build_valves(E)
    build_rockers(E)
    build_cam_housings(E)
    build_cams(E)
    build_timing(E)
