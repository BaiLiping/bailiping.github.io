# engine_dress.py: everything bolted round the mechanism.  Cooling, induction, turbocharger, exhaust, ignition,
# oil, mounts.  Positions come from photographs of 930/60 engines (see PARTS.md); few are dimensioned, so most
# numbers here are tagged [P] read off a photograph or [E] chosen to fit.
#
# Part list, in build order:
#   Fan_Housing, Fan [spins 1.67], Fan_Strap, Alternator, V_Belt, Fan_Shroud, Shroud_Label
#   Oil_Cooler, OilReturnTube_1..4, Oil_Filler, Oil_Pressure_Sender
#   Engine_Carrier, EngineMount_L/R, Carrier_Bracket
#   Intake_Manifold, Intake_Runner_1..6, Injector_1..6, Throttle_Body, Throttle_Plate [throttle]
#   Intercooler_Core, Intercooler_Tank_In/Out, Intercooler_Seal, Intermediate_Manifold, Bypass_Valve, Recirc_Hose
#   Air_Cleaner, Air_Filter, Airflow_Sensor, Sensor_Plate [airflow], Fuel_Distributor, FuelLine_1..6,
#   Warmup_Regulator, Inlet_Duct, Charge_Pipe
#   Turbo_Compressor_Housing, Turbo_Centre_Housing, Turbo_Turbine_Housing, Turbo_Rotor [turbo],
#   Turbo_Oil_Feed, Turbo_Oil_Drain, Turbo_Oil_Tank, Turbo_Scavenge_Pump
#   HeatExchanger_L/R, ExhaustFlangeNut_*, Crossover_Pipe, Turbine_Inlet_Pipe, Wastegate_Body,
#   Wastegate_Valve [wastegate], Wastegate_Pipe, Turbine_Outlet_Pipe, Muffler, Tailpipe_L/R, Muffler_Strap
#   Distributor_Body, Distributor_Cap, Distributor_Rotor [spins 0.5], Ignition_Coil, HT_Lead_1..6, HT_Lead_Coil
#   Air_Pump, Air_Pump_Belt, Cam_Pulley_L
import math
from mathutils import Vector as V, Matrix, Quaternion
from kit import Geo, frame, basis, circle, rounded_rect, fillet_path, smooth_path, TAU
import spec as S
from engine_core import bank_m, bank_q, XF, hull_of, bank_span

FZ = S.FAN_Z
FAN_X0, FAN_X1 = S.FAN_X[0] - 8.0, S.FAN_X[1] + 12.0       # fan housing ring along X
BAR_X = (-362.0, -336.0)                                    # engine carrier, behind the pulley [P]
TURBO = dict(y=185.0, z=-95.0, x_in=-346.0, x_comp=-366.0, x_turb=-462.0, x_out=-495.0)    # [P] for the place
IC = dict(x0=-440.0, x1=-161.0, y0=-5.0, y1=287.0, z0=325.0, z1=382.0, tank=63.0)          # core [S], place [P]
MUFFLER = dict(x=-575.0, z=-175.0, y0=-300.0, y1=330.0)
PLENUM = dict(x=0.0, z=272.0)
THROTTLE_DIR = V((-math.cos(math.radians(35)), 0.0, math.sin(math.radians(35))))            # 35 degrees up [S]


def pipe_part(E, name, pts, r, mat, group, label, r_in=0.0, n=16, bend=None, smooth=0, scales=None, **kw):
    """Build one pipe or hose as a part.  Takes the Engine, the name, control points, outer radius, material,
    group, label, an inner radius for a hollow pipe, segments round, a bend radius for filleted corners or a
    sample count for a smooth spline, and optional scales per path point.  Returns the path it used."""
    if bend is not None:
        path = fillet_path(pts, bend)
    elif smooth:
        path = smooth_path(pts, smooth)
    else:
        path = [V(p) for p in pts]
    g = Geo()
    g.pipe(path, r, r_in, n=n, scales=scales)
    E.add(name, g, mat, group, label, **kw)
    return path


def gas(E, gid, kind, path, cyl=0, r=20.0):
    """Record a gas path for the viewer's flow particles.  Takes the Engine, an id, the kind of gas, the path
    points, the cylinder it serves (0 for none) and the passage radius.  Returns nothing."""
    E.gas.append(dict(id=gid, kind=kind, cyl=cyl, r=r, pts=[[round(p[0], 1), round(p[1], 1), round(p[2], 1)] for p in path]))


def hose_clamp(g, p, d, r):
    """Add a worm-drive hose clamp to a Geo.  Takes the Geo, a point on the hose axis, the axis direction and the
    hose radius.  Returns nothing."""
    p, d = V(p), V(d).normalized()
    g.tube(p - d * 5, p + d * 5, r + 1.6, r - 0.5, n=20)
    side = d.orthogonal().normalized()
    g.at(basis(p + side * (r + 5), d.cross(side), d)).box((0, 0, 0), (14, 9, 8), bevel=1.5)
    g.at()


# ---------------------------------------------------------------------------------------------------------------
# cooling
# ---------------------------------------------------------------------------------------------------------------
def shroud_arch(W, H, ze, q, n=32, inset=0.0):
    """One station of the fan shroud: an arch from the left edge over the top to the right edge.  Takes the
    half-width, top height, edge height, a fullness exponent, the segment count and an inward offset.
    Returns [(y, z), ...]."""
    out = []
    for i in range(n + 1):
        phi = math.radians(-90.0 + 180.0 * i / n)
        y = (W - inset) * math.sin(phi)
        z = ze + (H - inset - ze) * (max(0.0, math.cos(phi)) ** q)
        out.append((-y, z))
    return out


SHROUD = [(-241.0, 136, 317, 181, 1.00), (-216.0, 152, 311, 150, 0.90), (-172.0, 200, 286, 110, 0.74),
          (-100.0, 214, 252, 92, 0.62), (0.0, 216, 230, 84, 0.55), (110.0, 216, 206, 82, 0.55), (204.0, 214, 182, 82, 0.55)]


def build_cooling(E):
    """Fan, alternator, belt, housing ring, strap and the moulded shroud.  Takes the Engine.  Returns nothing."""
    # housing ring with its foot on the crankcase
    g = Geo()
    g.at(frame((FAN_X0, 0, FZ), z=(1, 0, 0), x=(0, 1, 0)))
    L = FAN_X1 - FAN_X0
    g.lathe([(124.5, 0), (133.5, 0), (133.5, 5), (131, 5), (131, L - 5), (135, L - 5), (135, L), (124.5, L)], n=72, closed=True)
    g.at()
    g.box(((FAN_X0 + FAN_X1) / 2 + 6, 0, 50), (36, 110, 12), bevel=3)
    E.add('Fan_Housing', g, 'alu_cast', 'cooling', 'Fan housing', ex=(-60, 0, 260))
    # fan: hub, back plate and 11 blades [S]
    g = Geo()
    xm = (S.FAN_X[0] + S.FAN_X[1]) / 2
    g.at(frame((0, 0, 0), z=(1, 0, 0), x=(0, 1, 0)))
    g.lathe([(0, -17), (30, -17), (56, -8), (121, 13), (121, 15.5), (56, 15.5), (40, 20), (0, 20)], n=66)
    R_P = S.FAN_PULLEY_D / 2
    g.lathe([(0, -44), (20, -44), (R_P, -39), (R_P, -36), (R_P - 9, -32.5), (R_P - 9, -29.5), (R_P, -26), (R_P, -23), (24, -17), (0, -17)], n=40)
    g.hexa((0, 0, -44), (0, 0, -1), 22.0, 9.0)
    for k in range(S.FAN_BLADES):
        a = TAU * k / S.FAN_BLADES
        rad = V((0, math.cos(a), math.sin(a)))
        tan = V((1, 0, 0)).cross(rad)
        pitch = math.radians(32)
        chord = V((1, 0, 0)) * math.cos(pitch) + tan * math.sin(pitch)
        g.at(basis(rad * 89.0 + V((0.5, 0, 0)), rad, chord))
        g.box((0, 0, 0), (64, 31, 2.6), bevel=0.8)
    g.at()
    E.add('Fan', g, 'alu_mach', 'cooling', 'Cooling fan, 245 mm, 11 blades', cut=False, loc=(xm, 0, FZ), ex=(-150, 0, 260),
          spin=(1, 0, 0, S.FAN_RATIO))
    g = Geo()
    g.at(frame((0, 0, FZ), z=(1, 0, 0), x=(0, 1, 0)))
    xa = S.FAN_X[1] + 6
    g.lathe([(0, xa), (58, xa), (62, xa + 6), (62, xa + 104), (52, xa + 118), (30, xa + 122), (0, xa + 122)], n=40)
    for k in range(12):         # cooling slots suggested by ribs on the rear shell
        a = TAU * k / 12
        g.at(frame((0, 0, FZ), z=(1, 0, 0), x=(0, 1, 0)) @ Matrix.Rotation(a, 4, 'Z')).box((60, 0, xa + 60), (5, 7, 84), bevel=1)
    g.at()
    for dy in (-16, 16):
        g.nut((xa + 122, dy, FZ + 22), (1, 0, 0), 6.0, stud=7)
    E.add('Alternator', g, 'alu_cast', 'cooling', 'Alternator, inside the fan hub', loc=(0, 0, 0), ex=(-20, 0, 300))
    # strap round the ring, zinc plated [S]
    g = Geo()
    g.at(frame(((FAN_X0 + FAN_X1) / 2 - 9, 0, FZ), z=(1, 0, 0), x=(0, 1, 0)))
    g.lathe([(131.2, 0), (133.2, 0), (133.2, 18), (131.2, 18)], n=72, closed=True)
    g.at()
    g.box(((FAN_X0 + FAN_X1) / 2, 0, FZ + 138), (16, 30, 10), bevel=2)
    g.bolt(((FAN_X0 + FAN_X1) / 2, 15, FZ + 138), (0, 1, 0), 6.0)
    E.add('Fan_Strap', g, 'zinc', 'cooling', 'Fan housing strap', cut=False, ex=(-60, 0, 330))
    # V-belt round the two pulleys
    loop = hull_of([((0.0, 0.0), S.CRANK_PULLEY_D / 2 - 5.5), ((0.0, FZ), S.FAN_PULLEY_D / 2 - 5.5)], n=40)
    g = Geo()
    g.sweep([(S.X_PULLEY, p[0], p[1]) for p in loop], [(-4, -3.2), (4, -4.9), (4, 4.9), (-4, 3.2)], closed_path=True)
    E.add('V_Belt', g, 'rubber', 'cooling', 'Fan belt, 9.5 x 710', cut=False, ex=(-190, 0, 0))
    # moulded shroud that ducts the fan's air over both banks
    outer = [[(x, y, z) for (y, z) in shroud_arch(W, H, ze, q)] for (x, W, H, ze, q) in SHROUD]
    inner = [[(x, y, z) for (y, z) in shroud_arch(W, H, ze, q, inset=3.2)] for (x, W, H, ze, q) in SHROUD]
    g = Geo()
    g.shell(outer, inner)
    E.add('Fan_Shroud', g, 'black_grp', 'cooling', 'Fan shroud', ex=(40, 0, 330), sharp=50)
    g = Geo()
    g.at(basis((-150, -60, 281.5), (1, 0, 0), (0, 1, 0.12))).box((0, 0, 0), (70, 48, 1.2))
    g.at()
    E.add('Shroud_Label', g, 'label_yellow', 'cooling', 'Service label', cut=False, ex=(40, 0, 332))
    for name, pts in (('cool_L', [(-300, 40, FZ), (-250, 60, FZ + 30), (-150, 120, 220), (-20, 165, 150), (20, 170, 20), (20, 175, -110)]),
                      ('cool_R', [(-300, -40, FZ), (-250, -60, FZ + 30), (-150, -120, 220), (-80, -165, 150), (-40, -170, 20), (-40, -175, -110)])):
        gas(E, name, 'cooling', smooth_path(pts, 6), r=60)


def build_oil(E):
    """Engine oil cooler, the four oil return tubes, filler and pressure sender.  Takes the Engine.  Returns
    nothing."""
    g = Geo()
    g.box((180, -148, 55), (58, 92, 190), bevel=5)
    for k in range(16):         # fin rows [S: 15 rows of tubes]
        g.box((180, -148, -32 + k * 11.6), (62, 96, 2.2))
    g.cyl((180, -110, -44), (180, -110, -70), 11, n=14)
    g.cyl((166, -110, -44), (166, -100, -70), 9, n=12)
    E.add('Oil_Cooler', g, 'alu_cast', 'oil', 'Engine oil cooler', ex=(170, -150, 0))
    g = Geo()
    g.tube((0, 112.5, 0), (0, 205, 0), 11.0, 8.5, n=18)
    g.tube((0, 195, 0), (0, 283, 0), 13.0, 11.0, n=18)
    g.torus((0, 197, 0), (0, 1, 0), 13.5, 1.6, n=18, m=6)
    src = None
    k = 1
    for bank in (+1, -1):
        cx = sorted(c['x'] for c in S.CYLS if c['bank'] == bank)
        for xm in ((cx[0] + cx[1]) / 2, (cx[1] + cx[2]) / 2):
            src = E.fastener(src, 'OilReturnTube_%d' % k, g, (xm, 0, -79), rot=bank_q(bank), mat='alu_mach', label='Oil return tube',
                             group='oil', cut=True, ex=(0, bank * 60, -120))
            k += 1
    g = Geo()
    g.cyl((60, 22, 92), (60, 22, 126), 17, n=20)
    g.cyl((60, 22, 126), (60, 22, 136), 22, n=20)
    E.add('Oil_Breather', g, 'alu_cast', 'oil', 'Crankcase breather', ex=(0, 0, 200))
    g = Geo()
    g.cyl((110, -24, 92), (110, -24, 104), 9, n=12)
    g.cyl((110, -24, 104), (110, -24, 140), 15, n=18)
    g.cyl((110, -24, 140), (110, -24, 150), 4, n=8)
    E.add('Oil_Pressure_Sender', g, 'zinc', 'oil', 'Oil pressure sender', cut=False, ex=(0, 0, 220))


def build_mounts(E):
    """Engine carrier across the rear with its two rubber mounts and the bracket to the crankcase.  Takes the
    Engine.  Returns nothing."""
    bar = [(-338, 0), (-250, -14), (250, -14), (338, 0), (338, 24), (250, 46), (-250, 46), (-338, 24)]
    g = Geo()
    g.at(XF(BAR_X[0])).prism(bar, 0, BAR_X[1] - BAR_X[0], holes=[[(p[0], 8 + p[1]) for p in circle(40, 28)]])
    g.at(XF(BAR_X[0] - 5)).prism([(-250, 38), (250, 38), (250, 46), (-250, 46)], 0, 5)       # pressed lips
    g.at(XF(BAR_X[0] - 5)).prism([(-250, -14), (250, -14), (250, -6), (-250, -6)], 0, 5)
    g.at()
    E.add('Engine_Carrier', g, 'black_paint', 'mounts', 'Engine carrier', ex=(-230, 0, 0))
    g = Geo()
    xm = (BAR_X[0] + BAR_X[1]) / 2
    g.cyl((xm, 318, 12), (xm, 318, 40), 26, n=24)
    g.cyl((xm, 318, 40), (xm, 318, 46), 30, n=24)
    g.cyl((xm, 318, 46), (xm, 318, 76), 6, n=10)
    g.hexa((xm, 318, 60), (0, 0, 1), 19, 9)
    src = E.add('EngineMount_L', g, 'rubber', 'mounts', 'Engine mount', cut=False, ex=(-230, 60, 40))
    E.inst(src, 'EngineMount_R', loc=(0, -636, 0), ex=(-230, -60, 40))
    g = Geo()
    for sy in (-1, 1):
        g.box(((BAR_X[1] - 274) / 2, sy * 92, -26), (BAR_X[1] + 274, 24, 34), bevel=4)
        g.bolt((-278, sy * 92, -9), (0, 0, 1), 10.0)
        g.bolt((BAR_X[1] + 10, sy * 92, -9), (0, 0, 1), 10.0)
    E.add('Carrier_Bracket', g, 'black_paint', 'mounts', 'Engine carrier bracket', ex=(-200, 0, -60))


# ---------------------------------------------------------------------------------------------------------------
# induction
# ---------------------------------------------------------------------------------------------------------------
def port_point(c, d=0.0):
    """World position on a cylinder's intake port axis.  Takes a CYLS row and a distance beyond the flange.
    Returns a Vector."""
    return bank_m(c['bank'], c['x']) @ V((0, 243 + d * 0.22, 106 + d * 0.975))


def build_manifold(E):
    """Intake manifold: central plenum, a log over each bank, six runners with their injectors, and the
    throttle body.  Takes the Engine.  Returns nothing."""
    px, pz = PLENUM['x'], PLENUM['z']
    g = Geo()
    g.tube((px, -150, pz), (px, 150, pz), 36, 31, n=28)
    for bank in (+1, -1):
        cx = sorted(c['x'] for c in S.CYLS if c['bank'] == bank)
        ya, yb = bank * 132, bank * 214
        plan = [(px - 34, ya), (px + 34, ya), (cx[2] + 30, yb - bank * 22), (cx[2] + 30, yb), (cx[0] - 30, yb), (cx[0] - 30, yb - bank * 22)]
        if bank < 0:
            plan = plan[::-1]
        g.at(basis((0, 0, pz - 27), (1, 0, 0), (0, 1, 0))).prism(plan, 0, 54)
    g.at()
    tb0 = V((px, 0, pz)) + THROTTLE_DIR * 30
    g.tube(V((px, 0, pz)), tb0 + THROTTLE_DIR * 12, 44, 34, n=28)
    E.add('Intake_Manifold', g, 'alu_cast', 'induction', 'Intake manifold', ex=(0, 0, 400), sharp=32)
    # runners from the logs down to the ports, and the injectors beside them
    src = isrc = None
    inj = Geo()
    inj.cyl((0, 0, 0), (0, 0, 34), 6.5, n=10)
    inj.hexa((0, 0, 34), (0, 0, 1), 12, 7)
    inj.cyl((0, 0, 41), (0, 0, 52), 3.2, n=8)
    for c in S.CYLS:
        b = c['bank']
        top = V((c['x'], b * 206, pz - 4))
        pts = [top, V((c['x'], b * 232, pz - 34)), port_point(c, 60), port_point(c, 0)]
        path = smooth_path(pts, 5)
        g = Geo()
        g.pipe(path, 22.5, 16.5, n=20)
        g.at(basis(port_point(c, 0), (1, 0, 0), (0, b, 0) if b > 0 else (0, 1, 0))).prism(rounded_rect(76, 56, 12, 4), 0, 7, holes=[circle(16.5, 20)])
        g.at()
        E.add('Intake_Runner_%d' % c['n'], g, 'alu_cast', 'induction', 'Intake runner %d' % c['n'], ex=(0, b * 30, 380))
        seat = port_point(c, 26) + V((24, 0, 0))
        d = V((0.45, b * 0.2, 0.87)).normalized()
        q = V((0, 0, 1)).rotation_difference(d)
        if isrc is None:
            isrc = E.add('Injector_%d' % c['n'], inj, 'brass', 'fuel', 'Injector %d' % c['n'], cut=False, loc=seat, rot=q, ex=(0, b * 60, 420))
        else:
            E.inst(isrc, 'Injector_%d' % c['n'], label='Injector %d' % c['n'], loc=seat, rot=q, ex=(0, b * 60, 420))
        E.tmp_inj[c['n']] = seat + d * 52
        centre = [V((px, 0, pz)), V((px, b * 150, pz)), V((c['x'], b * 190, pz))] + path + [bank_m(b, c['x']) @ V((0, 232, 60)), bank_m(b, c['x']) @ V((0, S.VALVE['in']['seat'][0] + 6, S.VALVE['in']['seat'][1] + 4))]
        gas(E, 'intake_%d' % c['n'], 'charge', centre, cyl=c['n'], r=15)
    # throttle body and its plate
    g = Geo()
    g.tube(tb0 + THROTTLE_DIR * 12, tb0 + THROTTLE_DIR * 70, 42, 34, n=28)
    g.cyl(tb0 + THROTTLE_DIR * 40 + V((0, -52, 0)), tb0 + THROTTLE_DIR * 40 + V((0, 52, 0)), 5.0, n=10)
    g.box(tb0 + THROTTLE_DIR * 40 + V((0, 58, 0)), (26, 14, 30), bevel=2)
    E.add('Throttle_Body', g, 'alu_mach', 'induction', 'Throttle body, 68 mm', ex=(-30, 0, 440))
    g = Geo()
    g.at(frame((0, 0, 0), z=THROTTLE_DIR, x=(0, 1, 0))).lathe([(0, -1), (33.6, -1), (33.6, 1), (0, 1)], n=28)
    g.at()
    E.add('Throttle_Plate', g, 'brass', 'induction', 'Throttle plate', cut=False, loc=tb0 + THROTTLE_DIR * 40, ex=(-30, 0, 440),
          param=dict(p='throttle', axis=[0, 1, 0], deg=78))
    E.tmp_throttle_top = tb0 + THROTTLE_DIR * 70


def build_intercooler(E):
    """Intercooler with its two tanks and lid seal, the intermediate manifold with the bypass valve.  Takes the
    Engine.  Returns nothing."""
    x0, x1, y0, y1, z0, z1, tk = IC['x0'], IC['x1'], IC['y0'], IC['y1'], IC['z0'], IC['z1'], IC['tank']
    g = Geo()
    g.box(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2 - 2), (x1 - x0, y1 - y0, z1 - z0 - 6))
    n = 34
    for k in range(n):          # charge tubes run across; the fins stand along the car
        x = x0 + (x1 - x0) * (k + 0.5) / n
        g.box((x, (y0 + y1) / 2, z1 - 3), (2.4, y1 - y0, 6))
    E.add('Intercooler_Core', g, 'alu_cover', 'intercooler', 'Intercooler core, 279 x 292 x 57', ex=(-60, 60, 520))
    g = Geo()
    g.box(((x0 + x1) / 2, y1 + tk / 2, (z0 + z1) / 2), (x1 - x0, tk, z1 - z0 + 6), bevel=10, seg=2)
    g.tube((TURBO['x_comp'], y1 + tk / 2, z0 - 30), (TURBO['x_comp'], y1 + tk / 2, z0), 36, 30, n=24)
    E.add('Intercooler_Tank_In', g, 'black_paint', 'intercooler', 'Intercooler inlet tank', ex=(-60, 110, 520), sharp=35)
    g = Geo()
    g.box(((x0 + x1) / 2, y0 - tk / 2, (z0 + z1) / 2), (x1 - x0, tk, z1 - z0 + 6), bevel=10, seg=2)
    g.tube((x1, y0 - tk / 2, 353), (x1 + 16, y0 - tk / 2, 353), 36, 30, n=24)
    E.add('Intercooler_Tank_Out', g, 'black_paint', 'intercooler', 'Intercooler outlet tank', ex=(-60, 10, 520), sharp=35)
    g = Geo()
    ring = rounded_rect(x1 - x0 + 14, y1 - y0 + 2 * tk + 14, 22, 5)
    hole = rounded_rect(x1 - x0 - 10, y1 - y0 + 2 * tk - 10, 14, 5)
    g.at(basis(((x0 + x1) / 2, (y0 + y1) / 2, z1 + 3), (1, 0, 0), (0, 1, 0))).prism(ring, 0, 14, holes=[hole])
    g.at()
    E.add('Intercooler_Seal', g, 'rubber', 'intercooler', 'Seal to the engine lid grille', cut=False, ex=(-60, 60, 560))
    # intermediate manifold from the outlet tank down to the throttle body, carrying the bypass valve
    a = V((x1 + 16, y0 - tk / 2, 353))
    top = E.tmp_throttle_top
    pts = [a, a + V((26, 4, -2)), top - THROTTLE_DIR * -34 + V((0, -8, 0)), top]
    path = smooth_path(pts, 7)
    g = Geo()
    g.pipe(path, 40, 33, n=24)
    bp = path[len(path) // 2]
    g.cyl(bp, bp + V((0, -62, 6)), 24, n=20)
    g.cyl(bp + V((0, -62, 6)), bp + V((0, -72, 7)), 28, n=20)
    E.add('Intermediate_Manifold', g, 'alu_cast', 'intercooler', 'Intermediate manifold with the bypass valve housing', ex=(-40, -20, 470))
    g = Geo()
    g.cyl((0, 0, 0), (0, -30, 3), 17, n=16)
    g.helix((0, -30, 3), (0, -58, 5.6), 12, 1.6, 5, n=12, m=5)
    E.add('Bypass_Valve', g, 'steel', 'intercooler', 'Bypass valve piston: recirculates boost when the throttle shuts', cut=False,
          loc=bp + V((0, -4, 0)), ex=(-40, -90, 470), param=dict(p='bypass', move=[0, -18, 1.7]))
    E.tmp_bypass = bp + V((0, -72, 7))
    gas(E, 'charge_cool', 'charge', [V((TURBO['x_comp'], y1 + tk / 2, z0)), V(((x0 + x1) / 2, y1 + 20, 353)), V(((x0 + x1) / 2, (y0 + y1) / 2, 353)),
                                     V(((x0 + x1) / 2 + 60, y0 - tk / 2, 353)), a] + path + [V((PLENUM['x'], 0, PLENUM['z']))], r=26)


def build_airbox(E):
    """Air cleaner, airflow sensor, fuel distributor and lines, warm-up regulator, the long inlet duct to the
    compressor, the charge pipe and the recirculation hose.  Takes the Engine.  Returns nothing."""
    g = Geo()
    g.box((-110, -228, 318), (250, 160, 130), bevel=18, seg=2)
    g.tube((-232, -240, 325), (-322, -250, 334), 46, 41, n=24)
    for z in (300, 336):
        g.box((-110, -228, z), (254, 164, 3))
    E.add('Air_Cleaner', g, 'black_paint', 'induction', 'Air cleaner housing', ex=(-40, -200, 430), sharp=35)
    g = Geo()
    g.box((-110, -228, 318), (256, 166, 14), bevel=3)
    for sx in (-80, 60):
        g.box((-110 + sx, -310.5, 318), (22, 4, 44), bevel=1.5)
    E.add('Air_Cleaner_Band', g, 'filter_paper', 'induction', 'Air cleaner band and clips', cut=False, ex=(-40, -200, 430))
    g = Geo()
    for k in range(22):         # pleats of the filter element
        g.box((-215 + k * 10, -228, 318), (5, 140, 96))
    E.add('Air_Filter', g, 'filter_paper', 'induction', 'Air filter element', cut=False, ex=(-40, -200, 430))
    # airflow sensor housing with the fuel distributor on top
    sc = V((85, -196, 250))
    g = Geo()
    g.at(frame(sc, z=(0, 0, 1), x=(1, 0, 0))).lathe([(40, -30), (74, -30), (78, -10), (78, 22), (60, 30), (40, 30)], n=40, closed=True)
    g.at()
    g.box((38, -214, 286), (60, 96, 44), bevel=8)
    g.box((112, -150, 262), (70, 50, 40), bevel=6)
    E.add('Airflow_Sensor', g, 'alu_cast', 'induction', 'K-Jetronic airflow sensor', ex=(60, -220, 380))
    g = Geo()
    g.cyl((0, 0, -1.5), (0, 0, 1.5), 42, n=32)
    g.box((34, 14, 0), (110, 10, 6), bevel=1)
    E.add('Sensor_Plate', g, 'steel', 'induction', 'Airflow sensor plate: lifts with the air drawn in', cut=False, loc=sc + V((0, 0, 2)),
          ex=(60, -220, 380), param=dict(p='airflow', axis=[0, 1, 0], pivot=[96, 14, 0], deg=-16))
    fd = V((118, -150, 282))
    g = Geo()
    g.cyl(fd, fd + V((0, 0, 52)), 37, n=28)
    g.cyl(fd + V((0, 0, 52)), fd + V((0, 0, 60)), 30, n=28)
    fit = []
    for k in range(6):
        a = TAU * (k + 0.5) / 6
        p = fd + V((26 * math.cos(a), 26 * math.sin(a), 60))
        g.hexa(p, (0, 0, 1), 13, 9)
        fit.append(p + V((0, 0, 9)))
    g.hexa(fd + V((0, 0, 60)), (0, 0, 1), 15, 11)
    E.add('Fuel_Distributor', g, 'steel', 'fuel', 'K-Jetronic fuel distributor', ex=(60, -180, 470))
    order = sorted(S.CYLS, key=lambda c: math.atan2(c['bank'] * 240.0 + 150.0, c['x'] - 118.0))
    for k, c in enumerate(order):
        b = c['bank']
        end = E.tmp_inj[c['n']]
        rad = (fit[k] - fd).normalized()
        rad.z = 0.0
        up = fit[k] + V((0, 0, 12)) + rad * 4
        if b > 0:       # left bank: over the plenum
            pts = [fit[k], up, V((up.x + rad.x * 20, -96, 336 + k)), V((end.x, 60, 330 + k)), V((end.x, end.y - 34, 322)), end + V((0, 0, 30)), end]
        else:           # right bank: outboard of the sensor and down
            pts = [fit[k], up, V((up.x + rad.x * 16, -214 - k * 3, 332)), V((end.x, -214 - k * 3, 318)), end + V((0, 0, 34)), end]
        pipe_part(E, 'FuelLine_%d' % c['n'], pts, 2.4, 'fuel_line', 'fuel', 'Fuel line to injector %d' % c['n'], n=6, bend=14, cut=False, ex=(40, 0, 470))
    g = Geo()
    g.box((70, 150, 262), (64, 54, 46), bevel=6)
    g.cyl((70, 150, 285), (70, 150, 297), 16, n=16)
    E.add('Warmup_Regulator', g, 'alu_cast', 'fuel', 'Warm-up regulator', ex=(40, 60, 440))
    # inlet duct: from the sensor back along the right side, across behind the fan, down to the compressor
    ty, tz = TURBO['y'], TURBO['z']
    xd = -304.0
    pts = [sc + V((-30, -34, -22)), V((0, -296, 205)), V((-200, -302, 196)), V((xd, -296, 184)), V((xd, -262, 60)), V((xd, -240, -118)),
           V((xd, 60, -118)), V((xd, ty, tz - 6)), V((TURBO['x_in'], ty, tz))]
    path = fillet_path(pts, [70, 90, 70, 80, 60, 50, 30])
    g = Geo()
    g.pipe(path, 30, 26, n=22)
    for p, d in ((path[3], path[4] - path[3]), (path[-2], path[-1] - path[-2])):
        hose_clamp(g, p, d, 30)
    E.add('Inlet_Duct', g, 'rubber', 'induction', 'Inlet duct: metered air to the compressor', ex=(-170, -60, 330))
    box_in = [V((-322, -250, 334)), V((-232, -240, 325)), V((-110, -228, 318)), V((20, -214, 290)), sc + V((0, 0, 10))]
    gas(E, 'intake_ambient', 'ambient', box_in + path, r=24)
    E.tmp_duct = path
    # charge pipe: compressor outlet straight up to the intercooler inlet tank
    cp = [V((TURBO['x_comp'], ty + 56, tz + 4)), V((TURBO['x_comp'], ty + 74, tz + 90)), V((TURBO['x_comp'], IC['y1'] + IC['tank'] / 2, 110)),
          V((TURBO['x_comp'], IC['y1'] + IC['tank'] / 2, IC['z0'] - 28))]
    path = fillet_path(cp, 60)
    g = Geo()
    g.pipe(path, 35, 30, n=22)
    hose_clamp(g, path[-2], path[-1] - path[-2], 35)
    hose_clamp(g, path[1], path[1] - path[0], 35)
    E.add('Charge_Pipe', g, 'rubber', 'intercooler', 'Charge pipe: compressor to intercooler', ex=(-180, 110, 200))
    gas(E, 'charge_hot', 'boost', [V((TURBO['x_comp'], ty, tz)), V((TURBO['x_comp'], ty + 50, tz + 30))] + path, r=26)
    # recirculation hose from the bypass valve back into the inlet duct
    tgt = E.tmp_duct[len(E.tmp_duct) // 5]
    rp = pipe_part(E, 'Recirc_Hose', [E.tmp_bypass, E.tmp_bypass + V((0, -40, 0)), tgt + V((0, 30, 60)), tgt + V((0, 6, 24))], 15, 'rubber', 'intercooler',
                   'Recirculation hose', r_in=11, n=14, smooth=7, ex=(-40, -160, 430))
    gas(E, 'recirc', 'recirc', rp, r=10)


# ---------------------------------------------------------------------------------------------------------------
# turbocharger
# ---------------------------------------------------------------------------------------------------------------
def scroll(g, centre, R, r0, r1, start_deg, n=30, m=14):
    """Add a volute to a Geo: a tube wound once round the X axis whose section grows from r0 to r1.  Takes the
    Geo, the centre, the winding radius, the two section radii, the angle where the big end sits and the
    segment counts.  Returns the point and direction where the big end opens."""
    pts, sc = [], []
    for i in range(n + 1):
        f = i / n
        a = math.radians(start_deg + 334.0 * (1 - f))
        rr = R - (r1 - (r0 + (r1 - r0) * f)) * 0.5
        pts.append(V((centre[0], centre[1] + rr * math.cos(a), centre[2] + rr * math.sin(a))))
        sc.append((r0 + (r1 - r0) * f) / r1)
    g.sweep(pts, circle(r1, m), scales=sc, up=(1, 0, 0))
    return pts[-1], (pts[-1] - pts[-2]).normalized()


def build_turbo(E):
    """KKK K27 turbocharger: compressor, centre and turbine housings, the rotor, its oil lines, catch tank and
    scavenge pump.  Takes the Engine.  Returns nothing."""
    ty, tz = TURBO['y'], TURBO['z']
    xc, xt = TURBO['x_comp'], TURBO['x_turb']
    # compressor housing: inlet toward the engine, outlet up toward the intercooler
    g = Geo()
    end, d = scroll(g, (xc, ty, tz), 52, 12, 23, 20)
    g.tube(end, end + d * 22, 34, 25, n=22)
    g.at(frame((xc, ty, tz), z=(1, 0, 0), x=(0, 1, 0)))
    g.lathe([(27, 20), (36, 20), (36, 6), (52, 2), (52, -12), (44, -22), (27, -22)], n=44, closed=True)
    g.at()
    E.add('Turbo_Compressor_Housing', g, 'alu_cover', 'turbo', 'Compressor housing', ex=(-170, 190, -60), sharp=40)
    # centre housing with oil feed on top and drain underneath
    g = Geo()
    g.at(frame((xc, ty, tz), z=(-1, 0, 0), x=(0, 1, 0)))
    g.lathe([(9, 22), (46, 22), (46, 30), (32, 36), (32, 62), (48, 68), (48, 76), (9, 76)], n=36, closed=True)
    g.at()
    xm = xc - 50
    g.cyl((xm, ty, tz + 30), (xm, ty, tz + 44), 9, n=12)
    g.cyl((xm, ty, tz - 30), (xm, ty, tz - 46), 12, n=12)
    E.add('Turbo_Centre_Housing', g, 'iron_cast', 'turbo', 'Bearing housing', ex=(-250, 190, -60))
    # turbine housing: inlet underneath from the manifold pipe, outlet rearward into the muffler
    g = Geo()
    end, d = scroll(g, (xt, ty, tz), 50, 14, 27, -160)
    g.tube(end, end + d * 26, 33, 24, n=20)
    g.at(basis(end + d * 26, d.cross(V((1, 0, 0))), (1, 0, 0))).prism(rounded_rect(84, 70, 8, 3), 0, 9, holes=[circle(24, 20)])
    g.at(frame((xt, ty, tz), z=(-1, 0, 0), x=(0, 1, 0)))
    g.lathe([(33, -20), (48, -20), (56, -8), (56, 16), (44, 24), (44, TURBO['x_turb'] - TURBO['x_out']), (33, TURBO['x_turb'] - TURBO['x_out'])], n=40, closed=True)
    xo = TURBO['x_turb'] - TURBO['x_out']
    g.prism(circle(58, 28), xo - 9, xo, holes=[circle(33, 28)])
    for k in range(4):
        a = TAU * (k + 0.5) / 4
        g.nut((50 * math.cos(a), 50 * math.sin(a), xo), (0, 0, 1), 8.0, stud=5)
    g.at()
    E.add('Turbo_Turbine_Housing', g, 'iron_cast', 'turbo', 'Turbine housing', ex=(-340, 190, -60), sharp=40)
    E.tmp_turbine_in = end + d * 35
    E.tmp_turbine_dir = d
    # rotor: compressor wheel, shaft, turbine wheel, turning as one piece
    g = Geo()
    g.at(frame((0, 0, 0), z=(1, 0, 0), x=(0, 1, 0)))
    ci, ce = S.COMP_IND_D / 2, S.COMP_EXD_D / 2
    g.lathe([(0, 22), (7, 22), (9, 14), (15, 2), (26, -10), (ce, -14), (ce, -17), (0, -17)], n=28)
    for k in range(S.COMP_BLADES):
        full = (k % 2 == 0)
        a = TAU * k / S.COMP_BLADES
        g.at(frame((0, 0, 0), z=(1, 0, 0), x=(0, 1, 0)) @ Matrix.Rotation(a, 4, 'Z') @ Matrix.Rotation(math.radians(24), 4, 'X'))
        g.at(g.xf @ basis((0, 0, 0), (1, 0, 0), (0, 0, 1)))
        blade = [(8, 20 if full else 8), (ci, 20 if full else 8), (ce, -8), (ce, -14), (24, -10), (14, 3)]
        g.prism(blade, -0.6, 0.6)
    g.at()
    g.cyl((-104, 0, 0), (-16, 0, 0), 7.0, n=14)
    xw = xt - xc
    g.at(frame((xw, 0, 0), z=(-1, 0, 0), x=(0, 1, 0)))
    ti, te = S.TURB_IND_D / 2, S.TURB_EXD_D / 2
    g.lathe([(0, -16), (ti, -16), (ti, -13), (22, -6), (12, 12), (9, 26), (0, 30)], n=28)
    for k in range(S.TURB_BLADES):
        a = TAU * k / S.TURB_BLADES
        g.at(frame((xw, 0, 0), z=(-1, 0, 0), x=(0, 1, 0)) @ Matrix.Rotation(a, 4, 'Z') @ Matrix.Rotation(math.radians(-20), 4, 'X'))
        g.at(g.xf @ basis((0, 0, 0), (1, 0, 0), (0, 0, 1)))
        g.prism([(11, 24), (te, 24), (ti, 2), (ti, -13), (22, -6), (13, 10)], -0.8, 0.8)
    g.at()
    E.add('Turbo_Rotor', g, 'steel', 'turbo', 'Turbo rotor: turbine and compressor wheels on one shaft', cut=False, loc=(xc, ty, tz),
          ex=(-250, 320, -60), spin=(1, 0, 0, 0.0), param=dict(p='turbo'))
    # oil: feed from the case, drain into the catch tank, scavenge pump on the left camshaft's front end
    pipe_part(E, 'Turbo_Oil_Feed', [(xm, ty, tz + 44), (xm, ty, tz + 70), (-300, 120, 30), (-268, 60, 40), (-250, 40, 30)], 4.0, 'steel', 'turbo',
              'Turbo oil feed', n=8, smooth=6, cut=False, ex=(-200, 150, 0))
    pipe_part(E, 'Turbo_Oil_Drain', [(xm, ty, tz - 46), (xm, ty, tz - 78), (xm + 8, ty - 6, tz - 104)], 10.0, 'rubber', 'turbo', 'Turbo oil drain',
              n=12, smooth=5, cut=False, ex=(-250, 190, -150))
    g = Geo()
    g.box((xm + 8, ty - 6, tz - 126), (86, 110, 44), bevel=8, seg=2)
    g.cyl((xm + 44, ty - 30, tz - 108), (xm + 62, ty - 30, tz - 100), 7, n=10)
    E.add('Turbo_Oil_Tank', g, 'black_paint', 'turbo', 'Turbo oil catch tank', ex=(-250, 190, -230))
    xf = bank_span(+1)[0]
    g = Geo()
    g.cyl((xf + 16, S.CAM_U, 0), (xf + 44, S.CAM_U, 0), 27, n=24)
    g.cyl((xf + 44, S.CAM_U, 0), (xf + 50, S.CAM_U, 0), 20, n=20)
    g.cyl((xf + 30, S.CAM_U, -27), (xf + 30, S.CAM_U, -42), 8, n=10)
    g.cyl((xf + 30, S.CAM_U, 27), (xf + 30, S.CAM_U, 40), 7, n=10)
    E.add('Turbo_Scavenge_Pump', g, 'alu_cast', 'turbo', 'Turbo oil scavenge pump, driven by the left camshaft', ex=(120, 330, 0))


# ---------------------------------------------------------------------------------------------------------------
# exhaust
# ---------------------------------------------------------------------------------------------------------------
def build_exhaust(E):
    """Heat exchangers, crossover, the pipe to the turbine with its wastegate branch, the wastegate, muffler and
    tailpipes.  Takes the Engine.  Returns nothing."""
    nut = Geo(); nut.nut((0, 0, 0), (0, 0, -1), 8.0, stud=5)
    nsrc = None
    outlets = {}
    for bank in (+1, -1):
        tag = 'L' if bank > 0 else 'R'
        cyls = sorted([c for c in S.CYLS if c['bank'] == bank], key=lambda c: -c['x'])
        xa, xb = cyls[0]['x'] + 54, cyls[-1]['x'] - 52
        y, z = bank * 238.0, -168.0
        g = Geo()
        path = [(xa + 14, y, z), (xa, y, z), (xa - 20, y, z), (xb + 30, y, z), (xb, y, z), (xb - 16, y, z)]
        g.sweep(path, rounded_rect(84, 112, 30, 5), scales=[0.55, 0.9, 1.0, 1.0, 0.82, 0.42], up=(0, 0, 1))
        for c in cyls:      # a short stub from each exhaust port into the shell
            g.tube((c['x'], y, -109), (c['x'], y, z + 30), 23, 19.5, n=18)
            g.at(basis((c['x'], y, -109), (1, 0, 0), (0, -1, 0))).prism(rounded_rect(80, 58, 12, 4), 0, 7, holes=[circle(19.5, 18)])
            g.at()
        g.tube((xb - 10, y, z), (xb - 62, y, z), 27, 23.5, n=20)
        # heater air outlet at the front end
        g.sweep([(xa - 6, y - bank * 10, z + 20), (xa + 4, y - bank * 14, z + 70), (xa + 10, y - bank * 16, z + 150)],
                rounded_rect(70, 50, 14, 4), scales=[1.0, 0.8, 0.7], up=(1, 0, 0))
        E.add('HeatExchanger_' + tag, g, 'black_paint', 'exhaust', 'Heat exchanger, %s: exhaust pipes inside a heater shell' % ('left' if bank > 0 else 'right'),
              ex=(0, bank * 80, -260), sharp=40)
        outlets[bank] = V((xb - 62, y, z))
        for c in cyls:
            for dx in (-29, 29):
                nsrc = E.fastener(nsrc, 'ExhaustFlangeNut_%d%s' % (c['n'], 'a' if dx < 0 else 'b'), nut, (c['x'] + dx, y, -116), mat='copper',
                                  label='Exhaust flange nut', group='exhaust', ex=(0, bank * 80, -300))
            head = bank_m(bank, c['x'])
            seat = S.VALVE['ex']['seat']
            run = [head @ V((0, seat[0] + 6, seat[1] - 4)), head @ V((0, 232, -70)), V((c['x'], y, -109)), V((c['x'], y, z)), V((c['x'] - 60, y, z))]
            E.tmp_ex[c['n']] = run + [V((x, y, z)) for x in range(int(c['x'] - 120), int(xb - 62), -60)] + [outlets[bank]]
    ty, tz = TURBO['y'], TURBO['z']
    tin, tdir = E.tmp_turbine_in, E.tmp_turbine_dir
    # left exchanger straight into the turbine's inlet pipe; the right one crosses over behind the crankcase
    left = [outlets[+1], V((-330, 238, -172)), V((-420, 232, -192)), tin - tdir * 70 + V((0, 0, -6)), tin]
    lpath = fillet_path(left, [70, 60, 50])
    g = Geo()
    g.pipe(lpath, 30, 26, n=20)
    g.at(basis(tin, tdir.cross(V((1, 0, 0))), (1, 0, 0))).prism(rounded_rect(84, 70, 8, 3), -9, 0, holes=[circle(26, 20)])
    g.at()
    wg_branch = [V((-384, 150, -207)), V((-400, 60, -200)), V((-420, 20, -190)), V((-420, 20, -176))]
    bpath = fillet_path(wg_branch, [40, 26])
    g.pipe(bpath, 22, 18.5, n=16)
    E.add('Turbine_Inlet_Pipe', g, 'exhaust', 'exhaust', 'Pipe to the turbine, with the wastegate branch', ex=(-200, 120, -330))
    cross = [outlets[-1], V((-352, -238, -176)), V((-384, -200, -207)), V((-384, 150, -207)), V((-400, 215, -198)), lpath[len(lpath) // 2]]
    cpath = fillet_path(cross, [60, 44, 44, 30])
    g = Geo()
    g.pipe(cpath, 28, 24.5, n=20)
    bel = cpath[len(cpath) // 2]
    for k in range(5):          # flexible bellows section [P]
        g.torus(bel + V((0, -24 + k * 12, 0)), (0, 1, 0), 28, 4, n=20, m=6)
    E.add('Crossover_Pipe', g, 'exhaust', 'exhaust', 'Crossover pipe from the right bank', ex=(-160, -60, -360))
    for c in S.CYLS:
        tail = (lpath if c['bank'] > 0 else cpath + lpath[len(lpath) // 2:])
        gas(E, 'exhaust_%d' % c['n'], 'exhaust', E.tmp_ex[c['n']] + tail + [V((TURBO['x_turb'], ty, tz))], cyl=c['n'], r=17)
    # wastegate: poppet valve under a spring and diaphragm, standing on the branch
    wb = V((-420, 20, -176))
    g = Geo()
    g.at(frame(wb, z=(0, 0, 1), x=(1, 0, 0)))
    g.lathe([(18.5, 0), (34, 0), (34, 34), (20, 40), (20, 52), (56, 58), (56, 74), (48, 88), (22, 94), (22, 100), (10, 100), (10, 44), (18.5, 36)], n=36, closed=True)
    for k in range(8):
        a = TAU * k / 8
        g.bolt((50 * math.cos(a), 50 * math.sin(a), 74), (0, 0, 1), 6.0)
    g.at()
    g.tube(wb + V((-30, 0, 20)), wb + V((-62, 0, 20)), 21, 17, n=18)
    E.add('Wastegate_Body', g, 'iron_cast', 'exhaust', 'Wastegate: opens at 0.8 bar boost', ex=(-230, -40, -200), sharp=40)
    g = Geo()
    g.at(frame((0, 0, 0), z=(0, 0, 1), x=(1, 0, 0)))
    g.lathe([(0, -2), (22, -2), (22, 2), (6, 6), (5, 60), (36, 60), (36, 64), (0, 64)], n=24)
    g.helix((0, 0, 8), (0, 0, 58), 14, 2.4, 5, n=12, m=5)
    g.at()
    E.add('Wastegate_Valve', g, 'steel', 'exhaust', 'Wastegate valve, spring and diaphragm plate', cut=False, loc=wb + V((0, 0, 4)),
          ex=(-230, -40, -80), param=dict(p='wastegate', move=[0, 0, 13]))
    # muffler: one can across the car, twin outlets on the left
    mx, mz = MUFFLER['x'], MUFFLER['z']
    y0, y1 = MUFFLER['y0'], MUFFLER['y1']
    g = Geo()
    ys = [y0, y0 + 16, y0 + 50, y1 - 50, y1 - 16, y1]
    g.sweep([(mx, y, mz) for y in ys], [(p[1], p[0]) for p in rounded_rect(186, 170, 70, 8)], scales=[0.62, 0.88, 1.0, 1.0, 0.88, 0.62], up=(0, 0, 1))
    E.add('Muffler', g, 'black_paint', 'exhaust', 'Muffler', ex=(-380, 0, -200), sharp=40)
    out_path = fillet_path([V((TURBO['x_out'], ty, tz)), V((TURBO['x_out'] - 40, ty, tz)), V((mx + 10, ty, mz + 84)), V((mx + 10, ty, mz + 60))], [40, 30])
    g = Geo()
    g.pipe(out_path, 36, 32, n=22)
    g.at(frame(V((TURBO['x_out'], ty, tz)), z=(-1, 0, 0), x=(0, 1, 0))).prism(circle(58, 28), 0, 9, holes=[circle(32, 24)])
    g.at()
    E.add('Turbine_Outlet_Pipe', g, 'exhaust', 'exhaust', 'Pipe from the turbine to the muffler', ex=(-380, 190, -120))
    wpath = fillet_path([wb + V((-62, 0, 20)), V((-520, 20, -156)), V((mx + 60, 20, mz + 20))], 40)
    g = Geo()
    g.pipe(wpath, 21, 17.5, n=16)
    E.add('Wastegate_Pipe', g, 'exhaust', 'exhaust', 'Wastegate pipe into the muffler', ex=(-300, -40, -260))
    tails = {}
    for nm, yy in (('L', y1 - 8), ('R', y1 - 54)):
        tp = fillet_path([V((mx - 40, yy, mz - 60)), V((mx - 70, yy, mz - 78)), V((mx - 150, yy, mz - 82))], 30)
        g = Geo()
        g.pipe(tp, 22.25, 19.5, n=18)
        E.add('Tailpipe_' + nm, g, 'exhaust', 'exhaust', 'Tailpipe, 44.5 mm: %s' % ('wastegate gas only' if nm == 'L' else 'turbine exhaust'), ex=(-470, 60, -220))
        tails[nm] = tp
    g = Geo()
    g.at(frame((mx, 60, mz), z=(0, 1, 0), x=(1, 0, 0)))
    ring = [(p[0] * 1.03, p[1] * 1.03) for p in rounded_rect(186, 170, 70, 8)]
    g.prism(ring, 0, 22, holes=[[(p[0] * 0.985, p[1] * 0.985) for p in rounded_rect(186, 170, 70, 8)]])
    g.at()
    E.add('Muffler_Strap', g, 'steel', 'exhaust', 'Muffler strap', cut=False, ex=(-380, 0, -230))
    gas(E, 'turbine_out', 'exhaust_out', [V((TURBO['x_turb'], ty, tz))] + out_path + [V((mx, ty, mz)), V((mx, y1 - 80, mz - 20))] + tails['R'], r=24)
    gas(E, 'wastegate', 'wastegate', [lpath[len(lpath) // 2], V((-384, 150, -207))] + bpath + [wb + V((0, 0, 20))] + wpath +
        [V((mx, 120, mz)), V((mx, y1 - 60, mz - 20))] + tails['L'], r=14)


# ---------------------------------------------------------------------------------------------------------------
# ignition and the air pump
# ---------------------------------------------------------------------------------------------------------------
def build_ignition(E):
    """Distributor at the left rear, coil on the shroud and the seven high-tension leads.  Takes the Engine.
    Returns nothing."""
    base = V((S.X_DIST_GEAR, 62, 44))
    cap_c = V((S.X_DIST_GEAR, 150, 124))
    ax = (cap_c - base).normalized()
    fr = frame(base, z=ax, x=(1, 0, 0))
    L = (cap_c - base).length
    g = Geo()
    g.at(fr).lathe([(0, -30), (13, -30), (13, 8), (24, 14), (24, L - 46), (44, L - 40), (44, L - 16), (0, L - 16)], n=32)
    g.at(fr).box((0, 50, L - 30), (22, 22, 30), bevel=3)
    g.at()
    E.add('Distributor_Body', g, 'alu_cast', 'ignition', 'Distributor', ex=(-90, 190, 150))
    g = Geo()
    g.at(fr).lathe([(40, L - 16), (46, L - 16), (46, L + 10), (36, L + 28), (0, L + 30), (0, L + 24), (32, L + 22), (40, L + 8)], n=32, closed=False)
    towers = []
    for k in range(6):
        a = TAU * k / 6 + 0.3
        p = V((27 * math.cos(a), 27 * math.sin(a), L + 24))
        g.at(fr).cyl(p, p + V((0, 0, 20)), 7.5, n=12)
        towers.append(fr @ (p + V((0, 0, 20))))
    g.at(fr).cyl((0, 0, L + 26), (0, 0, L + 48), 8, n=12)
    centre_tower = fr @ V((0, 0, L + 48))
    g.at()
    E.add('Distributor_Cap', g, 'cap_orange', 'ignition', 'Distributor cap', ex=(-110, 240, 210))
    g = Geo()
    g.cyl((0, 0, 0), (0, 0, 14), 9, n=12)
    g.box((14, 0, 12), (34, 12, 5), bevel=1.5)
    E.add('Distributor_Rotor', g, 'plastic_brown', 'ignition', 'Distributor rotor, half crank speed', cut=False, loc=fr @ V((0, 0, L - 6)),
          rot=fr.to_quaternion(), ex=(-100, 215, 180), spin=(ax.x, ax.y, ax.z, 0.5))
    coil_c = V((-212, 96, 300))
    g = Geo()
    g.cyl(coil_c, coil_c + V((0, 0, 92)), 27, n=24)
    g.cyl(coil_c + V((0, 0, 92)), coil_c + V((0, 0, 112)), 14, n=16)
    g.tube(coil_c + V((0, 0, 28)), coil_c + V((0, 0, 44)), 30, 27, n=24)
    E.add('Ignition_Coil', g, 'black_paint', 'ignition', 'Ignition coil', ex=(-60, 60, 400))
    pipe_part(E, 'HT_Lead_Coil', [coil_c + V((0, 0, 112)), coil_c + V((10, 40, 130)), centre_tower + ax * 40 + V((0, 0, 30)), centre_tower], 3.6, 'lead_black',
              'ignition', 'Coil lead', n=8, smooth=7, cut=False, ex=(-90, 190, 260))
    # the cap serves the cylinders in firing order round its towers
    for k, n in enumerate(S.FIRING_ORDER):
        c = next(c for c in S.CYLS if c['n'] == n)
        b = c['bank']
        pt, pd = E.tmp_plug
        plug = bank_m(b, c['x']) @ (pt + pd * 108)
        lift = towers[k] + ax * 34
        if b > 0:
            pts = [towers[k], lift, V((-150, 268, 150)), V((plug.x - 40, 272, 142)), plug + V((-8, 6, 34)), plug]
        else:
            pts = [towers[k], lift, V((-196, 40, 108)), V((-196, -150, 106)), V((-170, -268, 146)), V((plug.x + 40, -272, 142)), plug + V((8, -6, 34)), plug]
        pipe_part(E, 'HT_Lead_%d' % n, pts, 3.6, 'lead_black', 'ignition', 'Plug lead, cylinder %d' % n, n=8, smooth=7, cut=False, ex=(0, b * 120, 240))


def build_air_pump(E):
    """Air injection pump at the left front with its toothed belt from the left camshaft.  Takes the Engine.
    Returns nothing."""
    xf = bank_span(+1)[0]
    xb = xf + 9
    pc = V((xb, 212, 262))
    g = Geo()
    g.cyl(pc + V((-92, 0, 0)), pc + V((-14, 0, 0)), 52, n=32)
    g.cyl(pc + V((-14, 0, 0)), pc + V((-6, 0, 0)), 30, n=24)
    g.cyl(pc + V((-70, 0, 52)), pc + V((-70, 0, 74)), 14, n=14)
    g.box(pc + V((-52, -6, -62)), (56, 60, 22), bevel=4)
    E.add('Air_Pump', g, 'alu_cast', 'induction', 'Air injection pump', ex=(150, 200, 330))
    g = Geo()
    g.at(frame(pc, z=(1, 0, 0), x=(0, 1, 0)))
    g.lathe([(0, -6), (33, -6), (36, -4), (36, 4), (33, 6), (0, 6)], n=32)
    for k in range(3):
        a = TAU * k / 3
        g.bolt((16 * math.cos(a), 16 * math.sin(a), 6), (0, 0, 1), 6.0)
    g.at()
    E.add('Air_Pump_Pulley', g, 'alu_mach', 'induction', 'Air pump pulley', cut=False, ex=(180, 200, 330))
    g = Geo()
    g.at(frame((xb, S.CAM_U, 0), z=(1, 0, 0), x=(0, 1, 0)))
    g.lathe([(0, -6), (25, -6), (28, -4), (28, 4), (25, 6), (0, 6)], n=28)
    g.at()
    E.add('Cam_Pulley_L', g, 'alu_mach', 'induction', 'Belt pulley on the left camshaft', cut=False, ex=(160, 330, 0))
    loop = hull_of([((pc.y, pc.z), 37.0), ((S.CAM_U, 0.0), 29.0)], n=32)
    g = Geo()
    g.sweep([(xb, p[0], p[1]) for p in loop], [(-1.2, -5), (1.2, -5), (1.2, 5), (-1.2, 5)], closed_path=True)
    E.add('Air_Pump_Belt', g, 'rubber', 'induction', 'Toothed belt to the air pump', cut=False, ex=(190, 260, 160))


def build_all(E):
    """Build the dress in list order.  Takes the Engine.  Returns nothing."""
    E.tmp_inj, E.tmp_ex = {}, {}
    build_cooling(E)
    build_oil(E)
    build_mounts(E)
    build_manifold(E)
    build_intercooler(E)
    build_airbox(E)
    build_turbo(E)
    build_exhaust(E)
    build_ignition(E)
    build_air_pump(E)
