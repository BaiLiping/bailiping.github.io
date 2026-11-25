# spec.py: every dimension and every motion law of the model, with no Blender dependency.
#
# Units are millimetres and degrees.  Axes: +X toward the front of the car (flywheel end), +Y the car's left
# (cylinders 1 to 3), +Z up.  Origin on the crank axis at the centre of main bearing 4.
# Tags after a value say where it comes from: [S] a source listed in PARTS.md, [D] derived, [P] read off a
# photograph, [E] an estimate chosen to fit.  PARTS.md explains each choice.
import math

# ---------------------------------------------------------------------------------------------------------------
# crank train
# ---------------------------------------------------------------------------------------------------------------
COMPRESSION = 7.0
CROWN_PROFILE = [(0, -6.5), (27, -6.5), (36, 0)]  # radius and height relative to the edge
FIN_LIMIT = 68.0
SHROUD_MATERIAL = 'black_grp'
BORE = 97.0                 # [S]
STROKE = 74.4               # [S]
R_CRANK = STROKE / 2.0      # 37.2 [D]
ROD = 127.0                 # centre to centre [S]
COMP_H = 32.8               # piston pin to crown edge [S]
PIN_D = 23.0                # piston pin [S]
MAIN_D = 60.0               # main journals 1 to 7 [S, sources conflict, see PARTS.md]
PIN_J_D = 55.0              # rod journals [S]
PIN_J_W = 22.0              # rod journal width [S]
ROD_BIG_D = 58.0            # big-end bore [S]
BORE_PITCH = 118.0          # [S]
STAGGER = BORE_PITCH / 2.0  # left bank sits this far ahead of the right [D]
DECK = 112.5                # crank axis to cylinder base seat [S, forum measurement]
CYL_H = 85.4                # cylinder base seat to head face [D] = 197.9 - 112.5
HEAD_FACE = DECK + CYL_H    # 197.9: crank axis to head sealing face
FIRING_ORDER = (1, 6, 2, 4, 3, 5)   # [S]

# Cylinder table: number, bank (+1 left, -1 right), axis position along X, crank pin angle at theta = 0 measured
# from +Y toward +Z, and the crank angle of its firing top dead centre.
CYLS = [
    dict(n=1, bank=+1, x=+147.5, alpha=0.0, fire=0.0),
    dict(n=2, bank=+1, x=+29.5, alpha=120.0, fire=240.0),
    dict(n=3, bank=+1, x=-88.5, alpha=240.0, fire=480.0),
    dict(n=4, bank=-1, x=+88.5, alpha=180.0, fire=360.0),
    dict(n=5, bank=-1, x=-29.5, alpha=300.0, fire=600.0),
    dict(n=6, bank=-1, x=-147.5, alpha=60.0, fire=120.0),
]
MAIN_X = [177.0, 118.0, 59.0, 0.0, -59.0, -118.0, -177.0]   # mains 1 to 7 [D]
MAIN_W = 22.0
WEB_T = 7.5                 # crank web thickness [D] from 59 mm pin pitch, 22 mm pin and 22 mm main
X_FLANGE = 232.0            # flywheel flange face [E]
X_TIMING_GEAR = -198.0      # crank timing gear, 28 teeth [E]
X_DIST_GEAR = -221.0        # distributor drive gear [E]
X_MAIN8 = -268.0            # nose bearing, in the rear wall of the case [E]
X_PULLEY = -300.0           # belt plane [E]


def piston_s(beta_deg):
    """Distance from the crank axis to the piston pin.  Takes the crank pin's angle from the cylinder axis in
    degrees.  Returns millimetres."""
    b = math.radians(beta_deg)
    return R_CRANK * math.cos(b) + math.sqrt(ROD * ROD - (R_CRANK * math.sin(b)) ** 2)


def beta(cyl, theta):
    """Angle of a cylinder's crank pin from its own cylinder axis.  Takes a CYLS row and the crank angle.
    Returns degrees."""
    return cyl['alpha'] + theta - (0.0 if cyl['bank'] > 0 else 180.0)


# ---------------------------------------------------------------------------------------------------------------
# timing drive
# ---------------------------------------------------------------------------------------------------------------
Z_CRANK_GEAR, Z_IMS_GEAR = 28, 48       # [S secondary]
Z_IMS_SPROCKET, Z_CAM_SPROCKET, Z_IDLER = 24, 28, 19
IMS_DROP = 84.0             # intermediate shaft below the crank [S secondary]
CHAIN_PITCH = 9.525         # 3/8 in duplex [M, standard for this chain]
CHAIN_LINKS = 98            # per chain [S]
CHAIN_ROLLER = 6.35
CHAIN_ROW = 10.24           # distance between the two rows of a duplex chain
X_CHAIN = {+1: -221.0, -1: -248.0}      # chain plane per bank [E]; the left chain is the forward one


def pitch_radius(teeth):
    """Pitch radius of a chain sprocket.  Takes the tooth count.  Returns millimetres."""
    return CHAIN_PITCH / (2.0 * math.sin(math.pi / teeth))


# ---------------------------------------------------------------------------------------------------------------
# valve gear, in the head frame: u outward along the cylinder axis from the crank axis, w up
# ---------------------------------------------------------------------------------------------------------------
VALVE = {
    'in': dict(head_d=49.0, stem_d=9.0, length=110.1, angle=25.5, seat=(212.0, 23.0), lift=9.35),      # [S]
    'ex': dict(head_d=41.5, stem_d=9.0, length=108.4, angle=-30.25, seat=(212.0, -21.0), lift=8.61),   # [S]
}
ROCKER_RATIO = 1.4          # [S]
HEAD_H = 85.0               # head sealing face to cam housing face [M]
CAM_U = 329.5               # camshaft axis from the crank axis [D]: a 98-link chain needs 340 mm between centres
CAM_BASE_R = 16.5           # [E]
CAM_JOURNAL_R = 24.45       # 49 mm housing bore [S]
ROCKER_SHAFT_D = 18.0       # [S]
SPRING_H = 41.0             # installed height [S]
SPRING_FROM_TIP = 6.0       # retainer face below the stem tip [E]
# Timing at 1 mm valve lift [S]: intake opens 3 after TDC and closes 37 after BDC, exhaust opens 27 before BDC
# and closes 5 before TDC.  Durations 214 and 202 degrees, lobe centres 108 apart.
IN_OPEN_ATDC, IN_CLOSE_ABDC = 3.0, 37.0
EX_OPEN_BBDC, EX_CLOSE_BTDC = 27.0, 5.0
CHECK_LIFT = 1.0


def _valve_window(kind):
    """Opening window of a valve relative to its cylinder's firing TDC.  Takes 'in' or 'ex'.  Returns
    (centre, half_duration_at_1mm) in crank degrees."""
    if kind == 'in':
        a, b = 360.0 + IN_OPEN_ATDC, 540.0 + IN_CLOSE_ABDC
    else:
        a, b = 180.0 - EX_OPEN_BBDC, 360.0 - EX_CLOSE_BTDC
    return (a + b) / 2.0, (b - a) / 2.0


def _full_half(kind):
    """Half-width of the whole lift event, seat to seat.  Takes 'in' or 'ex'.  Returns crank degrees.  The lift
    law is a raised cosine, so the half-width follows from where it passes the 1 mm checking lift."""
    _, half = _valve_window(kind)
    L = VALVE[kind]['lift']
    return math.pi * half / math.acos(2.0 * CHECK_LIFT / L - 1.0)


def valve_lift(kind, cyl, theta):
    """Lift of one valve.  Takes 'in' or 'ex', a CYLS row and the crank angle in degrees.  Returns millimetres.
    Raised-cosine event centred on the window from the published timing; the shape between the checking
    points is a placeholder, not the factory lift curve."""
    centre, _ = _valve_window(kind)
    H = _full_half(kind)
    d = ((theta - cyl['fire'] - centre + 360.0) % 720.0) - 360.0
    if abs(d) >= H:
        return 0.0
    return VALVE[kind]['lift'] * 0.5 * (1.0 + math.cos(math.pi * d / H))


def valve_axis(kind):
    """Unit direction of a valve stem in the head frame, from head to tip.  Takes 'in' or 'ex'.  Returns (u, w)."""
    a = math.radians(VALVE[kind]['angle'])
    return math.cos(a), math.sin(a)


def valve_tip(kind):
    """Stem tip position with the valve closed.  Takes 'in' or 'ex'.  Returns (u, w)."""
    v = VALVE[kind]
    au, aw = valve_axis(kind)
    return v['seat'][0] + au * v['length'], v['seat'][1] + aw * v['length']


def rocker_geom(kind):
    """Solve one rocker's layout so the pad arm is square to the cam's push and the valve arm is square to the
    stem.  Takes 'in' or 'ex'.  Returns a dict: shaft (u, w), pad (u, w) on the cam base circle, the pad
    arm b, the valve arm lam, and pad_angle, the direction from the cam axis to the pad in degrees."""
    sign = 1.0 if kind == 'in' else -1.0
    tu, tw = valve_tip(kind)
    au, aw = valve_axis(kind)
    tw, aw = tw * sign, aw * sign                   # work in the mirrored half-plane w > 0
    mu, mw = -aw, au                                # square to the stem, pointing up and inward
    lo, hi = 20.0, 60.0                             # bisection on the pad arm b
    for _ in range(60):
        b = 0.5 * (lo + hi)
        lam = ROCKER_RATIO * b
        su, sw = tu - lam * mu, tw - lam * mw
        f = (su - CAM_U) ** 2 + sw ** 2 - (CAM_BASE_R ** 2 + b ** 2)
        lo, hi = (lo, b) if f < 0 else (b, hi)
    lam = ROCKER_RATIO * b
    su, sw = tu - lam * mu, tw - lam * mw
    cs = math.atan2(sw, su - CAM_U)
    d_ang = cs - math.acos(CAM_BASE_R / math.hypot(su - CAM_U, sw))
    pu, pw = CAM_U + CAM_BASE_R * math.cos(d_ang), CAM_BASE_R * math.sin(d_ang)
    return dict(shaft=(su, sw * sign), pad=(pu, pw * sign), b=b, lam=lam, pad_angle=math.degrees(d_ang) * sign)


def cam_rho(bank, theta):
    """Camshaft rotation in its bank's head frame.  Takes the bank sign and the crank angle.  Returns degrees
    about the frame's x axis.  The gear pair reverses the crank's sense and the chain keeps it, so the cams
    turn against the crank at half speed; the right bank's frame is turned half a turn about Z."""
    return -0.5 * theta * bank


def lobe_nose_angle(kind, cyl):
    """Where a lobe's nose sits on its camshaft, as an angle in the cam's own frame.  Takes 'in' or 'ex' and a
    CYLS row.  Returns degrees.  The nose faces the rocker pad at the centre of the valve's event."""
    centre, _ = _valve_window(kind)
    g = rocker_geom(kind)
    return g['pad_angle'] - cam_rho(cyl['bank'], cyl['fire'] + centre)


def lobe_radius(kind, psi_from_nose):
    """Cam lobe radius.  Takes 'in' or 'ex' and the angle from the nose in cam degrees.  Returns millimetres."""
    H = _full_half(kind) / 2.0
    lift = VALVE[kind]['lift'] / ROCKER_RATIO
    d = ((psi_from_nose + 180.0) % 360.0) - 180.0
    if abs(d) >= H:
        return CAM_BASE_R
    return CAM_BASE_R + lift * 0.5 * (1.0 + math.cos(math.pi * d / H))


def rocker_angle(kind, cyl, theta):
    """Rocker rotation about its shaft in the head frame.  Takes 'in' or 'ex', a CYLS row and the crank angle.
    Returns degrees about the frame's x axis (positive turns u toward w)."""
    g = rocker_geom(kind)
    a = math.degrees(valve_lift(kind, cyl, theta) / g['lam'])
    return a if kind == 'in' else -a


# ---------------------------------------------------------------------------------------------------------------
# cooling, auxiliaries
# ---------------------------------------------------------------------------------------------------------------
FAN_D = 245.0               # [S]
FAN_BLADES = 11             # [S]
FAN_RATIO = 1.67            # fan over crank speed [S, workshop manual]
FAN_PULLEY_D = 82.0         # [S]
CRANK_PULLEY_D = FAN_PULLEY_D * FAN_RATIO   # 137 [D]
BELT_LEN = 710.0            # [S]


def fan_height():
    """Fan axis height above the crank axis from the belt length and the two pulleys.  Returns millimetres."""
    D, d = CRANK_PULLEY_D, FAN_PULLEY_D
    c = 170.0
    for _ in range(40):
        c = (BELT_LEN - math.pi * (D + d) / 2.0 - (D - d) ** 2 / (4.0 * c)) / 2.0
    return c


FAN_Z = fan_height()        # about 181 [D]
FAN_X = (-287.0, -253.0)    # blade row along X [E]

# Turbocharger [S for the wheels, from the later K27-7006; the 930/60's own unit is not confirmed], [P] for place
TURBO_AXIS_Y, TURBO_AXIS_Z = 170.0, -78.0
COMP_IND_D, COMP_EXD_D, COMP_BLADES = 52.5, 76.0, 12        # 6 full and 6 splitter blades
TURB_IND_D, TURB_EXD_D, TURB_BLADES = 76.1, 64.0, 12
BOOST_BAR = 0.8             # [S]
BOOST_RPM = 3500.0          # where full boost arrives [S]
REDLINE = 6700.0            # [S]
IDLE = 900.0                # [S]
POWER_PS, POWER_RPM = 300.0, 5500.0
TORQUE_NM, TORQUE_RPM = 412.0, 4000.0


# ---------------------------------------------------------------------------------------------------------------
# chain path
# ---------------------------------------------------------------------------------------------------------------
def _hull(pts):
    """Convex hull by monotone chain.  Takes [(y, z), ...].  Returns the hull counter-clockwise."""
    pts = sorted(set(pts))
    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) <= 0:
                h.pop()
            h.append(p)
        return h
    lo, up = half(pts), half(reversed(pts))
    return lo[:-1] + up[:-1]


def _loop(bank, h, n=720):
    """Chain centreline round the three sprockets for a given idler offset.  Takes the bank sign, the idler
    centre's offset from the line between shafts and the samples per circle.  Returns (hull points, length,
    idler centre)."""
    a, b = (0.0, -IMS_DROP), (bank * CAM_U, 0.0)
    dy, dz = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dy, dz)
    ny, nz = -dz / L, dy / L
    if nz < 0:
        ny, nz = -ny, -nz
    side = 1.0 if bank > 0 else -1.0        # slack run is on top on the left bank, underneath on the right
    c = (a[0] + dy * 0.45 + ny * h * side, a[1] + dz * 0.45 + nz * h * side)
    pts = []
    for (cy, cz), r in ((a, pitch_radius(Z_IMS_SPROCKET)), (b, pitch_radius(Z_CAM_SPROCKET)), (c, pitch_radius(Z_IDLER))):
        for i in range(n):
            t = 2.0 * math.pi * i / n
            pts.append((round(cy + r * math.cos(t), 6), round(cz + r * math.sin(t), 6)))
    hull = _hull(pts)
    length = sum(math.hypot(hull[i][0] - hull[i - 1][0], hull[i][1] - hull[i - 1][1]) for i in range(len(hull)))
    return hull, length, c


def chain_hull(bank, pad):
    """Outline that clears the chain by a margin, for the chain housing.  Takes the bank sign and the margin.
    Returns [(y, z), ...] counter-clockwise."""
    c = chain_path(bank)['idler']
    a, b = (0.0, -IMS_DROP), (bank * CAM_U, 0.0)
    pts = []
    for (cy, cz), r in ((a, pitch_radius(Z_IMS_SPROCKET)), (b, pitch_radius(Z_CAM_SPROCKET)), (c, pitch_radius(Z_IDLER))):
        for i in range(48):
            t = 2.0 * math.pi * i / 48
            pts.append((round(cy + (r + pad) * math.cos(t), 6), round(cz + (r + pad) * math.sin(t), 6)))
    return _hull(pts)


def chain_pins(bank, theta=0.0):
    """Positions of the chain's pins.  Takes the bank sign and the crank angle.  Returns [(y, z), ...], one per
    link, in the order the chain travels.  The chain advances 14 pitches per crank turn."""
    path = chain_path(bank)
    pts = path['pts']
    seg = []
    total = 0.0
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        seg.append((total, d, a, b))
        total += d
    phase = (theta / 360.0) * Z_IMS_SPROCKET * Z_CRANK_GEAR / Z_IMS_GEAR
    out = []
    k = 0
    for n in range(CHAIN_LINKS):
        s = ((n + phase) * CHAIN_PITCH) % total
        while not (seg[k][0] <= s < seg[k][0] + seg[k][1]):
            k = (k + 1) % len(seg)
        s0, d, a, b = seg[k]
        f = (s - s0) / d
        out.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
    return out


def chain_path(bank):
    """Closed chain centreline whose length is exactly CHAIN_LINKS pitches.  Takes the bank sign.  Returns a
    dict: pts [(y, z), ...] ordered in the direction the chain travels, idler (y, z) and length."""
    target = CHAIN_LINKS * CHAIN_PITCH
    lo, hi = -40.0, 80.0
    for _ in range(50):
        h = 0.5 * (lo + hi)
        _, length, _ = _loop(bank, h)
        lo, hi = (h, hi) if length < target else (lo, h)
    hull, length, c = _loop(bank, h)
    # The intermediate shaft turns against the crank, counter-clockwise seen from the rear, which is clockwise
    # in a (y, z) plot with +y to the right.  The hull is counter-clockwise there, so travel is the reverse.
    return dict(pts=list(reversed(hull)), idler=c, length=length)


def use_1978_sc():
    """Select the naturally aspirated 930/03 dimensions before loading the builders (see NOTES.md)."""
    global BORE, STROKE, R_CRANK, COMP_H, ROD, PIN_D, COMPRESSION, CROWN_PROFILE, FIN_LIMIT
    global POWER_PS, POWER_RPM, TORQUE_NM, TORQUE_RPM, BOOST_BAR, BOOST_RPM, REDLINE
    global IN_OPEN_ATDC, IN_CLOSE_ABDC, EX_OPEN_BBDC, EX_CLOSE_BTDC
    global FAN_D, FAN_RATIO, CRANK_PULLEY_D, BELT_LEN, FAN_Z, SHROUD_MATERIAL, SPRING_H, CAM_BASE_R
    BORE, STROKE, ROD, COMP_H, PIN_D = 95.0, 70.4, 127.8, 34.0, 22.0
    R_CRANK = STROKE / 2
    COMPRESSION = 8.5
    CROWN_PROFILE = [(0, 8), (13, 7.5), (26, 5), (37, 0)]  # schematic dome
    FIN_LIMIT = 79.0
    POWER_PS, POWER_RPM = 180, 5500
    TORQUE_NM, TORQUE_RPM = 265, 4200
    BOOST_BAR = BOOST_RPM = 0
    REDLINE = 6800  # nominal ignition cutoff of the European 930/03
    IN_OPEN_ATDC, IN_CLOSE_ABDC = -1.0, 53.0
    EX_OPEN_BBDC, EX_CLOSE_BTDC = 43.0, -3.0
    VALVE['in']['lift'], VALVE['ex']['lift'] = 11.4, 10.0
    CAM_BASE_R = 15.0
    SPRING_H = 34.5
    FAN_D, FAN_RATIO, BELT_LEN = 226.0, 1.81, 725.0
    CRANK_PULLEY_D = FAN_PULLEY_D * FAN_RATIO
    FAN_Z = fan_height()
    SHROUD_MATERIAL = 'sc_shroud'
