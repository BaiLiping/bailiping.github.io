"""Second-pass details from close-up references (BaT 2024 Raptor R photos 012/015/017/022/031/046, Ford 360 renders):
taillamps with the folded double-Γ light pipes, bed rear closures, hood nose, Power Dome extractor, CHMSL, headlamp internals."""
import numpy as np, bpy
from geo import pchip, resample
from mk import Surf, outline_panel, make_obj, grid_obj, mirror_y, extrude_profile
from parts_front import rrect, XF, sy, KY
import design as D

PAINT = 'Mat_Rapid_Red_Paint'

def circle(cy, cz, r, n=28):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(cy + r * np.cos(t), cz + r * np.sin(t)) for t in a]

def gamma(c_u, z_t, w, r_i, arm_end, z_b, corner_at_min=True, n=8):
    """Γ-shaped light pipe in (u, z): arm along the top from arm_end to the corner at c_u, quarter round, leg down to z_b.
    corner_at_min: the corner is at the smaller u (side face, rear = -x) else at the larger u (rear face, outboard = +y)."""
    r_o = r_i + w; s = 1.0 if corner_at_min else -1.0
    C = (c_u + s * r_o, z_t - r_o)
    pts = [(arm_end, z_t)] + [(C[0] + s * r_o * np.cos(a), C[1] + r_o * np.sin(a)) for a in np.linspace(np.pi / 2, np.pi, n)]
    pts += [(c_u, z_b), (c_u + s * w, z_b)]
    pts += [(C[0] + s * r_i * np.cos(a), C[1] + r_i * np.sin(a)) for a in np.linspace(np.pi, np.pi / 2, n)]
    pts += [(arm_end, z_t - w)]
    return pts

# ------------------------------------------------------------------ taillamps
X_LR = -2.990                                  # rear face of the lamp lens
def XR(y, z): return X_LR + 0.004 * (1 - np.clip((np.abs(y) - 0.83) / 0.10, 0, 1)) - 0.003 * np.clip((z - 1.12) / 0.18, -1, 1) ** 2
def YL(x, z): return D.YP(np.clip(np.asarray(x, float), -2.982, 0)) + 0.004
LAMP_SIDE = [(-2.982, 0.938), (-2.930, 0.938), (-2.890, 0.955), (-2.850, 0.990), (-2.800, 1.060), (-2.760, 1.130),
             (-2.730, 1.200), (-2.716, 1.288), (-2.732, 1.304), (-2.982, 1.304)]
LAMP_REAR = [(0.828, 0.938), (0.926, 0.938), (0.935, 0.952), (0.937, 1.288), (0.930, 1.304), (0.828, 1.304)]

def taillamps(col):
    out = []
    RS = lambda d=0.0: Surf('front', lambda y, z: XR(y, z) - d, sign=-1.0)
    SS = lambda d=0.0: Surf('side', lambda x, z: YL(x, z) + d)
    lens_r = outline_panel('RaptorR_Taillight_Lens_L', LAMP_REAR, RS(), h=0.012, bnd=0.004, hem=0.035, roll=0.008,
                           mat='Mat_Taillight_Ruby_Deep', collection=col)
    lens_s = outline_panel('RaptorR_Taillight_Side_L', LAMP_SIDE, SS(), h=0.012, bnd=0.004, hem=0.030, roll=0.008,
                           mat='Mat_Taillight_Ruby_Deep', collection=col)
    parts = [lens_r, lens_s]
    # light pipes (bright), 1.5 mm proud of the lens
    zt1, w1 = 1.240, 0.020        # outer Γ
    zt2, w2 = 1.205, 0.013        # inner Γ
    side_pipes = [('Top', rrect(-2.976, -2.742, 1.274, 1.286, 0.005)),
                  ('GammaO', gamma(-2.980, zt1, w1, 0.022, -2.756, 1.040, True)),
                  ('GammaI', gamma(-2.951, zt2, w2, 0.016, -2.776, 1.075, True))]
    rear_pipes = [('Top', rrect(0.836, 0.932, 1.274, 1.286, 0.005)),
                  ('GammaO', gamma(0.936, zt1, w1, 0.022, 0.868, 1.040, False)),
                  ('GammaI', gamma(0.907, zt2, w2, 0.016, 0.870, 1.075, False))]
    for nm, P in side_pipes:
        parts.append(outline_panel(f'RaptorR_Taillight_Blade_S{nm}_L', P, SS(0.0015), h=0.01, bnd=0.003, hem=0.004, roll=0.0015,
                                   mat='Mat_Taillight_Amber_Blade', collection=col))
    for nm, P in rear_pipes:
        parts.append(outline_panel(f'RaptorR_Taillight_Blade_R{nm}_L', P, RS(0.0015), h=0.01, bnd=0.003, hem=0.004, roll=0.0015,
                                   mat='Mat_Taillight_Amber_Blade', collection=col))
    # clear reverse strip on the inboard edge, lower clear block and red reflex band
    parts.append(outline_panel('RaptorR_Taillight_Reverse_L', rrect(0.834, 0.858, 0.962, 1.262, 0.006), RS(0.0012), h=0.01, bnd=0.003,
                               hem=0.003, roll=0.0015, mat='Mat_Taillight_Clear_Accent', collection=col))
    parts.append(outline_panel('RaptorR_Taillight_ReverseSide_L', [(-2.936, 0.975), (-2.880, 0.975), (-2.862, 1.000), (-2.936, 1.000)],
                               SS(0.0012), h=0.01, bnd=0.003, hem=0.003, roll=0.0015, mat='Mat_Taillight_Clear_Accent', collection=col))
    parts.append(outline_panel('RaptorR_Taillight_Reflex_L', rrect(0.866, 0.928, 0.952, 1.018, 0.006), RS(0.0010), h=0.01, bnd=0.003,
                               hem=0.003, roll=0.0015, mat='Mat_Taillight_Red_Lens', collection=col))
    for p in parts: out.append((p, mirror_y(p, p.name[:-2] + '_R')))
    return out

def bed_rear_closure(col):
    """rear-facing painted pillar of each bedside around the lamp (tailgate shut line to the body corner)"""
    def XRF(y, z): return -2.966 - 0.012 * np.clip((np.abs(y) - 0.82) / 0.10, 0, 1) - 0.004 * np.clip((z - 1.40) / 0.06, 0, 1)
    outer = [(0.821, 0.814), (0.944, 0.814), (0.952, 0.846), (0.948, 1.330), (0.936, 1.420), (0.912, 1.456), (0.821, 1.456)]
    hole = [(0.826, 0.934), (0.939, 0.934), (0.941, 1.308), (0.826, 1.308)]
    ob = outline_panel('RaptorR_Body_BedPillar_Rear_L', outer, Surf('front', XRF, sign=-1.0), holes=[hole], h=0.015, bnd=0.004,
                       hem=0.028, roll=0.008, hem_holes=False, mat=PAINT, collection=col)
    return ob, mirror_y(ob, 'RaptorR_Body_BedPillar_Rear_R')

# ------------------------------------------------------------------ hood nose (front lip down to the grille top)
def hood_nose(col):
    rows = []
    for y in np.linspace(-0.845, 0.845, 49):
        x0 = 2.640 - 0.050 * (y / 0.85) ** 2; z0 = float(D.Z_HOOD(np.array([x0]), np.array([y]))[0])
        dz = float((D.Z_HOOD(np.array([x0 + 0.005]), np.array([y])) - D.Z_HOOD(np.array([x0 - 0.005]), np.array([y])))[0] / 0.010)
        xg = float(XF(np.array([y]), np.array([1.30]))[0]) - 0.006
        if abs(y) > 0.62 * KY:   # over the lamps the face follows the lens wrap
            xg -= 0.060 * np.clip((abs(y) - 0.70) / 0.22, 0, 1) ** 1.6
        x3, z3 = max(xg, x0 + 0.004), 1.305
        P0 = np.array([x0, z0]); P1 = P0 + np.array([1.0, dz]) / np.hypot(1.0, dz) * 0.035
        P3 = np.array([x3, z3]); P2 = P3 + np.array([0.0, 0.020])
        sec = []
        for t in np.linspace(0, 1, 10):
            b = (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t ** 2 * P2 + t ** 3 * P3
            sec.append((b[0], y, b[1]))
        sec.append((x3 - 0.015, y, z3 - 0.0005))      # short return flange under the lip
        rows.append(sec)
    return grid_obj('RaptorR_Hood_Nose', np.array(rows), mat=PAINT, collection=col,
                    outward=lambda p: np.stack([np.ones(len(p)), np.zeros(len(p)), 0.6 * np.ones(len(p))], 1))

# ------------------------------------------------------------------ Power Dome heat extractor (photo 017)
def hood_extractor(col):
    # Both visible widths and the fore/aft location are constrained by the
    # 2024 side and high-front reference photos. The unit sits over the hood
    # crown. The fixed cowl and the painted Power Dome remain separate.
    out = []
    rear, nose, half_width = 1.594, 1.962, 0.505
    stations = np.linspace(0, 1, 68)
    lateral = np.linspace(-half_width, half_width, 201)
    T, Y = np.meshgrid(stations, lateral, indexing='ij')
    def smooth(x):
        v = np.clip(x, 0, 1)
        return v * v * (3 - 2 * v)
    def tray(t, y):
        # Round the two forward corners of the moulding in plan.
        corner = smooth((np.abs(y) - .438) / .067)
        x = rear + (nose - rear) * t - .035 * corner * t ** 5
        base = D.Z_HOOD(x, y)
        # The moulded tray bridges the dome shoulder out to the side badges.
        deck = D.Z_HOOD(x, np.clip(y, -.392, .392))
        roll = smooth((half_width - np.abs(y)) / .022)
        rim = .004 + .008 * np.sin(np.pi * t) + .009 * smooth((.99 - t) / .15)
        support = base + (deck - base) * roll + rim
        relief = np.zeros_like(y)
        for yc in (-.276, 0., .276):
            offset = np.abs(y - yc)
            # Forward-facing scoops with a raised left/right rim and recess.
            side = smooth((.092 - offset) / .024)
            toe = smooth((.85 - t) / .16)
            shoulder = smooth(t / .12) * toe
            peak = np.clip(1.568 - support, .018, .059)
            cup = smooth((.054 - offset) / .015) * smooth((t - .30) / .18) * smooth((.81 - t) / .16)
            relief = np.maximum(relief, shoulder * side * peak * (1 - .80 * cup))
        return x, support + relief
    X, Z = tray(T, Y)
    P = np.stack([X, Y, Z], axis=-1)
    material = 'Mat_Hood_Extractor_SatinGray'
    out.append(grid_obj('RaptorR_Hood_Extractor', P, mat=material, collection=col, outward=(0, 0, 1)))
    rim = np.concatenate((P[0], P[1:, -1], P[-1, -2::-1], P[-2:0:-1, 0]))
    foot = rim.copy()
    foot[:, 2] = D.Z_HOOD(foot[:, 0], foot[:, 1]) - .002
    out.append(grid_obj('RaptorR_Hood_Extractor_Skirt', np.stack([rim, foot], axis=1),
                        mat=material, collection=col, closed_u=True, outward=lambda p:
                        np.column_stack((p[:, 0] - (rear + nose) / 2, p[:, 1], np.zeros(len(p))))))
    for yc in (-.276, 0., .276):
        for dy in (-.028, -.009, .009, .028):
            y = yc + dy
            x, z = tray(.70, np.array(y))
            out.append(box_at('RaptorR_Hood_Extractor_Slot', float(x), y, float(z) + .001,
                              .013, .012, .002, 'Mat_Grille_Charcoal_Black', col))
    for side in (-1, 1):
        y = side * .455
        x, z = tray(.46, np.array(y))
        out.append(box_at('RaptorR_Hood_Extractor_Badge', float(x), y, float(z) + .003,
                          .079, .016, .004, 'Mat_Gloss_Black_Pillar', col))
    return out

def box_at(name, x, y, z, lx, ly, lz, mat, col):
    P = [(y - ly / 2, z - lz / 2), (y + ly / 2, z - lz / 2), (y + ly / 2, z + lz / 2), (y - ly / 2, z + lz / 2)]
    return extrude_profile(name, P, 'x', lx, origin=(x - lx / 2, 0, 0), mat=mat, collection=col)

# ------------------------------------------------------------------ CHMSL with cargo lamps and camera (photo 046)
def chmsl(col):
    out = []
    rows = []
    for t in np.linspace(-1, 1, 21):
        y = 0.152 * np.sign(t) * abs(t) ** 0.8
        zr = float(D.Z_ROOF(np.array([-1.000]), np.array([y]))[0])
        rows.append([(-1.101, y * 0.985, 1.934), (-1.099, y, 1.992), (-1.084, y * 0.985, 2.020), (-1.050, y * 0.97, 2.031),
                     (-1.000, y * 0.95, zr + 0.001)])
    out.append(grid_obj('RaptorR_Body_CHMSL_Housing', np.array(rows), mat='Mat_Taillight_Clear_Accent', collection=col,
                        outward=lambda p: np.stack([-np.ones(len(p)), np.zeros(len(p)), 0.4 * np.ones(len(p))], 1)))
    RS = Surf('front', lambda y, z: -1.1035 + 0 * y, sign=-1.0)
    for s in (1, -1):
        out.append(outline_panel('RaptorR_Body_CHMSL_Red', rrect(min(s * 0.088, s * 0.140), max(s * 0.088, s * 0.140), 1.960, 1.982, 0.005),
                                 RS, h=0.01, bnd=0.003, hem=0.003, roll=0.001, mat='Mat_CHMSL_Clear_Red', collection=col))
        out.append(outline_panel('RaptorR_Body_CHMSL_Cargo', rrect(min(s * 0.030, s * 0.078), max(s * 0.030, s * 0.078), 1.958, 1.984, 0.004),
                                 RS, h=0.01, bnd=0.003, hem=0.003, roll=0.001, mat='Mat_LED_Headlight_White', collection=col))
    out.append(outline_panel('RaptorR_Body_CHMSL_Camera', circle(0.0, 1.966, 0.011, 20), RS, h=0.01, bnd=0.002, hem=0.003, roll=0.001,
                             mat='Mat_Gloss_Black_Pillar', collection=col))
    return out

# ------------------------------------------------------------------ headlamp internals (photo 012, Ford frames 04/07)
def headlamp_internals(col):
    out = []
    def XL(y, z): return XF(y, z) + 0.004 - 0.060 * np.clip((np.abs(y) - 0.70) / 0.22, 0, 1) ** 1.6
    S = lambda d: Surf('front', lambda y, z: XL(y, z) - d)
    pieces = []
    # thin amber C-clamp DRL: top bar and outboard leg (≈14 mm light guides)
    pieces.append(('RaptorR_Headlight_AmberDRL_Top_L', rrect(0.578, 0.884, 1.263, 1.277, 0.006), 0.010, 'Mat_Amber_LED_DRL', 0.004))
    pieces.append(('RaptorR_Headlight_AmberDRL_Leg_L', [(0.847, 0.990), (0.861, 0.990), (0.868, 1.255), (0.854, 1.255)], 0.010,
                   'Mat_Amber_LED_DRL', 0.004))
    # reflector bowls (dark chrome) around the two projector modules
    pieces.append(('RaptorR_Headlight_Reflector_Upper_L', rrect(0.596, 0.834, 1.128, 1.242, 0.014), 0.034, 'Mat_Headlight_DarkChrome', 0.012))
    pieces.append(('RaptorR_Headlight_Reflector_Lower_L', rrect(0.596, 0.834, 0.972, 1.068, 0.014), 0.034, 'Mat_Headlight_DarkChrome', 0.012))
    # projector lenses and LED emitters
    pieces.append(('RaptorR_Headlight_Projector_Upper_L', circle(0.770, 1.184, 0.034), 0.024, 'Mat_Headlight_Projector_Lens', 0.008))
    pieces.append(('RaptorR_Headlight_Projector_Lower_L', circle(0.770, 1.020, 0.032), 0.024, 'Mat_Headlight_Projector_Lens', 0.008))
    pieces.append(('RaptorR_Headlight_LEDStrip_Upper_L', rrect(0.628, 0.712, 1.168, 1.196, 0.006), 0.026, 'Mat_LED_Headlight_White', 0.004))
    pieces.append(('RaptorR_Headlight_LEDStrip_Lower_L', rrect(0.640, 0.708, 1.004, 1.034, 0.006), 0.026, 'Mat_LED_Headlight_White', 0.004))
    # horizontal blade continuing the grille's middle bar through the lamp
    pieces.append(('RaptorR_Headlight_Blade_L', [(0.566, 1.076), (0.806, 1.084), (0.814, 1.104), (0.566, 1.110)], 0.014,
                   'Mat_Headlight_DarkChrome', 0.010))
    for name, P, d, mat, hem in pieces:
        ob = outline_panel(name, sy(P), S(d), h=0.012, bnd=0.003, hem=hem, roll=0.003, mat=mat, collection=col)
        out.append((ob, mirror_y(ob, name[:-2] + '_R')))
    return out

# ------------------------------------------------------------------ headlamp v3 (Ford frames 04-13 + BaT 012/015)
# plan of the lens: side face (fender corner) -> rounded corner -> front face; rear edge on the side is a straight slant
# from (2.235, 1.27) to (2.43, 0.94) (frame 13 back-projected); the C-clamp DRL sits ~9 cm inside the side lens.
def XLENS(y, z): return XF(y, z) + 0.004 - 0.060 * np.clip((np.abs(y) - 0.70) / 0.22, 0, 1) ** 1.6
LAMP_ZT = 1.292
LAMP_ZB = pchip([2.232, 2.236, 2.300, 2.370, 2.430, 2.470, 2.800], [1.290, 1.268, 1.158, 1.038, 0.950, 0.950, 0.952])
Y_DEEP = 0.868          # inner wall of the side chamber
def lamp_plan():
    from geo import spline
    fy = [0.578, 0.69, 0.79, 0.87]
    front_pts = [(float(XLENS(np.array([y]), np.array([1.12]))[0]), y) for y in fy]
    return spline([(2.232, 0.990), (2.330, 0.987), (2.420, 0.978), (2.500, 0.955), (2.536, 0.920)] + front_pts[::-1], 72)

def tube(name, pts, r, mat, col, n=10):
    P = np.asarray(pts, float); rows = []
    T = np.gradient(P, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True)
    up = np.array([0.0, 1.0, 0.0])
    for p, t in zip(P, T):
        a = np.cross(t, up); a = a / (np.linalg.norm(a) + 1e-9) if np.linalg.norm(a) > 1e-6 else np.array([1.0, 0, 0])
        b = np.cross(t, a)
        rows.append([p + r * (np.cos(th) * a + np.sin(th) * b) for th in np.linspace(0, 2 * np.pi, n, endpoint=False)])
    def away(q):
        d = np.linalg.norm(q[:, None, :] - P[None, :, :], axis=2)
        return q - P[np.argmin(d, axis=1)]
    return grid_obj(name, np.array(rows), mat=mat, collection=col, closed_v=True, outward=away)

def headlamps_v3(col):
    out = []
    plan = lamp_plan()
    rows = []
    for (x, y) in plan:
        z0, z1 = float(LAMP_ZB(min(x, 2.80))), LAMP_ZT
        rows.append([(x, y, z) for z in np.linspace(z0, z1, 16)])
    lens = grid_obj('RaptorR_Headlight_Lens_L', np.array(rows), mat='Mat_Headlight_Clear_Lens', collection=col)
    out.append(lens)
    # deep inner wall of the side chamber (black) and the dark-amber side reflector on it
    side_outline = [(2.236, 1.284), (2.236, 1.268), (2.300, 1.158), (2.370, 1.038), (2.430, 0.950), (2.555, 0.950), (2.555, 1.284)]
    out.append(outline_panel('RaptorR_Headlight_Housing_Side_L', side_outline, Surf('side', lambda x, z: Y_DEEP + 0 * x), h=0.02, bnd=0.006,
                             hem=0.0, hem_outer=False, mat='Mat_Gloss_Black_Pillar', collection=col))
    xe = lambda z: 2.236 + (1.268 - z) * 0.61 + 0.006       # just forward of the lens' rear edge
    xl = lambda z: 2.294 + (1.244 - z) * 0.482 - 0.009      # just behind the DRL leg
    zz = np.linspace(1.240, 1.004, 7)
    refl = [(xe(z), z) for z in zz] + [(xl(z), z) for z in zz[::-1][1:-1]] + [(xl(1.240), 1.240)]
    out.append(outline_panel('RaptorR_Headlight_SideAmber_L', refl, Surf('side', lambda x, z: Y_DEEP + 0.004 + 0 * x), h=0.01, bnd=0.004,
                             hem=0.0, hem_outer=False, mat='Mat_Amber_Side_Reflector', collection=col))
    # ceiling and floor of the side chamber (black) so the lamp reads as a solid housing from above/below
    for nm, zf in (('Top', LAMP_ZT - 0.004), ('Floor', None)):
        rows = []
        for (x, y) in plan:
            if x > 2.556: continue
            z = zf if zf is not None else float(LAMP_ZB(x)) + 0.004
            rows.append([(x, Y_DEEP, z), (x, y - 0.004, z)])
        out.append(grid_obj(f'RaptorR_Headlight_Housing_{nm}_L', np.array(rows), mat='Mat_Gloss_Black_Pillar', collection=col,
                            outward=(0, 0, -1) if nm == 'Top' else (0, 0, 1)))
    # front back-wall (behind the front lens face) with the grille-side bezel
    lens_front = [(0.565, 0.952), (0.860, 0.952), (0.895, 0.985), (0.905, 1.245), (0.885, 1.292), (0.565, 1.292)]
    out.append(outline_panel('RaptorR_Headlight_Housing_L', sy(lens_front), Surf('front', lambda y, z: XLENS(y, z) - 0.050), h=0.03,
                             hem=0.0, hem_outer=False, mat='Mat_Gloss_Black_Pillar', collection=col))
    # The illuminated close-up shows two thick guides in separate bezels.
    # The upright diffuser faces forward around the outer projector corner.
    # The narrow amber reflector farther aft is a different component.
    S = lambda depth: Surf('front', lambda y, z: XLENS(y, z) - depth)
    top = [(0.545, 1.282), (0.872, 1.282), (0.880, 1.275), (0.880, 1.253),
           (0.574, 1.253), (0.561, 1.259)]
    upright = [(0.852, 1.240), (0.879, 1.243), (0.882, 1.232), (0.869, 0.963),
               (0.862, 0.955), (0.838, 0.956), (0.837, 0.966)]
    for tag, outline in (('Top', top), ('Leg', upright)):
        contour = np.array(sy(outline))
        midpoint = (contour.min(axis=0) + contour.max(axis=0)) / 2
        extent = np.ptp(contour, axis=0)
        bezel = midpoint + (contour - midpoint) * (1 + 0.012 / extent)
        out.append(outline_panel(f'RaptorR_Headlight_DiffuserBezel_{tag}_L', bezel, S(0.013),
                                 h=0.010, bnd=0.003, hem=0.010, roll=0.002,
                                 mat='Mat_Headlight_DarkChrome', collection=col))
        out.append(outline_panel(f'RaptorR_Headlight_AmberDRL_{tag}_L', contour, S(0.008),
                                 h=0.009, bnd=0.003, hem=0.005, roll=0.002,
                                 mat='Mat_Amber_LED_DRL', collection=col))
    # reflectors (light satin chrome), projector lenses, LED strips, blade
    # Reflector cups around the projectors: transition from the rounded
    # rectangular opening to the circular projector. The grooved side walls
    # catch different studio reflections as the camera moves around the nose.
    angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    for tag, z0 in (('Upper', 1.185), ('Lower', 1.020)):
        yc = 0.774
        profiles = []
        for width, height, exponent, depth, center_y in (
                (0.087, 0.051, 0.34, 0.014, 0.746),
                (0.082, 0.047, 0.40, 0.018, 0.746),
                (0.063, 0.035, 0.57, 0.040, 0.752),
                (0.034, 0.030, 0.93, 0.038, yc),
                (0.029, 0.026, 1.00, 0.023, yc)):
            y = center_y + width * np.sign(np.cos(angles)) * abs(np.cos(angles)) ** exponent
            z = z0 + height * np.sign(np.sin(angles)) * abs(np.sin(angles)) ** exponent
            y *= KY
            profiles.append(np.column_stack((XLENS(y, z) - depth, y, z)))
        out.append(grid_obj(f'RaptorR_Headlight_Reflector_{tag}_L', np.array(profiles),
                            mat='Mat_Headlight_DarkChrome', collection=col, closed_v=True, outward=(1, 0, 0)))
        profiles = []
        for radius, depth in ((0.029, 0.024), (0.025, 0.020), (0.017, 0.017), (0.003, 0.015)):
            y = yc * KY + radius * np.cos(angles)
            z = z0 + radius * 0.91 * np.sin(angles)
            profiles.append(np.column_stack((XLENS(y, z) - depth, y, z)))
        out.append(grid_obj(f'RaptorR_Headlight_Projector_{tag}_L', np.array(profiles),
                            mat='Mat_Mirror_Chrome_Glass', collection=col, closed_v=True, outward=(1, 0, 0)))
        out.append(outline_panel(f'RaptorR_Headlight_Emitter_{tag}_L', circle(yc * KY, z0, 0.009), S(0.015),
                                 h=0.005, bnd=0.002, hem=0.002, mat='Mat_LED_Headlight_White', collection=col))
    pieces = [('RaptorR_Headlight_Blade_L', [(0.566, 1.076), (0.806, 1.084), (0.814, 1.104), (0.566, 1.110)], 0.014, 'Mat_Headlight_DarkChrome', 0.010),
              ('RaptorR_Headlight_InnerBezel_L', [(0.640, 0.956), (0.662, 0.956), (0.662, 1.246), (0.640, 1.246)], 0.006, 'Mat_Grille_Charcoal_Black', 0.040)]
    for name, P, d, mat, hem in pieces:
        out.append(outline_panel(name, sy(P), S(d), h=0.012, bnd=0.003, hem=hem, roll=0.003, mat=mat, collection=col))
    return [(o, mirror_y(o, o.name[:-2] + '_R')) for o in out]

# ------------------------------------------------------------------ tailgate FORD block letters (BaT 008, plate-scaled)
def ford_letter_polys():
    """the grille's FORD letters (u=y as seen from the front, v=z), outer loops and holes"""
    w, z0, z1 = 0.235, 1.065, 1.152
    def F_(y0):
        t = 0.026
        return [(y0, z0), (y0 + t, z0), (y0 + t, 1.098), (y0 + w * 0.78, 1.098), (y0 + w * 0.80, 1.120), (y0 + t, 1.120),
                (y0 + t, z1 - t), (y0 + w, z1 - t), (y0 + w, z1), (y0, z1)], []
    def O_(y0):
        c = 0.022
        return ([(y0 + c, z0), (y0 + w - c, z0), (y0 + w, z0 + c), (y0 + w, z1 - c), (y0 + w - c, z1), (y0 + c, z1), (y0, z1 - c), (y0, z0 + c)],
                [[(y0 + 0.040, 1.091), (y0 + w - 0.040, 1.091), (y0 + w - 0.040, 1.126), (y0 + 0.040, 1.126)]])
    def R_(y0):
        return ([(y0, z0), (y0 + 0.030, z0), (y0 + 0.030, 1.098), (y0 + 0.140, 1.098), (y0 + 0.205, z0), (y0 + w, z0),
                 (y0 + 0.170, 1.100), (y0 + w - 0.012, 1.104), (y0 + w, 1.118), (y0 + w, z1 - 0.016), (y0 + w - 0.016, z1), (y0, z1)],
                [[(y0 + 0.030, 1.118), (y0 + w - 0.032, 1.118), (y0 + w - 0.032, 1.132), (y0 + 0.030, 1.132)]])
    def Dl(y0):
        c = 0.030
        return ([(y0, z0), (y0 + w - c, z0), (y0 + w, z0 + c), (y0 + w, z1 - c), (y0 + w - c, z1), (y0, z1)],
                [[(y0 + 0.032, 1.091), (y0 + w - 0.040, 1.091), (y0 + w - 0.040, 1.126), (y0 + 0.032, 1.126)]])
    return [fn(y0) for fn, y0 in ((F_, -0.552), (O_, -0.272), (R_, 0.035), (Dl, 0.318))]

def tailgate_letters(col, XT):
    s = 0.102 / 0.087; zc0, zc1 = 1.1085, 1.111
    tf = lambda P: [(-(u * s), zc1 + (v - zc0) * s) for (u, v) in P]     # mirrored so it reads F-O-R-D from behind
    TSurf = Surf('front', lambda y, z: XT(y, z) - 0.016, sign=-1.0)
    out = []
    for outer, holes in ford_letter_polys():
        ob = outline_panel('RaptorR_Tailgate_FORD', tf(outer), TSurf, holes=[tf(h) for h in holes], h=0.02, bnd=0.004, hem=0.012,
                           roll=0.004, mat='Mat_FORD_Letter_Face', collection=col)
        me = ob.data; uvl = me.uv_layers.new(name='UVMap')
        for poly in me.polygons:
            for li in poly.loop_indices:
                co = me.vertices[me.loops[li].vertex_index].co
                uvl.data[li].uv = (-co.y * 6.0, co.z * 6.0)
        out.append(ob)
    # scribed lines on the applique (above the letters, and between the F-150 / RAPTOR R badges)
    for (y0, y1, z) in ((-0.735, 0.735, 1.214), (-0.455, 0.470, 1.022)):
        out.append(outline_panel('RaptorR_Tailgate_Applique_Line', [(y0, z - 0.002), (y1, z - 0.002), (y1, z + 0.002), (y0, z + 0.002)],
                                 Surf('front', lambda y, z: XT(y, z) - 0.0095, sign=-1.0), h=0.01, bnd=0.004, hem=0.002, roll=0.001,
                                 mat='Mat_Grille_Charcoal_Black', collection=col))
    return out

# ------------------------------------------------------------------ body-colour cover under the lamps/grille, one smooth wrap
def front_cover(col):
    plan = [p for p in lamp_plan() if p[0] >= 2.452]                # side end -> corner -> front inner (y 0.578)
    fy = np.linspace(0.578, 0.0, 15)[1:]
    pts = [(p[0], p[1]) for p in plan] + [(float(XLENS(np.array([y]), np.array([0.90]))[0]), y) for y in fy]
    pts = np.array(pts)
    # densify and compute plan normals (outward, in xy)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    s = np.linspace(0, d[-1], 90); P = np.stack([np.interp(s, d, pts[:, 0]), np.interp(s, d, pts[:, 1])], 1)
    T = np.gradient(P, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True)
    N = np.stack([-T[:, 1], T[:, 0]], 1)
    N *= np.sign(np.sum(N * (P - np.array([1.8, 0.0])), axis=1))[:, None]
    w = np.clip((s - 0.0) / 0.10, 0, 1)                             # no offset at the side end so it meets the fender
    ztop = np.where(P[:, 1] > 0.720 * KY, 0.952, np.where(P[:, 1] < 0.682, 0.872, 0.872 + 0.080 * (P[:, 1] - 0.682) / (0.720 * KY - 0.682)))
    rows = []
    for i in range(len(P)):
        row = []
        for t in np.linspace(0, 1, 9):
            z = 0.858 + t * (ztop[i] - 0.858)
            bulge = 0.012 * np.clip((0.95 - z) / 0.08, 0, 1) - 0.004
            q = P[i] + N[i] * bulge * w[i]
            row.append((q[0], q[1], z))
        rows.append(row)
    R = np.array(rows)
    full = np.concatenate([R[::-1][:-1] * np.array([1, -1, 1]), R], axis=0)    # mirror across the centreline
    return grid_obj('RaptorR_Body_Front_Valance', full, mat=PAINT, collection=col,
                    outward=lambda p: np.stack([p[:, 0] - 1.8, p[:, 1], np.zeros(len(p))], 1))

# ------------------------------------------------------------------ inner wheel-well aprons (hide the engine bay / bed void)
def wheel_aprons(col):
    from geo import spline
    out = []
    for tag, inner, zlo, ztop in (('F', D.FLARE_F_IN, 0.50, 0.965), ('Rr', D.FLARE_R_IN, 0.60, 1.20)):
        I = spline(inner, 60)
        hub_x = D.WB2 if tag == 'F' else -D.WB2
        cz = D.HUB_Z + 0.068
        arch = []
        for (xi, zi) in I:
            rx, rz = xi - hub_x, zi - cz; rl = np.hypot(rx, rz)
            arch.append((xi + rx / rl * 0.03, min(zi + rz / rl * 0.03, ztop)))
        arch = [p for p in arch if p[1] >= zlo]
        poly = arch + [(arch[-1][0], zlo), (arch[0][0], zlo)]
        ob = outline_panel(f'RaptorR_Wheelhouse_{tag}_Apron_L', poly, Surf('side', lambda x, z: 0.400 + 0 * x), h=0.03, bnd=0.01,
                           hem=0.0, hem_outer=False, mat='Mat_Trim_Matte_Black', collection=col)
        out.append((ob, mirror_y(ob, f'RaptorR_Wheelhouse_{tag}_Apron_R')))
    return out

# ------------------------------------------------------------------ fender nose over the lamp's outer top corner
def fender_nose(col):
    """continue the fender deck from x 2.556 to the lamp face and turn it down onto the lamp top (closes the corner notch)"""
    xs = np.linspace(2.540, 2.606, 8)
    rows = []
    for x in xs:
        xx = min(x, 2.60)
        zg = float(D.Z_DECK(np.array([xx]))[0]); yg = float(D.Y_GAP(np.array([xx]))[0]) + 0.002
        zs = float(D.Z_SHOULDER(np.array([xx]))[0]); ys = float(D.Y_SIDE(np.array([xx]), np.array([zs]))[0])
        sec = []
        ang = np.linspace(0, np.pi / 2, 10)
        r_v = zg - zs - 0.012; r_h = min(0.060, 0.7 * (ys - yg))
        sec = [(ys - r_h * (1 - np.cos(a)), zs + r_v * np.sin(a)) for a in ang]
        y1, z1 = sec[-1]
        for t in np.linspace(0, 1, 6)[1:]:
            sec.append((y1 + (yg - y1) * t, z1 + (zg - z1) * t))
        drop = np.clip((x - 2.585) / 0.021, 0, 1)
        rows.append([(x, y, z - drop * (z - LAMP_ZT - 0.004)) for (y, z) in sec])
    ob = grid_obj('RaptorR_Body_FenderNose_L', np.array(rows), mat=PAINT, collection=col)
    return ob, mirror_y(ob, 'RaptorR_Body_FenderNose_R')

# ------------------------------------------------------------------ roof clearance lamps and C-pillar appliques (Ford frames 19-25)
def roof_markers(col):
    out = []
    for s, n in ((1, 'L'), (-1, 'R')):
        rows = []
        for x in np.linspace(-1.050, -0.955, 9):
            t = (x + 1.050) / 0.095
            h = 0.020 * (1 - np.clip((t - 0.55) / 0.45, 0, 1) ** 1.5)
            row = []
            for v in np.linspace(-1, 1, 9):
                y = s * 0.480 + 0.026 * v
                z0 = float(D.Z_ROOF(np.array([x]), np.array([y]))[0])
                row.append((x, y, z0 - 0.002 + h * (1 - abs(v) ** 6)))
            rows.append(row)
        out.append(grid_obj(f'RaptorR_Body_RoofMarker_{n}', np.array(rows), mat='Mat_Gloss_Black_Pillar', collection=col, outward=(0, 0, 1)))
    return out

def cpillar_appliques(col):
    def Xb(y, z): return -1.152 - 0.004 * (y / 0.9) ** 2 + 0.135 * np.clip(z - 1.45, 0, None) ** 1.15
    edge = pchip([1.445, 1.60, 1.80, 1.925], [0.940, 0.910, 0.832, 0.745])
    zs = np.linspace(1.452, 1.918, 24)
    outer = [(float(edge(z)) - 0.002, z) for z in zs]
    inner = [(float(edge(z)) - 0.040, z) for z in zs[::-1]]
    ob = outline_panel('RaptorR_Body_Cab_CPillarTrim_L', outer + inner, Surf('front', lambda y, z: Xb(y, z) - 0.003, sign=-1.0), h=0.01,
                       bnd=0.004, hem=0.006, roll=0.003, mat='Mat_Gloss_Black_Pillar', collection=col)
    return ob, mirror_y(ob, 'RaptorR_Body_Cab_CPillarTrim_R')
