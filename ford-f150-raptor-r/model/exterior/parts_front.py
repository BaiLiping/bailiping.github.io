"""Front fascia: grille, FORD letters, amber markers, headlamps with C-clamp DRL, upper cover, bumper, skid plate, tow hooks."""
import numpy as np, bpy
from geo import pchip, resample
from mk import Surf, outline_panel, make_obj, grid_obj, mirror_y, extrude_profile
import design as D

PAINT = 'Mat_Rapid_Red_Paint'
KY = 1.018   # front-view camera scale correction (official 2210/2438 mm widths)
def XF(y, z):
    """front face of the upper fascia (grille surround / lamp lenses), slightly bowed in plan"""
    u = np.abs(y) / (0.62 * KY)
    return 2.722 - 0.058 * u ** 2 - 0.030 * np.clip(u - 1, 0, None) ** 2 - 0.008 * np.clip((z - 1.05) / 0.25, -1, 1)
FS = lambda d=0.0: Surf('front', lambda y, z: XF(y, z) + d)

def sy(P, k=None):
    k = KY if k is None else k
    return [(p[0] * k, p[1]) for p in P]

def rrect(y0, y1, z0, z1, r, n=6):
    """rounded rectangle polygon (u=y, v=z)"""
    pts = []
    for (cy, cz, a0) in ((y1 - r, z0 + r, -np.pi / 2), (y1 - r, z1 - r, 0), (y0 + r, z1 - r, np.pi / 2), (y0 + r, z0 + r, np.pi)):
        for a in np.linspace(a0, a0 + np.pi / 2, n): pts.append((cy + r * np.cos(a), cz + r * np.sin(a)))
    return pts

def grille(col):
    out = {}
    # grille opening outline (u=y, v=z): chamfered top corners, rounded bottom
    def gopen(s=1.0):
        return [(-0.600 * s, 0.890), (0.600 * s, 0.890), (0.640 * s, 0.925), (0.650 * s, 1.250), (0.618 * s, 1.292),
                (-0.618 * s, 1.292), (-0.650 * s, 1.250), (-0.640 * s, 0.925)]
    # surround frame (gloss black ring between lamps and opening)
    surround = [(-0.67, 0.872), (0.67, 0.872), (0.67, 1.296), (-0.67, 1.296)]
    out['frame'] = outline_panel('RaptorR_Front_Grille_Frame', sy(surround), FS(-0.004), holes=[sy(gopen())], h=0.03, hem=0.03, roll=0.006,
                                 mat='Mat_Grille_Charcoal_Black', collection=col)
    # recessed backing
    out['back'] = outline_panel('RaptorR_Front_Grille_Backing', sy(gopen(1.02)), FS(-0.075), h=0.05, hem=0.0, hem_outer=False,
                                mat='Mat_Trim_Matte_Black', collection=col)
    # three horizontal bars with hexagon-faceted fronts
    bars = []
    for (z0, z1, d) in ((0.935, 1.025, -0.030), (1.055, 1.160, -0.012), (1.185, 1.255, -0.030)):
        bar = rrect(-0.628, 0.628, z0, z1, 0.022)
        bars.append(outline_panel(f'RaptorR_Front_Grille_Bar_{len(bars)}', sy(bar), FS(d), h=0.03, hem=0.03, roll=0.008,
                                  mat='Mat_Grille_Charcoal_Black', collection=col))
    out['bars'] = bars
    # hexagonal lattice between bars (simple vertical struts)
    struts = []
    for y in np.linspace(-0.55, 0.55, 12):
        for (z0, z1) in ((0.895, 0.935), (1.025, 1.055), (1.160, 1.185), (1.255, 1.288)):
            P = [(y - 0.012, z0), (y + 0.012, z0), (y + 0.006, z1), (y - 0.006, z1)]
            struts.append(outline_panel('RaptorR_Front_Grille_Strut', sy(P), FS(-0.045), h=0.05, hem=0.03, roll=0.006,
                                        mat='Mat_Grille_Charcoal_Black', collection=col))
    out['struts'] = struts
    # FORD letters: chunky block letters standing proud of the middle bar
    letters = []
    def F_(y0):
        w, z0, z1 = 0.235, 1.065, 1.152; t = 0.026
        return [[(y0, z0), (y0 + t, z0), (y0 + t, 1.098), (y0 + w * 0.78, 1.098), (y0 + w * 0.80, 1.120), (y0 + t, 1.120),
                 (y0 + t, z1 - t), (y0 + w, z1 - t), (y0 + w, z1), (y0, z1)]], []
    def O_(y0):
        w, z0, z1 = 0.235, 1.065, 1.152; c = 0.022
        outer = [(y0 + c, z0), (y0 + w - c, z0), (y0 + w, z0 + c), (y0 + w, z1 - c), (y0 + w - c, z1), (y0 + c, z1), (y0, z1 - c), (y0, z0 + c)]
        inner = [(y0 + 0.040, 1.091), (y0 + w - 0.040, 1.091), (y0 + w - 0.040, 1.126), (y0 + 0.040, 1.126)]
        return [outer], [inner]
    def R_(y0):
        w, z0, z1 = 0.235, 1.065, 1.152
        outer = [(y0, z0), (y0 + 0.030, z0), (y0 + 0.030, 1.098), (y0 + 0.140, 1.098), (y0 + 0.205, z0), (y0 + w, z0),
                 (y0 + 0.170, 1.100), (y0 + w - 0.012, 1.104), (y0 + w, 1.118), (y0 + w, z1 - 0.016), (y0 + w - 0.016, z1), (y0, z1)]
        inner = [(y0 + 0.030, 1.118), (y0 + w - 0.032, 1.118), (y0 + w - 0.032, 1.132), (y0 + 0.030, 1.132)]
        return [outer], [inner]
    def Dl(y0):
        w, z0, z1 = 0.235, 1.065, 1.152; c = 0.030
        outer = [(y0, z0), (y0 + w - c, z0), (y0 + w, z0 + c), (y0 + w, z1 - c), (y0 + w - c, z1), (y0, z1)]
        inner = [(y0 + 0.032, 1.091), (y0 + w - 0.040, 1.091), (y0 + w - 0.040, 1.126), (y0 + 0.032, 1.126)]
        return [outer], [inner]
    # left side of vehicle is +y; seen from the front, F is on the vehicle's right (negative y)
    for fn, y0 in ((F_, -0.552), (O_, -0.272), (R_, 0.035), (Dl, 0.318)):
        outs, ins = fn(y0)
        for o in outs:
            # mirror letters left-right for correct reading from the front (camera right = +y)
            letters.append(outline_panel('RaptorR_Front_Grille_FORD', sy(o), FS(0.012), holes=[sy(i) for i in ins], h=0.02, bnd=0.004, hem=0.028,
                                         roll=0.004, mat='Mat_FORD_Letter_Face', collection=col))
    for ob in letters:
        me = ob.data; uvl = me.uv_layers.new(name='UVMap')
        for poly in me.polygons:
            for li in poly.loop_indices:
                co = me.vertices[me.loops[li].vertex_index].co
                uvl.data[li].uv = (co.y * 6.0, co.z * 6.0)
    out['letters'] = letters
    # amber marker lamps on the grille top
    mk_ = []
    for yc in (-0.255, 0.0, 0.255):
        P = rrect(yc - 0.046, yc + 0.046, 1.256, 1.281, 0.008)
        mk_.append(outline_panel('RaptorR_Front_AmberMarker', sy(P), FS(0.004), h=0.02, hem=0.012, roll=0.003, mat='Mat_Amber_LED_DRL', collection=col))
    out['markers'] = mk_
    # R badge
    rb = [(0.445, 0.968), (0.462, 0.968), (0.462, 0.985), (0.486, 0.968), (0.506, 0.968), (0.478, 0.988), (0.500, 0.996),
          (0.500, 1.010), (0.490, 1.018), (0.445, 1.018)]
    out['rbadge'] = outline_panel('RaptorR_Front_Grille_RBadge', sy(rb), FS(-0.006), h=0.01, bnd=0.003, hem=0.008, roll=0.002,
                                  mat='Mat_Code_Orange_Accent', collection=col)
    return out

def headlamps(col):
    out = {}
    # lens outline in front view (u=y, v=z) for the left lamp (+y side)
    lens = [(0.565, 0.952), (0.860, 0.952), (0.895, 0.985), (0.905, 1.245), (0.885, 1.292), (0.565, 1.292)]
    # lens surface wraps the corner: deeper toward the outside
    def XL(y, z): return XF(y, z) + 0.004 - 0.060 * np.clip((np.abs(y) - 0.70) / 0.22, 0, 1) ** 1.6
    LS = Surf('front', XL)
    # plan curve of the lens: side face -> rounded corner -> front face (left lamp)
    from geo import spline
    fy = [0.578, 0.69, 0.79, 0.87]
    front_pts = [(float(XL(np.array([y]), np.array([1.12]))[0]), y) for y in fy]
    plan = spline([(2.232, 0.990), (2.330, 0.987), (2.420, 0.978), (2.500, 0.955), (2.536, 0.920)] + front_pts[::-1], 60)
    zt = lambda x: 1.292
    zb = pchip([2.232, 2.250, 2.280, 2.310, 2.345, 2.40, 2.70], [1.285, 1.215, 1.115, 1.030, 0.962, 0.952, 0.952])
    rows = []
    for (x, y) in plan:
        z0, z1 = float(zb(x)), zt(x)
        rows.append([(x, y, z) for z in np.linspace(z0, z1, 14)])
    P = np.array(rows)
    lensob = grid_obj('RaptorR_Headlight_Lens_L', P, mat='Mat_Headlight_Clear_Lens', collection=col)
    out['lens'] = (lensob, mirror_y(lensob, 'RaptorR_Headlight_Lens_R'))
    # dark inner housing following the same plan, 4 cm inboard
    rows = []
    for (x, y) in plan:
        z0, z1 = float(zb(x)) + 0.004, zt(x) - 0.004
        rows.append([(x - 0.035, y - 0.030 * (x < 2.55), z) for z in np.linspace(z0, z1, 10)])
    hsob = grid_obj('RaptorR_Headlight_Housing_Side_L', np.array(rows), mat='Mat_Gloss_Black_Pillar', collection=col)
    out['hside'] = (hsob, mirror_y(hsob, 'RaptorR_Headlight_Housing_Side_R'))
    # amber C-clamp continuing onto the side face (vertical leg sits at the corner)
    rows = []
    sidepts = [p for p in plan if p[0] < 2.56]
    for (x, y) in sidepts[-8:]:
        rows.append([(x - 0.006, y - 0.006, z) for z in np.linspace(1.263, 1.277, 3)])
    dsob = grid_obj('RaptorR_Headlight_AmberDRL_Side_L', np.array(rows), mat='Mat_Amber_LED_DRL', collection=col)
    out['dside'] = (dsob, mirror_y(dsob, 'RaptorR_Headlight_AmberDRL_Side_R'))
    # housing (black) behind the lens
    LH = Surf('front', lambda y, z: XL(y, z) - 0.050)
    hs = outline_panel('RaptorR_Headlight_Housing_L', sy(lens), LH, h=0.03, hem=0.0, hem_outer=False, mat='Mat_Gloss_Black_Pillar', collection=col)
    out['housing'] = (hs, mirror_y(hs, 'RaptorR_Headlight_Housing_R'))
    # DRL light guides, reflectors, projectors and blade: parts_detail.headlamp_internals
    return out

def upper_cover(col):
    """painted bumper top cover below the lamps / grille, wrapping the corners"""
    out = {}
    def XU(y, z): return XF(y, z) - 0.006 + 0.012 * np.clip((0.95 - z) / 0.08, 0, 1)
    poly = [(-0.98, 0.858), (0.98, 0.858), (0.985, 0.955), (0.905, 0.955), (0.895, 0.985), (0.860, 0.952), (0.565, 0.952),
            (0.565, 0.872), (-0.565, 0.872), (-0.565, 0.952), (-0.860, 0.952), (-0.895, 0.985), (-0.905, 0.955), (-0.985, 0.955)]
    out['cover'] = outline_panel('RaptorR_Body_Front_Valance', sy(poly), Surf('front', XU), h=0.02, hem=0.015, mat=PAINT, collection=col)
    # side wraps: painted strip from the lamp corner back to the fender/flare (left; mirrored)
    def Yw(x, z): return D.YP(np.asarray(x)) - 0.012 - 0.03 * np.clip((x - 2.50) / 0.10, 0, 1) ** 2
    wrap = [(2.330, 0.860), (2.560, 0.860), (2.600, 0.900), (2.600, 0.955), (2.335, 0.955)]
    w = outline_panel('RaptorR_Body_Front_Wrap_L', wrap, Surf('side', Yw), h=0.02, hem=0.012, mat=PAINT, collection=col)
    out['wrap'] = (w, mirror_y(w, 'RaptorR_Body_Front_Wrap_R'))
    return out

def bumper(col):
    out = {}
    def XB(y, z):
        ay = np.abs(y)
        return (2.792 - 0.120 * (ay / 0.85) ** 3.5 - 0.220 * np.clip((ay - 0.80) / 0.20, 0, 1) ** 2
                - 0.050 * np.clip((z - 0.78) / 0.09, 0, 1) - 0.048 * np.clip((0.70 - z) / 0.12, 0, 1) ** 1.3)
    BS = Surf('front', XB)
    face = [(-1.000, 0.700), (-0.900, 0.595), (-0.640, 0.546), (0.640, 0.546), (0.900, 0.595), (1.000, 0.700), (1.005, 0.862),
            (-1.005, 0.862)]
    fogL = rrect(0.680, 0.960, 0.742, 0.828, 0.02); fogR = [(-y, z) for (y, z) in fogL][::-1]
    center = [(-0.430, 0.640), (0.430, 0.640), (0.470, 0.700), (0.440, 0.790), (-0.440, 0.790), (-0.470, 0.700)]
    out['face'] = outline_panel('RaptorR_Front_Bumper', face, BS, holes=[sy(fogL), sy(fogR), sy(center)], h=0.03, hem=0.04, roll=0.01,
                                mat='Mat_Bumper_Steel_Charcoal', collection=col)
    out['recess'] = outline_panel('RaptorR_Front_Bumper_Recess', [(-0.47, 0.63), (0.47, 0.63), (0.47, 0.80), (-0.47, 0.80)],
                                  Surf('front', lambda y, z: XB(y, z) - 0.050), h=0.04, hem=0.0, hem_outer=False, mat='Mat_Trim_Matte_Black', collection=col)
    slats = []
    for yc in (-0.30, -0.10, 0.10, 0.30):
        P = [(yc - 0.012, 0.645), (yc + 0.012, 0.645), (yc + 0.012, 0.785), (yc - 0.012, 0.785)]
        slats.append(outline_panel('RaptorR_Front_Bumper_Slat', sy(P), Surf('front', lambda y, z: XB(y, z) - 0.015), h=0.02, hem=0.035,
                                   roll=0.006, mat='Mat_Bumper_Steel_Charcoal', collection=col))
    out['slats'] = slats
    fogs = []
    for P, nm in ((fogL, 'L'), (fogR, 'R')):
        fogs.append(outline_panel(f'RaptorR_Front_FogPocket_{nm}', sy(P), Surf('front', lambda y, z: XB(y, z) - 0.035), h=0.02, hem=0.0,
                                  hem_outer=False, mat='Mat_Trim_Matte_Black', collection=col))
        sgn = 1 if nm == 'L' else -1
        for k, (a, b) in enumerate(((0.694, 0.772), (0.868, 0.946))):
            # Rigid LED fog modules: black housing with a lit lens inset (BaT 012)
            hs = rrect(min(sgn * a, sgn * b), max(sgn * a, sgn * b), 0.756, 0.814, 0.008)
            fogs.append(outline_panel(f'RaptorR_Front_FogHousing_{nm}{k}', sy(hs), Surf('front', lambda y, z: XB(y, z) - 0.024), h=0.01,
                                      bnd=0.004, hem=0.012, roll=0.004, mat='Mat_Gloss_Black_Pillar', collection=col))
            a2, b2 = a + 0.012, b - 0.012
            led = rrect(min(sgn * a2, sgn * b2), max(sgn * a2, sgn * b2), 0.768, 0.802, 0.005)
            fogs.append(outline_panel(f'RaptorR_Front_FogLights_{nm}{k}', sy(led), Surf('front', lambda y, z: XB(y, z) - 0.021), h=0.01,
                                      bnd=0.003, hem=0.004, roll=0.002, mat='Mat_LED_Headlight_White', collection=col))
        sen = [(sgn * 0.820 + 0.011 * np.cos(t), 0.785 + 0.011 * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
        fogs.append(outline_panel(f'RaptorR_Front_ParkSensor_{nm}', sy(sen), Surf('front', lambda y, z: XB(y, z) - 0.030), h=0.01,
                                  bnd=0.003, hem=0.004, roll=0.002, mat='Mat_Grille_Charcoal_Black', collection=col))
    out['fogs'] = fogs
    # top shelf of the bumper (horizontal, under the painted cover)
    xs = np.linspace(-1.0, 1.0, 41)
    rows = []
    for y in xs:
        x0 = XB(np.array([y]), np.array([0.862]))[0]
        rows.append([(x0, y, 0.862), (x0 - 0.10, y, 0.865), (x0 - 0.20, y, 0.866)])
    out['top'] = grid_obj('RaptorR_Front_Bumper_TopPad', np.array(rows), mat='Mat_Trim_Matte_Black', collection=col)
    # lower lip / skid plate
    skid = []
    # Ford side render (frame 13, back-projected): skid lower edge (2.84, 0.50) -> (2.69, 0.457) -> (2.49, 0.417);
    # front render: skid bottom 0.437 at x 2.55. Official 33.1 deg approach angle is then met by the skid's front edge.
    rows = []
    for y in np.linspace(-0.62, 0.62, 27):
        k = (y / 0.62) ** 2
        rows.append([(2.742 - 0.050 * k, y, 0.548), (2.762 - 0.050 * k, y, 0.512), (2.700 - 0.040 * k, y, 0.470), (2.550, y, 0.437),
                     (2.300, y, 0.425), (2.000, y, 0.422)])
    out['skid'] = grid_obj('RaptorR_Front_SkidPlate', np.array(rows), mat='Mat_Skid_Plate_Silver', collection=col, outward=(0.3, 0, -1))
    # tow hooks
    hooks = []
    for yc in (-0.425 * KY, 0.425 * KY):
        # vertical D-shaped hook plates protruding below the bumper's lower edge (BaT 003/012)
        a = np.linspace(-np.pi / 2, np.pi / 2, 13)
        D_ = [(2.700, 0.528)] + [(2.752 + 0.034 * np.cos(t), 0.556 + 0.028 * np.sin(t)) for t in a] + [(2.700, 0.584)]
        hooks.append(extrude_profile('RaptorR_Front_TowHooks', D_, 'y', 0.026, origin=(0, yc - 0.013, 0),
                                     mat='Mat_Code_Orange_Accent', bevel=0.005, collection=col))
    out['hooks'] = hooks
    return out
