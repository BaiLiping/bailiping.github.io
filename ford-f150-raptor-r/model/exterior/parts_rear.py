"""Rear: tailgate (outer, applique, FORD letters, handle, cap), taillamps, stepped bumper, exhaust tips, hitch, bed box."""
import numpy as np, bpy
from geo import pchip, resample
from mk import Surf, outline_panel, make_obj, grid_obj, mirror_y, extrude_profile
from parts_front import rrect
import design as D

PAINT = 'Mat_Rapid_Red_Paint'
TG_Z0, TG_Z1 = 0.812, 1.452
def XT(y, z):  # tailgate outer surface (rear facing): slight crown and a tumble near the top
    return -2.968 + 0.010 * (np.abs(y) / 0.82) ** 2 + 0.030 * np.clip((z - 1.30) / 0.15, 0, 1) ** 2 - 0.006 * np.clip((1.00 - z) / 0.2, 0, 1)
TS = lambda d=0.0: Surf('front', lambda y, z: XT(y, z) - d, sign=-1.0)

def tailgate(col):
    out = {}
    tg = [(-0.812, TG_Z0), (0.812, TG_Z0), (0.814, TG_Z1 - 0.03), (0.800, TG_Z1), (-0.800, TG_Z1), (-0.814, TG_Z1 - 0.03)]
    handle = rrect(-0.205, 0.205, 1.300, 1.392, 0.030)
    out['outer'] = outline_panel('RaptorR_Tailgate_Outer', tg, TS(), holes=[handle], h=0.03, hem=0.03, mat=PAINT, collection=col)
    out['bezel'] = outline_panel('RaptorR_Tailgate_HandleBezel', handle, TS(-0.020), h=0.02, hem=0.0, hem_outer=False,
                                 mat='Mat_Trim_Matte_Black', collection=col)
    out['handle'] = outline_panel('RaptorR_Tailgate_Handle', rrect(-0.110, 0.110, 1.312, 1.348, 0.012), TS(-0.010), h=0.01, hem=0.012,
                                  roll=0.004, mat='Mat_Trim_Matte_Black', collection=col)
    cam = rrect(-0.016, 0.016, 1.360, 1.384, 0.010)
    out['camera'] = outline_panel('RaptorR_Tailgate_Camera', cam, TS(-0.012), h=0.01, hem=0.008, roll=0.003, mat='Mat_Gloss_Black_Pillar', collection=col)
    app = [(-0.806, 0.972), (0.806, 0.972), (0.806, 1.262), (-0.806, 1.262)]
    out['applique'] = outline_panel('RaptorR_Tailgate_Applique', app, TS(0.008), h=0.03, hem=0.012, roll=0.004,
                                    mat='Mat_Tailgate_Applique_Satin', collection=col)
    # FORD letters: parts_detail.tailgate_letters
    # top cap (black) across the tailgate top
    rows = []
    for y in np.linspace(-0.80, 0.80, 33):
        x0 = XT(np.array([y]), np.array([TG_Z1]))[0]
        rows.append([(x0 + 0.002, y, TG_Z1 - 0.004), (x0 + 0.03, y, TG_Z1 + 0.006), (x0 + 0.07, y, TG_Z1 + 0.006)])
    out['cap'] = grid_obj('RaptorR_Tailgate_TopCap', np.array(rows), mat='Mat_Trim_Matte_Black', collection=col)
    # inner panel (bedliner) so the tailgate looks solid when lowered
    inner = [(-0.80, TG_Z0 + 0.02), (0.80, TG_Z0 + 0.02), (0.80, TG_Z1 - 0.01), (-0.80, TG_Z1 - 0.01)]
    out['inner'] = outline_panel('RaptorR_Tailgate_Inner', inner, Surf('front', lambda y, z: XT(y, z) + 0.075 + 0 * y, sign=1.0), h=0.04,
                                 hem=0.0, hem_outer=False, mat='Mat_Spray_Bedliner', collection=col)
    # side edges of the tailgate (closing faces)
    for s in (1, -1):
        rows = []
        for z in np.linspace(TG_Z0, TG_Z1, 20):
            x0 = XT(np.array([0.812 * s]), np.array([z]))[0]
            rows.append([(x0, 0.812 * s, z), (x0 + 0.075, 0.812 * s, z)])
        out['edge' + str(s)] = grid_obj('RaptorR_Tailgate_Edge', np.array(rows), mat='Mat_Trim_Matte_Black', collection=col)
    return out

def taillamps(col):
    out = {}
    # rear face (u=y, v=z) of the left lamp and the side wrap (u=x, v=z)
    rear = [(0.818, 0.938), (0.952, 0.938), (0.967, 0.96), (0.968, 1.282), (0.950, 1.304), (0.818, 1.304)]
    def XR(y, z): return -2.988 + 0.014 * np.clip((np.abs(y) - 0.90) / 0.07, 0, 1) ** 2
    lr_ = outline_panel('RaptorR_Taillight_Lens_L', rear, Surf('front', XR, sign=-1.0), h=0.015, hem=0.04, roll=0.010,
                        mat='Mat_Taillight_Red_Lens', collection=col)
    side = [(-2.975, 0.938), (-2.930, 0.938), (-2.890, 0.955), (-2.850, 0.990), (-2.800, 1.060), (-2.760, 1.130), (-2.730, 1.200),
            (-2.716, 1.288), (-2.732, 1.304), (-2.975, 1.304)]
    def YL(x, z): return D.YP(np.clip(np.asarray(x), -2.95, 0)) + 0.004 - 0.024 * np.clip((-2.950 - x) / 0.030, 0, 1) ** 2
    ls = outline_panel('RaptorR_Taillight_Side_L', side, Surf('side', YL), h=0.015, hem=0.03, roll=0.008, mat='Mat_Taillight_Red_Lens', collection=col)
    out['lens'] = [(lr_, mirror_y(lr_, 'RaptorR_Taillight_Lens_R')), (ls, mirror_y(ls, 'RaptorR_Taillight_Side_R'))]
    # C-signature light blades (bright) behind the lens
    sig = [(0.835, 1.195), (0.948, 1.195), (0.950, 1.240), (0.835, 1.240)]
    sig2 = [(0.885, 1.040), (0.935, 1.040), (0.935, 1.195), (0.885, 1.195)]
    blades = []
    for P, nm in ((sig, 'Top'), (sig2, 'Leg')):
        b = outline_panel(f'RaptorR_Taillight_Blade_{nm}_L', P, Surf('front', lambda y, z: XR(y, z) + 0.008, sign=-1.0), h=0.01, hem=0.006,
                          roll=0.003, mat='Mat_Taillight_Amber_Blade', collection=col)
        blades.append((b, mirror_y(b, f'RaptorR_Taillight_Blade_{nm}_R')))
    lower = [(0.825, 0.946), (0.955, 0.946), (0.955, 1.010), (0.825, 1.010)]
    bk = outline_panel('RaptorR_Taillight_Reverse_L', lower, Surf('front', lambda y, z: XR(y, z) + 0.010, sign=-1.0), h=0.01, hem=0.006,
                       roll=0.003, mat='Mat_Taillight_Clear_Accent', collection=col)
    blades.append((bk, mirror_y(bk, 'RaptorR_Taillight_Reverse_R')))
    out['blades'] = blades
    return out

def rear_bumper(col):
    out = {}
    # BaT rear photo scaled by the 305 mm plate: bumper ends at ~0.955 m, centre section down to ~0.54 m,
    # corner sections end at ~0.63 m above the exhaust tips
    # Ford side renders (frames 13/31): the face stands ~35 mm further back at z 0.65-0.72 and its top lip ~45 mm
    # further forward than a flat face, i.e. the step pad is set back above a proud lower face.
    def XRB(y, z):
        zz = np.asarray(z, float)
        prof = np.where(zz > 0.745, -3.075 + 0.040 * np.clip((zz - 0.745) / 0.027, 0, 1) ** 1.5,
                        -3.075 + 0.045 * np.clip((0.62 - zz) / 0.075, 0, 1) ** 1.4)
        return prof + 0.020 * (np.abs(y) / 1.0) ** 3 + 0.110 * np.clip((np.abs(y) - 0.82) / 0.135, 0, 1) ** 2
    face = [(-0.955, 0.645), (-0.70, 0.632), (-0.640, 0.600), (-0.600, 0.545), (0.600, 0.545), (0.640, 0.600), (0.70, 0.632),
            (0.955, 0.645), (0.962, 0.790), (0.94, 0.810), (0.82, 0.810), (0.80, 0.772), (-0.80, 0.772), (-0.82, 0.810),
            (-0.94, 0.810), (-0.962, 0.790)]
    plate = rrect(-0.232, 0.232, 0.622, 0.748, 0.012)
    out['face'] = outline_panel('RaptorR_Rear_Bumper', face, Surf('front', XRB, sign=-1.0), holes=[plate], h=0.03, hem=0.05, roll=0.012,
                                mat='Mat_Bumper_Steel_Charcoal', collection=col)
    out['platewell'] = outline_panel('RaptorR_Rear_Bumper_PlateWell', plate, Surf('front', lambda y, z: XRB(y, z) + 0.030, sign=-1.0), h=0.03,
                                     hem=0.0, hem_outer=False, mat='Mat_Trim_Matte_Black', collection=col)
    # step pads (top surfaces): centre (low) and corner steps (high)
    rows = []
    for y in np.linspace(-0.80, 0.80, 33):
        x0 = XRB(np.array([y]), np.array([0.772]))[0]
        rows.append([(x0, y, 0.772), (x0 + 0.06, y, 0.773), (x0 + 0.20, y, 0.774)])
    out['step'] = grid_obj('RaptorR_Rear_Bumper_StepPad_C', np.array(rows), mat='Mat_Trim_Matte_Black', collection=col)
    for s, nm in ((1, 'L'), (-1, 'R')):
        rows = []
        for y in np.linspace(0.82, 0.955, 8):
            x0 = XRB(np.array([y]), np.array([0.810]))[0]
            rows.append([(x0, s * y, 0.810), (x0 + 0.08, s * y, 0.811), (-2.86, s * y, 0.811)])
        out['corner' + nm] = grid_obj(f'RaptorR_Rear_Bumper_StepPad_{nm}', np.array(rows), mat='Mat_Trim_Matte_Black', collection=col, flip=(s < 0))
    # exhaust tips: the kept exhaust system supplies them (moved to Ford's positions in build_raptor)
    tips = []
    for s in ():
        ang = np.linspace(0, 2 * np.pi, 33)[:-1]
        rows = []
        for x in np.linspace(-2.80, -3.010, 6):
            r = 0.046 if x > -2.995 else 0.050
            rows.append([(x, s * 0.73 + r * np.cos(a), 0.612 + r * np.sin(a)) for a in ang])
        P = np.array(rows)
        tips.append(grid_obj('RaptorR_Exhaust_Tip_' + ('L' if s > 0 else 'R'), P, mat='Mat_Exhaust_Tip_SatinBlack', collection=col, closed_v=True))
        rows = [[(-3.010, s * 0.73 + r * np.cos(a), 0.612 + r * np.sin(a)) for a in ang] for r in (0.050, 0.042)]
        rows.append([(-2.96, s * 0.73 + 0.042 * np.cos(a), 0.612 + 0.042 * np.sin(a)) for a in ang])
        tips.append(grid_obj('RaptorR_Exhaust_TipBore', np.array(rows), mat='Mat_Underbody_Exhaust_Soot', collection=col, closed_v=True))
    out['tips'] = tips
    # 2.5 in receiver tucked flush under the bumper (BaT 008); with it the model meets Ford's 24.9 deg departure angle
    out['hitch'] = extrude_profile('RaptorR_Rear_Hitch', [(-0.032, 0.518), (0.032, 0.518), (0.032, 0.582), (-0.032, 0.582)], 'x', 0.30,
                                   origin=(-3.070, 0, 0), mat='Mat_Frame_Black_Steel', bevel=0.004, collection=col)
    out['hitch_bore'] = outline_panel('RaptorR_Rear_Hitch_Bore', rrect(-0.0254, 0.0254, 0.5246, 0.5754, 0.003),
                                      Surf('front', lambda y, z: -3.0715 + 0 * y, sign=-1.0), h=0.01, bnd=0.003, hem=0.02, roll=0.002,
                                      mat='Mat_Underbody_Exhaust_Soot', collection=col)
    return out

def bed_box(col):
    """inner bed: floor, front wall, side walls with wheel tubs, rail caps"""
    out = {}
    x0, x1 = D.BED_REAR_X + 0.075, D.BED_FRONT_X - 0.050
    zf, zt = D.BED_FLOOR_Z, D.BED_RAIL_Z
    # floor with corrugations
    xs = np.linspace(x0, x1, 40); ys = np.linspace(-0.828, 0.828, 61)
    P = np.zeros((len(xs), len(ys), 3))
    for i, x in enumerate(xs):
        P[i, :, 0] = x; P[i, :, 1] = ys
        P[i, :, 2] = zf + 0.010 * (np.cos(ys / 0.828 * np.pi * 7) > 0.2)
    out['floor'] = grid_obj('RaptorR_Bed_Floor', P, mat='Mat_Spray_Bedliner', collection=col, outward=(0, 0, 1))
    # side walls (inner) with wheel tubs
    for s, nm in ((1, 'L'), (-1, 'R')):
        zs = np.linspace(zf, zt - 0.02, 14); xs = np.linspace(x0, x1, 50)
        P = np.zeros((len(xs), len(zs), 3))
        for i, x in enumerate(xs):
            for j, z in enumerate(zs):
                yw = 0.828 + 0.010 * (z - zf) / (zt - zf)
                # wheel tub: bulge inward around rear axle below 1.12 m
                dx = (x + D.WB2) / 0.62; dz = (z - zf) / 0.24
                tub = max(0.0, 1 - dx * dx) * max(0.0, 1 - max(0.0, dz - 0.8) * 4)
                yw -= 0.185 * min(1.0, tub * 1.6)
                P[i, j] = (x, s * yw, z)
        out['wall' + nm] = grid_obj(f'RaptorR_Bed_InnerWall_{nm}', P, mat='Mat_Spray_Bedliner', collection=col, outward=(0, -s, 0))
        # rail cap (black) over the bedside top
        rows = []
        for x in np.linspace(D.BED_FRONT_X - 0.004, D.BED_REAR_X + 0.004, 60):
            rows.append([(x, s * 0.824, zt - 0.012), (x, s * 0.840, zt), (x, s * 0.905, zt + 0.003), (x, s * 0.930, zt - 0.004)])
        out['cap' + nm] = grid_obj(f'RaptorR_Bed_RailCap_{nm}', np.array(rows), mat='Mat_Trim_Matte_Black', collection=col, flip=(s < 0))
    # front wall
    ys = np.linspace(-0.828, 0.828, 30); zs = np.linspace(zf, zt - 0.01, 14)
    P = np.array([[(x1, y, z) for z in zs] for y in ys])
    out['front'] = grid_obj('RaptorR_Bed_Bulkhead', P, mat='Mat_Spray_Bedliner', collection=col, outward=(-1, 0, 0))
    return out
