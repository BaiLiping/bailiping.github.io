"""Closures' inner faces, tonneau cover, bed details, fuel pocket, plate, badges, markers, rear bumper wings."""
import numpy as np, bpy
from geo import pchip, resample, spline
from mk import Surf, outline_panel, make_obj, grid_obj, mirror_y, extrude_profile, perim
from parts_front import rrect
import design as D

PAINT = 'Mat_Rapid_Red_Paint'

def box(name, x0, x1, y0, y1, z0, z1, mat, col, bevel=0.0):
    P = [(y0, z0), (y1, z0), (y1, z1), (y0, z1)]
    return extrude_profile(name, P, 'x', x1 - x0, origin=(x0, 0, 0), mat=mat, bevel=bevel, collection=col)

def door_shell(col, name, outline, depth=0.024, inner_mat=PAINT):
    """closing faces of a door: band from the skin edge inward by `depth`, plus inner sheet"""
    O = resample(np.asarray(outline, float), max(40, int(perim(outline) / 0.01)), closed=True)
    ring_out = []; ring_in = []
    for (x, z) in O:
        yo = D.Y_SIDE(np.array([x]), np.array([z]))[0] - 0.012
        ring_out.append((x, yo, z)); ring_in.append((x, yo - depth, z))
    P = np.array([ring_out, ring_in]).transpose(1, 0, 2)
    band = grid_obj(name + '_Edge', P, mat=inner_mat, collection=col, closed_u=True,
                    outward=lambda p: np.stack([p[:, 0] - O[:, 0].mean(), np.zeros(len(p)), p[:, 2] - O[:, 1].mean()], 1))
    return band, None

def tonneau(col):
    out = {}
    zt = D.BED_RAIL_Z + 0.004      # cover top flush with the rail caps
    leaves = (('Header', -1.205, -1.398), ('Front', -1.401, -1.995), ('Middle', -1.998, -2.478), ('Rear', -2.481, -2.938))
    for nm, xa, xb in leaves:
        x0, x1 = min(xa, xb), max(xa, xb)
        out[nm] = [
            box(f'RaptorR_Tonneau_{nm}_Frame', x0, x1, -0.836, 0.836, zt - 0.026, zt - 0.004, 'Mat_Grille_Charcoal_Black', col),
            box(f'RaptorR_Tonneau_{nm}_Panel', x0 + 0.030, x1 - 0.030, -0.808, 0.808, zt - 0.004, zt, 'Mat_Tonneau_Powder_Coat', col),
            box(f'RaptorR_Tonneau_{nm}_FrontEdge', x1 - 0.030, x1, -0.818, 0.818, zt - 0.004, zt + 0.0005, 'Mat_Trim_Matte_Black', col),
            box(f'RaptorR_Tonneau_{nm}_RearEdge', x0, x0 + 0.030, -0.818, 0.818, zt - 0.004, zt + 0.0005, 'Mat_Trim_Matte_Black', col),
            box(f'RaptorR_Tonneau_{nm}_SideSeal_L', x0 + 0.002, x1 - 0.002, 0.808, 0.822, zt - 0.004, zt, 'Mat_Trim_Matte_Black', col),
            box(f'RaptorR_Tonneau_{nm}_SideSeal_R', x0 + 0.002, x1 - 0.002, -0.822, -0.808, zt - 0.004, zt, 'Mat_Trim_Matte_Black', col)]
    for s, n in ((1, 'L'), (-1, 'R')):
        out['guide' + n] = box(f'RaptorR_Tonneau_Guide_{n}', -2.945, -1.205, min(s * 0.838, s * 0.862), max(s * 0.838, s * 0.862),
                               zt - 0.030, zt - 0.003, 'Mat_Grille_Charcoal_Black', col)
    out['pivots'] = {'Pivot_Tonneau_Middle': (-1.9965, 0.0, zt + 0.023), 'Pivot_Tonneau_Rear': (-2.4795, 0.0, zt + 0.003)}
    return out

def bed_details(col):
    out = []
    for s, n in ((1, 'L'), (-1, 'R')):
        for x in (-1.285, -2.840):
            out.append(box(f'RaptorR_Bed_StakeCap_{n}', x - 0.029, x + 0.029, min(s * 0.86, s * 0.905), max(s * 0.86, s * 0.905),
                           D.BED_RAIL_Z + 0.002, D.BED_RAIL_Z + 0.006, 'Mat_Grille_Charcoal_Black', col))
        out.append(box(f'RaptorR_Bed_CargoLight_{n}', -1.300, -1.270, min(s * 0.818, s * 0.827), max(s * 0.818, s * 0.827), 1.36, 1.42,
                       'Mat_LED_Headlight_White', col))
        for x in (-1.32, -2.80):
            out.append(box(f'RaptorR_Bed_TieDown_{n}', x - 0.03, x + 0.03, min(s * 0.808, s * 0.826), max(s * 0.808, s * 0.826), 1.05, 1.10,
                           'Mat_Chrome_Hardware', col, bevel=0.006))
    return out

def fuel_pocket(col):
    ang = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    rows = []
    for d, r in ((0.000, 0.082), (0.010, 0.080), (0.045, 0.072), (0.050, 0.0)):
        rows.append([(-1.375 + r * np.cos(a), D.Y_SIDE(np.array([-1.375]), np.array([1.205]))[0] - 0.002 - d, 1.205 + r * np.sin(a)) for a in ang])
    P = np.array(rows).transpose(1, 0, 2)
    pocket = grid_obj('RaptorR_Fuel_Pocket', P, mat='Mat_Trim_Matte_Black', collection=col, closed_u=True, outward=(0, 1, 0))
    neck = []
    for d, r in ((0.038, 0.034), (0.046, 0.034), (0.046, 0.020)):
        neck.append([(-1.375 + r * np.cos(a), D.Y_SIDE(np.array([-1.375]), np.array([1.205]))[0] - 0.002 - d, 1.205 + r * np.sin(a)) for a in ang])
    nk = grid_obj('RaptorR_Fuel_CaplessNeck', np.array(neck).transpose(1, 0, 2), mat='Mat_Grille_Charcoal_Black', collection=col, closed_u=True, outward=(0, 1, 0))
    return pocket, nk

def plate(col):
    P = [(-0.1525, 0.610), (0.1525, 0.610), (0.1525, 0.762), (-0.1525, 0.762)]
    ob = outline_panel('RaptorR_Rear_LicensePlate', P, Surf('front', lambda y, z: -3.012 + 0 * y, sign=-1.0), h=0.05, hem=0.0, hem_outer=False,
                       mat='Mat_License_Plate_MN', collection=col)
    me = ob.data; uvl = me.uv_layers.new(name='UVMap')
    for poly in me.polygons:
        for li in poly.loop_indices:
            co = me.vertices[me.loops[li].vertex_index].co
            uvl.data[li].uv = ((0.1525 - co.y) / 0.305, (co.z - 0.610) / 0.152)   # reads correctly from behind
    return ob

def badges(col):
    out = []
    # tailgate: F-150 on the left end (vehicle left, +y), RAPTOR R on the right end
    TS = Surf('front', lambda y, z: -2.968 + 0.010 * (np.abs(y) / 0.82) ** 2 - 0.016, sign=-1.0)
    out.append(outline_panel('RaptorR_Tailgate_Badge_F150', rrect(0.585, 0.745, 1.008, 1.030, 0.006), TS, h=0.01, hem=0.004, roll=0.002,
                             mat='Mat_Chrome_Hardware', collection=col))
    out.append(outline_panel('RaptorR_Tailgate_Badge_RAPTOR', rrect(-0.760, -0.600, 1.008, 1.030, 0.006), TS, h=0.01, hem=0.004, roll=0.002,
                             mat='Mat_Chrome_Hardware', collection=col))
    rb = [(-0.592, 1.004), (-0.578, 1.004), (-0.578, 1.016), (-0.560, 1.004), (-0.545, 1.004), (-0.566, 1.019), (-0.548, 1.025),
          (-0.548, 1.034), (-0.556, 1.040), (-0.592, 1.040)]
    out.append(outline_panel('RaptorR_Tailgate_Badge_CodeOrangeR', rb, TS, h=0.008, hem=0.004, roll=0.002, mat='Mat_Code_Orange_Accent', collection=col))
    # front corner side markers (amber) and rear bedside reflectors (red)
    for s, n in ((1, 'L'), (-1, 'R')):
        S = Surf('side', lambda x, z: D.Y_SIDE(x, z) + 0.004)
        m = outline_panel(f'RaptorR_Front_SideMarker_{n}', rrect(2.336, 2.374, 0.884, 0.966, 0.008), S, h=0.01, hem=0.008, roll=0.003,
                          mat='Mat_Amber_Side_Reflector', collection=col)
        if s < 0:
            for v in m.data.vertices: v.co.y = -v.co.y
            m.data.flip_normals()
        out.append(m)
    return out

def rear_bumper_wings(col):
    """black bumper end caps wrapping forward under the tapered bedsides to the rear flares"""
    out = []
    for s, n in ((1, 'L'), (-1, 'R')):
        rows = []
        for x in np.linspace(-2.950, -2.500, 16):
            yb = D.Y_SIDE(np.array([x]), np.array([0.75]))[0] + 0.012
            t = np.clip((x + 2.95) / 0.45, 0, 1)
            z0 = 0.645 + 0.060 * t ** 1.5
            z1 = 0.846 - 0.040 * t ** 2
            rows.append([(x, s * (yb - 0.035), z0 - 0.012), (x, s * yb, z0 + 0.008), (x, s * (yb + 0.004), (z0 + z1) / 2),
                         (x, s * yb, z1 - 0.006), (x, s * (yb - 0.030), z1)])
        out.append(grid_obj(f'RaptorR_Rear_Bumper_Wing_{n}', np.array(rows), mat='Mat_Bumper_Steel_Charcoal', collection=col))
    return out

def hood_underside(col):
    HS = Surf('top', lambda x, y: D.Z_HOOD(x, y) - 0.028, sign=-1.0)
    xs = np.linspace(D.HOOD_REAR_X + 0.06, 2.56, 40)
    outline = [(x, -D.Y_GAP(x) + 0.06) for x in xs] + [(x, D.Y_GAP(x) - 0.06) for x in xs[::-1]]
    return outline_panel('RaptorR_Hood_InsulatorPad', outline, HS, h=0.06, hem=0.0, hem_outer=False, mat='Mat_Eng_Battery_Blanket', collection=col)
