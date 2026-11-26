"""Side region: re-cut the door skins to Honda's P24 shut lines, add the FK8 side sculpting, fix the fuel door.

apply(ctx) runs after the body-shape FFD. Every position is derived from the current (post-FFD) geometry or
from the P24 side-plate linework below (plate px: X = 2.5 - u/1000, Y = 1.65 - v/1000, glTF metres).
Steps:
  1. sample the outer side surface |z|(X,Y) from the old door skins + fender + quarter (ray casts),
  2. rebuild the four door skins as Delaunay meshes inside the CAD outlines (front edge L1, B-cut L2,
     rear edge B, bottom BOT) with crease-aligned vertex rows, perimeter return flanges and shut gaps,
  3. add the sculpting as an outward displacement field (shoulder line C1, lower diagonal crease C2 with the
     concave scoop down to C4, rear sweep C3/C3b running through the rear handle onto the quarter),
  4. add body-colour sill / rear dog-leg panels below the (now higher) door bottoms,
  5. tuck the fender under the forward-moved front door edge, trim the fender vent to it,
  6. move/resize the fuel door to the CAD rectangle and re-cut its pocket.
"""
import bpy, bmesh, json, math
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

CAD = json.loads('{"L1":[[1712,700],[1695,708],[1676,717],[1660,733],[1652,752],[1638,766],[1628,786],[1620,808],[1614,828],[1607,846],[1606,866],[1604,886],[1603,906],[1602,926],[1601,946],[1600,966],[1600,986],[1600,1006],[1600,1026],[1600,1046],[1600,1066],[1601,1086],[1602,1106],[1602,1126],[1603,1146],[1604,1166],[1604,1186],[1605,1206],[1607,1226],[1610,1247],[1610,1267],[1614,1285],[1620,1306],[1628,1327],[1637,1348],[1648,1368],[1654,1376]],"L2":[[2794,600],[2790,621],[2786,641],[2782,662],[2774,682],[2768,703],[2765,723],[2758,744],[2753,764],[2746,785],[2744,809],[2743,829],[2742,849],[2740,869],[2739,889],[2738,909],[2736,929],[2735,949],[2734,969],[2733,989],[2732,1009],[2732,1029],[2731,1049],[2731,1069],[2731,1089],[2732,1110],[2732,1130],[2734,1150],[2736,1170],[2737,1190],[2740,1210],[2742,1230],[2744,1250],[2747,1270],[2750,1291],[2750,1311],[2754,1331],[2756,1368]],"B":[[3762,700],[3710,795],[3656,891],[3644,910],[3634,928],[3620,942],[3608,961],[3602,979],[3590,998],[3578,1016],[3568,1034],[3554,1051],[3542,1068],[3528,1085],[3515,1102],[3501,1120],[3487,1137],[3473,1154],[3458,1171],[3444,1188],[3429,1204],[3414,1220],[3401,1238],[3384,1251],[3368,1264],[3320,1292],[3285,1318],[3250,1338],[3215,1349]],"BOT":[[1665,1384],[1685,1381],[1705,1381],[1725,1380],[1745,1380],[1765,1380],[1785,1379],[1805,1379],[1825,1379],[1846,1378],[1866,1378],[1886,1378],[1906,1378],[1926,1377],[1946,1376],[1966,1376],[1986,1376],[2006,1376],[2026,1376],[2046,1375],[2066,1374],[2086,1374],[2106,1374],[2126,1374],[2146,1373],[2166,1373],[2186,1372],[2206,1372],[2226,1372],[2246,1372],[2266,1371],[2286,1371],[2306,1370],[2326,1370],[2346,1370],[2366,1370],[2386,1369],[2406,1369],[2426,1368],[2446,1368],[2466,1368],[2486,1368],[2506,1367],[2526,1366],[2546,1366],[2566,1366],[2586,1366],[2606,1365],[2626,1365],[2646,1364],[2666,1364],[2686,1364],[2706,1364],[2726,1363],[2746,1363],[2766,1362],[2786,1362],[2806,1362],[2826,1361],[2846,1361],[2866,1360],[2886,1360],[2906,1360],[2926,1359],[2946,1358],[2966,1358],[2986,1358],[3006,1358],[3026,1357],[3046,1356],[3066,1356],[3086,1356],[3106,1355],[3126,1355],[3146,1354],[3166,1354],[3186,1354],[3206,1353],[3226,1352],[3246,1350],[3260,1345]],"C1":[[1600,878],[1620,876],[1640,875],[1660,874],[1680,872],[1700,870],[1720,869],[1740,867],[1760,866],[1780,864],[1800,862],[1820,861],[1840,859],[1860,858],[1880,856],[1900,854],[1920,853],[1940,852],[1960,850],[1980,848],[2000,847],[2020,845],[2040,844],[2060,842],[2080,840],[2100,839],[2120,837],[2140,836],[2160,834],[2180,832],[2200,831],[2220,830],[2240,828],[2260,826],[2280,825],[2300,823],[2320,822],[2340,820],[2360,818],[2380,817],[2400,815],[2420,814],[2446,812],[2466,810],[2486,808],[2506,807],[2526,806],[2546,804],[2566,802],[2586,801],[2606,800],[2626,798],[2646,798],[2664,796],[2682,794],[2702,792],[2722,792],[2742,790],[2762,790],[2782,788],[2802,787],[2822,786],[2842,784],[2862,783],[2882,782],[2902,781],[2922,780],[2942,778],[2962,777],[2982,776],[3002,774],[3022,773],[3042,772],[3062,771],[3082,770],[3102,768],[3122,767],[3142,766],[3162,764],[3182,764],[3202,762],[3222,761],[3242,760],[3250,759]],"C2":[[1700,1366],[1715,1351],[1732,1338],[1752,1326],[1771,1316],[1791,1306],[1812,1300],[1832,1295],[1852,1290],[1872,1288],[1892,1286],[1912,1284],[1932,1282],[1952,1280],[1972,1277],[1992,1275],[2012,1273],[2032,1270],[2052,1268],[2072,1266],[2092,1264],[2112,1262],[2133,1260],[2153,1258],[2173,1255],[2193,1253],[2213,1251],[2233,1249],[2253,1246],[2273,1244],[2293,1242],[2313,1240],[2333,1238],[2353,1236],[2373,1234],[2394,1231],[2414,1229],[2434,1227],[2454,1225],[2474,1222],[2494,1220],[2514,1218],[2534,1216],[2554,1214],[2574,1212],[2594,1210],[2614,1208],[2635,1205],[2655,1203],[2675,1201],[2695,1198],[2715,1196],[2735,1193],[2755,1192],[2775,1190],[2795,1188],[2815,1186],[2835,1184],[2855,1182],[2875,1180],[2896,1178],[2916,1176],[2936,1174],[2956,1172],[2976,1170],[2996,1168],[3016,1166],[3036,1164],[3056,1162],[3076,1160],[3096,1158],[3117,1156],[3137,1154],[3157,1152],[3177,1150],[3197,1148],[3217,1145],[3237,1143],[3257,1141],[3277,1139],[3297,1137],[3317,1135],[3338,1133],[3358,1131],[3378,1129],[3398,1127],[3418,1126],[3438,1125],[3440,1125]],"C4":[[1775,1380],[1795,1372],[1815,1367],[1835,1363],[1855,1360],[1875,1358],[1896,1356],[1916,1356],[1936,1355],[1956,1354],[1976,1354],[1996,1352],[2016,1352],[2036,1351],[2056,1350],[2076,1350],[2096,1349],[2116,1348],[2136,1348],[2156,1347],[2176,1346],[2196,1346],[2216,1345],[2236,1344],[2256,1344],[2276,1343],[2296,1342],[2316,1342],[2336,1340],[2356,1340],[2376,1339],[2396,1338],[2416,1338],[2436,1337],[2456,1336],[2476,1336],[2496,1335],[2516,1334],[2536,1334],[2556,1333],[2576,1332],[2596,1332],[2616,1331],[2636,1330],[2656,1330],[2676,1328],[2696,1328],[2716,1327],[2736,1327],[2756,1326],[2776,1325],[2796,1324],[2816,1322],[2836,1322],[2856,1320],[2876,1320],[2896,1318],[2916,1316],[2936,1313],[2956,1310],[2976,1305],[2996,1300],[3167,1167],[3180,1150],[3183,1149]],"C3b":[[3365,749],[3385,744],[3405,742],[3426,739],[3446,735],[3466,733],[3487,732],[3507,730],[3527,728],[3547,726],[3573,722],[3591,721],[3611,720],[3631,718],[3652,716],[3672,713],[3692,710],[3712,708],[3732,706],[3752,704],[3772,700],[3792,700],[3812,700],[3832,700],[3852,700],[3872,700],[3892,700],[3912,700],[3932,700],[3950,700]],"C3":[[3047,876],[3100,855],[3150,836],[3200,818],[3250,801],[3300,781],[3333,767],[3365,750]]}')

GAP = 0.0022            # half shut gap (m)
H = 0.009               # base mesh spacing on door skins (m)
FLANGE = 0.016          # return flange depth (m)
HANDLE_DX = -0.048      # front door handle centre X -0.024 -> P24 -0.072
HANDLE_DY = -0.035      # and Y 0.840 -> 0.805: P24 / Honda renders put the grip ~5 cm below shoulder line C1
# B-pillar above the belt (P24 side plate, measured): X = a + b * (Y - 0.98) for the front glass rear edge,
# the door shut line (continuation of L2; its belt end tied to the re-cut skin corner at X -0.274) and the rear glass front edge
PILLAR = {'glass_f': (-0.1923, -0.2597), 'shut': (-0.2759, -0.2094), 'glass_r': (-0.3628, -0.135)}


# ----------------------------------------------------------------------------------------------- helpers
def plate(name):
    """Takes a CAD line name; returns its polyline as glTF (X, Y) metres (n,2)."""
    p = np.array(CAD[name], float)
    return np.stack([2.5 - p[:, 0] / 1000.0, 1.65 - p[:, 1] / 1000.0], 1)


def resample(poly, step):
    """Takes a polyline (n,2) and spacing; returns it resampled at ~uniform arclength spacing."""
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(2, int(round(s[-1] / step)) + 1)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])], 1)


def smooth_poly(poly, it=3):
    """Takes a polyline; returns it Laplacian-smoothed with fixed endpoints."""
    p = poly.copy()
    for _ in range(it):
        p[1:-1] = 0.25 * p[:-2] + 0.5 * p[1:-1] + 0.25 * p[2:]
    return p


def closest_on_poly(P, poly):
    """Takes points (n,2) and a polyline (m,2); returns (signed distance, arclength param, total length).
    Sign is + on the left of the polyline direction. Processes points in chunks."""
    if len(P) > 3000:
        parts = [closest_on_poly(P[i:i + 3000], poly) for i in range(0, len(P), 3000)]
        return np.concatenate([a for a, _, _ in parts]), np.concatenate([b for _, b, _ in parts]), parts[0][2]
    A = poly[:-1]; B = poly[1:]; AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-12)
    seg = np.linalg.norm(AB, axis=1); s0 = np.concatenate([[0], np.cumsum(seg)])[:-1]
    AP = P[:, None, :] - A[None]
    t = np.clip((AP * AB[None]).sum(2) / L2[None], 0, 1)
    C = A[None] + t[..., None] * AB[None]
    d2 = ((P[:, None, :] - C) ** 2).sum(2)
    k = np.argmin(d2, 1); i = np.arange(len(P))
    cr = AB[k, 0] * AP[i, k, 1] - AB[k, 1] * AP[i, k, 0]
    dist = np.sqrt(d2[i, k]) * np.sign(cr + 1e-15)
    return dist, s0[k] + t[i, k] * seg[k], seg.sum()


def inside(P, poly):
    """Takes points (n,2) and a closed polygon (m,2); returns bool mask (even-odd rule)."""
    if len(P) > 3000:
        return np.concatenate([inside(P[i:i + 3000], poly) for i in range(0, len(P), 3000)])
    x, y = P[:, 0][:, None], P[:, 1][:, None]
    x0, y0 = poly[:, 0][None], poly[:, 1][None]
    x1, y1 = np.roll(poly[:, 0], -1)[None], np.roll(poly[:, 1], -1)[None]
    c = ((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-15) + x0)
    return (c.sum(1) % 2) == 1


def poly_dist(P, poly):
    """Takes points and a closed polygon; returns unsigned distance to its boundary."""
    d, _, _ = closest_on_poly(P, np.vstack([poly, poly[:1]]))
    return np.abs(d)


def offset_poly(poly, g):
    """Takes a closed polygon (m,2) and distance g; returns it offset inward by g (per-vertex normals)."""
    area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    t = np.roll(poly, -1, 0) - np.roll(poly, 1, 0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    n = np.stack([-t[:, 1], t[:, 0]], 1) * (1 if area > 0 else -1)      # inward normal
    return poly + n * g


def smoothstep(e0, e1, x):
    """Takes edges and values; returns the cubic smoothstep."""
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def world_tris(ob):
    """Takes a mesh object; returns (glTF verts (n,3), triangles (m,3))."""
    me = ob.data; me.calc_loop_triangles()
    co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
    M = np.array(ob.matrix_world); co = co @ M[:3, :3].T + M[:3, 3]
    tri = np.empty(len(me.loop_triangles) * 3, np.int64); me.loop_triangles.foreach_get('vertices', tri)
    return np.stack([co[:, 0], co[:, 2], -co[:, 1]], 1), tri.reshape(-1, 3)


def bvh_left(obs):
    """Takes mesh objects; returns a BVH (glTF coords) of their left-side (z<0) triangles."""
    V, F = [], []; off = 0
    for ob in obs:
        v, f = world_tris(ob)
        f = f[v[f][:, :, 2].mean(1) < 0]
        V.append(v); F.append(f + off); off += len(v)
    V = np.concatenate(V); F = np.concatenate(F)
    return BVHTree.FromPolygons([tuple(p) for p in V], [tuple(t) for t in F])


def set_world_mesh(ob, Vgl, F, smooth=True):
    """Takes an object, glTF world verts and faces; replaces its mesh data (keeps materials). Returns nothing."""
    mats = list(ob.data.materials)
    me = bpy.data.meshes.new(ob.data.name + '_side')
    Vb = np.stack([Vgl[:, 0], -Vgl[:, 2], Vgl[:, 1]], 1)
    Mi = np.array(ob.matrix_world.inverted()); Vl = Vb @ Mi[:3, :3].T + Mi[:3, 3]
    me.from_pydata([tuple(p) for p in Vl], [], [tuple(int(i) for i in f) for f in F])
    for m in mats:
        me.materials.append(m)
    me.polygons.foreach_set('use_smooth', [smooth] * len(me.polygons))
    me.update(); ob.data = me


def new_object(name, Vgl, F, mat, parent):
    """Takes a name, glTF world verts/faces, a material name and parent object; returns the new object."""
    me = bpy.data.meshes.new(name)
    me.materials.append(bpy.data.materials[mat])
    ob = bpy.data.objects.new(name, me)
    (parent.users_collection[0] if parent.users_collection else bpy.context.scene.collection).objects.link(ob)
    ob.parent = parent; ob.matrix_world = Matrix.Identity(4)
    set_world_mesh(ob, Vgl, F)
    return ob


def mirror(V, F):
    """Takes left-side glTF verts/faces; returns the right-side copy (z flipped, winding flipped)."""
    W = V.copy(); W[:, 2] *= -1
    return W, F[:, ::-1].copy()


def get_verts(ob):
    """Takes a mesh object; returns its world verts in glTF axes (n,3)."""
    co = np.empty(len(ob.data.vertices) * 3); ob.data.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
    M = np.array(ob.matrix_world); co = co @ M[:3, :3].T + M[:3, 3]
    return np.stack([co[:, 0], co[:, 2], -co[:, 1]], 1)


def put_verts(ob, Vgl):
    """Takes a mesh object and new world glTF verts; writes them back in local coords. Returns nothing."""
    Vb = np.stack([Vgl[:, 0], -Vgl[:, 2], Vgl[:, 1]], 1)
    Mi = np.array(ob.matrix_world.inverted()); Vl = Vb @ Mi[:3, :3].T + Mi[:3, 3]
    ob.data.vertices.foreach_set('co', Vl.ravel()); ob.data.update()


# ----------------------------------------------------------------------------------------------- surface map
def _fill_nan(Z):
    """Takes a 2D array with NaNs; returns it with NaNs filled by repeated neighbour averaging."""
    Z = Z.copy()
    for _ in range(400):
        bad = np.isnan(Z)
        if not bad.any():
            break
        P = np.pad(Z, 1, constant_values=np.nan)
        nb = np.stack([P[:-2, 1:-1], P[2:, 1:-1], P[1:-1, :-2], P[1:-1, 2:]])
        cnt = (~np.isnan(nb)).sum(0); sm = np.nansum(nb, 0)
        fill = bad & (cnt > 0)
        Z[fill] = sm[fill] / cnt[fill]
    return np.nan_to_num(Z)


def _filt(Z, k, fn):
    """Takes a 2D array, odd window k and np.max/np.min; returns the sliding-window filtered array."""
    P = np.pad(Z, k // 2, mode='edge')
    return fn(np.lib.stride_tricks.sliding_window_view(P, (k, k)), axis=(2, 3))


def _blur(Z, sigma):
    """Takes a 2D array and sigma (cells); returns a separable Gaussian blur."""
    r = int(math.ceil(3 * sigma)); x = np.arange(-r, r + 1); w = np.exp(-x * x / (2 * sigma * sigma)); w /= w.sum()
    P = np.pad(Z, r, mode='edge')
    P = np.apply_along_axis(lambda v: np.convolve(v, w, 'valid'), 0, P)
    return np.apply_along_axis(lambda v: np.convolve(v, w, 'valid'), 1, P)


class Surface:
    """Outer side surface |z|(X,Y) sampled on a 4 mm grid from the given objects, shut-line gaps closed."""

    def __init__(self, obs, bands=(), mask_line=None, moves=(), x0=-2.0, x1=1.10, y0=0.15, y1=1.12, h=0.004):
        bvh = bvh_left(obs); self.bands = bands
        self.x0, self.y0, self.h = x0, y0, h
        xs = np.arange(x0, x1 + 1e-9, h); ys = np.arange(y0, y1 + 1e-9, h)
        Z = np.full((len(ys), len(xs)), np.nan)
        d = Vector((0, 0, 1))
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                loc, nrm, idx, dist = bvh.ray_cast(Vector((x, y, -1.6)), d, 1.2)
                if loc is not None and abs(nrm[2]) > 0.35:
                    Z[j, i] = -loc[2]
        if mask_line is not None:                          # old diagonal shut line B: in-paint across it
            gx, gy = np.meshgrid(xs, ys)
            G = np.stack([gx.ravel(), gy.ravel()], 1)
            near = (G[:, 0] > mask_line[:, 0].min() - 0.03) & (G[:, 0] < mask_line[:, 0].max() + 0.03) & \
                   (G[:, 1] > mask_line[:, 1].min() - 0.03) & (G[:, 1] < mask_line[:, 1].max() + 0.03)
            dd = np.full(len(G), 1.0); dd[near] = np.abs(closest_on_poly(G[near], mask_line)[0])
            Z[(dd < 0.016).reshape(Z.shape)] = np.nan
        self.ok = ~np.isnan(Z)
        Zf = _fill_nan(Z)
        Zc = _filt(_filt(Zf, 11, np.max), 11, np.min)      # grey closing: fills shut-line gaps < 44 mm wide
        Zf = np.where(Zc - Zf > 0.0015, Zc, Zf)
        for b0, b1 in self.bands:                          # bridge old vertical shut lines row by row
            i0, i1 = int((b0 - x0) / h), int((b1 - x0) / h)
            w = np.linspace(0, 1, i1 - i0 + 1)[None]
            Zf[:, i0:i1 + 1] = Zf[:, i0:i0 + 1] * (1 - w) + Zf[:, i1:i1 + 1] * w
        for mx0, mx1, my0, my1, dx, dy in moves:          # relocate a local feature (handle recess) by dx, dy
            i0, i1 = int((mx0 - x0) / h), int((mx1 - x0) / h); j0, j1 = int((my0 - y0) / h), int((my1 - y0) / h)
            w = np.linspace(0, 1, i1 - i0 + 1)[None]
            base = Zf[j0:j1 + 1, i0:i0 + 1] * (1 - w) + Zf[j0:j1 + 1, i1:i1 + 1] * w
            R = Zf[j0:j1 + 1, i0:i1 + 1] - base            # feature relief relative to the bridged panel
            Zf[j0:j1 + 1, i0:i1 + 1] = base
            s, t = int(round(dx / h)), int(round(dy / h))
            Zf[j0 + t:j1 + 1 + t, i0 + s:i1 + 1 + s] += R
        self.Z = _blur(Zf, 3.0)                           # panels are smooth at 12 mm; creases come from Sculpt

    def __call__(self, X, Y):
        """Takes arrays X, Y; returns outward |z| by bilinear interpolation."""
        gy = np.clip((np.asarray(Y, float) - self.y0) / self.h, 0, self.Z.shape[0] - 1.001)
        gx = np.clip((np.asarray(X, float) - self.x0) / self.h, 0, self.Z.shape[1] - 1.001)
        i, j = gy.astype(int), gx.astype(int); fy, fx = gy - i, gx - j
        Z = self.Z
        return (Z[i, j] * (1 - fx) * (1 - fy) + Z[i, j + 1] * fx * (1 - fy)
                + Z[i + 1, j] * (1 - fx) * fy + Z[i + 1, j + 1] * fx * fy)


# ----------------------------------------------------------------------------------------------- sculpting
class Sculpt:
    """Outward displacement field of the FK8 side feature lines (metres, + = outboard)."""

    def __init__(self):
        self.C1 = smooth_poly(resample(plate('C1'), 0.01))
        self.C2 = smooth_poly(resample(plate('C2'), 0.01))
        self.C4 = smooth_poly(resample(plate('C4'), 0.01))
        c3b = plate('C3b'); c3b = c3b[c3b[:, 0] > 2.5 - 3.752]           # stop at the rear door edge B:
        c3 = np.vstack([plate('C3'), c3b[1:]])                              # the quarter mesh is too coarse
        self.C3 = smooth_poly(resample(c3, 0.01), 6)
        self.xj = self.C4[-1, 0]                           # C4 meets C2 here (rear door)

    @staticmethod
    def ridge(d, A, w, e=0.003):
        """Takes signed distances; returns a rounded-tip ridge profile of height A and half-width w."""
        r = np.sqrt(d * d + e * e) - e
        return A * np.clip(1 - r / w, 0, 1) ** 2

    def __call__(self, X, Y):
        """Takes arrays X, Y; returns outward displacement."""
        P = np.stack([X, Y], 1); out = np.zeros(len(P))
        # C1 shoulder line (front door -> rear door), fades at both ends
        d, s, L = closest_on_poly(P, self.C1)
        out += self.ridge(d, 0.006, 0.03) * smoothstep(0, 0.06, s) * smoothstep(0, 0.25, L - s)
        # C3 rear sweep: rises across the rear door, through the rear handle, along the quarter
        d, s, L = closest_on_poly(P, self.C3)
        out += self.ridge(d, 0.005, 0.035) * smoothstep(0, 0.20, s) * smoothstep(0, 0.05, L - s)
        # C2 lower diagonal crease with the concave scoop below it, down to C4
        d2, s2, L2 = closest_on_poly(P, self.C2)          # C2 runs front->rear: below = right = d2 < 0
        d4, s4, L4 = closest_on_poly(P, self.C4)          # below C4 = d4 < 0
        below2 = np.clip(-d2, 0, None); above4 = np.clip(d4, 0, None)
        wedge = (d2 < 0) & (d4 > 0) & (X > self.xj) & (X < self.C4[0, 0] + 0.03)
        depth = 0.014 * smoothstep(0, 0.018, below2) * smoothstep(0, 0.06, above4)
        out -= np.where(wedge, depth, 0.0)
        # behind the C4 junction the crease continues as a softer step down to the arch
        tail = (d2 < 0)
        tdepth = 0.006 * smoothstep(0, 0.015, below2) * (1 - smoothstep(0.03, 0.09, below2))
        tdepth *= smoothstep(self.xj + 0.02, self.xj - 0.10, X)        # ramps in behind the C4 junction
        tdepth *= smoothstep(0, 0.06, L2 - s2)
        out -= np.where(tail, tdepth, 0.0)
        # small ridge on the C2 edge itself so the crease catches light
        out += self.ridge(d2, 0.002, 0.02) * smoothstep(0, 0.08, s2) * smoothstep(0, 0.06, L2 - s2)
        return out

    def rows(self, poly_clip):
        """Takes a door outline; returns extra vertex rows (k,2) laid along the crease lines inside it."""
        pts = []
        for line, offs in ((self.C1, (0, .003, -.003, .008, -.008, .015, -.015, .024, -.024)),
                           (self.C3, (0, .003, -.003, .008, -.008, .015, -.015, .024, -.024)),
                           (self.C2, (0, .003, -.003, .007, -.007, .012, -.012, .019, -.019, -.028)),
                           (self.C4, (0, .008, -.008, .02, -.02))):
            r = resample(line, 0.006)
            t = np.gradient(r, axis=0); t /= np.linalg.norm(t, axis=1, keepdims=True)
            n = np.stack([-t[:, 1], t[:, 0]], 1)
            for o in offs:
                pts.append(r + n * o)
        pts = np.concatenate(pts)
        keep = inside(pts, poly_clip) & (poly_dist(pts, poly_clip) > 0.004)
        return pts[keep]


# ----------------------------------------------------------------------------------------------- panel mesh
def panel_mesh(outline, surf, sculpt, extra=None, h=H, flange=FLANGE, edge_roll=0.0012):
    """Takes a closed outline (m,2), surface, sculpt field and optional extra vertex rows.
    Returns left-side (V (n,3) glTF, F (k,3)) of the skin + perimeter return flange."""
    from mathutils.kdtree import KDTree
    from mathutils.geometry import delaunay_2d_cdt
    B = resample(np.vstack([outline, outline[:1]]), h * 0.6)[:-1]
    nb = len(B)
    lo = outline.min(0); hi = outline.max(0)
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0], h), np.arange(lo[1], hi[1], h))
    G = np.stack([gx.ravel(), gy.ravel()], 1)
    G = G[inside(G, outline)]
    G = G[poly_dist(G, outline) > 0.7 * h]
    pts = [B]
    if extra is not None and len(extra):
        kd = KDTree(len(extra))
        for i, p in enumerate(extra):
            kd.insert((p[0], p[1], 0), i)
        kd.balance()
        G = G[np.array([kd.find((p[0], p[1], 0))[2] > 0.6 * h for p in G], bool)]
        pts.append(G)
        pts.append(extra[poly_dist(extra, outline) > 0.35 * h])
    else:
        pts.append(G)
    P = np.vstack(pts)
    kd = KDTree(len(P))
    for i, p in enumerate(P):
        kd.insert((p[0], p[1], 0), i)
    kd.balance()
    keep = np.ones(len(P), bool)
    for i in range(nb, len(P)):
        if keep[i]:
            for (_, j, dd) in kd.find_range((P[i, 0], P[i, 1], 0), 0.0015):
                if j != i and j >= nb and j > i:
                    keep[j] = False
    P = P[keep]
    res = delaunay_2d_cdt([Vector(p) for p in P], [], [list(range(nb))], 1, 1e-7)
    P = np.array([tuple(v) for v in res[0]])
    tri = np.array([f for f in res[2] if len(f) == 3])
    nb = nb  # boundary verts keep their indices (first nb) unless merged
    z = surf(P[:, 0], P[:, 1]) + sculpt(P[:, 0], P[:, 1])
    z[:nb] -= edge_roll                                    # rolled panel edge reads as a shut line
    V = np.stack([P[:, 0], P[:, 1], -z], 1)
    # orient skin outward (-z normal on the left side)
    n = np.cross(V[tri[:, 1]] - V[tri[:, 0]], V[tri[:, 2]] - V[tri[:, 0]])
    tri[n[:, 2] > 0] = tri[n[:, 2] > 0][:, ::-1]
    # return flange: boundary ring pushed inboard
    Bi = offset_poly(B, 0.0015)
    zf = z[:nb] - flange
    Vf = np.stack([Bi[:, 0], Bi[:, 1], -zf], 1)
    V = np.vstack([V, Vf]); off = len(P)
    q = []
    area = 0.5 * np.sum(B[:, 0] * np.roll(B[:, 1], -1) - np.roll(B[:, 0], -1) * B[:, 1])
    for i in range(nb):
        j = (i + 1) % nb
        a, b, cc, d = i, j, off + j, off + i
        q += [(a, d, cc), (a, cc, b)] if area > 0 else [(a, b, cc), (a, cc, d)]
    F = np.vstack([tri, np.array(q)])
    return V, F


def old_top(ob, x0, x1):
    """Takes an old door skin and an X range; returns its top edge polyline (X, Ymax) front->rear order."""
    V = get_verts(ob); V = V[V[:, 2] < 0] if V[:, 2].mean() < 0 else V
    xs = np.arange(x0, x1 + 1e-9, 0.01)
    ys = []
    for x in xs:
        m = np.abs(V[:, 0] - x) < 0.02                    # upper envelope over a +-20 mm window
        ys.append(V[m, 1].max() if m.any() else np.nan)
    ys = np.array(ys); ok = ~np.isnan(ys)
    ys = np.interp(xs, xs[ok], ys[ok])
    ys = np.convolve(np.pad(ys, 2, mode='edge'), np.ones(5) / 5, 'valid')
    return np.stack([xs, ys - 0.001], 1)


def clip_top(line, top):
    """Takes a polyline and the top-edge polyline; returns the part of the line below the top edge."""
    yt = np.interp(line[:, 0], top[:, 0], top[:, 1])
    return line[line[:, 1] <= yt - 0.001]


def tuck_and_snap(ob, poly, full, depth=0.012, snap=True):
    """Takes a body panel, a cut outline (already widened by the shut gap) and the outer surface function.
    Outer-skin verts inside the outline are pushed inboard by depth; outer verts just outside that share an
    edge with a tucked vert are snapped onto the outline (on the surface), giving a clean shut line.
    Works on both sides (mirror). Returns nothing."""
    V = get_verts(ob); me = ob.data
    E = np.empty(len(me.edges) * 2, np.int64); me.edges.foreach_get('vertices', E); E = E.reshape(-1, 2)
    outer = np.abs(V[:, 2]) > 0.70
    P = V[:, :2]
    near = outer & (P[:, 0] > poly[:, 0].min() - 0.05) & (P[:, 0] < poly[:, 0].max() + 0.05) & \
        (P[:, 1] > poly[:, 1].min() - 0.05) & (P[:, 1] < poly[:, 1].max() + 0.05)
    ins = np.zeros(len(V), bool); ins[near] = inside(P[near], poly)
    cross = ins[E[:, 0]] != ins[E[:, 1]]
    out_v = np.unique(np.where(ins[E[cross, 0]], E[cross, 1], E[cross, 0]))
    out_v = out_v[outer[out_v]]
    if len(out_v) and snap:
        Q = np.vstack([poly, poly[:1]])
        A = Q[:-1]; AB = Q[1:] - A
        X = P[out_v]
        t = np.clip((((X[:, None] - A[None]) * AB[None]).sum(2)) / np.maximum((AB ** 2).sum(1), 1e-12)[None], 0, 1)
        C = A[None] + t[..., None] * AB[None]
        k = np.argmin(((X[:, None] - C) ** 2).sum(2), 1)
        S = C[np.arange(len(X)), k]
        ok = np.linalg.norm(S - X, axis=1) < 0.025
        idx = out_v[ok]; S = S[ok]
        sg = np.sign(V[idx, 2])
        V[idx, 0], V[idx, 1] = S[:, 0], S[:, 1]
        V[idx, 2] = sg * (full(S[:, 0], S[:, 1]) - 0.0012)
    V[ins, 2] -= np.sign(V[ins, 2]) * depth
    put_verts(ob, V)
    print('side: %s tucked %d snapped %d' % (ob.name, ins.sum(), len(out_v)))


# ----------------------------------------------------------------------------------------------- main
def move_front_door_hinges(ctx, x_hinge=0.885):
    """Takes the build context; moves Pivot_Door_L/R forward to the re-cut front door edge (P24 L1 sits at
    X ~0.90 between hinge heights) so the door swings about its real hinge line instead of through the
    fender. Children keep their world placement. Returns nothing."""
    from mathutils import Matrix
    for name in ('Pivot_Door_L', 'Pivot_Door_R'):
        pv = ctx.obj(name)
        kids = [(c, c.matrix_world.copy()) for c in pv.children]
        M = pv.matrix_world.copy(); M.translation.x = x_hinge          # Blender x == glTF x (forward)
        pv.matrix_world = M
        bpy.context.view_layer.update()
        for c, Mw in kids:
            c.matrix_world = Mw


def pillar_x(name, Y):
    """Takes a PILLAR line name and heights Y; returns the line's X at those heights (Y clamped to the DLO)."""
    a, b = PILLAR[name]
    return a + b * (np.clip(Y, 0.93, 1.36) - 0.98)


def warp_front_sash(V, ramp=0.30):
    """Takes front-door frame/glass glTF verts; returns them with the B-pillar sash re-leaned: the sash face
    (X -0.222..-0.196) stretched from the shut line to the glass lip, the glass lip (X -0.196..-0.171) moved
    rigidly onto the P24 front glass edge, and everything ahead of it eased back to rest over ramp metres."""
    X, Y = V[:, 0], V[:, 1]
    tr = pillar_x('shut', Y) + 0.002; tf = pillar_x('glass_f', Y)
    face = tr + (X + 0.222) * ((tf - 0.025) - tr) / 0.026
    lip = tf + (X + 0.171)
    ahead = X + (tf + 0.171) * (1 - smoothstep(-0.171, -0.171 + ramp, X))
    W = V.copy(); W[:, 0] = np.where(X <= -0.196, face, np.where(X <= -0.171, lip, ahead))
    return W


def warp_rear_sash(V, ramp=0.30):
    """Takes rear-door frame/glass glTF verts; returns them with the B-pillar sash re-leaned: the sash face
    (X -0.246..-0.218) stretched from the shut line to the glass lip, the lip (X -0.275..-0.246) moved rigidly
    onto the P24 rear glass front edge, and everything behind it eased back to rest over ramp metres."""
    X, Y = V[:, 0], V[:, 1]
    tf = pillar_x('shut', Y) - 0.002; tr = pillar_x('glass_r', Y)
    face = (tr + 0.029) + (X + 0.246) * (tf - tr - 0.029) / 0.028
    lip = X + (tr + 0.275)
    behind = X + (tr + 0.275) * (1 - smoothstep(-0.275, -0.275 - ramp, X))
    W = V.copy(); W[:, 0] = np.where(X >= -0.246, face, np.where(X >= -0.275, lip, behind))
    return W


def lean_b_pillar(ctx):
    """Takes the build context; re-leans the B-pillar above the belt (door frames + door glass, both sides)
    so it continues the re-cut L2 shut line instead of standing vertical at X -0.22. Returns nothing."""
    for tag, fn in (('L', warp_front_sash), ('R', warp_front_sash), ('RL', warp_rear_sash), ('RR', warp_rear_sash)):
        for pre in ('Civic_Door_Frame_', 'Civic_Glass_Door_'):
            ob = ctx.obj(pre + tag)
            put_verts(ob, fn(get_verts(ob)))


def apply(ctx):
    """Takes the build context; rebuilds the side region in place. Returns nothing."""
    apply_round1(ctx)
    lean_b_pillar(ctx)
    move_front_door_hinges(ctx)


def apply_round1(ctx):
    """Takes the build context; round-1 side rebuild (doors, sculpting, sills, fender, fuel door). Returns nothing."""
    O = ctx.obj
    fd_old, rd_old = O('Civic_Door_Shell_L'), O('Civic_Door_Shell_RL')
    fender, quarter = O('Civic_Body_Shell_FrontFender'), O('Civic_Body_Shell_RearQuarter')
    xa = get_verts(fd_old)[:, 0]; xr = get_verts(rd_old)[:, 0]
    bands = ((xa.max() - 0.028, xa.max() + 0.03), (xa.min() - 0.025, xa.min() + 0.025))   # old front / B shut lines
    surf = Surface([fd_old, rd_old, fender, quarter], bands, resample(plate('B'), 0.005),
                   moves=[(-0.140, 0.092, 0.772, 0.904, HANDLE_DX, HANDLE_DY)])     # front handle recess follows the handle
    sculpt = Sculpt()

    # ---- door outlines (glTF X,Y, left side)
    top_f = old_top(fd_old, -0.36, 0.95)
    top_r = old_top(rd_old, -1.32, -0.15)
    L1 = smooth_poly(resample(plate('L1'), 0.006), 4)
    L2 = smooth_poly(resample(plate('L2'), 0.006), 6)
    Bl = smooth_poly(resample(plate('B'), 0.006), 30)          # B crosses the flare arc: smooth trace hops
    BOT = smooth_poly(resample(plate('BOT'), 0.006), 6)
    L1 = clip_top(L1, top_f)
    L2f = clip_top(L2, top_f); L2r = clip_top(L2, top_r)
    xb2 = L2[-1, 0]
    bot_f = BOT[(BOT[:, 0] < L1[-1, 0] - 0.004) & (BOT[:, 0] > xb2 + 0.002)]
    bot_r = BOT[(BOT[:, 0] < xb2 - 0.002) & (BOT[:, 0] > Bl[-1, 0] + 0.004)]
    tf = top_f[(top_f[:, 0] > L2f[0, 0] + 0.003) & (top_f[:, 0] < L1[0, 0] - 0.003)]
    # front door: L1 top->bottom, bottom front->rear, L2 bottom->top, top rear->front
    fpoly = np.vstack([L1, bot_f, L2f[::-1], tf])
    # rear door: B extended straight up to the old top
    yb_top = np.interp(Bl[0, 0], top_r[:, 0], top_r[:, 1])
    up = np.array([[Bl[0, 0] - 0.002, y] for y in np.arange(yb_top - 0.002, Bl[0, 1], -0.01)])
    tr = top_r[(top_r[:, 0] < L2r[0, 0] - 0.003) & (top_r[:, 0] > Bl[0, 0] + 0.003)]
    rpoly = np.vstack([L2r, bot_r, Bl[::-1], up[::-1], tr])
    fpoly = offset_poly(resample(np.vstack([fpoly, fpoly[:1]]), 0.004)[:-1], GAP)
    rpoly = offset_poly(resample(np.vstack([rpoly, rpoly[:1]]), 0.004)[:-1], GAP)
    print('side: front door X %.3f..%.3f Y %.3f..%.3f' % (fpoly[:, 0].min(), fpoly[:, 0].max(), fpoly[:, 1].min(), fpoly[:, 1].max()))
    print('side: rear  door X %.3f..%.3f Y %.3f..%.3f' % (rpoly[:, 0].min(), rpoly[:, 0].max(), rpoly[:, 1].min(), rpoly[:, 1].max()))

    # ---- rebuild door skins
    for tag, poly in (('L', fpoly), ('RL', rpoly)):
        V, F = panel_mesh(poly, surf, sculpt, sculpt.rows(poly))
        set_world_mesh(O('Civic_Door_Shell_' + tag), V, F)
        Vr, Fr = mirror(V, F)
        set_world_mesh(O('Civic_Door_Shell_' + {'L': 'R', 'RL': 'RR'}[tag]), Vr, Fr)
        print('side: door', tag, len(V), 'verts', len(F), 'tris')

    # ---- sill + rear dog-leg panel below the doors
    wc = np.array([-1.3484, 0.3195]); r_arch = 0.445
    xf_s = L1[-1, 0] - 0.002
    bot_s = BOT[(BOT[:, 0] < xf_s) & (BOT[:, 0] > Bl[-1, 0])] + [0, -2 * GAP]
    bseg = Bl[::-1]                                         # bottom -> top
    rr = np.linalg.norm(bseg - wc, axis=1)
    k = np.argmax(rr < r_arch) if (rr < r_arch).any() else len(bseg) - 1
    bup = offset_poly(np.vstack([bseg[:k], bseg[k - 1:k] + [0.0, -0.5]]), -GAP)[:-1] if k > 1 else bseg[:1]
    p_end = bup[-1]
    a0 = math.atan2(p_end[1] - wc[1], p_end[0] - wc[0])
    ya = 0.20
    a1 = math.asin((ya - wc[1]) / r_arch)                  # front side of the rear arch
    arc = np.array([wc + r_arch * np.array([math.cos(a), math.sin(a)]) for a in np.linspace(a0, a1, 16)])
    spoly = np.vstack([bot_s, bup, arc, [[xf_s, ya]]])
    spoly = resample(np.vstack([spoly, spoly[:1]]), 0.004)[:-1]
    V, F = panel_mesh(spoly, surf, lambda X, Y: np.zeros(len(X)), None, h=0.01, flange=0.012)
    body = O('Civic_Body_Shell_Rocker').parent
    for tag, (VV, FF) in (('L', (V, F)), ('R', mirror(V, F))):
        nm = 'Civic_Body_Shell_Sill_' + tag
        if nm in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[nm])
        new_object(nm, VV, FF, fd_old.data.materials[0].name, body)

    # ---- tuck body panels under the new door / sill skins and snap their cut edge onto the shut line
    tuck_and_snap(fender, offset_poly(fpoly, -2 * GAP), surf)          # front door edge moved forward 6 cm
    tuck_and_snap(fender, offset_poly(spoly, -2 * GAP), surf)
    # rear quarter untouched along B: the old rear door edge already followed B and the quarter sits under it

    # ---- fender vent / gills: compress in X so they end ahead of the new door front edge
    for nm in ('Civic_Fender_Vent', 'Civic_Fender_Side_Gills'):
        ob = O(nm); Vb = get_verts(ob)
        xe = np.interp(Vb[:, 1], L1[::-1, 1], L1[::-1, 0]) + 0.004
        xf, xmin = Vb[:, 0].max(), Vb[:, 0].min()
        s = np.clip((xf - xe) / max(xf - xmin, 1e-6), 0, 1)
        Vb[:, 0] = xf - (xf - Vb[:, 0]) * s
        put_verts(ob, Vb)

    # ---- door handles ride on the new skin (shift outward by the local sculpt displacement); the front
    #      handles move rearward to the P24 position (recess moved with them in Surface)
    for tag in ('L', 'R', 'RL', 'RR'):
        ob = O('Civic_Door_Handle_' + tag); Vb = get_verts(ob); c = Vb.mean(0)
        front = tag in ('L', 'R')
        dz = sculpt(np.array([c[0] + (HANDLE_DX if front else 0)]), np.array([c[1] + (HANDLE_DY if front else 0)]))[0]
        if front:                                          # base-surface change, sampled just above the recess
            dz += (surf(np.array([c[0] + HANDLE_DX]), np.array([0.912 + HANDLE_DY]))
                   - surf(np.array([c[0]]), np.array([0.912])))[0]
            Vb[:, 0] += HANDLE_DX; Vb[:, 1] += HANDLE_DY
        Vb[:, 2] += -np.sign(c[2]) * dz
        put_verts(ob, Vb)

    fuel_door(ctx, surf, sculpt)


def rounded_rect(x0, x1, y0, y1, rad, n=8):
    """Takes rectangle bounds and corner radius; returns a CCW closed outline (4n,2)."""
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1); hx, hy = 0.5 * (x1 - x0) - rad, 0.5 * (y1 - y0) - rad
    out = []
    for q, (sx, sy) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
        for a in np.linspace(q * math.pi / 2, (q + 1) * math.pi / 2, n):
            out.append((cx + sx * hx + rad * math.cos(a), cy + sy * hy + rad * math.sin(a)))
    return np.array(out)


def fuel_door(ctx, surf, sculpt):
    """Takes context, surface and sculpt; moves/resizes the fuel flap to the P24 rounded rectangle, re-cuts
    the pocket in the quarter, adds a dark pocket liner, moves the cap and the hinge pivot. Returns nothing."""
    O = ctx.obj
    x0, x1, y0, y1 = -1.572, -1.339, 0.850, 1.006           # P24 side plate fuel door (glTF m)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    loop = resample(np.vstack([rounded_rect(x0, x1, y0, y1, 0.024)] * 1), 0.004)
    loop = resample(np.vstack([loop, loop[:1]]), 0.004)[:-1]
    full = lambda X, Y: surf(X, Y) + sculpt(X, Y)
    flat = lambda X, Y: np.zeros(len(X))
    V, F = panel_mesh(offset_poly(loop, 0.0015), lambda X, Y: full(X, Y) + 0.0025, flat, None, h=0.010, flange=0.006)
    flap = O('Civic_Fuel_Flap'); pv = O('Pivot_Fuel_Flap')
    keep = {c.name: c.matrix_world.copy() for c in pv.children}
    zc = full(np.array([x1]), np.array([cy]))[0]
    new = Vector((x1 + 0.002, zc - 0.012, cy))               # Blender axes: (x, -z, y)
    pv.matrix_world = Matrix.Translation(new - pv.matrix_world.translation) @ pv.matrix_world
    bpy.context.view_layer.update()
    for c in pv.children:
        c.matrix_world = keep[c.name]
    bpy.context.view_layer.update()
    set_world_mesh(flap, V, F)
    quarter = O('Civic_Body_Shell_RearQuarter')
    # the quarter is too coarse (18 mm tris) to cut cleanly: the opening is a dark liner laid on the skin,
    # the liner sits 1.5 mm proud of the coarse skin (no z-fighting) and the flap 2.5 mm; the old flap area needs no repair (the quarter was never cut there)
    Vp, Fp = panel_mesh(offset_poly(loop, 0.0002), lambda X, Y: full(X, Y) + 0.0015, flat, None, h=0.015, flange=0.003)
    nm = 'Civic_Body_Shell_Fuel_Pocket'
    if nm in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[nm])
    new_object(nm, Vp, Fp, 'Mat_Aero_Matte_Black', quarter.parent)
    cap = O('Civic_Fuel_Filler_Cap'); Vc = get_verts(cap); cc = Vc.mean(0)
    Vc[:, 0] += cx - cc[0]; Vc[:, 1] += cy - cc[1]
    zs = full(np.array([cx]), np.array([cy]))[0]
    Vc[:, 2] += (-(zs + 0.0018)) - Vc[:, 2].min()          # cap face just above the liner
    put_verts(cap, Vc)
