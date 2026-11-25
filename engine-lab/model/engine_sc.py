"""1978 911 SC induction and exhaust. Coordinates are millimetres in the shared crank frame."""
import math

from mathutils import Vector as V
from kit import Geo, circle, fillet_path, frame, rounded_rect, smooth_path, TAU
import spec as S
from engine_core import bank_m
from engine_dress import (
    build_cooling,
    build_ignition,
    build_mounts,
    build_oil,
    gas,
    hose_clamp,
    pipe_part,
    port_point,
)


def induction(E):
    # The unfiltered chamber and filter lid lie across the car, immediately behind the fan.
    # Above the clean side, metered air goes over the rubber elbow and DOWN through the throttle.
    plenum = V((29, 0, 278))
    throttle = V((51, 38, 310))
    meter = V((90, -57, 319))
    g = Geo()
    g.box(plenum, (202, 166, 58), bevel=13, seg=3)
    for c in S.CYLS:
        b = c['bank']
        x = plenum.x + (c['x'] - (29.5 if b > 0 else -29.5)) * 0.57
        g.tube((x, b * 72, 278), (x, b * 105, 278), 23, 19, n=24)
    E.add('Intake_Manifold', g, 'plastic_black', 'induction', 'Intake plenum · feeds six runners', ex=(100, 0, 270))

    g = Geo()
    # A tapered lid and intake trumpet, with the rectangular filter below.
    g.sweep(
        [(-59, y, 350) for y in (-144, -133, -100, 5, 87, 130, 141)],
        rounded_rect(126, 70, 25, 6),
        scales=[0.62, 0.86, 1, 1, 0.9, 0.7, 0.50],
        up=(1, 0, 0),
    )
    snorkel = smooth_path([(-59, 102, 349), (-61, 152, 350), (-63, 199, 346), (-70, 251, 334)], 4)
    g.pipe(snorkel, 28, 25, n=32, scales=[1.18 - 0.18 * i / (len(snorkel) - 1) for i in range(len(snorkel))])
    g.box((-54, 0, 319), (140, 275, 12), bevel=5)
    E.add('Air_Cleaner', g, 'plastic_black', 'induction', 'Air cleaner lid and intake snorkel', ex=(-55, 0, 440))
    g = Geo()
    for k in range(27):
        g.box((-54, -124 + k * 9.5, 313), (122, 4, 21), bevel=1)
    E.add('Air_Filter', g, 'filter_paper', 'induction', 'Pleated air filter', ex=(-45, 0, 389))
    g = Geo()
    g.box((-41, 0, 290), (162, 272, 32), bevel=10, seg=2)
    g.box((-54, 0, 304), (140, 275, 7), bevel=3)
    E.add('Airbox_Base', g, 'plastic_black', 'induction', 'Air filter housing', ex=(-40, 0, 344))
    g = Geo()
    for side in (-83, 83):
        g.pipe(
            smooth_path(
                [
                    (-122, side, 309),
                    (-125, side, 349),
                    (-109, side, 378),
                    (-65, side, 385),
                    (-14, side, 380),
                    (6, side, 350),
                    (11, side, 319),
                ],
                3,
            ),
            4,
            n=10,
        )
    E.add('Air_Cleaner_Straps', g, 'rubber', 'induction', 'Air cleaner retaining straps', ex=(-80, 0, 484))

    g = Geo()
    g.tube(meter - V((0, 0, 29)), meter + V((0, 0, 23)), 42, 37, n=34)
    g.box((71, -103, 324), (70, 75, 36), bevel=8, seg=2)
    g.box((13, -77, 289), (88, 106, 26), bevel=8, seg=2)
    E.add('Airflow_Sensor', g, 'alu_cast', 'induction', 'K-Jetronic airflow meter', ex=(50, -80, 360))
    g = Geo()
    g.cyl((0, 0, -1.5), (0, 0, 1.5), 36, n=32)
    E.add(
        'Sensor_Plate',
        g,
        'steel',
        'induction',
        'Air metering plate',
        loc=meter,
        ex=(72, -84, 395),
        param=dict(p='airflow', axis=[1, 0, 0], pivot=[0, -42, 0], deg=-20),
    )
    g = Geo()
    g.tube(throttle - V((0, 0, 8)), throttle + V((0, 0, 36)), 36, 31, n=36)
    g.cyl(throttle + V((-45, 0, 14)), throttle + V((45, 0, 14)), 4, n=12)
    E.add('Throttle_Body', g, 'alu_cast', 'induction', 'Throttle body', ex=(30, 50, 365))
    g = Geo()
    g.cyl((0, 0, -1), (0, 0, 1), 30.5, n=32)
    E.add(
        'Throttle_Plate',
        g,
        'brass',
        'induction',
        'Throttle butterfly',
        loc=throttle + V((0, 0, 14)),
        ex=(35, 55, 399),
        param=dict(p='throttle', axis=[1, 0, 0], deg=78),
    )
    elbow = smooth_path(
        [
            meter + V((0, 0, 22)),
            meter + V((0, 0, 48)),
            V((76, -35, 385)),
            V((60, 21, 384)),
            throttle + V((0, 0, 59)),
            throttle + V((0, 0, 35)),
        ],
        5,
    )
    g = Geo()
    g.pipe(elbow, 35.5, 31, n=28)
    for i in range(3, len(elbow) - 2, 2):
        g.torus(elbow[i], elbow[i + 1] - elbow[i - 1], 35, 3.2, n=28, m=8)
    E.add('Metering_Boot', g, 'rubber', 'induction', 'Rubber air boot · meter to throttle', ex=(65, 0, 460))
    airpath = list(reversed(snorkel)) + [V((-54, 0, 351)), V((-54, 0, 310)), V((-4, -44, 291)), meter]
    gas(E, 'fresh_air', 'ambient', airpath + elbow + [throttle, plenum], r=21)

    distributor = V((84, -122, 342))
    g = Geo()
    g.cyl(distributor, distributor + V((0, 0, 27)), 31, n=32)
    g.cyl(distributor + V((0, 0, 27)), distributor + V((0, 0, 38)), 27, n=32)
    for c in S.CYLS:
        n, b = c['n'], c['bank']
        x = plenum.x + (c['x'] - (29.5 if b > 0 else -29.5)) * 0.57
        start = V((x, b * 101, 278))
        points = smooth_path(
            [
                start,
                V((x, b * 140, 277)),
                V((c['x'], b * 196, 263)),
                V((c['x'], b * 239, 221)),
                port_point(c, 60),
                port_point(c),
            ],
            6,
        )
        runner = Geo()
        runner.pipe(points, 22.5, 19, n=22)
        runner.at(frame(port_point(c), z=(0, b * 0.22, 0.975))).prism(
            rounded_rect(76, 56, 12, 4), 0, 6, holes=[circle(19, 24)]
        )
        E.add(
            f'Intake_Runner_{n}',
            runner,
            'alu_cast',
            'induction',
            f'Curved intake runner · cylinder {n}',
            ex=(0, b * 125, 260),
        )
        coupling = Geo()
        coupling.tube(start - V((0, b * 12, 0)), start + V((0, b * 21, 0)), 25, 21, n=24)
        for offset in (-9, 17):
            hose_clamp(coupling, start + V((0, b * offset, 0)), (0, b, 0), 24)
        E.add(
            f'Runner_Sleeve_{n}', coupling, 'rubber', 'induction', f'Plenum sleeve · runner {n}', ex=(0, b * 90, 315)
        )
        seat = S.VALVE['in']['seat']
        end = bank_m(b, c['x']) @ V((0, seat[0] + 6, seat[1] + 4))
        gas(E, f'intake_{n}', 'ambient', [plenum, start] + points[1:] + [end], cyl=n, r=15)

        injector = port_point(c, 25) + V((15, 0, 0))
        tip = injector + V((14, b * 6, 35))
        jet = Geo()
        jet.cyl(injector, tip, 5.7, n=12)
        jet.hexa(tip, (0, 0, 1), 12, 7)
        E.add(f'Injector_{n}', jet, 'brass', 'fuel', f'Continuous injector · cylinder {n}', ex=(0, b * 150, 310))
        a = TAU * (n - 1) / 6
        p = distributor + V((22 * math.cos(a), 22 * math.sin(a), 38))
        g.hexa(p, (0, 0, 1), 12, 8)
        pipe_part(
            E,
            f'FuelLine_{n}',
            [
                p + V((0, 0, 8)),
                p + V((0, 0, 25)),
                V((137 + n * 4, -67, 375 + n * 2)),
                V((c['x'] + 20, b * 166, 323 + n * 2)),
                tip + V((0, 0, 26)),
                tip + V((0, 0, 7)),
            ],
            2.6,
            'lead_black',
            'fuel',
            f'Fuel line · cylinder {n}',
            n=8,
            smooth=5,
            ex=(70, b * 55, 415),
        )
    E.add('Fuel_Distributor', g, 'steel_dark', 'fuel', 'K-Jetronic fuel distributor', ex=(100, -100, 410))


def exhaust(E):
    # Each three-into-one collector exits toward the front (flywheel) end. Both banks
    # discharge through the longitudinal pre-silencer on the left and rear transverse silencer.
    branches = {}
    outlets = {}
    for b, tag in [(1, 'L'), (-1, 'R')]:
        y = b * 240
        bank = [c for c in S.CYLS if c['bank'] == b]
        front = max(c['x'] for c in bank) + 53
        rear = min(c['x'] for c in bank) - 54
        shell = Geo()
        shell.sweep(
            [(rear, y, -171), (rear + 24, y, -171), (front - 24, y, -171), (front, y, -171)],
            rounded_rect(74, 112, 30, 5),
            scales=[0.65, 1, 1, 0.65],
            up=(0, 0, 1),
        )
        shell.tube((rear, y, -169), (rear - 35, y, -161), 27, 24, n=24)
        shell.tube((front - 25, y, -165), (front + 15, y, -79), 30, 27, n=24)
        E.add(
            'HeatExchanger_' + tag,
            shell,
            'alu_cover',
            'exhaust',
            f'Heater shell around exhaust headers · {tag}',
            ex=(0, b * 155, -310),
        )
        out = V((front - 8, b * 303, -171))
        outlets[b] = out
        for c in bank:
            n, x = c['n'], c['x']
            inlet = V((x, b * 238, -109))
            pts = smooth_path(
                [
                    inlet,
                    V((x, b * 238, -143)),
                    V((x + 13, b * 249, -169)),
                    V(((x + front) * 0.5, b * 260, -169)),
                    V((front - 12, b * 278, -171)),
                    out,
                ],
                4,
            )
            pipes = Geo()
            pipes.pipe(pts, 20, 17, n=18)
            pipes.at(frame(inlet, z=(0, 0, -1))).prism(rounded_rect(77, 54, 12, 4), 0, 7, holes=[circle(17, 20)])
            E.add(
                f'Header_{n}', pipes, 'steel_hot', 'exhaust', f'Exhaust primary · cylinder {n}', ex=(0, b * 90, -200)
            )
            seat = S.VALVE['ex']['seat']
            start = bank_m(b, x) @ V((0, seat[0] + 6, seat[1] - 4))
            branches[n] = [start] + pts
    junction = V((184, 345, -168))
    cross = pipe_part(
        E,
        'Crossover_Pipe',
        [outlets[-1], (198, -300, -168), (235, -230, -166), (235, 250, -166), junction],
        26,
        'exhaust',
        'exhaust',
        'Crossover · right bank to left bank',
        r_in=23,
        n=24,
        bend=33,
        ex=(100, -50, -300),
    )
    front_left = pipe_part(
        E,
        'Collector_Left',
        [outlets[1], junction],
        27,
        'exhaust',
        'exhaust',
        'Left-bank collector',
        r_in=24,
        n=24,
        ex=(50, 110, -255),
    )
    prem = fillet_path(
        [
            junction,
            V((152, 352, -168)),
            V((-155, 352, -168)),
            V((-257, 346, -163)),
            V((-340, 319, -150)),
            V((-389, 278, -145)),
        ],
        25,
    )
    g = Geo()
    g.pipe(prem, 27, 24, n=24)
    g.at(frame((135, 352, -168), z=(-1, 0, 0))).lathe(
        [(24, 0), (30, 0), (52, 28), (54, 230), (47, 258), (28, 286), (24, 286)], n=32, closed=True
    )
    E.add('Premuffler', g, 'exhaust', 'exhaust', 'Front silencer · both banks', ex=(0, 190, -235))
    mx, mz = -428, -147
    g = Geo()
    g.sweep(
        [(mx, y, mz) for y in (-289, -272, -233, 236, 275, 292)],
        rounded_rect(135, 147, 57, 7),
        scales=[0.59, 0.88, 1, 1, 0.88, 0.6],
        up=(0, 0, 1),
    )
    # The pressed shell seam and the two bands remain visible from underneath the car.
    g.box((mx - 73, 0, mz), (4, 476, 7), bevel=1)
    E.add('Muffler', g, 'steel_hot', 'exhaust', 'Rear silencer · single outlet', ex=(-270, 0, -172))
    bands = Geo()
    for y in (-94, 94):
        bands.at(frame((mx, y, mz), z=(0, 1, 0), x=(0, 0, 1)))
        bands.prism(rounded_rect(140, 152, 59, 7), 0, 14, holes=[rounded_rect(134, 146, 56, 7)])
        bands.at()
        bands.box((mx + 65, y + 7, -68), (18, 19, 83), bevel=2)
        bands.box((mx + 70, y + 7, -26), (43, 25, 8), bevel=2)
    E.add('Muffler_Straps', bands, 'steel', 'exhaust', 'Rear silencer mounting bands', ex=(-250, 0, -215))
    tail = pipe_part(
        E,
        'Tailpipe_L',
        [(mx, 259, mz), (mx - 37, 285, mz), (mx - 88, 292, mz - 8), (mx - 129, 300, mz - 16)],
        28,
        'alu_mach',
        'exhaust',
        'Single left tailpipe',
        r_in=25,
        n=28,
        bend=24,
        ex=(-340, 90, -175),
    )
    for c in S.CYLS:
        collectors = cross if c['bank'] < 0 else front_left
        gas(E, f'exhaust_{c["n"]}', 'exhaust', branches[c['n']] + collectors[1:], cyl=c['n'], r=14)
    gas(
        E,
        'exhaust_out',
        'exhaust_out',
        prem + [V((mx, 205, mz)), V((mx, 9, mz)), V((mx - 18, -192, mz)), V((mx - 29, 120, mz))] + tail,
        r=17,
    )


def build_all(E):
    build_cooling(E)
    build_oil(E)
    build_mounts(E)
    induction(E)
    exhaust(E)
    build_ignition(E)
