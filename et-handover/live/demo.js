      'use strict';
      /* ================================================================
         Part 1 — deterministic toy-example timeline (mirrors the paper's
         Appendix toy schematic: 3 FOV discs r=2.5, A(0,1.5) B(-1.5,-1)
         C(1.5,-1), smooth trajectory through 15 waypoints).
         ================================================================ */
      const R_FOV = 2.5;
      const CENTERS = { A: [0, 1.5], B: [-1.5, -1], C: [1.5, -1] };
      const BS_IDS = ['A', 'B', 'C'];
      const BS_COLOR = { A: '#2a78d6', B: '#eb6834', C: '#1baf7a' };
      const BS_DARK = { A: '#1c5cab', B: '#a53e13', C: '#0f7250' };

      const WAYPOINTS = [
        [-3.35, 3.80], [-2.65, 3.05], [-1.95, 2.20], [-1.25, 1.40], [-0.60, 0.80],
        [-0.05, 0.38], [0.45, 0.18], [0.95, 0.02], [1.35, -0.18], [1.78, -0.55],
        [2.10, -1.10], [2.35, -1.90], [2.62, -2.95], [2.95, -4.20], [3.30, -5.00],
      ];
      const N_FRAMES = 360, PATH_END = 0.94, LOOKAHEAD = 0.55;
      const R_MIN = 0.6, MU_SCALE = 7, MU_CAP = 8, LAMBDA_TH = 2;
      const EXT_A = 0.40, EXT_B = 0.20; // target extent semi-axes (world units)
      const BYTES = { prior: 196, posterior: 196, meas: 16, ctrl: 24 };

      // catmullRom: sample a uniform Catmull-Rom spline through pts.
      // Takes waypoints and samples-per-segment; returns a dense [x,y] polyline.
      function catmullRom(pts, per) {
        const out = [];
        for (let i = 0; i < pts.length - 1; i++) {
          const p0 = pts[Math.max(0, i - 1)], p1 = pts[i],
                p2 = pts[i + 1], p3 = pts[Math.min(pts.length - 1, i + 2)];
          for (let j = 0; j < per; j++) {
            const t = j / per, t2 = t * t, t3 = t2 * t;
            out.push([0, 1].map(k =>
              0.5 * ((2 * p1[k]) + (-p0[k] + p2[k]) * t +
              (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2 +
              (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3)));
          }
        }
        out.push(pts[pts.length - 1].slice());
        return out;
      }

      // resampleByArc: re-space a polyline into n equal arc-length points over
      // [0, endFrac] of its total length. Returns an [x,y] list of length n.
      function resampleByArc(poly, n, endFrac) {
        const cum = [0];
        for (let i = 1; i < poly.length; i++) {
          cum.push(cum[i - 1] + Math.hypot(poly[i][0] - poly[i - 1][0], poly[i][1] - poly[i - 1][1]));
        }
        const total = cum[cum.length - 1] * endFrac;
        const out = [];
        let j = 0;
        for (let i = 0; i < n; i++) {
          const s = total * i / (n - 1);
          while (j < cum.length - 2 && cum[j + 1] < s) j++;
          const seg = cum[j + 1] - cum[j] || 1;
          const t = (s - cum[j]) / seg;
          out.push([poly[j][0] + (poly[j + 1][0] - poly[j][0]) * t,
                    poly[j][1] + (poly[j + 1][1] - poly[j][1]) * t]);
        }
        return out;
      }

      const dist = (p, c) => Math.hypot(p[0] - c[0], p[1] - c[1]);

      // visFrac: fraction of the extent footprint visible to station s — the
      // delta factor of the detection model. Takes center [x,y], heading, and a
      // station id; returns a number in [0,1] from a grid sample of the ellipse.
      function visFrac(pos, heading, s) {
        const c = CENTERS[s], cos = Math.cos(heading), sin = Math.sin(heading);
        let inside = 0, total = 0;
        for (let i = -3; i <= 3; i++) {
          for (let j = -3; j <= 3; j++) {
            const u = i / 3, v = j / 3;
            if (u * u + v * v > 1) continue;
            total++;
            const x = pos[0] + u * EXT_A * cos - v * EXT_B * sin;
            const y = pos[1] + u * EXT_A * sin + v * EXT_B * cos;
            if (Math.hypot(x - c[0], y - c[1]) <= R_FOV) inside++;
          }
        }
        return total ? inside / total : 0;
      }

      // muAt: expected detection count mu = rho(r) * A * delta at station s —
      // the paper's range- and visibility-aware detection model. Takes a
      // position, heading, and station id; returns a non-negative number.
      function muAt(pos, heading, s) {
        const r = Math.max(R_MIN, dist(pos, CENTERS[s]));
        return MU_SCALE * visFrac(pos, heading, s) / r;
      }

      // mulberry32: deterministic PRNG so measurement dots are stable when
      // scrubbing. Takes an integer seed; returns a () => float in [0,1).
      function mulberry32(seed) {
        let a = seed >>> 0;
        return function () {
          a |= 0; a = (a + 0x6D2B79F5) | 0;
          let t = Math.imul(a ^ (a >>> 15), 1 | a);
          t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
          return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
      }

      // gauss: one standard normal drawn from prng via Box-Muller.
      function gauss(rng) {
        const u = Math.max(rng(), 1e-9), v = rng();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      }

      // buildTimeline: precompute the full deterministic scenario — per-frame
      // pose/scores, the six narrative events, co-observation spans, and
      // cumulative communication per variant. Takes nothing; returns the
      // {frames, events, spans, comm, measAt} bundle used by the renderer.
      function buildTimeline() {
        const dense = catmullRom(WAYPOINTS, 60);
        const pts = resampleByArc(dense, N_FRAMES, PATH_END);

        const frames = [];
        for (let k = 0; k < N_FRAMES; k++) {
          const pos = pts[k];
          const nxt = pts[Math.min(N_FRAMES - 1, k + 1)];
          const dx = nxt[0] - pos[0], dy = nxt[1] - pos[1];
          const L = Math.hypot(dx, dy) || 1;
          const heading = Math.atan2(dy, dx);
          const pred = [pos[0] + dx / L * LOOKAHEAD, pos[1] + dy / L * LOOKAHEAD];
          const d = {}, inside = {}, delta = {}, mu = {}, muPred = {};
          for (const s of BS_IDS) {
            d[s] = dist(pos, CENTERS[s]);
            inside[s] = d[s] <= R_FOV;
            delta[s] = visFrac(pos, heading, s);
            mu[s] = muAt(pos, heading, s);
            muPred[s] = muAt(pred, heading, s);
          }
          frames.push({ pos, heading, pred, d, inside, delta, mu, muPred });
        }

        const birth = frames.findIndex(f => f.inside.A);
        for (let k = 0; k < N_FRAMES; k++) {
          frames[k].exist = k < birth ? 0 : Math.min(0.95, 0.5 + 0.05 * (k - birth));
        }
        for (let k = 0; k < N_FRAMES; k++) {
          const f = frames[k];
          f.lambda = {};
          for (const o of BS_IDS) {
            f.lambda[o] = {};
            for (const r of BS_IDS) {
              if (r === o) continue;
              // Lambda = existence gate x expected detections at the receiver,
              // i.e. the detection model's mu evaluated at the predicted pose
              const gate = f.exist >= 0.6 ? 1 : 0;
              f.lambda[o][r] = gate * f.muPred[r];
            }
          }
        }

        const firstAt = (from, pred2) => {
          for (let k = from; k < N_FRAMES; k++) if (pred2(frames[k])) return k;
          return -1;
        };
        const e1 = birth;
        const e2 = firstAt(e1, f => f.lambda.A.B >= LAMBDA_TH);
        const e3 = firstAt(e1, f => f.lambda.A.C >= LAMBDA_TH);
        const bIn = firstAt(e1, f => f.inside.B);
        const cIn = firstAt(e1, f => f.inside.C);
        const e4 = firstAt(bIn, f => f.d.B > R_FOV + 0.15);
        const e5 = firstAt(e4, f => f.d.A > R_FOV);
        const e6 = firstAt(e5, f => f.d.C > R_FOV);
        const events = { e1, e2, e3, e4, e5, e6, ack: e5 + 32 };
        const spans = { flowB: [bIn, e4], flowC: [cIn, e5], bIn, cIn };

        // measAt: deterministic measurement dots for (frame k, station s).
        // The count realizes the detection model's mu_s (rounded stochastically)
        // and every dot lies in the visible part of the extent, so density
        // falls with range and thins at FOV boundaries. Returns [x,y] list.
        const measAt = (k, s) => {
          const f = frames[k];
          const mu = f.mu[s];
          if (mu <= 0.02) return [];
          const rng = mulberry32(k * 7919 + s.charCodeAt(0) * 131);
          const n = Math.floor(mu) + (rng() < mu - Math.floor(mu) ? 1 : 0);
          const cos = Math.cos(f.heading), sin = Math.sin(f.heading);
          const out = [];
          let tries = 0;
          while (out.length < n && tries < n * 6 + 8) {
            tries++;
            const u = gauss(rng) * 0.30, v = gauss(rng) * 0.15;
            const p = [f.pos[0] + u * cos - v * sin, f.pos[1] + u * sin + v * cos];
            if (dist(p, CENTERS[s]) <= R_FOV) out.push(p);
          }
          return out;
        };

        const comm = {};
        for (const v of ['H', 'HM', 'HL', 'HP']) {
          comm[v] = { evtMsgs: [], evtBytes: [], flowMsgs: [], flowBytes: [] };
          let em = 0, eb = 0, fm = 0, fb = 0;
          for (let k = 0; k < N_FRAMES; k++) {
            if (k === e2) { em += 1; eb += BYTES.prior; }
            if (k === e3) { em += 1; eb += BYTES.prior; }
            if (k === e4) { em += 1; eb += BYTES.ctrl; }
            if (k === e5 || k === e5 + 16) { em += 1; eb += BYTES.ctrl; }
            if (v !== 'H') {
              for (const [s, span] of [['B', spans.flowB], ['C', spans.flowC]]) {
                if (k >= span[0] && k < span[1]) {
                  if (v === 'HM') {
                    const n = measAt(k, s).length;
                    if (n > 0) { fm += 2; fb += n * BYTES.meas + BYTES.posterior; }
                  } else {
                    fm += 2; fb += BYTES.posterior * 2;
                  }
                }
              }
            }
            comm[v].evtMsgs.push(em); comm[v].evtBytes.push(eb);
            comm[v].flowMsgs.push(fm); comm[v].flowBytes.push(fb);
          }
        }

        return { frames, events, spans, comm, measAt };
      }

      /* ================================================================
         Part 2 — toy-example renderer (SVG scene + side panel + controls)
         ================================================================ */
      const SVG_NS = 'http://www.w3.org/2000/svg';
      const T = buildTimeline();
      const EV_KEYS = ['e1', 'e2', 'e3', 'e4', 'e5', 'e6'];

      // world → screen: x in [-5,5], y in [-4.35,4.6], 96 px per unit
      const SCALE = 96;
      const W = 960, H = Math.round((4.6 + 3.95) * SCALE);
      const sx = wx => (wx + 5) * SCALE;
      const sy = wy => (4.6 - wy) * SCALE;

      // el: create an SVG element with attributes and optional text content.
      // Takes tag name, attribute map, optional string; returns the element.
      function el(tag, attrs, text) {
        const e = document.createElementNS(SVG_NS, tag);
        for (const k in attrs) e.setAttribute(k, attrs[k]);
        if (text !== undefined) e.textContent = text;
        return e;
      }

      // fmtBytes: human-format a byte count. Takes a number; returns a string.
      function fmtBytes(n) {
        if (n < 1000) return n + ' B';
        return (n / 1024).toFixed(1) + ' KB';
      }

      const EVENT_TEXT = {
        e1: ['Target appears in A — track born, label (A, 3)', '—', 0],
        e2: ['Score to B reaches 2 — prior A → B; B becomes a shadow', '1 prior packet', BYTES.prior],
        e3: ['Score to C reaches 2 — prior A → C; C becomes a shadow', '1 prior packet', BYTES.prior],
        e4: ['Target leaves B — shadow pruned, B notifies owner', '1 control msg', BYTES.ctrl],
        e5: ['A loses sight — transfer request; C owns only after the acknowledgment', 'transfer + ack', 2 * BYTES.ctrl],
        e6: ['Target leaves C — track pruned', '—', 0],
      };

      const scene = document.getElementById('toy-scene');
      const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img',
        'aria-label': 'Animated toy scenario: one target crossing the fields of view of base stations A, B and C, with handover events marked.' });
      scene.appendChild(svg);

      // ---- static layers ----
      const defs = el('defs', {});
      for (const s of BS_IDS) {
        const cp = el('clipPath', { id: 'clip-' + s });
        cp.appendChild(el('circle', { cx: sx(CENTERS[s][0]), cy: sy(CENTERS[s][1]), r: R_FOV * SCALE }));
        defs.appendChild(cp);
      }
      const unionClip = el('clipPath', { id: 'clip-union' });
      for (const s of BS_IDS) {
        unionClip.appendChild(el('circle', { cx: sx(CENTERS[s][0]), cy: sy(CENTERS[s][1]), r: R_FOV * SCALE }));
      }
      defs.appendChild(unionClip);
      const mkArrow = (id, color) => {
        const m = el('marker', { id, viewBox: '0 0 10 10', refX: 9, refY: 5,
          markerWidth: 7.5, markerHeight: 7.5, orient: 'auto-start-reverse' });
        m.appendChild(el('path', { d: 'M0,0 L10,5 L0,10 z', fill: color }));
        defs.appendChild(m);
      };
      mkArrow('arr-ink', '#1d2520');
      mkArrow('arr-transfer', BS_DARK.A);
      mkArrow('arr-ack', BS_DARK.C);
      svg.appendChild(defs);

      // backhaul links between stations
      const gBackhaul = el('g', { stroke: '#c9c3b2', 'stroke-width': 1.5, 'stroke-dasharray': '3 6' });
      for (const [a, b] of [['A', 'B'], ['A', 'C'], ['B', 'C']]) {
        gBackhaul.appendChild(el('line', {
          x1: sx(CENTERS[a][0]), y1: sy(CENTERS[a][1]),
          x2: sx(CENTERS[b][0]), y2: sy(CENTERS[b][1]),
        }));
      }
      svg.appendChild(gBackhaul);

      // FOV discs — stacked translucent fills so overlaps read darker
      const gFov = el('g', {});
      for (const s of BS_IDS) {
        gFov.appendChild(el('circle', {
          cx: sx(CENTERS[s][0]), cy: sy(CENTERS[s][1]), r: R_FOV * SCALE,
          fill: '#4a4a40', 'fill-opacity': 0.075,
        }));
        gFov.appendChild(el('circle', {
          cx: sx(CENTERS[s][0]), cy: sy(CENTERS[s][1]), r: R_FOV * SCALE,
          fill: 'none', stroke: BS_COLOR[s], 'stroke-opacity': 0.5, 'stroke-width': 2,
        }));
      }
      svg.appendChild(gFov);

      // station glyphs + labels (label spots follow the paper's figure)
      const BS_LABEL_POS = { A: [0, 4.18], B: [-3.2, -2.5], C: [3.2, -2.5] };
      const gBs = el('g', {});
      const stationRoles = {};
      const ownerRings = {};
      for (const s of BS_IDS) {
        const [cx, cy] = [sx(CENTERS[s][0]), sy(CENTERS[s][1])];
        ownerRings[s] = el('circle', { cx, cy, r:22, fill:'none', stroke:BS_COLOR[s], 'stroke-width':4, visibility:'hidden' });
        gBs.appendChild(ownerRings[s]);
        const tower = el('path', {
          d: `M ${cx - 7} ${cy + 9} L ${cx} ${cy - 11} L ${cx + 7} ${cy + 9} Z`,
          fill: BS_COLOR[s], stroke: '#ffffff', 'stroke-width': 1.5,
        });
        gBs.appendChild(tower);
        gBs.appendChild(el('circle', { cx, cy: cy - 11, r: 3.2, fill: BS_COLOR[s], stroke: '#fff', 'stroke-width': 1.2 }));
        gBs.appendChild(el('text', {
          x: sx(BS_LABEL_POS[s][0]), y: sy(BS_LABEL_POS[s][1]),
          'text-anchor': 'middle', 'font-size': 21, 'font-weight': 700, fill: BS_DARK[s],
        }, 'BS ' + s));
        stationRoles[s] = el('text', {
          x:sx(BS_LABEL_POS[s][0]), y:sy(BS_LABEL_POS[s][1])+24,
          'text-anchor':'middle', 'font-size':19, 'font-weight':700,
          fill:BS_DARK[s], 'paint-order':'stroke', stroke:'#fff', 'stroke-width':4,
        });
        gBs.appendChild(stationRoles[s]);
      }
      svg.appendChild(gBs);

      // ---- dynamic layers (updated per frame) ----
      const gTruePath = el('g', { 'clip-path': 'url(#clip-union)' });
      const guidePoly = el('polyline', {
        fill: 'none', stroke: '#1d2520', 'stroke-width': 2,
        'stroke-dasharray': '7 7', 'stroke-opacity': 0.18, points: '',
      });
      const truePoly = el('polyline', {
        fill: 'none', stroke: '#1d2520', 'stroke-width': 2.4,
        'stroke-dasharray': '7 7', 'stroke-opacity': 0.75, points: '',
      });
      gTruePath.appendChild(guidePoly);
      gTruePath.appendChild(truePoly);
      svg.appendChild(gTruePath);
      guidePoly.setAttribute('points',
        T.frames.map(f => sx(f.pos[0]).toFixed(1) + ',' + sy(f.pos[1]).toFixed(1)).join(' '));

      const TRACK_OFFSET = { A: [0.04, 0.04], B: [0.105, 0.105], C: [0.17, 0.17] };
      const trackPolys = {};
      for (const s of BS_IDS) {
        const g = el('g', { 'clip-path': `url(#clip-${s})` });
        const p = el('polyline', { fill: 'none', stroke: BS_COLOR[s], 'stroke-width': 2.8,
          'stroke-linejoin': 'round', 'stroke-linecap': 'round', points: '' });
        g.appendChild(p);
        svg.appendChild(g);
        trackPolys[s] = p;
      }

      // ghost shadow ellipses (prior received, target not yet visible locally)
      const ghosts = {};
      for (const s of ['B', 'C']) {
        const g = el('g', { visibility: 'hidden' });
        g.appendChild(el('ellipse', { rx: 0.40 * SCALE, ry: 0.20 * SCALE, fill: 'none',
          stroke: BS_COLOR[s], 'stroke-width': 2, 'stroke-dasharray': '5 5', 'stroke-opacity': 0.8 }));
        const tg = el('g', { class: 'ghost-label' });
        tg.appendChild(el('text', { 'font-size': 13, 'font-weight': 600, fill: BS_DARK[s],
          x: 0, y: -0.34 * SCALE, 'text-anchor': 'middle',
          'paint-order': 'stroke', stroke: '#ffffff', 'stroke-width': 4 }, 'shadow (predicted)'));
        g.appendChild(tg);
        svg.appendChild(g);
        ghosts[s] = g;
      }

      // per-frame stream (variant HM/HP): one animated dashed line owner↔shadow
      // with a labeled chip near the shadow-holder end
      const flows = {};
      for (const s of ['B', 'C']) {
        const g = el('g', { visibility: 'hidden' });
        const a = CENTERS.A, b = CENTERS[s];
        const dx = b[0] - a[0], dy = b[1] - a[1];
        const L = Math.hypot(dx, dy);
        const nx = -dy / L, ny = dx / L; // unit normal, for the label offset
        g.appendChild(el('line', {
          x1: sx(a[0]), y1: sy(a[1]), x2: sx(b[0]), y2: sy(b[1]),
          stroke: '#647067', 'stroke-width': 2.6, 'stroke-dasharray': '6 7',
          class: 'flow-ants',
        }));
        const lp = [a[0] + dx * 0.74 + nx * 0.45, a[1] + dy * 0.74 + ny * 0.45];
        const lbl = el('text', { x: sx(lp[0]), y: sy(lp[1]),
          'font-size': 13.5, 'font-weight': 650, fill: '#3d463f', 'text-anchor': 'middle',
          'paint-order': 'stroke', stroke: '#ffffff', 'stroke-width': 5, 'stroke-linejoin': 'round' });
        g.appendChild(lbl);
        svg.appendChild(g);
        flows[s] = { g, lbl };
      }

      // event badges + annotation arrows (positions from the paper's figure)
      const EV_ANCHOR = {
        e1: [-2.209, 2.671], e2: [-1.266, 1.489], e3: [-0.296, 0.739],
        e4: [0.725, 0.140], e5: T.frames[T.events.e5].pos, e6: [2.798, -3.136],
      };
      const EV_BADGE = {
        e1: [-1.72, 2.62], e2: [1.10, 1.80], e3: [1.80, 0.78],
        e4: [1.48, 0.14], e5: [2.95, -0.15], e6: [3.62, -3.14],
      };
      const EV_LABEL = {
        e1: [['(A, 3)', false]],
        e2: [['(A, 3)', false]],
        e3: [['(A, 3)', false]],
        e4: [['(A, 3)', true]],
        e5: [['A → C', false]],
        e6: [['(C, 7)', true]],
      };
      const EV_ARROW = { e2: true, e3: true, e5: true }; // packet / transfer arrows
      const gEvents = el('g', {});
      const badgeEls = {};
      EV_KEYS.forEach((k, i) => {
        const g = el('g', { cursor: 'pointer' });
        const [bx, by] = [sx(EV_BADGE[k][0]), sy(EV_BADGE[k][1])];
        const [ax, ay] = [sx(EV_ANCHOR[k][0]), sy(EV_ANCHOR[k][1])];
        // leader line / arrow from badge to its anchor on the trajectory
        g.appendChild(el('line', {
          x1: bx, y1: by, x2: ax, y2: ay,
          stroke: '#1d2520', 'stroke-width': EV_ARROW[k] ? 2.2 : 1.3,
          'stroke-opacity': EV_ARROW[k] ? 0.85 : 0.5,
          'marker-end': EV_ARROW[k] ? 'url(#arr-ink)' : 'none',
        }));
        g.appendChild(el('circle', { cx: bx, cy: by, r: 13, fill: '#ffffff', stroke: '#1d2520', 'stroke-width': 1.6 }));
        g.appendChild(el('text', { x: bx, y: by + 4.5, 'text-anchor': 'middle', 'font-size': 13.5, 'font-weight': 700, fill: '#1d2520' }, String(i + 1)));
        // label text beside the badge
        let lx = bx + 19;
        for (const [txt, struck] of EV_LABEL[k]) {
          const t = el('text', { x: lx, y: by + 4.5, 'font-size': 14, 'font-weight': 650,
            fill: '#1d2520', 'font-family': 'ui-monospace, Menlo, Consolas, monospace' }, txt);
          if (struck) t.setAttribute('text-decoration', 'line-through');
          g.appendChild(t);
          lx += txt.length * 8.6 + 6;
        }
        g.addEventListener('click', () => seek(T.events[k]));
        const title = el('title', {});
        title.textContent = `Event ${i + 1}: ${EVENT_TEXT[k][0]} (click to jump)`;
        g.appendChild(title);
        gEvents.appendChild(g);
        badgeEls[k] = g;
      });
      svg.appendChild(gEvents);

      // Keep the boundary crossing and handshake visible throughout event 5.
      const gTransfer = el('g', { visibility:'hidden', 'aria-hidden':'true' });
      const exitPoint = T.frames[T.events.e5].pos;
      gTransfer.appendChild(el('circle', { cx:sx(exitPoint[0]), cy:sy(exitPoint[1]), r:19,
        fill:'none', stroke:BS_DARK.A, 'stroke-width':4, 'stroke-dasharray':'5 4' }));
      gTransfer.appendChild(el('text', { x:sx(exitPoint[0])+28, y:sy(exitPoint[1])+39,
        'font-size':22, 'font-weight':700, fill:BS_DARK.A,
        'paint-order':'stroke', stroke:'#fff', 'stroke-width':5 }, 'Exit BS A'));
      const transferArrow = el('line', { 'stroke-width':5, 'stroke-dasharray':'9 5', display:'none' });
      gTransfer.appendChild(transferArrow);
      svg.appendChild(gTransfer);

      // traveling packet pills for events 2, 3 and the transfer handshake at 5
      const gPackets = el('g', {});
      svg.appendChild(gPackets);
      const PACKETS = [
        { at: T.events.e2, from: 'A', to: 'B', text: 'predicted prior', dur: 22 },
        { at: T.events.e3, from: 'A', to: 'C', text: 'predicted prior', dur: 22 },
        { at: T.events.e5, from: 'A', to: 'C', text: 'request A → C', dur: 16 },
        { at: T.events.e5 + 16, from: 'C', to: 'A', text: 'ack (C, 7)', dur: 16 },
      ];

      // measurement dots + target glyph
      const gMeas = el('g', {});
      svg.appendChild(gMeas);
      const gTarget = el('g', { visibility: 'hidden' });
      const targetEllipse = el('ellipse', { rx: 0.40 * SCALE, ry: 0.20 * SCALE,
        fill: 'rgba(29,37,32,0.06)', stroke: '#1d2520', 'stroke-width': 2.4 });
      gTarget.appendChild(targetEllipse);
      gTarget.appendChild(el('line', { x1: 0, y1: 0, x2: 0.40 * SCALE, y2: 0,
        stroke: '#1d2520', 'stroke-width': 2 }));
      svg.appendChild(gTarget);

      // ---- scene legend ----
      const legend = document.getElementById('scene-legend');
      // legendItem: append one swatch+label pair to the scene legend.
      // Takes an inline-SVG snippet and a label string; returns nothing.
      function legendItem(svgSnippet, label) {
        const span = document.createElement('span');
        span.className = 'item';
        span.innerHTML = '<span aria-hidden="true">' + svgSnippet + '</span> ' + label;
        legend.appendChild(span);
      }
      legendItem('<svg width="22" height="14"><rect x="1" y="1" width="9" height="12" fill="#4a4a40" fill-opacity="0.10"/><rect x="10" y="1" width="11" height="12" fill="#4a4a40" fill-opacity="0.22"/></svg>', 'FOV / overlap');
      legendItem('<svg width="26" height="14"><line x1="1" y1="7" x2="25" y2="7" stroke="#1d2520" stroke-width="2.2" stroke-dasharray="5 4"/></svg>', 'true trajectory');
      legendItem('<svg width="26" height="14"><line x1="1" y1="7" x2="25" y2="7" stroke="#2a78d6" stroke-width="3"/></svg>', 'A track (schematic)');
      legendItem('<svg width="26" height="14"><line x1="1" y1="7" x2="25" y2="7" stroke="#eb6834" stroke-width="3"/></svg>', 'B track (schematic)');
      legendItem('<svg width="26" height="14"><line x1="1" y1="7" x2="25" y2="7" stroke="#1baf7a" stroke-width="3"/></svg>', 'C track (schematic)');
      legendItem('<svg width="30" height="14"><circle cx="6" cy="7" r="4" fill="#2a78d6" stroke="#fff" stroke-width="1.6"/><circle cx="15" cy="7" r="4" fill="#eb6834" stroke="#fff" stroke-width="1.6"/><circle cx="24" cy="7" r="4" fill="#1baf7a" stroke="#fff" stroke-width="1.6"/></svg>', 'measurements (by BS)');
      legendItem('<svg width="26" height="14"><line x1="1" y1="7" x2="20" y2="7" stroke="#1d2520" stroke-width="2.2"/><path d="M20,3 L25,7 L20,11 z" fill="#1d2520"/></svg>', 'event packet');
      legendItem('<svg width="26" height="14"><line x1="1" y1="7" x2="25" y2="7" stroke="#647067" stroke-width="2.2" stroke-dasharray="4 6"/></svg>', 'per-frame stream');

      /* ---- side panel ---- */
      const meterHost = document.getElementById('lambda-meters');
      const meterEls = {};
      // buildMeters: (re)create the two Λ meter rows for the current owner.
      // Takes the owner id; returns nothing (mutates the panel DOM).
      function buildMeters(owner) {
        meterHost.textContent = '';
        meterEls.rows = {};
        for (const r of BS_IDS) {
          if (r === owner) continue;
          const row = document.createElement('div');
          row.className = 'meter-row';
          const name = document.createElement('span');
          name.className = 'name';
          name.textContent = `${owner} → ${r}`;
          const track = document.createElement('div');
          track.className = 'meter-track';
          const fill = document.createElement('div');
          fill.className = 'meter-fill';
          fill.style.background = BS_COLOR[r];
          const th = document.createElement('div');
          th.className = 'meter-th';
          th.style.left = (LAMBDA_TH / MU_CAP * 100) + '%';
          track.appendChild(fill); track.appendChild(th);
          const val = document.createElement('span');
          val.className = 'val';
          val.textContent = '0.0';
          row.appendChild(name); row.appendChild(track); row.appendChild(val);
          meterHost.appendChild(row);
          meterEls.rows[r] = { row, fill, val };
        }
        meterEls.owner = owner;
      }
      buildMeters('A');

      // static mu meters, one per station, in station colors
      const muHost = document.getElementById('mu-meters');
      const muEls = {};
      for (const bs of BS_IDS) {
        const row = document.createElement('div');
        row.className = 'meter-row';
        const name = document.createElement('span');
        name.className = 'name';
        name.textContent = 'BS ' + bs;
        const track = document.createElement('div');
        track.className = 'meter-track';
        const fill = document.createElement('div');
        fill.className = 'meter-fill';
        fill.style.background = BS_COLOR[bs];
        track.appendChild(fill);
        const val = document.createElement('span');
        val.className = 'val';
        val.textContent = '0.0';
        row.appendChild(name); row.appendChild(track); row.appendChild(val);
        muHost.appendChild(row);
        muEls[bs] = { fill, val };
      }

      const logHost = document.getElementById('event-log');
      const logItems = {};
      EV_KEYS.forEach((k, i) => {
        const li = document.createElement('li');
        li.tabIndex = 0;
        li.setAttribute('role', 'button');
        const n = document.createElement('span');
        n.className = 'n';
        n.textContent = String(i + 1);
        const txt = document.createElement('span');
        txt.textContent = EVENT_TEXT[k][0];
        li.appendChild(n); li.appendChild(txt);
        li.addEventListener('click', () => seek(T.events[k]));
        li.addEventListener('keydown', evk => {
          if (evk.key === 'Enter' || evk.key === ' ') { evk.preventDefault(); seek(T.events[k]); }
        });
        logHost.appendChild(li);
        logItems[k] = li;
      });

      const evtTableBody = document.querySelector('#event-table tbody');
      EV_KEYS.forEach((k, i) => {
        const tr = document.createElement('tr');
        const cells = [String(i + 1), EVENT_TEXT[k][0], EVENT_TEXT[k][1],
          EVENT_TEXT[k][2] ? EVENT_TEXT[k][2] + ' B' : '—'];
        cells.forEach((c, j) => {
          const td = document.createElement('td');
          if (j === 3) td.className = 'num';
          td.textContent = c;
          tr.appendChild(td);
        });
        evtTableBody.appendChild(tr);
      });

      /* ---- playback state & render loop ---- */
      let frame = 0, playing = false, variant = 'H', lastTick = 0, animationId = 0;
      const FPS = 30;
      const scrub = document.getElementById('scrub');
      const btnPlay = document.getElementById('btn-play');
      const frameLabel = document.getElementById('frame-label');

      // ownerAt: which station owns the track at frame k. Takes a frame index;
      // returns 'A' before the transfer, 'C' after, or 'A' pre-birth (unused).
      const ownerAt = k => (k >= T.events.ack ? 'C' : 'A');

      // shadowsAt: current shadow-holder set at frame k. Takes a frame index;
      // returns an array of station ids.
      function shadowsAt(k) {
        const out = [];
        if (k >= T.events.e2 && k < T.events.e4) out.push('B');
        if (k >= T.events.e3 && k < T.events.ack) out.push('C');
        return out;
      }

      // render: draw frame k across the scene and the side panel. Takes a
      // frame index; returns nothing (pure DOM update, no state change).
      function render(k) {
        const f = T.frames[k];
        const ev = T.events;
        const trackAlive = k >= ev.e1 && k < ev.e6;
        const currentOwner = ownerAt(k);
        for (const s of BS_IDS) {
          const owns = trackAlive && s === currentOwner;
          ownerRings[s].setAttribute('visibility', owns ? 'visible' : 'hidden');
          stationRoles[s].textContent = owns ? 'OWNER ' + (s === 'A' ? '(A, 3)' : '(C, 7)')
            : trackAlive && shadowsAt(k).includes(s) ? 'SHADOW (A, 3)' : '';
        }
        const transferVisible = k >= ev.e5 && k < ev.e6;
        gTransfer.setAttribute('visibility', transferVisible ? 'visible' : 'hidden');
        const transferInProgress = k >= ev.e5 && k < ev.ack;
        const awaitingAck = k >= ev.e5 + 16 && k < ev.ack;
        const from = CENTERS[awaitingAck ? 'C' : 'A'], to = CENTERS[awaitingAck ? 'A' : 'C'];
        transferArrow.setAttribute('x1', sx(from[0])); transferArrow.setAttribute('y1', sy(from[1]));
        transferArrow.setAttribute('x2', sx(to[0])); transferArrow.setAttribute('y2', sy(to[1]));
        transferArrow.setAttribute('stroke', awaitingAck ? BS_DARK.C : BS_DARK.A);
        transferArrow.setAttribute('marker-end', awaitingAck ? 'url(#arr-ack)' : 'url(#arr-transfer)');
        transferArrow.setAttribute('display', transferInProgress ? 'inline' : 'none');

        // true trajectory drawn up to k
        const ptsStr = [];
        for (let i = 0; i <= k; i++) {
          const p = T.frames[i].pos;
          ptsStr.push(sx(p[0]).toFixed(1) + ',' + sy(p[1]).toFixed(1));
        }
        truePoly.setAttribute('points', ptsStr.join(' '));

        // per-station estimated tracks over their active spans
        const spanOf = { A: [ev.e1, ev.e5], B: [ev.e2, ev.e4], C: [ev.e3, ev.e6] };
        for (const s of BS_IDS) {
          const [a, b] = spanOf[s];
          const upto = Math.min(k, b);
          const pts2 = [];
          if (upto >= a) {
            const [ox, oy] = TRACK_OFFSET[s];
            for (let i = a; i <= upto; i++) {
              const rng = mulberry32(i * 913 + s.charCodeAt(0));
              const p = T.frames[i].pos;
              pts2.push((sx(p[0] + ox + (rng() - 0.5) * 0.05)).toFixed(1) + ',' +
                        (sy(p[1] + oy + (rng() - 0.5) * 0.05)).toFixed(1));
            }
          }
          trackPolys[s].setAttribute('points', pts2.join(' '));
        }

        // ghost shadows between prior receipt and local visibility
        for (const [s, from, until] of [['B', ev.e2, T.spans.bIn], ['C', ev.e3, T.spans.cIn]]) {
          const on = k >= from && k < until;
          ghosts[s].setAttribute('visibility', on ? 'visible' : 'hidden');
          if (on) {
            const deg = -f.heading * 180 / Math.PI;
            ghosts[s].setAttribute('transform',
              `translate(${sx(f.pred[0])}, ${sy(f.pred[1])}) rotate(${deg})`);
            ghosts[s].querySelector('.ghost-label').setAttribute('transform', `rotate(${-deg})`);
          }
        }

        // per-frame streams (variants HM / HL / HP)
        for (const [s, span] of [['B', T.spans.flowB], ['C', T.spans.flowC]]) {
          const on = variant !== 'H' && k >= span[0] && k < span[1];
          flows[s].g.setAttribute('visibility', on ? 'visible' : 'hidden');
          if (on) flows[s].lbl.textContent = variant === 'HM' ? 'group ↔ fused' : variant === 'HL' ? 'likelihood ↔ fused' : 'posterior ↔ fused';
        }

        // traveling packet pills
        gPackets.textContent = '';
        for (const p of PACKETS) {
          if (k >= p.at && k < p.at + p.dur) {
            const t = (k - p.at) / p.dur;
            const a = CENTERS[p.from], b = CENTERS[p.to];
            const x = sx(a[0] + (b[0] - a[0]) * t), y = sy(a[1] + (b[1] - a[1]) * t) - 16;
            const g = el('g', {});
            const wPill = p.text.length * 8 + 18;
            g.appendChild(el('rect', { x: x - wPill / 2, y: y - 13, width: wPill, height: 24,
              rx: 12, fill: '#1d2520' }));
            g.appendChild(el('text', { x, y: y + 4, 'text-anchor': 'middle', 'font-size': 13,
              'font-weight': 650, fill: '#ffffff' }, p.text));
            gPackets.appendChild(g);
          }
        }

        // measurements — drawn whenever the detection model yields any, so
        // density visibly falls with range and thins at FOV boundaries
        gMeas.textContent = '';
        for (const s of BS_IDS) {
          for (const p of T.measAt(k, s)) {
            gMeas.appendChild(el('circle', { cx: sx(p[0]), cy: sy(p[1]), r: 4.5,
              fill: BS_COLOR[s], stroke: '#ffffff', 'stroke-width': 2 }));
          }
        }

        // target glyph
        const alive = k >= ev.e1 && k < Math.min(N_FRAMES, ev.e6 + 14);
        gTarget.setAttribute('visibility', alive ? 'visible' : 'hidden');
        if (alive) {
          const deg = -f.heading * 180 / Math.PI;
          gTarget.setAttribute('transform', `translate(${sx(f.pos[0])}, ${sy(f.pos[1])}) rotate(${deg})`);
        }

        // badge reached-state
        EV_KEYS.forEach(kk => {
          badgeEls[kk].setAttribute('opacity', k >= T.events[kk] ? 1 : 0.28);
        });

        // ---- side panel ----
        const owner = ownerAt(k);
        const bornYet = k >= ev.e1;
        const gone = k >= ev.e6;
        document.getElementById('owner-swatch').style.background = BS_COLOR[owner];
        document.getElementById('owner-label').textContent =
          !bornYet ? '—' : owner === 'A' ? '(A, 3)' : '(C, 7)';
        document.getElementById('owner-who').textContent =
          !bornYet ? 'no track yet' : gone ? 'track pruned' : k >= ev.e5 && k < ev.ack ? 'A owns until acknowledgment' : 'owner: BS ' + owner;
        document.getElementById('exist-val').textContent = gone ? '—' : f.exist.toFixed(2);

        const chips = document.getElementById('shadow-chips');
        chips.textContent = '';
        const sh = gone ? [] : shadowsAt(k);
        if (!sh.length) {
          const s = document.createElement('span');
          s.className = 'none';
          s.textContent = 'none';
          chips.appendChild(s);
        } else {
          for (const s of sh) {
            const c = document.createElement('span');
            c.className = 'chip';
            const sw = document.createElement('span');
            sw.className = 'swatch';
            sw.style.background = BS_COLOR[s];
            c.appendChild(sw);
            c.appendChild(document.createTextNode('BS ' + s));
            chips.appendChild(c);
          }
        }

        for (const bs of BS_IDS) {
          muEls[bs].fill.style.width = Math.min(100, f.mu[bs] / MU_CAP * 100) + '%';
          muEls[bs].val.textContent = f.mu[bs].toFixed(1);
        }

        if (meterEls.owner !== owner) buildMeters(owner);
        for (const r in meterEls.rows) {
          const lam = bornYet && !gone ? f.lambda[owner][r] : 0;
          const { row, fill, val } = meterEls.rows[r];
          fill.style.width = Math.min(100, lam / MU_CAP * 100) + '%';
          val.textContent = lam.toFixed(1);
          row.classList.toggle('fired', lam >= LAMBDA_TH);
        }

        const c = T.comm[variant];
        document.getElementById('stat-evt').textContent = c.evtMsgs[k] + ' · ' + fmtBytes(c.evtBytes[k]);
        document.getElementById('stat-flow').textContent = c.flowMsgs[k] + ' · ' + fmtBytes(c.flowBytes[k]);
        document.getElementById('stat-total').textContent = fmtBytes(c.evtBytes[k] + c.flowBytes[k]);

        let current = null;
        EV_KEYS.forEach(kk => { if (k >= T.events[kk]) current = kk; });
        EV_KEYS.forEach(kk => {
          logItems[kk].classList.toggle('reached', k >= T.events[kk]);
          logItems[kk].classList.toggle('current', kk === current);
        });

        frameLabel.textContent = 'frame ' + k;
        scrub.value = String(k);
      }

      // seek: jump the playhead to frame k and redraw. Takes a frame index;
      // returns nothing.
      function seek(k) {
        frame = Math.max(0, Math.min(N_FRAMES - 1, k));
        render(frame);
      }

      // tick: rAF playback loop advancing ~FPS frames/sec while playing.
      // Takes the rAF timestamp; returns nothing (reschedules itself).
      function tick(ts) {
        if (!playing) return;
        const stepMs = 1000 / FPS;
        if (ts - lastTick >= stepMs) {
          // accumulate the frame budget so pacing stays at FPS on any refresh
          // rate; resync after long gaps (tab in background, first frame)
          lastTick = ts - lastTick > 200 ? ts : lastTick + stepMs;
          frame++;
          if (frame >= N_FRAMES) { frame = N_FRAMES - 1; setPlaying(false); }
          render(frame);
        }
        if (playing) animationId = requestAnimationFrame(tick);
      }

      // setPlaying: start or stop playback (restarts from the top if the
      // timeline is finished). Takes a boolean; returns nothing.
      function setPlaying(p) {
        cancelAnimationFrame(animationId);
        playing = p;
        btnPlay.textContent = p ? '⏸ Pause' : '▶ Play';
        if (p) {
          if (frame >= N_FRAMES - 1) frame = 0;
          lastTick = 0;
          animationId = requestAnimationFrame(tick);
        }
      }

      btnPlay.addEventListener('click', () => setPlaying(!playing));
      window.addEventListener('message', e => { if (e.source === parent && e.data?.type === 'bento-live-pause') setPlaying(false); });
      document.addEventListener('visibilitychange', () => { if (document.hidden) setPlaying(false); });
      scrub.addEventListener('input', () => { setPlaying(false); seek(Number(scrub.value)); });
      document.getElementById('btn-next').addEventListener('click', () => {
        const nxt = EV_KEYS.map(k => T.events[k]).find(v => v > frame);
        if (nxt !== undefined) { setPlaying(false); seek(nxt); }
      });
      document.getElementById('btn-prev').addEventListener('click', () => {
        const prevs = EV_KEYS.map(k => T.events[k]).filter(v => v < frame);
        setPlaying(false);
        seek(prevs.length ? prevs[prevs.length - 1] : 0);
      });
      document.getElementById('btn-back1').addEventListener('click', () => { setPlaying(false); seek(frame - 1); });
      document.getElementById('btn-fwd1').addEventListener('click', () => { setPlaying(false); seek(frame + 1); });
      document.querySelectorAll('input[name="variant"]').forEach(r => {
        r.addEventListener('change', () => { variant = r.value; render(frame); });
      });

      // deep-link support: #f=<frame> seeks the playhead, &v=<variant> picks a
      // variant — e.g. #f=160&v=HM. Applied once on load.
      (function applyHash() {
        const m = location.hash.match(/f=(\d+)/);
        const mv = location.hash.match(/v=(HM|HL|HP|H)/);
        if (mv) {
          variant = mv[1];
          const rb = document.querySelector(`input[name="variant"][value="${variant}"]`);
          if (rb) rb.checked = true;
        }
        frame = m ? Math.max(0, Math.min(N_FRAMES - 1, Number(m[1]))) : 0;
      })();
      render(frame);

      /* ================================================================
         Part 3 — handover-score widget (draggable target, live Λ)
         ================================================================ */
      (function scoreWidget() {
        const host = document.getElementById('score-scene');
        const SW = 560, SH = 470;
        const s = el('svg', { viewBox: `0 0 ${SW} ${SH}`, role: 'img',
          style: 'touch-action: none',
          'aria-label': 'Draggable target beside a base-station field of view; the handover score rises as the target enters.' });
        host.appendChild(s);

        const C = [200, 235], RR = 175;      // receiver BS and its FOV (px)
        const RMIN = 40;                      // minimum-range clamp (px)

        // radial shading: detections are denser close to the BS (rho ~ 1/r)
        const sdefs = el('defs', {});
        const grad = el('radialGradient', { id: 'rho-grad' });
        grad.appendChild(el('stop', { offset: '0%', 'stop-color': '#4a4a40', 'stop-opacity': 0.22 }));
        grad.appendChild(el('stop', { offset: '55%', 'stop-color': '#4a4a40', 'stop-opacity': 0.10 }));
        grad.appendChild(el('stop', { offset: '100%', 'stop-color': '#4a4a40', 'stop-opacity': 0.05 }));
        sdefs.appendChild(grad);
        s.appendChild(sdefs);
        s.appendChild(el('circle', { cx: C[0], cy: C[1], r: RR, fill: 'url(#rho-grad)' }));
        s.appendChild(el('circle', { cx: C[0], cy: C[1], r: RR, fill: 'none', stroke: '#1baf7a', 'stroke-width': 2.5, 'stroke-opacity': 0.6 }));
        s.appendChild(el('path', { d: `M ${C[0] - 8} ${C[1] + 10} L ${C[0]} ${C[1] - 12} L ${C[0] + 8} ${C[1] + 10} Z`,
          fill: '#1baf7a', stroke: '#fff', 'stroke-width': 1.5 }));
        s.appendChild(el('text', { x: C[0], y: C[1] + 30, 'text-anchor': 'middle', 'font-size': 15, 'font-weight': 700, fill: '#0f7250' }, 'BS r'));
        s.appendChild(el('text', { x: SW - 14, y: 26, 'text-anchor': 'end', 'font-size': 13.5, fill: '#647067' }, 'drag the target ⤦'));

        const rangeLine = el('line', { stroke: '#647067', 'stroke-width': 1.4, 'stroke-dasharray': '4 5' });
        s.appendChild(rangeLine);

        const gDotsW = el('g', {});
        s.appendChild(gDotsW);
        const tgt = el('g', { cursor: 'grab' });
        const tgtHit = el('circle', { r: 46, fill: 'transparent' }); // generous drag target
        const tgtEll = el('ellipse', { rx: 52, ry: 26, fill: 'rgba(29,37,32,0.07)', stroke: '#1d2520', 'stroke-width': 2.6 });
        tgt.appendChild(tgtHit); tgt.appendChild(tgtEll);
        s.appendChild(tgt);

        let px = 430, py = 130, extent = 1;

        // visibleFraction: fraction of the extent footprint inside the FOV,
        // estimated on a grid over the ellipse. Takes center [x,y] and the two
        // semi-axes; returns a number in [0,1].
        function visibleFraction(cx, cy, a, b) {
          let inside = 0, total = 0;
          for (let i = -4; i <= 4; i++) {
            for (let j = -4; j <= 4; j++) {
              const u = i / 4, v = j / 4;
              if (u * u + v * v > 1) continue;
              total++;
              const x = cx + u * a, y = cy + v * b;
              if (Math.hypot(x - C[0], y - C[1]) <= RR) inside++;
            }
          }
          return total ? inside / total : 0;
        }

        // update: recompute the score from the current target pose and paint
        // scene + readouts. Takes nothing; returns nothing.
        function update() {
          const a = 52 * extent, b = 26 * extent;
          tgt.setAttribute('transform', `translate(${px}, ${py})`);
          tgtEll.setAttribute('rx', a); tgtEll.setAttribute('ry', b);
          rangeLine.setAttribute('x1', C[0]); rangeLine.setAttribute('y1', C[1]);
          rangeLine.setAttribute('x2', px); rangeLine.setAttribute('y2', py);

          const r = Math.max(RMIN, Math.hypot(px - C[0], py - C[1]));
          const rho = 90 / r;                       // ~1/range scatter density
          const area = (a * b) / (52 * 26);         // footprint, ref-normalized
          const vis = visibleFraction(px, py, a, b);
          const lambda = 6 * rho * area * vis / (90 / 175); // Λ = 6 at rim-range, fully visible

          document.getElementById('sr-range').textContent = (r / 35).toFixed(1) + ' u';
          document.getElementById('sr-rho').textContent = rho.toFixed(2);
          document.getElementById('sr-area').textContent = area.toFixed(2) + ' ×';
          document.getElementById('sr-vis').textContent = (vis * 100).toFixed(0) + ' %';
          document.getElementById('sr-lambda').textContent = lambda.toFixed(2);
          const meter = document.getElementById('sr-meter');
          const cap = 8;
          meter.style.width = Math.min(100, lambda / cap * 100) + '%';
          document.getElementById('sr-th').style.left = (LAMBDA_TH / cap * 100) + '%';
          document.getElementById('sr-meter-val').textContent = lambda.toFixed(1);
          // sample detection dots whose count realizes the current mu — the
          // density the reader sees is the number the score is computing
          gDotsW.textContent = '';
          const rngW = mulberry32(Math.round(px) * 7919 + Math.round(py) * 131 + Math.round(extent * 100));
          let nW = Math.round(Math.min(lambda, 12));
          let triesW = 0;
          while (nW > 0 && triesW < nW * 6 + 8) {
            triesW++;
            const dxW = gauss(rngW) * 0.42 * a, dyW = gauss(rngW) * 0.42 * b;
            const xW = px + dxW, yW = py + dyW;
            if (Math.hypot(xW - C[0], yW - C[1]) <= RR) {
              gDotsW.appendChild(el('circle', { cx: xW, cy: yW, r: 4,
                fill: '#1baf7a', stroke: '#ffffff', 'stroke-width': 1.6 }));
              nW--;
            }
          }

          const verdict = document.getElementById('sr-verdict');
          const fires = lambda >= LAMBDA_TH;
          verdict.textContent = fires
            ? 'Threshold reached — send the prior packet'
            : 'below threshold — stay silent';
          verdict.className = 'score-verdict ' + (fires ? 'yes' : 'no');
          meter.style.background = fires ? 'var(--accent)' : '#9aa79e';
        }

        // toSvg: convert a pointer event to this SVG's viewBox coordinates.
        // Takes a PointerEvent; returns [x, y].
        function toSvg(evp) {
          const point = new DOMPoint(evp.clientX, evp.clientY).matrixTransform(s.getScreenCTM().inverse());
          return [point.x, point.y];
        }

        let dragging = false;
        tgt.addEventListener('pointerdown', evp => {
          dragging = true;
          tgt.setAttribute('cursor', 'grabbing');
          tgt.setPointerCapture(evp.pointerId);
          evp.preventDefault();
        });
        tgt.addEventListener('pointermove', evp => {
          if (!dragging) return;
          const [x, y] = toSvg(evp);
          px = Math.max(30, Math.min(SW - 30, x));
          py = Math.max(30, Math.min(SH - 30, y));
          update();
        });
        // endDrag: release the drag on any pointer end, including cancellation
        // (touch scroll takeover, capture loss). Takes nothing; returns nothing.
        const endDrag = () => { dragging = false; tgt.setAttribute('cursor', 'grab'); };
        tgt.addEventListener('pointerup', endDrag);
        tgt.addEventListener('pointercancel', endDrag);
        tgt.addEventListener('lostpointercapture', endDrag);
        document.getElementById('sr-extent').addEventListener('input', evi => {
          extent = Number(evi.target.value);
          update();
        });
        update();
      })();
