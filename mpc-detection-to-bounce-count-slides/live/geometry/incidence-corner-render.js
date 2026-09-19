(function (root) {
  'use strict';
  const C = { bs: '#1874b8', ue: '#0e8f7e', ink: '#16222e', muted: '#51606e', faint: '#a2adb7', point: '#e8720c' };
  const label = (p, text, color = C.muted, size = 13, anchor = 'start') =>
    `<text x="${p[0]}" y="${p[1]}" fill="${color}" font-family="Arial,Helvetica,sans-serif" font-size="${size}" text-anchor="${anchor}">${text}</text>`;
  const line = (a, b, color, width = 2, dash = '') =>
    `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="${color}" stroke-width="${width}" ${dash ? `stroke-dasharray="${dash}"` : ''}/>`;
  const point = (p, color, radius = 6) => `<circle cx="${p[0]}" cy="${p[1]}" r="${radius}" fill="${color}" stroke="#fff" stroke-width="2"/>`;
  function arrow(a, b, color, dash = '') {
    const angle = Math.atan2(b[1] - a[1], b[0] - a[0]);
    const tip = offset => [b[0] - 8 * Math.cos(angle + offset), b[1] - 8 * Math.sin(angle + offset)];
    return line(a, b, color, 2.5, dash) + `<path d="M${tip(-.4)} L${b} L${tip(.4)}" fill="none" stroke="${color}" stroke-width="2.5"/>`;
  }
  function render({ bs, ue, result, step = 3, reference, showReference = false }) {
    const points = [bs, ue, result.single.point, result.p1, result.p2].filter(Boolean);
    if (showReference && reference) points.push(reference.points.single, reference.points.p1, reference.points.p2);
    const minX = Math.min(...points.map(p => p[0])), maxX = Math.max(...points.map(p => p[0]));
    const minY = Math.min(...points.map(p => p[1])), maxY = Math.max(...points.map(p => p[1]));
    const scale = Math.min(560 / Math.max(32, maxX - minX + 10), 225 / Math.max(26, maxY - minY + 4));
    const camera = { scale, x: 350 - (minX + maxX) * .5 * scale, y: 175 + (minY + maxY) * .5 * scale };
    const project = p => [camera.x + p[0] * scale, camera.y - p[1] * scale];
    const B = project(bs), U = project(ue);
    const limit = (p, d, maximum = 1200) => Math.max(0, Math.min(maximum,
      d[0] > 0 ? (738 - p[0]) / d[0] : d[0] < 0 ? (22 - p[0]) / d[0] : Infinity,
      d[1] > 0 ? (302 - p[1]) / d[1] : d[1] < 0 ? (64 - p[1]) / d[1] : Infinity));
    function ray(origin, direction, length, color, dash = '') {
      if (!direction) return '';
      const d = [direction[0], -direction[1]], distance = limit(origin, d, length);
      return arrow(origin, [origin[0] + d[0] * distance, origin[1] + d[1] * distance], color, dash);
    }
    function wall(w, length, color = C.ink, dash = '') {
      const p = project(w.point), d = [w.tangent[0], -w.tangent[1]];
      const back = limit(p, [-d[0], -d[1]], length), forward = limit(p, d, length);
      return line([p[0] - d[0] * back, p[1] - d[1] * back], [p[0] + d[0] * forward, p[1] + d[1] * forward], color, dash ? 6 : 3.5, dash);
    }
    let markup = '<rect width="760" height="340" fill="#fff"/>';
    markup += line([22, 23], [46, 23], C.bs, 3) + label([54, 27], 'BS AoD / reflected ray', C.bs, 12);
    markup += line([243, 23], [267, 23], C.ue, 3) + label([275, 27], 'UE AoA ray', C.ue, 12);
    markup += line([399, 23], [423, 23], C.ink, 3.5) + label([431, 27], 'inferred wall', C.ink, 12);
    if (showReference && reference) {
      markup += wall({ point: reference.points.single, tangent: reference.wallA.tangent }, 225, C.faint, '5 6');
      markup += wall({ point: reference.points.p2, tangent: reference.wallB.tangent }, 95, C.faint, '5 6');
      markup += label([738, 47], 'dashed gray: reference only', C.muted, 11, 'end');
    }
    if (result.wallA) markup += wall(result.wallA, 290);
    if (step === 3 && result.wallB) markup += wall(result.wallB, 94);
    const first = result.single;
    markup += `<g opacity="${step === 1 ? 1 : .33}">`;
    markup += ray(B, first.departure, first.status === 'ok' ? first.bsDistance * scale + 23 : 220, C.bs, step > 1 ? '5 5' : '');
    markup += ray(U, first.arrival, first.status === 'ok' ? first.ueDistance * scale + 23 : 220, C.ue, step > 1 ? '5 5' : '');
    markup += '</g>';
    if (first.point) {
      const S = project(first.point);
      markup += point(S, step === 1 ? C.ue : C.muted);
      markup += label([S[0] - 12, S[1] - 18], 'S · single bounce', step === 1 ? C.ue : C.muted, 13, 'end');
      markup += label([S[0] - 60, S[1] + 21], 'wall A', C.ink, 13);
    }
    if (step >= 2) {
      markup += ray(B, result.departure, result.p1 ? result.firstDistance * scale : 245, C.bs);
      if (result.p1) {
        const P1 = project(result.p1);
        markup += ray(P1, result.reflected, result.p2 ? result.middleDistance * scale + 35 : 250, C.bs);
        markup += point(P1, C.bs, 7);
        markup += label([P1[0] + 8, P1[1] - 18], 'P1 · first hit', C.bs, 13);
        const length = result.p2 ? result.middleDistance * scale : 160;
        markup += label([P1[0] + result.reflected[0] * length * .4, P1[1] - result.reflected[1] * length * .4 + 27], 'reflected AoD', C.bs, 12, 'middle');
      }
    }
    if (step === 3) {
      markup += ray(U, result.arrival, result.p2 ? result.lastDistance * scale + 32 : 245, C.ue);
      if (result.p2) {
        const P2 = project(result.p2);
        markup += point(P2, C.point, 8);
        markup += label([P2[0] + 19, P2[1] - 17], 'P2 · second hit', C.point, 14);
        markup += label([P2[0] + 34, P2[1] + 35], 'wall B', C.ink, 13);
      }
    }
    markup += `<rect x="${B[0] - 6}" y="${B[1] - 6}" width="12" height="12" fill="${C.ink}"/>`;
    markup += label([B[0] - 22, B[1] + 24], 'BS', C.ink, 14);
    markup += point(U, '#2ca02c', 8) + `<circle cx="${U[0]}" cy="${U[1]}" r="14" fill="none" stroke="#2ca02c" stroke-dasharray="3 3"/>`;
    markup += label([U[0], U[1] + 28], 'UE · drag', '#1d7a1d', 13, 'middle');
    if (step === 1 && first.status !== 'ok' || step >= 2 && !result.p1 || step === 3 && !result.p2) {
      markup += label([380, 326], 'No forward construction for these bearings.', '#89520e', 13, 'middle');
    } else {
      markup += label([22, 326], step === 1 ? 'One single-bounce MPC fixes a point and the orientation of wall A.' : 'S and P1 share wall A; they belong to different measured paths.', C.muted, 12);
    }
    return { markup, camera, svg: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 340" role="img" aria-label="Two-bounce incidence points constructed from a single-bounce wall and two measured rays">${markup}</svg>` };
  }
  const api = { render };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.IncidenceCornerDrawing = api;
})(globalThis);
