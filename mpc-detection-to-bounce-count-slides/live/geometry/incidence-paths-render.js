(function (root) {
  'use strict';
  const C = { bs: '#1874b8', ue: '#0e8f7e', ink: '#16222e', muted: '#51606e', faint: '#a2adb7', point: '#e8720c' };
  const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const label = (p, text, color = C.muted, size = 13, anchor = 'start') =>
    `<text x="${p[0]}" y="${p[1]}" fill="${color}" font-family="Arial,Helvetica,sans-serif" font-size="${size}" text-anchor="${anchor}" stroke="#fff" stroke-width="3" paint-order="stroke">${text}</text>`;
  const line = (a, b, color, width = 2, dash = '') =>
    `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="${color}" stroke-width="${width}" ${dash ? `stroke-dasharray="${dash}"` : ''}/>`;
  const dot = (p, color, radius = 6) => `<circle cx="${p[0]}" cy="${p[1]}" r="${radius}" fill="${color}" stroke="#fff" stroke-width="2"/>`;
  function arrow(a, b, color, dash = '') {
    const angle = Math.atan2(b[1] - a[1], b[0] - a[0]);
    const tip = offset => [b[0] - 8 * Math.cos(angle + offset), b[1] - 8 * Math.sin(angle + offset)];
    return line(a, b, color, 2.5, dash) + `<path d="M${tip(-.4)} L${b} L${tip(.4)}" fill="none" stroke="${color}" stroke-width="2.5"/>`;
  }
  function frame(points) {
    const minX = Math.min(...points.map(p => p[0])), maxX = Math.max(...points.map(p => p[0]));
    const minY = Math.min(...points.map(p => p[1])), maxY = Math.max(...points.map(p => p[1]));
    const scale = Math.min(560 / Math.max(32, maxX - minX + 10), 225 / Math.max(26, maxY - minY + 5));
    const camera = { scale, x: 350 - (minX + maxX) * .5 * scale, y: 175 + (minY + maxY) * .5 * scale };
    const project = p => [camera.x + p[0] * scale, camera.y - p[1] * scale];
    const limit = (p, d, maximum = 1000) => Math.max(0, Math.min(maximum,
      d[0] > 0 ? (738 - p[0]) / d[0] : d[0] < 0 ? (22 - p[0]) / d[0] : Infinity,
      d[1] > 0 ? (301 - p[1]) / d[1] : d[1] < 0 ? (59 - p[1]) / d[1] : Infinity));
    function ray(origin, direction, length, color, dash = '') {
      if (!direction) return '';
      const p = project(origin), d = [direction[0], -direction[1]], t = limit(p, d, length);
      return arrow(p, [p[0] + d[0] * t, p[1] + d[1] * t], color, dash);
    }
    function wall(w, color = C.ink, dash = '', extent = 270) {
      const p = project(w.point), d = [w.tangent[0], -w.tangent[1]];
      const back = limit(p, [-d[0], -d[1]], extent), forward = limit(p, d, extent);
      return line([p[0] - d[0] * back, p[1] - d[1] * back], [p[0] + d[0] * forward, p[1] + d[1] * forward], color, dash ? 6 : 3.5, dash);
    }
    function endpoints(bs, ues, draggable = true) {
      const B = project(bs);
      let out = `<rect x="${B[0] - 6}" y="${B[1] - 6}" width="12" height="12" fill="${C.ink}"/>` + label([B[0] - 13, B[1] + 24], 'BS', C.ink, 14, 'end');
      ues.forEach((ue, i) => {
        const U = project(ue);
        out += dot(U, '#2ca02c', 8);
        if (draggable) out += `<circle cx="${U[0]}" cy="${U[1]}" r="14" fill="none" stroke="#2ca02c" stroke-dasharray="3 3"/>`;
        out += label([U[0] - 12, U[1] + (ues.length === 1 ? 25 : 19)], ues.length === 1 ? 'UE · drag' : `UE ${i ? 'b' : 'a'}`, '#1d7a1d', 13, 'end');
      });
      return out;
    }
    const legend = '<rect width="760" height="340" fill="#fff"/>' + line([22, 23], [46, 23], C.bs, 3) + label([54, 27], 'BS AoD / reflected ray', C.bs, 12)
      + line([244, 23], [268, 23], C.ue, 3) + label([276, 27], 'UE AoA ray', C.ue, 12)
      + line([399, 23], [423, 23], C.ink, 3.5) + label([431, 27], 'wall from rays', C.ink, 12);
    return { camera, project, ray, wall, endpoints, legend };
  }
  function finish(markup, camera, description) {
    return { markup, camera, svg: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 340" role="img" aria-label="${description}">${markup}</svg>` };
  }

  function render({ bs, ue, result, view, names = ['A', 'B', 'C'], maxOrder = 3, reference, showReference = false }) {
    const points = [bs, ue, ...result.paths.flatMap(p => p.hits)];
    if (showReference && reference) points.push(...reference.measurements.flatMap(p => p.hits || []));
    const f = frame(points), { project, ray, wall, camera } = f;
    let markup = f.legend;
    if (showReference && reference) {
      reference.walls.forEach(w => { markup += wall(w, C.faint, '5 6'); });
      markup += label([738, 47], 'dashed gray: reference only', C.muted, 11, 'end');
    }
    result.walls.slice(0, view.walls).forEach((w, i) => {
      markup += wall(w);
      const p = project(w.point), vertical = Math.abs(w.tangent[1]) > .75;
      markup += label(vertical ? [p[0] + (i === 1 ? -20 : 20), 68] : [Math.max(75, p[0] - 95), p[1] + 20], `wall ${names[i]}`, C.ink, 13, vertical && i === 1 ? 'end' : 'start');
    });
    // Shorter paths locate the walls, but their hits are not the next path's hits.
    result.paths.slice(0, view.path).forEach((p, index) => {
      if (p.status !== 'ok') return;
      const nodes = [bs, ...p.hits, ue];
      markup += '<g opacity=".23">';
      nodes.slice(1).forEach((q, i) => { markup += line(project(nodes[i]), project(q), C.muted, 1.4, '4 5'); });
      markup += '</g>';
      const q = project(p.hits.at(-1));
      markup += dot(q, C.faint, 4) + label([q[0] + 12, q[1] + 5], index === 0 ? 'S' : 'Q2', C.muted, 11);
    });
    const path = result.paths[view.path];
    if (path) {
      const visibleHits = path.hits.slice(0, view.hits);
      const origins = [bs, ...visibleHits];
      const limit = Math.min(path.directions.length, view.hits + (view.arrival ? 0 : 1));
      for (let i = 0; i < limit; i++) {
        if (!origins[i]) break;
        const target = path.hits[i], length = target ? distance(origins[i], target) * camera.scale : 235;
        const isLast = i === limit - 1;
        markup += ray(origins[i], path.directions[i], length + (isLast ? 24 : 0), C.bs);
      }
      if (view.arrival) markup += ray(ue, path.arrival, path.status === 'ok' ? path.lastDistance * camera.scale + 25 : 230, C.ue);
      visibleHits.forEach((p, i) => {
        const q = project(p), last = i === view.path && view.arrival;
        const w = result.walls[i] || result.walls[0], vertical = w && Math.abs(w.tangent[1]) > .75;
        const left = vertical && i === 1;
        const name = view.path === 0 ? 'S' : view.path === 1 && maxOrder === 3 ? `Q${i + 1}` : `P${i + 1}`;
        const offset = vertical ? [left ? -14 : 14, -10] : [9, q[1] > 230 ? 23 : -17];
        markup += dot(q, last ? C.point : C.bs, 7) + label([q[0] + offset[0], q[1] + offset[1]], `${name}${last ? ' · intersection' : ''}`, last ? C.point : C.bs, last ? 14 : 13, left ? 'end' : 'start');
      });
      if (view.arrival && result.closure && !result.consistent && result.closure.expected.status === 'ok' && view.path === 2) {
        markup += line(project(path.hits[2]), project(result.closure.expected.point), '#b45607', 2, '4 4');
      }
    }
    markup += f.endpoints(bs, [ue]);
    markup += label([22, 326], path && (path.status === 'ok' || path.hits.length >= view.hits) ? 'Each measured path has its own incidence points on the shared walls.' : 'This construction stops when a required forward ray does not meet.', C.muted, 12);
    return finish(markup, camera, 'Incidence points reconstructed by reflecting measured rays at walls inferred from shorter paths');
  }

  function renderAmbiguity({ bs, ues, result, baseline, step = 3, showReference = false, fitPoints }) {
    const f = frame([bs, ...ues, ...(fitPoints || [...result.paths.flatMap(p => p.hits), ...baseline.paths.flatMap(p => p.hits)])]);
    const { project, wall, ray, camera } = f;
    let markup = f.legend;
    if (showReference) {
      markup += wall(baseline.wallA, C.faint, '5 6') + wall(baseline.wallB, C.faint, '5 6');
      markup += label([738, 47], 'dashed gray: original walls', C.muted, 11, 'end');
    }
    if (step >= 2 && result.wallA) {
      markup += wall(result.wallA);
      const a = project(result.paths[0]?.hits[0] || result.wallA.point);
      markup += label([a[0] - 150, a[1] + 18], result.singlePath ? 'wall A · fixed by single path' : 'candidate wall A', C.ink, 12);
      if (result.wallB) {
        markup += wall(result.wallB, C.ink, '', 96);
        const b = project(result.wallB.point);
        markup += label([b[0] + 25, b[1] + 55], 'wall B · from rays', C.ink, 12);
      }
    }
    result.paths.slice(0, step === 2 ? 1 : 2).forEach((p, index) => {
      const dash = index ? '6 5' : '';
      markup += `<g opacity="${index ? .6 : 1}">`;
      if (step === 1) {
        markup += ray(bs, p.departure, 220, C.bs, dash) + ray(ues[index], p.arrival, 220, C.ue, dash);
      } else {
        const origins = [bs, p.hits[0]];
        for (let i = 0; i < Math.min(2, p.directions.length); i++) {
          if (origins[i]) markup += ray(origins[i], p.directions[i], p.hits[i] ? distance(origins[i], p.hits[i]) * camera.scale + (i ? 24 : 0) : 220, C.bs, dash);
        }
        markup += ray(ues[index], p.arrival, p.status === 'ok' ? p.lastDistance * camera.scale + 22 : 220, C.ue, dash);
        p.hits.forEach((p, i) => {
          const q = project(p);
          markup += dot(q, i ? C.point : C.bs, index ? 5 : 7);
          if (!index) markup += label([q[0] + 13, q[1] - 15], `P${i + 1}`, i ? C.point : C.bs, 14);
        });
      }
      markup += '</g>';
    });
    if (result.singlePath && step >= 2) {
      const p = result.singlePath, S = p.hits[0];
      markup += '<g opacity=".35">' + ray(bs, p.departure, distance(bs, S) * camera.scale, C.muted, '4 5') + ray(ues[0], p.arrival, distance(ues[0], S) * camera.scale, C.muted, '4 5') + '</g>';
      const q = project(S); markup += dot(q, C.muted, 5) + label([q[0] - 12, q[1] - 16], 'S · single path', C.muted, 12, 'end');
    }
    markup += f.endpoints(bs, ues, false);
    markup += label([22, 326], step === 1 ? 'Known poses and fixed endpoint rays do not specify the first reflecting wall.' : 'Solid: UE a. Dashed: UE b. Each pose observes its own double-bounce MPC.', C.muted, 12);
    return finish(markup, camera, 'Two known UE poses share an ambiguity family of incidence points until a single-bounce measurement fixes the first wall');
  }
  const api = { render, renderAmbiguity };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.IncidencePathsDrawing = api;
})(globalThis);
