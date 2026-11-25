(function (root) {
  'use strict';
  const WIDTH = 760, HEIGHT = 400;
  const colors = { bs: '#1874b8', ue: '#0e8f7e', ink: '#16222e', muted: '#51606e', reference: '#a2adb7' };
  const label = (x, y, text, color = colors.muted, size = 13, anchor = 'start') =>
    `<text x="${x}" y="${y}" fill="${color}" font-family="Arial,Helvetica,sans-serif" font-size="${size}" text-anchor="${anchor}">${text}</text>`;
  const line = (a, b, color, width = 2, dash = '') =>
    `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="${color}" stroke-width="${width}" ${dash ? `stroke-dasharray="${dash}"` : ''}/>`;
  function arrow(a, b, color) {
    const angle = Math.atan2(b[1] - a[1], b[0] - a[0]);
    const end = offset => [b[0] - 9 * Math.cos(angle + offset), b[1] - 9 * Math.sin(angle + offset)];
    return line(a, b, color, 2.4) + `<path d="M${end(-0.4)} L${b} L${end(0.4)}" fill="none" stroke="${color}" stroke-width="2.4"/>`;
  }
  function render({ bs, ue, result, reference, showReference = false }) {
    const valid = result.status === 'ok';
    const points = [bs, ue, ...(valid ? [result.point] : []), ...(showReference && reference ? [reference.point] : [])];
    let minX = Math.min(...points.map(p => p[0])), maxX = Math.max(...points.map(p => p[0]));
    let minY = Math.min(...points.map(p => p[1])), maxY = Math.max(...points.map(p => p[1]));
    const spanX = Math.max(28, maxX - minX + 10), spanY = Math.max(18, maxY - minY + 8);
    const scale = Math.min(540 / spanX, 255 / spanY);
    const camera = { scale, x: 320 - (minX + maxX) / 2 * scale, y: 195 + (minY + maxY) / 2 * scale };
    const project = p => [camera.x + scale * p[0], camera.y - scale * p[1]];
    const B = project(bs), U = project(ue), P = valid ? project(result.point) : null;
    const rayLimit = (p, d, maximum) => Math.max(0, Math.min(maximum,
      d[0] > 0 ? (738 - p[0]) / d[0] : d[0] < 0 ? (22 - p[0]) / d[0] : Infinity,
      d[1] > 0 ? (342 - p[1]) / d[1] : d[1] < 0 ? (60 - p[1]) / d[1] : Infinity));
    const wallEnds = (p, t, maximum) => {
      const d = [t[0], -t[1]], back = [-d[0], -d[1]];
      const a = rayLimit(p, back, maximum), b = rayLimit(p, d, maximum);
      return [[p[0] - d[0] * a, p[1] - d[1] * a], [p[0] + d[0] * b, p[1] + d[1] * b], a, b];
    };
    let markup = '<rect width="760" height="400" fill="#fff"/>';
    markup += line([22, 23], [48, 23], colors.bs, 3) + label(56, 27, 'BS AoD ray', colors.bs, 12);
    markup += line([181, 23], [207, 23], colors.ue, 3) + label(215, 27, 'UE AoA ray', colors.ue, 12);
    if (valid) markup += line([341, 23], [367, 23], colors.ink, 4) + label(375, 27, 'inferred wall', colors.ink, 12);
    if (showReference && reference) {
      const center = project(reference.point), t = reference.tangent;
      const [a, b] = wallEnds(center, t, 116);
      markup += line(a, b, colors.reference, 7, '5 5');
      markup += label(Math.max(110, a[0] - 12), Math.max(58, a[1] - 9), 'reference only', colors.reference, 11, 'end');
    }
    if (valid) {
      const t = result.wallTangent, n = result.wallNormal;
      const [a, b, back, forward] = wallEnds(P, t, 96);
      markup += line(a, b, colors.ink, 4);
      for (let v = -back + 8; v <= forward - 8; v += 18) {
        const q = [P[0] + v * t[0], P[1] - v * t[1]];
        markup += line(q, [q[0] + 7 * (n[0] + t[0]), q[1] - 7 * (n[1] + t[1])], colors.reference, 1.3);
      }
    }
    for (const [origin, d, distance, color, name, sign] of [
      [B, result.departure, result.bsDistance, colors.bs, 'AoD', -1],
      [U, result.arrival, result.ueDistance, colors.ue, 'AoA', 1]
    ]) {
      if (!d) continue;
      const length = valid ? distance * scale : 230;
      const drawnLength = rayLimit(origin, [d[0], -d[1]], length + 40);
      const end = [origin[0] + d[0] * drawnLength, origin[1] - d[1] * drawnLength];
      markup += arrow(origin, end, color);
      const mid = [origin[0] + d[0] * Math.min(length, drawnLength) * .43, origin[1] - d[1] * Math.min(length, drawnLength) * .43];
      markup += label(mid[0], mid[1] + sign * 17, name, color, 14, 'middle');
    }
    markup += `<rect x="${B[0] - 6}" y="${B[1] - 6}" width="12" height="12" fill="${colors.ink}"/>`;
    markup += label(B[0] - 14, B[1] + 22, 'BS', colors.ink, 14);
    markup += `<circle cx="${U[0]}" cy="${U[1]}" r="8" fill="#2ca02c" stroke="#fff" stroke-width="2"/><circle cx="${U[0]}" cy="${U[1]}" r="14" fill="none" stroke="#2ca02c" stroke-dasharray="3 3"/>`;
    markup += label(U[0] - 12, U[1] + 30, 'UE · drag', '#1d7a1d', 13);
    if (valid) {
      markup += `<circle cx="${P[0]}" cy="${P[1]}" r="8" fill="${colors.ue}" stroke="#fff" stroke-width="2"/>`;
      markup += label(P[0] + 22, P[1] - 34, 'P · incidence point', colors.ue, 15);
    } else {
      markup += label(380, 345, result.status === 'behind' ? 'The forward rays do not meet.' : 'No unique forward-ray intersection.', '#b45607', 16, 'middle');
    }
    markup += label(22, 381, valid ? 'Angles locate P. The wall orientation follows from the reflection law.' : 'A single bounce requires a unique intersection in front of both endpoints.', colors.muted, 12);
    return { markup, camera, svg: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${WIDTH} ${HEIGHT}" role="img" aria-label="Incidence point at the intersection of the BS departure ray and UE arrival ray">${markup}</svg>` };
  }
  const api = { render };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.IncidenceSingleDrawing = api;
})(globalThis);
