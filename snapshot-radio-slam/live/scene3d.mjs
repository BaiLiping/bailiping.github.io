/**
 * Small dependency-free, z-up perspective scene for the radio SLAM explainer.
 *
 * The caller owns layout: give the canvas an explicit CSS width and height.
 * Rotations map local vectors to world vectors and may be nested 3x3 arrays or
 * flat, row-major arrays. `showTruth` hides the truth marker and orientation;
 * path visibility is controlled by the caller's `paths` array.
 *
 * Optional callbacks: onPathClick(id, event), onCameraChange(cameraState).
 */

const TAU = Math.PI * 2;
const PALETTE = Object.freeze({
  blue: '#1874B8', teal: '#0A6B5E', orange: '#D76809',
  magenta: '#A64D91', ink: '#16222E', bg: '#F4F6F8',
  grey: '#7D8993', grid: '#DCE3E8', muted: '#AFBBC3',
});
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const add = (a, b) => a.map((x, i) => x + b[i]);
const sub = (a, b) => a.map((x, i) => x - b[i]);
const mul = (a, s) => a.map(x => x * s);
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = a => Math.hypot(...a);
const unit = a => { const n = norm(a); return n > 1e-12 ? mul(a, 1 / n) : [0, 0, 0]; };
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const vec = p => p && p.length >= 3 && [p[0], p[1], p[2]].every(Number.isFinite) ? [p[0], p[1], p[2]] : null;
const safeCall = (fn, ...args) => { if (typeof fn === 'function') fn(...args); };

function rotationColumn(rotation, i) {
  if (!rotation) return null;
  const column = Array.isArray(rotation[0]) || ArrayBuffer.isView(rotation[0])
    ? [rotation[0]?.[i], rotation[1]?.[i], rotation[2]?.[i]]
    : [rotation[i], rotation[3 + i], rotation[6 + i]];
  return vec(column);
}

function niceStep(span) {
  const rough = Math.max(span / 10, 1e-3);
  const power = 10 ** Math.floor(Math.log10(rough));
  const f = rough / power;
  return power * (f > 5 ? 10 : f > 2 ? 5 : f > 1 ? 2 : 1);
}

function clipBox(a, b, lo, hi) {
  const delta = sub(b, a);
  let enter = 0, leave = 1;
  for (let i = 0; i < 3; i += 1) {
    if (Math.abs(delta[i]) < 1e-12) {
      if (a[i] < lo[i] || a[i] > hi[i]) return null;
    } else {
      let t0 = (lo[i] - a[i]) / delta[i], t1 = (hi[i] - a[i]) / delta[i];
      if (t0 > t1) [t0, t1] = [t1, t0];
      enter = Math.max(enter, t0);
      leave = Math.min(leave, t1);
      if (enter > leave) return null;
    }
  }
  return [add(a, mul(delta, enter)), add(a, mul(delta, leave))];
}

function screenDistance(p, a, b) {
  const dx = b.x - a.x, dy = b.y - a.y;
  const t = clamp(((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx * dx + dy * dy || 1), 0, 1);
  return Math.hypot(p.x - a.x - t * dx, p.y - a.y - t * dy);
}

export class Scene3D {
  constructor(canvas, options = {}) {
    if (!canvas?.getContext) throw new TypeError('Scene3D requires a canvas element.');
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    if (!this.ctx) throw new Error('A 2D canvas context is required.');
    this.options = options;
    this.scene = {};
    this.width = 1;
    this.height = 1;
    this.dpr = 1;
    this.destroyed = false;
    this.pending = 0;
    this.pointers = new Map();
    this.hitSegments = [];
    this.dragged = false;
    this.bounds = { center: [0, 0, 2], radius: 12, lo: [-10, -10, 0], hi: [10, 10, 8] };
    this.camera = {
      azimuth: -0.95,
      elevation: 0.57,
      zoom: 1,
      reset: () => this.resetView(),
    };
    this.savedTouchAction = canvas.style.touchAction;
    this.savedCursor = canvas.style.cursor;
    this.addedTabIndex = !canvas.hasAttribute('tabindex');
    if (this.addedTabIndex) canvas.tabIndex = 0;
    canvas.style.touchAction = 'none';
    canvas.style.cursor = 'grab';
    this.listeners = [];
    const listen = (target, name, handler, opts) => {
      target.addEventListener(name, handler, opts);
      this.listeners.push(() => target.removeEventListener(name, handler, opts));
    };
    listen(canvas, 'pointerdown', e => this._pointerDown(e));
    listen(canvas, 'pointermove', e => this._pointerMove(e));
    listen(canvas, 'pointerup', e => this._pointerUp(e));
    listen(canvas, 'pointercancel', e => this._pointerUp(e, true));
    listen(canvas, 'lostpointercapture', e => this._pointerUp(e, true));
    listen(canvas, 'wheel', e => {
      e.preventDefault();
      const scale = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? this.height : 1;
      this.camera.zoom = clamp(this.camera.zoom * Math.exp(clamp(e.deltaY * scale, -800, 800) * 0.001), 0.38, 3.5);
      this._cameraChanged();
    }, { passive: false });
    listen(canvas, 'keydown', e => this._key(e));
    listen(canvas, 'dblclick', () => this.resetView());
    if (typeof ResizeObserver !== 'undefined') {
      this.resizeObserver = new ResizeObserver(() => this._resize());
      this.resizeObserver.observe(canvas);
    } else {
      listen(window, 'resize', () => this._resize());
    }
    this._resize();
  }

  setScene(scene = {}) {
    if (this.destroyed) return;
    this.scene = scene;
    this.selected = scene.selectedIds == null ? null : new Set(scene.selectedIds);
    this._computeBounds();
    this._dirty();
  }

  resetView() {
    this.camera.azimuth = -0.95;
    this.camera.elevation = 0.57;
    this.camera.zoom = 1;
    this._cameraChanged();
  }

  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    if (this.pending) cancelAnimationFrame(this.pending);
    this.resizeObserver?.disconnect();
    this.listeners.forEach(remove => remove());
    this.listeners.length = 0;
    this.pointers.clear();
    this.canvas.style.touchAction = this.savedTouchAction;
    this.canvas.style.cursor = this.savedCursor;
    if (this.addedTabIndex) this.canvas.removeAttribute('tabindex');
  }

  _resize() {
    if (this.destroyed) return;
    const rect = this.canvas.getBoundingClientRect();
    this.width = Math.max(1, rect.width || this.canvas.clientWidth || 640);
    this.height = Math.max(1, rect.height || this.canvas.clientHeight || 420);
    this.dpr = clamp(window.devicePixelRatio || 1, 1, 3);
    const w = Math.round(this.width * this.dpr), h = Math.round(this.height * this.dpr);
    if (this.canvas.width !== w) this.canvas.width = w;
    if (this.canvas.height !== h) this.canvas.height = h;
    this._dirty();
  }

  _dirty() {
    if (this.destroyed || this.pending) return;
    this.pending = requestAnimationFrame(() => {
      this.pending = 0;
      if (!this.destroyed) this._render();
    });
  }

  _cameraChanged() {
    this.camera.azimuth = ((this.camera.azimuth % TAU) + TAU) % TAU;
    this.camera.elevation = clamp(this.camera.elevation, 0.08, 1.43);
    this._dirty();
    safeCall(this.options.onCameraChange, {
      azimuth: this.camera.azimuth,
      elevation: this.camera.elevation,
      zoom: this.camera.zoom,
    });
  }

  _point(e) {
    const rect = this.canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  _pointerDown(e) {
    if (e.button != null && e.button !== 0 && e.pointerType !== 'touch') return;
    this.canvas.focus({ preventScroll: true });
    this.canvas.setPointerCapture?.(e.pointerId);
    const point = this._point(e);
    this.pointers.set(e.pointerId, point);
    if (this.pointers.size === 1) {
      this.pressPoint = point;
      this.dragged = false;
    } else {
      this.dragged = true;
    }
    this.canvas.style.cursor = 'grabbing';
  }

  _pointerMove(e) {
    if (!this.pointers.has(e.pointerId)) return;
    const old = this.pointers.get(e.pointerId), next = this._point(e);
    const previous = [...this.pointers.values()];
    this.pointers.set(e.pointerId, next);
    if (this.pointers.size >= 2) {
      const current = [...this.pointers.values()];
      const d0 = Math.hypot(previous[0].x - previous[1].x, previous[0].y - previous[1].y);
      const d1 = Math.hypot(current[0].x - current[1].x, current[0].y - current[1].y);
      if (d0 > 2 && d1 > 2) this.camera.zoom = clamp(this.camera.zoom * d0 / d1, 0.38, 3.5);
      this.dragged = true;
    } else {
      const dx = next.x - old.x, dy = next.y - old.y;
      this.camera.azimuth -= dx * 0.006;
      this.camera.elevation += dy * 0.006;
      if (this.pressPoint && Math.hypot(next.x - this.pressPoint.x, next.y - this.pressPoint.y) > 3) this.dragged = true;
    }
    this._cameraChanged();
  }

  _pointerUp(e, cancelled = false) {
    if (!this.pointers.has(e.pointerId)) return;
    const point = this._point(e);
    const wasSingle = this.pointers.size === 1;
    this.pointers.delete(e.pointerId);
    if (!cancelled && wasSingle && !this.dragged && typeof this.options.onPathClick === 'function') {
      let best = null, distance = 11;
      for (const segment of this.hitSegments) {
        const d = screenDistance(point, segment.a, segment.b);
        if (d < distance) { distance = d; best = segment.id; }
      }
      if (best != null) this.options.onPathClick(best, e);
    }
    if (!this.pointers.size) this.canvas.style.cursor = 'grab';
  }

  _key(e) {
    const step = e.shiftKey ? 0.18 : 0.08;
    switch (e.key) {
      case 'ArrowLeft': this.camera.azimuth += step; break;
      case 'ArrowRight': this.camera.azimuth -= step; break;
      case 'ArrowUp': this.camera.elevation += step; break;
      case 'ArrowDown': this.camera.elevation -= step; break;
      case '+': case '=': this.camera.zoom = Math.max(0.38, this.camera.zoom / 1.12); break;
      case '-': case '_': this.camera.zoom = Math.min(3.5, this.camera.zoom * 1.12); break;
      case 'Home': case 'r': case 'R': e.preventDefault(); this.resetView(); return;
      default: return;
    }
    e.preventDefault();
    this._cameraChanged();
  }

  _computeBounds() {
    // Estimated positions deliberately never affect the camera's scene scale.
    const bs = vec(this.scene.bsPosition) || [0, 0, 2];
    const truth = vec(this.scene.truePosition);
    const anchor = truth ? mul(add(bs, truth), 0.5) : bs;
    const anchorSpan = truth ? norm(sub(bs, truth)) : 10;
    const cap = Math.max(25, anchorSpan * 5);
    const points = [bs, ...(truth ? [truth] : [])];
    for (const path of this.scene.paths || []) {
      for (const raw of path.points || []) {
        const p = vec(raw);
        if (p && norm(sub(p, anchor)) <= cap) points.push(p);
      }
    }
    const lo = [0, 1, 2].map(i => Math.min(...points.map(p => p[i])));
    const hi = [0, 1, 2].map(i => Math.max(...points.map(p => p[i])));
    lo[2] = Math.min(0, lo[2]);
    const maxSpan = Math.max(6, hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
    for (let i = 0; i < 2; i += 1) {
      if (hi[i] - lo[i] < maxSpan * 0.55) {
        const center = (hi[i] + lo[i]) / 2;
        lo[i] = center - maxSpan * 0.275;
        hi[i] = center + maxSpan * 0.275;
      }
      lo[i] -= maxSpan * 0.12;
      hi[i] += maxSpan * 0.12;
    }
    hi[2] = Math.max(hi[2], lo[2] + maxSpan * 0.28);
    const center = mul(add(lo, hi), 0.5);
    const radius = Math.max(4, norm(sub(hi, lo)) * 0.5);
    this.bounds = { center, radius, lo, hi };
  }

  _view() {
    const { azimuth: az, elevation: el, zoom } = this.camera;
    const target = this.bounds.center;
    const aspect = this.width / this.height;
    const fit = aspect < 1.2 ? 3.5 / Math.max(aspect, 0.5) : 3.05;
    const distance = this.bounds.radius * fit * zoom;
    const eye = add(target, mul([Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)], distance));
    const forward = unit(sub(target, eye));
    const right = unit(cross(forward, [0, 0, 1]));
    const up = cross(right, forward);
    const focal = Math.min(this.width, this.height) * 1.13;
    const near = Math.max(0.015, this.bounds.radius * 0.025);
    const toCamera = p => {
      const relative = sub(p, eye);
      return [dot(relative, right), dot(relative, up), dot(relative, forward)];
    };
    const fromCamera = p => ({
      x: this.width / 2 + focal * p[0] / p[2],
      y: this.height * 0.51 - focal * p[1] / p[2],
      depth: p[2],
    });
    const project = p => {
      const c = toCamera(p);
      return c[2] >= near ? fromCamera(c) : null;
    };
    const segment = (a, b) => {
      let ca = toCamera(a), cb = toCamera(b);
      if (ca[2] < near && cb[2] < near) return null;
      if (ca[2] < near) ca = add(ca, mul(sub(cb, ca), (near - ca[2]) / (cb[2] - ca[2])));
      if (cb[2] < near) cb = add(cb, mul(sub(ca, cb), (near - cb[2]) / (ca[2] - cb[2])));
      return [fromCamera(ca), fromCamera(cb)];
    };
    return { project, segment, toCamera, fromCamera, eye, target };
  }

  _render() {
    const ctx = this.ctx, w = this.width, h = this.height;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const background = ctx.createLinearGradient(0, 0, 0, h);
    background.addColorStop(0, '#FFFFFF');
    background.addColorStop(1, PALETTE.bg);
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, w, h);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    const view = this._view();
    const primitives = [], labels = [];
    this.hitSegments = [];
    const radius = this.bounds.radius;
    const clipLo = sub(this.bounds.center, [radius * 2, radius * 2, radius * 2]);
    const clipHi = add(this.bounds.center, [radius * 2, radius * 2, radius * 2]);
    const visible = p => p && p.x > -30 && p.x < w + 30 && p.y > -30 && p.y < h + 30;
    const line = (a, b, style = {}) => {
      const clipped = clipBox(a, b, clipLo, clipHi);
      if (!clipped) return;
      const projected = view.segment(...clipped);
      if (!projected) return;
      const [pa, pb] = projected;
      const primitive = { type: 'line', a: pa, b: pb, depth: (pa.depth + pb.depth) / 2, ...style };
      primitives.push(primitive);
      if (style.id != null) this.hitSegments.push({ a: pa, b: pb, id: style.id });
    };
    const marker = (p, style = {}) => {
      const projected = view.project(p);
      if (visible(projected)) primitives.push({ type: 'marker', ...projected, ...style });
      return projected;
    };
    const label = (p, text, color = PALETTE.ink, dx = 12, dy = -12, priority = 0) => {
      const projected = view.project(p);
      if (visible(projected)) labels.push({ ...projected, text, color, dx, dy, priority });
    };
    const orientation = (p, rotation, muted = false) => {
      if (!rotation) return;
      const length = radius * 0.13;
      const colors = muted ? [PALETTE.grey, PALETTE.grey, PALETTE.grey] : [PALETTE.blue, PALETTE.teal, PALETTE.orange];
      for (let i = 0; i < 3; i += 1) {
        const column = rotationColumn(rotation, i);
        if (!column) continue;
        line(p, add(p, mul(unit(column), length)), { color: colors[i], width: muted ? 1.25 : 2, alpha: muted ? 0.55 : 0.9, arrow: true });
      }
    };

    // A world-fixed floor and exact perspective projection make orbiting useful
    // for distinguishing coincident planar rays from true spatial intersections.
    const { lo, hi } = this.bounds;
    const step = niceStep(Math.max(hi[0] - lo[0], hi[1] - lo[1]));
    const x0 = Math.floor(lo[0] / step) * step, x1 = Math.ceil(hi[0] / step) * step;
    const y0 = Math.floor(lo[1] / step) * step, y1 = Math.ceil(hi[1] / step) * step;
    for (let x = x0; x <= x1 + step * 0.1; x += step) line([x, y0, 0], [x, y1, 0], { color: PALETTE.grid, width: 0.9, alpha: 0.85 });
    for (let y = y0; y <= y1 + step * 0.1; y += step) line([x0, y, 0], [x1, y, 0], { color: PALETTE.grid, width: 0.9, alpha: 0.85 });

    const axesOrigin = [x0, y0, 0], axesLength = Math.min(radius * 0.27, step * 1.5);
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]].forEach((axis, i) => {
      const end = add(axesOrigin, mul(axis, axesLength));
      const color = [PALETTE.blue, PALETTE.teal, PALETTE.orange][i];
      line(axesOrigin, end, { color, width: 1.5, alpha: 0.7, arrow: true });
      label(end, ['x', 'y', 'z'][i], color, 4, -5, -1);
    });

    for (const path of this.scene.paths || []) {
      const points = (path.points || []).map(vec).filter(Boolean);
      if (points.length < 2) continue;
      const selected = this.selected == null || this.selected.has(path.id);
      const highlighted = this.scene.highlightId === path.id;
      const color = path.kind === 'double' ? PALETTE.magenta : path.kind === 'los' ? PALETTE.blue : PALETTE.teal;
      const alpha = highlighted ? 1 : selected ? 0.83 : 0.18;
      for (let i = 1; i < points.length; i += 1) {
        line(points[i - 1], points[i], { color, alpha, width: highlighted ? 3.5 : selected ? 2.3 : 1.2, dash: path.kind === 'double' ? [7, 5] : [], id: path.id });
      }
      for (let i = 1; i < points.length - 1; i += 1) {
        marker(points[i], { color, fill: '#FFFFFF', alpha, radius: highlighted ? 5.6 : 4.4, strokeWidth: 1.8 });
        if (highlighted) label(points[i], `IP ${path.id}${points.length > 3 ? `.${i}` : ''}`, color, 9, -10, 1);
      }
    }

    for (const ray of this.scene.rays || []) {
      const origin = vec(ray.origin), rawDirection = vec(ray.direction);
      if (!origin || !rawDirection || norm(rawDirection) < 1e-12) continue;
      const length = Number.isFinite(ray.length) && ray.length > 0 ? Math.min(ray.length, radius * 8) : radius * 1.4;
      const endpoint = add(origin, mul(unit(rawDirection), length));
      line(origin, endpoint, {
        color: ray.color || PALETTE.orange,
        width: this.scene.highlightId === ray.id ? 2.7 : 1.8,
        alpha: 0.7,
        dash: [5, 5],
        arrow: true,
      });
    }

    const bs = vec(this.scene.bsPosition);
    if (bs) {
      if (bs[2] > 0) line([bs[0], bs[1], 0], bs, { color: PALETTE.blue, width: 1.1, alpha: 0.35, dash: [3, 4] });
      marker(bs, { color: PALETTE.blue, fill: PALETTE.blue, radius: 7.2, halo: true });
      label(bs, 'BS', PALETTE.blue, 12, -13, 3);
    }
    const truth = vec(this.scene.truePosition);
    if (truth && this.scene.showTruth !== false) {
      orientation(truth, this.scene.trueRotation, true);
      marker(truth, { color: PALETTE.grey, fill: '#FFFFFF', radius: 8.5, strokeWidth: 2 });
      label(truth, 'UE truth', PALETTE.grey, 12, 20, 2);
    }
    const estimate = vec(this.scene.estimatedPosition);
    let offscreenEstimate = null;
    if (estimate) {
      const bounded = estimate.every((value, i) => value >= clipLo[i] && value <= clipHi[i]);
      const projected = bounded ? view.project(estimate) : null;
      const onScreen = projected && projected.x > 14 && projected.x < w - 14 && projected.y > 14 && projected.y < h - 14;
      if (onScreen) {
        if (estimate[2] > 0) line([estimate[0], estimate[1], 0], estimate, { color: PALETTE.orange, width: 1, alpha: 0.27, dash: [3, 4] });
        orientation(estimate, this.scene.candidateRotation, false);
        marker(estimate, { color: PALETTE.orange, fill: PALETTE.orange, radius: 6.1, halo: true });
        label(estimate, 'UE estimate', PALETTE.orange, 12, -14, 4);
      } else {
        offscreenEstimate = estimate;
      }
    }

    // Painter's algorithm: far primitives first, near markers last at a tie.
    primitives.sort((a, b) => b.depth - a.depth || (a.type === 'line' ? -1 : 1));
    for (const primitive of primitives) this._drawPrimitive(primitive);
    this._drawLabels(labels);
    if (offscreenEstimate) this._drawOffscreen(offscreenEstimate, view);

    ctx.globalAlpha = 1;
    ctx.setLineDash([]);
    ctx.fillStyle = '#73818D';
    ctx.font = '11px system-ui, -apple-system, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText(`Grid ${Number(step.toPrecision(3))} m`, 14, h - 11);
    if (w > 460) {
      ctx.textAlign = 'right';
      ctx.fillText('Drag to orbit · Scroll to zoom · R to reset', w - 14, h - 11);
    }
  }

  _drawPrimitive(p) {
    const ctx = this.ctx;
    ctx.globalAlpha = p.alpha ?? 1;
    ctx.strokeStyle = p.color || PALETTE.ink;
    ctx.fillStyle = p.fill || p.color || PALETTE.ink;
    if (p.type === 'line') {
      ctx.lineWidth = p.width || 1;
      ctx.setLineDash(p.dash || []);
      ctx.beginPath();
      ctx.moveTo(p.a.x, p.a.y);
      ctx.lineTo(p.b.x, p.b.y);
      ctx.stroke();
      if (p.arrow && Math.hypot(p.b.x - p.a.x, p.b.y - p.a.y) > 12) {
        const angle = Math.atan2(p.b.y - p.a.y, p.b.x - p.a.x);
        const size = 5 + (p.width || 1);
        ctx.setLineDash([]);
        ctx.fillStyle = p.color || PALETTE.ink;
        ctx.beginPath();
        ctx.moveTo(p.b.x, p.b.y);
        ctx.lineTo(p.b.x - size * Math.cos(angle - 0.44), p.b.y - size * Math.sin(angle - 0.44));
        ctx.lineTo(p.b.x - size * Math.cos(angle + 0.44), p.b.y - size * Math.sin(angle + 0.44));
        ctx.closePath();
        ctx.fill();
      }
    } else {
      ctx.setLineDash([]);
      const r = p.radius || 4;
      if (p.halo) {
        ctx.globalAlpha = (p.alpha ?? 1) * 0.13;
        ctx.beginPath(); ctx.arc(p.x, p.y, r + 5, 0, TAU); ctx.fill();
        ctx.globalAlpha = p.alpha ?? 1;
      }
      ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, TAU); ctx.fill();
      ctx.lineWidth = p.strokeWidth || 1.5;
      ctx.stroke();
      if (p.halo) {
        ctx.strokeStyle = '#FFFFFF'; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(p.x, p.y, Math.max(1, r - 1), 0, TAU); ctx.stroke();
      }
    }
  }

  _drawLabels(labels) {
    const ctx = this.ctx, placed = [];
    ctx.globalAlpha = 1;
    ctx.setLineDash([]);
    ctx.font = '600 12px system-ui, -apple-system, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    labels.sort((a, b) => b.priority - a.priority);
    for (const label of labels) {
      const textWidth = ctx.measureText(label.text).width;
      let x = clamp(label.x + label.dx, 8, Math.max(8, this.width - textWidth - 12));
      let y = clamp(label.y + label.dy, 16, this.height - 29);
      for (let attempt = 0; attempt < 6; attempt += 1) {
        const box = { x: x - 4, y: y - 9, w: textWidth + 8, h: 18 };
        if (!placed.some(p => box.x < p.x + p.w && box.x + box.w > p.x && box.y < p.y + p.h && box.y + box.h > p.y)) break;
        y = clamp(y + (label.dy >= 0 ? 19 : -19), 16, this.height - 29);
      }
      placed.push({ x: x - 4, y: y - 9, w: textWidth + 8, h: 18 });
      ctx.fillStyle = 'rgba(255,255,255,0.88)';
      ctx.fillRect(x - 4, y - 9, textWidth + 8, 18);
      ctx.fillStyle = label.color;
      ctx.fillText(label.text, x, y);
    }
  }

  _drawOffscreen(position, view) {
    const ctx = this.ctx, w = this.width, h = this.height;
    const cameraPoint = view.toCamera(position);
    let dx = cameraPoint[0], dy = -cameraPoint[1];
    const n = Math.hypot(dx, dy);
    if (n < 1e-8) { dx = 0; dy = -1; } else { dx /= n; dy /= n; }
    const halfW = Math.max(20, w / 2 - 28), halfH = Math.max(20, h / 2 - 35);
    const scale = Math.min(Math.abs(dx) > 1e-8 ? halfW / Math.abs(dx) : Infinity, Math.abs(dy) > 1e-8 ? halfH / Math.abs(dy) : Infinity);
    const x = w / 2 + dx * scale, y = h / 2 + dy * scale;
    ctx.globalAlpha = 1;
    ctx.fillStyle = PALETTE.orange;
    const side = [-dy, dx];
    ctx.beginPath();
    ctx.moveTo(x + dx * 7, y + dy * 7);
    ctx.lineTo(x - dx * 5 + side[0] * 5, y - dy * 5 + side[1] * 5);
    ctx.lineTo(x - dx * 5 - side[0] * 5, y - dy * 5 - side[1] * 5);
    ctx.closePath(); ctx.fill();
    const text = 'UE estimate outside view';
    ctx.font = '600 12px system-ui, -apple-system, sans-serif';
    const tw = ctx.measureText(text).width;
    const tx = clamp(x - tw / 2, 12, Math.max(12, w - tw - 12));
    const ty = clamp(y - dy * 23, 18, h - 31);
    ctx.fillStyle = 'rgba(255,255,255,0.95)';
    ctx.fillRect(tx - 5, ty - 10, tw + 10, 20);
    ctx.fillStyle = PALETTE.orange;
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText(text, tx, ty);
  }
}

export default Scene3D;
