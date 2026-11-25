/**
 * Computed teaching models for the Gaussian-splatting training lesson.
 *
 * The image experiment is a 2D reduction, with a fixed front-to-back order and
 * constant RGB colors. It is not the original 3DGS CUDA rasterizer. Its loss is
 * the original implementation's L1 + (1 - SSIM) convention, including the
 * 11 x 11, sigma=1.5 Gaussian SSIM window and zero padding. All gradients below
 * are analytic. Adam receives target pixels only, never the generating splats.
 *
 * The geometry experiment estimates one 3D center from known orthographic
 * cameras. Scale, opacity, and color are fixed, deliberately removing their
 * ambiguities so the missing depth constraint of one camera is visible.
 */

export const BACKGROUND = Object.freeze([0.045, 0.065, 0.075]);
export const PARAMETER_GROUPS = Object.freeze(['position', 'shape', 'opacity', 'color']);
export const MODEL_NOTES = Object.freeze({
  image: '2D anisotropic Gaussians; fixed layer order; constant RGB; full footprints without tile culling; analytic gradients and Adam.',
  loss: 'Mean RGB L1 plus 1 minus mean local SSIM. SSIM uses an 11 by 11 Gaussian window, sigma 1.5, zero padding, C1 = 0.01² and C2 = 0.03².',
  geometry: 'One 3D Gaussian center; orthographic cameras; known scale, color and opacity; two training cameras and a separate 45-degree evaluation camera.',
});

const PARAMS_PER_SPLAT = 9;
const FULL_TRAIN = Object.freeze({position: true, shape: true, opacity: true, color: true});
const SSIM_RADIUS = 5;
const SSIM_C1 = 0.01 ** 2;
const SSIM_C2 = 0.03 ** 2;
const TWO_PI = 2 * Math.PI;
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const sigmoid = x => x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
const logit = x => Math.log(clamp(x, 1e-6, 1 - 1e-6) / (1 - clamp(x, 1e-6, 1 - 1e-6)));
const SSIM_KERNEL = (() => {
  const kernel = Float64Array.from({length: 11}, (_, i) => Math.exp(-0.5 * ((i - 5) / 1.5) ** 2));
  const total = kernel.reduce((sum, x) => sum + x, 0);
  return kernel.map(x => x / total);
})();

function validateSize(size) {
  if (!Number.isInteger(size) || size < 12 || size > 128) throw new RangeError('Image size must be an integer from 12 to 128.');
}

function gaussianBlur(image, size) {
  const temp = new Float64Array(image.length);
  const result = new Float64Array(image.length);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const offset = 3 * (y * size + x);
      for (let k = -SSIM_RADIUS; k <= SSIM_RADIUS; k++) {
        const xx = x + k;
        if (xx < 0 || xx >= size) continue;
        const from = 3 * (y * size + xx);
        const w = SSIM_KERNEL[k + SSIM_RADIUS];
        temp[offset] += w * image[from];
        temp[offset + 1] += w * image[from + 1];
        temp[offset + 2] += w * image[from + 2];
      }
    }
  }
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const offset = 3 * (y * size + x);
      for (let k = -SSIM_RADIUS; k <= SSIM_RADIUS; k++) {
        const yy = y + k;
        if (yy < 0 || yy >= size) continue;
        const from = 3 * (yy * size + x);
        const w = SSIM_KERNEL[k + SSIM_RADIUS];
        result[offset] += w * temp[from];
        result[offset + 1] += w * temp[from + 1];
        result[offset + 2] += w * temp[from + 2];
      }
    }
  }
  return result;
}

function targetStatistics(target, size) {
  const mean = gaussianBlur(target, size);
  const second = gaussianBlur(target.map(x => x * x), size);
  const variance = second.map((x, i) => x - mean[i] * mean[i]);
  return {mean, variance};
}

/**
 * Evaluate image loss and, optionally, its exact derivative with respect to RGB.
 * RGB arrays are interleaved, square, in [0,1]. No factor 1/2 in the SSIM term.
 */
export function imageLoss(image, target, size, {lambda = 0.2, gradient = false, targetStats} = {}) {
  if (image.length !== target.length || image.length !== size * size * 3) throw new RangeError('Incompatible image dimensions.');
  lambda = clamp(lambda, 0, 1);
  const m = image.length;
  const reference = targetStats || targetStatistics(target, size);
  const mean = gaussianBlur(image, size);
  const second = gaussianBlur(image.map(x => x * x), size);
  const cross = gaussianBlur(image.map((x, i) => x * target[i]), size);
  const residual = new Float64Array(m);
  const ssimMap = new Float64Array(m);
  const dMean = gradient ? new Float64Array(m) : null;
  const dVariance = gradient ? new Float64Array(m) : null;
  const dCovariance = gradient ? new Float64Array(m) : null;
  let absoluteSum = 0;
  let similaritySum = 0;
  for (let i = 0; i < m; i++) {
    const mx = mean[i], my = reference.mean[i];
    const vx = second[i] - mx * mx;
    const vy = reference.variance[i];
    const covariance = cross[i] - mx * my;
    const a = 2 * mx * my + SSIM_C1;
    const b = 2 * covariance + SSIM_C2;
    const c = mx * mx + my * my + SSIM_C1;
    const d = vx + vy + SSIM_C2;
    const similarity = a * b / (c * d);
    ssimMap[i] = similarity;
    similaritySum += similarity;
    residual[i] = Math.abs(image[i] - target[i]);
    absoluteSum += residual[i];
    if (gradient) {
      const dsDVar = -similarity / d;
      const dsDCov = 2 * a / (c * d);
      const dsDMu = 2 * my * b / (c * d) - similarity * 2 * mx / c;
      // The second-moment and covariance definitions also depend on the mean.
      dMean[i] = dsDMu - 2 * mx * dsDVar - my * dsDCov;
      dVariance[i] = dsDVar;
      dCovariance[i] = dsDCov;
    }
  }
  const l1 = absoluteSum / m;
  const ssim = similaritySum / m;
  const dssim = 1 - ssim;
  const loss = (1 - lambda) * l1 + lambda * dssim;
  let pixelGradient;
  if (gradient) {
    // The symmetric zero-padded convolution is its own transpose.
    const a = gaussianBlur(dMean, size);
    const b = gaussianBlur(dVariance, size);
    const c = gaussianBlur(dCovariance, size);
    pixelGradient = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      const dsDx = a[i] + 2 * image[i] * b[i] + target[i] * c[i];
      pixelGradient[i] = ((1 - lambda) * Math.sign(image[i] - target[i]) - lambda * dsDx) / m;
    }
  }
  return {loss, l1, ssim, dssim, residual, ssimMap, pixelGradient};
}

/** Parameter order: x, y, log sx, log sy, angle, logit opacity, RGB logits. */
export function encodeSplats(splats) {
  return Float64Array.from(splats.flatMap(s => [s.x, s.y, Math.log(s.sx), Math.log(s.sy), s.angle, logit(s.opacity), ...s.color.map(logit)]));
}

export function decodeSplats(params) {
  const splats = [];
  for (let i = 0; i < params.length; i += PARAMS_PER_SPLAT) {
    splats.push({
      x: params[i], y: params[i + 1], sx: Math.exp(params[i + 2]), sy: Math.exp(params[i + 3]),
      angle: params[i + 4], opacity: sigmoid(params[i + 5]),
      color: [sigmoid(params[i + 6]), sigmoid(params[i + 7]), sigmoid(params[i + 8])],
    });
  }
  return splats;
}

function renderWithCache(splats, size, background, keepCache = false) {
  const count = splats.length;
  const image = new Float64Array(size * size * 3);
  const alpha = keepCache ? new Float64Array(size * size * count) : null;
  const transmittance = keepCache ? new Float64Array(size * size * count) : null;
  const rotatedX = keepCache ? new Float64Array(size * size * count) : null;
  const rotatedY = keepCache ? new Float64Array(size * size * count) : null;
  const transforms = splats.map(s => ({cos: Math.cos(s.angle), sin: Math.sin(s.angle), ix: 1 / s.sx ** 2, iy: 1 / s.sy ** 2}));
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const p = y * size + x;
      const pixelX = (x + 0.5) / size;
      const pixelY = (y + 0.5) / size;
      let t = 1, r = 0, g = 0, b = 0;
      for (let i = 0; i < count; i++) {
        const s = splats[i], tr = transforms[i];
        const dx = pixelX - s.x, dy = pixelY - s.y;
        const u = tr.cos * dx + tr.sin * dy;
        const v = -tr.sin * dx + tr.cos * dy;
        const a = s.opacity * Math.exp(-0.5 * (u * u * tr.ix + v * v * tr.iy));
        if (keepCache) {
          const at = p * count + i;
          alpha[at] = a;
          transmittance[at] = t;
          rotatedX[at] = u;
          rotatedY[at] = v;
        }
        const weight = t * a;
        r += weight * s.color[0];
        g += weight * s.color[1];
        b += weight * s.color[2];
        t *= 1 - a;
      }
      image[3 * p] = r + t * background[0];
      image[3 * p + 1] = g + t * background[1];
      image[3 * p + 2] = b + t * background[2];
    }
  }
  return {image, alpha, transmittance, rotatedX, rotatedY, transforms};
}

/** Front-to-back splatting, with the last transmittance multiplying background. */
export function renderSplats(splats, {size = 32, background = BACKGROUND} = {}) {
  validateSize(size);
  return renderWithCache(splats, size, background).image;
}

function parameterGradient(splats, size, background, cache, upstream) {
  const count = splats.length;
  const result = new Float64Array(count * PARAMS_PER_SPLAT);
  for (let p = 0; p < size * size; p++) {
    const ur = upstream[3 * p], ug = upstream[3 * p + 1], ub = upstream[3 * p + 2];
    // B is the color that would be seen immediately behind the current splat.
    let br = background[0], bg = background[1], bb = background[2];
    for (let i = count - 1; i >= 0; i--) {
      const s = splats[i], tr = cache.transforms[i], at = p * count + i, j = i * PARAMS_PER_SPLAT;
      const a = cache.alpha[at], t = cache.transmittance[at];
      const u = cache.rotatedX[at], v = cache.rotatedY[at];
      const dAlpha = t * (ur * (s.color[0] - br) + ug * (s.color[1] - bg) + ub * (s.color[2] - bb));
      const dLogG = dAlpha * a;
      result[j] += dLogG * (tr.cos * u * tr.ix - tr.sin * v * tr.iy);
      result[j + 1] += dLogG * (tr.sin * u * tr.ix + tr.cos * v * tr.iy);
      result[j + 2] += dLogG * u * u * tr.ix;
      result[j + 3] += dLogG * v * v * tr.iy;
      result[j + 4] += dLogG * u * v * (tr.iy - tr.ix);
      result[j + 5] += dAlpha * a * (1 - s.opacity);
      result[j + 6] += ur * t * a * s.color[0] * (1 - s.color[0]);
      result[j + 7] += ug * t * a * s.color[1] * (1 - s.color[1]);
      result[j + 8] += ub * t * a * s.color[2] * (1 - s.color[2]);
      br = a * s.color[0] + (1 - a) * br;
      bg = a * s.color[1] + (1 - a) * bg;
      bb = a * s.color[2] + (1 - a) * bb;
    }
  }
  return result;
}

function adamState(length) {
  return {m: new Float64Array(length), v: new Float64Array(length), ticks: new Uint32Array(length)};
}

function adamUpdate(params, gradient, optimizer, rates, enabled) {
  for (let i = 0; i < params.length; i++) {
    if (!enabled[i]) continue;
    const g = gradient[i];
    optimizer.m[i] = 0.9 * optimizer.m[i] + 0.1 * g;
    optimizer.v[i] = 0.999 * optimizer.v[i] + 0.001 * g * g;
    const t = ++optimizer.ticks[i];
    const m = optimizer.m[i] / (1 - 0.9 ** t);
    const v = optimizer.v[i] / (1 - 0.999 ** t);
    params[i] -= rates[i] * m / (Math.sqrt(v) + 1e-8);
  }
}

function sceneForTarget() {
  return [
    {x: 0.30, y: 0.33, sx: 0.135, sy: 0.063, angle: -0.45, opacity: 0.90, color: [0.94, 0.39, 0.23]},
    {x: 0.65, y: 0.29, sx: 0.095, sy: 0.116, angle: 0.65, opacity: 0.86, color: [0.17, 0.76, 0.69]},
    {x: 0.54, y: 0.57, sx: 0.153, sy: 0.060, angle: 0.30, opacity: 0.91, color: [0.97, 0.76, 0.27]},
    {x: 0.28, y: 0.70, sx: 0.075, sy: 0.105, angle: -0.20, opacity: 0.86, color: [0.39, 0.50, 0.94]},
    {x: 0.77, y: 0.70, sx: 0.074, sy: 0.105, angle: 0.62, opacity: 0.88, color: [0.86, 0.35, 0.62]},
  ];
}

function initialSplats() {
  // This is a fixed, deliberately imperfect initialization, independent of any
  // target parameter comparison in the optimizer. No correspondence loss exists.
  return [
    {x: 0.38, y: 0.27, sx: 0.160, sy: 0.084, angle: 0.12, opacity: 0.65, color: [0.68, 0.54, 0.39]},
    {x: 0.56, y: 0.36, sx: 0.125, sy: 0.090, angle: 0.12, opacity: 0.64, color: [0.32, 0.58, 0.53]},
    {x: 0.60, y: 0.63, sx: 0.120, sy: 0.090, angle: -0.20, opacity: 0.67, color: [0.75, 0.62, 0.43]},
    {x: 0.23, y: 0.62, sx: 0.100, sy: 0.087, angle: 0.38, opacity: 0.66, color: [0.45, 0.46, 0.68]},
    {x: 0.82, y: 0.76, sx: 0.095, sy: 0.078, angle: 0.05, opacity: 0.65, color: [0.68, 0.45, 0.55]},
  ];
}

/** Create reproducible pixel-fitting experiment. Hidden source splats discarded. */
export function createTraining({size = 32, scenario = 'standard', lambda = 0.2} = {}) {
  validateSize(size);
  if (scenario !== 'standard') throw new RangeError('This lab supports the standard five-splat experiment.');
  const target = renderSplats(sceneForTarget(), {size});
  const params = encodeSplats(initialSplats());
  const state = {
    size, scenario, params, target, lambda: clamp(lambda, 0, 1), background: [...BACKGROUND],
    iteration: 0, history: [], optimizer: adamState(params.length),
    targetStats: targetStatistics(target, size),
  };
  const evaluation = evaluateTraining(state);
  state.initialLoss = evaluation.loss;
  state.history.push(historyPoint(evaluation));
  return state;
}

function historyPoint(evaluation) {
  return {iteration: evaluation.iteration, loss: evaluation.loss, l1: evaluation.l1, ssim: evaluation.ssim, dssim: evaluation.dssim, lambda: evaluation.lambda};
}

export function evaluateTraining(state) {
  const splats = decodeSplats(state.params);
  const rendered = renderWithCache(splats, state.size, state.background, true);
  const metrics = imageLoss(rendered.image, state.target, state.size, {lambda: state.lambda, gradient: true, targetStats: state.targetStats});
  const gradients = parameterGradient(splats, state.size, state.background, rendered, metrics.pixelGradient);
  return {
    size: state.size, image: rendered.image, target: state.target, splats, gradients,
    iteration: state.iteration, history: state.history, lambda: state.lambda,
    ...metrics,
  };
}

/** Mutates state; returns actual post-update render, objective and gradients. */
export function stepTraining(state, {steps = 1, learnRate = 0.025, train = FULL_TRAIN, lambda = state.lambda} = {}) {
  state.lambda = clamp(lambda, 0, 1);
  const groups = {...FULL_TRAIN, ...train};
  const enabled = Array.from(state.params, (_, i) => {
    const k = i % PARAMS_PER_SPLAT;
    return k < 2 ? groups.position : k < 5 ? groups.shape : k === 5 ? groups.opacity : groups.color;
  });
  let result = evaluateTraining(state);
  for (let step = 0; step < Math.max(0, Math.floor(steps)); step++) {
    // Group rates differ because position is in [0,1], while shape and appearance
    // use log/logit coordinates. Gradual decay makes L1's final kink less noisy.
    const base = clamp(learnRate, 0, 0.2) / (1 + state.iteration / 180);
    const rates = Float64Array.from(state.params, (_, i) => {
      const k = i % PARAMS_PER_SPLAT;
      return base * (k < 2 ? 0.22 : k < 4 ? 0.80 : k === 4 ? 0.80 : 1.0);
    });
    adamUpdate(state.params, result.gradients, state.optimizer, rates, enabled);
    // Numerical safety bounds are part of this small teaching reduction.
    for (let i = 0; i < state.params.length; i++) {
      const k = i % PARAMS_PER_SPLAT;
      if (!enabled[i]) continue;
      if (k < 2) state.params[i] = clamp(state.params[i], -0.15, 1.15);
      else if (k < 4) state.params[i] = clamp(state.params[i], Math.log(0.02), Math.log(0.5));
      else if (k === 4) state.params[i] = ((state.params[i] + Math.PI) % TWO_PI + TWO_PI) % TWO_PI - Math.PI;
      else state.params[i] = clamp(state.params[i], -8, 8);
    }
    state.iteration++;
    result = evaluateTraining(state);
    state.history.push(historyPoint(result));
  }
  return result;
}

/** Exact front-to-back contributions at a selected integer pixel coordinate. */
export function inspectPixel(state, pixelX, pixelY) {
  const x = clamp(Math.floor(pixelX), 0, state.size - 1);
  const y = clamp(Math.floor(pixelY), 0, state.size - 1);
  const px = (x + 0.5) / state.size, py = (y + 0.5) / state.size;
  const splats = decodeSplats(state.params);
  const prediction = [0, 0, 0];
  let t = 1;
  const contributions = splats.map((splat, index) => {
    const dx = px - splat.x, dy = py - splat.y;
    const c = Math.cos(splat.angle), s = Math.sin(splat.angle);
    const u = c * dx + s * dy, v = -s * dx + c * dy;
    const gaussian = Math.exp(-0.5 * (u * u / splat.sx ** 2 + v * v / splat.sy ** 2));
    const alpha = splat.opacity * gaussian;
    const weight = t * alpha;
    const rgb = splat.color.map(value => weight * value);
    rgb.forEach((value, channel) => { prediction[channel] += value; });
    const row = {index, gaussian, opacity: splat.opacity, alpha, transmittance: t, weight, color: [...splat.color], contribution: rgb};
    t *= 1 - alpha;
    return row;
  });
  const background = state.background.map(value => t * value);
  background.forEach((value, channel) => { prediction[channel] += value; });
  const offset = 3 * (y * state.size + x);
  const target = Array.from(state.target.slice(offset, offset + 3));
  return {x, y, prediction, target, residual: prediction.map((value, channel) => value - target[channel]), contributions, background, remainingTransmittance: t};
}

const CAMERAS = Object.freeze([
  {id: 'front', label: 'Front · x–y', angle: 0},
  {id: 'side', label: 'Side · z–y', angle: Math.PI / 2},
  {id: 'held-out', label: 'Held-out · 45°', angle: Math.PI / 4},
]);

export function projectCenter(center, angle) {
  const c = Math.abs(Math.cos(angle)) < 1e-15 ? 0 : Math.cos(angle);
  const s = Math.abs(Math.sin(angle)) < 1e-15 ? 0 : Math.sin(angle);
  return [0.5 + c * (center[0] - 0.5) + s * (center[2] - 0.5), center[1]];
}

function projectedSplat(center, camera) {
  const [x, y] = projectCenter(center, camera.angle);
  return {x, y, sx: 0.105, sy: 0.105, angle: 0, opacity: 0.92, color: [0.20, 0.82, 0.68]};
}

export function createMultiView({size = 32, views = 1, lambda = 0.2} = {}) {
  validateSize(size);
  // Kept solely for explanatory overlays and evaluation. Neither render loss
  // nor gradient reads reference, as the independence test verifies.
  const reference = [0.64, 0.40, 0.70];
  const center = Float64Array.from([0.33, 0.60, 0.27]);
  const cameras = CAMERAS.map(camera => {
    const target = renderSplats([projectedSplat(reference, camera)], {size});
    return {...camera, target, targetStats: targetStatistics(target, size)};
  });
  const state = {
    size, views: views >= 2 ? 2 : 1, lambda: clamp(lambda, 0, 1), center, reference,
    initialCenter: Array.from(center), background: [...BACKGROUND], cameras,
    optimizer: adamState(3), iteration: 0, history: [],
  };
  const result = evaluateMultiView(state);
  state.initialLoss = result.trainingLoss;
  state.history.push(multiHistoryPoint(result));
  return state;
}

function multiHistoryPoint(result) {
  return {iteration: result.iteration, trainingLoss: result.trainingLoss, heldOutLoss: result.heldOutLoss, depthError: result.depthError, views: result.views};
}

export function evaluateMultiView(state) {
  const gradients = new Float64Array(3);
  const views = state.views >= 2 ? 2 : 1;
  let trainingLoss = 0;
  const cameras = state.cameras.map((camera, i) => {
    const splat = projectedSplat(state.center, camera);
    const rendered = renderWithCache([splat], state.size, state.background, i < views);
    const metrics = imageLoss(rendered.image, camera.target, state.size, {
      lambda: state.lambda, gradient: i < views, targetStats: camera.targetStats,
    });
    if (i < views) {
      trainingLoss += metrics.loss / views;
      const gradient = parameterGradient([splat], state.size, state.background, rendered, metrics.pixelGradient);
      const c = Math.abs(Math.cos(camera.angle)) < 1e-15 ? 0 : Math.cos(camera.angle);
      const s = Math.abs(Math.sin(camera.angle)) < 1e-15 ? 0 : Math.sin(camera.angle);
      gradients[0] += gradient[0] * c / views;
      gradients[1] += gradient[1] / views;
      gradients[2] += gradient[0] * s / views;
    }
    return {
      id: camera.id, label: camera.label, angle: camera.angle, training: i < views,
      image: rendered.image, target: camera.target, projection: [splat.x, splat.y],
      ...metrics,
    };
  });
  return {
    size: state.size, center: Array.from(state.center), reference: [...state.reference],
    cameras, gradients, trainingLoss, heldOutLoss: cameras[2].loss,
    depthError: Math.abs(state.center[2] - state.reference[2]),
    iteration: state.iteration, history: state.history, views, lambda: state.lambda,
  };
}

/** The reference center is never consulted here. Only image gradients update it. */
export function stepMultiView(state, {steps = 1, views = state.views, learnRate = 0.035} = {}) {
  state.views = views >= 2 ? 2 : 1;
  let result = evaluateMultiView(state);
  for (let step = 0; step < Math.max(0, Math.floor(steps)); step++) {
    const rate = clamp(learnRate, 0, 0.2) / (1 + state.iteration / 100);
    // Freeze truly unobserved depth, including its momentum, in the one-view case.
    adamUpdate(state.center, result.gradients, state.optimizer, [rate, rate, rate], [true, true, state.views > 1]);
    for (let i = 0; i < 3; i++) state.center[i] = clamp(state.center[i], -0.1, 1.1);
    state.iteration++;
    result = evaluateMultiView(state);
    state.history.push(multiHistoryPoint(result));
  }
  return result;
}
