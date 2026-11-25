import test from 'node:test';
import assert from 'node:assert/strict';
import {
  BACKGROUND, createTraining, evaluateTraining, stepTraining, inspectPixel,
  createMultiView, evaluateMultiView, stepMultiView, imageLoss, renderSplats,
  encodeSplats, decodeSplats, projectCenter,
} from '../live/model.mjs';

const close = (actual, expected, tolerance = 1e-10, message = '') => {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${message}: ${actual} versus ${expected}`);
};

test('front-to-back weights and the surviving background reproduce the selected pixel', () => {
  const state = createTraining({size: 16});
  const evaluation = evaluateTraining(state);
  for (const [x, y] of [[6, 5], [8, 8], [0, 0], [15, 15]]) {
    const probe = inspectPixel(state, x, y);
    close(probe.contributions.reduce((sum, c) => sum + c.weight, 0) + probe.remainingTransmittance, 1);
    let t = 1;
    for (const contribution of probe.contributions) {
      close(contribution.transmittance, t);
      close(contribution.alpha, contribution.opacity * contribution.gaussian);
      t *= 1 - contribution.alpha;
    }
    for (let channel = 0; channel < 3; channel++) {
      close(probe.prediction[channel], evaluation.image[3 * (y * state.size + x) + channel]);
      close(probe.background[channel], t * BACKGROUND[channel]);
    }
  }
});

test('the splat Gaussian is an unnormalized footprint and covariance remains positive', () => {
  const splat = {x: 0.5, y: 0.5, sx: 0.11, sy: 0.21, angle: 0.7, opacity: 0.6, color: [0.4, 0.8, 0.2]};
  const [decoded] = decodeSplats(encodeSplats([splat]));
  for (const key of ['x', 'y', 'sx', 'sy', 'angle', 'opacity']) close(decoded[key], splat[key]);
  const c = Math.cos(splat.angle), s = Math.sin(splat.angle);
  const a = c * c * splat.sx ** 2 + s * s * splat.sy ** 2;
  const b = c * s * (splat.sx ** 2 - splat.sy ** 2);
  const d = s * s * splat.sx ** 2 + c * c * splat.sy ** 2;
  assert.ok(a > 0 && d > 0 && a * d - b * b > 0);
  // Odd image size places a pixel exactly at the Gaussian center. There is no
  // probability-density determinant factor multiplying the peak opacity.
  const image = renderSplats([splat], {size: 17});
  for (let channel = 0; channel < 3; channel++) {
    close(image[3 * (8 * 17 + 8) + channel], 0.6 * splat.color[channel] + 0.4 * BACKGROUND[channel]);
  }
});

test('the loss uses mean RGB L1 and exactly 1 minus SSIM, without a half factor', () => {
  const state = createTraining({size: 16});
  const r = evaluateTraining(state);
  close(r.loss, 0.8 * r.l1 + 0.2 * (1 - r.ssim));
  close(r.dssim, 1 - r.ssim);
  const identity = imageLoss(state.target, state.target, state.size, {gradient: true});
  close(identity.l1, 0);
  close(identity.ssim, 1);
  close(identity.loss, 0);
  assert.ok(Math.max(...identity.pixelGradient.map(Math.abs)) < 1e-10);
  const l1Only = imageLoss(r.image, state.target, state.size, {lambda: 0, gradient: true});
  close(l1Only.loss, l1Only.l1);
  for (let i = 0; i < r.image.length; i++) {
    close(l1Only.pixelGradient[i], Math.sign(r.image[i] - state.target[i]) / r.image.length);
  }
});

test('local SSIM agrees with an independent direct 11 by 11 window computation', () => {
  const state = createTraining({size: 12});
  const r = evaluateTraining(state);
  const k = Array.from({length: 11}, (_, i) => Math.exp(-0.5 * ((i - 5) / 1.5) ** 2));
  const total = k.reduce((sum, v) => sum + v, 0);
  const kernel = k.map(v => v / total);
  let ssimSum = 0;
  for (let y = 0; y < state.size; y++) {
    for (let x = 0; x < state.size; x++) {
      for (let ch = 0; ch < 3; ch++) {
        let mx = 0, my = 0, exx = 0, eyy = 0, exy = 0;
        for (let ky = -5; ky <= 5; ky++) {
          for (let kx = -5; kx <= 5; kx++) {
            const yy = y + ky, xx = x + kx;
            if (yy < 0 || yy >= state.size || xx < 0 || xx >= state.size) continue;
            const w = kernel[kx + 5] * kernel[ky + 5];
            const idx = 3 * (yy * state.size + xx) + ch;
            const a = r.image[idx], b = state.target[idx];
            mx += w * a; my += w * b;
            exx += w * a * a; eyy += w * b * b; exy += w * a * b;
          }
        }
        const vx = exx - mx * mx, vy = eyy - my * my, covariance = exy - mx * my;
        ssimSum += ((2 * mx * my + 0.01 ** 2) * (2 * covariance + 0.03 ** 2)) /
          ((mx * mx + my * my + 0.01 ** 2) * (vx + vy + 0.03 ** 2));
      }
    }
  }
  close(r.ssim, ssimSum / r.image.length, 1e-12);
});

test('all 45 analytic splat derivatives agree with central finite differences', () => {
  const state = createTraining({size: 16});
  const analytic = evaluateTraining(state).gradients;
  const h = 1e-6;
  for (let i = 0; i < state.params.length; i++) {
    const original = state.params[i];
    state.params[i] = original + h;
    const plus = evaluateTraining(state).loss;
    state.params[i] = original - h;
    const minus = evaluateTraining(state).loss;
    state.params[i] = original;
    const numerical = (plus - minus) / (2 * h);
    close(analytic[i], numerical, 2e-7, `parameter ${i}`);
  }
});

test('actual Adam steps fit the target image and report the post-update objective', () => {
  const state = createTraining();
  const initial = evaluateTraining(state);
  const fitted = stepTraining(state, {steps: 200});
  assert.equal(state.iteration, 200);
  assert.equal(state.history.length, 201);
  assert.ok(fitted.loss < initial.loss * 0.01, `${initial.loss} to ${fitted.loss}`);
  assert.ok(fitted.ssim > 0.999);
  assert.ok(fitted.image.every(Number.isFinite));
  assert.ok(fitted.image.every(v => v >= 0 && v <= 1));
  close(state.history.at(-1).loss, evaluateTraining(state).loss);
  assert.ok(state.params.some((v, i) => Math.abs(v - initialParameterCopy()[i]) > 0.01));
});

function initialParameterCopy() { return createTraining({size: 12}).params; }

test('freezing a parameter group preserves its exact values', () => {
  const state = createTraining({size: 16});
  const before = Array.from(state.params);
  stepTraining(state, {steps: 25, train: {position: false, opacity: false}});
  let someShapeOrColorChanged = false;
  for (let i = 0; i < state.params.length; i++) {
    const k = i % 9;
    if (k < 2 || k === 5) assert.equal(state.params[i], before[i]);
    else if (state.params[i] !== before[i]) someShapeOrColorChanged = true;
  }
  assert.ok(someShapeOrColorChanged);
});

test('multi-view center derivatives follow the calibrated projection Jacobian', () => {
  const state = createMultiView({size: 16, views: 2});
  const analytic = evaluateMultiView(state).gradients;
  const h = 1e-6;
  for (let i = 0; i < 3; i++) {
    const original = state.center[i];
    state.center[i] = original + h;
    const plus = evaluateMultiView(state).trainingLoss;
    state.center[i] = original - h;
    const minus = evaluateMultiView(state).trainingLoss;
    state.center[i] = original;
    close(analytic[i], (plus - minus) / (2 * h), 2e-7, `center coordinate ${i}`);
  }
  assert.deepEqual(projectCenter([0.2, 0.3, 0.7], 0), [0.2, 0.3]);
  assert.deepEqual(projectCenter([0.2, 0.3, 0.7], Math.PI / 2), [0.7, 0.3]);
});

test('one camera fits its image with wrong depth; a second camera repairs held-out rendering', () => {
  const state = createMultiView();
  const initial = evaluateMultiView(state);
  assert.equal(initial.gradients[2], 0);
  const frontOnly = stepMultiView(state, {steps: 100});
  assert.equal(frontOnly.center[2], state.initialCenter[2]);
  assert.ok(frontOnly.trainingLoss < initial.trainingLoss * 0.01);
  assert.ok(frontOnly.heldOutLoss > 0.08);
  assert.ok(frontOnly.depthError > 0.4);
  const both = stepMultiView(state, {steps: 150, views: 2});
  assert.ok(both.depthError < 0.005, `z=${both.center[2]}`);
  assert.ok(both.heldOutLoss < frontOnly.heldOutLoss * 0.01);
  assert.equal(both.cameras[2].training, false);
  assert.equal(both.cameras[0].training, true);
  assert.equal(both.cameras[1].training, true);
  assert.equal(state.history.length, 251);
});

test('the displayed true center does not enter training or its gradient', () => {
  const original = createMultiView({size: 16, views: 2});
  const changedOverlay = createMultiView({size: 16, views: 2});
  changedOverlay.reference = [0.01, 0.99, -20];
  const a = stepMultiView(original, {steps: 20});
  const b = stepMultiView(changedOverlay, {steps: 20});
  assert.deepEqual(a.center, b.center);
  assert.deepEqual(a.gradients, b.gradients);
  close(a.trainingLoss, b.trainingLoss);
  close(a.heldOutLoss, b.heldOutLoss);
  assert.notEqual(a.depthError, b.depthError);
});

test('changing loss weight or camera selection without stepping preserves the fitted parameters', () => {
  const image = createTraining({size: 16});
  const imageBefore = Array.from(image.params);
  const l1Only = stepTraining(image, {steps: 0, lambda: 0});
  close(l1Only.loss, l1Only.l1);
  assert.deepEqual(Array.from(image.params), imageBefore);
  const multi = createMultiView({size: 16});
  const centerBefore = Array.from(multi.center);
  const two = stepMultiView(multi, {steps: 0, views: 2});
  assert.equal(two.views, 2);
  assert.deepEqual(two.center, centerBefore);
});
