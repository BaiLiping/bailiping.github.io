"use strict";
const assert = require("node:assert/strict");
const CRB = require("../live/model.js");

function near(actual, expected, relative = 2e-10, absolute = 2e-12) {
  assert.ok(Math.abs(actual - expected) <= absolute + relative * Math.abs(expected),
    `Expected ${actual} to equal ${expected}`);
}
function ratio(actual, expected, value) { near(actual / expected, value); }

// An independent explicit complex derivative calculation. Unlike the model,
// this constructs every antenna, pilot, and subcarrier observation and sums
// the derivative Gram matrix. Balanced DFT pilots have L >= N_T and each
// TX port uses 1/(N_T K) watts in each symbol, so total P = 1 watt.
function bruteForceFIM(p, gamma) {
  const c = 299792458, rad = Math.PI / 180;
  const lambda = c / (p.fcGHz * 1e9), spacing = p.dLambda * lambda;
  const k0 = 2 * Math.PI / lambda;
  const nt = p.txY * p.txZ, nr = p.rxY * p.rxZ, K = p.tones, L = p.symbols;
  const alpha = Math.sqrt(gamma / (nr * L)); // complex noise variance = 1
  const pilotMagnitude = Math.sqrt(1 / (nt * K));
  const deltaF = p.bandwidthMHz * 1e6 / K;
  function array(ny, nz, azDeg, elDeg) {
    const az = azDeg * rad, el = elDeg * rad;
    const u = [Math.cos(el) * Math.sin(az), Math.sin(el)];
    const da = [Math.cos(el) * Math.cos(az) * rad, 0];
    const de = [-Math.sin(el) * Math.sin(az) * rad, Math.cos(el) * rad];
    const result = [];
    for (let y = 0; y < ny; y++) for (let z = 0; z < nz; z++) {
      const xyz = [(y - (ny - 1) / 2) * spacing, (z - (nz - 1) / 2) * spacing];
      const dot = v => xyz[0] * v[0] + xyz[1] * v[1];
      result.push({ phase: k0 * dot(u), az: k0 * dot(da), el: k0 * dot(de) });
    }
    return result;
  }
  const tx = array(p.txY, p.txZ, p.txAz, p.txEl);
  const rx = array(p.rxY, p.rxZ, p.rxAz, p.rxEl);
  const J = Array.from({ length: 7 }, () => Array(7).fill(0));
  let energy = 0;
  const mul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
  const imul = (a, s) => [-s * a[1], s * a[0]];
  const scale = (a, s) => [s * a[0], s * a[1]];
  for (let k = 0; k < K; k++) {
    const f = (k - (K - 1) / 2) * deltaF;
    for (let l = 0; l < L; l++) {
      let sum = [0, 0], sumAz = [0, 0], sumEl = [0, 0];
      tx.forEach((t, ti) => {
        const phase = -t.phase + 2 * Math.PI * ti * l / L;
        const z = [pilotMagnitude * Math.cos(phase), pilotMagnitude * Math.sin(phase)];
        const za = imul(z, -t.az), ze = imul(z, -t.el);
        sum = [sum[0] + z[0], sum[1] + z[1]];
        sumAz = [sumAz[0] + za[0], sumAz[1] + za[1]];
        sumEl = [sumEl[0] + ze[0], sumEl[1] + ze[1]];
      });
      rx.forEach(r => {
        // Nonzero tau and nuisance phase ensure phase bookkeeping is tested.
        const phase = r.phase - 2 * Math.PI * f * 23e-9 + 0.47;
        const gain = [alpha * Math.cos(phase), alpha * Math.sin(phase)];
        const mu = mul(gain, sum);
        const deriv = [
          imul(mu, -2 * Math.PI * f * 1e-9),
          imul(mu, r.az), imul(mu, r.el),
          mul(gain, sumAz), mul(gain, sumEl),
          scale(mu, -Math.log(10) / 20), imul(mu, 1)
        ];
        energy += mu[0] ** 2 + mu[1] ** 2;
        for (let i = 0; i < 7; i++) for (let j = 0; j < 7; j++) {
          J[i][j] += 2 * (deriv[i][0] * deriv[j][0] + deriv[i][1] * deriv[j][1]);
        }
      });
    }
  }
  near(energy, gamma);
  return J;
}

const base = CRB.compute();
assert.equal(base.valid, true);
assert.equal(base.parameters.length, 7);
assert.ok(base.parameters.every(p => p.identifiable && p.std > 0));
near(base.grid.spacingHz, 400e6 / 3300);
near(base.grid.spanHz, 3299 * 400e6 / 3300);
near(base.grid.betaHz, 400e6 * Math.sqrt((1 - 1 / 3300 ** 2) / 12));
near(base.grid.usefulTimeS, 64 / base.grid.spacingHz);
near(base.signal.gamma, 1000);
near(base.bounds.tauNs, 0.03082022361, 2e-9);
near(base.bounds.lossDb, 20 / (Math.log(10) * Math.sqrt(2000)));
near(base.bounds.phaseRad, 1 / Math.sqrt(2000));

const small = { ...CRB.defaults, txY: 3, txZ: 2, rxY: 2, rxZ: 3,
  symbols: 8, tones: 5, fcGHz: 3, dLambda: 0.41, bandwidthMHz: 50,
  rxAz: 31, rxEl: -18, txAz: -24, txEl: 26, snrDb: 14 };
const closed = CRB.compute(small);
const brute = bruteForceFIM(small, closed.signal.gamma);
for (let i = 0; i < 7; i++) for (let j = 0; j < 7; j++) near(closed.fim[i][j], brute[i][j]);

// Full inversion must recover identity and account for azimuth/elevation
// coupling. A diagonal reciprocal would give an overly optimistic bound.
for (let i = 0; i < 7; i++) for (let j = 0; j < 7; j++) {
  near(closed.fim[i].reduce((s, value, k) => s + value * closed.covariance[k][j], 0), i === j ? 1 : 0);
}
assert.ok(closed.covariance[1][1] > 1 / closed.fim[1][1]);
assert.ok(closed.covariance[3][3] > 1 / closed.fim[3][3]);

// Energy normalization: no TX-count multiplier at fixed power or fixed
// aggregate energy. Adding RX elements or symbols matters only in the modes
// whose energy budget actually increases with those observations.
const perTone = CRB.compute({ mode: "perTone", snrDb: -20 });
near(perTone.signal.gamma, 16 * 3300 * 64 * 0.01);
const halfTx = CRB.compute({ mode: "perTone", snrDb: -20, txY: 4 });
near(perTone.signal.gamma, halfTx.signal.gamma);
const doubleL = CRB.compute({ mode: "perTone", snrDb: -20, symbols: 128 });
ratio(doubleL.signal.gamma, perTone.signal.gamma, 2);
ratio(doubleL.bounds.tauNs, perTone.bounds.tauNs, 1 / Math.sqrt(2));
const fixedEnergyMoreL = CRB.compute({ symbols: 128 });
near(fixedEnergyMoreL.bounds.tauNs, base.bounds.tauNs);
const moreRx = CRB.compute({ mode: "perTone", snrDb: -20, rxY: 8 });
ratio(moreRx.signal.gamma, perTone.signal.gamma, 2);

const link = CRB.compute({ mode: "link" });
near(link.signal.noiseDbm, -173.975 + 9 + 10 * Math.log10(400e6));
near(link.signal.rhoPerTone, 10 ** ((35 - 120 - link.signal.noiseDbm) / 10));
near(link.signal.gamma, 16 * 3300 * 64 * link.signal.rhoPerTone);
const widerFixedPower = CRB.compute({ mode: "link", bandwidthMHz: 800 });
ratio(widerFixedPower.signal.gamma, link.signal.gamma, 0.5);
ratio(widerFixedPower.bounds.tauNs, link.bounds.tauNs, 1 / Math.sqrt(2));
ratio(widerFixedPower.bounds.rxAzDeg, link.bounds.rxAzDeg, Math.sqrt(2));
const doublePower = CRB.compute({ mode: "link", txPowerDbm: 35 + 10 * Math.log10(2) });
ratio(doublePower.bounds.lossDb, link.bounds.lossDb, 1 / Math.sqrt(2));

// Same bandwidth / aggregate energy: merely denser tone sampling has no
// arbitrary sqrt(K) gain. At fixed tone spacing the occupied band expands.
const denser = CRB.compute({ tones: 6600 });
near(denser.signal.gamma, base.signal.gamma);
ratio(denser.bounds.tauNs, base.bounds.tauNs,
  Math.sqrt((1 - 1 / 3300 ** 2) / (1 - 1 / 6600 ** 2)));
const exactSpacing = CRB.compute({ gridMode: "spacing", spacingKHz: 120 });
near(exactSpacing.grid.bandwidthHz, 396e6);

// At fixed d/lambda carrier frequency changes neither angular aperture in
// wavelengths nor the envelope delay CRB. Fixed physical spacing does.
const higherCarrier = CRB.compute({ fcGHz: 54.4 });
near(higherCarrier.bounds.rxAzDeg, base.bounds.rxAzDeg);
near(higherCarrier.bounds.tauNs, base.bounds.tauNs);
const physical = CRB.compute({ spacingMode: "physical", spacingMm: 4, fcGHz: 27.2 });
const physicalHigher = CRB.compute({ spacingMode: "physical", spacingMm: 4, fcGHz: 54.4 });
ratio(physicalHigher.bounds.rxAzDeg, physical.bounds.rxAzDeg, 0.5);
assert.ok(physicalHigher.warnings.some(w => w.includes("Grating")));

// Singular direction handling never treats a pseudoinverse's zeros as zero
// uncertainty. Other independent blocks remain available.
const oneTone = CRB.compute({ tones: 1 });
assert.equal(oneTone.bounds.tauNs, Infinity);
assert.ok(Number.isFinite(oneTone.bounds.rxAzDeg));
assert.equal(oneTone.correlation[0][0] !== oneTone.correlation[0][0], true);
const pointRx = CRB.compute({ rxY: 1, rxZ: 1 });
assert.equal(pointRx.bounds.rxAzDeg, Infinity);
assert.equal(pointRx.bounds.rxElDeg, Infinity);
assert.ok(Number.isFinite(pointRx.bounds.txAzDeg));
const verticalRx = CRB.compute({ rxY: 1, rxZ: 4 });
assert.equal(verticalRx.bounds.rxAzDeg, Infinity);
assert.ok(Number.isFinite(verticalRx.bounds.rxElDeg));
const horizontalGeneric = CRB.compute({ rxY: 4, rxZ: 1, rxAz: 25, rxEl: 10 });
assert.equal(horizontalGeneric.geometry.rx.rank, 1);
assert.equal(horizontalGeneric.bounds.rxAzDeg, Infinity);
assert.equal(horizontalGeneric.bounds.rxElDeg, Infinity);
const horizontalBroadside = CRB.compute({ rxY: 4, rxZ: 1, rxAz: 0, rxEl: 0 });
assert.ok(Number.isFinite(horizontalBroadside.bounds.rxAzDeg));
assert.equal(horizontalBroadside.bounds.rxElDeg, Infinity);
const grazing = CRB.compute({ rxAz: 90 });
assert.equal(grazing.bounds.rxAzDeg, Infinity);
assert.ok(Number.isFinite(grazing.bounds.rxElDeg));
const polar = CRB.compute({ rxEl: 90, rxAz: 20 });
assert.equal(polar.bounds.rxAzDeg, Infinity);
assert.ok(Number.isFinite(polar.bounds.rxElDeg));

const invalid = CRB.compute({ symbols: 63 });
assert.equal(invalid.valid, false);
assert.ok(invalid.errors.some(e => e.includes("L >= N_T")));
assert.ok(Number.isNaN(invalid.bounds.tauNs));
assert.equal(CRB.compute({ txY: 1.5 }).valid, false);
assert.equal(CRB.compute({ bandwidthMHz: 0 }).valid, false);
assert.equal(CRB.compute({ mode: "made-up" }).valid, false);
assert.equal(CRB.compute({ snrDb: Infinity }).valid, false);
assert.equal(CRB.compute({ txY: 1, txZ: 1, symbols: 1 }).valid, true);
assert.equal(CRB.sweep({}, "symbols", [32, 64, 128])[0].result.valid, false);

console.log("CRB model verified: explicit complex derivative Gram, marginal inverse, energy/power scaling, grid/carrier effects, and singular/pilot-rank handling.");
