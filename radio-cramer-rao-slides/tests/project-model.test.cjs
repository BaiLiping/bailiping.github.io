"use strict";
const assert = require("node:assert/strict");
const M = require("../project/model.js");
function near(a, b, rel = 3e-9, abs = 1e-11) {
  assert.ok(Math.abs(a - b) <= abs + rel * Math.abs(b), `${a} != ${b}`);
}

// Independent raw-IQ implementation: construct ONE physical X[T,S], then
// explicitly calculate all complex observations. This does not use the
// engine's pilot moments, array moments, FIM, or matrix inverse.
function rawIQ(p, theta) {
  const nt = p.txY * p.txZ, nr = p.rxY * p.rxZ, K = p.tones, L = p.symbols;
  const rad = Math.PI / 180, P = 10 ** ((p.txPowerDbm - 30) / 10);
  const magnitude = Math.sqrt(P / (nt * K));
  let state = (p.pilotSeed >>> 0) || 0x9e3779b9;
  const X = Array.from({ length: nt }, () => []);
  for (let l = 0; l < L; l++) for (let t = 0; t < nt; t++) {
    state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
    const quadrant = state >>> 30;
    const qpsk = [[1, 0], [0, 1], [-1, 0], [0, -1]][quadrant];
    X[t].push(qpsk.map(v => v * magnitude));
  }
  for (let l = 0; l < L; l++) {
    near(K * X.reduce((sum, row) => sum + row[l][0] ** 2 + row[l][1] ** 2, 0), P);
  }
  function array(ny, nz, az, el) {
    az *= rad; el *= rad;
    const result = [];
    for (let y = 0; y < ny; y++) for (let z = 0; z < nz; z++) {
      const py = (y - (ny - 1) / 2) * p.dLambda;
      const pz = (z - (nz - 1) / 2) * p.dLambda;
      result.push(2 * Math.PI * (py * Math.cos(el) * Math.sin(az) + pz * Math.sin(el)));
    }
    return result;
  }
  const tx = array(p.txY, p.txZ, theta[3], theta[4]);
  const rx = array(p.rxY, p.rxZ, theta[1], theta[2]);
  const alpha = 10 ** (-theta[5] / 20), result = [];
  for (let k = 0; k < K; k++) {
    const f = (k - Math.floor(K / 2)) * p.bandwidthMHz * 1e6 / K;
    for (let r = 0; r < nr; r++) for (let l = 0; l < L; l++) {
      let re = 0, im = 0;
      for (let t = 0; t < nt; t++) {
        const angle = rx[r] - tx[t] - 2 * Math.PI * f * theta[0] * 1e-9 + theta[6];
        re += alpha * (Math.cos(angle) * X[t][l][0] - Math.sin(angle) * X[t][l][1]);
        im += alpha * (Math.sin(angle) * X[t][l][0] + Math.cos(angle) * X[t][l][1]);
      }
      result.push([re, im]);
    }
  }
  return result;
}

const defaults = M.compute();
assert.equal(defaults.valid, true);
assert.equal(defaults.pilot.rank, 50);
assert.equal(defaults.pilot.rankUpperBound, 50);
assert.equal(defaults.rank, 7);
assert.equal(defaults.parameters[0].label, "Apparent delay");
assert.ok(defaults.parameters.every(p => p.identifiable && Number.isFinite(p.std) && p.std > 0));
assert.equal(defaults.signal.txPorts, 384);
assert.equal(defaults.signal.rxPorts, 32);
assert.equal(defaults.signal.complexObservations, 3300 * 32 * 50);
near(defaults.grid.usefulTimeS, 50 / (400e6 / 3300));
near(defaults.signal.noiseVarianceW, 10 ** ((-174 + 9 - 30) / 10) * (400e6 / 3300));
near(defaults.pilot.energyFactor, 0.9511495853608255);

// Even K, fewer symbols than TX ports, off-broadside angles, and a nonzero
// delay/phase exercise all actual coded-pilot nuisance cross terms.
const p = { ...M.defaults, txY: 3, txZ: 2, rxY: 2, rxZ: 3, symbols: 4,
  tones: 6, bandwidthMHz: 36, txAz: 28, txEl: -17, rxAz: -21, rxEl: 14,
  txPowerDbm: 10, lossDb: 75, nfDb: 7, dLambda: 0.43, pilotSeed: 1827 };
const r = M.compute(p);
assert.equal(r.valid, true);
assert.equal(r.rank, 7);
assert.equal(r.pilot.rank, 4);
const theta = [83, p.rxAz, p.rxEl, p.txAz, p.txEl, p.lossDb, 0.61];
const iq = rawIQ(p, theta);
near(iq.reduce((sum, z) => sum + z[0] ** 2 + z[1] ** 2, 0) / r.signal.noiseVarianceW, r.signal.gamma);
const steps = [1e-3, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5];
const derivatives = theta.map((_, i) => {
  const plus = theta.slice(), minus = theta.slice();
  plus[i] += steps[i]; minus[i] -= steps[i];
  const hi = rawIQ(p, plus), lo = rawIQ(p, minus);
  return hi.map((z, k) => z.map((v, axis) => (v - lo[k][axis]) / (2 * steps[i])));
});
for (let i = 0; i < 7; i++) for (let j = 0; j < 7; j++) {
  const gram = 2 / r.signal.noiseVarianceW * derivatives[i].reduce((sum, z, k) =>
    sum + z[0] * derivatives[j][k][0] + z[1] * derivatives[j][k][1], 0);
  const error = Math.abs(gram - r.fim[i][j]) / Math.sqrt(r.fim[i][i] * r.fim[j][j]);
  assert.ok(error < 8e-8, `Raw-IQ derivative FIM mismatch [${i},${j}]: ${error}`);
}
assert.notEqual(r.fim[0][6], 0, "Even-K baseband grid must retain delay/phase coupling.");
assert.notEqual(r.fim[3][5], 0, "Coded TX must retain angle/amplitude coupling.");
assert.notEqual(r.fim[3][6], 0, "Coded TX must retain angle/phase coupling.");
for (let i = 0; i < 7; i++) for (let j = 0; j < 7; j++) {
  near(r.fim[i].reduce((sum, entry, k) => sum + entry * r.covariance[k][j], 0), i === j ? 1 : 0, 1e-8, 1e-8);
}

// Profiling unknown complex gain centers the frequency derivative exactly,
// despite the coded TX angle/gain correlations and noncentered even-K grid.
near(r.bounds.tauNs, 1e9 / (2 * Math.PI * r.grid.betaHz * Math.sqrt(2 * r.signal.gamma)));
near(defaults.bounds.tauNs, 1e9 / (2 * Math.PI * defaults.grid.betaHz * Math.sqrt(2 * defaults.signal.gamma)));
assert.ok(r.boundsKnownPhase.tauNs <= r.bounds.tauNs * (1 + 1e-10));
assert.ok(r.boundsKnownGainPhase.txAzDeg <= r.bounds.txAzDeg * (1 + 1e-10));
assert.ok(r.boundsKnownGainPhase.txElDeg <= r.bounds.txElDeg * (1 + 1e-10));
near(defaults.signal.gamma, 32 * 3300 * 50 * defaults.pilot.energyFactor *
  (10 ** ((55 - 30 - 130) / 10)) / (10 ** ((-174 + 9 - 30) / 10) * 400e6));

const morePower = M.compute({ txPowerDbm: 55 + 10 * Math.log10(2) });
near(morePower.signal.gamma / defaults.signal.gamma, 2);
near(morePower.bounds.txAzDeg / defaults.bounds.txAzDeg, 1 / Math.sqrt(2));
const moreNoise = M.compute({ nfDb: 9 + 10 * Math.log10(2) });
near(moreNoise.signal.gamma / defaults.signal.gamma, 0.5);
const wider = M.compute({ bandwidthMHz: 800 });
near(wider.signal.gamma / defaults.signal.gamma, 0.5);
near(wider.bounds.tauNs / defaults.bounds.tauNs, 1 / Math.sqrt(2));
near(wider.bounds.txAzDeg / defaults.bounds.txAzDeg, Math.sqrt(2));
const fixedGamma = M.compute({ mode: "total", snrDb: 30 });
near(fixedGamma.signal.gamma, 1000);
const higherCarrier = M.compute({ fcGHz: 54.4 });
near(higherCarrier.bounds.txAzDeg, defaults.bounds.txAzDeg);
near(higherCarrier.bounds.tauNs, defaults.bounds.tauNs);
assert.notEqual(M.compute({ pilotSeed: 20260923 }).pilot.energyFactor, defaults.pilot.energyFactor);

// An underdetermined unstructured channel matrix and an identifiable path
// are distinct cases. L=1 really does lose TX/gain information with fixed X.
const oneSymbol = M.compute({ symbols: 1 });
assert.equal(oneSymbol.valid, true);
assert.equal(oneSymbol.pilot.rank, 1);
assert.equal(oneSymbol.rank, 5);
assert.equal(oneSymbol.bounds.txAzDeg, Infinity);
assert.equal(oneSymbol.bounds.txElDeg, Infinity);
assert.equal(oneSymbol.bounds.lossDb, Infinity);
assert.equal(oneSymbol.bounds.phaseRad, Infinity);
assert.ok(Number.isFinite(oneSymbol.bounds.tauNs));
assert.ok(Number.isFinite(oneSymbol.bounds.rxAzDeg));
const vertical = M.compute({ rxY: 1, rxZ: 4 });
assert.equal(vertical.bounds.rxAzDeg, Infinity);
assert.ok(Number.isFinite(vertical.bounds.rxElDeg));
const horizontal = M.compute({ rxY: 4, rxZ: 1 });
assert.equal(horizontal.bounds.rxAzDeg, Infinity);
assert.equal(horizontal.bounds.rxElDeg, Infinity);
const pointTX = M.compute({ txY: 1, txZ: 1 });
assert.equal(pointTX.bounds.txAzDeg, Infinity);
assert.equal(pointTX.bounds.txElDeg, Infinity);
assert.ok(Number.isFinite(pointTX.bounds.lossDb));
assert.ok(Number.isFinite(pointTX.bounds.phaseRad));
const oneTone = M.compute({ tones: 1 });
assert.equal(oneTone.bounds.tauNs, Infinity);
assert.ok(Number.isFinite(oneTone.bounds.txAzDeg));
assert.equal(M.compute({ rxAz: 90 }).bounds.rxAzDeg, Infinity);
assert.equal(M.compute({ symbols: 0 }).valid, false);
assert.equal(M.compute({ txY: 2.5 }).valid, false);
assert.equal(M.compute({ rxAz: 100 }).valid, false);
const tooLarge = M.compute({ symbols: 100000000 });
assert.equal(tooLarge.valid, false);
assert.ok(tooLarge.errors.some(e => e.includes("browser work limit")));
assert.equal(M.compute({ rxY: 100000000 }).valid, false);

const clock = M.clock();
assert.equal(clock.valid, true);
near(clock.weights.reduce((sum, p) => sum + p.weight, 0), 1);
near(clock.meanFrequencyHz, -(400e6 / 3300) / 2, 1e-8);
near(clock.bounds.unknownPhaseNs, 0.030820223618144973);
near(clock.fim[0][0] - clock.fim[0][1] ** 2 / clock.fim[1][1],
  1 / clock.bounds.unknownPhaseNs ** 2);
const two = M.clock({ tones: 2 });
near(two.bounds.unknownPhaseNs / two.bounds.knownPhaseNs, Math.sqrt(2));
const tilted = M.clock({ spectralTilt: 3 });
assert.ok(tilted.meanFrequencyHz > 0);
assert.ok(tilted.bounds.unknownPhaseNs > tilted.bounds.knownPhaseNs);
near(tilted.fim[0][0] - tilted.fim[0][1] ** 2 / tilted.fim[1][1],
  1 / tilted.bounds.unknownPhaseNs ** 2);
const narrow = M.clock({ occupiedFraction: 0.5 });
assert.equal(narrow.grid.activeTones, 1650);
assert.ok(narrow.betaHz < clock.betaHz);
assert.ok(narrow.bounds.unknownPhaseNs > clock.bounds.unknownPhaseNs);
const oneNonzeroTone = M.clock({ tones: 8, occupiedFraction: 0.125, spectralCenter: 1 });
assert.equal(oneNonzeroTone.bounds.unknownPhaseNs, Infinity);
assert.ok(Number.isFinite(oneNonzeroTone.bounds.knownPhaseNs));
const oneZeroTone = M.clock({ tones: 1 });
assert.equal(oneZeroTone.bounds.unknownPhaseNs, Infinity);
assert.equal(oneZeroTone.bounds.knownPhaseNs, Infinity);
assert.equal(M.clock({ occupiedFraction: 0 }).valid, false);
assert.equal(M.clock({ tones: 100000000 }).valid, false);
assert.equal(M.clock({ snrDb: 4000 }).valid, false);

console.log("Project CRB verified: fixed physical QPSK X, independent raw-IQ finite-difference FIM, marginal nuisance elimination, pilot/channel rank distinction, power/noise normalization, and weighted clock bounds.");
