/* RadioProjectCRB: an isolated-path CRB with one actual-style coded pilot.
 *
 * mu[k,r,s] = alpha aR[r] sum_t(conj(aT[t]) X[t,s]) exp(-j2pi f[k] tau_a).
 * tau_a = tau_geometric + clock_offset is the apparent path delay.
 * ONE X[T,S] is reused on every tone. X=sqrt(Ptotal/(T K))*Q, Q in {1,i,-1,-i}.
 * Arrays have unit-modulus responses and centered coordinates in the yz plane.
 * The effective attenuation includes calibrated element-pattern amplitudes.
 * The seeded JavaScript pilot is illustrative. It does not reproduce a saved
 * NumPy pilot or a full ray-traced scene, and it is not made orthogonal.
 * f[k]=(k-floor(K/2))*B/K. sigma_complex^2=10^((-174+NF-30)/10)*B/K watts.
 * J_ij=2 Re{d_i^H d_j}/sigma_complex^2, real coordinates
 * [tau ns, RX az deg, RX el deg, TX az deg, TX el deg, loss dB, phase rad].
 * Nuisance cross terms are retained. Rank(X)<T does not itself imply that
 * this structured seven-parameter channel is unidentifiable.
 * The channel is unknown to the estimator: all seven parameters are estimated
 * from Y and X. The chosen path only sets where the CRB is evaluated.
 */
(function (root, factory) {
  "use strict";
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.RadioProjectCRB = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";
  var C = 299792458, DEG = Math.PI / 180, ALOSS = Math.log(10) / 20;
  var defaults = Object.freeze({ txY: 24, txZ: 16, rxY: 8, rxZ: 4, symbols: 50,
    tones: 3300, bandwidthMHz: 400, fcGHz: 27.2, dLambda: 0.5,
    txAz: 20, txEl: 10, rxAz: 25, rxEl: 10, txPowerDbm: 55, nfDb: 9,
    lossDb: 130, pilotSeed: 20260922, mode: "link", snrDb: 30 });
  var defs = [
    { key: "tauNs", label: "Apparent delay", unit: "ns" },
    { key: "rxAzDeg", label: "AoA azimuth", unit: "deg" },
    { key: "rxElDeg", label: "AoA elevation", unit: "deg" },
    { key: "txAzDeg", label: "AoD azimuth", unit: "deg" },
    { key: "txElDeg", label: "AoD elevation", unit: "deg" },
    { key: "lossDb", label: "Effective attenuation", unit: "dB" },
    { key: "phaseRad", label: "Reference phase", unit: "rad" }
  ];
  function matrix(n, fill) { return Array.from({ length: n }, function () { return Array(n).fill(fill); }); }
  function snap(x) { return Math.abs(x) < 1e-15 ? 0 : x; }
  function uniformGrid(K, B) {
    var df = B / K, meanIndex = (K - 1) / 2 - Math.floor(K / 2);
    var variance = df * df * (K * K - 1) / 12;
    return { bandwidthHz: B, spacingHz: df, spanHz: (K - 1) * df,
      firstFrequencyHz: -Math.floor(K / 2) * df,
      lastFrequencyHz: (K - 1 - Math.floor(K / 2)) * df,
      meanFrequencyHz: meanIndex * df, varianceHz2: variance,
      secondMomentHz2: variance + meanIndex * meanIndex * df * df,
      betaHz: Math.sqrt(variance), tones: K };
  }
  function array(ny, nz, dLambda, az, el, lambda) {
    var sa = snap(Math.sin(az * DEG)), ca = snap(Math.cos(az * DEG));
    var se = snap(Math.sin(el * DEG)), ce = snap(Math.cos(el * DEG));
    var pos = [];
    for (var y = 0; y < ny; y++) for (var z = 0; z < nz; z++) {
      var py = (y - (ny - 1) / 2) * dLambda, pz = (z - (nz - 1) / 2) * dLambda;
      pos.push({ phase: 2 * Math.PI * (py * ce * sa + pz * se),
        az: 2 * Math.PI * py * ce * ca * DEG,
        el: 2 * Math.PI * (-py * se * sa + pz * ce) * DEG });
    }
    var vy = dLambda * dLambda * (ny * ny - 1) / 12;
    var vz = dLambda * dLambda * (nz * nz - 1) / 12;
    var s = 4 * Math.PI * Math.PI * DEG * DEG;
    return { ny: ny, nz: nz, count: ny * nz, azDeg: az, elDeg: el, positions: pos,
      direction: [ce * ca, ce * sa, se], varY: vy * lambda * lambda, varZ: vz * lambda * lambda,
      apertureYM: (ny - 1) * dLambda * lambda, apertureZM: (nz - 1) * dLambda * lambda,
      derivativeMoment: [[s * vy * ce * ce * ca * ca, -s * vy * ce * ca * se * sa],
        [-s * vy * ce * ca * se * sa, s * (vy * se * se * sa * sa + vz * ce * ce)]] };
  }
  // Symmetric Jacobi eigendecomposition. Input is the dimensionless scaled
  // FIM, which avoids a rank decision dominated by ns/deg/dB unit choices.
  function eigenSymmetric(input) {
    var n = input.length, a = input.map(function (r) { return r.slice(); }), v = matrix(n, 0);
    for (var i = 0; i < n; i++) v[i][i] = 1;
    for (var iteration = 0; iteration < 100 * n * n; iteration++) {
      var largest = 0, p = 0, q = 0;
      for (i = 0; i < n; i++) for (var j = i + 1; j < n; j++) {
        if (Math.abs(a[i][j]) > largest) { largest = Math.abs(a[i][j]); p = i; q = j; }
      }
      if (largest < 1e-14) break;
      var tau = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      var t = (tau < 0 ? -1 : 1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
      var c = 1 / Math.sqrt(1 + t * t), s = t * c;
      var app = a[p][p], aqq = a[q][q], apq = a[p][q];
      a[p][p] = app - t * apq; a[q][q] = aqq + t * apq; a[p][q] = a[q][p] = 0;
      for (i = 0; i < n; i++) {
        if (i !== p && i !== q) {
          var aip = a[i][p], aiq = a[i][q];
          a[i][p] = a[p][i] = c * aip - s * aiq;
          a[i][q] = a[q][i] = s * aip + c * aiq;
        }
        var vip = v[i][p], viq = v[i][q];
        v[i][p] = c * vip - s * viq; v[i][q] = s * vip + c * viq;
      }
    }
    return { values: a.map(function (r, i) { return r[i]; }), vectors: v };
  }
  function invertEstimable(fim, indices) {
    var n = indices.length;
    var d = indices.map(function (i) { return Math.sqrt(Math.max(0, fim[i][i])); });
    var corr = indices.map(function (i, a) { return indices.map(function (j, b) {
      return d[a] > 0 && d[b] > 0 ? fim[i][j] / (d[a] * d[b]) : 0;
    }); });
    var eig = eigenSymmetric(corr), max = Math.max.apply(null, eig.values.concat([1]));
    var tolerance = max * 1e-11;
    var rank = eig.values.filter(function (e) { return e > tolerance; }).length;
    var estimable = d.map(function (di, i) {
      var nullProjection = 0;
      eig.values.forEach(function (e, k) { if (e <= tolerance) nullProjection += eig.vectors[i][k] * eig.vectors[i][k]; });
      return di > 0 && nullProjection < 1e-9;
    });
    var covariance = matrix(n, NaN), bounds = {};
    for (var i = 0; i < n; i++) {
      if (!estimable[i]) covariance[i][i] = Infinity;
      else for (var j = 0; j < n; j++) if (estimable[j]) {
        var sum = 0;
        eig.values.forEach(function (e, k) {
          if (e > tolerance) sum += eig.vectors[i][k] * eig.vectors[j][k] / e;
        });
        covariance[i][j] = sum / (d[i] * d[j]);
      }
      bounds[defs[indices[i]].key] = Math.sqrt(covariance[i][i]);
    }
    return { covariance: covariance, bounds: bounds, rank: rank, eigenvalues: eig.values,
      estimable: estimable, tolerance: tolerance };
  }
  function pilotMoments(tx, symbols, seed) {
    var nt = tx.count, state = (seed >>> 0) || 0x9e3779b9;
    var E = 0, CaR = 0, CaI = 0, CeR = 0, CeI = 0, AA = 0, EE = 0, AE = 0;
    var columns = [], cos = tx.positions.map(function (p) { return Math.cos(-p.phase); });
    var sin = tx.positions.map(function (p) { return Math.sin(-p.phase); });
    for (var l = 0; l < symbols; l++) {
      var sr = 0, si = 0, ar = 0, ai = 0, er = 0, ei = 0;
      var column = new Float64Array(2 * nt);
      for (var t = 0; t < nt; t++) {
        // xorshift32, stream order symbol then TX, top two bits choose QPSK.
        state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
        var q = state >>> 30, wr, wi;
        if (q === 0) { wr = cos[t]; wi = sin[t]; column[2 * t] = 1; }
        else if (q === 1) { wr = -sin[t]; wi = cos[t]; column[2 * t + 1] = 1; }
        else if (q === 2) { wr = -cos[t]; wi = -sin[t]; column[2 * t] = -1; }
        else { wr = sin[t]; wi = -cos[t]; column[2 * t + 1] = -1; }
        sr += wr; si += wi;
        ar += tx.positions[t].az * wi; ai -= tx.positions[t].az * wr;
        er += tx.positions[t].el * wi; ei -= tx.positions[t].el * wr;
      }
      columns.push(column);
      E += sr * sr + si * si; AA += ar * ar + ai * ai; EE += er * er + ei * ei;
      CaR += sr * ar + si * ai; CaI += sr * ai - si * ar;
      CeR += sr * er + si * ei; CeI += sr * ei - si * er;
      AE += ar * er + ai * ei;
    }
    // Modified Gram-Schmidt with reorthogonalization measures rank of the
    // complex pilot matrix itself. It is unrelated to seven-parameter rank.
    var basis = [];
    for (l = 0; l < columns.length; l++) {
      if (basis.length === nt) break;
      var v = columns[l].slice();
      for (var pass = 0; pass < 2; pass++) for (var b = 0; b < basis.length; b++) {
        var br = 0, bi = 0, u = basis[b];
        for (t = 0; t < nt; t++) {
          br += u[2 * t] * v[2 * t] + u[2 * t + 1] * v[2 * t + 1];
          bi += u[2 * t] * v[2 * t + 1] - u[2 * t + 1] * v[2 * t];
        }
        for (t = 0; t < nt; t++) {
          v[2 * t] -= u[2 * t] * br - u[2 * t + 1] * bi;
          v[2 * t + 1] -= u[2 * t] * bi + u[2 * t + 1] * br;
        }
      }
      var norm = Math.sqrt(v.reduce(function (s, x) { return s + x * x; }, 0));
      if (norm > 1e-10 * Math.sqrt(nt)) {
        for (t = 0; t < v.length; t++) v[t] /= norm;
        basis.push(v);
      }
    }
    return { E: E, CaR: CaR, CaI: CaI, CeR: CeR, CeI: CeI, AA: AA, EE: EE, AE: AE,
      rank: basis.length, energyFactor: E / (nt * symbols) };
  }
  var pilotCache = new Map();
  function compute(raw) {
    var p = Object.assign({}, defaults, raw || {}), errors = [], warnings = [];
    Object.keys(defaults).forEach(function (k) {
      if (typeof defaults[k] === "number") { p[k] = Number(p[k]); if (!Number.isFinite(p[k])) errors.push(k + " must be finite."); }
    });
    ["txY", "txZ", "rxY", "rxZ", "symbols", "tones"].forEach(function (k) {
      if (!Number.isSafeInteger(p[k]) || p[k] < 1) errors.push(k + " must be a positive integer.");
    });
    ["bandwidthMHz", "fcGHz", "dLambda"].forEach(function (k) { if (!(p[k] > 0)) errors.push(k + " must be positive."); });
    ["txAz", "rxAz", "txEl", "rxEl"].forEach(function (k) { if (Math.abs(p[k]) > 90) errors.push(k + " must lie in the front-hemisphere angle range [-90,90]."); });
    if (["link", "total"].indexOf(p.mode) < 0) errors.push("mode must be 'link' or 'total'.");
    var txCount = p.txY * p.txZ, rxCount = p.rxY * p.rxZ;
    if (txCount > 4096 || rxCount > 4096 || txCount * p.symbols > 200000 ||
        txCount * p.symbols * Math.min(txCount, p.symbols) > 20000000) {
      errors.push("This interactive calculation exceeds its browser work limit. Reduce TX/RX ports or symbols. This is a computation limit, not a physical identifiability restriction.");
    }
    if (!Number.isSafeInteger(p.pilotSeed) || p.pilotSeed < 0 || p.pilotSeed > 4294967295) errors.push("pilotSeed must be an integer from 0 through 4294967295.");
    if (errors.length) return { valid: false, errors: errors, warnings: [], inputs: p, bounds: {} };
    var nt = p.txY * p.txZ, nr = p.rxY * p.rxZ, L = p.symbols, K = p.tones;
    var grid = uniformGrid(K, p.bandwidthMHz * 1e6);
    grid.usefulTimeS = L / grid.spacingHz; grid.symbols = L;
    var lambda = C / (p.fcGHz * 1e9);
    if (!Number.isFinite(grid.bandwidthHz) || !(grid.spacingHz > 0) || !(lambda > 0)) {
      return { valid: false, errors: ["Frequency values exceed the numerical range of this calculation."], warnings: [], inputs: p, bounds: {} };
    }
    var tx = array(p.txY, p.txZ, p.dLambda, p.txAz, p.txEl, lambda);
    var rx = array(p.rxY, p.rxZ, p.dLambda, p.rxAz, p.rxEl, lambda);
    var cacheKey = [p.txY, p.txZ, p.dLambda, p.txAz, p.txEl, L, p.pilotSeed].join("|");
    var m = pilotCache.get(cacheKey);
    if (!m) {
      m = pilotMoments(tx, L, p.pilotSeed); pilotCache.set(cacheKey, m);
      if (pilotCache.size > 8) pilotCache.delete(pilotCache.keys().next().value);
    }
    var powerW = Math.pow(10, (p.txPowerDbm - 30) / 10);
    var noiseVarianceW = Math.pow(10, (-174 + p.nfDb - 30) / 10) * grid.spacingHz;
    var pilotAmplitude = Math.sqrt(powerW / (nt * K));
    var gainPower = Math.pow(10, -p.lossDb / 10);
    var w = nr * K * gainPower * pilotAmplitude * pilotAmplitude / noiseVarianceW;
    if (p.mode === "total") w = Math.pow(10, p.snrDb / 10) / m.E;
    var gamma = w * m.E;
    if (!(gamma > 0) || !Number.isFinite(gamma)) return { valid: false, errors: ["Invalid signal-energy scale or a complete pilot null."], warnings: [], inputs: p, bounds: {} };
    var J = matrix(7, 0), meanOmega = 2 * Math.PI * grid.meanFrequencyHz * 1e-9;
    var secondOmega = 4 * Math.PI * Math.PI * grid.secondMomentHz2 * 1e-18;
    function put(i, j, value) { J[i][j] = J[j][i] = 2 * w * value; }
    put(0, 0, secondOmega * m.E);
    put(0, 3, -meanOmega * m.CaI); put(0, 4, -meanOmega * m.CeI);
    put(0, 6, -meanOmega * m.E);
    put(3, 3, m.AA); put(4, 4, m.EE); put(3, 4, m.AE);
    put(3, 5, -ALOSS * m.CaR); put(4, 5, -ALOSS * m.CeR);
    put(3, 6, m.CaI); put(4, 6, m.CeI);
    put(5, 5, ALOSS * ALOSS * m.E); put(6, 6, m.E);
    for (var a = 0; a < 2; a++) for (var b = 0; b < 2; b++) J[1 + a][1 + b] = 2 * gamma * rx.derivativeMoment[a][b];
    var inverse = invertEstimable(J, [0, 1, 2, 3, 4, 5, 6]);
    var correlation = J.map(function (row, i) { return row.map(function (value, j) {
      return J[i][i] > 0 && J[j][j] > 0 ? Math.max(-1, Math.min(1, value / Math.sqrt(J[i][i] * J[j][j]))) : NaN;
    }); });
    if (inverse.rank < 7) warnings.push("The full Fisher matrix is singular. Infinite bounds mark individual parameters that cannot be estimated locally with the others unknown.");
    if (p.dLambda > 0.5 + 1e-12) warnings.push("Spacing above half a wavelength can create global grating ambiguities even when a local bound is finite.");
    if (L < nt) warnings.push("The coded pilot has fewer symbols than TX ports. A structured path can still be identifiable. The full channel matrix cannot be recovered without additional structure.");
    if (gamma < 10) warnings.push("At low aggregate SNR, actual estimation errors can greatly exceed the local CRB.");
    var parameters = defs.map(function (d, i) { return Object.assign({}, d, {
      std: inverse.bounds[d.key], bound: inverse.bounds[d.key], variance: inverse.covariance[i][i], identifiable: inverse.estimable[i]
    }); });
    delete tx.positions; delete rx.positions;
    return { valid: true, errors: [], warnings: warnings, inputs: p, grid: grid,
      geometry: { wavelengthM: lambda, spacingM: p.dLambda * lambda, dLambda: p.dLambda, tx: tx, rx: rx },
      signal: { gamma: gamma, snrTotalDb: 10 * Math.log10(gamma), rhoPerTone: gamma / (nr * K * L),
        snrPerToneDb: 10 * Math.log10(gamma / (nr * K * L)), powerW: powerW,
        noiseVarianceW: noiseVarianceW, noisePerToneDbm: 10 * Math.log10(noiseVarianceW) + 30,
        noiseDbm: 10 * Math.log10(noiseVarianceW * K) + 30, pilotAmplitudeSqrtW: pilotAmplitude,
        txPorts: nt, rxPorts: nr, complexObservations: nr * K * L, energyMode: p.mode },
      pilot: { seed: p.pilotSeed >>> 0, rank: m.rank, rankUpperBound: Math.min(nt, L),
        energyFactor: m.energyFactor, construction: "One illustrative xorshift32 QPSK X[T,S], reused on every tone; not the saved NumPy X." },
      fim: J, correlation: correlation, covariance: inverse.covariance, parameters: parameters,
      bounds: inverse.bounds,
      rank: inverse.rank, eigenvaluesScaled: inverse.eigenvalues,
      rangeEquivalentM: C * inverse.bounds.tauNs * 1e-9,
      notes: ["Apparent delay equals geometric delay plus receiver clock offset. A single path alone does not separate those two contributions.",
        "Static coherent isolated path, frequency-flat unknown complex gain, calibrated array patterns and orientation.",
        "CRBs are local lower bounds and do not imply detection, unique global ambiguity resolution, or a full RT-scene bound."] };
  }
  return Object.freeze({ defaults: defaults, compute: compute,
    constants: Object.freeze({ speedOfLight: C, degreesToRadians: DEG, thermalDbmHz: -174 }) });
});
