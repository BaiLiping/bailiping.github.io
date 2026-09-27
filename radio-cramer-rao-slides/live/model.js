/*
 * Single isolated-path, deterministic complex-Gaussian CRB.
 *
 * Y_k = alpha exp(-j 2 pi f_k tau) a_R a_T^H X_k + N_k,
 * [a(u)]_n = exp(j 2 pi p_n.u / lambda), with unit-modulus entries.
 * X_k X_k^H = (L P_total / (N_T K)) I, hence L >= N_T.
 * E[|N_k(r,l)|^2] = sigma_tone^2: this is COMPLEX noise variance.
 * J_ij = 2 Re{(d mu / d theta_i)^H (d mu / d theta_j)}/sigma_tone^2.
 *
 * Centering the array coordinates and the subcarrier frequencies removes
 * the angle/phase and delay/phase cross terms without claiming known phase.
 * The phase parameter refers to these array centers and the center frequency.
 * Summing the derivatives of the balanced pilot matrix removes TX/RX cross
 * terms. Delay, RX angles, TX angles, attenuation, and phase form blocks.
 *
 * Gamma = ||mu||^2 / sigma_tone^2 = N_R K L rho.
 * rho is the SNR per RX element / subcarrier / symbol, averaged over the
 * balanced pilot schedule and INCLUDING the sum of TX-port pilot energies.
 * There is no extra N_T multiplier at fixed total transmitted power.
 *
 * Frequency grid: Delta_f = B/K, span = (K-1)Delta_f,
 * beta^2 = Delta_f^2 (K^2-1)/12. Carrier frequency is not delay bandwidth:
 * the unknown complex path phase absorbs the center-frequency delay phase.
 *
 * The seven coordinates, and therefore the displayed FIM units, are
 * [delay ns, RX azimuth deg, RX elevation deg, TX azimuth deg,
 *  TX elevation deg, attenuation dB, phase rad]. Bounds are marginal
 * standard-deviation lower bounds, not confidence intervals or resolutions.
 *
 * Source derivations and assumptions are documented in the surrounding deck.
 * Core complex-Gaussian FIM: https://arxiv.org/html/2301.10689v1
 * Array/delay bounds and pilot dependence: https://arxiv.org/html/1702.01605v2
 * Identifiability and nuisance parameters: https://arxiv.org/html/2002.04481v3
 */
(function (root, factory) {
  "use strict";
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.RadioCRB = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  var C = 299792458;
  var DEG = Math.PI / 180;
  var LOSS_DERIVATIVE = Math.log(10) / 20;
  var THERMAL_DBM_HZ = -173.975;
  var defaults = Object.freeze({
    txY: 8, txZ: 8, rxY: 4, rxZ: 4, symbols: 64, tones: 3300,
    gridMode: "bandwidth", bandwidthMHz: 400, spacingKHz: 120,
    fcGHz: 27.2, txAz: 20, txEl: 15, rxAz: 25, rxEl: 10,
    spacingMode: "wavelength", dLambda: 0.5, spacingMm: 5.5,
    mode: "total", snrDb: 30, lossDb: 120, txPowerDbm: 35, nfDb: 9
  });
  var definitions = [
    { key: "tauNs", label: "Delay", unit: "ns" },
    { key: "rxAzDeg", label: "AoA azimuth", unit: "deg" },
    { key: "rxElDeg", label: "AoA elevation", unit: "deg" },
    { key: "txAzDeg", label: "AoD azimuth", unit: "deg" },
    { key: "txElDeg", label: "AoD elevation", unit: "deg" },
    { key: "lossDb", label: "Path attenuation", unit: "dB" },
    { key: "phaseRad", label: "Path phase", unit: "rad" }
  ];

  function matrix(n, fill) {
    return Array.from({ length: n }, function () { return Array(n).fill(fill); });
  }

  function cleanTrig(value) {
    // Numerical cos(pi/2) should not create an apparently finite CRB at a
    // direction with a mathematically zero first-order angular derivative.
    return Math.abs(value) < 1e-15 ? 0 : value;
  }

  function normalize(raw) {
    var p = Object.assign({}, defaults, raw || {});
    var errors = [];
    Object.keys(defaults).forEach(function (key) {
      if (typeof defaults[key] === "number") {
        p[key] = Number(p[key]);
        if (!Number.isFinite(p[key])) errors.push(key + " must be a finite number.");
      }
    });
    ["txY", "txZ", "rxY", "rxZ", "symbols", "tones"].forEach(function (key) {
      if (!Number.isSafeInteger(p[key]) || p[key] < 1) {
        errors.push(key + " must be a positive integer.");
      }
    });
    if (["bandwidth", "spacing"].indexOf(p.gridMode) < 0) {
      errors.push("gridMode must be 'bandwidth' or 'spacing'.");
    }
    if (["wavelength", "physical"].indexOf(p.spacingMode) < 0) {
      errors.push("spacingMode must be 'wavelength' or 'physical'.");
    }
    if (["total", "perTone", "link"].indexOf(p.mode) < 0) {
      errors.push("mode must be 'total', 'perTone', or 'link'.");
    }
    ["fcGHz", "bandwidthMHz", "spacingKHz", "dLambda", "spacingMm"].forEach(function (key) {
      if (!(p[key] > 0)) errors.push(key + " must be positive.");
    });
    ["txEl", "rxEl"].forEach(function (key) {
      if (p[key] < -90 || p[key] > 90) errors.push(key + " must be between -90 and 90 degrees.");
    });
    ["txAz", "rxAz"].forEach(function (key) {
      if (p[key] < -180 || p[key] > 180) errors.push(key + " must be between -180 and 180 degrees.");
    });
    if (p.symbols < p.txY * p.txZ) {
      errors.push("Balanced time-orthogonal TX pilots require L >= N_T. Increase symbols or reduce TX ports.");
    }
    return { inputs: p, errors: errors };
  }

  function emptyResult(p, errors) {
    var bounds = {};
    var parameters = definitions.map(function (d) {
      bounds[d.key] = NaN;
      return Object.assign({}, d, { bound: NaN, std: NaN, variance: NaN, identifiable: false });
    });
    return {
      valid: false, errors: errors, warnings: [], inputs: p,
      grid: null, signal: null, geometry: null,
      fim: matrix(7, NaN), correlation: matrix(7, NaN), covariance: matrix(7, NaN),
      parameters: parameters, bounds: bounds
    };
  }

  function arrayGeometry(ny, nz, spacingM, azDeg, elDeg) {
    var az = azDeg * DEG;
    var el = elDeg * DEG;
    var sa = cleanTrig(Math.sin(az));
    var ca = cleanTrig(Math.cos(az));
    var se = cleanTrig(Math.sin(el));
    var ce = cleanTrig(Math.cos(el));
    var varY = spacingM * spacingM * (ny * ny - 1) / 12;
    var varZ = spacingM * spacingM * (nz * nz - 1) / 12;
    // Position-projected direction derivatives, for angles in radians.
    var azY = ce * ca;
    var elY = -se * sa;
    var elZ = ce;
    var a = varY * azY * azY;
    var b = varY * azY * elY;
    var c = varY * elY * elY + varZ * elZ * elZ;
    var determinant = varY * varZ * azY * azY * elZ * elZ;
    var rank = determinant > 0 ? 2 : (a > 0 || c > 0 ? 1 : 0);
    return {
      ny: ny, nz: nz, count: ny * nz, azDeg: azDeg, elDeg: elDeg,
      varY: varY, varZ: varZ,
      apertureYM: (ny - 1) * spacingM,
      apertureZM: (nz - 1) * spacingM,
      direction: [ce * ca, ce * sa, se],
      moment: [[a, b], [b, c]], determinant: determinant, rank: rank,
      derivatives: { azY: azY, elY: elY, elZ: elZ }
    };
  }

  function putAngleBlock(fim, covariance, start, array, factor) {
    var a = array.moment[0][0];
    var b = array.moment[0][1];
    var c = array.moment[1][1];
    fim[start][start] = factor * a;
    fim[start][start + 1] = fim[start + 1][start] = factor * b;
    fim[start + 1][start + 1] = factor * c;
    if (array.rank === 2) {
      // Use the analytic Gram determinant to avoid cancellation in a*c-b*b.
      // Invert the whole angular block: 1/J_ii is generally conditional.
      var denominator = factor * array.determinant;
      covariance[start][start] = c / denominator;
      covariance[start + 1][start + 1] = a / denominator;
      covariance[start][start + 1] = covariance[start + 1][start] = -b / denominator;
    } else if (a > 0 && c === 0) {
      covariance[start][start] = 1 / (factor * a);
      covariance[start + 1][start + 1] = Infinity;
    } else if (c > 0 && a === 0) {
      covariance[start][start] = Infinity;
      covariance[start + 1][start + 1] = 1 / (factor * c);
    } else {
      // A rank-one line array observes one angular combination. When both
      // coordinates contribute to it, neither has a finite marginal CRB.
      // A Moore-Penrose inverse would incorrectly suggest finite variances.
      covariance[start][start] = Infinity;
      covariance[start + 1][start + 1] = Infinity;
    }
  }

  function compute(raw) {
    var normalized = normalize(raw);
    var p = normalized.inputs;
    if (normalized.errors.length) return emptyResult(p, normalized.errors);
    var warnings = [];
    var K = p.tones;
    var L = p.symbols;
    var Nr = p.rxY * p.rxZ;
    var Nt = p.txY * p.txZ;
    var spacingHz = p.gridMode === "spacing" ? p.spacingKHz * 1e3 : p.bandwidthMHz * 1e6 / K;
    var bandwidthHz = K * spacingHz;
    var spanHz = (K - 1) * spacingHz;
    var betaHz = spacingHz * Math.sqrt((K * K - 1) / 12);
    // Normalize both coupled grid fields for UI readouts and exported state.
    p.bandwidthMHz = bandwidthHz / 1e6;
    p.spacingKHz = spacingHz / 1e3;
    var grid = {
      bandwidthHz: bandwidthHz, spacingHz: spacingHz, spanHz: spanHz,
      betaHz: betaHz, usefulTimeS: L / spacingHz,
      symbols: L, tones: K, observations: Nr * K * L
    };
    var noiseDbm = THERMAL_DBM_HZ + p.nfDb + 10 * Math.log10(bandwidthHz);
    var gamma;
    var rho;
    if (p.mode === "total") {
      gamma = Math.pow(10, p.snrDb / 10);
      rho = gamma / (Nr * K * L);
    } else if (p.mode === "perTone") {
      rho = Math.pow(10, p.snrDb / 10);
      gamma = Nr * K * L * rho;
    } else {
      rho = Math.pow(10, (p.txPowerDbm - p.lossDb - noiseDbm) / 10);
      gamma = Nr * K * L * rho;
    }
    if (!Number.isFinite(gamma) || !(gamma > 0) || !Number.isFinite(bandwidthHz)) {
      return emptyResult(p, ["The requested energy or frequency scale is outside the numerical range."]);
    }
    var signal = {
      gamma: gamma, rhoPerTone: rho,
      snrTotalDb: 10 * Math.log10(gamma), snrPerToneDb: 10 * Math.log10(rho),
      noiseDbm: noiseDbm, noisePerToneDbm: noiseDbm - 10 * Math.log10(K),
      receivedPowerPerRxDbm: p.txPowerDbm - p.lossDb,
      txPorts: Nt, rxPorts: Nr, complexObservations: Nr * K * L,
      energyMode: p.mode
    };
    var wavelengthM = C / (p.fcGHz * 1e9);
    var spacingM = p.spacingMode === "physical" ? p.spacingMm * 1e-3 : p.dLambda * wavelengthM;
    var dLambda = spacingM / wavelengthM;
    p.dLambda = dLambda;
    p.spacingMm = spacingM * 1e3;
    var tx = arrayGeometry(p.txY, p.txZ, spacingM, p.txAz, p.txEl);
    var rx = arrayGeometry(p.rxY, p.rxZ, spacingM, p.rxAz, p.rxEl);
    var geometry = { wavelengthM: wavelengthM, spacingM: spacingM, dLambda: dLambda, tx: tx, rx: rx };

    var fim = matrix(7, 0);
    var covariance = matrix(7, 0);
    var angleFactor = 8 * Math.PI * Math.PI * gamma / (wavelengthM * wavelengthM) * DEG * DEG;
    fim[0][0] = 8 * Math.PI * Math.PI * gamma * betaHz * betaHz * 1e-18;
    fim[5][5] = 2 * gamma * LOSS_DERIVATIVE * LOSS_DERIVATIVE;
    fim[6][6] = 2 * gamma;
    [0, 5, 6].forEach(function (i) {
      covariance[i][i] = fim[i][i] > 0 ? 1 / fim[i][i] : Infinity;
    });
    putAngleBlock(fim, covariance, 1, rx, angleFactor);
    putAngleBlock(fim, covariance, 3, tx, angleFactor);
    // Covariances with an unidentifiable coordinate are undefined, rather
    // than a purported zero that could be mistaken for a finite inverse.
    for (var i = 0; i < 7; i++) {
      if (!Number.isFinite(covariance[i][i])) {
        for (var j = 0; j < 7; j++) {
          if (i !== j) covariance[i][j] = covariance[j][i] = NaN;
        }
      }
    }
    var correlation = fim.map(function (row, i) {
      return row.map(function (value, j) {
        var scale = Math.sqrt(fim[i][i] * fim[j][j]);
        return scale > 0 ? Math.max(-1, Math.min(1, value / scale)) : NaN;
      });
    });
    var bounds = {};
    var parameters = definitions.map(function (definition, i) {
      var variance = covariance[i][i];
      var std = Math.sqrt(variance);
      bounds[definition.key] = std;
      return Object.assign({}, definition, {
        bound: std, std: std, variance: variance, identifiable: Number.isFinite(variance)
      });
    });
    if (K === 1) warnings.push("One tone with unknown path phase does not identify delay.");
    if (rx.rank < 2) warnings.push("The RX angular block is singular. Infinite entries mark unidentifiable angle coordinates.");
    if (tx.rank < 2) warnings.push("The TX angular block is singular. Infinite entries mark unidentifiable angle coordinates.");
    if (dLambda > 0.5 + 1e-12) warnings.push("Spacing exceeds half a wavelength. Grating ambiguities may exist despite a small local CRB.");
    if (gamma < 10) warnings.push("At low aggregate SNR, estimator threshold effects can make actual errors much larger than this local bound.");
    if (Math.abs(p.rxEl) > 80 || Math.abs(p.txEl) > 80 || Math.abs(Math.cos(p.rxAz * DEG)) < 0.17 || Math.abs(Math.cos(p.txAz * DEG)) < 0.17) {
      warnings.push("Near grazing or an angle-coordinate singularity, first-order angular information is weak or singular.");
    }
    return {
      valid: true, errors: [], warnings: warnings, inputs: p,
      grid: grid, signal: signal, geometry: geometry,
      fim: fim, correlation: correlation, covariance: covariance,
      parameters: parameters, bounds: bounds,
      rangeEquivalentM: C * bounds.tauNs * 1e-9
    };
  }

  function sweep(raw, key, values) {
    return values.map(function (value) {
      var changed = Object.assign({}, raw || {});
      changed[key] = value;
      return { value: value, result: compute(changed) };
    });
  }

  return Object.freeze({ defaults: defaults, compute: compute, sweep: sweep,
    constants: Object.freeze({ speedOfLight: C, degreesToRadians: DEG, thermalDbmHz: THERMAL_DBM_HZ }) });
});
