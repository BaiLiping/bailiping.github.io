(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.ASRModel = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const EPS = 1e-12;

  function clamp(value, lo, hi) {
    return Math.max(lo, Math.min(hi, value));
  }

  function linspace(a, b, count) {
    return Array.from({ length: count }, (_, i) => a + (b - a) * i / Math.max(1, count - 1));
  }

  function matMul(A, B) {
    return A.map(row => B[0].map((_, j) => row.reduce((sum, value, k) => sum + value * B[k][j], 0)));
  }

  function transpose(A) {
    return A[0].map((_, j) => A.map(row => row[j]));
  }

  function determinant2(A) {
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];
  }

  function frobeniusOrthogonalityError(A) {
    const gram = matMul(transpose(A), A);
    return Math.hypot(gram[0][0] - 1, gram[0][1], gram[1][0], gram[1][1] - 1);
  }

  function rotation(angle) {
    const c = Math.cos(angle), s = Math.sin(angle);
    return [[c, -s], [s, c]];
  }

  function rotationExperiment(incrementDeg = 12, steps = 6, startDeg = -35) {
    const delta = incrementDeg * Math.PI / 180;
    const exactStep = rotation(delta);
    const eulerStep = [[1, -delta], [delta, 1]];
    let exact = rotation(startDeg * Math.PI / 180);
    let euler = rotation(startDeg * Math.PI / 180);
    const exactPath = [startDeg * Math.PI / 180];
    const eulerPath = [[euler[0][0], euler[1][0]]];
    for (let i = 0; i < steps; i += 1) {
      exact = matMul(exact, exactStep);
      euler = matMul(euler, eulerStep);
      exactPath.push(Math.atan2(exact[1][0], exact[0][0]));
      eulerPath.push([euler[0][0], euler[1][0]]);
    }
    return {
      incrementDeg,
      steps,
      startDeg,
      exact,
      euler,
      exactPath,
      eulerPath,
      exactDeterminant: determinant2(exact),
      eulerDeterminant: determinant2(euler),
      exactOrthogonality: frobeniusOrthogonalityError(exact),
      eulerOrthogonality: frobeniusOrthogonalityError(euler),
      exactAngleDeg: Math.atan2(exact[1][0], exact[0][0]) * 180 / Math.PI,
      eulerAngleDeg: Math.atan2(euler[1][0], euler[0][0]) * 180 / Math.PI,
      eulerAxisLength: Math.hypot(euler[0][0], euler[1][0])
    };
  }

  const BASE_CONTROL_POINTS = [
    [0.00, 0.34], [0.10, 0.18], [0.20, 0.46], [0.30, 0.61],
    [0.40, 0.43], [0.50, 0.67], [0.60, 0.39], [0.70, 0.75],
    [0.80, 0.54], [0.90, 0.72], [1.00, 0.58]
  ];

  function openUniformKnots(n, degree) {
    const knotCount = n + degree + 1;
    const inner = n - degree - 1;
    const knots = [];
    for (let i = 0; i < knotCount; i += 1) {
      if (i <= degree) knots.push(0);
      else if (i >= n) knots.push(1);
      else knots.push((i - degree) / (inner + 1));
    }
    return knots;
  }

  function basisWeights(t, n, degree) {
    const u = clamp(t, 0, 1);
    const knots = openUniformKnots(n, degree);
    let weights = Array.from({ length: n }, (_, i) =>
      (knots[i] <= u && u < knots[i + 1]) || (u === 1 && i === n - 1) ? 1 : 0
    );
    for (let p = 1; p <= degree; p += 1) {
      weights = weights.map((_, i) => {
        const leftDen = knots[i + p] - knots[i];
        const rightDen = knots[i + p + 1] - knots[i + 1];
        const left = leftDen > EPS ? (u - knots[i]) / leftDen * weights[i] : 0;
        const right = rightDen > EPS && i + 1 < n ? (knots[i + p + 1] - u) / rightDen * weights[i + 1] : 0;
        return left + right;
      });
    }
    return weights.map(value => Math.abs(value) < EPS ? 0 : value);
  }

  function splinePoint(points, degree, t) {
    const weights = basisWeights(t, points.length, degree);
    return {
      point: weights.reduce((sum, weight, i) => [sum[0] + weight * points[i][0], sum[1] + weight * points[i][1]], [0, 0]),
      weights,
      active: weights.map((weight, i) => ({ i, weight })).filter(item => item.weight > 1e-8)
    };
  }

  function splineExperiment(options = {}) {
    const degree = options.degree === 1 ? 1 : 3;
    const query = clamp(Number(options.query ?? 0.54), 0, 1);
    const selected = clamp(Math.round(Number(options.selected ?? 5)), 0, BASE_CONTROL_POINTS.length - 1);
    const shift = clamp(Number(options.shift ?? 0.18), -0.35, 0.35);
    const base = BASE_CONTROL_POINTS.map(point => point.slice());
    const points = base.map(point => point.slice());
    points[selected][1] = clamp(points[selected][1] + shift, 0.04, 0.96);
    const times = linspace(0, 1, 161);
    const reference = times.map(t => splinePoint(base, degree, t).point);
    const curve = times.map(t => splinePoint(points, degree, t).point);
    const queryResult = splinePoint(points, degree, query);
    const h = 1e-4;
    const before = splinePoint(points, degree, clamp(query - h, 0, 1)).point;
    const after = splinePoint(points, degree, clamp(query + h, 0, 1)).point;
    const span = Math.max(h, clamp(query + h, 0, 1) - clamp(query - h, 0, 1));
    const velocity = [(after[0] - before[0]) / span, (after[1] - before[1]) / span];
    const influence = times.filter((_, i) => Math.hypot(curve[i][0] - reference[i][0], curve[i][1] - reference[i][1]) > 1e-7);
    return {
      degree,
      query,
      selected,
      shift,
      points,
      base,
      times,
      curve,
      reference,
      queryPoint: queryResult.point,
      weights: queryResult.weights,
      active: queryResult.active,
      speed: Math.hypot(...velocity),
      influenceStart: influence.length ? influence[0] : null,
      influenceEnd: influence.length ? influence[influence.length - 1] : null
    };
  }

  function zeros(rows, cols) {
    return Array.from({ length: rows }, () => Array(cols).fill(0));
  }

  function solveLinear(A, b) {
    const n = A.length;
    const M = A.map((row, i) => [...row, b[i]]);
    for (let col = 0; col < n; col += 1) {
      let pivot = col;
      for (let row = col + 1; row < n; row += 1) if (Math.abs(M[row][col]) > Math.abs(M[pivot][col])) pivot = row;
      if (Math.abs(M[pivot][col]) < EPS) throw new Error('Singular linear system');
      [M[col], M[pivot]] = [M[pivot], M[col]];
      const scale = M[col][col];
      for (let j = col; j <= n; j += 1) M[col][j] /= scale;
      for (let row = 0; row < n; row += 1) {
        if (row === col) continue;
        const factor = M[row][col];
        for (let j = col; j <= n; j += 1) M[row][j] -= factor * M[col][j];
      }
    }
    return M.map(row => row[n]);
  }

  function inverse(A) {
    const n = A.length;
    const result = zeros(n, n);
    for (let j = 0; j < n; j += 1) {
      const e = Array(n).fill(0); e[j] = 1;
      const column = solveLinear(A, e);
      for (let i = 0; i < n; i += 1) result[i][j] = column[i];
    }
    return result;
  }

  function interpolationRow(time, count) {
    const t = clamp(time, 0, count - 1);
    const left = Math.min(count - 2, Math.floor(t));
    const alpha = t - left;
    const row = Array(count).fill(0);
    row[left] = 1 - alpha;
    row[left + 1] = alpha;
    return row;
  }

  function quad(row, matrix) {
    let value = 0;
    for (let i = 0; i < row.length; i += 1) for (let j = 0; j < row.length; j += 1) value += row[i] * matrix[i][j] * row[j];
    return value;
  }

  const GP_MEASUREMENTS = [
    { t: 0.18, z: 0.18 }, { t: 0.92, z: 0.76 }, { t: 1.72, z: 0.38 },
    { t: 2.58, z: 1.42 }, { t: 3.37, z: 1.11 }, { t: 4.44, z: 2.08 },
    { t: 5.36, z: 1.66 }, { t: 6.68, z: 2.54 }
  ];

  // Finite random-walk control model with deterministic linear interpolation.
  // quad(row, covariance) is control-interpolation variance, not Wiener-bridge variance.
  function gpExperiment(options = {}) {
    const count = 8;
    const processVariance = clamp(Number(options.processVariance ?? 0.18), 0.02, 1.2);
    const measurementSigma = clamp(Number(options.measurementSigma ?? 0.22), 0.06, 0.9);
    const query = clamp(Number(options.query ?? 3.65), 0, count - 1);
    const information = zeros(count, count);
    const eta = Array(count).fill(0);
    const addFactor = (row, target, variance) => {
      const weight = 1 / variance;
      for (let i = 0; i < count; i += 1) {
        eta[i] += weight * row[i] * target;
        for (let j = 0; j < count; j += 1) information[i][j] += weight * row[i] * row[j];
      }
    };
    const prior = Array(count).fill(0); prior[0] = 1;
    addFactor(prior, 0, 0.04 ** 2);
    for (let i = 1; i < count; i += 1) {
      const difference = Array(count).fill(0); difference[i - 1] = -1; difference[i] = 1;
      addFactor(difference, 0, processVariance);
    }
    for (const measurement of GP_MEASUREMENTS) addFactor(interpolationRow(measurement.t, count), measurement.z, measurementSigma ** 2);
    const mean = solveLinear(information, eta);
    const covariance = inverse(information);
    const times = linspace(0, count - 1, 181);
    const curve = times.map(t => {
      const row = interpolationRow(t, count);
      return {
        t,
        mean: row.reduce((sum, value, i) => sum + value * mean[i], 0),
        sigma: Math.sqrt(Math.max(0, quad(row, covariance)))
      };
    });
    const queryRow = interpolationRow(query, count);
    const queryMean = queryRow.reduce((sum, value, i) => sum + value * mean[i], 0);
    const querySigma = Math.sqrt(Math.max(0, quad(queryRow, covariance)));
    let nonzeros = 0;
    for (const row of information) for (const value of row) if (Math.abs(value) > 1e-10) nonzeros += 1;
    const roughness = mean.slice(1).reduce((sum, value, i) => sum + (value - mean[i]) ** 2, 0);
    return {
      count,
      processVariance,
      measurementSigma,
      query,
      measurements: GP_MEASUREMENTS.map(item => ({ ...item })),
      information,
      covariance,
      mean,
      curve,
      queryMean,
      querySigma,
      nonzeros,
      density: nonzeros / (count * count),
      roughness
    };
  }

  return {
    clamp,
    rotation,
    determinant2,
    frobeniusOrthogonalityError,
    rotationExperiment,
    basisWeights,
    splinePoint,
    splineExperiment,
    solveLinear,
    inverse,
    interpolationRow,
    gpExperiment,
    BASE_CONTROL_POINTS: BASE_CONTROL_POINTS.map(point => point.slice()),
    GP_MEASUREMENTS: GP_MEASUREMENTS.map(item => ({ ...item }))
  };
});
