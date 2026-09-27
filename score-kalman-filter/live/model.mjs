/**
 * Educational one-dimensional quartic example for The Score Kalman Filter.
 *
 * Quadrature creates consistent synthetic moments and reference density plots.
 * It is not part of score fitting, Stein closure, or the coefficient update.
 * This module does not implement time propagation or the paper's truncated
 * posterior moment-recovery procedure.
 */

export const DEFAULTS = Object.freeze({ a: 0.35, b: -1.3, c: 0.25, y: 1.1, R: 0.45 });

export function quarticParameters({ a, b, c }) {
  if (!(a > 0) || ![a, b, c].every(Number.isFinite)) throw new Error('The quartic coefficient a must be positive.');
  return [-c, b, 0, a];
}

export function logKernel(lambda, x) {
  return -x * (lambda[0] + x * (lambda[1] + x * (lambda[2] + x * lambda[3])));
}

export function score(lambda, x) {
  return -(lambda[0] + x * (2 * lambda[1] + x * (3 * lambda[2] + 4 * lambda[3] * x)));
}

/** Composite Simpson quadrature, with log scaling and compensated summation. */
export function referenceQuadrature(lambda, maxOrder = 8, intervals = 4096) {
  if (lambda.length !== 4 || !lambda.every(Number.isFinite)) throw new Error('Expected four finite parameters.');
  const normalizable = lambda[3] > 0 || (lambda[3] === 0 && lambda[2] === 0 && lambda[1] > 0);
  if (!normalizable) throw new Error('The selected polynomial does not define an integrable density.');
  if (intervals < 2 || intervals % 2 !== 0) throw new Error('Simpson quadrature requires an even interval count.');

  let radius = 4;
  let logShift = -Infinity;
  for (let attempt = 0; attempt < 12; attempt += 1) {
    logShift = -Infinity;
    for (let i = 0; i <= 256; i += 1) {
      logShift = Math.max(logShift, logKernel(lambda, -radius + 2 * radius * i / 256));
    }
    const tail = Math.max(logKernel(lambda, -radius), logKernel(lambda, radius));
    const inwardTails = score(lambda, -radius) > 0 && score(lambda, radius) < 0;
    if (inwardTails && tail - logShift < -48) break;
    radius *= 1.5;
    if (attempt === 11) throw new Error('Could not bracket the reference density.');
  }

  const sums = Array(maxOrder + 1).fill(0);
  const correction = Array(maxOrder + 1).fill(0);
  const step = 2 * radius / intervals;
  for (let i = 0; i <= intervals; i += 1) {
    const x = -radius + i * step;
    const simpsonWeight = i === 0 || i === intervals ? 1 : i % 2 === 0 ? 2 : 4;
    let term = simpsonWeight * Math.exp(logKernel(lambda, x) - logShift);
    for (let order = 0; order <= maxOrder; order += 1) {
      const adjusted = term - correction[order];
      const sum = sums[order] + adjusted;
      correction[order] = (sum - sums[order]) - adjusted;
      sums[order] = sum;
      term *= x;
    }
  }
  const moments = sums.map(value => value / sums[0]);
  // Exact symmetry prevents floating-point cancellation from obscuring odd moments.
  if (lambda[0] === 0 && lambda[2] === 0) {
    for (let k = 1; k <= maxOrder; k += 2) moments[k] = 0;
  }
  return { lambda: [...lambda], moments, radius, logShift, normScaled: sums[0] * step / 3, intervals };
}

export function referenceDensity(reference, x) {
  return Math.exp(logKernel(reference.lambda, x) - reference.logShift) / reference.normScaled;
}

/** Eqs. (8)-(9), specialized to phi_j(x) = x^j for j=1,...,4. */
export function scoreSystem(moments) {
  if (moments.length < 7) throw new Error('The quartic score fit requires moments m0 through m6.');
  const A = Array.from({ length: 4 }, (_, row) => Array.from({ length: 4 }, (_, col) => {
    const j = row + 1;
    const k = col + 1;
    return j * k * moments[j + k - 2];
  }));
  const rhs = Array.from({ length: 4 }, (_, row) => {
    const j = row + 1;
    return j === 1 ? 0 : j * (j - 1) * moments[j - 2];
  });
  return { A, rhs };
}

/** Scaled partial-pivot elimination. Inputs are copied and never mutated. */
export function solveLinear(A, rhs) {
  const n = rhs.length;
  const augmented = A.map((row, i) => [...row, rhs[i]]);
  const scales = A.map(row => Math.max(...row.map(Math.abs)));
  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < n; row += 1) {
      if (Math.abs(augmented[row][col]) / scales[row] > Math.abs(augmented[pivot][col]) / scales[pivot]) pivot = row;
    }
    if (!scales[pivot] || Math.abs(augmented[pivot][col]) <= 1e-13 * scales[pivot]) throw new Error('The moment system is numerically singular.');
    [augmented[col], augmented[pivot]] = [augmented[pivot], augmented[col]];
    [scales[col], scales[pivot]] = [scales[pivot], scales[col]];
    for (let row = col + 1; row < n; row += 1) {
      const factor = augmented[row][col] / augmented[col][col];
      augmented[row][col] = 0;
      for (let k = col + 1; k <= n; k += 1) augmented[row][k] -= factor * augmented[col][k];
    }
  }
  const solution = Array(n).fill(0);
  for (let row = n - 1; row >= 0; row -= 1) {
    let value = augmented[row][n];
    for (let col = row + 1; col < n; col += 1) value -= augmented[row][col] * solution[col];
    solution[row] = value / augmented[row][row];
  }
  return solution;
}

export function fitScore(moments) {
  const { A, rhs } = scoreSystem(moments);
  // Symmetric diagonal equilibration improves the scale of the four monomials.
  const diagonal = A.map((row, i) => Math.sqrt(row[i]));
  const scaledA = A.map((row, i) => row.map((value, j) => value / (diagonal[i] * diagonal[j])));
  const scaledRhs = rhs.map((value, i) => value / diagonal[i]);
  const lambda = solveLinear(scaledA, scaledRhs).map((value, i) => value / diagonal[i]);
  const residual = A.map((row, i) => row.reduce((sum, value, j) => sum + value * lambda[j], 0) - rhs[i]);
  return { A, rhs, lambda, residual, maxResidual: Math.max(...residual.map(Math.abs)) };
}

/** Eq. (12): use beta=q-3 to compute m_q from already known lower moments. */
export function steinClosure(knownMoments, lambda, maxOrder = 8) {
  if (knownMoments.length < 7) throw new Error('This example starts with known moments m0 through m6.');
  if (!(lambda[3] > 0)) throw new Error('The quartic recurrence requires a positive lambda4.');
  const moments = knownMoments.slice(0, 7);
  const steps = [];
  for (let order = 7; order <= maxOrder; order += 1) {
    const beta = order - 3;
    const rhs = beta * moments[beta - 1];
    const knownTerms = lambda[0] * moments[beta] + 2 * lambda[1] * moments[beta + 1] + 3 * lambda[2] * moments[beta + 2];
    const value = (rhs - knownTerms) / (4 * lambda[3]);
    moments[order] = value;
    const residual = knownTerms + 4 * lambda[3] * value - rhs;
    steps.push({ order, beta, value, rhs, knownTerms, residual });
  }
  return { moments, steps };
}

/** Eqs. (13)-(14) for y=x+v, v~N(0,R). Exactly one likelihood factor. */
export function gaussianObservationUpdate(priorLambda, y, R) {
  if (!(R > 0) || !Number.isFinite(y)) throw new Error('Measurement variance R must be positive.');
  return [priorLambda[0] - y / R, priorLambda[1] + 1 / (2 * R), priorLambda[2], priorLambda[3]];
}
