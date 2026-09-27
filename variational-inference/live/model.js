((root, factory) => {
  'use strict';
  const api = Object.freeze(factory());
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (root) root.VariationalEMModel = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, () => {
  'use strict';

  const TAU = 2 * Math.PI;
  const MIN_WEIGHT = 1e-9;

  function mulberry32(seed) {
    let value = seed >>> 0;
    return () => {
      value += 0x6D2B79F5;
      let t = value;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function standardNormal(rng) {
    const u = Math.max(Number.EPSILON, rng());
    const v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * v);
  }

  function generateDataset({ separation = 3.2, sigma = 0.72, seed = 20260901 } = {}) {
    const rng = mulberry32(seed);
    const means = [-separation / 2, separation / 2];
    const labels = Array.from({ length: 48 }, (_, index) => index < 26 ? 0 : 1);
    for (let index = labels.length - 1; index > 0; index -= 1) {
      const swap = Math.floor(rng() * (index + 1));
      [labels[index], labels[swap]] = [labels[swap], labels[index]];
    }
    const data = labels.map((label, index) => ({
      x: means[label] + sigma * standardNormal(rng),
      label,
      jitter: ((index * 37) % 11) / 10
    }));
    data.sort((a, b) => a.x - b.x);
    return data;
  }

  function initialParams(center = 0.45) {
    return {
      means: [center - 0.9, center + 0.9],
      weights: [0.5, 0.5]
    };
  }

  function uniformResponsibilities(data) {
    return data.map(() => [0.5, 0.5]);
  }

  function logNormal(x, mean, sigma) {
    const residual = (x - mean) / sigma;
    return -0.5 * (Math.log(TAU * sigma * sigma) + residual * residual);
  }

  function density(x, mean, sigma) {
    return Math.exp(logNormal(x, mean, sigma));
  }

  function logSumExp2(a, b) {
    const maximum = Math.max(a, b);
    return maximum + Math.log(Math.exp(a - maximum) + Math.exp(b - maximum));
  }

  function eStep(data, params, sigma) {
    return data.map(({ x }) => {
      const logA = Math.log(Math.max(MIN_WEIGHT, params.weights[0])) + logNormal(x, params.means[0], sigma);
      const logB = Math.log(Math.max(MIN_WEIGHT, params.weights[1])) + logNormal(x, params.means[1], sigma);
      const normalizer = logSumExp2(logA, logB);
      const first = Math.exp(logA - normalizer);
      return [first, 1 - first];
    });
  }

  function mStep(data, responsibilities) {
    const counts = [0, 0];
    const sums = [0, 0];
    data.forEach(({ x }, index) => {
      for (let component = 0; component < 2; component += 1) {
        const value = responsibilities[index][component];
        counts[component] += value;
        sums[component] += value * x;
      }
    });
    const means = counts.map((count, component) => sums[component] / Math.max(MIN_WEIGHT, count));
    const weights = counts.map(count => count / data.length);
    if (means[0] <= means[1]) return { means, weights };
    return { means: [means[1], means[0]], weights: [weights[1], weights[0]] };
  }

  function logLikelihood(data, params, sigma) {
    return data.reduce((total, { x }) => {
      const logA = Math.log(Math.max(MIN_WEIGHT, params.weights[0])) + logNormal(x, params.means[0], sigma);
      const logB = Math.log(Math.max(MIN_WEIGHT, params.weights[1])) + logNormal(x, params.means[1], sigma);
      return total + logSumExp2(logA, logB);
    }, 0);
  }

  function elbo(data, params, sigma, responsibilities) {
    return data.reduce((total, { x }, index) => {
      let value = total;
      for (let component = 0; component < 2; component += 1) {
        const r = Math.max(MIN_WEIGHT, responsibilities[index][component]);
        value += r * (
          Math.log(Math.max(MIN_WEIGHT, params.weights[component])) +
          logNormal(x, params.means[component], sigma) -
          Math.log(r)
        );
      }
      return value;
    }, 0);
  }

  function summarizeResponsibilities(responsibilities) {
    const counts = [0, 0];
    let entropy = 0;
    let ambiguous = 0;
    responsibilities.forEach(pair => {
      counts[0] += pair[0];
      counts[1] += pair[1];
      pair.forEach(value => {
        if (value > 0) entropy -= value * Math.log(value);
      });
      if (Math.max(...pair) < 0.75) ambiguous += 1;
    });
    return {
      counts,
      meanEntropy: entropy / responsibilities.length,
      ambiguous,
      ambiguousFraction: ambiguous / responsibilities.length
    };
  }

  function maxParameterChange(before, after) {
    return Math.max(
      Math.abs(before.means[0] - after.means[0]),
      Math.abs(before.means[1] - after.means[1]),
      Math.abs(before.weights[0] - after.weights[0]),
      Math.abs(before.weights[1] - after.weights[1])
    );
  }

  return {
    density,
    eStep,
    elbo,
    generateDataset,
    initialParams,
    logLikelihood,
    mStep,
    maxParameterChange,
    summarizeResponsibilities,
    uniformResponsibilities
  };
});
