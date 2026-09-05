(() => {
  'use strict'

  function scalarPosterior({ priorMean, priorSigma, measurement, measurementSigma }) {
    const P = priorSigma * priorSigma
    const R = measurementSigma * measurementSigma
    const K = P / (P + R)
    const postMean = priorMean + K * (measurement - priorMean)
    const postVar = (1 / P + 1 / R) ** -1
    const routes = {
      Bayes: postMean,
      WLS: (priorMean / P + measurement / R) / (1 / P + 1 / R),
      Information: postVar * (priorMean / P + measurement / R),
      Conditioning: priorMean + (P / (P + R)) * (measurement - priorMean)
    }
    const delta = Math.max(...Object.values(routes).map(value => Math.abs(value - postMean)))
    return {
      m: priorMean,
      sp: priorSigma,
      z: measurement,
      sr: measurementSigma,
      P,
      R,
      K,
      postMean,
      postVar,
      routes,
      delta
    }
  }

  function det2(P) {
    return P[0][0] * P[1][1] - P[0][1] * P[1][0]
  }

  function covarianceGeometry({ sx, sy, rho, angleDeg, z, measurementSigma }) {
    const angle = angleDeg * Math.PI / 180
    const P = [[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]]
    const h = [Math.cos(angle), Math.sin(angle)]
    const m = [-1.05, .55]
    const Ph = [P[0][0] * h[0] + P[0][1] * h[1], P[1][0] * h[0] + P[1][1] * h[1]]
    const S = h[0] * Ph[0] + h[1] * Ph[1] + measurementSigma * measurementSigma
    const K = [Ph[0] / S, Ph[1] / S]
    const innovation = z - (h[0] * m[0] + h[1] * m[1])
    const mp = [m[0] + K[0] * innovation, m[1] + K[1] * innovation]
    const Pp = [
      [P[0][0] - K[0] * S * K[0], P[0][1] - K[0] * S * K[1]],
      [P[1][0] - K[1] * S * K[0], P[1][1] - K[1] * S * K[1]]
    ]
    const areaRatio = Math.sqrt(Math.max(0, det2(Pp)) / det2(P))
    const gainAngleSine = Math.abs(K[1] * h[0] - K[0] * h[1]) /
      Math.max(1e-12, Math.hypot(...K) * Math.hypot(...h))
    return { sx, sy, rho, angleDeg, angle, z, sr: measurementSigma, P, Pp, h, m, mp, S, K, innovation, areaRatio, gainAngleSine }
  }

  function eigen2(P) {
    const a = P[0][0]
    const b = (P[0][1] + P[1][0]) / 2
    const d = P[1][1]
    const disc = Math.sqrt(Math.max(0, (a - d) ** 2 + 4 * b * b))
    return {
      l1: Math.max(1e-12, (a + d + disc) / 2),
      l2: Math.max(1e-12, (a + d - disc) / 2),
      angle: .5 * Math.atan2(2 * b, a - d)
    }
  }

  window.KalmanModel = Object.freeze({ scalarPosterior, covarianceGeometry, eigen2 })
})()
