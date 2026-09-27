# Max Entropy Moment Kalman Filter

28 native Bento slides and four browser-local numerical labs for [arXiv:2506.00838v1](https://arxiv.org/abs/2506.00838). Published under Random Thoughts at /mem-kf/.

## Rebuild

Run `node mem-kf/build.mjs` from the repository root, then `node --test mem-kf/test.mjs`. The build preserves the native Bento runtime in the generated index. On the first build only, it clones the existing advanced-state-representations Bento shell. It also generates study.html and four model-derived SVG snapshots, and inserts one scoped homepage card.

## Numerical scope

These are original scalar teaching models, not the authors' code or benchmark reproduction. The solver uses finite trapezoidal quadrature, scaled Legendre features, a log-partition dual, covariance Hessian, and damped Newton with line search. Independent polynomial moment propagation is exact relative to supplied moments. All reference curves are numerical. MAP labels refer to grid argmax; no SDP or optimality certificate is implemented. The quartic sensor demo uses fixed illustrative observations; the filtering demo uses seeded Gaussian noise.

## Files

- bento-deck.mjs: canonical content, equations, notes, and inline-lab mappings.
- model.mjs: pure numerical functions shared by snapshots, tests, and live controls.
- live.html: responsive, accessible lab interface and Bento keyboard integration.
- build.mjs: deterministic output generation and homepage integration.
- test.mjs and browser.cjs: numerical, layout, and integration checks.

Source attribution is included on every slide and in the study companion. The retained Bento runtime keeps its existing license notices.
