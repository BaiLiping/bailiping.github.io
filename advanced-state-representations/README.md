# Advanced State Variable Representations

A worked, 43-slide native Bento lesson for `/advanced-state-representations/`, with seven inline experiments and a scrollable `study.html` companion. All 24 original named slide routes are preserved. The lesson develops geometry, frame conventions, optimization, splines, SDE-based Gaussian processes, uncertainty, and group-valued trajectories in small steps.

## Source and generated files

- `bento-deck.mjs`: retained canonical import entry point; re-exports `lesson-deck.mjs`.
- `lesson-deck.mjs`: slide content, equations, explanatory notes, source references, static fallbacks, and inline-lab map.
- `lesson-model.js`: pure, dependency-free numerical models shared by Node tests and browser experiments.
- `lesson-figures.js`: deterministic SVG figures and numerical readouts from those models.
- `build.mjs`: validates route continuity and slide geometry, preserves this deck's existing Bento vendor payload, and generates `index.html`, `study.html`, and `figures/*.svg`.
- `live/index.html`, `live/lesson.js`, `live/lesson.css`: accessible controls and responsive layouts.
- `tests/lesson-math.test.cjs`: model identities, independent matrix exponentials, finite-difference checks, spline formulas, and independently assembled dense GP conditioning.
- `tests/lesson-browser.cjs`: Chromium checks and screenshots for the slides, controls, math, focus/navigation, print, and mobile routes.
- `math-audit.md`: corrections, assumptions, reference links, and validation boundaries.

The old `model.js`, `test-model.cjs`, `live/app.js`, and `live/styles.css` are retained as legacy regression fixtures. They are no longer imported by the live entry point. In particular, their finite-control interpolation model is not presented as the new exact continuous-time GP.

## Rebuild and validate

From the repository root:

```sh
node --test advanced-state-representations/tests/lesson-math.test.cjs
node advanced-state-representations/test-model.cjs
node advanced-state-representations/build.mjs
```

The dedicated GitHub Actions workflow installs isolated Playwright dependencies, runs browser checks, checks deterministic rebuilding, and commits only validated generated output. QA screenshots and numerical logs are retained as a workflow artifact.

## Laboratories

| Direct query under `live/` | What it demonstrates |
| --- | --- |
| `?demo=tangent` | Exact SO(2) tangent geometry: the straight step leaves the circle. |
| `?demo=manifold` | Exact repeated rotations versus first-order matrix accumulation, including scale and angular bias. |
| `?demo=optimize` | An actual right-perturbation Gauss–Newton fit of a planar pose with fixed point correspondences. |
| `?demo=adjoint` | Equivalent right/left increments after adjoint conversion, versus using the same numbers on the wrong side. |
| `?demo=spline` | Linear/cubic B-spline bases, exact local support, coefficient influence, and analytic derivatives. |
| `?demo=gp` | Exact random-walk or position–velocity SDE inference, bridge uncertainty, and information/covariance structure. |
| `?demo=pose` | Constant-twist interpolation, a valid split model, and invalid matrix blending. |

Bento owns slide identity, navigation, notes, scaling, overview, and printing. Each lab automatically mounts over its regular slide's exact static region. There are no launcher buttons or hidden state slides. In an embedded lab, Page Up/Down navigate the deck; Escape returns focus to Bento; focused controls retain their own arrow keys. On narrow screens, controls stack and wide mathematical plots can scroll rather than cropping the page.

## Scientific scope

The examples are original deterministic teaching models, not reproduced paper datasets or timing benchmarks. Spatial demos are planar where labeled; the surrounding derivations explain the full SO(3)/SE(3) conventions. The GP demo includes states at every observation time so that its asynchronous likelihood and conditional query distribution are exact for the stated linear SDE. It uses small dense matrix solves to expose the mathematics, not to claim a sparse implementation or fewer support states than observations. Lie-group GP construction is explained separately and is not claimed to be implemented by the scalar lab.
