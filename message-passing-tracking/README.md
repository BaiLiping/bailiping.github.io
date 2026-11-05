# Message passing for multitarget tracking

A 12-slide Bento presentation and an interactive walkthrough of **Figure 4** in Meyer et al., *Message Passing Algorithms for Scalable Multitarget Tracking*, Proceedings of the IEEE 106(2), 221–259 (2018), DOI: https://doi.org/10.1109/JPROC.2018.2789427.

Public entry: `/message-passing-tracking/`, under Random thoughts. Figure 4 is on printed p. 242 (PDF p. 22); the walkthrough implements the single-sensor schedule in Section IX-A, pp. 242–243. The earlier time slice is summarized by its already-computed posterior inputs. The current slice preserves the complete factor topology for two legacy candidates and two measurements, including both possible new-target branches.

The default 17 steps expose the prior, prediction, prediction copy, beta, xi, initialization, three nu/phi association sweeps, kappa, iota, gamma, varsigma, and normalized beliefs. Select any target–measurement pair and any output entry to inspect the actual contributions. Orange edges identify the inspected path; green edges identify its inputs. Other pairs update in parallel during each half-sweep. Controls vary measurement 2, detection probability, expected newly detected targets, and sweep count.

## Numerical scope

This is an original finite-state teaching example, not a reported experiment from the paper. Every target has states `[absent, L, R]`, with position cells at 0 and 1. The absent state aggregates the paper's dummy density, whose integral is one. The two previous posteriors are `[.1,.8,.1]` and `[.2,.15,.65]`. Survival is .95; a surviving target stays in its cell with probability .85. Measurement density is Gaussian with standard deviation .35. Measurement 1 is .3; measurement 2 defaults to .62. Clutter intensity is .5, and new-target position mass is `[.5,.5]`.

The birth control is mu_n, the expected number of **newly detected** targets (no additional detection factor in v). q and v follow (72)–(74); beta and xi follow (78)–(79). Association uses the full vector initialization (29) and updates (27)–(28), with each outgoing vector rescaled so its nonmatching entries equal 1. It deliberately does not use the alternative scalar initialization stated after (31). All finite normalized beliefs remain approximate on this loopy graph. Exact references enumerate all seven valid matchings and are independently checked against full enumeration of the finite-state joint model.

## Files and validation

- `model.js`: pure numerical model, all message rounds, and exact inference.
- `graph.js`: topology, geometry, active message paths, and SVG rendering.
- `live/lesson.js`: formulas, arithmetic tables, and explanatory copy.
- `live/app.js`: canonical controls and rendering; playback pauses when hidden.
- `deck.mjs` and `build.mjs`: slide source and generated Bento/static reading views.
- `test.cjs`: inference checks, scalar/vector agreement, tree exactness, and parameter bounds.
- `browser-test.cjs`: all slides and message steps, controls, arithmetic, math, focus, print, narrow and mobile layouts.

From the site root:

```sh
node message-passing-tracking/build.mjs
node --test message-passing-tracking/test.cjs
# With Playwright available through NODE_PATH, optionally set BP_CHROMIUM:
node message-passing-tracking/browser-test.cjs
```

Serve the repository on port 8793 for browser QA, or set `TRACKING_BASE_URL`. The existing Bento runtime and shared adapters are reused without modification. MathJax is pinned to 3.2.2. The downloaded source PDF is not redistributed; slides link its DOI and the ANU repository copy.
