# Likelihood & Density

A separate 25-slide teaching deck in **Random thoughts**, with three deterministic interactive experiments. It uses the site's existing Bento runtime and visual language; it does not replace or modify the multidensity-fusion deck.

## Learning path

- Probability mass versus continuous density; probability is a sum or an area, not generally a density height.
- The central distinction: hold the parameter fixed and vary possible data for a sampling density; hold the observed data fixed and vary candidate parameters for a likelihood.
- Coin-flip table and binomial likelihood, Gaussian sensor gain, likelihood ratios, and parameter-dependent normalizers.
- Prior, likelihood, evidence, and posterior; Beta-binomial updates; MLE, MAP, and the posterior mean.
- Conditional independence, shared-prior density fusion, and partially observed states.
- Technical extensions: parameter reparameterization, Jacobians, non-normalizable likelihoods, and unknown Gaussian variance.

Every mathematical object is labeled by its varying argument. A Gaussian-looking likelihood is not automatically presented as a posterior, a likelihood ratio is not a probability, and a posterior density is distinguished from a point estimate. Assumptions and boundary cases appear in the speaker notes.

## Experiments

1. `live/?demo=coin`: compare a PMF over possible head counts with a likelihood over the coin parameter. Change the parameter, sample size, or observed heads.
2. `live/?demo=gaussian`: compare the sampling density and likelihood for `Y | theta ~ Normal(b theta, sigma^2)`. The former has unit area in the measurement; the latter has area `1/b` in the parameter, for positive gain.
3. `live/?demo=prior`: change a Beta prior while keeping the observed counts fixed. The posterior changes; the likelihood does not.

Plots use their own labeled vertical scales and show actual heights, not unit-peak rescalings. Gaussian area metrics refer to the full real line; displayed windows omit tails. The finite-range Beta controls use positive integer shape parameters. Examples are deterministic calculations, not empirical calibration studies.

## Files and dependencies

- `bento-deck.mjs`: slide content, notes, primary source links, and three inline-live map entries.
- `boot.mjs`: serializes the deck, resolves named routes, and loads only the compressed runtime blocks from `../kalman-filter-derivations/index.html`.
- `index.html`: shared MathJax and inline-live styles. MathJax is pinned to 3.2.2 on jsDelivr, following the existing deck.
- `math.mjs`: numerical kernels without external dependencies.
- `live/`: responsive standalone/embedded experiments.
- `test.mjs`: mathematical and document-structure regression tests.
- `test-browser.py`: full-checkout HTTP/Bento integration regression.

The website version requires the repository's existing `assets/bento-inline-live.js`, `assets/bento-inline-live.css`, `assets/mathjax-dynamic.js`, and Kalman deck runtime. Serve the **repository root**, not this folder in isolation. Named routes such as `#likelihood`, `#bayes`, and `#fusion` are supported.

## Validation

Run from a full checkout:

```sh
node --test likelihood-vs-density/test.mjs
python likelihood-vs-density/test-browser.py
```

The browser test requires Python Playwright and its Chromium installation. `CHROMIUM` can specify a browser executable.

Preparation checks passed: nine Node tests against the actual authored deck; 50 mathematical expressions compiled to SVG; all 25 slides rendered in a standalone preview with no detected overflow or JavaScript errors; all three experiments checked at 1132 x 448 and 390 x 844, including 44 total control-endpoint edits, reset, and prior/likelihood invariance. Mobile demos intentionally scroll vertically.

**The full shared Bento-runtime HTTP regression has not been run in the preparation environment.** The offline preview uses a separate lightweight viewer with pre-rendered equations and inlined experiments. Its successful checks do not establish end-to-end compatibility with the deployed Bento runtime. Run the full-checkout browser test before merging.

## Sources

Linked on slides and in notes: Chris Piech's Stanford probability course (continuous distributions, maximum likelihood, and MAP); the Stan User's Guide (change of variables); Wu et al., *Bayesian Data Fusion With Shared Priors* (2024). Numerical examples and algebraic walkthroughs are original derivations with executable checks.
