# BP / PMBM lesson audit — September 7, 2026

This audit covers the article, native Bento slides, and their three embedded demonstrations. It preserves the existing site layout and slide runtime.

## Corrections

- The article previously stopped BP after five sweeps while labeling the answer converged. A shared solver now reports its actual stopping status, checks both log-message change and edge-belief consistency, and distinguishes a safety cap from convergence.
- The previous embedded slide application was an unrelated radio-SLAM walkthrough with hard-coded particle values. It is replaced with real shared one-scan assignment, BP, and top-k hypothesis computations.
- BP is an inference algorithm; PMBM is a posterior family. Exact assignment enumeration is not a full PMBM filter. Both views can be combined.
- The article retains the valid PMBM set-partition density from the earlier site audit and adds the actual conditional-parent association weights, Bernoulli existence updates, PPP evidence, and undetected-PPP update. New-target existence is explicitly conditional on being unassigned to existing tracks.
- Marginalization, product-of-marginals projection, and Gaussian moment matching are distinguished. TOMB/P retains an undetected-target PPP. Coalescence and mode merging are not universal properties of BP.
- Gating is identified as a numerical likelihood restriction in this demonstration. Complexity is stated per sweep and per actual inference task; no universal cubic PMBM runtime or monotonic error-versus-loopiness claim is made.
- Top-k marginals are explicitly renormalized. Their approximation error is separate from BP error. Discarded posterior mass bounds each marginal error; that mass is available here because all events are enumerated.
- Nineteen native slides include two additional explanations: full PMBM evidence and dependence lost by a product projection. Primary references are linked on teaching slides. Graph endpoints and text placement are corrected.

## Reproducible default benchmark

The default scene has 3 certain existing tracks, 4 measurements, PD = 0.9, homogeneous clutter intensity 5e-5, zero undetected PPP, and optional 99% two-dimensional Gaussian gating.

| Quantity | Value |
|---|---:|
| Positive-weight gated assignments | 22 |
| Active edges / independent graph cycles | 7 / 2 |
| BP sweeps to stopping tolerance 1e-10 | 30 |
| Maximum fixed-point BP–exact difference | 6.9262384463 percentage points |
| Difference between five-sweep and final BP beliefs | 0.4276701178 percentage points |
| Probability retained by the five best assignments | 85.8364402014% |

These are association results for one teaching scene, not a comparison of complete trackers or evidence that either framework universally performs better.

## Validation

Run from the repository root:

```sh
node bp-vs-pmbm-slides/build-deck.mjs
node --test tests/bp-pmbm.test.cjs
```

The 15 automated numerical/content tests include an independent Cartesian-product enumeration oracle, 100 randomized small problems, exactness on acyclic graphs, symmetry, half-step bookkeeping, normalization, gated-out probabilities, convergence-cap reporting, row-scale invariance, log-domain enumeration, top-k total variation, and PMBM normalization with uncertain existence and PPP evidence.

Browser checks use real locally served URLs and the actual sandboxed iframe configuration:

```sh
python -m pip install playwright==1.57.0
python -m playwright install chromium
python tests/bp-pmbm-browser.py
```

The GitHub Actions workflow is read-only and fails when generated content is stale. It checks controls, matching results across the article and all three demos, nineteen slides, embedded-demo readiness, text overflow, and JavaScript errors. Screenshots and a JSON report are retained as a short-lived workflow artifact.

## Primary sources

1. J. L. Williams and R. A. Lau, “Approximate Evaluation of Marginal Association Probabilities with Belief Propagation,” IEEE TAES, 50(4), 2014. https://arxiv.org/abs/1209.6299
2. J. L. Williams, “Marginal Multi-Bernoulli Filters: RFS Derivation of MHT, JIPDA and Association-Based MeMBer,” IEEE TAES, 51(3), 2015. https://arxiv.org/abs/1203.2995
3. F. Meyer et al., “Message Passing Algorithms for Scalable Multitarget Tracking,” Proceedings of the IEEE, 106(2), 2018. https://doi.org/10.1109/JPROC.2018.2789427
4. A. F. García-Fernández, J. L. Williams, K. Granström, and L. Svensson, “Poisson Multi-Bernoulli Mixture Filter: Direct Derivation and Implementation,” IEEE TAES, 54(4), 2018. https://arxiv.org/abs/1703.04264
