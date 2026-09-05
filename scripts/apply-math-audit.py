#!/usr/bin/env python3
"""Apply the 2026-09-05 audit corrections, preserving existing demos/layouts.

Run from the repository root. Idempotent, fails on unrecognized source, and
stages all edits in memory before writing. Canonical radio/deck sources are
updated as well as deployed HTML; regenerate with the existing build scripts.
"""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]
FILES = {}
COUNTS = {}


def read(path):
    if path not in FILES:
        FILES[path] = (ROOT / path).read_text()
    return FILES[path]


def replace(path, old, new, label, optional=False):
    s = read(path)
    if old in new and new in s:
        return
    n = s.count(old)
    if not n:
        if new in s or optional:
            return
        raise ValueError(f'{label}: expected text missing in {path}: {old[:110]!r}')
    FILES[path] = s.replace(old, new)
    COUNTS[label] = COUNTS.get(label, 0) + n


def rx(path, pattern, new, label, done=None, optional=False):
    s = read(path)
    if done and done in s:
        return
    s, n = re.subn(pattern, lambda m: new(m) if callable(new) else new, s, flags=re.S)
    if not n and not optional:
        raise ValueError(f'{label}: expected pattern missing in {path}')
    FILES[path] = s
    COUNTS[label] = COUNTS.get(label, 0) + n


# F01: finite-set convolution, not an unspecified product of component sets.
p = 'bp-vs-pmbm/index.html'
replace(p,
    'f(X) = Σ<sub>h</sub> w<sub>h</sub> · f<sup>ppp</sup>(X<sup>u</sup>) Π<sub>i</sub> f<sup>&thinsp;bern</sup><sub>h,i</sub>(X<sub>i</sub>)',
    'f(X) = Σ<sub>X<sup>u</sup> ⊎ X<sup>d</sup> = X</sub> f<sup>ppp</sup>(X<sup>u</sup>) Σ<sub>h</sub> w<sub>h</sub> f<sup>MB</sup><sub>h</sub>(X<sup>d</sup>)', 'F01')
replace(p,
    '<span class="num">(4) undetected Poisson ⊎ multi-Bernoulli mixture</span></div>',
    '<span class="num">(4a) undetected Poisson ⊎ multi-Bernoulli mixture</span></div>\n'
    '    <div class="eq"><span class="m">f<sup>MB</sup><sub>h</sub>(X<sup>d</sup>) = Σ<sub>X₁ ⊎ ··· ⊎ X<sub>n<sub>h</sub></sub> = X<sup>d</sup></sub> Π<sub>i=1</sub><sup>n<sub>h</sub></sup> f<sup>bern</sup><sub>h,i</sub>(X<sub>i</sub>)</span><span class="num">(4b) sum over disjoint component-set decompositions</span></div>\n'
    '    <div class="eqnote">The component sets may be empty; each Bernoulli density is zero on sets with more than one element. The hypothesis weights are nonnegative and sum to one. Both disjoint-set sums are part of the unlabelled PMBM density.</div>', 'F01')

# F02: summing associations produces the marginal trajectory-map posterior.
p = 'jvs-slam/index.html'
replace(p, '<i>X</i><sub>0:k</sub>, <i>M</i>, <i>A</i><sub>1:k</sub> | <i>Z</i><sub>1:k</sub>',
        '<i>X</i><sub>0:k</sub>, <i>M</i> | <i>Z</i><sub>1:k</sub>', 'F02')
replace(p, 'and the data associations:</p>',
        'and the data associations. Summing over association histories gives the marginal trajectory–map posterior:</p>', 'F02')
replace(p, '<p class="lede">Each association hypothesis <i>A</i>',
        '<p class="lede">Here <i>w</i><sup>(A)</sup> = <i>p</i>(<i>A</i><sub>1:k</sub> = <i>A</i> | <i>Z</i><sub>1:k</sub>), and the weights sum to one. Each association hypothesis <i>A</i>', 'F02')
replace(p, 'mixture over globally coherent explanations — calibrated, multi-modal',
        'mixture over globally coherent explanations — multi-modal; calibration depends on the model', 'F02')
replace(p, 'carrying the 9% it deserves.', 'carrying an illustrative weight of 9%.', 'F02')

p = 'frame-registration/index.html'
replace(p,
    'And in the limit the family closes over the ladder below it: as $\\sigma\\to 0$ the responsibilities collapse onto the nearest neighbor, so ICP is this family at zero temperature, with the outlier bin as its gate.',
    'The zero-noise limit needs a qualification: with no outlier component and a unique nearest neighbor, responsibilities concentrate on that neighbor as $\\sigma\\to0$. With a positive outlier weight, this nearest-neighbor limit applies only conditional on an inlier assignment. At fixed nonzero distances to every Gaussian center, all unconditional inlier responsibilities instead tend to zero and the outlier probability tends to one.', 'F03')
replace(p,
    'Here is what the Gaussians buy that ICP never gets: the score is an analytic function of the pose, so its gradient <em>and Hessian</em> come in closed form.',
    'For fixed point-to-cell assignments, the Gaussian score is analytic in the pose, so its gradient <em>and Hessian</em> have closed forms. With hard cell lookup, the full objective is generally only piecewise analytic; the formulas below apply within a region of unchanged cell assignments.', 'F05')
replace(p, 'No single cell border is a border of all four grids, so the worst seams cancel.',
    'The shifted grids reduce sensitivity to individual cell boundaries, but their sum is not guaranteed to be continuous or globally analytic.', 'F05')
rx(p, r'And because the optimum \$\\theta\^\\star\$ is defined implicitly.*?learn an initializer plus a per-problem kernel scale\.',
    r'Implicit differentiation requires more than stationarity. Let $\mathcal D$ denote the input clouds and suppose $F(\theta,\mathcal D)$ is twice continuously differentiable with a nonsingular pose Hessian at the selected local optimum. Then the implicit function theorem gives $\partial\theta^\star/\partial\mathcal D=-[\nabla^2_{\theta\theta}F]^{-1}\nabla^2_{\theta\mathcal D}F$, evaluated there. Gauge freedom, degeneracy, or switching between minima can invalidate this local derivative. Under these conditions, Neural MMD-Reg can differentiate through its solver.',
    'F06', done='Implicit differentiation requires more than stationarity.')

# F04: supplied geometric inputs can still be differentiation variables.
p = 'differentiable-ray-tracing/index.html'
replace(p,
    'Every quantity on this diagram — hit point, angle, length, AoD/AoA — is a deterministic function of the specified scene. There is no free variable left here, which is exactly why gradients in a radio ray tracer attach to other quantities: the electromagnetic and system parameters covered in the next section.',
    'Hit point, angle, length, and AoD/AoA are deterministic functions of the scene inputs, not independent optimization variables. They can still be differentiated with respect to those inputs. This demo holds the wall fixed; that is a modeling choice, not a mathematical restriction on geometric gradients.</p><div class="formula-block" role="math" tabindex="0">∂ℓ/∂h = 2(2h − y<sub>S</sub> − y<sub>R</sub>) / ℓ(h)</div><p>This derivative applies to a nonzero-length path while the same valid reflection branch is retained.', 'F04')
replace(p, "These are the parameters Sionna's gradients actually attach to — the exact list, with what stays non-differentiable, is in the next section.",
    'Supported material, system, and geometric parameters can be optimized. The next section separates smooth geometric derivatives from discrete path selection.', 'F04')
rx(p, r'paths, CIRs, and radio maps with respect to — in the documentation\x27s.*?geometry you specified is not on it\.',
    'channel responses and radio maps with respect to supported material, pattern, and geometric parameters. The official introduction explicitly includes object positions and orientations. These are examples, not an exhaustive list. Support for arbitrary mesh deformations must be checked for the chosen solver and parameterization.',
    'F04', done='These are examples, not an exhaustive list.')
replace(p, '<p class="card-kicker">not differentiable ✗</p>\n          <h3>The things this page derives or enumerates</h3>',
    '<p class="card-kicker">support and smoothness limits</p>\n          <h3>Separate geometry from discrete path selection</h3>', 'F04')
replace(p,
    '<strong>Mesh geometry.</strong> No official parameter list includes scene mesh vertices or a wall coordinate. The wall from section&nbsp;02 stays a constant of the forward model — exactly as you would expect.',
    '<strong>Geometric parameterization.</strong> A wall coordinate or object pose can influence path length and angles differentiably on a fixed valid branch. Supported rigid poses do not establish automatic-differentiation support for every mesh-vertex deformation. The wall in section&nbsp;02 is held fixed only for that demonstration.', 'F04')

# F13 and F14: preserve the numerical demonstrations, fix their interpretation.
p = 'graph-slam/index.html'
rx(p, r'Those small errors, like the scale, are frozen into the solution:\s*nothing in the bearings can ever correct them — the monocular gauge ambiguity in action\.',
    'The relative-heading and baseline-direction errors are frozen by these extra hard constraints, not by monocular gauge freedom: sufficiently informative bearings can generally constrain them. Only the four planar-similarity degrees of freedom are gauge. A minimal gauge choice fixes the first pose and one nonzero baseline length.',
    'F13', done='not by monocular gauge freedom:')
p = 'sampling-playground/index.html'
replace(p, 'consistent with the ~0.07 standard error that a 388-draw effective size implies.',
    'consistent with the approximately 0.07 large-sample standard error for this particular mean estimator. The 388-draw Kish weight ESS is a weight-degeneracy diagnostic, not a general function-specific standard-error guarantee.', 'F14')

# F07-F10: one discrete variable names a COMPLETE path, including LoS/clutter.
RADIO = [
    'mpc-detection-to-bounce-count/index.html',
    'scripts/enhance-radio-slam-companion.mjs',
    'scripts/add-radio-slam-problem-formulation.mjs',
    'scripts/upgrade-radio-slam-formulation-3d.mjs',
]
bridge = r'''\1
    <p class="lede">Collect continuous unknowns in \(\boldsymbol\Theta\). Each entry of \(A\) selects a complete path hypothesis: clutter, LoS, or an ordered wall chain. Known BS states are conditioned on; uncertain BS and calibration states are included in \(\boldsymbol\Theta\).</p>
    <div class="eq math-eq">
      \[
      \boldsymbol\Theta=\{\mathbf T_{1:T},\mathcal M,\boldsymbol\delta,\boldsymbol\xi,\mathbf B_{\mathrm{uncertain}}\},
      \]
      \[
      a_{ts\ell}\in\mathcal H=\{0,\mathrm{LoS}\}\cup
      \{(k,j_1,\ldots,j_k):k\ge1,\text{ ordered wall chain}\}.
      \]
    </div>
    <p class="eq-note">Here \(0\) denotes clutter, not a map entity. LoS has no reflector; a reflected path selects every wall in its ordered chain. Bounce count is determined by the hypothesis, not an additional independent association variable. Finite-support and visibility tests determine admissible candidates.</p>
    <div class="eq math-eq">
      \[
      p(\boldsymbol\Theta,A\mid Z,U,\mathbf B_{\mathrm{known}})
      \propto p(\boldsymbol\Theta)
      \prod_{t=2}^{T}f_t^{\mathrm{rel}}
      \prod_{t,s}\mathcal L_{\mathrm{set}}(\mathcal Z_{ts},A_{ts}\mid\boldsymbol\Theta,\mathbf B_s).
      \]
    </div>
    <p class="eq-note">\(U\) is optional motion information, and \(f_t^{\mathrm{rel}}\) is the corresponding relative-pose factor. The joint set likelihood \(\mathcal L_{\mathrm{set}}\) includes association weights, clutter, missed detections, count terms, and the admissibility constraints of the declared observation model. A product of independent per-MPC mixture factors is not automatically this set likelihood. One-to-one constraints, when assumed, apply to a physical path hypothesis within a scan, not to every use of the same wall.</p>
    <div class="eq math-eq">
      \[
      \begin{aligned}
      \boldsymbol\Theta^*=\arg\min_{\boldsymbol\Theta}\;&
      \|\mathbf r^{\mathrm{prior}}\|^2_{\boldsymbol\Omega_0}
      +\sum_{t=2}^{T}\|\mathbf r_t^{\mathrm{rel}}\|^2_{\boldsymbol\Omega_t}\\
      &+\sum_{(t,s,\ell):a_{ts\ell}\ne0}
      \rho\!\left(\|\mathbf r^{\mathrm{rad}}_{ts\ell}(\mathbf x_t,\mathcal M,a_{ts\ell})\|^2_{\boldsymbol\Omega_{ts\ell}}\right).
      \end{aligned}
      \]
    </div>
    <p class="eq-note">This is the fixed-association geometric back-end objective; it includes LoS. Ordinary Gaussian least squares uses \(\rho(s)=s\), fixed covariances, and state-independent omitted likelihood terms. A robust loss is a modeling choice. Retain state-dependent log-determinants, detection/count, association, and visibility terms when claiming the full MAP objective.</p>
    <div class="companion-steps">'''
for p in RADIO:
    rx(p, r'(<div class="subsection-block" id="radio-graph-objective">\s*<h3[^>]*>.*?</h3>).*?<div class="companion-steps">',
       lambda m: bridge.replace('\\1', m.group(1)), 'F07/F08', done='Each entry of \\(A\\) selects a complete path hypothesis', optional=True)
    replace(p, r'The map entity assigned to MPC \(\ell\); \(a=0\) denotes clutter or no accepted assignment.',
       r'The complete path assigned to MPC \(\ell\): LoS or an ordered wall chain; \(a=0\) denotes clutter.', 'F07', optional=True)
    replace(p, r'With fixed \(A,Q\), solve nonlinear least squares. With uncertain \(A,Q\), alternate, marginalize, branch, or use mixture/max-mixture and robust factors.',
       r'With fixed complete path assignments \(A\), solve the conditioned problem. With uncertain \(A\), alternate, marginalize, or branch; mixture/max-mixture and robust factors are practical approximations.', 'F07', optional=True)
    # The 2D hybrid expression needs a JOINT association/observation likelihood.
    replace(p, r'p(\mathcal Z_{ts}\mid\mathbf x_t,\mathcal M,A_{ts},\mathcal B_s)',
       r'p(\mathcal Z_{ts},A_{ts}\mid\mathbf x_t,\mathcal M,\mathcal B_s)', 'F08', optional=True)
    # Replace the independent-assignment posterior in the full 3D formulation.
    rx(p, r'p\(\\boldsymbol\\Theta,A\\mid Z,\{\}\^W\\mathbf T_\{B_\{1:S\}\}\).*?\\mathbf T_\{B_s\}\\right\),',
       r'p(\boldsymbol\Theta,A\mid Z,{}^W\mathbf T_{B_{1:S}})'
       '\n      '+r'\propto p(\boldsymbol\Theta)\prod_{t,s}'
       '\n      '+r'\mathcal L_{\mathrm{set}}(\mathcal Z_{ts},A_{ts}\mid\boldsymbol\Theta,{}^W\mathbf T_{B_s}),',
       'F08', optional=True)
    # Use the covariance of the RESIDUAL, rather than raw seconds/metres mixed.
    replace(p, r'\mathbf z_{ts\ell};'+'\n      '+r'\mathbf h(\mathbf x_t,\mathcal M,h;\mathcal B_s),',
       r'\mathbf r^{\mathrm{rad}}_{ts\ell}(\mathbf x_t,\mathcal M,h);\mathbf0,', 'F09', optional=True)
    old = r'Multiplying the delay residual by \(c\) expresses it in metres. The covariance must use the same units and should retain delay–angle correlations when the channel estimator provides them.'
    new = old + r' Explicitly, for raw delay/angle/gain covariance \(\boldsymbol\Sigma_z\), let \(D=\operatorname{diag}(c,1,\ldots,1)\). Then \(\boldsymbol\Sigma_h=D\boldsymbol\Sigma_zD^{\mathsf T}\) and \(\boldsymbol\Omega_h=\boldsymbol\Sigma_h^{-1}\), including all cross-covariances.'
    if new not in read(p): replace(p, old, new, 'F09', optional=True)
    old = r'The covariance \(\boldsymbol\Sigma_h\) must use the same coordinates and should preserve delay–direction correlations provided by the channel estimator.'
    new = old + r' If the estimator supplies azimuth/elevation coordinates, first use the local residual Jacobian \(J\): \(\boldsymbol\Sigma_h\approx J\boldsymbol\Sigma_zJ^{\mathsf T}\). If directions are already in the chosen tangent bases, the delay-unit conversion is exactly \(D\boldsymbol\Sigma_zD^{\mathsf T}\), with \(D=\operatorname{diag}(c,1,\ldots,1)\). The spherical logarithm has domain \(0\le\theta<\pi\), with its continuous zero-angle limit. At the antipode \(\theta=\pi\), its direction is not unique; gate out that candidate or use an explicitly chosen alternative residual. Clamp dot products to \([-1,1]\) before numerical arccos evaluation.'
    if new not in read(p): replace(p, old, new, 'F09/F10', optional=True)
    # Clarify both mixture equations immediately next to their definitions.
    rx(p, r'(<p class="eq-note">\\\(\\kappa(?:_\{ts\})?\\\) is (?:a )?clutter intensity.*?</p>)',
       lambda m: m.group(1)+r'''
    <p class="eq-note" data-math-audit="mixture-scope">The joint set likelihood includes association weights and the detection/count and admissibility terms of the chosen model. The displayed \(\phi\) is a local mixture surrogate, not automatically a normalized observation density or a complete set likelihood. Clutter and path terms must share measurement coordinates and base measure. For a PPP intensity \(\lambda_\Theta\), the set likelihood includes \(e^{-\Lambda_\Theta}\prod_{z\in Z}\lambda_\Theta(z)\), where \(\Lambda_\Theta=\int\lambda_\Theta(z)\,dz\); a Bernoulli-path model instead requires its own missed-detection terms. Independent mixtures do not enforce one-to-one assignment. The conditioned least-squares form also assumes that omitted likelihood normalizers and detection terms are constant, or is only a geometric surrogate.</p>''',
       'F08', done='data-math-audit="mixture-scope"', optional=True)

# Repair the audited escaped-LaTeX rendering defect in the delay-resolution note.
p = RADIO[0]
rx(p, r'<p class="lede">Finite bandwidth gives finite delay resolution\..*?</p>',
    r'<p class="lede">Finite bandwidth gives finite delay resolution. As a useful scale, \(\Delta\tau\sim1/B\) and \(\Delta L\sim c/B\), although waveform, windowing, SNR, array processing, and super-resolution estimation determine the practical limit. Two physical paths inside one resolution cell may appear as a single component.</p>',
    'TeX escaping')
p = RADIO[1]
FILES[p] = '\n'.join(re.sub(r'(?<!\\)\\(?=(?:[()]|Delta|tau|sim))', lambda m: '\\\\', line)
    if "'    <h3" in line and 'Finite bandwidth gives finite delay resolution.' in line else line
    for line in read(p).split('\n'))

# F11/F12: keep the finite-control teaching model, make its assumptions explicit.
p = 'advanced-state-representations/bento-deck.mjs'
replace(p, 'The normal equations can be written directly in pose updates:<br>',
    'For a positive-definite kernel, the pose-update equations are:<br>', 'F11')
replace(p,
    'A kernel function ${I(R`K(t,t\')`)} replaces the explicit inner product of basis functions.<br><br><b>New choice:</b> kernel family and hyperparameters.',
    'A finite-basis ${I(R`K`)} can be singular. The inverse-free form remains valid:<br>${M(R`\\delta^*=K A^{\\mathsf T}(I+A K A^{\\mathsf T})^{-1}b`)}', 'F11')
replace(p, 'The inverse kernel acts as a smoothing prior in the normal equations.',
    'The inverse-kernel equation requires K positive definite. The equivalent inverse-free solution remains valid for a positive-semidefinite, rank-deficient finite-basis kernel.', 'F11')
replace(p, 'STATIC FALLBACK · RANDOM-WALK GP', 'STATIC FALLBACK · INTERPOLATED CONTROL MODEL', 'F12')
replace(p, 'Scalar random-walk motion factors connect neighbors; measurements interpolate between adjacent controls.',
    'Random-walk control prior; deterministic linear interpolation between controls. The band describes this finite-control model, not full continuous-time GP uncertainty.', 'F12')
replace(p, 'then inverts the information matrix for the displayed interpolation uncertainty.',
    'then inverts the information matrix for the uncertainty of the deterministically interpolated controls. It omits continuous-process bridge uncertainty; adding a query-only bridge term would not repair the asynchronous measurement model.', 'F12')
p = 'advanced-state-representations/live/app.js'
replace(p, '>95% band</span>', '>≈95% interpolation band</span>', 'F12')
replace(p, 'without making the graph dense.`;',
    'without making the graph dense. The band is for deterministic interpolation of uncertain controls, not the full continuous-time GP.`;', 'F12')
p = 'advanced-state-representations/model.js'
replace(p, '  function gpExperiment(options = {}) {',
    '  // Finite random-walk control model with deterministic linear interpolation.\n'
    '  // quad(row, covariance) is control-interpolation variance, not Wiener-bridge variance.\n'
    '  function gpExperiment(options = {}) {', 'F12')

# Do not write anything until every required correction above has matched.
changed = []
for p, s in FILES.items():
    if (ROOT / p).read_text() != s:
        (ROOT / p).write_text(s)
        changed.append(p)
print(json.dumps({'corrections': COUNTS, 'changed': changed}, indent=2))

# Report relevant companion-slide text for verification; do not rewrite diagrams
# blindly or make changes to unrelated slides.
for p in ['frame-registration-slides/index.html', 'bp-vs-pmbm-slides/index.html', 'mpc-detection-to-bounce-count-slides/index.html']:
    s = (ROOT / p).read_text()
    m = re.search(r'<script[^>]*id="bento-doc"[^>]*>(.*?)</script>', s, re.S)
    if not m:
        print('COMPANION no embedded Bento document:', p)
        continue
    deck = json.loads(m.group(1))
    for slide in deck.get('slides', []):
        text = json.dumps(slide, ensure_ascii=False)
        if re.search(r'CPD|zero.temperature|implicit function|analytic|PMBM|a_\{ts', text, re.I):
            snippets = []
            for el in slide.get('elements', []):
                h = el.get('html', '')
                if re.search(r'CPD|zero.temperature|implicit|analytic|PMBM|posterior|association', h, re.I):
                    snippets.append([el.get('id'), re.sub('<[^>]+>', ' ', h)[:900]])
            print('COMPANION', p, slide.get('id'), json.dumps(snippets, ensure_ascii=False))
