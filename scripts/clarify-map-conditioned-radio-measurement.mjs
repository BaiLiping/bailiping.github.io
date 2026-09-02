import { readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

const twoDPath = resolve('scripts/add-radio-slam-problem-formulation.mjs')
const threeDPath = resolve('scripts/upgrade-radio-slam-formulation-3d.mjs')

function replaceRegion(text, startMarker, endMarker, replacement, sentinel, label) {
  if (text.includes(sentinel)) return text
  const start = text.indexOf(startMarker)
  const end = text.indexOf(endMarker, start + startMarker.length)
  if (start < 0 || end < 0) throw new Error(`Could not isolate ${label}`)
  return text.slice(0, start) + replacement + text.slice(end)
}

function replaceOptional(text, before, after) {
  if (text.includes(after) || !text.includes(before)) return text
  return text.replace(before, after)
}

function patch2D(source) {
  source = replaceOptional(
    source,
    '<p class="lede">The channel estimator returns one noisy tuple per resolvable MPC. The geometric prediction depends on the current UE state, the known BS, and zero, one, or two ordered wall variables according to the hypothesis.</p>',
    '<p class="lede">The channel estimator returns one noisy tuple per resolvable MPC. At the scene level, its distribution is conditioned on the current UE state, the complete map \\(\\mathcal M\\), the known BS, and a latent path hypothesis. Once that hypothesis selects an ordered reflector sequence, the corresponding GraphSLAM factor uses only the selected subset of map variables.</p>'
  )

  source = replaceOptional(
    source,
    `      =\\mathbf h_{h_{ts\\ell}}\\!\\left(
      \\mathbf x_t,\\mathbf m_{j_1},\\ldots,\\mathbf m_{j_k};\\mathcal B_s
      \\right)+\\boldsymbol\\varepsilon_{ts\\ell},
      \\qquad
      \\boldsymbol\\varepsilon_{ts\\ell}\\sim\\mathcal N(\\mathbf 0,\\boldsymbol\\Sigma_{h}).`,
    `      =\\mathbf h\\!\\left(
      \\mathbf x_t,\\mathcal M,h_{ts\\ell};\\mathcal B_s
      \\right)+\\boldsymbol\\varepsilon_{ts\\ell},
      \\qquad
      h_{ts\\ell}\\in\\mathcal H_{ts\\ell}(\\mathcal M),
      \\quad
      \\boldsymbol\\varepsilon_{ts\\ell}\\sim\\mathcal N(\\mathbf 0,\\boldsymbol\\Sigma_{h}).`
  )

  const startMarker = '    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">The three explicit cases</h4>'
  const endMarker = '    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Path-loss component</h4>'
  const replacement = String.raw`    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Map-conditioned model and hypothesis-local measurement functions</h4>
    <p>Let the complete radio map contain all persistent map entities. A path hypothesis selects an <em>ordered tuple</em> from that map:</p>
    <div class="eq math-eq">
      \[
      \mathcal M=\{\mathbf m_j\}_{j=1}^{J},
      \qquad
      \mathcal M_h=\mathcal S_h(\mathcal M)
      =\begin{cases}
      \varnothing,&h=\mathrm{LoS},\\
      (\mathbf m_j),&h=(1,j),\\
      (\mathbf m_{j_1},\mathbf m_{j_2}),&h=(2,j_1,j_2),\\
      (\mathbf m_{j_1},\ldots,\mathbf m_{j_k}),&h=(k,j_{1:k}).
      \end{cases}
      \]
      \[
      \mathbf h(\mathbf x_t,\mathcal M,h;\mathcal B_s)
      \equiv
      \mathbf h_h(\mathbf x_t,\mathcal M_h;\mathcal B_s).
      \]
    </div>
    <p class="eq-note">The left-hand side is the complete map-conditioned model. The right-hand side exposes the sparse geometric computation after \(h\) has selected the active map variables. Candidate generation, path prior, finite-support validity, visibility, and blockage can still depend on the rest of \(\mathcal M\).</p>
    <div class="accuracy"><strong>Global model versus one factor.</strong> Before association, use \(p(\mathbf z_{ts\ell}\mid\mathbf x_t,\mathcal M,\mathcal B_s)\). After conditioning on \(h=(k,j_{1:k})\), a sparse back-end factor normally connects only to the UE state and \(\mathcal M_h\). It would connect to additional map variables only if occlusion or other scene-wide effects were differentiated through the optimizer rather than fixed by the front end.</div>

    <div class="eq math-eq">
      \[
      \mathbf h(\mathbf x_t,\mathcal M,\mathrm{LoS};\mathcal B_s)
      \equiv \mathbf h_0(\mathbf x_t;\mathcal B_s)
      =\begin{bmatrix}
      \|\mathbf p_t-\mathbf b_s\|/c+\delta_t\\
      \operatorname{ang}\!\big(\mathbf R_t^{\mathsf T}(\mathbf b_s-\mathbf p_t)\big)\\
      \operatorname{ang}\!\big(\mathbf R_s^{\mathsf T}(\mathbf p_t-\mathbf b_s)\big)\\
      \widehat{\mathrm{PL}}^{(0)}
      \end{bmatrix},
      \]
      \[
      \mathbf h(\mathbf x_t,\mathcal M,(1,j);\mathcal B_s)
      \equiv \mathbf h_1(\mathbf x_t,\mathbf m_j;\mathcal B_s)
      =\begin{bmatrix}
      \|\mathbf p_t-\mathbf v_s^{(1)}\|/c+\delta_t\\
      \operatorname{ang}\!\big(\mathbf R_t^{\mathsf T}(\mathbf q_1-\mathbf p_t)\big)\\
      \operatorname{ang}\!\big(\mathbf R_s^{\mathsf T}(\mathbf q_1-\mathbf b_s)\big)\\
      \widehat{\mathrm{PL}}^{(1)}
      \end{bmatrix},
      \]
      \[
      \mathbf h(\mathbf x_t,\mathcal M,(2,j_1,j_2);\mathcal B_s)
      \equiv \mathbf h_2(\mathbf x_t,\mathbf m_{j_1},\mathbf m_{j_2};\mathcal B_s)
      =\begin{bmatrix}
      \|\mathbf p_t-\mathbf v_s^{(2)}\|/c+\delta_t\\
      \operatorname{ang}\!\big(\mathbf R_t^{\mathsf T}(\mathbf q_2-\mathbf p_t)\big)\\
      \operatorname{ang}\!\big(\mathbf R_s^{\mathsf T}(\mathbf q_1-\mathbf b_s)\big)\\
      \widehat{\mathrm{PL}}^{(2)}
      \end{bmatrix}.
      \]
    </div>

`
  source = replaceRegion(
    source,
    startMarker,
    endMarker,
    replacement,
    'Map-conditioned model and hypothesis-local measurement functions',
    'the 2D explicit measurement-function block'
  )

  source = replaceOptional(
    source,
    '\\mathbf r^{\\mathrm{rad}}_{ts\\ell}(h)=',
    '\\mathbf r^{\\mathrm{rad}}_{ts\\ell}(\\mathbf x_t,\\mathcal M,h)='
  )
  source = replaceOptional(
    source,
    '\\|\\mathbf r^{\\mathrm{rad}}_{ts\\ell}(a_{ts\\ell})\\|^2_{\\boldsymbol\\Omega_{a_{ts\\ell}}}',
    '\\|\\mathbf r^{\\mathrm{rad}}_{ts\\ell}(\\mathbf x_t,\\mathcal M,a_{ts\\ell})\\|^2_{\\boldsymbol\\Omega_{a_{ts\\ell}}}'
  )
  source = replaceOptional(
    source,
    '+\\sum_{h\\in\\mathcal H_{ts\\ell}}',
    '+\\sum_{h\\in\\mathcal H_{ts\\ell}(\\mathcal M)}'
  )
  source = replaceOptional(
    source,
    '\\mathbf h_h(\\mathbf x_t,\\mathcal M;\\mathcal B_s),',
    '\\mathbf h(\\mathbf x_t,\\mathcal M,h;\\mathcal B_s),'
  )
  source = replaceOptional(
    source,
    '<p class="eq-note">\\(\\kappa\\) is a clutter intensity, \\(\\mathcal H_{ts\\ell}\\) is a gated candidate set, \\(w_h\\) is a prior weight, and \\(p_{\\mathrm D}(h)\\) is path-detection probability. Exact one-to-one assignment is combinatorial; practical systems use front-end association, alternating inference, branching, or sum-/max-mixture factors.</p>',
    '<p class="eq-note">\\(\\kappa\\) is a clutter intensity and \\(\\mathcal H_{ts\\ell}(\\mathcal M)\\) is the gated set of LoS and ordered-reflection hypotheses generated from the complete map. The weight \\(w_h\\), detection probability \\(p_{\\mathrm D}(h)\\), and validity/visibility of each candidate may depend on the whole scene, whereas its conditioned geometric factor uses only \\(\\mathcal M_h\\). Exact one-to-one assignment is combinatorial; practical systems use front-end association, alternating inference, branching, or sum-/max-mixture factors.</p>'
  )
  source = replaceOptional(
    source,
    '<p class="lede">With the BS fixed, the arity of a radio factor grows with bounce count. This is the cleanest way to see how LoS and multipath coexist in one GraphSLAM model.</p>',
    '<p class="lede">The complete measurement likelihood is conditioned on \\(\\mathcal M\\), but after a path hypothesis is selected the corresponding factor touches only the UE state and the selected tuple \\(\\mathcal M_h\\). Its arity therefore grows with bounce count while the global map remains the object being jointly estimated.</p>'
  )

  return source
}

function patch3D(source) {
  source = replaceOptional(
    source,
    '<p class="lede">In three dimensions each angle measurement is an azimuth/elevation pair, or equivalently a unit direction on \\(\\mathbb S^2\\). Unit directions make the coordinate frames and the optimizer residual unambiguous.</p>',
    '<p class="lede">In three dimensions each angle measurement is an azimuth/elevation pair, or equivalently a unit direction on \\(\\mathbb S^2\\). At the scene level the likelihood is conditioned on the complete map \\(\\mathcal M\\); a discrete path hypothesis then selects the ordered reflector tuple used by the sparse geometric factor.</p>'
  )

  const startMarker = '    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">LoS, one-bounce, and two-bounce measurement functions</h4>'
  const endMarker = '    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Direction residual on \\(\\mathbb S^2\\)</h4>'
  const replacement = String.raw`    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Map-conditioned model and hypothesis-local measurement functions</h4>
    <p>Let \(\mathcal M\) contain every persistent plane, finite patch, VA, or other radio-map entity. The path hypothesis \(h=(k,j_{1:k})\) selects the ordered tuple used to construct the current ray:</p>
    <div class="eq math-eq">
      \[
      \mathcal M=\{\mathbf m_j^{\mathrm W}\}_{j=1}^{J},
      \qquad
      \mathcal M_h=\mathcal S_h(\mathcal M)
      =\begin{cases}
      \varnothing,&h=\mathrm{LoS},\\
      (\mathbf m_j^{\mathrm W}),&h=(1,j),\\
      (\mathbf m_{j_1}^{\mathrm W},\mathbf m_{j_2}^{\mathrm W}),&h=(2,j_1,j_2),\\
      (\mathbf m_{j_1}^{\mathrm W},\ldots,\mathbf m_{j_k}^{\mathrm W}),&h=(k,j_{1:k}).
      \end{cases}
      \]
      \[
      \mathbf z_{ts\ell}
      =\mathbf h\!\left(\mathbf x_t,\mathcal M,h_{ts\ell};{}^W\mathbf T_{B_s}\right)
      +\boldsymbol\varepsilon_{ts\ell},
      \qquad
      h_{ts\ell}\in\mathcal H_{ts\ell}(\mathcal M),
      \]
      \[
      \mathbf h(\mathbf x_t,\mathcal M,h;{}^W\mathbf T_{B_s})
      \equiv
      \mathbf h_h(\mathbf x_t,\mathcal M_h;{}^W\mathbf T_{B_s}).
      \]
    </div>
    <p class="eq-note">This notation separates the complete scene-level model from its hypothesis-conditioned sparse computation. The candidate set, path prior, finite-patch validity, visibility, occlusion, and blockage may depend on all of \(\mathcal M\). Once those decisions are conditioned on, the geometric prediction uses only \(\mathcal M_h\).</p>
    <div class="accuracy"><strong>Do not confuse global conditioning with factor adjacency.</strong> Writing \(p(\mathbf z_{ts\ell}\mid\mathbf x_t,\mathcal M,{}^W\mathbf T_{B_s})\) is correct before association. In a standard sparse GraphSLAM back end, a selected one-bounce factor is adjacent only to the UE state and \(\mathbf m_j^{\mathrm W}\), while a selected two-bounce factor is adjacent only to the UE state and the ordered pair \((\mathbf m_{j_1}^{\mathrm W},\mathbf m_{j_2}^{\mathrm W})\). A fully differentiable scene-wide visibility model would create additional dependencies.</div>

    <div class="eq math-eq">
      \[
      \begin{aligned}
      \mathbf h(\mathbf x_t,\mathcal M,\mathrm{LoS};{}^W\mathbf T_{B_s})
      &\equiv\mathbf h_0(\mathbf x_t;{}^W\mathbf T_{B_s})
      =\big[\widehat\tau_0,{}^{U_t}\widehat{\mathbf u}^{\mathrm A}_0,{}^{B_s}\widehat{\mathbf u}^{\mathrm D}_0,\widehat g_0^{\mathrm{dB}}\big],\\
      \mathbf h(\mathbf x_t,\mathcal M,(1,j);{}^W\mathbf T_{B_s})
      &\equiv\mathbf h_1(\mathbf x_t,\mathbf m_j^{\mathrm W};{}^W\mathbf T_{B_s})
      =\big[\widehat\tau_1,{}^{U_t}\widehat{\mathbf u}^{\mathrm A}_1,{}^{B_s}\widehat{\mathbf u}^{\mathrm D}_1,\widehat g_1^{\mathrm{dB}}\big],\\
      \mathbf h(\mathbf x_t,\mathcal M,(2,j_1,j_2);{}^W\mathbf T_{B_s})
      &\equiv\mathbf h_2(\mathbf x_t,\mathbf m_{j_1}^{\mathrm W},\mathbf m_{j_2}^{\mathrm W};{}^W\mathbf T_{B_s})
      =\big[\widehat\tau_2,{}^{U_t}\widehat{\mathbf u}^{\mathrm A}_2,{}^{B_s}\widehat{\mathbf u}^{\mathrm D}_2,\widehat g_2^{\mathrm{dB}}\big].
      \end{aligned}
      \]
    </div>

`
  source = replaceRegion(
    source,
    startMarker,
    endMarker,
    replacement,
    'Map-conditioned model and hypothesis-local measurement functions',
    'the 3D explicit measurement-function block'
  )

  source = replaceOptional(
    source,
    '\\mathbf r^{\\mathrm{rad}}_{ts\\ell}(h)',
    '\\mathbf r^{\\mathrm{rad}}_{ts\\ell}(\\mathbf x_t,\\mathcal M,h)'
  )
  source = replaceOptional(
    source,
    '\\|\\mathbf r_{ts\\ell}^{\\mathrm{rad}}(a_{ts\\ell})\\|^2_{\\boldsymbol\\Sigma^{-1}_{a_{ts\\ell}}}',
    '\\|\\mathbf r_{ts\\ell}^{\\mathrm{rad}}(\\mathbf x_t,\\mathcal M,a_{ts\\ell})\\|^2_{\\boldsymbol\\Sigma^{-1}_{a_{ts\\ell}}}'
  )
  source = replaceOptional(
    source,
    '+\\sum_{h\\in\\mathcal H_{ts\\ell}}',
    '+\\sum_{h\\in\\mathcal H_{ts\\ell}(\\mathcal M)}'
  )
  source = replaceOptional(
    source,
    '\\mathcal N\\!\\left(\n      \\mathbf r^{\\mathrm{rad}}_{ts\\ell}(\\mathbf x_t,\\mathcal M,h);\\mathbf0,\\boldsymbol\\Sigma_h',
    '\\mathcal N\\!\\left(\n      \\mathbf r^{\\mathrm{rad}}_{ts\\ell}(\\mathbf x_t,\\mathcal M,h);\\mathbf0,\\boldsymbol\\Sigma_h'
  )
  source = replaceOptional(
    source,
    '<p class="eq-note">\\(\\kappa_{ts}\\) is clutter intensity, \\(\\mathcal H_{ts\\ell}\\) is a gated set of LoS and ordered-reflection candidates, \\(w_h\\) is a prior hypothesis weight, and \\(p_{\\mathrm D}(h)\\) is path-detection probability. Exact one-to-one association is combinatorial; practical systems use a front end, alternating optimization, branching, marginalization, or mixture/max-mixture factors.</p>',
    '<p class="eq-note">\\(\\kappa_{ts}\\) is clutter intensity and \\(\\mathcal H_{ts\\ell}(\\mathcal M)\\) is the gated set of LoS and ordered-reflection candidates generated from the complete map. The prior weight \\(w_h\\), detection probability \\(p_{\\mathrm D}(h)\\), and validity/visibility of a candidate may depend on all of \\(\\mathcal M\\), while its conditioned geometric residual uses only \\(\\mathcal M_h\\). Exact one-to-one association is combinatorial; practical systems use a front end, alternating optimization, branching, marginalization, or mixture/max-mixture factors.</p>'
  )
  source = replaceOptional(
    source,
    '<p class="lede">The known BS transform is conditioned on. Continuous variables live on a product manifold; bounce order and association are discrete front-end or hybrid-inference variables.</p>',
    '<p class="lede">The known BS transform is conditioned on and the complete likelihood is a function of the whole map \\(\\mathcal M\\). Conditional on one association and bounce-order hypothesis, however, the corresponding sparse factor touches only the selected map tuple \\(\\mathcal M_h\\). Continuous variables live on a product manifold; bounce order and association are discrete front-end or hybrid-inference variables.</p>'
  )

  return source
}

let twoD = readFileSync(twoDPath, 'utf8')
let threeD = readFileSync(threeDPath, 'utf8')
const patchedTwoD = patch2D(twoD)
const patchedThreeD = patch3D(threeD)

for (const [label, text] of [['2D', patchedTwoD], ['3D', patchedThreeD]]) {
  for (const required of [
    'Map-conditioned model and hypothesis-local measurement functions',
    '\\mathcal M_h=\\mathcal S_h(\\mathcal M)',
    '\\mathbf h(\\mathbf x_t,\\mathcal M,h;',
    '\\mathcal H_{ts\\ell}(\\mathcal M)',
    '\\mathbf r^{\\mathrm{rad}}_{ts\\ell}(\\mathbf x_t,\\mathcal M,h)'
  ]) {
    if (!text.includes(required)) throw new Error(`${label} map-conditioned validation failed: ${required}`)
  }
}

if (patchedTwoD !== twoD) writeFileSync(twoDPath, patchedTwoD)
if (patchedThreeD !== threeD) writeFileSync(threeDPath, patchedThreeD)

if (patchedTwoD !== twoD || patchedThreeD !== threeD) {
  console.log('Clarified complete-map conditioning and hypothesis-local factor dependencies in 2D and 3D.')
} else {
  console.log('The 2D and 3D measurement models already distinguish complete-map conditioning from local factors.')
}
