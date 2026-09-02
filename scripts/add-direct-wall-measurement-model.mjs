import { readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

const twoDPath = resolve('scripts/add-radio-slam-problem-formulation.mjs')
const threeDPath = resolve('scripts/upgrade-radio-slam-formulation-3d.mjs')

function insertBeforeRequired(source, marker, block, sentinel, label) {
  if (source.includes(sentinel)) return source
  const index = source.indexOf(marker)
  if (index < 0) throw new Error(`Could not find ${label}`)
  return source.slice(0, index) + block + source.slice(index)
}

function replaceRequired(source, before, after, label) {
  if (source.includes(after)) return source
  if (!source.includes(before)) throw new Error(`Could not find ${label}`)
  return source.replace(before, after)
}

const headingStyle = '<h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">'

const twoDWallBlock = String.raw`    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Direct wall-state measurement model — no VA in the factor definition</h4>
    <p>The same ideal specular path can be defined directly from the physical wall states. In this form the optimization variables are the walls themselves; the physical reflection points are implicit functions of the UE pose, the known BS, and the ordered wall tuple selected from the complete map.</p>
    <div class="eq math-eq">
      \[
      \mathcal M^{\mathrm W}=\{\mathbf m_j^{\mathrm W}\}_{j=1}^{J},
      \qquad
      \mathcal M_h^{\mathrm W}=\mathcal S_h(\mathcal M^{\mathrm W})
      =(\mathbf m_{j_1}^{\mathrm W},\ldots,\mathbf m_{j_k}^{\mathrm W}),
      \]
      \[
      \mathbf q_{1:k}^{\mathrm W}
      =\operatorname{SpecularSolve}\!\left(
      \mathbf b_s,\mathbf p_t;\mathcal M_h^{\mathrm W}
      \right),
      \qquad
      \mathbf q_0=\mathbf b_s,
      \quad
      \mathbf q_{k+1}=\mathbf p_t.
      \]
    </div>
    <p class="eq-note">\(\operatorname{SpecularSolve}\) denotes the geometric operation that returns a valid ordered set of physical reflection points, or rejects the hypothesis when no such path exists. It does not introduce extra optimization variables.</p>

    <div class="eq math-eq">
      \[
      \mathbf q_r\in\mathcal S_{j_r},
      \qquad
      \mathbf u_r^{-}
      =\frac{\mathbf q_r-\mathbf q_{r-1}}
      {\|\mathbf q_r-\mathbf q_{r-1}\|},
      \qquad
      \mathbf u_r^{+}
      =\frac{\mathbf q_{r+1}-\mathbf q_r}
      {\|\mathbf q_{r+1}-\mathbf q_r\|},
      \]
      \[
      \mathbf u_r^{+}
      =\underbrace{\left(\mathbf I-2\mathbf n_{j_r}\mathbf n_{j_r}^{\mathsf T}\right)}_{\mathbf H_{j_r}}
      \mathbf u_r^{-},
      \qquad r=1,\ldots,k,
      \qquad
      \chi_h^{\mathrm W}(\mathbf x_t,\mathcal M^{\mathrm W};\mathcal B_s)=1.
      \]
    </div>
    <p class="eq-note">These equations state the law of specular reflection directly at each wall. The validity indicator additionally enforces finite-segment support, positive ordered legs, and visibility. The sign choice of the wall normal does not matter because \(\mathbf n_j\mathbf n_j^{\mathsf T}\) is unchanged.</p>

    <div class="eq math-eq">
      \[
      L_h^{\mathrm W}
      =\sum_{r=0}^{k}\|\mathbf q_{r+1}^{\mathrm W}-\mathbf q_r^{\mathrm W}\|,
      \]
      \[
      \mathbf h_h^{\mathrm W}(\mathbf x_t,\mathcal M_h^{\mathrm W};\mathcal B_s)
      =\begin{bmatrix}
      L_h^{\mathrm W}/c+\delta_t\\
      \operatorname{ang}\!\left(\mathbf R_t^{\mathsf T}
      (\mathbf q_k^{\mathrm W}-\mathbf q_{k+1}^{\mathrm W})\right)\\
      \operatorname{ang}\!\left(\mathbf R_s^{\mathsf T}
      (\mathbf q_1^{\mathrm W}-\mathbf q_0^{\mathrm W})\right)\\
      \widehat{\mathrm{PL}}_{h}^{\mathrm W}
      \end{bmatrix},
      \qquad
      \mathbf h^{\mathrm W}(\mathbf x_t,\mathcal M^{\mathrm W},h;\mathcal B_s)
      \equiv
      \mathbf h_h^{\mathrm W}(\mathbf x_t,\mathcal M_h^{\mathrm W};\mathcal B_s).
      \]
    </div>
    <p class="eq-note">For LoS, \(k=0\), use \(\mathbf q_0=\mathbf b_s\) and \(\mathbf q_1=\mathbf p_t\); the selected wall tuple is empty. For one bounce, \(\mathbf q_1\) is the wall hit. For two bounces, \(\mathbf q_1\) controls AoD and \(\mathbf q_2\) controls AoA.</p>

    <div class="companion-math-grid">
      <article class="companion-math-card"><h4>LoS · direct wall form</h4><p>\(\mathcal M_h^{\mathrm W}=\varnothing\) and no reflection equation is active:</p><p>\(\mathbf h_0^{\mathrm W}(\mathbf x_t;\mathcal B_s)=\mathbf h_0(\mathbf x_t;\mathcal B_s).\)</p></article>
      <article class="companion-math-card"><h4>One bounce · direct wall form</h4><p>For \(h=(1,j)\), solve directly for \(\mathbf q_1^{\mathrm W}\in\mathcal S_j\), then evaluate</p><p>\(\mathbf h^{\mathrm W}(\mathbf x_t,\mathcal M^{\mathrm W},(1,j);\mathcal B_s)=\mathbf h_1^{\mathrm W}(\mathbf x_t,\mathbf m_j^{\mathrm W};\mathcal B_s).\)</p></article>
      <article class="companion-math-card"><h4>Two bounces · direct wall form</h4><p>For \(h=(2,j_1,j_2)\), solve jointly for \((\mathbf q_1^{\mathrm W},\mathbf q_2^{\mathrm W})\) on the ordered walls, then evaluate</p><p>\(\mathbf h_2^{\mathrm W}(\mathbf x_t,\mathbf m_{j_1}^{\mathrm W},\mathbf m_{j_2}^{\mathrm W};\mathcal B_s).\)</p></article>
      <article class="companion-math-card"><h4>What the factor touches</h4><p>The global likelihood is conditioned on \(\mathcal M^{\mathrm W}\). Once \(h\) is fixed, the sparse factor is adjacent only to the UE state and the selected wall tuple \(\mathcal M_h^{\mathrm W}\).</p></article>
    </div>

    <div class="accuracy"><strong>Same ideal physics, two useful forms.</strong> When each VA is generated by reflecting the known BS through the selected physical walls and the resulting path is valid, the image-source and direct-wall predictions agree: \(\mathbf h_h^{\mathrm{VA}}=\mathbf h_h^{\mathrm W}\). The VA form is usually the faster geometric evaluation; the wall-direct form makes the estimated physical map, finite support, reflection law, and factor adjacency explicit.</div>

`

const threeDWallBlock = String.raw`    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Direct planar-wall measurement model — no VA in the factor definition</h4>
    <p>The same path can be formulated directly with the physical plane or finite-patch states. For a selected ordered wall hypothesis \(h=(k,j_{1:k})\), the bounce points are determined by the wall constraints and the specular reflection law. They are deterministic intermediate quantities, not additional graph variables.</p>
    <div class="eq math-eq">
      \[
      \mathcal M^{\mathrm W}=\{\mathbf m_j^{\mathrm W}\}_{j=1}^{J},
      \qquad
      \mathcal M_h^{\mathrm W}
      =\mathcal S_h(\mathcal M^{\mathrm W})
      =(\mathbf m_{j_1}^{\mathrm W},\ldots,\mathbf m_{j_k}^{\mathrm W}),
      \]
      \[
      \mathbf q_{1:k}^{\mathrm W}
      =\operatorname{SpecularSolve}\!\left(
      {}^W\mathbf b_s,{}^W\mathbf p_{U_t};\mathcal M_h^{\mathrm W}
      \right),
      \qquad
      \mathbf q_0={}^W\mathbf b_s,
      \quad
      \mathbf q_{k+1}={}^W\mathbf p_{U_t}.
      \]
    </div>
    <p class="eq-note">\(\operatorname{SpecularSolve}\) returns the physical reflection-point branch satisfying all of the equations below, or rejects the candidate. For flat planes it may be implemented efficiently with image sources, but its state arguments remain the physical walls.</p>

    <div class="eq math-eq">
      \[
      \mathbf q_r\in\mathcal S_{j_r}\subset\Pi_{j_r},
      \qquad
      \mathbf n_{j_r}^{\mathsf T}\mathbf q_r=d_{j_r},
      \]
      \[
      \mathbf u_r^{-}
      =\frac{\mathbf q_r-\mathbf q_{r-1}}
      {\|\mathbf q_r-\mathbf q_{r-1}\|},
      \qquad
      \mathbf u_r^{+}
      =\frac{\mathbf q_{r+1}-\mathbf q_r}
      {\|\mathbf q_{r+1}-\mathbf q_r\|},
      \]
      \[
      \mathbf u_r^{+}
      =\underbrace{\left(\mathbf I-2\mathbf n_{j_r}\mathbf n_{j_r}^{\mathsf T}\right)}_{\mathbf H_{j_r}}
      \mathbf u_r^{-},
      \qquad r=1,\ldots,k,
      \qquad
      \chi_h^{\mathrm W}(\mathbf x_t,\mathcal M^{\mathrm W};{}^W\mathbf T_{B_s})=1.
      \]
    </div>
    <p class="eq-note">The last equation is the vector form of equal incidence and reflection angles. The validity indicator enforces finite-patch support, the prescribed interaction order, positive segment lengths, and visibility of every physical leg.</p>

    <div class="eq math-eq">
      \[
      L_h^{\mathrm W}
      =\sum_{r=0}^{k}\|\mathbf q_{r+1}^{\mathrm W}-\mathbf q_r^{\mathrm W}\|,
      \]
      \[
      \mathbf h_h^{\mathrm W}(\mathbf x_t,\mathcal M_h^{\mathrm W};{}^W\mathbf T_{B_s})
      =\begin{bmatrix}
      L_h^{\mathrm W}/c+\delta_t\\[.25em]
      ({}^{W}\mathbf R_{U_t})^{\mathsf T}
      \dfrac{\mathbf q_k^{\mathrm W}-\mathbf q_{k+1}^{\mathrm W}}
      {\|\mathbf q_k^{\mathrm W}-\mathbf q_{k+1}^{\mathrm W}\|}\\[.7em]
      ({}^{W}\mathbf R_{B_s})^{\mathsf T}
      \dfrac{\mathbf q_1^{\mathrm W}-\mathbf q_0^{\mathrm W}}
      {\|\mathbf q_1^{\mathrm W}-\mathbf q_0^{\mathrm W}\|}\\[.7em]
      \widehat g_{h,\mathrm W}^{\mathrm{dB}}
      \end{bmatrix},
      \]
      \[
      \mathbf h^{\mathrm W}(\mathbf x_t,\mathcal M^{\mathrm W},h;{}^W\mathbf T_{B_s})
      \equiv
      \mathbf h_h^{\mathrm W}(\mathbf x_t,\mathcal M_h^{\mathrm W};{}^W\mathbf T_{B_s}).
      \]
    </div>
    <p class="eq-note">The second row is the UE-frame AoA unit vector and the third row is the BS-frame AoD unit vector. For LoS, \(k=0\), set \(\mathbf q_0={}^W\mathbf b_s\) and \(\mathbf q_1={}^W\mathbf p_{U_t}\), so the same formula gives UE→BS AoA and BS→UE AoD.</p>

    <div class="companion-math-grid">
      <article class="companion-math-card"><h4>LoS · wall-direct model</h4><p>\(\mathcal M_h^{\mathrm W}=\varnothing\). The measurement depends on the UE state and known BS; the complete map still determines whether the direct segment is visible.</p></article>
      <article class="companion-math-card"><h4>One bounce · wall-direct model</h4><p>For \(h=(1,j)\), solve \(\mathbf q_1^{\mathrm W}\in\mathcal S_j\) and evaluate</p><p>\(\mathbf h_1^{\mathrm W}(\mathbf x_t,\mathbf m_j^{\mathrm W};{}^W\mathbf T_{B_s}).\)</p></article>
      <article class="companion-math-card"><h4>Two bounces · wall-direct model</h4><p>For \(h=(2,j_1,j_2)\), solve the coupled equations for \((\mathbf q_1^{\mathrm W},\mathbf q_2^{\mathrm W})\) and evaluate</p><p>\(\mathbf h_2^{\mathrm W}(\mathbf x_t,\mathbf m_{j_1}^{\mathrm W},\mathbf m_{j_2}^{\mathrm W};{}^W\mathbf T_{B_s}).\)</p></article>
      <article class="companion-math-card"><h4>Radiometric term</h4><p>\(\widehat g_{h,\mathrm W}^{\mathrm{dB}}\) is evaluated from the physical leg lengths, incidence angles, wall materials, antenna patterns, polarization, and blockage—not from bounce count alone.</p></article>
    </div>

    <div class="accuracy"><strong>VA and wall formulations are complementary.</strong> If the VA or image-source chain is deterministically generated from \(\mathcal M_h^{\mathrm W}\), then a valid ideal-specular path satisfies \(\mathbf h_h^{\mathrm{VA}}=\mathbf h_h^{\mathrm W}\). The VA form gives a compact and efficient evaluation; the wall-direct form is preferable when the output map must contain physical planes shared across poses or BSs, when finite support matters, or when multi-bounce AoD must remain physically interpretable.</div>

`

let twoD = readFileSync(twoDPath, 'utf8')
const originalTwoD = twoD

twoD = replaceRequired(
  twoD,
  `${headingStyle}Map-conditioned model and hypothesis-local measurement functions</h4>`,
  `${headingStyle}Map-conditioned VA / image-source measurement model</h4>\n    <p>The equations in this first form retain the virtual-anchor or image-source construction. The complete map supplies the selected walls or VAs, and the factor evaluates the corresponding unfolded path.</p>`,
  '2D VA/image-source heading'
)
twoD = insertBeforeRequired(
  twoD,
  `${headingStyle}Path-loss component</h4>`,
  twoDWallBlock,
  'Direct wall-state measurement model — no VA in the factor definition',
  '2D path-loss insertion marker'
)

let threeD = readFileSync(threeDPath, 'utf8')
const originalThreeD = threeD

threeD = replaceRequired(
  threeD,
  `${headingStyle}Map-conditioned model and hypothesis-local measurement functions</h4>`,
  `${headingStyle}Map-conditioned VA / image-source measurement model</h4>\n    <p>This first formulation keeps the virtual-anchor or image-source computation already used above. The selected physical walls may still be the optimized states; the VAs are then deterministic intermediate quantities used to unfold the path.</p>`,
  '3D VA/image-source heading'
)
threeD = insertBeforeRequired(
  threeD,
  `${headingStyle}Direction residual on \\(\\mathbb S^2\\)</h4>`,
  threeDWallBlock,
  'Direct planar-wall measurement model — no VA in the factor definition',
  '3D direction-residual insertion marker'
)

for (const [source, required, label] of [
  [twoD, 'mathbf h_h^{\\mathrm W}(\\mathbf x_t,\\mathcal M_h^{\\mathrm W};\\mathcal B_s)', '2D wall-direct measurement function'],
  [twoD, 'mathbf h_h^{\\mathrm{VA}}=\\mathbf h_h^{\\mathrm W}', '2D VA/wall equivalence'],
  [threeD, 'mathbf h_h^{\\mathrm W}(\\mathbf x_t,\\mathcal M_h^{\\mathrm W};{}^W\\mathbf T_{B_s})', '3D wall-direct measurement function'],
  [threeD, 'mathbf u_r^{+}', '3D reflection law'],
  [threeD, 'mathbf h_h^{\\mathrm{VA}}=\\mathbf h_h^{\\mathrm W}', '3D VA/wall equivalence']
]) {
  if (!source.includes(required)) throw new Error(`Validation failed: ${label}`)
}

if (twoD !== originalTwoD) writeFileSync(twoDPath, twoD)
if (threeD !== originalThreeD) writeFileSync(threeDPath, threeD)

if (twoD !== originalTwoD || threeD !== originalThreeD) {
  console.log('Added direct physical-wall measurement models while retaining the VA/image-source forms.')
} else {
  console.log('The VA/image-source and direct wall-state measurement models are already present.')
}
