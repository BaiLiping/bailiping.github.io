import { readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

const pagePath = resolve('mpc-detection-to-bounce-count/index.html')
let html = readFileSync(pagePath, 'utf8')
let changed = false

function replaceOptional(before, after) {
  if (html.includes(after) || !html.includes(before)) return
  html = html.replace(before, after)
  changed = true
}

function replaceBlock(startMarker, endMarker, replacement, sentinel, label) {
  if (html.includes(sentinel)) return
  const start = html.indexOf(startMarker)
  const end = html.indexOf(endMarker, start + startMarker.length)
  if (start < 0 || end < 0) throw new Error(`Could not find ${label}`)
  html = html.slice(0, start) + replacement + html.slice(end)
  changed = true
}

replaceOptional(
  '<a href="#problem-formulation">Problem formulation</a>',
  '<a href="#problem-formulation">3D formulation</a>'
)

replaceOptional(
  '<!-- ============ 05 bistatic radio to GraphSLAM ============ -->',
  '<!-- ============ 06 bistatic radio to GraphSLAM ============ -->'
)

replaceOptional(
  '<tr><td>\\(\\mathbf B_s=(\\mathbf b_s,\\theta_s)\\)</td><td>known or calibrated</td><td>BS position and array heading. Promote it to a graph variable only when uncertain.</td></tr>',
  '<tr><td>\\({}^{W}\\mathbf T_{B_s}\\in SE(3)\\)</td><td>known or calibrated</td><td>Known 3D BS pose: array orientation and position. If the installation is level, a known yaw/heading plus fixed roll and pitch determines the rotation.</td></tr>'
)
replaceOptional(
  '<tr><td>\\(\\mathbf T_t=(\\mathbf p_t,\\theta_t)\\)</td><td>unknown UE pose</td><td>UE position and orientation in 2D; use an \\(SE(3)\\) pose in 3D.</td></tr>',
  '<tr><td>\\({}^{W}\\mathbf T_{U_t}\\in SE(3)\\)</td><td>unknown UE pose</td><td>Full 3D UE pose: \\(SO(3)\\) orientation and \\(\\mathbb R^3\\) position. The planar demos are visual cross-sections of this state.</td></tr>'
)

const formulation3D = String.raw`
<!-- ============ 05 unified 3D radio-SLAM problem formulation ============ -->
<section class="sec companion-section" id="problem-formulation" data-formulation-dimension="3d">
  <h2><span class="no">05</span>A complete 3D radio-SLAM problem formulation</h2>
  <p class="lede">The formal estimation problem is three-dimensional. The base-station pose and array orientation are known; the continuous unknowns are the UE trajectory, optional clock states, and a persistent map of planar reflectors or virtual anchors. Every detected MPC also carries a discrete explanation: clutter, LoS, one reflection, two ordered reflections, and so on. The interactive drawings above remain useful 2D cross-sections of this 3D model.</p>

  <div class="accuracy"><strong>Recommended map state.</strong> Optimize physical finite planes when the goal is to recover the propagation environment and use delay, AoA, and AoD jointly. Compute virtual anchors and physical bounce points deterministically inside each factor. A directly optimized VA is compact and valid for a one-bounce path from one BS, but one final composite-VA point is generally insufficient to recover all constituent planes or predict the first departure direction of a multi-bounce path.</div>

  <nav class="subsection-tiles is-four" aria-label="Three-dimensional radio SLAM problem-formulation subsections">
    <a href="#formulation-state"><span>5.1</span><strong>3D state</strong><small>UE pose · known BS · plane · VA</small></a>
    <a href="#formulation-hypothesis"><span>5.2</span><strong>Path geometry</strong><small>LoS · one bounce · two bounce</small></a>
    <a href="#formulation-measurement"><span>5.3</span><strong>Measurement model</strong><small>delay · spherical AoA/AoD · gain</small></a>
    <a href="#formulation-map"><span>5.4</span><strong>Joint inference</strong><small>factors · association · manifold MAP</small></a>
  </nav>

  <div class="subsection-block" id="formulation-state">
    <h3 class="subh"><span class="no">5.1</span>Continuous state in 3D</h3>
    <p class="lede">Let \(F^W\) be the world/map frame, \(F^{U_t}\) the UE body/array frame at time \(t\), and \(F^{B_s}\) the array frame of BS \(s\). The transform \({}^{W}\mathbf T_{U_t}\) maps coordinates expressed in the UE frame into the world frame.</p>

    <h4 style="margin:20px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">UE pose and clock state</h4>
    <div class="eq math-eq">
      \[
      {}^{W}\mathbf T_{U_t}
      =\begin{bmatrix}
      {}^{W}\mathbf R_{U_t} & {}^{W}\mathbf p_{U_t}\\
      \mathbf 0^{\mathsf T} & 1
      \end{bmatrix}\in SE(3),
      \qquad
      {}^{W}\mathbf R_{U_t}\in SO(3),
      \quad {}^{W}\mathbf p_{U_t}\in\mathbb R^3,
      \]
      \[
      \mathbf x_t=\big({}^{W}\mathbf T_{U_t},\delta_t\big)
      \in SE(3)\times\mathbb R,
      \qquad
      \mathcal X=\{\mathbf x_t\}_{t=0}^{T}.
      \]
    </div>
    <table class="companion-symbols">
      <thead><tr><th>Quantity</th><th>Status</th><th>Meaning</th></tr></thead>
      <tbody>
        <tr><td>\({}^{W}\mathbf p_{U_t}\)</td><td>unknown</td><td>UE antenna-reference-point position in the world frame.</td></tr>
        <tr><td>\({}^{W}\mathbf R_{U_t}\)</td><td>unknown or aided</td><td>Full roll–pitch–yaw orientation of the UE array. AoA is measured in \(F^{U_t}\), so this orientation is needed to relate AoA to the map.</td></tr>
        <tr><td>\(\delta_t\)</td><td>optional unknown</td><td>UE clock plus residual hardware-delay offset relative to the calibrated BS time reference, in seconds. Set it to zero only for a synchronized and calibrated system.</td></tr>
        <tr><td>\({}^{W}\mathbf v_t,{}^{U_t}\boldsymbol\omega_t,\dot\delta_t\)</td><td>optional extension</td><td>Velocity, angular velocity, and clock drift may be added when an IMU, motion model, spline, or continuous-time prior uses them. They are not required by the radio likelihood itself.</td></tr>
      </tbody>
    </table>

    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Known base-station pose</h4>
    <div class="eq math-eq">
      \[
      {}^{W}\mathbf T_{B_s}
      =\begin{bmatrix}
      {}^{W}\mathbf R_{B_s} & {}^{W}\mathbf b_s\\
      \mathbf 0^{\mathsf T} & 1
      \end{bmatrix}\in SE(3),
      \qquad
      {}^{W}\mathbf T_{B_s}\;\text{is known}.
      \]
    </div>
    <p class="eq-note">A 3D AoD measurement needs the array orientation, not only the BS position. If the installed array is level and its roll and pitch are fixed, the surveyed yaw/heading completes \({}^{W}\mathbf R_{B_s}\). Otherwise the full orientation must be calibrated. Known BS quantities are arguments of a factor, not optimized variable nodes.</p>

    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Plane and finite-wall states</h4>
    <div class="eq math-eq">
      \[
      \boldsymbol\pi_j=(\mathbf n_j,d_j)\in\mathbb S^2\times\mathbb R,
      \qquad
      \Pi_j=\{\mathbf y\in\mathbb R^3:\mathbf n_j^{\mathsf T}\mathbf y=d_j\},
      \qquad \|\mathbf n_j\|=1.
      \]
    </div>
    <p class="eq-note">The pair \((\mathbf n_j,d_j)\) is the minimal three-degree-of-freedom state of an infinite plane. The representation has the sign symmetry \((\mathbf n_j,d_j)\equiv(-\mathbf n_j,-d_j)\).</p>
    <div class="eq math-eq">
      \[
      \mathbf m_j^{\mathrm W}
      =\big({}^{W}\mathbf T_{W_j},a_j,b_j,\boldsymbol\xi_j\big),
      \qquad
      {}^{W}\mathbf T_{W_j}
      =\begin{bmatrix}{}^{W}\mathbf R_{W_j}&{}^{W}\mathbf c_j\\\mathbf0^{\mathsf T}&1\end{bmatrix},
      \]
      \[
      \mathcal S_j=
      \left\{{}^{W}\mathbf c_j+{}^{W}\mathbf R_{W_j}
      \begin{bmatrix}u\\v\\0\end{bmatrix}:
      |u|\le a_j,\ |v|\le b_j\right\},
      \qquad
      \mathbf n_j={}^W\mathbf R_{W_j}\mathbf e_3,
      \quad d_j=\mathbf n_j^{\mathsf T}{}^W\mathbf c_j.
      \]
    </div>
    <p class="eq-note">For a rectangular reflector, \({}^{W}\mathbf c_j\) is its center, the first two columns of \({}^{W}\mathbf R_{W_j}\) orient its edges, \(a_j,b_j\) are half-extents, and \(\boldsymbol\xi_j\) contains optional material or roughness parameters. Unlike an infinite plane, rotation about the normal matters for finite support.</p>

    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Reflection and virtual anchor</h4>
    <div class="eq math-eq">
      \[
      \mathcal R_j(\mathbf y)
      =\mathbf H_j\mathbf y+2d_j\mathbf n_j,
      \qquad
      \mathbf H_j=\mathbf I-2\mathbf n_j\mathbf n_j^{\mathsf T},
      \qquad
      \mathbf H_j\in O(3),\quad\det\mathbf H_j=-1,
      \]
      \[
      \mathbf v_{sj}=\mathcal R_j({}^{W}\mathbf b_s)\in\mathbb R^3.
      \]
    </div>
    <p class="eq-note">A specular reflection is an improper orthogonal transform: it is in \(O(3)\), not \(SO(3)\). The VA itself is a Euclidean point, not a pose or a rotation.</p>
    <div class="companion-math-grid">
      <article class="companion-math-card"><h4>Direct one-bounce VA state</h4><p>For one known BS, an estimator may store \(\mathbf m^{\mathrm{VA}}_{sj}=\mathbf v_{sj}\in\mathbb R^3\). The associated infinite plane is recovered from</p><p>\(\mathbf n_{sj}=\dfrac{\mathbf v_{sj}-{}^{W}\mathbf b_s}{\|\mathbf v_{sj}-{}^{W}\mathbf b_s\|},\quad d_{sj}=\tfrac12\mathbf n_{sj}^{\mathsf T}(\mathbf v_{sj}+{}^{W}\mathbf b_s).\)</p></article>
      <article class="companion-math-card"><h4>Why walls are preferred for \(k\ge2\)</h4><p>A final composite image point predicts total length and the final unfolded arrival bearing, but generally does not identify each physical plane, finite support, reflection point, or initial departure direction. Keep the ordered wall states—or the full affine reflection-chain transform—when AoD and a physical map matter.</p></article>
    </div>

    <table class="companion-symbols">
      <thead><tr><th>Object</th><th>Space</th><th>Role in the formulation</th></tr></thead>
      <tbody>
        <tr><td>UE / BS pose</td><td>\(SE(3)\)</td><td>Rigid transformation: translation plus proper rotation.</td></tr>
        <tr><td>Array orientation</td><td>\(SO(3)\)</td><td>Maps local AoA/AoD direction vectors into the world frame.</td></tr>
        <tr><td>Plane normal / direction</td><td>\(\mathbb S^2\)</td><td>Unit vector on the sphere; not itself an \(SO(3)\) state.</td></tr>
        <tr><td>VA / point scatterer</td><td>\(\mathbb R^3\)</td><td>Euclidean point landmark.</td></tr>
        <tr><td>Single reflection matrix</td><td>\(O(3)\setminus SO(3)\)</td><td>Orientation-reversing operator with determinant \(-1\).</td></tr>
        <tr><td>Local pose increment</td><td>\(\mathfrak{se}(3)\simeq\mathbb R^6\)</td><td>Tangent-space variable used by Gauss–Newton, LM, and iSAM2.</td></tr>
      </tbody>
    </table>
  </div>

  <div class="subsection-block" id="formulation-hypothesis">
    <h3 class="subh"><span class="no">5.2</span>LoS and ordered specular paths in 3D</h3>
    <p class="lede">For MPC \(\ell\) observed at UE time \(t\) from BS \(s\), let the hypothesis state the bounce count and the ordered sequence of planes.</p>
    <div class="eq math-eq">
      \[
      h_{ts\ell}=\big(k_{ts\ell},\mathbf j_{ts\ell}\big),
      \qquad
      k_{ts\ell}\in\{0,1,2,\ldots,K_{\max}\},
      \qquad
      \mathbf j_{ts\ell}=(j_1,\ldots,j_k).
      \]
    </div>
    <p class="eq-note">\(k=0\) is LoS and uses the empty wall sequence. For two bounces, \((j_1,j_2)\) and \((j_2,j_1)\) are generally different: AoD is set by the first physical leg and AoA by the last.</p>

    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Line–plane intersection</h4>
    <div class="eq math-eq">
      \[
      \operatorname{LP}(\mathbf a,\mathbf b;\boldsymbol\pi_j)
      =\mathbf a+\lambda_j(\mathbf b-\mathbf a),
      \qquad
      \lambda_j=\frac{d_j-\mathbf n_j^{\mathsf T}\mathbf a}
      {\mathbf n_j^{\mathsf T}(\mathbf b-\mathbf a)}.
      \]
    </div>
    <p class="eq-note">The denominator must be nonzero. For a finite wall, the resulting point must also lie inside \(\mathcal S_j\).</p>

    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Unfold, then fold back</h4>
    <div class="eq math-eq">
      \[
      \mathbf v_s^{(0)}={}^W\mathbf b_s,
      \qquad
      \mathbf v_s^{(r)}=\mathcal R_{j_r}\!\left(\mathbf v_s^{(r-1)}\right),
      \quad r=1,\ldots,k,
      \]
      \[
      \mathbf q_{k+1}={}^W\mathbf p_{U_t},
      \qquad
      \mathbf q_r=\operatorname{LP}\!\left(\mathbf q_{r+1},\mathbf v_s^{(r)};\boldsymbol\pi_{j_r}\right),
      \quad r=k,k-1,\ldots,1,
      \qquad
      \mathbf q_0={}^W\mathbf b_s.
      \]
    </div>
    <p class="eq-note">The image sources are created in transmitter-to-receiver wall order. The physical bounce points are recovered in reverse order. The notation \(\mathbf q_0,\ldots,\mathbf q_{k+1}\) always lists the physical path from BS to UE.</p>

    <div class="companion-math-grid">
      <article class="companion-math-card"><h4>LoS · \(h_0\)</h4><p>\(k=0\), no wall state:</p><p>\(\mathbf q_0={}^W\mathbf b_s,\quad \mathbf q_1={}^W\mathbf p_{U_t},\)</p><p>\(L_0=\|{}^W\mathbf p_{U_t}-{}^W\mathbf b_s\|.\)</p></article>
      <article class="companion-math-card"><h4>One bounce · \(h_1(j)\)</h4><p>\(\mathbf v_s^{(1)}=\mathcal R_j({}^W\mathbf b_s),\)</p><p>\(\mathbf q_1=\operatorname{LP}({}^W\mathbf p_{U_t},\mathbf v_s^{(1)};\boldsymbol\pi_j),\)</p><p>\(L_1=\|{}^W\mathbf p_{U_t}-\mathbf v_s^{(1)}\|.\)</p></article>
      <article class="companion-math-card"><h4>Two bounces · \(h_2(j_1,j_2)\)</h4><p>\(\mathbf v_s^{(1)}=\mathcal R_{j_1}({}^W\mathbf b_s),\quad \mathbf v_s^{(2)}=\mathcal R_{j_2}(\mathbf v_s^{(1)}),\)</p><p>\(\mathbf q_2=\operatorname{LP}({}^W\mathbf p_{U_t},\mathbf v_s^{(2)};\boldsymbol\pi_{j_2}),\)</p><p>\(\mathbf q_1=\operatorname{LP}(\mathbf q_2,\mathbf v_s^{(1)};\boldsymbol\pi_{j_1}),\)</p><p>\(L_2=\|{}^W\mathbf p_{U_t}-\mathbf v_s^{(2)}\|.\)</p></article>
      <article class="companion-math-card"><h4>Validity indicator</h4><p>\(\chi_h(\mathbf x_t,\mathcal M;{}^W\mathbf T_{B_s})=1\) only if every required intersection exists, every \(\mathbf q_r\) lies on its finite patch, the legs occur in the stated order with positive length, and no physical leg is blocked.</p></article>
    </div>

    <div class="eq math-eq">
      \[
      L_h
      =\sum_{r=0}^{k}\|\mathbf q_{r+1}-\mathbf q_r\|
      =\|{}^W\mathbf p_{U_t}-\mathbf v_s^{(k)}\|
      \qquad\text{for a valid ideal specular path.}
      \]
    </div>
  </div>

  <div class="subsection-block" id="formulation-measurement">
    <h3 class="subh"><span class="no">5.3</span>3D delay, AoA, AoD, and path-gain measurement model</h3>
    <p class="lede">In three dimensions each angle measurement is an azimuth/elevation pair, or equivalently a unit direction on \(\mathbb S^2\). At the scene level the likelihood is conditioned on the complete map \(\mathcal M\); a discrete path hypothesis then selects the ordered reflector tuple used by the sparse geometric factor.</p>

    <div class="eq math-eq">
      \[
      \mathbf z_{ts\ell}
      =\begin{bmatrix}
      \tau_{ts\ell} &
      \varphi^{U,\mathrm{az}}_{ts\ell} & \varphi^{U,\mathrm{el}}_{ts\ell} &
      \psi^{B,\mathrm{az}}_{ts\ell} & \psi^{B,\mathrm{el}}_{ts\ell} &
      g^{\mathrm{dB}}_{ts\ell}
      \end{bmatrix}^{\mathsf T},
      \]
      \[
      \operatorname{sph}(\alpha,\beta)
      =\begin{bmatrix}
      \cos\beta\cos\alpha\\
      \cos\beta\sin\alpha\\
      \sin\beta
      \end{bmatrix},
      \quad
      {}^{U_t}\mathbf u^{\mathrm A}_{ts\ell}=\operatorname{sph}(\varphi^{U,\mathrm{az}},\varphi^{U,\mathrm{el}}),
      \quad
      {}^{B_s}\mathbf u^{\mathrm D}_{ts\ell}=\operatorname{sph}(\psi^{B,\mathrm{az}},\psi^{B,\mathrm{el}}).
      \]
    </div>
    <p class="eq-note">Superscript A denotes AoA and D denotes AoD. This page defines AoA as the direction from the UE toward the previous interaction/source. A forward propagation-vector convention would reverse that unit vector.</p>

    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Prediction under any valid hypothesis \(h\)</h4>
    <div class="eq math-eq">
      \[
      \widehat\tau_h=\frac{L_h}{c}+\delta_t,
      \]
      \[
      {}^{B_s}\widehat{\mathbf u}^{\mathrm D}_h
      =({}^{W}\mathbf R_{B_s})^{\mathsf T}
      \frac{\mathbf q_1-\mathbf q_0}{\|\mathbf q_1-\mathbf q_0\|},
      \qquad
      {}^{U_t}\widehat{\mathbf u}^{\mathrm A}_h
      =({}^{W}\mathbf R_{U_t})^{\mathsf T}
      \frac{\mathbf q_k-\mathbf q_{k+1}}{\|\mathbf q_k-\mathbf q_{k+1}\|}.
      \]
    </div>
    <p class="eq-note">The formulas also cover LoS: when \(k=0\), \(\mathbf q_0={}^W\mathbf b_s\), \(\mathbf q_1={}^W\mathbf p_{U_t}\), and the AoA numerator is \(\mathbf q_0-\mathbf q_1\). AoD therefore points BS→UE, while AoA points UE→BS.</p>

    <div class="eq math-eq">
      \[
      \widehat g_h^{\mathrm{dB}}
      =\gamma_h\!\left(
      \mathbf q_{0:k+1},\mathbf n_{j_{1:k}},\boldsymbol\xi_{j_{1:k}},
      f_c,\text{array patterns},\text{polarization},\text{blockage}
      \right).
      \]
    </div>
    <p class="eq-note">\(\gamma_h\) is a calibrated radiometric model. Geometry alone does not determine path loss. When transmit power, antenna patterns, material coefficients, or receiver calibration are unavailable, omit this residual from the geometric optimizer and use gain only for soft gating or association.</p>

    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Map-conditioned VA / image-source measurement model</h4>
    <p>This first formulation keeps the virtual-anchor or image-source computation already used above. The selected physical walls may still be the optimized states; the VAs are then deterministic intermediate quantities used to unfold the path.</p>
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

        <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Direct planar-wall measurement model — no VA in the factor definition</h4>
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

<h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Direction residual on \(\mathbb S^2\)</h4>
    <p>Do not subtract 3D azimuth and elevation as ordinary Euclidean coordinates near wrap boundaries or the poles. For measured \(\mathbf u\) and predicted \(\widehat{\mathbf u}\), define</p>
    <div class="eq math-eq">
      \[
      \vartheta=\arccos\!\big(\operatorname{clip}(\widehat{\mathbf u}^{\mathsf T}\mathbf u,-1,1)\big),
      \qquad
      \operatorname{Log}_{\mathbb S^2,\widehat{\mathbf u}}(\mathbf u)
      =\frac{\vartheta}{\sin\vartheta}
      \left(\mathbf u-\cos\vartheta\,\widehat{\mathbf u}\right)
      \in T_{\widehat{\mathbf u}}\mathbb S^2.
      \]
    </div>
    <p class="eq-note">Use the continuous limit at \(\vartheta=0\). If \(\mathbf E(\widehat{\mathbf u})\in\mathbb R^{3\times2}\) is an orthonormal tangent basis, the minimal residual is \(\mathbf r_{\mathbb S^2}=\mathbf E^{\mathsf T}\operatorname{Log}_{\mathbb S^2,\widehat{\mathbf u}}(\mathbf u)\in\mathbb R^2\).</p>

    <div class="eq math-eq">
      \[
      \mathbf r^{\mathrm{rad}}_{ts\ell}(\mathbf x_t,\mathcal M,h)
      =\begin{bmatrix}
      c\big(\tau_{ts\ell}-\widehat\tau_h\big)\\
      \mathbf r_{\mathbb S^2}\!\left({}^{U_t}\mathbf u^{\mathrm A}_{ts\ell},{}^{U_t}\widehat{\mathbf u}^{\mathrm A}_h\right)\\
      \mathbf r_{\mathbb S^2}\!\left({}^{B_s}\mathbf u^{\mathrm D}_{ts\ell},{}^{B_s}\widehat{\mathbf u}^{\mathrm D}_h\right)\\
      g^{\mathrm{dB}}_{ts\ell}-\widehat g_h^{\mathrm{dB}}
      \end{bmatrix},
      \qquad
      \mathbf r^{\mathrm{rad}}_{ts\ell}(\mathbf x_t,\mathcal M,h)\sim\mathcal N(\mathbf0,\boldsymbol\Sigma_h).
      \]
    </div>
    <p class="eq-note">With gain included this is a six-dimensional local residual: one range-equivalent delay component, two AoA tangent coordinates, two AoD tangent coordinates, and one gain component. The covariance \(\boldsymbol\Sigma_h\) must use the same coordinates and should preserve delay–direction correlations provided by the channel estimator. If the estimator supplies azimuth/elevation coordinates, first use the local residual Jacobian \(J\): \(\boldsymbol\Sigma_h\approx J\boldsymbol\Sigma_zJ^{\mathsf T}\). If directions are already in the chosen tangent bases, the delay-unit conversion is exactly \(D\boldsymbol\Sigma_zD^{\mathsf T}\), with \(D=\operatorname{diag}(c,1,\ldots,1)\). The spherical logarithm has domain \(0\le\theta<\pi\), with its continuous zero-angle limit. At the antipode \(\theta=\pi\), its direction is not unique; gate out that candidate or use an explicitly chosen alternative residual. Clamp dot products to \([-1,1]\) before numerical arccos evaluation.</p>
  </div>

  <div class="subsection-block" id="formulation-map">
    <h3 class="subh"><span class="no">5.4</span>Factor graph and joint manifold MAP inference</h3>
    <p class="lede">The known BS transform is conditioned on and the complete likelihood is a function of the whole map \(\mathcal M\). Conditional on one association and bounce-order hypothesis, however, the corresponding sparse factor touches only the selected map tuple \(\mathcal M_h\). Continuous variables live on a product manifold; bounce order and association are discrete front-end or hybrid-inference variables.</p>

    <div class="eq math-eq">
      \[
      \boldsymbol\Theta
      =\left\{
      ({}^{W}\mathbf T_{U_t},\delta_t)_{t=0}^{T},
      (\mathbf m_j^{\mathrm W})_{j=1}^{J},
      \text{optional calibration states}
      \right\},
      \]
      \[
      a_{ts\ell}\in
      \left\{0,\mathrm{LoS},(1,j),(2,j_1,j_2),\ldots\right\}.
      \]
    </div>
    <p class="eq-note">\(a=0\) denotes clutter. Otherwise the assignment selects both a bounce count and an ordered surface sequence. The per-scan MPC index \(\ell\) is not a persistent map identity.</p>

    <table class="companion-symbols">
      <thead><tr><th>Hypothesis</th><th>Unknown continuous nodes touched</th><th>Known input</th></tr></thead>
      <tbody>
        <tr><td>clutter / false alarm</td><td>none, apart from a latent assignment variable</td><td>clutter intensity over measurement space</td></tr>
        <tr><td>LoS, \(k=0\)</td><td>\(({}^{W}\mathbf T_{U_t},\delta_t)\)</td><td>\({}^{W}\mathbf T_{B_s}\)</td></tr>
        <tr><td>one bounce, \((1,j)\)</td><td>UE state and \(\mathbf m_j^{\mathrm W}\) or \(\mathbf v_{sj}\)</td><td>\({}^{W}\mathbf T_{B_s}\)</td></tr>
        <tr><td>two bounce, \((2,j_1,j_2)\)</td><td>UE state, \(\mathbf m_{j_1}^{\mathrm W}\), and \(\mathbf m_{j_2}^{\mathrm W}\)</td><td>\({}^{W}\mathbf T_{B_s}\)</td></tr>
        <tr><td>\(k\) bounce</td><td>UE state and the \(k\) ordered reflector states</td><td>\({}^{W}\mathbf T_{B_s}\)</td></tr>
      </tbody>
    </table>

    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Conditioned nonlinear least squares</h4>
    <div class="eq math-eq">
      \[
      \widehat{\boldsymbol\Theta}
      =\arg\min_{\boldsymbol\Theta}\;
      J_{\mathrm{prior}}(\boldsymbol\Theta)
      +\lambda_{\mathrm{mot}}J_{\mathrm{mot}}(\mathcal X)
      +\sum_{t,s,\ell:\,a_{ts\ell}\ne0}
      \rho\!\left(
      \|\mathbf r^{\mathrm{rad}}_{ts\ell}(\mathbf x_t,\mathcal M,a_{ts\ell})\|^2_{\boldsymbol\Sigma^{-1}_{a_{ts\ell}}}
      \right).
      \]
    </div>
    <p>Set \(\lambda_{\mathrm{mot}}=0\) for motion-model-free radio bundle adjustment. Re-observations of the same plane or VA still couple distinct UE poses. Add IMU, odometry, smoothness, clock-drift, or calibration factors only when those information sources are available.</p>

    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Manifold update used by Gauss–Newton / LM / iSAM2</h4>
    <div class="eq math-eq">
      \[
      \delta\boldsymbol\xi_t=
      \begin{bmatrix}\delta\boldsymbol\rho_t\\\delta\boldsymbol\phi_t\end{bmatrix}
      \in\mathbb R^6\simeq\mathfrak{se}(3),
      \qquad
      {}^{W}\mathbf T_{U_t}\leftarrow
      {}^{W}\mathbf T_{U_t}\operatorname{Exp}(\delta\boldsymbol\xi_t),
      \]
      \[
      {}^{W}\mathbf T_{W_j}\leftarrow
      {}^{W}\mathbf T_{W_j}\operatorname{Exp}(\delta\boldsymbol\eta_j),
      \qquad
      \delta_t\leftarrow\delta_t+\Delta\delta_t.
      \]
    </div>
    <p class="eq-note">This page uses a right perturbation convention. The pose lives in \(SE(3)\); the six-vector is only a local tangent-space increment. A solver must use one left/right convention consistently when deriving Jacobians and covariances.</p>

    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Unknown association and bounce order</h4>
    <div class="eq math-eq">
      \[
      p(\boldsymbol\Theta,A\mid Z,{}^W\mathbf T_{B_{1:S}})
      \propto p(\boldsymbol\Theta)\prod_{t,s}
      \mathcal L_{\mathrm{set}}(\mathcal Z_{ts},A_{ts}\mid\boldsymbol\Theta,{}^W\mathbf T_{B_s}),
      \]
      \[
      \phi_{ts\ell}
      =\kappa_{ts}(\mathbf z_{ts\ell})
      +\sum_{h\in\mathcal H_{ts\ell}(\mathcal M)}
      w_h p_{\mathrm D}(h)\,
      \mathcal N\!\left(
      \mathbf r^{\mathrm{rad}}_{ts\ell}(\mathbf x_t,\mathcal M,h);\mathbf0,\boldsymbol\Sigma_h
      \right).
      \]
    </div>
    <p class="eq-note">\(\kappa_{ts}\) is clutter intensity and \(\mathcal H_{ts\ell}(\mathcal M)\) is the gated set of LoS and ordered-reflection candidates generated from the complete map. The prior weight \(w_h\), detection probability \(p_{\mathrm D}(h)\), and validity/visibility of a candidate may depend on all of \(\mathcal M\), while its conditioned geometric residual uses only \(\mathcal M_h\). Exact one-to-one association is combinatorial; practical systems use a front end, alternating optimization, branching, marginalization, or mixture/max-mixture factors.</p>
    <p class="eq-note" data-math-audit="mixture-scope">The joint set likelihood includes association weights and the detection/count and admissibility terms of the chosen model. The displayed \(\phi\) is a local mixture surrogate, not automatically a normalized observation density or a complete set likelihood. Clutter and path terms must share measurement coordinates and base measure. For a PPP intensity \(\lambda_\Theta\), the set likelihood includes \(e^{-\Lambda_\Theta}\prod_{z\in Z}\lambda_\Theta(z)\), where \(\Lambda_\Theta=\int\lambda_\Theta(z)\,dz\); a Bernoulli-path model instead requires its own missed-detection terms. Independent mixtures do not enforce one-to-one assignment. The conditioned least-squares form also assumes that omitted likelihood normalizers and detection terms are constant, or is only a geometric surrogate.</p>

    <div class="companion-steps">
      <article><span>Front end</span><strong>Resolve and propose in 3D</strong>Estimate delay, azimuth/elevation AoA and AoD, gain, and covariance; convert directions to the stated frames; propose LoS and ordered-plane hypotheses; reject invalid finite-support and visibility cases.</article>
      <article><span>Back end</span><strong>Optimize the product manifold</strong>Jointly update \(SE(3)\) UE poses, clock terms, and plane/VA states with sparse Gauss–Newton, LM, bundle adjustment, or iSAM2.</article>
      <article><span>Validation</span><strong>Report unresolved modes</strong>A surveyed BS fixes the global frame, but NLoS-only symmetries, unknown clocks, parallel-plane configurations, poor elevation diversity, and non-unique multi-plane decompositions can remain weakly observable.</article>
    </div>
  </div>
</section>

`

replaceBlock(
  '<!-- ============ 05 unified radio-SLAM problem formulation ============ -->',
  '<!-- ============ 06 bistatic radio to GraphSLAM ============ -->',
  formulation3D,
  'data-formulation-dimension="3d"',
  'existing 2D problem-formulation section'
)

const bistaticFactor3D = String.raw`  <div class="subsection-block" id="bistatic-factor" data-radio-factor-dimension="3d">
    <h3 class="subh"><span class="no">6.2</span>One MPC becomes one sparse 3D bistatic likelihood factor</h3>
    <p class="lede">The factor evaluates one discrete path hypothesis using a known BS transform, the current UE pose in \(SE(3)\), and the associated map entity or ordered reflector chain. For a point interaction \({}^{W}\mathbf s_j\in\mathbb R^3\),</p>
    <div class="eq math-eq">
      \[
      \widehat L_{tsj}=\|{}^{W}\mathbf s_j-{}^{W}\mathbf b_s\|+\|{}^{W}\mathbf p_{U_t}-{}^{W}\mathbf s_j\|,
      \qquad
      \widehat\tau_{tsj}=\widehat L_{tsj}/c+\delta_t,
      \]
      \[
      {}^{B_s}\widehat{\mathbf u}^{\mathrm D}_{tsj}
      =({}^{W}\mathbf R_{B_s})^{\mathsf T}
      \frac{{}^{W}\mathbf s_j-{}^{W}\mathbf b_s}{\|{}^{W}\mathbf s_j-{}^{W}\mathbf b_s\|},
      \qquad
      {}^{U_t}\widehat{\mathbf u}^{\mathrm A}_{tsj}
      =({}^{W}\mathbf R_{U_t})^{\mathsf T}
      \frac{{}^{W}\mathbf s_j-{}^{W}\mathbf p_{U_t}}{\|{}^{W}\mathbf s_j-{}^{W}\mathbf p_{U_t}\|}.
      \]
    </div>
    <p class="eq-note">For LoS, the first outgoing point is the UE and the previous source seen by the UE is the BS. For a specular chain, replace the point interaction by the first and last folded reflection points defined in Section 5.</p>
    <div class="eq math-eq">
      \[
      \mathbf r^{\mathrm{geom}}_{ts\ell}(h)=
      \begin{bmatrix}
      c(\tau_{ts\ell}-\widehat\tau_h)\\
      \mathbf r_{\mathbb S^2}({}^{U_t}\mathbf u^{\mathrm A}_{ts\ell},{}^{U_t}\widehat{\mathbf u}^{\mathrm A}_h)\\
      \mathbf r_{\mathbb S^2}({}^{B_s}\mathbf u^{\mathrm D}_{ts\ell},{}^{B_s}\widehat{\mathbf u}^{\mathrm D}_h)
      \end{bmatrix},
      \qquad
      f^{\mathrm{rad}}_{ts\ell}\propto
      \exp\!\left[-\tfrac12\|\mathbf r^{\mathrm{geom}}_{ts\ell}(h)\|^2_{\boldsymbol\Sigma_h^{-1}}\right].
      \]
    </div>
    <div class="companion-math-grid">
      <article class="companion-math-card"><h4>Spherical directions</h4><p>AoA and AoD are unit vectors on \(\mathbb S^2\). The local residual has two tangent coordinates and avoids azimuth wrap and elevation-pole singularities.</p></article>
      <article class="companion-math-card"><h4>Gain residual</h4><p>Only after calibration should gain enter as \(r_g=g^{\mathrm{dB}}-\gamma_h(\mathbf x_t,\mathcal M;{}^W\mathbf T_{B_s},\boldsymbol\xi)\). Otherwise keep gain as association or gating evidence.</p></article>
    </div>
  </div>

`

replaceBlock(
  '  <div class="subsection-block" id="bistatic-factor">',
  '  <div class="subsection-block" id="va-factor">',
  bistaticFactor3D,
  'data-radio-factor-dimension="3d"',
  '2D bistatic-factor subsection'
)

const vaFactor3D = String.raw`  <div class="subsection-block" id="va-factor" data-va-factor-dimension="3d">
    <h3 class="subh"><span class="no">6.3</span>A 3D specular plane becomes a landmark through an image source</h3>
    <p class="lede">For an ideal plane \(\Pi_j\), reflect the known BS position through the plane. The resulting VA behaves like a point landmark for total path length and arrival direction, while the physical reflection point is retained for AoD, finite support, and visibility.</p>
    <div class="eq math-eq">
      \[
      \Pi_j=\{\mathbf y\in\mathbb R^3:\mathbf n_j^{\mathsf T}\mathbf y=d_j\},
      \qquad \mathbf n_j\in\mathbb S^2,
      \]
      \[
      \mathbf H_j=\mathbf I-2\mathbf n_j\mathbf n_j^{\mathsf T}\in O(3)\setminus SO(3),
      \qquad
      \mathbf v_{sj}=\mathbf H_j{}^W\mathbf b_s+2d_j\mathbf n_j,
      \]
      \[
      \lambda_{tsj}=\frac{d_j-\mathbf n_j^{\mathsf T}{}^W\mathbf p_{U_t}}
      {\mathbf n_j^{\mathsf T}(\mathbf v_{sj}-{}^W\mathbf p_{U_t})},
      \qquad
      \mathbf q_1={}^W\mathbf p_{U_t}+\lambda_{tsj}(\mathbf v_{sj}-{}^W\mathbf p_{U_t}),
      \qquad
      \widehat L_{tsj}=\|{}^W\mathbf p_{U_t}-\mathbf v_{sj}\|.
      \]
    </div>
    <p class="eq-note">A valid finite reflector needs \(0&lt;\lambda_{tsj}&lt;1\), \(\mathbf q_1\in\mathcal S_j\), and unoccluded BS→\(\mathbf q_1\) and \(\mathbf q_1\)→UE legs.</p>
    <div class="eq math-eq">
      \[
      {}^{U_t}\widehat{\mathbf u}^{\mathrm A}_{tsj}
      =({}^{W}\mathbf R_{U_t})^{\mathsf T}
      \frac{\mathbf q_1-{}^W\mathbf p_{U_t}}{\|\mathbf q_1-{}^W\mathbf p_{U_t}\|},
      \qquad
      {}^{B_s}\widehat{\mathbf u}^{\mathrm D}_{tsj}
      =({}^{W}\mathbf R_{B_s})^{\mathsf T}
      \frac{\mathbf q_1-{}^W\mathbf b_s}{\|\mathbf q_1-{}^W\mathbf b_s\|}.
      \]
    </div>
    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Ordered multi-bounce path</h4>
    <div class="eq math-eq">
      \[
      \mathbf v_s^{(0)}={}^W\mathbf b_s,
      \qquad
      \mathbf v_s^{(r)}=\mathcal R_{j_r}(\mathbf v_s^{(r-1)}),\quad r=1,\ldots,k,
      \qquad
      \widehat L_h=\|{}^W\mathbf p_{U_t}-\mathbf v_s^{(k)}\|.
      \]
    </div>
    <p>Fold the UE→final-VA line back through the planes in reverse order to recover \(\mathbf q_k,\ldots,\mathbf q_1\). The ordered chain is part of the factor hypothesis. A final VA point alone is not a substitute for the wall chain when the factor must predict AoD, finite support, or a physical propagation map.</p>
    <table class="companion-symbols">
      <thead><tr><th>Map state</th><th>Best use</th><th>Important limitation</th></tr></thead>
      <tbody>
        <tr><td>VA \(\mathbf v_{sj}\in\mathbb R^3\)</td><td>Compact one-BS, one-plane factor for delay and AoA; wall is recoverable for one bounce.</td><td>Different BSs have different VAs for the same physical plane.</td></tr>
        <tr><td>Plane \((\mathbf n_j,d_j)\)</td><td>Minimal infinite reflector shared across poses, paths, and BSs.</td><td>No finite support or in-plane extent information.</td></tr>
        <tr><td>Finite wall frame \(({}^W\mathbf T_{W_j},a_j,b_j)\)</td><td>Physical 3D environment reconstruction, support checks, and shared radiometric parameters.</td><td>Higher-dimensional state and visibility logic are required.</td></tr>
        <tr><td>Point interaction \({}^W\mathbf s_j\)</td><td>Point-like diffuse scatterer or localized interaction.</td><td>Does not enforce equal-angle specular reflection.</td></tr>
        <tr><td>Composite image / transform</td><td>Compact known ordered multi-bounce geometry.</td><td>May identify the composite transform without uniquely identifying every plane.</td></tr>
      </tbody>
    </table>
  </div>

`

replaceBlock(
  '  <div class="subsection-block" id="va-factor">',
  '  <div class="subsection-block" id="radio-graph-objective">',
  vaFactor3D,
  'data-va-factor-dimension="3d"',
  '2D virtual-anchor subsection'
)

const required = [
  'data-formulation-dimension="3d"',
  'A complete 3D radio-SLAM problem formulation',
  '{}^{W}\\mathbf T_{U_t}',
  '\\in SE(3)\\times\\mathbb R',
  '\\boldsymbol\\pi_j=(\\mathbf n_j,d_j)\\in\\mathbb S^2\\times\\mathbb R',
  '\\mathbf H_j\\in O(3),\\quad\\det\\mathbf H_j=-1',
  '\\operatorname{Log}_{\\mathbb S^2,\\widehat{\\mathbf u}}',
  'data-radio-factor-dimension="3d"',
  'data-va-factor-dimension="3d"',
  'O(3)\\setminus SO(3)',
  'two bounce, \\((2,j_1,j_2)\\)'
]
for (const value of required) {
  if (!html.includes(value)) throw new Error(`3D formulation validation failed: ${value}`)
}

const sectionStart = html.indexOf('data-formulation-dimension="3d"')
const sectionEnd = html.indexOf('<!-- ============ 06 bistatic radio to GraphSLAM ============ -->', sectionStart)
if (sectionStart < 0 || sectionEnd < 0) throw new Error('Could not isolate the 3D formulation section')
const formulation = html.slice(sectionStart, sectionEnd)
for (const forbidden of [
  'planar \\(SE(2)\\) model',
  '\\mathbb R^2\\times\\mathbb S^1',
  'Physical wall state in 2D',
  '\\mathbf m^{\\mathrm{VA}}_{sj}=\\mathbf v_{sj}\\in\\mathbb R^2'
]) {
  if (formulation.includes(forbidden)) throw new Error(`Legacy 2D formulation remains: ${forbidden}`)
}

if (changed) {
  writeFileSync(pagePath, html)
  console.log('Upgraded the formal radio-SLAM problem and companion factors to 3D.')
} else {
  console.log('No 3D radio-SLAM formulation changes needed.')
}
