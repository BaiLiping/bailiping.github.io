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

function insertBefore(marker, addition, sentinel, label) {
  if (html.includes(sentinel)) return
  const index = html.indexOf(marker)
  if (index < 0) throw new Error(`Could not find ${label}`)
  html = html.slice(0, index) + addition + html.slice(index)
  changed = true
}

replaceOptional(
  '    <a href="#bistatic-graphslam">GraphSLAM bridge</a>',
  '    <a href="#problem-formulation">Problem formulation</a>\n    <a href="#bistatic-graphslam">GraphSLAM bridge</a>'
)

const problemFormulationSection = String.raw`
<!-- ============ 05 unified radio-SLAM problem formulation ============ -->
<section class="sec companion-section" id="problem-formulation">
  <h2><span class="no">05</span>A complete radio-SLAM problem formulation</h2>
  <p class="lede">This section collects the problem in one place. The base-station pose is known. The continuous unknowns are the UE trajectory and a persistent radio map. Each detected MPC also has a discrete explanation: clutter, LoS, one bounce on one surface, two ordered bounces on two surfaces, and so on.</p>

  <div class="accuracy"><strong>Recommended representation for this page.</strong> Use <em>physical wall variables</em> as the map state and compute virtual anchors and bounce points deterministically inside each radio factor. This preserves finite-wall support, AoD, visibility, and a shared physical surface across different BSs. A directly optimized VA is an excellent compact state for a one-bounce path, but a single composite-VA point is generally insufficient to identify the individual walls of a multi-bounce path.</div>

  <nav class="subsection-tiles is-four" aria-label="Radio SLAM problem-formulation subsections">
    <a href="#formulation-state"><span>5.1</span><strong>Continuous state</strong><small>UE · known BS · wall · VA</small></a>
    <a href="#formulation-hypothesis"><span>5.2</span><strong>Path hypothesis</strong><small>LoS · one bounce · two bounce</small></a>
    <a href="#formulation-measurement"><span>5.3</span><strong>Measurement model</strong><small>delay · AoA · AoD · path loss</small></a>
    <a href="#formulation-map"><span>5.4</span><strong>Joint inference</strong><small>factors · association · MAP</small></a>
  </nav>

  <div class="subsection-block" id="formulation-state">
    <h3 class="subh"><span class="no">5.1</span>Continuous state: UE trajectory, known BS, and radio map</h3>
    <p class="lede">The drawings on this page use a planar \(SE(2)\) model. The same construction extends to \(SE(3)\) by replacing headings with rotation matrices and wall segments with finite planes.</p>

    <h4 style="margin:20px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">UE state</h4>
    <div class="eq math-eq">
      \[
      \mathbf x_t
      =\big(\mathbf p_t,\theta_t,\delta_t\big)
      \in \mathbb R^2\times\mathbb S^1\times\mathbb R,
      \qquad
      \mathcal X=\{\mathbf x_t\}_{t=0}^{T}.
      \]
    </div>
    <table class="companion-symbols">
      <thead><tr><th>Quantity</th><th>Status</th><th>Meaning</th></tr></thead>
      <tbody>
        <tr><td>\(\mathbf p_t=[p_{x,t},p_{y,t}]^{\mathsf T}\)</td><td>unknown</td><td>UE position in the world/map frame.</td></tr>
        <tr><td>\(\theta_t\)</td><td>unknown or aided</td><td>UE array/body heading. AoA is measured in this local frame, so heading must be estimated unless supplied by an IMU or another attitude source.</td></tr>
        <tr><td>\(\delta_t\)</td><td>optional unknown</td><td>UE-to-BS clock and hardware-delay offset in seconds. Set \(\delta_t=0\) only for a synchronized and calibrated system.</td></tr>
        <tr><td>\(\mathbf v_t,\omega_t,\dot\delta_t\)</td><td>optional extension</td><td>Add velocity, turn rate, or clock drift only when a dynamic or continuous-time prior requires them. They are not required by the radio measurement factor itself.</td></tr>
      </tbody>
    </table>

    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Known base station</h4>
    <div class="eq math-eq">
      \[
      \mathcal B_s=(\mathbf b_s,\theta_s),
      \qquad
      \mathbf b_s\in\mathbb R^2,
      \quad \theta_s\in\mathbb S^1,
      \qquad \mathcal B_s\;\text{is known}.
      \]
    </div>
    <p class="eq-note">The BS position, array heading, transmit calibration, and any known hardware delay enter the likelihood as fixed parameters. They become graph variables only when the calibration itself is to be estimated.</p>

    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Physical wall state in 2D</h4>
    <div class="eq math-eq">
      \[
      \mathbf m_j^{\mathrm W}
      =\big(\mathbf c_j,\alpha_j,\ell_j,\boldsymbol\xi_j\big),
      \qquad
      \mathbf n_j=
      \begin{bmatrix}\cos\alpha_j\\ \sin\alpha_j\end{bmatrix},
      \qquad
      \mathbf t_j=
      \begin{bmatrix}-\sin\alpha_j\\ \cos\alpha_j\end{bmatrix},
      \]
      \[
      \mathcal S_j
      =\left\{\mathbf c_j+u\mathbf t_j:\;|u|\leq \ell_j/2\right\}.
      \]
    </div>
    <p class="eq-note">\(\mathbf c_j\) is the segment center, \(\mathbf n_j\) its unit normal, \(\mathbf t_j\) its tangent, and \(\ell_j\) its finite length. \(\boldsymbol\xi_j\) denotes optional material parameters used by the path-loss model. If the wall is treated as infinite, omit \(\ell_j\).</p>

    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Virtual-anchor alternative</h4>
    <div class="eq math-eq">
      \[
      \mathcal R_j(\mathbf y)
      =\mathbf y-2\mathbf n_j\mathbf n_j^{\mathsf T}(\mathbf y-\mathbf c_j),
      \qquad
      \mathbf v_{sj}=\mathcal R_j(\mathbf b_s).
      \]
    </div>
    <div class="companion-math-grid">
      <article class="companion-math-card"><h4>One-bounce VA state</h4><p>With BS \(s\) known, \(\mathbf m^{\mathrm{VA}}_{sj}=\mathbf v_{sj}\in\mathbb R^2\) is a compact landmark state. The corresponding infinite wall is the perpendicular bisector of \(\mathbf b_s\) and \(\mathbf v_{sj}\).</p></article>
      <article class="companion-math-card"><h4>Multi-bounce warning</h4><p>For two or more bounces, the final point \(\mathbf v_s^{(k)}\) gives path length and the final unfolded bearing, but generally does not uniquely encode every physical wall, finite support, or first departure direction. Keep the ordered wall variables, or store the complete composite reflection transform rather than only one point.</p></article>
    </div>

    <div class="accuracy"><strong>3D extension.</strong> Replace \(\theta_t\) and \(\theta_s\) by \(\mathbf R_t,\mathbf R_s\in SO(3)\). A finite rectangular surface needs a center \(\mathbf c_j\), unit normal \(\mathbf n_j\in\mathbb S^2\), an in-plane orientation, length, and width. Center, normal, length, and width alone do not determine rotation about the normal.</div>
  </div>

  <div class="subsection-block" id="formulation-hypothesis">
    <h3 class="subh"><span class="no">5.2</span>Discrete path hypothesis and specular geometry</h3>
    <p class="lede">For detected MPC \(\ell\) at time \(t\) from BS \(s\), let the path hypothesis contain a bounce count and an <em>ordered</em> sequence of wall identities.</p>
    <div class="eq math-eq">
      \[
      h_{ts\ell}=\big(k_{ts\ell},\mathbf j_{ts\ell}\big),
      \qquad
      k_{ts\ell}\in\{0,1,2,\ldots,K_{\max}\},
      \qquad
      \mathbf j_{ts\ell}=(j_1,\ldots,j_k).
      \]
    </div>
    <p class="eq-note">\(k=0\) means LoS and the wall sequence is empty. For two bounces, \((j_1,j_2)\) and \((j_2,j_1)\) are generally different paths because the first and last interactions—and therefore AoD and AoA—change.</p>

    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Image-source recursion</h4>
    <div class="eq math-eq">
      \[
      \mathbf v_s^{(0)}=\mathbf b_s,
      \qquad
      \mathbf v_s^{(r)}=\mathcal R_{j_r}\!\left(\mathbf v_s^{(r-1)}\right),
      \quad r=1,\ldots,k.
      \]
    </div>
    <p>For a valid ideal specular path, the physical broken-path length equals the straight distance from the UE to the final image source. The reflection points are recovered by folding the line back through the walls in reverse order.</p>

    <div class="companion-math-grid">
      <article class="companion-math-card"><h4>LoS: \(k=0\)</h4><p>No wall is involved:</p><p>\(L^{(0)}_{ts}=\|\mathbf p_t-\mathbf b_s\|.\)</p><p>The first and last path points are the UE and BS themselves.</p></article>
      <article class="companion-math-card"><h4>One bounce: \(k=1\)</h4><p>For wall \(j\):</p><p>\(\mathbf v_s^{(1)}=\mathcal R_j(\mathbf b_s),\)</p><p>\(\mathbf q_1=[\mathbf p_t,\mathbf v_s^{(1)}]\cap\mathcal S_j,\)</p><p>\(L^{(1)}_{tsj}=\|\mathbf p_t-\mathbf v_s^{(1)}\|.\)</p></article>
      <article class="companion-math-card"><h4>Two bounces: \(k=2\)</h4><p>For ordered walls \((j_1,j_2)\):</p><p>\(\mathbf v_s^{(1)}=\mathcal R_{j_1}(\mathbf b_s),\quad \mathbf v_s^{(2)}=\mathcal R_{j_2}(\mathbf v_s^{(1)}),\)</p><p>\(\mathbf q_2=[\mathbf p_t,\mathbf v_s^{(2)}]\cap\mathcal S_{j_2},\)</p><p>\(\mathbf q_1=[\mathbf q_2,\mathbf v_s^{(1)}]\cap\mathcal S_{j_1},\)</p><p>\(L^{(2)}_{tsj_1j_2}=\|\mathbf p_t-\mathbf v_s^{(2)}\|.\)</p></article>
      <article class="companion-math-card"><h4>Validity indicator</h4><p>Define \(\chi(h,\mathbf x_t,\mathcal M,\mathcal B_s)\in\{0,1\}\). It is one only when every required line intersection exists, all ordered segment lengths are positive, every hit lies on the finite surface, and all physical legs are visible.</p></article>
    </div>

    <div class="eq math-eq">
      \[
      L^{(k)}_{ts\ell}
      =\sum_{r=0}^{k}\|\mathbf q_{r+1}-\mathbf q_r\|,
      \qquad
      \mathbf q_0=\mathbf b_s,
      \quad \mathbf q_{k+1}=\mathbf p_t,
      \qquad
      L^{(k)}_{ts\ell}=\|\mathbf p_t-\mathbf v_s^{(k)}\|
      \;\text{for a valid specular path}.
      \]
    </div>
  </div>

  <div class="subsection-block" id="formulation-measurement">
    <h3 class="subh"><span class="no">5.3</span>Unified measurement model for delay, AoA, AoD, and path loss</h3>
    <p class="lede">The channel estimator returns one noisy tuple per resolvable MPC. At the scene level, its distribution is conditioned on the current UE state, the complete map \(\mathcal M\), the known BS, and a latent path hypothesis. Once that hypothesis selects an ordered reflector sequence, the corresponding GraphSLAM factor uses only the selected subset of map variables.</p>
    <div class="eq math-eq">
      \[
      \mathbf z_{ts\ell}
      =\begin{bmatrix}
      \tau_{ts\ell} & \varphi^U_{ts\ell} & \psi^B_{ts\ell} & \mathrm{PL}_{ts\ell}
      \end{bmatrix}^{\mathsf T}
      =\mathbf h\!\left(
      \mathbf x_t,\mathcal M,h_{ts\ell};\mathcal B_s
      \right)+\boldsymbol\varepsilon_{ts\ell},
      \qquad
      h_{ts\ell}\in\mathcal H_{ts\ell}(\mathcal M),
      \quad
      \boldsymbol\varepsilon_{ts\ell}\sim\mathcal N(\mathbf 0,\boldsymbol\Sigma_{h}).
      \]
    </div>

    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Common prediction shared by all bounce orders</h4>
    <div class="eq math-eq">
      \[
      \widehat\tau^{(k)}=\frac{L^{(k)}}{c}+\delta_t,
      \qquad
      \widehat\psi^{B,(k)}
      =\operatorname{ang}\!\left(
      \mathbf R_s^{\mathsf T}
      \frac{\mathbf s^{\mathrm{first}}-\mathbf b_s}
      {\|\mathbf s^{\mathrm{first}}-\mathbf b_s\|}
      \right),
      \]
      \[
      \widehat\varphi^{U,(k)}
      =\operatorname{ang}\!\left(
      \mathbf R_t^{\mathsf T}
      \frac{\mathbf s^{\mathrm{last}}-\mathbf p_t}
      {\|\mathbf s^{\mathrm{last}}-\mathbf p_t\|}
      \right),
      \]
      \[
      \mathbf s^{\mathrm{first}}=
      \begin{cases}
      \mathbf p_t,&k=0,\\
      \mathbf q_1,&k\geq1,
      \end{cases}
      \qquad
      \mathbf s^{\mathrm{last}}=
      \begin{cases}
      \mathbf b_s,&k=0,\\
      \mathbf q_k,&k\geq1.
      \end{cases}
      \]
    </div>
    <p class="eq-note">\(\mathbf R_s=\mathbf R(\theta_s)\) and \(\mathbf R_t=\mathbf R(\theta_t)\) map local vectors to the world frame; their transposes express a world direction in the BS or UE frame. In 2D, \(\operatorname{ang}([x,y]^{\mathsf T})=\operatorname{atan2}(y,x)\). AoA points from the UE toward the last interaction/source.</p>

    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Map-conditioned VA / image-source measurement model</h4>
    <p>The equations in this first form retain the virtual-anchor or image-source construction. The complete map supplies the selected walls or VAs, and the factor evaluates the corresponding unfolded path.</p>
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

        <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Direct wall-state measurement model — no VA in the factor definition</h4>
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

<h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Path-loss component</h4>
    <div class="eq math-eq">
      \[
      \widehat{\mathrm{PL}}^{(k)}
      =\mathcal P_k\!\left(
      \{\|\mathbf q_{r+1}-\mathbf q_r\|\}_{r=0}^{k},
      \{\mathbf q_r,\mathbf n_{j_r},\boldsymbol\xi_{j_r}\}_{r=1}^{k},
      f_c,\widehat\varphi^U,\widehat\psi^B,\boldsymbol\eta
      \right).
      \]
    </div>
    <p>The function \(\mathcal P_k\) is deliberately modular. In dB it may contain propagation spreading, BS and UE antenna gains, one interaction loss per bounce as a function of incidence angle and material, blockage, polarization, and calibration offsets. The interaction sum is empty for LoS. Do not infer bounce count from path loss alone unless these nuisance terms are calibrated.</p>

    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Residual used by the optimizer</h4>
    <div class="eq math-eq">
      \[
      \mathbf r^{\mathrm{rad}}_{ts\ell}(\mathbf x_t,\mathcal M,h)=
      \begin{bmatrix}
      c\big(\tau_{ts\ell}-\widehat\tau_h\big)\\
      \operatorname{wrap}_{\pi}\!\big(\varphi^U_{ts\ell}-\widehat\varphi^U_h\big)\\
      \operatorname{wrap}_{\pi}\!\big(\psi^B_{ts\ell}-\widehat\psi^B_h\big)\\
      \mathrm{PL}_{ts\ell}-\widehat{\mathrm{PL}}_h
      \end{bmatrix},
      \qquad
      \|\mathbf r\|^2_{\boldsymbol\Omega_h}
      =\mathbf r^{\mathsf T}\boldsymbol\Sigma_h^{-1}\mathbf r.
      \]
    </div>
    <p class="eq-note">Multiplying the delay residual by \(c\) expresses it in metres. The covariance must use the same units and should retain delay–angle correlations when the channel estimator provides them.</p>
  </div>

  <div class="subsection-block" id="formulation-map">
    <h3 class="subh"><span class="no">5.4</span>Factor graph, unknown association, and the joint MAP problem</h3>
    <p class="lede">The complete measurement likelihood is conditioned on \(\mathcal M\), but after a path hypothesis is selected the corresponding factor touches only the UE state and the selected tuple \(\mathcal M_h\). Its arity therefore grows with bounce count while the global map remains the object being jointly estimated.</p>
    <table class="companion-symbols">
      <thead><tr><th>Hypothesis</th><th>Unknown nodes touched</th><th>Known input</th></tr></thead>
      <tbody>
        <tr><td>clutter / false alarm</td><td>none, or a latent assignment variable</td><td>clutter intensity over measurement space</td></tr>
        <tr><td>LoS, \(k=0\)</td><td>\(\mathbf x_t\)</td><td>\(\mathcal B_s\)</td></tr>
        <tr><td>one bounce, \((1,j)\)</td><td>\(\mathbf x_t,\mathbf m_j\)</td><td>\(\mathcal B_s\)</td></tr>
        <tr><td>two bounce, \((2,j_1,j_2)\)</td><td>\(\mathbf x_t,\mathbf m_{j_1},\mathbf m_{j_2}\)</td><td>\(\mathcal B_s\)</td></tr>
        <tr><td>\(k\) bounce</td><td>\(\mathbf x_t,\mathbf m_{j_1},\ldots,\mathbf m_{j_k}\)</td><td>\(\mathcal B_s\)</td></tr>
      </tbody>
    </table>

    <div class="eq math-eq">
      \[
      a_{ts\ell}\in
      \left\{0,\mathrm{LoS},(1,j),(2,j_1,j_2),\ldots\right\},
      \]
    </div>
    <p class="eq-note">\(a=0\) denotes clutter. Otherwise \(a\) selects both the bounce order and the ordered surface identities. The current scan index \(\ell\) is not itself a persistent feature identity.</p>

    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Conditioned problem: the front end has selected one hypothesis</h4>
    <div class="eq math-eq">
      \[
      \begin{aligned}
      (\widehat{\mathcal X},\widehat{\mathcal M})
      =\arg\min_{\mathcal X,\mathcal M}\;&
      J_{\mathrm{prior}}(\mathcal X,\mathcal M)
      +\lambda_{\mathrm{mot}}J_{\mathrm{mot}}(\mathcal X)\\
      &+\sum_{t,s,\ell:\,a_{ts\ell}\neq0}
      \rho\!\left(
      \|\mathbf r^{\mathrm{rad}}_{ts\ell}(\mathbf x_t,\mathcal M,a_{ts\ell})\|^2_{\boldsymbol\Omega_{a_{ts\ell}}}
      \right),
      \end{aligned}
      \]
    </div>
    <p>Set \(\lambda_{\mathrm{mot}}=0\) for motion-model-free radio bundle adjustment. Then different UE poses are coupled only through re-observation of common wall or VA variables. Add odometry, IMU, clock-drift, smoothness, or calibration factors when those measurements are available.</p>

    <h4 style="margin:22px 0 8px;font:700 11px var(--mono);letter-spacing:.08em;text-transform:uppercase">Hybrid problem: association and bounce order are unknown</h4>
    <div class="eq math-eq">
      \[
      (\widehat{\mathcal X},\widehat{\mathcal M},\widehat A)
      =\arg\max_{\mathcal X,\mathcal M,A}
      p(\mathcal X,\mathcal M)
      \prod_{t,s}p(\mathcal Z_{ts}\mid\mathbf x_t,\mathcal M,A_{ts},\mathcal B_s),
      \]
      \[
      \phi_{ts\ell}(\mathbf x_t,\mathcal M)
      =\kappa_{ts}(\mathbf z_{ts\ell})
      +\sum_{h\in\mathcal H_{ts\ell}(\mathcal M)}
      w_h p_{\mathrm D}(h)
      \mathcal N\!\left(
      \mathbf z_{ts\ell};
      \mathbf h(\mathbf x_t,\mathcal M,h;\mathcal B_s),
      \boldsymbol\Sigma_h
      \right).
      \]
    </div>
    <p class="eq-note">\(\kappa\) is a clutter intensity and \(\mathcal H_{ts\ell}(\mathcal M)\) is the gated set of LoS and ordered-reflection hypotheses generated from the complete map. The weight \(w_h\), detection probability \(p_{\mathrm D}(h)\), and validity/visibility of each candidate may depend on the whole scene, whereas its conditioned geometric factor uses only \(\mathcal M_h\). Exact one-to-one assignment is combinatorial; practical systems use front-end association, alternating inference, branching, or sum-/max-mixture factors.</p>

    <div class="companion-steps">
      <article><span>Front end</span><strong>Resolve and propose</strong>Estimate MPC tuples and covariances; normalize frames; propose LoS, ordered wall sequences, and clutter; reject impossible visibility cases.</article>
      <article><span>Back end</span><strong>Optimize continuous states</strong>Given active hypotheses, jointly update UE poses, clock terms, and wall or VA states with sparse Gauss–Newton, LM, bundle adjustment, or iSAM2.</article>
      <article><span>Validation</span><strong>Inspect ambiguity</strong>A known BS anchors the coordinate frame, but NLoS-only mirror ambiguities, corridor null spaces, clock ambiguity, and non-unique multi-wall decompositions can remain.</article>
    </div>
  </div>
</section>

`

insertBefore(
  '<!-- ============ 05 bistatic radio to GraphSLAM ============ -->',
  problemFormulationSection,
  'id="formulation-state"',
  'bistatic GraphSLAM section marker'
)

const numberingUpdates = [
  ['<h2><span class="no">05</span>Bistatic radio SLAM as GraphSLAM</h2>', '<h2><span class="no">06</span>Bistatic radio SLAM as GraphSLAM</h2>'],
  ['<a href="#factor-language"><span>5.1</span>', '<a href="#factor-language"><span>6.1</span>'],
  ['<a href="#bistatic-factor"><span>5.2</span>', '<a href="#bistatic-factor"><span>6.2</span>'],
  ['<a href="#va-factor"><span>5.3</span>', '<a href="#va-factor"><span>6.3</span>'],
  ['<a href="#radio-graph-objective"><span>5.4</span>', '<a href="#radio-graph-objective"><span>6.4</span>'],
  ['<h3 class="subh"><span class="no">5.1</span>The sensor placement changes', '<h3 class="subh"><span class="no">6.1</span>The sensor placement changes'],
  ['<h3 class="subh"><span class="no">5.2</span>One MPC becomes', '<h3 class="subh"><span class="no">6.2</span>One MPC becomes'],
  ['<h3 class="subh"><span class="no">5.3</span>A specular wall becomes', '<h3 class="subh"><span class="no">6.3</span>A specular wall becomes'],
  ['<h3 class="subh"><span class="no">5.4</span>Joint trajectory', '<h3 class="subh"><span class="no">6.4</span>Joint trajectory']
]
for (const [before, after] of numberingUpdates) replaceOptional(before, after)

const required = [
  'id="problem-formulation"',
  'id="formulation-state"',
  'id="formulation-hypothesis"',
  'id="formulation-measurement"',
  'id="formulation-map"',
  '\\mathbf x_t',
  '\\mathbf h_0',
  '\\mathbf h_1',
  '\\mathbf h_2',
  'two bounce,',
  '<h2><span class="no">06</span>Bistatic radio SLAM as GraphSLAM</h2>'
]
for (const value of required) {
  if (!html.includes(value)) throw new Error(`Problem-formulation validation failed: ${value}`)
}

if (changed) {
  writeFileSync(pagePath, html)
  console.log('Inserted the unified LoS/one-bounce/two-bounce radio-SLAM problem formulation.')
} else {
  console.log('No problem-formulation changes needed.')
}
