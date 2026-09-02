import { readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

const sourcePath = resolve('mpc-detection-to-bounce-count-slides/radio-slam-extra.mjs')
let source = readFileSync(sourcePath, 'utf8')
let changed = false

const BT = '`'
const graphMarker = 'function graphEquationSlide(ctx) {'

if (!source.includes('function isam2FramingSlide(ctx) {')) {
  if (!source.includes(graphMarker)) throw new Error('Could not find GraphSLAM slide insertion marker')

  const isam2Slides = String.raw`function isam2FramingSlide(ctx) {
  const { regular, text, card, shape, line, C, SANS, MONO, tex, texBlock } = ctx
  const elements = [
    card('isam-frame-model-card', 96, 206, 314, 112, C.mapSoft, { stroke: C.map, strokeWidth: 2, radius: 8 }),
    text('isam-frame-model-k', 116, 222, 274, 16, '1 · RADIO MODEL', 9, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .9 }),
    text('isam-frame-model-v', 116, 250, 274, 52, 'Choose pose and map variables, path geometry, noise, gauge priors, and optional clock/calibration states.', 11.5, { color: C.mapDeep, fontFamily: SANS, fontWeight: 700, align: 'center', lineHeight: 1.35 }),

    card('isam-frame-front-card', 96, 334, 314, 112, C.measurementSoft, { stroke: C.measurement, strokeWidth: 2, radius: 8 }),
    text('isam-frame-front-k', 116, 350, 274, 16, '2 · FRONT END', 9, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .9 }),
    text('isam-frame-front-v', 116, 378, 274, 52, 'Detect MPCs, propose association and bounce order, gate hypotheses, initialize new variables, and attach covariances.', 11.5, { color: C.measurementDeep, fontFamily: SANS, fontWeight: 700, align: 'center', lineHeight: 1.35 }),

    card('isam-frame-back-card', 96, 462, 314, 112, C.poseSoft, { stroke: C.pose, strokeWidth: 2, radius: 8 }),
    text('isam-frame-back-k', 116, 478, 274, 16, '3 · iSAM2 BACK END', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .9 }),
    text('isam-frame-back-v', 116, 506, 274, 52, 'Incrementally update the nonlinear MAP estimate and Bayes tree; relinearize and reorder only where needed.', 11.5, { color: C.poseDeep, fontFamily: SANS, fontWeight: 700, align: 'center', lineHeight: 1.35 }),

    text('isam-frame-arrow-1', 214, 312, 78, 22, '↓', 20, { color: C.faint, fontFamily: SANS, fontWeight: 700, align: 'center' }),
    text('isam-frame-arrow-2', 214, 440, 78, 22, '↓', 20, { color: C.faint, fontFamily: SANS, fontWeight: 700, align: 'center' }),

    card('isam-frame-objective-card', 438, 206, 746, 176, C.paper, { stroke: C.line, radius: 8 }),
    text('isam-frame-objective-k', 462, 222, 420, 18, 'SAME POSTERIOR AS BATCH BA / GRAPHSAM', 9.5, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .9 }),
    text('isam-frame-objective-eq', 452, 248, 718, 100, texBlock§\begin{aligned}
      \boldsymbol\Theta_t&=\{\mathbf T_{0:t},\mathcal M_t,\boldsymbol\kappa_t\},\\[-.1em]
      \boldsymbol\Theta_t^*&=\arg\min_{\boldsymbol\Theta_t}\;\|\mathbf r^{\rm prior}\|^2_{\Omega_0}
      +\sum_{i\in\mathcal F_{0:t}}\rho_i\!\left(\|\mathbf r_i(\boldsymbol\Theta_t)\|^2_{\Omega_i}\right),\\[-.1em]
      \mathcal F_{0:t}&=\mathcal F_{0:t-1}\cup\Delta\mathcal F_t,\qquad
      \boldsymbol\Theta_t^{0}=\boldsymbol\Theta_{t-1}^{*}\cup\Delta\boldsymbol\Theta_t^{0}.
    \end{aligned}§, 13.2, { fontWeight: 700, align: 'center', lineHeight: 1.25 }),
    text('isam-frame-objective-v', 462, 350, 698, 20, 'iSAM2 changes the update strategy—not the measurement likelihood, map representation, or MAP objective.', 10.5, { color: C.soft, fontFamily: SANS, fontWeight: 700, align: 'center' }),

    card('isam-frame-stream-card', 438, 400, 746, 174, C.poseSoft, { stroke: C.pose, radius: 8 }),
    text('isam-frame-stream-k', 462, 416, 420, 18, 'ONE STREAMING UPDATE AT TIME t', 9.5, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .9 }),
    text('isam-frame-stream-eq', 456, 442, 710, 72, texBlock§\begin{aligned}
      \Delta\mathcal F_t={}&\{f_t^{\rm prior/odo},\ f_{t\ell}^{\rm rad},\ f_{ij}^{\rm revisit},\ldots\},\\[-.1em]
      \Delta\boldsymbol\Theta_t^0={}&\{\mathbf T_t^0,\ \mathbf m_{j_1}^0,\ldots\},\\[-.1em]
      \operatorname{iSAM2.update}(&\Delta\mathcal F_t,\Delta\boldsymbol\Theta_t^0)\;\longrightarrow\;\boldsymbol\Theta_t^*.
    \end{aligned}§, 13.5, { fontWeight: 700, align: 'center', lineHeight: 1.3 }),
    text('isam-frame-stream-v', 462, 524, 698, 32, 'Only newly introduced keys need initial values. Existing estimates and the Bayes tree are retained across scans.', 10.5, { color: C.poseDeep, fontFamily: SANS, fontWeight: 700, align: 'center', lineHeight: 1.25 }),

    card('isam-frame-warning-card', 96, 592, 1088, 48, C.measurementSoft, { stroke: C.measurement, radius: 7 }),
    text('isam-frame-warning-k', 116, 604, 184, 16, 'MODELING BOUNDARY', 9, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .9 }),
    text('isam-frame-warning-v', 302, 600, 862, 28, 'Ordinary iSAM2 optimizes the factors it is given. MPC association, bounce-order selection, births/deaths, and outlier admission remain front-end or hybrid-inference decisions.', 11.2, { color: C.measurementDeep, fontFamily: SANS, fontWeight: 700, align: 'center', valign: 'middle' }),

    text('isam-frame-ref', 96, 653, 1088, 12, 'Ref · Kaess et al., iSAM2: Incremental Smoothing and Mapping Using the Bayes Tree, IJRR 2012', 7, { color: C.faint, fontFamily: MONO, align: 'center' })
  ]

  return regular(
    's-isam2-framing', '04 · KNOWN BS POSE, UNKNOWN UE POSE AND MAP',
    'Frame the radio-SLAM problem for iSAM2',
    'Keep the radio factor graph unchanged; make its nonlinear MAP solution incremental.',
    'Use this page to separate the scientific model from the numerical backend. The radio work must still define variables, map representation, likelihoods, association, bounce order, initialization, and gauge constraints. iSAM2 receives accepted nonlinear factors and initial values, maintains the entire smoothing posterior through a Bayes tree, and updates the MAP estimate as factors arrive. It is therefore best described as the incremental inference engine for the same bundle-adjustment or GraphSLAM objective, not as a new radio measurement model.',
    elements, { accent: C.pose, titleSize: 33, transition: 'none' }
  )
}

function isam2BayesTreeSlide(ctx) {
  const { regular, text, card, shape, line, C, SANS, MONO, tex, texBlock } = ctx
  const elements = [
    card('isam-tree-linearize-card', 96, 206, 322, 132, C.paper, { stroke: C.line, radius: 8 }),
    text('isam-tree-linearize-k', 116, 222, 282, 16, '1 · LINEARIZE LOCALLY', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .8 }),
    text('isam-tree-linearize-eq', 108, 248, 298, 56, texBlock§\mathbf r_i(\bar{\boldsymbol\Theta}\oplus\delta)\approx\mathbf r_i(\bar{\boldsymbol\Theta})+\mathbf J_i\delta,\qquad\mathbf A\delta\approx\mathbf b§, 12.5, { fontWeight: 700, align: 'center' }),
    text('isam-tree-linearize-v', 116, 310, 282, 18, 'Only selected nonlinear factors are relinearized.', 9.5, { color: C.soft, fontFamily: SANS, fontWeight: 700, align: 'center' }),

    card('isam-tree-eliminate-card', 438, 206, 322, 132, C.paper, { stroke: C.line, radius: 8 }),
    text('isam-tree-eliminate-k', 458, 222, 282, 16, '2 · ELIMINATE SPARSELY', 9, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .8 }),
    text('isam-tree-eliminate-eq', 450, 248, 298, 56, texBlock§p(\delta\mid Z)\propto\prod_{C\in\mathcal T}p(\delta_{F_C}\mid\delta_{S_C})§, 13.2, { fontWeight: 700, align: 'center' }),
    text('isam-tree-eliminate-v', 458, 310, 282, 18, 'Each Bayes-tree clique stores frontal and separator variables.', 9.2, { color: C.soft, fontFamily: SANS, fontWeight: 700, align: 'center' }),

    card('isam-tree-update-card', 780, 206, 404, 132, C.poseSoft, { stroke: C.pose, strokeWidth: 2, radius: 8 }),
    text('isam-tree-update-k', 802, 222, 360, 16, '3 · EDIT ONLY THE AFFECTED TREE', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .8 }),
    text('isam-tree-update-v', 802, 250, 360, 62, 'New factors mark connected variables. Their cliques and ancestors are removed, relinearized/reordered as needed, re-eliminated, and attached back to the untouched tree.', 10.7, { color: C.poseDeep, fontFamily: SANS, fontWeight: 700, align: 'center', lineHeight: 1.35 }),

    text('isam-tree-arrow-1', 414, 252, 26, 28, '→', 20, { color: C.faint, fontFamily: SANS, fontWeight: 700, align: 'center' }),
    text('isam-tree-arrow-2', 756, 252, 26, 28, '→', 20, { color: C.faint, fontFamily: SANS, fontWeight: 700, align: 'center' }),

    card('isam-tree-visual-card', 96, 360, 570, 264, C.paper, { stroke: C.line, radius: 8 }),
    text('isam-tree-visual-k', 116, 376, 530, 16, 'BAYES TREE AFTER A NEW RADIO FACTOR ' + tex§f(\mathbf T_5,\mathbf m_B)§, 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .75 }),

    line('isam-tree-edge-root-left', 382, 428, 258, 486, C.soft, 2, { opacity: .72 }),
    line('isam-tree-edge-root-right', 382, 428, 504, 486, C.pose, 3, { opacity: .95 }),
    line('isam-tree-edge-left-child', 258, 486, 190, 552, C.soft, 2, { opacity: .5 }),
    line('isam-tree-edge-right-child', 504, 486, 548, 552, C.pose, 3, { opacity: .95 }),

    card('isam-tree-root', 310, 404, 144, 48, C.poseSoft, { stroke: C.pose, strokeWidth: 2, radius: 22 }),
    text('isam-tree-root-v', 322, 418, 120, 18, tex§C_0:\;m_A,m_B\mid T_3§, 9.5, { color: C.poseDeep, fontWeight: 700, align: 'center' }),
    card('isam-tree-left', 194, 462, 128, 48, '#FBFCFD', { stroke: C.line, radius: 22 }),
    text('isam-tree-left-v', 204, 476, 108, 18, tex§C_1:\;T_1,T_2\mid T_3§, 8.8, { color: C.soft, fontWeight: 700, align: 'center' }),
    card('isam-tree-right', 440, 462, 128, 48, C.poseSoft, { stroke: C.pose, strokeWidth: 2, radius: 22 }),
    text('isam-tree-right-v', 450, 476, 108, 18, tex§C_2:\;T_4\mid T_3,m_B§, 8.6, { color: C.poseDeep, fontWeight: 700, align: 'center' }),
    card('isam-tree-left-child', 126, 528, 128, 48, '#FBFCFD', { stroke: C.line, radius: 22 }),
    text('isam-tree-left-child-v', 136, 542, 108, 18, tex§C_3:\;T_0\mid T_1§, 8.8, { color: C.soft, fontWeight: 700, align: 'center' }),
    card('isam-tree-right-child', 484, 528, 128, 48, C.measurementSoft, { stroke: C.measurement, strokeWidth: 2, radius: 22 }),
    text('isam-tree-right-child-v', 494, 542, 108, 18, tex§C_4:\;T_5\mid T_4,m_B§, 8.6, { color: C.measurementDeep, fontWeight: 700, align: 'center' }),
    text('isam-tree-untouched', 120, 590, 210, 16, 'gray branch remains cached and untouched', 8.5, { color: C.faint, fontFamily: MONO, fontWeight: 700, align: 'center' }),
    text('isam-tree-affected', 406, 590, 240, 16, 'colored path is removed and re-eliminated', 8.5, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, align: 'center' }),

    card('isam-tree-policy-card', 688, 360, 496, 264, C.paper, { stroke: C.line, radius: 8 }),
    text('isam-tree-policy-k', 712, 376, 448, 16, 'FLUID RELINEARIZATION + INCREMENTAL REORDERING', 9, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .7 }),
    text('isam-tree-policy-eq', 704, 404, 464, 54, texBlock§\|\delta_k\|>\beta_k\;\Rightarrow\;\text{relinearize variable k and affected factors}§, 13, { fontWeight: 700, align: 'center' }),
    card('isam-tree-policy-1', 712, 470, 132, 74, C.poseSoft, { stroke: C.pose, radius: 6 }),
    text('isam-tree-policy-1-k', 724, 482, 108, 14, 'THRESHOLD', 8, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, align: 'center' }),
    text('isam-tree-policy-1-v', 722, 506, 112, 28, 'Set per variable type and physical units.', 9.2, { color: C.poseDeep, fontFamily: SANS, fontWeight: 700, align: 'center', lineHeight: 1.2 }),
    card('isam-tree-policy-2', 858, 470, 132, 74, C.mapSoft, { stroke: C.map, radius: 6 }),
    text('isam-tree-policy-2-k', 870, 482, 108, 14, 'SKIP', 8, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, align: 'center' }),
    text('isam-tree-policy-2-v', 868, 506, 112, 28, 'Check relinearization every N updates.', 9.2, { color: C.mapDeep, fontFamily: SANS, fontWeight: 700, align: 'center', lineHeight: 1.2 }),
    card('isam-tree-policy-3', 1004, 470, 132, 74, C.measurementSoft, { stroke: C.measurement, radius: 6 }),
    text('isam-tree-policy-3-k', 1016, 482, 108, 14, 'ORDERING', 8, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, align: 'center' }),
    text('isam-tree-policy-3-v', 1014, 506, 112, 28, 'Constrain new map variables to preserve sparsity.', 9.2, { color: C.measurementDeep, fontFamily: SANS, fontWeight: 700, align: 'center', lineHeight: 1.2 }),
    text('isam-tree-policy-warning', 712, 566, 448, 40, 'A global loop closure or dense factor can touch the root and make one update approach a batch solve. Incremental does not mean constant time.', 10.3, { color: C.soft, fontFamily: SANS, fontWeight: 700, align: 'center', lineHeight: 1.3 }),

    text('isam-tree-ref', 96, 653, 1088, 12, 'Refs · Kaess et al., IJRR 2012 · GTSAM ISAM2 documentation and ISAM2Params', 7, { color: C.faint, fontFamily: MONO, align: 'center' })
  ]

  return regular(
    's-isam2-bayes-tree', '04 · KNOWN BS POSE, UNKNOWN UE POSE AND MAP',
    'How iSAM2 updates the solution: edit the Bayes tree',
    'New radio factors trigger partial relinearization and re-elimination instead of rebuilding the full normal equations.',
    'Walk from nonlinear factors to the Bayes tree. At the current linearization point, selected factors are linearized. Sparse elimination produces conditionals organized as cliques. A new factor marks the cliques containing its variables; the affected cliques and their ancestors are removed, while disconnected subtrees remain cached. Variables whose delta exceeds a threshold are fluidly relinearized, and the affected subproblem may be reordered before re-elimination. This is the main computational contribution of iSAM2. Warn that a factor connecting distant parts of the graph may affect the root, so worst-case work can still approach a batch update.',
    elements, { accent: C.pose, titleSize: 32, transition: 'none' }
  )
}

function isam2RadioFactorsSlide(ctx) {
  const { regular, text, card, shape, line, C, SANS, MONO, tex, texBlock } = ctx
  const elements = [
    card('isam-factor-state-card', 96, 206, 344, 180, C.poseSoft, { stroke: C.pose, strokeWidth: 2, radius: 8 }),
    text('isam-factor-state-k', 118, 222, 300, 16, 'VARIABLE KEYS', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .9 }),
    text('isam-factor-state-eq', 108, 248, 320, 104, texBlock§\begin{aligned}
      X_t&:\mathbf T_t\in SE(2)\text{ or }SE(3),\\
      M_j&:\mathbf m_j\;\text{(VA, wall, scatterer, or primitive)},\\
      B_t&:\text{clock bias / delay offset (optional)},\\
      C&:\text{array or propagation calibration (optional)}.
    \end{aligned}§, 12.2, { fontWeight: 700, align: 'center', lineHeight: 1.35 }),
    text('isam-factor-state-v', 118, 358, 300, 16, 'Keep each factor connected to the smallest possible state subset.', 9.2, { color: C.poseDeep, fontFamily: SANS, fontWeight: 700, align: 'center' }),

    card('isam-factor-residual-card', 460, 206, 724, 180, C.paper, { stroke: C.line, radius: 8 }),
    text('isam-factor-residual-k', 484, 222, 410, 16, 'CUSTOM RADIO-PATH FACTOR', 9, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .9 }),
    text('isam-factor-residual-eq', 474, 246, 696, 108, texBlock§\begin{aligned}
      \mathbf r_{t\ell}^{\rm rad}(X_t,M_j;q)&=\begin{bmatrix}
      (c\tau_{t\ell}-L_q(X_t,M_j,\mathbf b))/\sigma_L\\
      \operatorname{wrap}(\varphi_{t\ell}^{\rm b}-\widehat\varphi_q^{\rm b}(X_t,M_j,\mathbf b))/\sigma_{\varphi}\\
      \operatorname{wrap}(\psi_{t\ell}^{\rm g}-\widehat\psi_q^{\rm g}(X_t,M_j,\mathbf b))/\sigma_{\psi}
      \end{bmatrix},\\[-.1em]
      f_{t\ell}^{\rm rad}&\propto\exp\!\left[-\tfrac12\rho(\|\mathbf r_{t\ell}^{\rm rad}\|^2)\right].
    \end{aligned}§, 11.3, { fontWeight: 700, align: 'center', lineHeight: 1.18 }),
    text('isam-factor-residual-v', 484, 356, 676, 16, 'q fixes the hypothesized ordered reflector chain; the geometry function must reject non-forward or invalid paths.', 9.2, { color: C.soft, fontFamily: SANS, fontWeight: 700, align: 'center' }),

    card('isam-factor-family-card', 96, 404, 344, 210, C.paper, { stroke: C.line, radius: 8 }),
    text('isam-factor-family-k', 118, 420, 300, 16, 'FACTOR FAMILIES IN THE GRAPH', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .8 }),
    ...[
      ['PRIOR', 'anchor gauge, heading, clock, or calibration'],
      ['BETWEEN', 'odometry / registration between poses'],
      ['RADIO', 'path length + AoA + AoD to one map entity'],
      ['REVISIT', 'verified nonconsecutive relative constraint'],
      ['SMOOTH', 'optional kinematic or regularization prior']
    ].flatMap((row, index) => {
      const y = 450 + index * 31
      return [
        text('isam-factor-family-head-' + index, 118, y, 74, 15, row[0], 8.2, { color: index === 2 ? C.measurementDeep : C.poseDeep, fontFamily: MONO, fontWeight: 700 }),
        text('isam-factor-family-v-' + index, 196, y - 1, 222, 24, row[1], 9.3, { color: C.soft, fontFamily: SANS, fontWeight: 700, lineHeight: 1.2 })
      ]
    }),

    card('isam-factor-assoc-card', 460, 404, 350, 210, C.measurementSoft, { stroke: C.measurement, radius: 8 }),
    text('isam-factor-assoc-k', 482, 420, 306, 16, 'ASSOCIATION IS A FACTOR-LIFECYCLE PROBLEM', 8.7, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .55 }),
    text('isam-factor-assoc-eq', 474, 446, 322, 42, texBlock§a_{t\ell}=j,\quad q_{t\ell}=q\quad\Longrightarrow\quad f_{t\ell}^{\rm rad}(X_t,M_j;q)§, 12.2, { color: C.measurementDeep, fontWeight: 700, align: 'center' }),
    text('isam-factor-assoc-v', 482, 492, 306, 102, 'Recommended staging:<br>1. gate and score hypotheses;<br>2. initialize only geometrically supported entities;<br>3. delay insertion or use robust/switchable factors for uncertain matches;<br>4. remove and replace rejected factors when the API and bookkeeping permit.', 9.8, { color: C.measurementDeep, fontFamily: SANS, fontWeight: 700, lineHeight: 1.35 }),

    card('isam-factor-impl-card', 830, 404, 354, 210, C.mapSoft, { stroke: C.map, radius: 8 }),
    text('isam-factor-impl-k', 852, 420, 310, 16, 'IMPLEMENTATION OF THE RADIO FACTOR', 9, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .7 }),
    text('isam-factor-impl-v', 852, 450, 310, 132, '• Prefer a typed C++ NonlinearFactor for production speed.<br>• A Python CustomFactor is suitable for prototyping but incurs Python-call/GIL overhead.<br>• Return analytical Jacobians in GTSAM’s right-perturbation convention.<br>• Unit-test every Jacobian against numerical derivatives.<br>• Use diagonal/full covariance in consistent units; wrap angle residuals only.', 9.8, { color: C.mapDeep, fontFamily: SANS, fontWeight: 700, lineHeight: 1.35 }),
    text('isam-factor-impl-foot', 852, 588, 310, 16, 'Sparse factor connectivity is as important as fast residual evaluation.', 8.8, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, align: 'center' }),

    text('isam-factor-ref', 96, 653, 1088, 12, 'Refs · GTSAM NonlinearFactor and CustomFactor documentation · robust incremental alternatives include riSAM for severe outliers', 7, { color: C.faint, fontFamily: MONO, align: 'center' })
  ]

  return regular(
    's-isam2-radio-factors', '04 · KNOWN BS POSE, UNKNOWN UE POSE AND MAP',
    'Translate radio geometry into sparse iSAM2 factors',
    'Each accepted MPC should become a small nonlinear factor with explicit residuals, covariance, association, and bounce order.',
    'Define stable keys for poses, persistent radio-map entities, and any clock or calibration variables. The radio factor should normally connect one pose to one compact map variable plus known BS parameters. Its residual contains normalized path-length and wrapped angular errors. Bounce order and association determine which geometry function and map key the factor uses; ordinary iSAM2 does not infer those discrete choices. Incorrect factors can corrupt the Bayes tree, so use conservative gating, delayed insertion, robust losses, switchable or graduated factors, or a separate multi-hypothesis front end. For implementation, prototype with GTSAM CustomFactor, but move performance-critical factors to typed C++ and validate analytical Jacobians numerically.',
    elements, { accent: C.pose, titleSize: 32, transition: 'none' }
  )
}

function isam2ImplementationSlide(ctx) {
  const { regular, text, card, shape, line, C, SANS, MONO, tex, texBlock } = ctx
  const elements = [
    card('isam-impl-loop-card', 96, 206, 526, 416, C.paper, { stroke: C.line, radius: 8 }),
    text('isam-impl-loop-k', 118, 222, 482, 16, 'GTSAM-STYLE ONLINE LOOP', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .9 }),
    text('isam-impl-loop-code', 118, 250, 482, 334,
      '<span style="font-family:Menlo,Consolas,monospace;font-size:11px;line-height:1.55">'
      + '<b>params</b> = ISAM2Params()<br>'
      + 'params.relinearizeThreshold = thresholds<br>'
      + 'params.relinearizeSkip = N<br>'
      + '<b>isam</b> = ISAM2(params)<br><br>'
      + '<b>for</b> scan t:<br>'
      + '&nbsp;&nbsp;new_factors = NonlinearFactorGraph()<br>'
      + '&nbsp;&nbsp;new_values  = Values()<br>'
      + '&nbsp;&nbsp;insert X(t) from odometry/registration<br>'
      + '&nbsp;&nbsp;add prior or BetweenFactor to X(t)<br>'
      + '&nbsp;&nbsp;<b>for</b> accepted MPC hypothesis (ℓ,j,q):<br>'
      + '&nbsp;&nbsp;&nbsp;&nbsp;<b>if</b> M(j) is new: initialize and insert M(j)<br>'
      + '&nbsp;&nbsp;&nbsp;&nbsp;add RadioPathFactor(X(t), M(j), z, q)<br>'
      + '&nbsp;&nbsp;result = isam.update(new_factors,new_values)<br>'
      + '&nbsp;&nbsp;estimate = isam.calculateEstimate()<br>'
      + '&nbsp;&nbsp;publish pose/map; retain graph state for t+1<br>'
      + '</span>',
      11, { color: C.ink, fontFamily: MONO, lineHeight: 1.45 }),
    card('isam-impl-loop-key', 118, 586, 482, 24, C.poseSoft, { stroke: C.pose, radius: 5 }),
    text('isam-impl-loop-key-v', 128, 591, 462, 14, 'All keys referenced by a new factor must already exist or be inserted in the same update.', 8.4, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, align: 'center' }),

    card('isam-impl-init-card', 646, 206, 256, 196, C.mapSoft, { stroke: C.map, radius: 8 }),
    text('isam-impl-init-k', 666, 222, 216, 16, 'INITIALIZATION', 9, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .8 }),
    text('isam-impl-init-v', 666, 252, 216, 128, 'Pose ' + tex§X_t§ + ': propagate registration/odometry estimate.<br><br>Map ' + tex§M_j§ + ': initialize from VA inversion, multi-pose triangulation, wall fitting, or a prior.<br><br>Do not collapse a one-path unobservable family to an arbitrary point; delay the birth or parameterize the family.', 9.7, { color: C.mapDeep, fontFamily: SANS, fontWeight: 700, lineHeight: 1.35 }),

    card('isam-impl-tune-card', 922, 206, 262, 196, C.poseSoft, { stroke: C.pose, radius: 8 }),
    text('isam-impl-tune-k', 942, 222, 222, 16, 'TUNING + DIAGNOSTICS', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .7 }),
    text('isam-impl-tune-v', 942, 252, 222, 128, '• per-type relinearization thresholds<br>• relinearization skip interval<br>• constrained ordering for new variables<br>• nonlinear error before/after update<br>• re-eliminated and relinearized counts<br>• marginal covariance for pose/map confidence<br>• factor-removal bookkeeping', 9.5, { color: C.poseDeep, fontFamily: SANS, fontWeight: 700, lineHeight: 1.32 }),

    card('isam-impl-current-card', 646, 422, 538, 96, C.measurementSoft, { stroke: C.measurement, radius: 8 }),
    text('isam-impl-current-k', 668, 438, 494, 16, 'HOW THIS REFRAMES THE CURRENT IMPLEMENTATION', 9, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .7 }),
    text('isam-impl-current-v', 668, 466, 494, 38, 'Current: position graph + map rebuilt outside the graph. Proposed iSAM2: persistent pose and map keys inside one graph, with radio factors added scan by scan and no full backend rebuild per frame.', 10.3, { color: C.measurementDeep, fontFamily: SANS, fontWeight: 700, align: 'center', lineHeight: 1.3 }),

    card('isam-impl-eval-card', 646, 536, 538, 86, C.paper, { stroke: C.line, radius: 8 }),
    text('isam-impl-eval-k', 668, 550, 494, 16, 'EXPERIMENTAL CLAIM TO TEST', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .8 }),
    text('isam-impl-eval-v', 668, 574, 494, 34, 'Hold factors, initialization, and robust loss fixed. Compare batch BA vs iSAM2 on final accuracy, per-scan latency, affected variables/cliques, memory, and sensitivity to delayed or wrong associations.', 9.6, { color: C.soft, fontFamily: SANS, fontWeight: 700, align: 'center', lineHeight: 1.28 }),

    card('isam-impl-key-card', 96, 636, 1088, 38, C.poseDeep, { stroke: C.poseDeep, radius: 6 }),
    text('isam-impl-key-v', 116, 641, 1048, 15, 'RESEARCH FRAMING · contribution = radio-specific variables, factors, association/initialization, and observability; iSAM2 supplies the scalable incremental optimizer.', 9.2, { color: C.paper, fontFamily: SANS, fontWeight: 700, align: 'center' }),
    text('isam-impl-ref', 116, 658, 1048, 10, 'Refs · GTSAM ISAM2 API · ISAM2Params · CustomFactor · Kaess et al., IJRR 2012', 6.8, { color: '#DDEEFF', fontFamily: MONO, align: 'center' })
  ]

  return regular(
    's-isam2-implementation', '04 · KNOWN BS POSE, UNKNOWN UE POSE AND MAP',
    'Implementation blueprint and evaluation plan',
    'Initialize new keys, add only new factors, update the Bayes tree, and benchmark against the identical batch objective.',
    'Describe the concrete GTSAM implementation. Construct one persistent ISAM2 object. For each scan, create temporary containers holding only new factors and initial values. Initialize the new pose from registration or odometry and new map entities from a geometrically supported estimator. Add prior, relative-pose, radio, and verified revisit factors, then call update and calculateEstimate. Tune relinearization by variable type and record update diagnostics. The scientifically fair benchmark compares iSAM2 with batch optimization using the same factors, noise, robust loss, associations, and initialization. This isolates incremental inference speed and approximation effects from changes in the radio model.',
    elements, { accent: C.pose, titleSize: 32, transition: 'none' }
  )
}`.replaceAll('§', BT)

  source = source.replace(graphMarker, isam2Slides + '\n\n' + graphMarker)
  changed = true
}

if (!source.includes('ctx.slides.push(isam2FramingSlide(ctx))')) {
  const sequenceBefore = "    ctx.slides.push(bundleAdjustmentSlide(ctx))\n    ctx.slides.push(graphEquationSlide(ctx))"
  const sequenceAfter = "    ctx.slides.push(bundleAdjustmentSlide(ctx))\n    ctx.slides.push(isam2FramingSlide(ctx))\n    ctx.slides.push(isam2BayesTreeSlide(ctx))\n    ctx.slides.push(isam2RadioFactorsSlide(ctx))\n    ctx.slides.push(isam2ImplementationSlide(ctx))\n    ctx.slides.push(graphEquationSlide(ctx))"
  if (!source.includes(sequenceBefore)) throw new Error('Could not find bundle-adjustment to GraphSLAM sequence')
  source = source.replace(sequenceBefore, sequenceAfter)
  changed = true
}

for (const marker of [
  'function isam2FramingSlide(ctx) {',
  'function isam2BayesTreeSlide(ctx) {',
  'function isam2RadioFactorsSlide(ctx) {',
  'function isam2ImplementationSlide(ctx) {',
  "'s-isam2-framing'",
  "'s-isam2-bayes-tree'",
  "'s-isam2-radio-factors'",
  "'s-isam2-implementation'",
  'ctx.slides.push(isam2FramingSlide(ctx))'
]) {
  if (!source.includes(marker)) throw new Error(`iSAM2 patch validation failed: ${marker}`)
}

if (changed) {
  writeFileSync(sourcePath, source)
  console.log(`Inserted iSAM2 framing slides into ${sourcePath}`)
} else {
  console.log(`No iSAM2 changes needed in ${sourcePath}`)
}
