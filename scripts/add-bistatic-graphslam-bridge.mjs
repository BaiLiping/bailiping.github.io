import { readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

const sourcePath = resolve('mpc-detection-to-bounce-count-slides/radio-slam-extra.mjs')
const buildPath = resolve('mpc-detection-to-bounce-count-slides/build.mjs')
let source = readFileSync(sourcePath, 'utf8')
let build = readFileSync(buildPath, 'utf8')
let sourceChanged = false
let buildChanged = false

const BT = '`'
const functionMarker = 'function bundleAdjustmentSlide(ctx) {'

if (!source.includes('function bistaticGraphSlamBridgeSlide(ctx) {')) {
  if (!source.includes(functionMarker)) throw new Error('Could not find bundle-adjustment slide insertion marker')

  const bridgeSlides = String.raw`function bistaticGraphSlamBridgeSlide(ctx) {
  const { regular, text, card, shape, line, C, SANS, MONO, tex, texBlock } = ctx
  const elements = [
    card('bi-compare-onboard-card', 96, 202, 486, 350, C.paper, { stroke: C.line, radius: 8 }),
    text('bi-compare-onboard-k', 120, 220, 438, 18, 'COLLOCATED / ONBOARD MEASUREMENT', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .85 }),
    text('bi-compare-onboard-sub', 120, 246, 438, 30, 'The sensor pose is the robot pose.', 11, { color: C.soft, fontFamily: SANS, fontWeight: 700, align: 'center' }),
    line('bi-onboard-edge-map', 338, 330, 338, 382, C.map, 2),
    line('bi-onboard-edge-pose', 338, 410, 338, 466, C.pose, 2),
    shape('bi-onboard-map', 318, 302, 40, 40, C.mapSoft, { shape: 'ellipse', stroke: C.map, strokeWidth: 2 }),
    text('bi-onboard-map-v', 316, 313, 44, 18, tex§\mathbf m_j§, 10, { color: C.mapDeep, fontWeight: 700, align: 'center' }),
    shape('bi-onboard-factor', 327, 382, 22, 22, C.measurementSoft, { stroke: C.measurement, strokeWidth: 2, radius: 0 }),
    text('bi-onboard-factor-v', 272, 410, 132, 18, tex§f(\mathbf T_t,\mathbf m_j)§, 9, { color: C.measurementDeep, fontWeight: 700, align: 'center' }),
    shape('bi-onboard-pose', 318, 466, 40, 40, C.poseSoft, { shape: 'ellipse', stroke: C.pose, strokeWidth: 2 }),
    text('bi-onboard-pose-v', 316, 477, 44, 18, tex§\mathbf T_t§, 10, { color: C.poseDeep, fontWeight: 700, align: 'center' }),
    text('bi-onboard-eq', 120, 510, 438, 28, texBlock§\mathbf z_{tj}=\mathbf h(\mathbf T_t,\mathbf m_j)+\boldsymbol\varepsilon§, 12.5, { fontWeight: 700, align: 'center' }),

    card('bi-compare-radio-card', 606, 202, 578, 350, C.poseSoft, { stroke: C.pose, strokeWidth: 2, radius: 8 }),
    text('bi-compare-radio-k', 630, 220, 530, 18, 'RADIO SLAM · DISTINCT BS AND UE ENDPOINTS', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .85 }),
    text('bi-compare-radio-sub', 630, 246, 530, 30, 'The known transmitter enters the radio likelihood as a fixed parameter.', 11, { color: C.poseDeep, fontFamily: SANS, fontWeight: 700, align: 'center' }),
    shape('bi-radio-bs', 682, 316, 36, 36, C.ink, { shape: 'ellipse', stroke: C.paper, strokeWidth: 2 }),
    text('bi-radio-bs-v', 664, 291, 72, 18, tex§\mathbf B_s§, 10, { color: C.ink, fontWeight: 700, align: 'center' }),
    text('bi-radio-bs-k', 646, 354, 108, 16, 'known constant', 8, { color: C.faint, fontFamily: MONO, fontWeight: 700, align: 'center' }),
    shape('bi-radio-map', 1036, 316, 40, 40, C.mapSoft, { shape: 'ellipse', stroke: C.map, strokeWidth: 2 }),
    text('bi-radio-map-v', 1034, 327, 44, 18, tex§\mathbf m_j§, 10, { color: C.mapDeep, fontWeight: 700, align: 'center' }),
    shape('bi-radio-factor', 873, 392, 24, 24, C.measurementSoft, { stroke: C.measurement, strokeWidth: 2, radius: 0 }),
    text('bi-radio-factor-v', 806, 420, 158, 18, tex§f^{\rm rad}_{ts\ell}§, 9, { color: C.measurementDeep, fontWeight: 700, align: 'center' }),
    shape('bi-radio-pose', 865, 476, 40, 40, C.paper, { shape: 'ellipse', stroke: C.pose, strokeWidth: 2 }),
    text('bi-radio-pose-v', 863, 487, 44, 18, tex§\mathbf T_t§, 10, { color: C.poseDeep, fontWeight: 700, align: 'center' }),
    line('bi-radio-edge-bs', 718, 340, 873, 398, C.known, 2, { opacity: .58 }),
    line('bi-radio-edge-map', 1036, 342, 897, 398, C.map, 2, { opacity: .72 }),
    line('bi-radio-edge-pose', 885, 416, 885, 476, C.pose, 2),
    text('bi-radio-eq', 626, 510, 538, 28, texBlock§\mathbf z_{ts\ell}=\mathbf h_q(\mathbf T_t,\mathbf m_j;\mathbf B_s,\boldsymbol\kappa)+\boldsymbol\varepsilon§, 12.5, { fontWeight: 700, align: 'center' }),

    card('bi-rule-card', 96, 570, 1088, 70, C.measurementSoft, { stroke: C.measurement, radius: 8 }),
    text('bi-rule-k', 118, 585, 176, 16, 'THE “CONVERSION”', 9, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .95 }),
    text('bi-rule-v', 300, 578, 860, 38, 'Keep the same pose–map graph. Absorb the known BS into the measurement function; promote it to a variable node only when its pose, orientation, clock, or calibration is uncertain.', 12, { color: C.measurementDeep, fontFamily: SANS, fontWeight: 700, align: 'center', valign: 'middle', lineHeight: 1.3 }),
    text('bi-rule-ref', 118, 622, 1042, 10, 'Ref · Dellaert & Kaess, Factor Graphs for Robot Perception, 2017', 6.8, { color: C.measurementDeep, fontFamily: MONO, align: 'center' })
  ]

  return regular(
    's-bistatic-graphslam-bridge', '04 · BISTATIC RADIO → GRAPHSLAM',
    'Radio SLAM is bistatic; GraphSLAM is sensor-agnostic',
    'The off-board transmitter changes the measurement model—not the factor-graph machinery.',
    'Correct the common misconception that GraphSLAM requires every sensor to be physically onboard. A factor graph only records which unknowns enter each likelihood. In downlink radio SLAM, the BS and UE are distinct endpoints, but a calibrated BS pose can be conditioned on as a fixed parameter. The resulting factor still connects the UE pose to the persistent radio-map entity. If a BS pose, array orientation, clock, or hardware delay is uncertain, make that quantity a variable and attach an appropriate prior or calibration factor.',
    elements, { accent: C.pose, titleSize: 33, transition: 'none' }
  )
}

function bistaticMpcFactorSlide(ctx) {
  const { regular, text, card, C, SANS, MONO, tex, texBlock } = ctx
  const elements = [
    card('bi-mpc-geometry-card', 96, 202, 650, 250, C.paper, { stroke: C.line, radius: 8 }),
    text('bi-mpc-geometry-k', 120, 220, 602, 18, 'POINT-SCATTERER EXAMPLE · EXPLICIT ANGLE CONVENTION', 9, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .75 }),
    text('bi-mpc-geometry-eq', 112, 248, 618, 156, texBlock§\begin{aligned}
      L_{tsj}&=\|\mathbf s_j-\mathbf b_s\|+\|\mathbf p_t-\mathbf s_j\|,\\
      \widehat\tau_{tsj}&=L_{tsj}/c+\delta_{ts},\\
      \widehat\psi^{\rm AoD}_{tsj}&=\operatorname{wrap}_{\pi}\!\big(\angle(\mathbf s_j-\mathbf b_s)-\theta_s\big),\\
      \widehat\varphi^{\rm AoA}_{tsj}&=\operatorname{wrap}_{\pi}\!\big(\angle(\mathbf s_j-\mathbf p_t)-\theta_t\big).
    \end{aligned}§, 15, { fontWeight: 700, align: 'center', lineHeight: 1.35 }),
    text('bi-mpc-geometry-v', 120, 412, 602, 24, 'Convention used here: AoA points from the UE toward the previous interaction/source.', 10, { color: C.soft, fontFamily: SANS, fontWeight: 700, align: 'center' }),

    card('bi-mpc-residual-card', 770, 202, 414, 250, C.measurementSoft, { stroke: C.measurement, strokeWidth: 2, radius: 8 }),
    text('bi-mpc-residual-k', 792, 220, 370, 18, 'ONE SPARSE RADIO FACTOR', 9, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .85 }),
    text('bi-mpc-residual-eq', 784, 252, 386, 92, texBlock§\mathbf r^{\rm geom}_{ts\ell}=\begin{bmatrix}
      c(\tau_{ts\ell}-\delta_{ts})-\widehat L_q\\
      \operatorname{wrap}_{\pi}(\varphi_{ts\ell}-\widehat\varphi_q)\\
      \operatorname{wrap}_{\pi}(\psi_{ts\ell}-\widehat\psi_q)
    \end{bmatrix}§, 13.2, { fontWeight: 700, align: 'center', lineHeight: 1.28 }),
    text('bi-mpc-residual-v', 792, 356, 370, 66, tex§f^{\rm rad}_{ts\ell}\propto\exp\!\left[-\tfrac12\|\mathbf r^{\rm geom}_{ts\ell}\|^2_{\Sigma^{-1}_{ts\ell}}\right]§, 11.5, { color: C.measurementDeep, fontWeight: 700, align: 'center', valign: 'middle' }),
    text('bi-mpc-residual-n', 792, 426, 370, 14, 'The factor touches only the unknown pose and associated map variable(s).', 8.2, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, align: 'center' }),

    card('bi-mpc-clock-card', 96, 474, 334, 144, C.poseSoft, { stroke: C.pose, radius: 8 }),
    text('bi-mpc-clock-k', 116, 490, 294, 16, 'DELAY / CLOCK', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .9 }),
    text('bi-mpc-clock-v', 116, 518, 294, 72, 'Only a synchronized and hardware-calibrated delay obeys L=cτ. Otherwise estimate δ, add clock factors, or form delay differences.', 11, { color: C.poseDeep, fontFamily: SANS, fontWeight: 700, align: 'center', lineHeight: 1.35 }),

    card('bi-mpc-angle-card', 450, 474, 334, 144, C.paper, { stroke: C.line, radius: 8 }),
    text('bi-mpc-angle-k', 470, 490, 294, 16, 'AOA SIGN / FRAME', 9, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .9 }),
    text('bi-mpc-angle-v', 470, 518, 294, 72, 'State whether AoA is the direction of arrival or the forward propagation vector. The two conventions differ by π; both must be rotated by the UE heading.', 10.8, { color: C.soft, fontFamily: SANS, fontWeight: 700, align: 'center', lineHeight: 1.35 }),

    card('bi-mpc-gain-card', 804, 474, 380, 144, C.mapSoft, { stroke: C.map, radius: 8 }),
    text('bi-mpc-gain-k', 824, 490, 340, 16, 'PATH GAIN / PATH LOSS', 9, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .9 }),
    text('bi-mpc-gain-v', 824, 514, 340, 82, tex§r_g=g^{\rm dB}-\gamma_q(\mathbf T,\mathcal M,\mathbf B;\boldsymbol\xi)§ + '<br>Use this residual only with calibrated transmit power, antenna patterns, material coefficients, and blockage terms; otherwise use gain for gating/association.', 9.8, { color: C.mapDeep, fontFamily: SANS, fontWeight: 700, align: 'center', lineHeight: 1.3 }),

    text('bi-mpc-ref', 96, 648, 1088, 12, 'Refs · Leitinger et al., IEEE TWC 2019 · Li et al., IEEE OJ-COMS 2024', 7, { color: C.faint, fontFamily: MONO, align: 'center' })
  ]

  return regular(
    's-bistatic-mpc-factor', '04 · BISTATIC RADIO → GRAPHSLAM',
    'One MPC becomes one sparse bistatic factor',
    'Delay, AoA, and AoD are predicted jointly from the BS–map–UE path; gain is a separate calibrated radiometric model.',
    'Use the point-scatterer path to make the factor explicit. The path length is the sum of the BS-to-scatterer and scatterer-to-UE legs. AoD is expressed in the BS array frame, while AoA is expressed in the UE array frame. This slide adopts the direction-from-which-the-wave-arrives convention for AoA. If the estimator uses the forward wave-vector convention, shift by pi. The clock or hardware-delay term must be included unless synchronization/calibration makes it zero. Received path gain or path loss is not a pure geometric range measurement; include it as a residual only after defining a calibrated propagation and nuisance-parameter model.',
    elements, { accent: C.measurement, titleSize: 33, transition: 'none' }
  )
}

function bistaticVirtualAnchorSlide(ctx) {
  const { regular, text, card, shape, line, C, SANS, MONO, tex, texBlock } = ctx
  const elements = [
    card('bi-va-scene-card', 96, 202, 486, 416, C.paper, { stroke: C.line, radius: 8 }),
    text('bi-va-scene-k', 118, 220, 442, 18, 'ONE SPECULAR WALL · IMAGE-SOURCE CONSTRUCTION', 9, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .65 }),
    line('bi-va-wall', 350, 268, 350, 570, C.map, 5),
    text('bi-va-wall-v', 306, 248, 88, 16, tex§\mathcal W_j§, 10, { color: C.mapDeep, fontWeight: 700, align: 'center' }),
    shape('bi-va-bs', 170, 302, 18, 18, C.ink, { radius: 0 }),
    text('bi-va-bs-v', 132, 326, 94, 16, 'physical BS', 8.5, { color: C.ink, fontFamily: MONO, fontWeight: 700, align: 'center' }),
    shape('bi-va-virtual', 512, 302, 18, 18, C.mapSoft, { shape: 'ellipse', stroke: C.map, strokeWidth: 2 }),
    text('bi-va-virtual-v', 468, 326, 106, 16, 'virtual anchor', 8.5, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, align: 'center' }),
    shape('bi-va-ue', 188, 536, 20, 20, C.pose, { shape: 'ellipse', stroke: C.paper, strokeWidth: 2 }),
    text('bi-va-ue-v', 160, 562, 76, 16, 'UE pose', 8.5, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, align: 'center' }),
    shape('bi-va-reflection', 342, 424, 16, 16, C.measurementSoft, { shape: 'ellipse', stroke: C.measurement, strokeWidth: 3 }),
    text('bi-va-reflection-v', 360, 424, 94, 16, tex§\mathbf r_{tsj}§, 9, { color: C.measurementDeep, fontWeight: 700 }),
    line('bi-va-physical-a', 179, 311, 350, 432, C.measurement, 3),
    line('bi-va-physical-b', 350, 432, 198, 546, C.measurement, 3),
    line('bi-va-unfolded', 198, 546, 521, 311, C.map, 2, { opacity: .42 }),
    text('bi-va-unfolded-v', 374, 486, 154, 30, 'unfolded straight path', 8.5, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, align: 'center' }),
    text('bi-va-scene-foot', 118, 590, 442, 16, 'Fold the UE→VA line at the wall to recover the physical reflection point.', 8.5, { color: C.faint, fontFamily: SANS, fontWeight: 700, align: 'center' }),

    card('bi-va-eq-card', 606, 202, 578, 224, C.mapSoft, { stroke: C.map, strokeWidth: 2, radius: 8 }),
    text('bi-va-eq-k', 630, 220, 530, 18, 'WALL → VIRTUAL ANCHOR → PREDICTED MPC', 9, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .7 }),
    text('bi-va-eq-v', 620, 246, 550, 148, texBlock§\begin{aligned}
      \mathcal W_j&=\{\mathbf x:\mathbf n_j^{\mathsf T}\mathbf x=d_j\},\qquad\|\mathbf n_j\|=1,\\
      \mathbf v_{sj}&=\mathbf b_s+2(d_j-\mathbf n_j^{\mathsf T}\mathbf b_s)\mathbf n_j,\\
      \widehat L_{tsj}&=\|\mathbf p_t-\mathbf v_{sj}\|,\\
      \widehat\varphi^{\rm AoA}_{tsj}&=\operatorname{wrap}_{\pi}\!\big(\angle(\mathbf v_{sj}-\mathbf p_t)-\theta_t\big),\\
      \mathbf r_{tsj}&=[\mathbf p_t,\mathbf v_{sj}]\cap\mathcal W_j,\quad
      \widehat\psi^{\rm AoD}_{tsj}=\operatorname{wrap}_{\pi}\!\big(\angle(\mathbf r_{tsj}-\mathbf b_s)-\theta_s\big).
    \end{aligned}§, 12.4, { fontWeight: 700, align: 'center', lineHeight: 1.28 }),

    card('bi-va-param-card', 606, 446, 278, 172, C.paper, { stroke: C.line, radius: 8 }),
    text('bi-va-param-k', 626, 462, 238, 16, 'MAP PARAMETERIZATION', 8.5, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .7 }),
    text('bi-va-param-v', 626, 490, 238, 104, 'VA node: compact for one BS + one specular wall.<br><br>Wall node: shares one physical surface across several BSs and paths.', 10.5, { color: C.soft, fontFamily: SANS, fontWeight: 700, align: 'center', lineHeight: 1.35 }),

    card('bi-va-valid-card', 906, 446, 278, 172, C.measurementSoft, { stroke: C.measurement, radius: 8 }),
    text('bi-va-valid-k', 926, 462, 238, 16, 'VALIDITY + HIGHER ORDER', 8.5, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .7 }),
    text('bi-va-valid-v', 926, 488, 238, 108, 'Require finite-wall support, visibility, and positive ordered legs.<br><br>Multiple bounces use an ordered reflector chain or a higher-arity factor—not one generic landmark.', 10, { color: C.measurementDeep, fontFamily: SANS, fontWeight: 700, align: 'center', lineHeight: 1.32 }),

    text('bi-va-ref', 606, 648, 578, 12, 'Image-source / virtual-anchor model · exact for ideal specular reflection', 7, { color: C.faint, fontFamily: MONO, align: 'center' })
  ]

  return regular(
    's-bistatic-virtual-anchor', '04 · BISTATIC RADIO → GRAPHSLAM',
    'Specular reflection becomes a landmark through a virtual anchor',
    'Unfold the known BS across a wall, then use the resulting VA inside an ordinary pose–landmark factor.',
    'For an ideal one-bounce specular path, reflecting the known BS across a wall converts the broken path into a straight UE-to-virtual-anchor segment. The VA therefore behaves like a static landmark for path length and AoA. AoD is recovered by intersecting the unfolded segment with the wall and evaluating the BS-to-reflection-point bearing. A VA is convenient, but a wall parameterization is often better when the same physical surface must be shared across multiple BSs. For multiple bounces, retain the ordered reflector chain or use a composite state with explicit observability caveats. Always reject unfolded solutions whose folded reflection points lie outside finite surfaces, behind rays, or behind occluders.',
    elements, { accent: C.map, titleSize: 32, transition: 'none' }
  )
}

function bistaticGraphConstructionSlide(ctx) {
  const { regular, text, card, shape, line, C, SANS, MONO, tex, texBlock } = ctx
  const graph = [
    card('bi-gc-graph-card', 96, 202, 568, 416, C.paper, { stroke: C.line, radius: 8 }),
    text('bi-gc-graph-k', 118, 220, 524, 18, 'RESULTING SPARSE FACTOR GRAPH', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .85 }),
    shape('bi-gc-bs', 352, 274, 28, 28, C.ink, { shape: 'ellipse', stroke: C.paper, strokeWidth: 2 }),
    text('bi-gc-bs-v', 308, 248, 116, 18, tex§\mathbf B_s\;\text{known}§, 9, { color: C.ink, fontWeight: 700, align: 'center' }),
    shape('bi-gc-map-a', 190, 350, 36, 36, C.mapSoft, { shape: 'ellipse', stroke: C.map, strokeWidth: 2 }),
    text('bi-gc-map-a-v', 186, 360, 44, 18, tex§\mathbf m_A§, 9, { color: C.mapDeep, fontWeight: 700, align: 'center' }),
    shape('bi-gc-map-b', 500, 350, 36, 36, C.knownSoft, { shape: 'ellipse', stroke: C.known, strokeWidth: 2 }),
    text('bi-gc-map-b-v', 496, 360, 44, 18, tex§\mathbf m_B§, 9, { color: C.knownDeep, fontWeight: 700, align: 'center' })
  ]
  const poseXs = [150, 282, 414, 546]
  const poseLabels = [tex§\mathbf T_1§, tex§\mathbf T_2§, tex§\mathbf T_3§, tex§\mathbf T_4§]
  poseXs.forEach((x, index) => {
    if (index < poseXs.length - 1) {
      const fx = 0.5 * (x + poseXs[index + 1])
      graph.push(line('bi-gc-rel-a-' + index, x + 16, 536, fx - 7, 536, C.soft, 2))
      graph.push(line('bi-gc-rel-b-' + index, fx + 7, 536, poseXs[index + 1] - 16, 536, C.soft, 2))
      graph.push(shape('bi-gc-rel-f-' + index, fx - 7, 529, 14, 14, C.paper, { stroke: C.soft, strokeWidth: 2, radius: 0 }))
    }
    graph.push(shape('bi-gc-pose-' + index, x - 16, 520, 32, 32, index === 0 ? C.poseSoft : C.paper, { shape: 'ellipse', stroke: C.pose, strokeWidth: 2 }))
    graph.push(text('bi-gc-pose-v-' + index, x - 18, 527, 36, 18, poseLabels[index], 8.5, { color: C.poseDeep, fontWeight: 700, align: 'center' }))
  })
  const radios = [
    { x: 178, y: 442, pose: 0, mapX: 208, mapY: 368, color: C.map },
    { x: 326, y: 430, pose: 1, mapX: 208, mapY: 368, color: C.map },
    { x: 400, y: 430, pose: 2, mapX: 518, mapY: 368, color: C.known },
    { x: 548, y: 442, pose: 3, mapX: 518, mapY: 368, color: C.known }
  ]
  radios.forEach((item, index) => {
    graph.push(line('bi-gc-rad-map-' + index, item.x, item.y - 7, item.mapX, item.mapY + 18, item.color, 1.8, { opacity: .62 }))
    graph.push(line('bi-gc-rad-pose-' + index, item.x, item.y + 7, poseXs[item.pose], 520, C.measurement, 1.8, { opacity: .68 }))
    graph.push(line('bi-gc-rad-bs-' + index, item.x, item.y - 7, 366, 302, C.known, 1.2, { opacity: .28 }))
    graph.push(shape('bi-gc-rad-f-' + index, item.x - 7, item.y - 7, 14, 14, C.measurementSoft, { stroke: C.measurement, strokeWidth: 2, radius: 0 }))
  })
  graph.push(text('bi-gc-legend', 118, 580, 524, 20, 'Known BS = conditioned parameter · circles = unknowns · squares = factors', 8.2, { color: C.faint, fontFamily: MONO, fontWeight: 700, align: 'center' }))

  const elements = [
    ...graph,
    card('bi-gc-objective-card', 688, 202, 496, 180, C.poseSoft, { stroke: C.pose, strokeWidth: 2, radius: 8 }),
    text('bi-gc-objective-k', 712, 220, 448, 18, 'ORDINARY GRAPHSLAM OBJECTIVE', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .85 }),
    text('bi-gc-objective-eq', 700, 246, 472, 104, texBlock§\begin{aligned}
      \boldsymbol\Theta&=\{\mathbf T_{1:T},\mathcal M,\boldsymbol\kappa,\mathbf B_{\rm uncertain}\},\\[-.1em]
      \boldsymbol\Theta^*&=\arg\min_{\boldsymbol\Theta}\;\sum_i\|\mathbf r_i^{\rm prior}\|^2_{\Omega_i}
      +\sum_t\|\mathbf r_t^{\rm rel}\|^2_{\Omega_t}\\[-.1em]
      &\qquad+\sum_{t,s,\ell}\rho\!\left(\|\mathbf r_{ts\ell}^{\rm rad}\|^2_{\Omega_{ts\ell}}\right).
    \end{aligned}§, 12.2, { fontWeight: 700, align: 'center', lineHeight: 1.28 }),
    text('bi-gc-objective-v', 712, 354, 448, 16, 'Omit variables that are known; add factors only for measurements actually available.', 8.7, { color: C.poseDeep, fontFamily: SANS, fontWeight: 700, align: 'center' }),

    card('bi-gc-assoc-card', 688, 402, 238, 128, C.measurementSoft, { stroke: C.measurement, radius: 8 }),
    text('bi-gc-assoc-k', 708, 418, 198, 16, 'DISCRETE FRONT END', 8.5, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .7 }),
    text('bi-gc-assoc-v', 708, 446, 198, 62, 'Association, LoS/NLoS class, bounce order, visibility, and births/deaths are fixed, alternated, marginalized, or represented by mixture factors.', 9.8, { color: C.measurementDeep, fontFamily: SANS, fontWeight: 700, align: 'center', lineHeight: 1.3 }),

    card('bi-gc-gauge-card', 946, 402, 238, 128, C.mapSoft, { stroke: C.map, radius: 8 }),
    text('bi-gc-gauge-k', 966, 418, 198, 16, 'GAUGE ≠ OBSERVABILITY', 8.5, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .6 }),
    text('bi-gc-gauge-v', 966, 446, 198, 62, 'A known BS pose defines the global frame, but corridor, reflector-chain, clock, or heading null spaces can still remain.', 10, { color: C.mapDeep, fontFamily: SANS, fontWeight: 700, align: 'center', lineHeight: 1.3 }),

    card('bi-gc-solver-card', 688, 550, 496, 68, C.paper, { stroke: C.line, radius: 8 }),
    text('bi-gc-solver-k', 708, 566, 128, 16, 'SAME GRAPH', 8.5, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .8 }),
    text('bi-gc-solver-v', 836, 558, 328, 40, 'Batch Gauss–Newton / LM, Schur-complement BA, or incremental iSAM2 differ in solution strategy—not in whether the transmitter is onboard.', 10.4, { color: C.soft, fontFamily: SANS, fontWeight: 700, align: 'center', valign: 'middle', lineHeight: 1.28 }),

    card('bi-gc-key-card', 96, 636, 1088, 38, C.poseDeep, { stroke: C.poseDeep, radius: 6 }),
    text('bi-gc-key-v', 116, 644, 1048, 18, 'BOTTOM LINE · bistatic geometry changes h(·) and factor arity; it does not require a different SLAM backend.', 9.6, { color: C.paper, fontFamily: SANS, fontWeight: 700, align: 'center' })
  ]

  return regular(
    's-bistatic-graph-construction', '04 · BISTATIC RADIO → GRAPHSLAM',
    'The resulting graph is ordinary GraphSLAM',
    'Pose variables, persistent map variables, optional calibration states, and sparse radio factors are optimized together.',
    'Close the bridge by showing the complete graph. The known BS is a conditioned parameter feeding many radio factors; it is not an unknown node unless calibration is part of the problem. Re-observing a persistent reflector or virtual anchor couples distant UE poses just as re-observing a visual landmark does. Relative-pose or odometry factors are optional additional information, not a requirement for calling the formulation GraphSLAM. Fixing association and bounce order gives standard nonlinear least squares. Unknown discrete hypotheses require a front end, alternating inference, marginalization, or mixture/max-mixture style factors. A known BS pose removes the free choice of global frame, but it cannot remove genuine geometry-induced or clock-induced null spaces.',
    elements, { accent: C.pose, titleSize: 33, transition: 'none' }
  )
}`.replaceAll('§', BT)

  source = source.replace(functionMarker, bridgeSlides + '\n\n' + functionMarker)
  sourceChanged = true
}

if (!source.includes('ctx.slides.push(bistaticGraphSlamBridgeSlide(ctx))')) {
  const callMarker = '    ctx.slides.push(bundleAdjustmentSlide(ctx))'
  if (!source.includes(callMarker)) throw new Error('Could not find Section 04 bundle-adjustment call')
  const calls = [
    '    ctx.slides.push(bistaticGraphSlamBridgeSlide(ctx))',
    '    ctx.slides.push(bistaticMpcFactorSlide(ctx))',
    '    ctx.slides.push(bistaticVirtualAnchorSlide(ctx))',
    '    ctx.slides.push(bistaticGraphConstructionSlide(ctx))'
  ].join('\n') + '\n'
  source = source.replace(callMarker, calls + callMarker)
  sourceChanged = true
}

function replaceSourceRequired(before, after, label) {
  if (source.includes(after)) return
  if (!source.includes(before)) throw new Error(`Could not find ${label}`)
  source = source.replace(before, after)
  sourceChanged = true
}

replaceSourceRequired(
  'The radio analogue minimizes path-length, UE-frame AoA, and BS-frame AoD residuals over UE poses and persistent radio-map entities.',
  'The radio analogue minimizes path-length, UE-frame AoA, and BS-frame AoD residuals over UE poses and persistent radio-map entities. Delay becomes path length only after clock and hardware-delay calibration; gain enters the objective only through an explicit radiometric model.',
  'bundle-adjustment measurement-model clarification'
)

function replaceBuildRequired(before, after, label) {
  if (build.includes(after)) return
  if (!build.includes(before)) throw new Error(`Could not find ${label}`)
  build = build.replace(before, after)
  buildChanged = true
}

replaceBuildRequired(
  'delay ${tex`\\tau`} → ${tex`L=c\\tau`}<br>AoA ${tex`\\varphi`} → arrival bearing at the UE<br>AoD ${tex`\\psi`} → departure bearing at the BS<br>${tex`\\alpha\\in\\mathbb C`} → complex MPC gain; ${tex`|\\alpha|^2`} → power gain',
  'delay ${tex`\\tau`} → ${tex`L=c(\\tau-\\delta_\\tau)`}; synchronized: ${tex`\\delta_\\tau=0`}<br>AoA ${tex`\\varphi`} → arrival bearing in the UE frame<br>AoD ${tex`\\psi`} → departure bearing in the BS frame<br>${tex`\\alpha\\in\\mathbb C`} → complex MPC gain; ${tex`|\\alpha|^2`} → power gain',
  'measurement tuple delay and angle-frame mapping'
)

replaceBuildRequired(
  'Delay becomes path length, AoA and AoD are local until headings are known, and the calibrated complex MPC gain remains radiometric evidence rather than a direct bounce counter.',
  'After synchronization and hardware-delay calibration, delay becomes path length. AoA and AoD must use explicit UE/BS array-frame conventions, and the calibrated complex MPC gain remains radiometric evidence rather than a direct bounce counter.',
  'measurement-slide speaker-note clarification'
)

replaceBuildRequired(
  '${tex`-10\\log_{10}|\\alpha|^2`} → path loss (dB)',
  '${tex`-10\\log_{10}|\\alpha|^2`} → calibrated / normalized path loss (dB)',
  'path-loss normalization label'
)

replaceBuildRequired(
  "'The PDP separates paths; calibrated gain grades them'",
  "'When delay-resolved, the PDP separates paths'",
  'PDP resolvability title'
)

replaceBuildRequired(
  'Each resolved peak contributes one tuple, while the calibrated power gain from the squared magnitude of the complex MPC gain can penalize implausible material, roughness, interaction-count, or blockage hypotheses.',
  'Each resolvable peak contributes one tuple; components inside the delay resolution remain superposed. The calibrated power gain from the squared magnitude of the complex MPC gain can penalize implausible material, roughness, interaction-count, or blockage hypotheses.',
  'PDP resolution clarification'
)

replaceBuildRequired(
  "['01', 'interactions', 'more bounces usually spend more power']",
  "['01', 'interactions', 'each interaction often adds loss—not a strict ordering']",
  'non-monotonic bounce-loss wording'
)

for (const marker of [
  'function bistaticGraphSlamBridgeSlide(ctx) {',
  "'s-bistatic-graphslam-bridge'",
  "'s-bistatic-mpc-factor'",
  "'s-bistatic-virtual-anchor'",
  "'s-bistatic-graph-construction'",
  'ctx.slides.push(bistaticGraphConstructionSlide(ctx))',
  'GraphSLAM is sensor-agnostic',
  'Only a synchronized and hardware-calibrated delay obeys L=cτ',
  'Multiple bounces use an ordered reflector chain or a higher-arity factor'
]) {
  if (!source.includes(marker)) throw new Error(`Bistatic bridge validation failed: ${marker}`)
}

for (const marker of [
  'L=c(\\tau-\\delta_\\tau)',
  'calibrated / normalized path loss',
  'When delay-resolved, the PDP separates paths',
  'each interaction often adds loss—not a strict ordering'
]) {
  if (!build.includes(marker)) throw new Error(`Deck math-audit validation failed: ${marker}`)
}

if (sourceChanged) writeFileSync(sourcePath, source)
if (buildChanged) writeFileSync(buildPath, build)

if (sourceChanged || buildChanged) {
  console.log('Inserted bistatic GraphSLAM bridge and corrected radio-measurement wording.')
} else {
  console.log('No bistatic bridge or math-audit changes needed.')
}
