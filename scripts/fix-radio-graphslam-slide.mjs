import { readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

const sourcePath = resolve('mpc-detection-to-bounce-count-slides/radio-slam-extra.mjs')
const liveIndexPath = resolve('mpc-detection-to-bounce-count-slides/radio-graphslam-live/index.html')
const liveAppPath = resolve('mpc-detection-to-bounce-count-slides/radio-graphslam-live/app.js')

function replaceRequired(source, before, after, label, { all = false } = {}) {
  if (source.includes(after)) return source
  if (!source.includes(before)) throw new Error(`Could not find ${label}`)
  return all ? source.split(before).join(after) : source.replace(before, after)
}

let source = readFileSync(sourcePath, 'utf8')
const graphStartMarker = 'function graphEquationSlide(ctx) {'
const methodLiveMarker = '\nfunction methodLiveSlide(kind, ctx) {'
const graphStart = source.indexOf(graphStartMarker)
const methodLiveStart = source.indexOf(methodLiveMarker, graphStart)
if (graphStart < 0 || methodLiveStart < 0) {
  throw new Error('Could not locate the GraphSLAM slide block in radio-slam-extra.mjs')
}

const BT = '`'
const graphSlides = String.raw`function graphEquationSlide(ctx) {
  const { regular, text, card, shape, line, C, SANS, MONO, tex, texBlock } = ctx
  const graph = [
    card('gs-joint-graph-card', 96, 202, 520, 420, C.paper, { stroke: C.line, radius: 8 }),
    text('gs-joint-graph-k', 120, 220, 472, 18, 'CANONICAL JOINT FACTOR GRAPH · SETUP S1', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .9 }),
    text('gs-joint-graph-sub', 120, 244, 472, 20, 're-observing one map entity from distant poses supplies global consistency', 9, { color: C.soft, fontFamily: SANS, fontWeight: 700, align: 'center' })
  ]

  const poseY = 532
  const poseXs = [140, 245, 350, 455, 560]
  const bs = { x: 350, y: 280 }
  const mapA = { x: 205, y: 326 }
  const mapB = { x: 505, y: 326 }
  const radioFactors = [
    { x: 170, y: 420, pose: 0, map: mapA },
    { x: 305, y: 420, pose: 2, map: mapA },
    { x: 395, y: 420, pose: 2, map: mapB },
    { x: 530, y: 420, pose: 4, map: mapB }
  ]

  graph.push(line('gs-joint-prior-edge', 114, poseY, poseXs[0] - 15, poseY, C.pose, 2))
  poseXs.slice(0, -1).forEach((x, index) => {
    const next = poseXs[index + 1]
    const factorX = 0.5 * (x + next)
    graph.push(line('gs-joint-relative-edge-a-' + index, x + 15, poseY, factorX - 6, poseY, C.soft, 2, { opacity: .85 }))
    graph.push(line('gs-joint-relative-edge-b-' + index, factorX + 6, poseY, next - 15, poseY, C.soft, 2, { opacity: .85 }))
  })
  radioFactors.forEach((factor, index) => {
    const poseX = poseXs[factor.pose]
    graph.push(line('gs-joint-radio-pose-' + index, factor.x, factor.y + 7, poseX, poseY - 15, C.measurement, 1.8, { opacity: .62 }))
    graph.push(line('gs-joint-radio-map-' + index, factor.x, factor.y - 7, factor.map.x, factor.map.y + 16, C.map, 1.8, { opacity: .58 }))
    graph.push(line('gs-joint-radio-bs-' + index, factor.x, factor.y - 7, bs.x, bs.y + 13, C.known, 1.4, { opacity: .34 }))
  })

  graph.push(shape('gs-joint-bs-node', bs.x - 14, bs.y - 14, 28, 28, C.ink, { shape: 'ellipse', stroke: C.paper, strokeWidth: 2 }))
  graph.push(text('gs-joint-bs-label', bs.x - 48, bs.y - 39, 96, 18, tex§\mathbf b\;\text{known}§, 9, { color: C.ink, fontWeight: 700, align: 'center' }))
  ;[
    [tex§\mathbf m_A§, mapA, C.map, C.mapSoft],
    [tex§\mathbf m_B§, mapB, C.known, C.knownSoft]
  ].forEach((entry, index) => {
    const label = entry[0], node = entry[1], color = entry[2], fill = entry[3]
    graph.push(shape('gs-joint-map-node-' + index, node.x - 17, node.y - 17, 34, 34, fill, { shape: 'ellipse', stroke: color, strokeWidth: 2 }))
    graph.push(text('gs-joint-map-label-' + index, node.x - 22, node.y - 8, 44, 18, label, 9, { color, fontWeight: 700, align: 'center' }))
  })

  graph.push(shape('gs-joint-prior-factor', 108, poseY - 6, 12, 12, C.poseSoft, { stroke: C.pose, strokeWidth: 2, radius: 0 }))
  graph.push(text('gs-joint-prior-label', 96, poseY - 29, 48, 16, 'prior', 8, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, align: 'center' }))
  poseXs.slice(0, -1).forEach((x, index) => {
    const factorX = 0.5 * (x + poseXs[index + 1])
    graph.push(shape('gs-joint-relative-factor-' + index, factorX - 6, poseY - 6, 12, 12, C.paper, { stroke: C.soft, strokeWidth: 2, radius: 0 }))
    graph.push(text('gs-joint-relative-label-' + index, factorX - 21, poseY + 12, 42, 15, tex§f^{\mathrm{rel}}§, 6.5, { color: C.faint, fontWeight: 700, align: 'center' }))
  })
  const poseLabels = [tex§\mathbf T_1§, tex§\mathbf T_2§, tex§\mathbf T_3§, tex§\mathbf T_4§, tex§\mathbf T_5§]
  poseXs.forEach((x, index) => {
    graph.push(shape('gs-joint-pose-node-' + index, x - 15, poseY - 15, 30, 30, index === 0 ? C.poseSoft : C.paper, { shape: 'ellipse', stroke: C.pose, strokeWidth: 2 }))
    graph.push(text('gs-joint-pose-label-' + index, x - 16, poseY - 8, 32, 17, poseLabels[index], 8.5, { color: C.poseDeep, fontWeight: 700, align: 'center' }))
  })
  const radioLabels = [tex§f^{\mathrm{rad}}_{1A}§, tex§f^{\mathrm{rad}}_{3A}§, tex§f^{\mathrm{rad}}_{3B}§, tex§f^{\mathrm{rad}}_{5B}§]
  radioFactors.forEach((factor, index) => {
    graph.push(shape('gs-joint-radio-factor-' + index, factor.x - 7, factor.y - 7, 14, 14, C.measurementSoft, { stroke: C.measurement, strokeWidth: 2, radius: 0 }))
    graph.push(text('gs-joint-radio-label-' + index, factor.x - 24, factor.y - 28, 48, 16, radioLabels[index], 7.5, { color: C.measurementDeep, fontWeight: 700, align: 'center' }))
  })
  graph.push(text('gs-joint-map-caption', 120, 364, 472, 30, 'Map variables may be VAs, walls, point scatterers, or reflector-chain parameters.', 9, { color: C.faint, fontFamily: SANS, fontWeight: 700, align: 'center', lineHeight: 1.25 }))
  graph.push(text('gs-joint-relative-caption', 126, 568, 460, 17, 'consecutive squares = noisy relative-pose / odometry measurements', 8, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, align: 'center' }))
  graph.push(text('gs-joint-legend', 112, 592, 488, 16, '○ continuous variable · □ factor · black node = known BS parameter', 8, { color: C.faint, fontFamily: MONO, align: 'center' }))
  graph.push(text('gs-joint-legend-2', 112, 607, 488, 14, 'shared map observations couple nonconsecutive poses without adding a physical dynamics law', 7.5, { color: C.faint, fontFamily: MONO, align: 'center' }))

  return regular(
    's-radio-graphslam-equations', '04 · KNOWN BS POSE, UNKNOWN UE POSE AND MAP',
    'Radio GraphSLAM: optimize the UE trajectory and radio map jointly',
    'Relative-pose factors encode measured pose changes. They are not, by themselves, a governing motion model.',
    'This is the canonical Section 04 factor-graph model, not a literal description of the current graph_slam.py implementation. The original 2006 GraphSLAM paper wrote control-based motion arcs, while modern pose-graph SLAM commonly uses an odometry likelihood p(T_tilde | T_{t-1}, T_t). For this deck, the front end measures relative motion, so the odometry form is the accurate one. An explicit kinematic transition or a smoothness/constant-velocity prior may be added as a separate factor, but pose-graph optimization does not require it. Radio factors couple each pose to persistent map entities. Re-observing the same entity from distant poses produces the global-consistency effect of a loop closure. Covariances set factor weights through Omega = Sigma^{-1}; robust losses or mixture factors are needed for wrong MPC associations and clutter. The geometry residual uses path length and angles; gain in dB is not part of the current implementation residual unless a calibrated propagation model is introduced.',
    [
      ...graph,
      card('gs-joint-state-card', 640, 202, 544, 106, C.poseSoft, { stroke: C.pose, radius: 8 }),
      text('gs-joint-state-k', 664, 218, 310, 16, 'UNKNOWN VARIABLES + ASSOCIATION LABELS', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .75 }),
      text('gs-joint-state-eq', 650, 240, 524, 50, texBlock§\begin{aligned}
        \mathbf X&=\{\mathbf T_t\}_{t=1}^{T},&\mathbf T_t&=(\mathbf p_t,\theta_t)\in SE(2),&\mathcal M&=\{\mathbf m_j\}_{j=1}^{J}\\
        a_{t\ell}&\in\{0,1,\ldots,J\},&&q_{t\ell}\in\{\mathrm{LoS},1,2,\ldots\}
      \end{aligned}§, 10.5, { fontWeight: 700, align: 'center', lineHeight: 1.3 }),
      text('gs-joint-state-v', 664, 292, 496, 12, tex§a_{t\ell}=0\;\text{means clutter / no map assignment}§, 7.5, { color: C.poseDeep, fontWeight: 700, align: 'center' }),

      card('gs-joint-factor-card', 640, 322, 544, 146, C.paper, { stroke: C.line, radius: 8 }),
      text('gs-joint-factor-k', 664, 338, 330, 16, 'ODOMETRY LIKELIHOOD, NOT A REQUIRED DYNAMICS LAW', 8.5, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .55 }),
      text('gs-joint-factor-eq', 650, 358, 524, 74, texBlock§\begin{aligned}
        p(\mathbf X,\mathcal M,A,Q\mid Z,\widetilde U,\mathbf b)\propto{}&p(\mathbf T_1)\prod_{t=2}^{T}p(\widetilde{\mathbf T}_{t-1,t}\mid\mathbf T_{t-1},\mathbf T_t)\prod_jp(\mathbf m_j)\\[-.2em]
        &\times\prod_{t,\ell}p(a_{t\ell},q_{t\ell})\,p(\mathbf z_{t\ell}\mid\mathbf T_t,\mathbf m_{a_{t\ell}},q_{t\ell},\mathbf b)
      \end{aligned}§, 9.2, { fontWeight: 700, align: 'center', lineHeight: 1.25 }),
      text('gs-joint-factor-v', 664, 438, 496, 18, tex§\mathbf r_t^{\mathrm{rel}}=\operatorname{Log}(\widetilde{\mathbf T}_{t-1,t}^{-1}\mathbf T_{t-1}^{-1}\mathbf T_t)\quad\text{with covariance }\Sigma_t^{\mathrm{rel}}§, 8, { color: C.soft, fontWeight: 700, align: 'center' }),

      card('gs-joint-cost-card', 640, 482, 544, 140, C.measurementSoft, { stroke: C.measurement, radius: 8 }),
      text('gs-joint-cost-k', 664, 498, 310, 16, 'COVARIANCE-WEIGHTED MAP OBJECTIVE', 8.5, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .75 }),
      text('gs-joint-cost-eq', 650, 518, 524, 62, texBlock§\begin{aligned}
        (\mathbf X^*,\mathcal M^*)&=\arg\min_{\mathbf X,\mathcal M}\;\|\mathbf r_1^{\mathrm{prior}}\|_{\Omega_1}^{2}+\sum\nolimits_{t=2}^{T}\|\mathbf r_t^{\mathrm{rel}}\|_{\Omega_t^{\mathrm{rel}}}^{2}\\[-.1em]
        &\quad+\sum\nolimits_{t,\ell}\rho\!\left(\|\mathbf r_{t\ell}^{\mathrm{rad}}(a_{t\ell},q_{t\ell})\|_{\Omega_{t\ell}^{\mathrm{rad}}}^{2}\right),\qquad \Omega=\Sigma^{-1}\\[-.1em]
        \mathbf r_{t\ell}^{\mathrm{rad}}(j,q)&=[c\tau,\varphi^{\mathrm{AoA}},\varphi^{\mathrm{AoD}}]^{\mathsf T}\boxminus\mathbf h_q(\mathbf T_t,\mathbf m_j,\mathbf b)
      \end{aligned}§, 8.8, { fontWeight: 700, align: 'center', lineHeight: 1.2 }),
      text('gs-joint-cost-v', 664, 584, 496, 24, 'Fixed A,Q → ordinary nonlinear least squares. Unknown A,Q → marginalize, maximize, or alternate association and continuous-state updates.', 7.5, { color: C.measurementDeep, fontFamily: SANS, fontWeight: 700, align: 'center', lineHeight: 1.25 }),

      card('gs-joint-key-card', 96, 632, 1088, 42, C.poseDeep, { stroke: C.poseDeep, radius: 6 }),
      text('gs-joint-key-v', 116, 638, 1048, 16, 'KEY DISTINCTION · full radio GraphSLAM optimizes both trajectory X and map M; relative-pose edges encode measured changes, not mandatory dynamics.', 9, { color: C.paper, fontFamily: SANS, fontWeight: 700, align: 'center' }),
      text('gs-joint-source', 116, 657, 1048, 12, 'Refs · SLAM Handbook Ch. 1, Fig. 1.4 & eqs. 1.16–1.18 · Thrun & Montemerlo, IJRR 2006 · Leitinger et al., ICCW 2017 / TWC 2019', 6.8, { color: '#DDEEFF', fontFamily: MONO, align: 'center' })
    ], { accent: C.pose, titleSize: 30, transition: 'none' }
  )
}

function graphFallback(ctx) {
  const { text, card, shape, line, C, MONO, SANS, LIVE_BOUNDS, tex } = ctx
  const x0 = LIVE_BOUNDS.x, y0 = LIVE_BOUNDS.y, w = LIVE_BOUNDS.width, h = LIVE_BOUNDS.height
  const stageX = x0 + 14, stageY = y0 + 14, stageW = 766, stageH = h - 28
  const railX = stageX + stageW + 12, railW = w - stageW - 40
  const poseY = [stageY + 70, stageY + 142, stageY + 214, stageY + 286, stageY + 358]
  const poseX = stageX + 180, factorX = stageX + 400, mapX = stageX + 650
  const elements = [
    card('graph-fallback-bg', x0, y0, w, h, '#F8FAFB', { stroke: C.line, radius: 0 }),
    card('graph-fallback-stage', stageX, stageY, stageW, stageH, C.paper, { stroke: C.line, radius: 6 }),
    card('graph-fallback-rail', railX, stageY, railW, stageH, C.paper, { stroke: C.line, radius: 6 }),
    text('graph-fallback-k', stageX + 22, stageY + 16, stageW - 44, 18, 'FIXED-ASSOCIATION TEACHING GRAPH · RELATIVE-POSE + RADIO FACTORS', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .7 })
  ]
  poseY.slice(0, -1).forEach((y, index) => {
    const fy = 0.5 * (y + poseY[index + 1])
    elements.push(line('graph-relative-a-' + index, poseX, y + 13, poseX, fy - 6, C.soft, 2))
    elements.push(shape('graph-relative-factor-' + index, poseX - 6, fy - 6, 12, 12, C.paper, { stroke: C.soft, strokeWidth: 2, radius: 0 }))
    elements.push(line('graph-relative-b-' + index, poseX, fy + 6, poseX, poseY[index + 1] - 13, C.soft, 2))
  })
  const fallbackPoseLabels = [tex§\mathbf x_1§, tex§\mathbf x_2§, tex§\mathbf x_3§, tex§\mathbf x_4§, tex§\mathbf x_5§]
  poseY.forEach((y, index) => {
    elements.push(shape('graph-pose-' + index, poseX - 13, y - 13, 26, 26, index === 2 ? C.poseSoft : C.paper, { shape: 'ellipse', stroke: C.pose, strokeWidth: 2 }))
    elements.push(text('graph-pose-label-' + index, poseX - 18, y - 7, 36, 14, fallbackPoseLabels[index], 8, { color: C.poseDeep, fontWeight: 700, align: 'center' }))
    ;[0, 1].forEach(mapIndex => {
      const targetY = stageY + 140 + mapIndex * 170
      const fy = y + (mapIndex ? 8 : -8)
      const color = mapIndex ? C.measurement : C.map
      elements.push(line('graph-radio-pose-' + index + '-' + mapIndex, poseX + 14, fy, factorX - 7, fy, color, 1.4, { opacity: index === 2 ? .9 : .22 }))
      elements.push(shape('graph-radio-factor-' + index + '-' + mapIndex, factorX - 7, fy - 7, 14, 14, C.paper, { stroke: color, strokeWidth: 1.6, radius: 0, opacity: index === 2 ? 1 : .35 }))
      elements.push(line('graph-radio-map-' + index + '-' + mapIndex, factorX + 7, fy, mapX - 16, targetY, color, 1.4, { opacity: index === 2 ? .9 : .22 }))
    })
  })
  ;[[stageY + 140, C.map, C.mapSoft, tex§\mathbf m_A§], [stageY + 310, C.measurement, C.measurementSoft, tex§\mathbf m_B§]].forEach((item, index) => {
    elements.push(shape('graph-map-' + index, mapX - 16, item[0] - 16, 32, 32, item[2], { shape: 'ellipse', stroke: item[1], strokeWidth: 2 }))
    elements.push(text('graph-map-label-' + index, mapX - 25, item[0] - 7, 50, 16, item[3], 9, { color: item[1], fontWeight: 700, align: 'center' }))
  })
  elements.push(text('graph-stage-caption', stageX + 36, stageY + 404, stageW - 72, 20, 'Inter-pose squares measure relative motion; radio squares connect poses to persistent map variables.', 10, { color: C.soft, fontFamily: SANS, fontWeight: 700, align: 'center' }))
  elements.push(text('graph-rail-k', railX + 18, stageY + 18, railW - 36, 18, 'GAUSS–NEWTON CONTROLS', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.05 }))
  ;['delay only ↔ delay + angles', 'correct ↔ wrong association', 'quadratic ↔ Huber loss', 'one step ↔ optimize'].forEach((label, index) => {
    const y = stageY + 60 + index * 63
    elements.push(card('graph-control-' + index, railX + 18, y, railW - 36, 45, index === 3 ? C.poseSoft : '#FBFCFD', { stroke: index === 3 ? C.pose : C.line, radius: 5 }))
    elements.push(text('graph-control-v-' + index, railX + 27, y + 13, railW - 54, 20, label, 10, { color: index === 3 ? C.poseDeep : C.soft, fontFamily: SANS, fontWeight: 700, align: 'center' }))
  })
  elements.push(card('graph-result-card', railX + 18, stageY + 326, railW - 36, 82, C.poseSoft, { stroke: C.pose, radius: 6 }))
  elements.push(text('graph-result-k', railX + 30, stageY + 342, railW - 60, 15, 'JOINT MAP ESTIMATE', 8, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1 }))
  elements.push(text('graph-result-v', railX + 30, stageY + 370, railW - 60, 26, 'UE trajectory + VA map', 13, { fontWeight: 700, align: 'center' }))
  return elements
}

function graphLiveSlide(ctx) {
  const { regular, liveMount, C } = ctx
  return regular(
    's-radio-graphslam-live', '04 · KNOWN BS POSE, UNKNOWN UE POSE AND MAP',
    'Fixed-association GraphSLAM live: relative-pose + radio factors',
    'Step Gauss–Newton, compare delay-only with delay+angles, and inject one wrong association to see why robust loss matters.',
    'This is a canonical fixed-association teaching example. Its continuous variables are five 2D UE poses and two virtual anchors. Consecutive-pose factors use noisy relative odometry measurements; they are not a governing dynamics law. Radio factors predict unfolded path length, UE-frame AoA, and BS-frame AoD. The simulator supplies association and bounce order only to make the optimization visible. The next slide separately describes the current production implementation, whose graph contains only 3D position nodes and whose map is rebuilt outside the graph.',
    [...graphFallback(ctx), liveMount()], { accent: C.pose, titleSize: 29, transition: 'none' }
  )
}

function graphIterationSlide(ctx) {
  const { regular, text, card, C, SANS, MONO, tex, texBlock } = ctx
  const steps = [
    ['01', 'FRONT END', 'unlabeled MPCs → bounce/VA clouds<br>registration → relative increments + covariance'],
    ['02', 'MAP OUTSIDE GRAPH', 'current positions → NDT map<br>exclude the target scan and its nearby window'],
    ['03', 'CREATE FACTORS', 'refresh registration + gated LoS<br>keep odometry + verified revisits fixed'],
    ['04', 'LINEAR SOLVE', 'solve weighted position graph exactly<br>rebuild map and repeat · five passes']
  ]
  const elements = []
  steps.forEach((step, index) => {
    const x = 96 + index * 276
    const accent = [C.faint, C.measurement, C.pose, C.map][index]
    const fill = [C.paper, C.measurementSoft, C.poseSoft, C.mapSoft][index]
    elements.push(card('gs-impl-card-' + index, x, 216, 244, 150, fill, { stroke: accent, strokeWidth: 2, radius: 8 }))
    elements.push(text('gs-impl-num-' + index, x + 18, 236, 36, 20, step[0], 11, { color: accent, fontFamily: MONO, fontWeight: 700 }))
    elements.push(text('gs-impl-head-' + index, x + 55, 235, 169, 20, step[1], 9.5, { color: accent, fontFamily: MONO, fontWeight: 700, letterSpacing: .65 }))
    elements.push(text('gs-impl-copy-' + index, x + 18, 282, 208, 68, step[2], 11, { color: C.soft, fontFamily: SANS, fontWeight: 700, align: 'center', lineHeight: 1.4 }))
    if (index < steps.length - 1) elements.push(text('gs-impl-arrow-' + index, x + 246, 277, 28, 32, '→', 22, { color: C.faint, fontFamily: SANS, fontWeight: 700, align: 'center' }))
  })

  elements.push(card('gs-impl-eq-card', 96, 386, 1088, 116, C.paper, { stroke: C.line, radius: 8 }))
  elements.push(text('gs-impl-eq-k', 120, 402, 420, 16, 'ACTUAL STATE AND QUADRATIC FACTORS FOR A FIXED MAP', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .7 }))
  elements.push(text('gs-impl-eq-v', 108, 424, 1064, 66, texBlock§\begin{aligned}
    \mathbf p^*=\arg\min_{\mathbf p_{0:K-1}}{}&\|\mathbf p_0-\bar{\mathbf p}_0\|_{\Omega_0}^{2}+\sum\nolimits_k\|(\mathbf p_k-\mathbf p_{k-1})-\mathbf d_k^{\mathrm{odo}}\|_{\Omega_k^{\mathrm{odo}}}^{2}\\[-.1em]
    &+\sum\nolimits_k\|\mathbf p_{k-1}-2\mathbf p_k+\mathbf p_{k+1}\|_{\Omega^{\mathrm{smooth}}}^{2}+\sum\nolimits_k\|\mathbf p_k-\mathbf z_k^{\mathrm{reg/LoS}}\|_{\Omega_k}^{2}\\[-.1em]
    &+\sum\nolimits_{(i,j)\in\mathcal L}\|(\mathbf p_j-\mathbf p_i)-\mathbf d_{ij}^{\mathrm{loop}}\|_{\Omega_{ij}}^{2},\qquad \Omega=\Sigma^{-1}
  \end{aligned}§, 9.8, { fontWeight: 700, align: 'center', lineHeight: 1.2 }))

  elements.push(card('gs-impl-state-card', 96, 520, 344, 102, C.poseSoft, { stroke: C.pose, radius: 8 }))
  elements.push(text('gs-impl-state-k', 118, 536, 210, 16, 'STATE INSIDE THE GRAPH', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .8 }))
  elements.push(text('gs-impl-state-eq', 118, 560, 300, 24, tex§\mathbf p_k\in\mathbb R^3\quad\text{only}§, 14, { color: C.poseDeep, fontWeight: 700, align: 'center' }))
  elements.push(text('gs-impl-state-v', 116, 590, 304, 22, 'UE orientation is fixed/world-aligned; the map is not a graph variable.', 9, { color: C.soft, fontFamily: SANS, fontWeight: 700, align: 'center', lineHeight: 1.2 }))

  elements.push(card('gs-impl-prior-card', 468, 520, 344, 102, C.measurementSoft, { stroke: C.measurement, radius: 8 }))
  elements.push(text('gs-impl-prior-k', 490, 536, 250, 16, 'WHAT COUNTS AS A MOTION PRIOR?', 8.5, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .6 }))
  elements.push(text('gs-impl-prior-v', 490, 562, 300, 46, 'Odometry factors measure pose change. The second-difference smoothness term is the explicit constant-velocity prior—and it is optional.', 9.5, { color: C.measurementDeep, fontFamily: SANS, fontWeight: 700, align: 'center', lineHeight: 1.28 }))

  elements.push(card('gs-impl-variant-card', 840, 520, 344, 102, C.mapSoft, { stroke: C.map, radius: 8 }))
  elements.push(text('gs-impl-variant-k', 862, 536, 230, 16, 'GRAPH_SLAM_VA VARIANT', 9, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .8 }))
  elements.push(text('gs-impl-variant-v', 862, 562, 300, 46, 'Only the local odometry channel switches to VA clouds. Global registration, revisit verification, LoS factors, and the delivered map remain bounce-point based.', 9, { color: C.mapDeep, fontFamily: SANS, fontWeight: 700, align: 'center', lineHeight: 1.28 }))

  elements.push(card('gs-impl-key-card', 96, 636, 1088, 38, C.poseDeep, { stroke: C.poseDeep, radius: 6 }))
  elements.push(text('gs-impl-key-v', 116, 641, 1048, 15, 'CLASSIFICATION · current code is a position-graph smoother with external map re-registration—not the full joint {X,M} GraphSLAM model on the previous slide.', 8.8, { color: C.paper, fontFamily: SANS, fontWeight: 700, align: 'center' }))
  elements.push(text('gs-impl-source', 116, 657, 1048, 12, 'Implementation source · Gaussian_Splatting_Test/slam/graph_slam.py + graph_slam_va.py · map dependence handled by outer alternation', 6.8, { color: '#DDEEFF', fontFamily: MONO, align: 'center' }))

  return regular(
    's-radio-graphslam-iteration', '04 · KNOWN BS POSE, UNKNOWN UE POSE AND MAP',
    'Current code: position-graph smoothing with map re-registration',
    'This implementation is accurate for what the code does, but it is an approximation to the canonical joint trajectory–map model.',
    'Read this slide as an implementation audit. graph_slam.py retains one 3D position node per frame. A known-start prior anchors translation. Relative odometry factors and their empirical information matrices come from the hybrid registration front end; verified revisits add nonconsecutive relative factors. The second-difference term is a separate smoothness/constant-velocity prior, not the odometry model. Registration and LoS factors are absolute-position pseudo-measurements generated from the current map. Because every factor is linear in positions for a fixed map, the graph solve uses one exact normal-equation solve. The nonlinear pose–map dependence is handled outside the graph by rebuilding leave-window-out NDT maps, re-inverting MPCs, refreshing registration/LoS factors, and repeating. graph_slam_va changes only the local odometry cloud source. This should not be described as joint map optimization, and it does not yet estimate unknown UE orientation.',
    elements, { accent: C.pose, titleSize: 29, transition: 'none' }
  )
}
`.replaceAll('§', BT)

source = source.slice(0, graphStart) + graphSlides + source.slice(methodLiveStart)

const appendStartMarker = 'export function appendRadioSlamSlidesAfterSection(unit, ctx) {'
const liveEntriesMarker = '\nexport function radioSlamLiveEntries({ slides, LIVE_BOUNDS }) {'
const appendStart = source.indexOf(appendStartMarker)
const liveEntriesStart = source.indexOf(liveEntriesMarker, appendStart)
if (appendStart < 0 || liveEntriesStart < 0) {
  throw new Error('Could not locate slide append/live-entry functions')
}
const appendFunction = `export function appendRadioSlamSlidesAfterSection(unit, ctx) {
  if (unit.id === 'map') {
    ctx.slides.push(sharedSetupSlide(ctx))
    ctx.slides.push(methodEquationSlide('bp', ctx))
    ctx.slides.push(methodLiveSlide('bp', ctx))
    ctx.slides.push(methodEquationSlide('pmbm', ctx))
    ctx.slides.push(methodLiveSlide('pmbm', ctx))
  }
  if (unit.id === 'pose') {
    ctx.slides.push(graphEquationSlide(ctx))
    ctx.slides.push(graphLiveSlide(ctx))
    ctx.slides.push(graphIterationSlide(ctx))
  }
}
`
source = source.slice(0, appendStart) + appendFunction + source.slice(liveEntriesStart)

const liveEntriesStart2 = source.indexOf('export function radioSlamLiveEntries({ slides, LIVE_BOUNDS }) {')
if (liveEntriesStart2 < 0) throw new Error('Could not relocate radioSlamLiveEntries()')
const liveEntriesFunction = `export function radioSlamLiveEntries({ slides, LIVE_BOUNDS }) {
  const definitions = [
    {
      introSlide: 's-bp-slam-equations', slide: 's-bp-slam-live',
      src: '../bp-vs-pmbm-slides/live/?demo=bp&embed=region', source: '../bp-vs-pmbm-slides/live/?demo=bp',
      title: 'Section 03 · BP-SLAM on shared setup S1'
    },
    {
      introSlide: 's-pmbm-slam-equations', slide: 's-pmbm-slam-live',
      src: '../bp-vs-pmbm-slides/live/?demo=pmbm&embed=region', source: '../bp-vs-pmbm-slides/live/?demo=pmbm',
      title: 'Section 03 · PMBM-SLAM on shared setup S1'
    },
    {
      introSlide: 's-radio-graphslam-equations', slide: 's-radio-graphslam-live',
      src: './radio-graphslam-live/?embed=region', source: './radio-graphslam-live/',
      title: 'Section 04 · fixed-association radio GraphSLAM teaching lab'
    }
  ]
  return definitions.map(entry => ({
    ...entry,
    slideIndex: slides.findIndex(slide => slide.id === entry.slide),
    inline: true, layout: 'region', bounds: LIVE_BOUNDS,
    sandbox: 'allow-scripts', hideSource: true, readyMessage: true, unloadWhenHidden: true
  }))
}
`
source = source.slice(0, liveEntriesStart2) + liveEntriesFunction

source = replaceRequired(
  source,
  'the implemented GraphSLAM uses unlabeled MPC sets, estimates pose nodes only, and constructs registration and gated-LoS factors in its front end.',
  'the implemented GraphSLAM uses unlabeled MPC sets, estimates 3-D position nodes only (orientation is fixed/world-aligned), and constructs registration and gated-LoS factors in its front end.',
  'shared-setup implementation note'
)

for (const marker of [
  "'s-radio-graphslam-equations'",
  "'s-radio-graphslam-live'",
  "'s-radio-graphslam-iteration'",
  'ODOMETRY LIKELIHOOD, NOT A REQUIRED DYNAMICS LAW',
  'position-graph smoothing with map re-registration',
  "src: './radio-graphslam-live/?embed=region'"
]) {
  if (!source.includes(marker)) throw new Error(`GraphSLAM source validation failed: ${marker}`)
}
writeFileSync(sourcePath, source)

let liveIndex = readFileSync(liveIndexPath, 'utf8')
const indexReplacements = [
  ['<title>Archived oracle-associated S1 teaching toy</title>', '<title>Fixed-association radio GraphSLAM teaching lab</title>', 'live page title'],
  ['<meta name="description" content="Archived teaching toy with truth-conditioned path associations; this is not the implemented pose-only radio GraphSLAM estimator.">', '<meta name="description" content="Canonical fixed-association radio GraphSLAM teaching lab with relative-pose and MPC factors; explicitly separated from the current production position-graph implementation.">', 'live page description'],
  ['ARCHIVED S1 TOY · ORACLE PATH ASSOCIATIONS', 'CANONICAL TEACHING LAB · FIXED ASSOCIATIONS', 'live page eyebrow'],
  ['Truth-conditioned trajectory/VA optimization toy.', 'Joint trajectory–VA GraphSLAM with relative-pose factors.', 'live page heading'],
  ['title="Truth path identities are supplied to this archived teaching toy; they are not available to the implemented GraphSLAM estimator.">Not the implemented estimator', 'title="Association and path order are supplied only for this teaching solve; the next slide describes the current production implementation.">Fixed-association demo', 'live page chip'],
  ['ORACLE-ASSOCIATED TEACHING TOY', 'CANONICAL FACTOR-GRAPH TEACHING MODEL', 'live page method kicker'],
  ['<h2>Path identities are supplied here</h2>', '<h2>Relative-pose factors + radio factors</h2>', 'live page method heading'],
  ['<p>This archived optimizer is given which return is LoS and which VA generated each reflection. The implemented estimator does not receive those labels.</p>', '<p>Inter-pose edges are noisy odometry measurements, not a governing dynamics law. MPC association and bounce order are supplied only to expose the joint optimization.</p>', 'live page method copy'],
  ['CONDITIONED TRAJECTORY–VA TOY', 'JOINT TRAJECTORY–VA MAP ESTIMATE', 'live stage label'],
  ['Oracle-associated teaching geometry with a known base station, UE trajectory, virtual anchors, reflector walls, and truth-labelled multipath routes.', 'Fixed-association teaching geometry with a known base station, UE trajectory, virtual anchors, reflector walls, and simulated multipath routes.', 'live canvas label'],
  ['<p><strong>Scope:</strong> oracle-associated S1 toy: LoS/reflection identity and ordering are fixed inside each solve.</p>', '<p><strong>Scope:</strong> canonical teaching solve with fixed association and path order; it demonstrates relative-pose, radio, covariance, and robust factors.</p>', 'live footer scope'],
  ['<p><strong>Not the implementation:</strong> the delivered GraphSLAM uses unlabeled MPCs, pose-only variables, and front-end-derived factors.</p>', '<p><strong>Current code differs:</strong> it estimates world-aligned 3D positions in a linear graph and rebuilds the radio map outside the graph.</p>', 'live footer implementation'],
  ['<h1>Truth-conditioned trajectory/VA optimization toy</h1>', '<h1>Joint trajectory–VA GraphSLAM with relative-pose factors</h1>', 'noscript heading'],
  ['<p>This archived deterministic example is supplied the LoS/reflection identities. It is not the implemented pose-only GraphSLAM estimator, which receives unlabeled MPC sets.</p>', '<p>This deterministic teaching example supplies MPC association and path order. Consecutive factors represent noisy relative-pose measurements. The production position-graph implementation is described separately in the deck.</p>', 'noscript copy']
]
for (const [before, after, label] of indexReplacements) {
  liveIndex = replaceRequired(liveIndex, before, after, label, { all: label === 'live page eyebrow' })
}
for (const marker of ['CANONICAL TEACHING LAB · FIXED ASSOCIATIONS', 'Relative-pose factors + radio factors', 'Current code differs:']) {
  if (!liveIndex.includes(marker)) throw new Error(`Live index validation failed: ${marker}`)
}
writeFileSync(liveIndexPath, liveIndex)

let liveApp = readFileSync(liveAppPath, 'utf8')
liveApp = replaceRequired(
  liveApp,
  'label(ctx, "teaching toy | oracle-fixed A,Q", 14, 34, COLORS.faint, 8, "left", 400);',
  'label(ctx, "fixed-association demo | A,Q supplied", 14, 34, COLORS.faint, 8, "left", 400);',
  'factor-graph lab status label'
)
liveApp = replaceRequired(
  liveApp,
  ': "Truth overlay: orange is oracle-labelled LoS; green/blue are oracle-labelled specular routes. They are supplied to this toy, not to the implemented estimator.";',
  ': "Simulated routes generated the measurements; their association labels are supplied only to this teaching solve, not to the current production estimator.";',
  'factor-graph lab hint'
)
for (const marker of ['fixed-association demo | A,Q supplied', 'association labels are supplied only to this teaching solve']) {
  if (!liveApp.includes(marker)) throw new Error(`Live app validation failed: ${marker}`)
}
writeFileSync(liveAppPath, liveApp)

console.log('Corrected canonical and implementation-specific radio GraphSLAM slides, restored the fixed-association live lab, and updated its framing.')
