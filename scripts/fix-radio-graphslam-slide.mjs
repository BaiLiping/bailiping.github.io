import { readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

const sourcePath = resolve('mpc-detection-to-bounce-count-slides/radio-slam-extra.mjs')
const source = readFileSync(sourcePath, 'utf8')
const startMarker = 'function graphEquationSlide(ctx) {'
const endMarker = '\nfunction graphIterationSlide(ctx) {'
const start = source.indexOf(startMarker)
const end = source.indexOf(endMarker, start)

if (start < 0 || end < 0) {
  throw new Error('Could not locate graphEquationSlide() in radio-slam-extra.mjs')
}

const BT = '`'
const replacement = String.raw`function graphEquationSlide(ctx) {
  const { regular, text, card, shape, line, C, SANS, MONO, tex, texBlock } = ctx
  const graph = [
    card('gs-joint-graph-card', 96, 202, 520, 420, C.paper, { stroke: C.line, radius: 8 }),
    text('gs-joint-graph-k', 120, 222, 472, 18, 'JOINT FACTOR GRAPH · SETUP S1', 10, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.15 }),
    text('gs-joint-graph-sub', 120, 246, 472, 18, 'same map feature observed from several poses = radio loop closure', 10, { color: C.soft, fontFamily: SANS, fontWeight: 700, align: 'center' })
  ]

  const poseY = 526
  const poseXs = [140, 245, 350, 455, 560]
  const bs = { x: 350, y: 286 }
  const mapA = { x: 205, y: 332 }
  const mapB = { x: 505, y: 332 }
  const radioFactors = [
    { id: '1A', x: 170, y: 420, pose: 0, map: mapA },
    { id: '3A', x: 305, y: 420, pose: 2, map: mapA },
    { id: '3B', x: 395, y: 420, pose: 2, map: mapB },
    { id: '5B', x: 530, y: 420, pose: 4, map: mapB }
  ]

  // Draw edges first so variable and factor nodes remain legible.
  graph.push(line('gs-joint-prior-edge', 114, poseY, poseXs[0] - 15, poseY, C.pose, 2))
  poseXs.slice(0, -1).forEach((x, index) => {
    const next = poseXs[index + 1]
    const factorX = 0.5 * (x + next)
    graph.push(line('gs-joint-motion-edge-a-' + index, x + 15, poseY, factorX - 6, poseY, C.soft, 2, { opacity: .8 }))
    graph.push(line('gs-joint-motion-edge-b-' + index, factorX + 6, poseY, next - 15, poseY, C.soft, 2, { opacity: .8 }))
  })
  radioFactors.forEach((factor, index) => {
    const poseX = poseXs[factor.pose]
    graph.push(line('gs-joint-radio-pose-' + index, factor.x, factor.y + 7, poseX, poseY - 15, C.measurement, 1.8, { opacity: .62 }))
    graph.push(line('gs-joint-radio-map-' + index, factor.x, factor.y - 7, factor.map.x, factor.map.y + 16, C.map, 1.8, { opacity: .58 }))
    graph.push(line('gs-joint-radio-bs-' + index, factor.x, factor.y - 7, bs.x, bs.y + 13, C.known, 1.4, { opacity: .34 }))
  })

  graph.push(shape('gs-joint-bs-node', bs.x - 14, bs.y - 14, 28, 28, C.ink, { shape: 'ellipse', stroke: C.paper, strokeWidth: 2 }))
  graph.push(text('gs-joint-bs-label', bs.x - 42, bs.y - 38, 84, 18, tex§\mathbf b\;\text{known}§, 10, { color: C.ink, fontWeight: 700, align: 'center' }))

  ;[
    [tex§\mathbf m_A§, mapA, C.map, C.mapSoft],
    [tex§\mathbf m_B§, mapB, C.known, C.knownSoft]
  ].forEach((entry, index) => {
    const label = entry[0], node = entry[1], color = entry[2], fill = entry[3]
    graph.push(shape('gs-joint-map-node-' + index, node.x - 17, node.y - 17, 34, 34, fill, { shape: 'ellipse', stroke: color, strokeWidth: 2 }))
    graph.push(text('gs-joint-map-label-' + index, node.x - 22, node.y - 8, 44, 18, label, 10, { color, fontWeight: 700, align: 'center' }))
  })

  graph.push(shape('gs-joint-prior-factor', 108, poseY - 6, 12, 12, C.poseSoft, { stroke: C.pose, strokeWidth: 2, radius: 0 }))
  graph.push(text('gs-joint-prior-label', 96, poseY - 29, 48, 16, 'prior', 8, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, align: 'center' }))
  poseXs.slice(0, -1).forEach((x, index) => {
    const factorX = 0.5 * (x + poseXs[index + 1])
    graph.push(shape('gs-joint-motion-factor-' + index, factorX - 6, poseY - 6, 12, 12, C.paper, { stroke: C.soft, strokeWidth: 2, radius: 0 }))
  })
  const poseLabels = [tex§\mathbf x_1§, tex§\mathbf x_2§, tex§\mathbf x_3§, tex§\mathbf x_4§, tex§\mathbf x_5§]
  poseXs.forEach((x, index) => {
    graph.push(shape('gs-joint-pose-node-' + index, x - 15, poseY - 15, 30, 30, index === 0 ? C.poseSoft : C.paper, { shape: 'ellipse', stroke: C.pose, strokeWidth: 2 }))
    graph.push(text('gs-joint-pose-label-' + index, x - 16, poseY - 8, 32, 17, poseLabels[index], 9, { color: C.poseDeep, fontWeight: 700, align: 'center' }))
  })
  const radioLabels = [tex§f^{\mathrm{rad}}_{1A}§, tex§f^{\mathrm{rad}}_{3A}§, tex§f^{\mathrm{rad}}_{3B}§, tex§f^{\mathrm{rad}}_{5B}§]
  radioFactors.forEach((factor, index) => {
    graph.push(shape('gs-joint-radio-factor-' + index, factor.x - 7, factor.y - 7, 14, 14, C.measurementSoft, { stroke: C.measurement, strokeWidth: 2, radius: 0 }))
    graph.push(text('gs-joint-radio-label-' + index, factor.x - 24, factor.y - 28, 48, 16, radioLabels[index], 8, { color: C.measurementDeep, fontWeight: 700, align: 'center' }))
  })
  graph.push(text('gs-joint-map-caption', 120, 366, 472, 18, 'map nodes may parameterize a VA, a wall, a point scatterer, or a reflector chain', 9, { color: C.faint, fontFamily: SANS, fontWeight: 700, align: 'center' }))
  graph.push(text('gs-joint-legend', 112, 586, 488, 16, '○ continuous variable · □ factor · black node = known BS pose', 9, { color: C.faint, fontFamily: MONO, align: 'center' }))
  graph.push(text('gs-joint-legend-2', 112, 604, 488, 14, 'representative radio edges shown; every accepted/soft hypothesis adds one factor', 8, { color: C.faint, fontFamily: MONO, align: 'center' }))

  return regular(
    's-radio-graphslam-equations', '04 · KNOWN BS POSE, UNKNOWN UE POSE AND MAP',
    'Radio GraphSLAM: optimize the UE trajectory and radio map jointly',
    'The known BS anchors the frame; motion factors link poses, while each unlabeled MPC contributes a geometry-and-association factor to map hypotheses.',
    'This slide replaces the previous pose-only formulation. Full GraphSLAM for Section 04 must include both the UE trajectory and the radio map as variables. A map node may represent a virtual anchor, wall, point scatterer, or ordered reflector chain. The association a and interaction class q are latent. With fixed assignments the problem is ordinary nonlinear least squares; with unknown assignments use sum/max-mixture factors or alternate association updates with Gauss–Newton or Levenberg–Marquardt. The complex MPC gain alpha belongs in the likelihood or association weight unless a calibrated propagation model predicts it directly.',
    [
      ...graph,
      card('gs-joint-state-card', 640, 202, 544, 112, C.poseSoft, { stroke: C.pose, radius: 8 }),
      text('gs-joint-state-k', 666, 218, 270, 16, 'CONTINUOUS + DISCRETE UNKNOWN QUANTITIES', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .85 }),
      text('gs-joint-state-eq', 650, 240, 524, 58, texBlock§\begin{aligned}
        \Theta_c&=\{\mathbf X,\mathcal M\},&\mathbf X&=\{\mathbf x_t\}_{t=1}^{T},&\mathbf x_t&=[\mathbf p_t^{\mathsf T},\theta_t]^{\mathsf T}\\
        \mathcal M&=\{\mathbf m_j\}_{j=1}^{J},&A&=\{a_{t\ell}\},&Q&=\{q_{t\ell}\}\\
        a_{t\ell}&\in\{0,1,\ldots,J\},&&&q_{t\ell}&\in\{\mathrm{LoS},1,2,\ldots\}
      \end{aligned}§, 11, { fontWeight: 700, align: 'center', lineHeight: 1.35 }),
      text('gs-joint-state-v', 666, 298, 492, 12, tex§a_{t\ell}=0\;\text{denotes clutter / no map assignment}§, 8, { color: C.poseDeep, fontWeight: 700, align: 'center' }),

      card('gs-joint-posterior-card', 640, 330, 544, 142, C.paper, { stroke: C.line, radius: 8 }),
      text('gs-joint-posterior-k', 666, 348, 290, 16, 'RADIO-SLAM POSTERIOR FACTORIZATION', 9, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .9 }),
      text('gs-joint-meas-eq', 656, 370, 512, 24, texBlock§\mathbf z_{t\ell}=(\tau_{t\ell},\varphi_{t\ell},\psi_{t\ell},\alpha_{t\ell}),\qquad \mathbf z^{\mathrm{geo}}_{t\ell}=[c\tau_{t\ell},\varphi_{t\ell},\psi_{t\ell}]^{\mathsf T}§, 11, { fontWeight: 700, align: 'center' }),
      text('gs-joint-posterior-eq', 650, 398, 524, 58, texBlock§\begin{aligned}
        p(\mathbf X,\mathcal M,A,Q\mid Z,U,\mathbf b)\propto{}&p(\mathbf x_1)\prod_{t=2}^{T}p(\mathbf x_t\mid\mathbf x_{t-1},\mathbf u_t)\prod_{j=1}^{J}p(\mathbf m_j)\\[-.1em]
        &\times\prod_{t,\ell}p(a_{t\ell},q_{t\ell})\,p(\mathbf z_{t\ell}\mid\mathbf x_t,\mathbf m_{a_{t\ell}},q_{t\ell},\mathbf b)
      \end{aligned}§, 10, { fontWeight: 700, align: 'center', lineHeight: 1.25 }),
      text('gs-joint-gain-v', 666, 456, 492, 14, tex§\alpha_{t\ell}\;\text{supports detection/association; include it in }\mathbf h_q\text{ only with a calibrated gain model}§, 7, { color: C.soft, fontWeight: 700, align: 'center' }),

      card('gs-joint-cost-card', 640, 488, 544, 134, C.measurementSoft, { stroke: C.measurement, radius: 8 }),
      text('gs-joint-cost-k', 666, 506, 290, 16, 'NEGATIVE LOG POSTERIOR → GRAPHSLAM', 9, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .9 }),
      text('gs-joint-cost-eq', 650, 530, 524, 58, texBlock§\begin{aligned}
        (\mathbf X^*,\mathcal M^*)&=\arg\min_{\mathbf X,\mathcal M}\;\|\mathbf r_1^{\mathrm{prior}}\|_{\Omega_1}^2+\sum_{t=2}^{T}\|\mathbf r_t^{\mathrm{mot}}\|_{\Omega_t}^2+\sum_{t,\ell}\Phi^{\mathrm{rad}}_{t\ell}(\mathbf X,\mathcal M)\\[-.1em]
        \mathbf r_{t\ell}^{\mathrm{rad}}(j,q)&=\mathbf z^{\mathrm{geo}}_{t\ell}\boxminus\mathbf h_q(\mathbf x_t,\mathbf m_j,\mathbf b)
      \end{aligned}§, 10, { fontWeight: 700, align: 'center', lineHeight: 1.25 }),
      text('gs-joint-cost-v', 666, 590, 492, 20, tex§\Phi_{t\ell}^{\mathrm{rad}}=-2\log[\pi_{\mathrm{FA}}p_{\mathrm{FA}}(\mathbf z_{t\ell})+\sum_{j,q}\pi_{t\ell jq}p(\mathbf z_{t\ell}\mid\mathbf x_t,\mathbf m_j,q,\mathbf b)]§, 7.5, { color: C.measurementDeep, fontWeight: 700, align: 'center' }),

      card('gs-joint-key-card', 96, 638, 1088, 36, C.poseDeep, { stroke: C.poseDeep, radius: 6 }),
      text('gs-joint-key-v', 116, 647, 1048, 18, 'KEY DISTINCTION · full radio GraphSLAM optimizes both trajectory X and map M; a pose-only fixed-map loop is an approximation, not the Section 04 model.', 10, { color: C.paper, fontFamily: SANS, fontWeight: 700, align: 'center' })
    ], { accent: C.pose, titleSize: 30, transition: 'none' }
  )
}
`.replaceAll('§', BT)

const next = source.slice(0, start) + replacement + source.slice(end)
if (next === source) {
  console.log('Slide 17 already uses the joint trajectory-map GraphSLAM formulation.')
} else {
  writeFileSync(sourcePath, next)
  console.log('Replaced slide 17 with a joint trajectory-map radio GraphSLAM formulation.')
}
