import { readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

const sourcePath = resolve('mpc-detection-to-bounce-count-slides/radio-slam-extra.mjs')
let source = readFileSync(sourcePath, 'utf8')
let changed = false

const BT = '`'
const graphMarker = 'function graphEquationSlide(ctx) {'

if (!source.includes('function bundleAdjustmentSlide(ctx) {')) {
  if (!source.includes(graphMarker)) throw new Error('Could not find GraphSLAM slide insertion marker')

  const bundleSlide = String.raw`function bundleAdjustmentSlide(ctx) {
  const { regular, text, card, shape, line, C, SANS, MONO, tex, texBlock } = ctx
  const elements = [
    card('ba-graph-card', 96, 202, 430, 420, C.paper, { stroke: C.line, radius: 8 }),
    text('ba-graph-k', 118, 220, 386, 18, 'BIPARTITE OBSERVATION GRAPH · FIXED ASSOCIATIONS', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .75 }),
    text('ba-graph-sub', 118, 244, 386, 18, 'the same map entity observed from several poses couples those poses indirectly', 9, { color: C.soft, fontFamily: SANS, fontWeight: 700, align: 'center' }),
    shape('ba-bs-node', 296, 266, 28, 28, C.ink, { shape: 'ellipse', stroke: C.paper, strokeWidth: 2 }),
    text('ba-bs-label', 250, 248, 120, 16, tex§\mathbf b\;\text{known}§, 9, { color: C.ink, fontWeight: 700, align: 'center' }),
    text('ba-bs-caption', 224, 296, 172, 14, 'fixed parameter in every radio factor', 7.5, { color: C.faint, fontFamily: MONO, align: 'center' })
  ]

  const poses = [
    { x: 154, y: 338, label: tex§\mathbf T_1§ },
    { x: 154, y: 416, label: tex§\mathbf T_2§ },
    { x: 154, y: 494, label: tex§\mathbf T_3§ },
    { x: 154, y: 572, label: tex§\mathbf T_4§ }
  ]
  const maps = [
    { x: 468, y: 346, label: tex§\mathbf m_A§, color: C.map, fill: C.mapSoft },
    { x: 468, y: 460, label: tex§\mathbf m_B§, color: C.known, fill: C.knownSoft },
    { x: 468, y: 566, label: tex§\mathbf m_C§, color: C.measurement, fill: C.measurementSoft }
  ]
  const observations = [
    { x: 292, y: 338, pose: 0, map: 0 },
    { x: 320, y: 372, pose: 0, map: 1 },
    { x: 292, y: 416, pose: 1, map: 0 },
    { x: 310, y: 474, pose: 2, map: 1 },
    { x: 332, y: 510, pose: 2, map: 2 },
    { x: 304, y: 572, pose: 3, map: 2 }
  ]

  observations.forEach((obs, index) => {
    const pose = poses[obs.pose], map = maps[obs.map]
    elements.push(line('ba-observation-pose-' + index, pose.x + 16, pose.y, obs.x - 7, obs.y, C.measurement, 1.6, { opacity: .68 }))
    elements.push(line('ba-observation-map-' + index, obs.x + 7, obs.y, map.x - 18, map.y, map.color, 1.6, { opacity: .62 }))
    elements.push(shape('ba-observation-factor-' + index, obs.x - 7, obs.y - 7, 14, 14, C.measurementSoft, { stroke: C.measurement, strokeWidth: 2, radius: 0 }))
  })
  poses.forEach((pose, index) => {
    elements.push(shape('ba-pose-node-' + index, pose.x - 16, pose.y - 16, 32, 32, index === 0 ? C.poseSoft : C.paper, { shape: 'ellipse', stroke: C.pose, strokeWidth: 2 }))
    elements.push(text('ba-pose-label-' + index, pose.x - 18, pose.y - 8, 36, 17, pose.label, 8.5, { color: C.poseDeep, fontWeight: 700, align: 'center' }))
  })
  maps.forEach((map, index) => {
    elements.push(shape('ba-map-node-' + index, map.x - 18, map.y - 18, 36, 36, map.fill, { shape: 'ellipse', stroke: map.color, strokeWidth: 2 }))
    elements.push(text('ba-map-label-' + index, map.x - 24, map.y - 8, 48, 17, map.label, 8.5, { color: map.color, fontWeight: 700, align: 'center' }))
  })
  elements.push(text('ba-graph-pose-k', 112, 312, 84, 14, 'UE POSES', 8, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, align: 'center', letterSpacing: .7 }))
  elements.push(text('ba-graph-map-k', 426, 312, 84, 14, 'MAP ENTITIES', 8, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, align: 'center', letterSpacing: .7 }))
  elements.push(card('ba-no-motion-card', 118, 590, 386, 22, C.poseSoft, { stroke: C.pose, radius: 5 }))
  elements.push(text('ba-no-motion-v', 128, 594, 366, 14, 'PURE BA: no pose–pose chain and no required motion law', 8.2, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, align: 'center' }))

  elements.push(card('ba-objective-card', 548, 202, 636, 174, C.poseSoft, { stroke: C.pose, strokeWidth: 2, radius: 8 }))
  elements.push(text('ba-objective-k', 572, 218, 400, 18, 'JOINT NONLINEAR LEAST SQUARES', 9.5, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1 }))
  elements.push(text('ba-objective-eq', 560, 244, 612, 92, texBlock§\begin{aligned}
    (\mathbf X^*,\mathcal M^*)&=\arg\min_{\mathbf X,\mathcal M}\sum_{(t,\ell)\in\mathcal O}\rho\!\left(\left\|\mathbf z_{t\ell}\boxminus\mathbf h_{q_{t\ell}}(\mathbf T_t,\mathbf m_{a_{t\ell}},\mathbf b)\right\|_{\Omega_{t\ell}}^2\right)\\[-.1em]
    &\qquad+\left\|\mathbf r_{\mathrm{anchor}}\right\|_{\Omega_{\mathrm{anchor}}}^{2},\qquad \Omega=\Sigma^{-1},\\[-.1em]
    \mathbf h_{\mathrm{cam}}&=\pi(\mathbf T_t\mathbf P_j),\qquad
    \mathbf h_{\mathrm{rad},q}=[L_q,\varphi_q^{\mathrm{AoA}},\varphi_q^{\mathrm{AoD}}]^{\mathsf T}.
  \end{aligned}§, 11.1, { fontWeight: 700, align: 'center', lineHeight: 1.24 }))
  elements.push(text('ba-objective-v', 572, 344, 588, 20, 'Classical BA uses image reprojection; the radio analogue replaces pixels with path length and angle residuals.', 9.5, { color: C.poseDeep, fontFamily: SANS, fontWeight: 700, align: 'center' }))

  elements.push(card('ba-schur-card', 548, 392, 636, 142, C.paper, { stroke: C.line, radius: 8 }))
  elements.push(text('ba-schur-k', 572, 408, 410, 18, 'GAUSS–NEWTON + SCHUR COMPLEMENT', 9.5, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .9 }))
  elements.push(text('ba-schur-eq', 560, 432, 612, 70, texBlock§\begin{aligned}
    \begin{bmatrix}\mathbf H_{XX}&\mathbf H_{XM}\\\mathbf H_{MX}&\mathbf H_{MM}\end{bmatrix}
    \begin{bmatrix}\Delta\mathbf X\\\Delta\mathcal M\end{bmatrix}&=-\begin{bmatrix}\mathbf g_X\\\mathbf g_M\end{bmatrix},\\[-.1em]
    (\mathbf H_{XX}-\mathbf H_{XM}\mathbf H_{MM}^{-1}\mathbf H_{MX})\Delta\mathbf X&=-(\mathbf g_X-\mathbf H_{XM}\mathbf H_{MM}^{-1}\mathbf g_M).
  \end{aligned}§, 11.2, { fontWeight: 700, align: 'center', lineHeight: 1.2 }))
  elements.push(text('ba-schur-v', 572, 506, 588, 18, 'Eliminate independent map blocks, solve the sparse pose system, then back-substitute the map update.', 9.5, { color: C.soft, fontFamily: SANS, fontWeight: 700, align: 'center' }))

  elements.push(card('ba-bridge-card', 548, 550, 636, 72, C.measurementSoft, { stroke: C.measurement, radius: 8 }))
  elements.push(text('ba-bridge-k', 572, 562, 234, 16, 'BRIDGE TO THE NEXT SLIDE', 9, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .8 }))
  elements.push(text('ba-bridge-eq', 566, 582, 600, 24, texBlock§\mathcal F_{\mathrm{GraphSLAM}}=\mathcal F_{\mathrm{BA}}\cup\mathcal F_{\mathrm{rel}}\cup\mathcal F_{\mathrm{loop}}\cup\mathcal F_{\mathrm{prior}}§, 12.5, { color: C.measurementDeep, fontWeight: 700, align: 'center' }))
  elements.push(text('ba-bridge-v', 572, 606, 588, 12, 'BA cannot invent observability: anchor the gauge and retain any geometry-induced null space.', 8.5, { color: C.measurementDeep, fontFamily: SANS, fontWeight: 700, align: 'center' }))

  elements.push(card('ba-key-card', 96, 636, 1088, 38, C.poseDeep, { stroke: C.poseDeep, radius: 6 }))
  elements.push(text('ba-key-v', 116, 641, 1048, 15, 'KEY IDEA · repeated pose–map observations provide global consistency even without a governing motion model.', 9.5, { color: C.paper, fontFamily: SANS, fontWeight: 700, align: 'center' }))
  elements.push(text('ba-reference-v', 116, 658, 1048, 10, 'Refs · Triggs et al., Bundle Adjustment—A Modern Synthesis, 2000 · Agarwal et al., Bundle Adjustment in the Large, ECCV 2010', 6.8, { color: '#DDEEFF', fontFamily: MONO, align: 'center' }))

  return regular(
    's-radio-bundle-adjustment', '04 · KNOWN BS POSE, UNKNOWN UE POSE AND MAP',
    'Bundle adjustment: the measurement-only core of joint radio SLAM',
    'Jointly refine UE poses and persistent map entities from repeated MPC observations; association and bounce order are fixed for this optimization.',
    'Introduce bundle adjustment as the observation-only joint pose–map problem. Classical visual BA minimizes reprojection residuals over camera poses and 3D points. The radio analogue minimizes path-length, UE-frame AoA, and BS-frame AoD residuals over UE poses and persistent radio-map entities. No explicit motion model is required: two poses become coupled when they observe the same map entity. The known BS and any pose or heading priors anchor the gauge, but BA cannot remove a geometry-induced null space. With fixed association and bounce order, Gauss–Newton produces the usual block normal equations; eliminating map blocks with the Schur complement yields a sparse reduced pose system. The next slide generalizes this measurement-only graph by adding relative-pose, loop, and other factors, which is the GraphSLAM view.',
    elements, { accent: C.pose, titleSize: 31, transition: 'none' }
  )
}`.replaceAll('§', BT)

  source = source.replace(graphMarker, bundleSlide + '\n\n' + graphMarker)
  changed = true
}

if (!source.includes('ctx.slides.push(bundleAdjustmentSlide(ctx))')) {
  const appendBefore = "  if (unit.id === 'pose') {\n    ctx.slides.push(graphEquationSlide(ctx))"
  const appendAfter = "  if (unit.id === 'pose') {\n    ctx.slides.push(bundleAdjustmentSlide(ctx))\n    ctx.slides.push(graphEquationSlide(ctx))"
  if (!source.includes(appendBefore)) throw new Error('Could not find Section 04 slide sequence')
  source = source.replace(appendBefore, appendAfter)
  changed = true
}

for (const marker of [
  'function bundleAdjustmentSlide(ctx) {',
  "'s-radio-bundle-adjustment'",
  'ctx.slides.push(bundleAdjustmentSlide(ctx))',
  '\\mathcal F_{\\mathrm{GraphSLAM}}=\\mathcal F_{\\mathrm{BA}}'
]) {
  if (!source.includes(marker)) throw new Error(`Bundle-adjustment patch validation failed: ${marker}`)
}

if (changed) {
  writeFileSync(sourcePath, source)
  console.log(`Inserted bundle-adjustment slide into ${sourcePath}`)
} else {
  console.log(`No bundle-adjustment changes needed in ${sourcePath}`)
}
