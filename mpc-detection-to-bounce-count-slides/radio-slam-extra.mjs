import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)
const RADIO_S1 = require('../assets/radio-slam-s1.js')

function sharedSetupSlide(ctx) {
  const { regular, text, card, shape, line, C, SANS, MONO, tex, texBlock } = ctx
  const setup = RADIO_S1.setup
  const scan = RADIO_S1.scan(3, 1)
  const scene = { x: 116, y: 224, width: 614, height: 360 }
  const point = value => ({
    x: scene.x + value.x / 16 * scene.width,
    y: scene.y + scene.height - value.y / 13 * scene.height
  })
  const bs = point(setup.bs)
  const activePose = point(scan.pose)
  const elements = [
    card('s1-scene-card', 96, 202, 660, 420, C.paper, { stroke: C.line, radius: 8 }),
    text('s1-scene-k', 118, 218, 480, 18, 'SIMULATION-TRUTH OVERLAY · HIGHLIGHTED SCAN 4', 10, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.05 })
  ]

  for (let x = 0; x <= 16; x += 2) {
    const a = point({ x, y: 0 })
    elements.push(shape(`s1-grid-x-${x}`, a.x, scene.y, 1, scene.height, C.line, { radius: 0, opacity: .5 }))
  }
  for (let y = 0; y <= 12; y += 2) {
    const a = point({ x: 0, y }), b = point({ x: 16, y })
    elements.push(line(`s1-grid-y-${y}`, a.x, a.y, b.x, b.y, C.line, 1, { opacity: .5 }))
  }

  const wallA0 = point({ x: 0, y: setup.walls[0].value })
  const wallA1 = point({ x: 16, y: setup.walls[0].value })
  const wallB0 = point({ x: setup.walls[1].value, y: 0 })
  const wallB1 = point({ x: setup.walls[1].value, y: 13 })
  elements.push(line('s1-wall-a', wallA0.x, wallA0.y, wallA1.x, wallA1.y, C.map, 4))
  elements.push(line('s1-wall-b', wallB0.x, wallB0.y, wallB1.x, wallB1.y, C.pose, 4))
  elements.push(text('s1-wall-a-label', wallA1.x - 112, wallA1.y - 22, 100, 16, `${tex`\mathcal W_A`}: ${tex`y=7`}`, 10, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, align: 'right' }))
  elements.push(text('s1-wall-b-label', wallB1.x + 8, wallB1.y + 8, 108, 16, `${tex`\mathcal W_B`}: ${tex`x=8.5`}`, 10, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700 }))

  setup.poses.forEach((pose, index) => {
    const p = point(pose)
    if (index) {
      const previous = point(setup.poses[index - 1])
      elements.push(line(`s1-trajectory-${index}`, previous.x, previous.y, p.x, p.y, C.pose, 3))
    }
    elements.push(shape(`s1-pose-${index}`, p.x - 7, p.y - 7, 14, 14, index === 3 ? C.pose : C.paper, { shape: 'ellipse', stroke: C.pose, strokeWidth: 2 }))
    const headingEnd = point({ x: pose.x + .42 * Math.cos(pose.heading), y: pose.y + .42 * Math.sin(pose.heading) })
    elements.push(line(`s1-heading-${index}`, p.x, p.y, headingEnd.x, headingEnd.y, index === 3 ? C.paper : C.poseDeep, 2))
    elements.push(text(`s1-pose-label-${index}`, p.x + 8, p.y - 17, 34, 14, tex`x_${index + 1}`, 8, { color: C.poseDeep, fontWeight: 700 }))
  })

  const routeColors = [C.measurement, C.map, C.pose]
  scan.predictions.forEach((prediction, index) => {
    if (prediction.kind === 'los') {
      elements.push(line('s1-route-los', bs.x, bs.y, activePose.x, activePose.y, routeColors[index], 4))
      return
    }
    const reflection = point(prediction.reflectionPoint)
    const va = point(setup.virtualAnchors[prediction.pathIndex - 1])
    elements.push(line(`s1-route-${index}-a`, bs.x, bs.y, reflection.x, reflection.y, routeColors[index], 4))
    elements.push(line(`s1-route-${index}-b`, reflection.x, reflection.y, activePose.x, activePose.y, routeColors[index], 4))
    elements.push(line(`s1-unfold-${index}`, activePose.x, activePose.y, va.x, va.y, routeColors[index], 1.5, { opacity: .38, dash: [5, 4] }))
    elements.push(shape(`s1-reflection-${index}`, reflection.x - 5, reflection.y - 5, 10, 10, C.paper, { shape: 'ellipse', stroke: routeColors[index], strokeWidth: 2 }))
  })

  elements.push(shape('s1-bs', bs.x - 7, bs.y - 7, 14, 14, C.ink, { radius: 0 }))
  elements.push(text('s1-bs-label', bs.x + 10, bs.y + 5, 58, 16, 'BS', 9, { color: C.ink, fontFamily: MONO, fontWeight: 700 }))
  setup.virtualAnchors.forEach((va, index) => {
    const p = point(va)
    elements.push(shape(`s1-va-${index}`, p.x - 9, p.y - 9, 18, 18, index === 0 ? C.mapSoft : C.poseSoft, { shape: 'ellipse', stroke: index === 0 ? C.map : C.pose, strokeWidth: 2 }))
    elements.push(text(`s1-va-label-${index}`, p.x - 29, p.y - 29, 58, 16, tex`v_${index === 0 ? 'A' : 'B'}`, 9, { color: index === 0 ? C.mapDeep : C.poseDeep, fontWeight: 700, align: 'center' }))
  })
  elements.push(text('s1-route-legend', 126, 590, 600, 18, `TRUTH ONLY · ${tex`\mathrm{LoS}`} · ${tex`\mathcal W_A`} reflection · ${tex`\mathcal W_B`} reflection · faint = unfolded VA ray`, 9, { color: C.faint, fontFamily: MONO, fontWeight: 700, align: 'center' }))

  elements.push(card('s1-geometry-card', 782, 202, 402, 142, C.mapSoft, { stroke: C.map, radius: 8 }))
  elements.push(text('s1-geometry-k', 806, 220, 354, 17, 'KNOWN BS · WALL / VA GENERATOR TRUTH', 9, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .75 }))
  elements.push(text('s1-geometry-eq', 806, 248, 354, 82, texBlock`\begin{aligned}
    \mathbf b&=[2,\,2]^{\mathsf T}\ \mathrm m\\
    \mathcal W_A&:y=7\ \mathrm m,\qquad \mathcal W_B:x=8.5\ \mathrm m\\
    \mathbf v_A&=[2,\,12]^{\mathsf T}\ \mathrm m,\qquad \mathbf v_B=[15,\,2]^{\mathsf T}\ \mathrm m
  \end{aligned}`, 15, { fontWeight: 700, align: 'center', lineHeight: 1.4 }))

  elements.push(card('s1-poses-card', 782, 362, 402, 132, C.paper, { stroke: C.line, radius: 8 }))
  elements.push(text('s1-poses-k', 806, 380, 354, 17, 'REFERENCE UE TRAJECTORY', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.1 }))
  elements.push(text('s1-poses-eq', 806, 406, 354, 74, texBlock`\begin{aligned}
    \mathbf x_1&=(2.8,6.2,-18^\circ)&\mathbf x_2&=(3.9,5.6,-13^\circ)\\
    \mathbf x_3&=(5.0,5.0,-6^\circ)&\mathbf x_4&=(6.1,4.3,3^\circ)\\
    \mathbf x_5&=(7.2,3.5,12^\circ)
  \end{aligned}`, 12, { fontWeight: 700, align: 'center', lineHeight: 1.45 }))

  elements.push(card('s1-data-card', 782, 512, 402, 110, C.measurementSoft, { stroke: C.measurement, radius: 8 }))
  elements.push(text('s1-data-k', 806, 530, 354, 17, 'UNLABELED MPC REALIZATION', 9, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.05 }))
  elements.push(text('s1-data-eq', 806, 554, 354, 24, texBlock`\mathbf z_{t\ell}=(\tau_{t\ell},\varphi_{t\ell}^{\mathrm{AoA}},\varphi_{t\ell}^{\mathrm{AoD}},g_{t\ell}^{\mathrm{dB}})`, 13, { fontWeight: 700, align: 'center' }))
  elements.push(text('s1-data-v', 806, 586, 354, 24, `row index ${tex`\ell`} is not a path label · ${tex`\sigma_L=0.08\,\mathrm m`} · ${tex`\sigma_{\angle}=1.4^\circ`}`, 10, { color: C.soft, fontFamily: SANS, fontWeight: 700, align: 'center' }))

  elements.push(card('s1-state-card', 96, 638, 1088, 36, C.poseDeep, { stroke: C.poseDeep, radius: 6 }))
  elements.push(text('s1-state-v', 116, 647, 1048, 18, 'NEXT: BP/PMBM condition truth associations to expose structure   ·   LATER: implemented GraphSLAM consumes unlabeled MPC sets', 11, { color: C.paper, fontFamily: SANS, fontWeight: 700, align: 'center' }))

  return regular(
    's-radio-slam-s1', 'CONDITIONED TEACHING SETUP',
    'Setup S1: a truth overlay for two structural walkthroughs',
    'Route colors, walls, VAs, and ordering are generator truth—not estimator inputs. BP/PMBM condition the associations only to expose structure; implemented GraphSLAM later consumes unlabeled MPC sets.',
    'Use this slide only as the controlled teaching setup for the following BP and PMBM reductions. The BS pose is known; the plotted poses, route colors, walls, and virtual anchors are simulation-truth annotations, and the measurement row index is not a semantic path label. The next two walkthroughs deliberately condition association variables so their state–map structures stay readable. Do not carry that conditioning into Section 04: the implemented GraphSLAM uses unlabeled MPC sets, estimates 3-D position nodes only (orientation is fixed/world-aligned), and constructs registration and gated-LoS factors in its front end. S1 stores gain magnitude / dB rather than complex phase; the current GraphSLAM implementation discards amplitude.',
    elements,
    { accent: C.map, titleSize: 34, transition: 'none' }
  )
}

function methodEquationSlide(kind, ctx) {
  const { regular, text, card, shape, line, C, SANS, MONO, tex, texBlock } = ctx
  if (kind === 'bp') {
    return regular(
      's-bp-slam-equations', '03 · UNKNOWN UE STATE AND MAP',
      'BP-SLAM on S1: one graph, two marginal families',
      'Motion factors connect the latent trajectory; radio factors couple every UE state to the shared map; sum–product returns state and map beliefs.',
      'Read the top equation as the state–map factorization used by the following worked example. The known inputs are the BS, odometry, and radio tuples. To keep the structural coupling visible, the S1 route labels A are conditioned in this reduction. Full multipath BP-SLAM also represents feature-existence and association variables. The important result here is that trajectory and map are both variable nodes, and BP returns a marginal belief for each rather than one joint point estimate.',
      [
        card('bp-post-card', 96, 202, 1088, 132, C.mapSoft, { stroke: C.map, strokeWidth: 2, radius: 8 }),
        text('bp-post-k', 122, 218, 330, 18, 'JOINT STATE–MAP FACTORIZATION', 10, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.2 }),
        text('bp-post-eq', 118, 248, 1044, 70,
          texBlock`\begin{aligned}
            p(\mathbf X,\mathcal M\mid Z,U,\mathbf b,A)\propto\;&p(\mathbf x_1)\prod_{t=2}^{5}f_t^{\mathrm{mot}}(\mathbf x_{t-1},\mathbf x_t;\mathbf u_t)\\[-.1em]
            &\times\prod_{j\in\{A,B\}}p(\mathbf m_j)\prod_{t=1}^{5}f_t^{\mathrm{rad}}(\mathbf x_t,\mathbf m_A,\mathbf m_B;Z_t,A_t,\mathbf b)
          \end{aligned}`,
          16, { fontWeight: 700, align: 'center', valign: 'middle', lineHeight: 1.35 }),

        card('bp-msg-card', 96, 356, 526, 212, C.paper, { stroke: C.line, radius: 8 }),
        text('bp-msg-k', 122, 376, 470, 18, 'FACTOR FAMILIES', 10, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.2 }),
        text('bp-msg-eq-1', 122, 410, 470, 42,
          texBlock`f_t^{\mathrm{mot}}(\mathbf x_{t-1},\mathbf x_t;\mathbf u_t)`,
          18, { fontWeight: 700, align: 'center' }),
        text('bp-msg-v-1', 122, 452, 470, 28, 'Odometry couples consecutive UE states.', 12, { color: C.soft, fontFamily: SANS, align: 'center' }),
        text('bp-msg-eq-2', 122, 486, 470, 42,
          texBlock`f_t^{\mathrm{rad}}(\mathbf x_t,\mathbf m_A,\mathbf m_B;Z_t,A_t,\mathbf b)`,
          17, { fontWeight: 700, align: 'center' }),
        text('bp-msg-v-2', 122, 530, 470, 24, 'Each scan couples one UE state to the shared map.', 12, { color: C.soft, fontFamily: SANS, align: 'center' }),

        card('bp-radio-card', 658, 356, 526, 212, C.measurementSoft, { stroke: C.measurement, radius: 8 }),
        text('bp-radio-k', 684, 376, 470, 18, 'SUM–PRODUCT RETURNS TWO BELIEF FAMILIES', 10, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.05 }),
        text('bp-radio-eq', 684, 408, 470, 104,
          texBlock`\begin{aligned}
            b(\mathbf x_t)&\propto\prod_{f\in\mathcal N(\mathbf x_t)}\mu_{f\to\mathbf x_t}(\mathbf x_t)\\[.6em]
            b(\mathbf m_j)&\propto p(\mathbf m_j)\prod_{t=1}^{5}\mu_{f_t^{\mathrm{rad}}\to\mathbf m_j}(\mathbf m_j)
          \end{aligned}`,
          17, { fontWeight: 700, align: 'center', lineHeight: 1.5 }),
        text('bp-radio-note', 684, 520, 470, 34, 'Radio-factor messages make state and map inform each other.', 12, { color: C.soft, fontFamily: SANS, lineHeight: 1.35, align: 'center' }),

        card('bp-return-card', 96, 590, 1088, 46, C.mapDeep, { stroke: C.mapDeep, radius: 7 }),
        text('bp-return', 120, 602, 1040, 22, texBlock`\mathrm{RETURN}\ \cdot\ \{b(\mathbf x_t)\}_{t=1}^{5}\quad\text{and}\quad\{b(\mathbf m_j)\}_{j\in\{A,B\}}`, 14, { color: C.paper, fontWeight: 700, align: 'center' }),
        text('bp-source', 96, 650, 1088, 17, 'Structural S1 reduction: A is conditioned; full multipath BP-SLAM augments this graph with existence and association variables.', 9, { color: C.faint, fontFamily: MONO, align: 'center' })
      ], { accent: C.map, titleSize: 33, transition: 'none' }
    )
  }

  return regular(
    's-pmbm-slam-equations', '03 · UNKNOWN UE STATE AND MAP',
    'PMBM-SLAM on S1: trajectory density × conditional RFS map',
    'First factor the joint posterior into vehicle trajectory and conditional map; then represent each particle-conditioned map as a PPP plus an MBM.',
    'Start with the exact state–map factorization. A Rao–Blackwellized particle filter represents the vehicle-trajectory density. For every trajectory particle, the conditional map is PMBM: a Poisson point process for never-detected map features plus a multi-Bernoulli mixture for detected features. Global hypotheses remain inside the conditional map representation; they are supporting structure, not the headline.',
    [
      card('pmbm-post-card', 96, 202, 1088, 112, C.measurementSoft, { stroke: C.measurement, strokeWidth: 2, radius: 8 }),
      text('pmbm-post-k', 122, 218, 290, 18, 'EXACT JOINT FACTORIZATION', 10, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.2 }),
      text('pmbm-post-eq', 122, 246, 1036, 52,
        texBlock`f(\mathbf X,\mathcal M\mid Z,U)=f(\mathbf X\mid Z,U)\,f(\mathcal M\mid\mathbf X,Z,U)`,
        21, { fontWeight: 700, align: 'center', valign: 'middle' }),

      card('pmbm-poisson-card', 96, 338, 344, 224, C.paper, { stroke: C.line, radius: 8 }),
      text('pmbm-poisson-k', 120, 358, 296, 18, 'VEHICLE TRAJECTORY PARTICLES', 10, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.0 }),
      text('pmbm-poisson-eq', 120, 400, 296, 72, texBlock`f(\mathbf X\mid Z,U)\approx\sum_{n=1}^{N}w^{(n)}\delta(\mathbf X-\mathbf X^{(n)})`, 18, { fontWeight: 700, align: 'center' }),
      text('pmbm-poisson-v', 120, 488, 296, 50, 'Each weighted particle is one complete UE-trajectory sample.', 13, { color: C.soft, fontFamily: SANS, lineHeight: 1.4, align: 'center' }),

      card('pmbm-bern-card', 468, 338, 344, 224, C.mapSoft, { stroke: C.map, radius: 8 }),
      text('pmbm-bern-k', 492, 358, 296, 18, 'CONDITIONAL MAP · PPP', 10, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.1 }),
      text('pmbm-bern-eq', 492, 400, 296, 72, texBlock`f_{\mathrm P}^{u,(n)}(\mathcal M^u)=e^{-\Lambda^{u,(n)}}\prod_{\mathbf m\in\mathcal M^u}\lambda^{u,(n)}(\mathbf m)`, 16, { fontWeight: 700, align: 'center', lineHeight: 1.35 }),
      text('pmbm-bern-v', 492, 488, 296, 50, 'The PPP represents map features not yet detected under that trajectory.', 13, { color: C.soft, fontFamily: SANS, lineHeight: 1.4, align: 'center' }),

      card('pmbm-hyp-card', 840, 338, 344, 224, C.measurementSoft, { stroke: C.measurement, radius: 8 }),
      text('pmbm-hyp-k', 864, 358, 296, 18, 'CONDITIONAL MAP · MBM', 10, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.05 }),
      text('pmbm-hyp-eq', 864, 396, 296, 80, texBlock`\sum_{h\in\mathcal H^{(n)}}w_h^{(n)}\prod_i f_{\mathrm B}^{(n,h,i)}(\mathcal M^i;r_i^{(n,h)},p_i^{(n,h)})`, 16, { fontWeight: 700, align: 'center', lineHeight: 1.4 }),
      text('pmbm-hyp-v', 864, 488, 296, 52, 'The MBM represents detected map features; global histories live inside this conditional map.', 13, { color: C.soft, fontFamily: SANS, lineHeight: 1.4, align: 'center' }),

      card('pmbm-return-card', 96, 590, 1088, 46, C.measurementDeep, { stroke: C.measurementDeep, radius: 7 }),
      text('pmbm-return', 120, 602, 1040, 22, texBlock`f(\mathbf X,\mathcal M\mid Z,U)\approx\sum_{n=1}^{N}w^{(n)}\delta(\mathbf X-\mathbf X^{(n)})\,f_{\mathrm{PMBM}}^{(n)}(\mathcal M)`, 14, { color: C.paper, fontWeight: 700, align: 'center' }),
      text('pmbm-source', 96, 650, 1088, 17, 'The following live example selects a trajectory particle and opens its PPP + MBM conditional map.', 9, { color: C.faint, fontFamily: MONO, align: 'center' })
    ], { accent: C.measurement, titleSize: 32, transition: 'none' }
  )
}

function stateMapFallback(kind, ctx) {
  const { text, card, shape, line, C, MONO, SANS, LIVE_BOUNDS, tex } = ctx
  const x0 = LIVE_BOUNDS.x, y0 = LIVE_BOUNDS.y, w = LIVE_BOUNDS.width, h = LIVE_BOUNDS.height
  const stageX = x0 + 14, stageY = y0 + 14, stageW = 744, stageH = h - 28
  const railX = stageX + stageW + 12, railW = w - stageW - 40
  const accent = kind === 'bp' ? C.map : C.measurement
  const deep = kind === 'bp' ? C.mapDeep : C.measurementDeep
  const soft = kind === 'bp' ? C.mapSoft : C.measurementSoft
  const elements = [
    card('state-map-' + kind + '-bg', x0, y0, w, h, '#F8FAFB', { stroke: C.line, radius: 0 }),
    card('state-map-' + kind + '-stage', stageX, stageY, stageW, stageH, C.paper, { stroke: C.line, radius: 6 }),
    card('state-map-' + kind + '-rail', railX, stageY, railW, stageH, C.paper, { stroke: C.line, radius: 6 })
  ]

  if (kind === 'bp') {
    const mapY = stageY + 90, radioY = stageY + 218, stateY = stageY + 352
    const mapNodes = [[stageX + 245, mapY, tex`\mathbf m_A`, C.map, C.mapSoft], [stageX + 515, mapY, tex`\mathbf m_B`, C.measurement, C.measurementSoft]]
    const stateXs = [stageX + 82, stageX + 226, stageX + 370, stageX + 514, stageX + 658]
    elements.push(text('bp-state-map-title', stageX + 22, stageY + 18, stageW - 44, 18, 'BP-SLAM · ONE FACTOR GRAPH COUPLES UE STATE AND MAP', 10, { color: deep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.1 }))
    elements.push(text('bp-map-family', stageX + 26, stageY + 68, 190, 16, 'LATENT MAP · ' + tex`\mathcal M`, 8, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700 }))
    elements.push(text('bp-state-family', stageX + 26, stageY + 326, 220, 16, 'LATENT TRAJECTORY · ' + tex`\mathbf X`, 8, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700 }))
    mapNodes.forEach((item, index) => {
      elements.push(shape('bp-map-node-' + index, item[0] - 18, item[1] - 18, 36, 36, item[4], { shape: 'ellipse', stroke: item[3], strokeWidth: 3 }))
      elements.push(text('bp-map-label-' + index, item[0] - 24, item[1] - 7, 48, 14, item[2], 9, { color: index ? C.measurementDeep : C.mapDeep, fontWeight: 700, align: 'center' }))
    })
    stateXs.forEach((x, index) => {
      mapNodes.forEach((mapNode, mapIndex) => elements.push(line('bp-radio-map-edge-' + index + '-' + mapIndex, mapNode[0], mapNode[1] + 19, x, radioY - 8, mapIndex ? C.measurement : C.map, 2, { opacity: .34 })))
      elements.push(line('bp-radio-state-edge-' + index, x, radioY + 8, x, stateY - 17, C.pose, 2))
      elements.push(shape('bp-radio-factor-' + index, x - 8, radioY - 8, 16, 16, C.paper, { stroke: C.measurement, strokeWidth: 2, radius: 0 }))
      elements.push(text('bp-radio-factor-label-' + index, x - 28, radioY + 13, 56, 14, tex`f_{${index + 1}}^{\mathrm{rad}}`, 7, { color: C.measurementDeep, fontWeight: 700, align: 'center' }))
      elements.push(shape('bp-state-node-' + index, x - 16, stateY - 16, 32, 32, C.paper, { shape: 'ellipse', stroke: C.pose, strokeWidth: 3 }))
      elements.push(text('bp-state-label-' + index, x - 18, stateY - 7, 36, 14, tex`\mathbf x_{${index + 1}}`, 8, { color: C.poseDeep, fontWeight: 700, align: 'center' }))
      if (index < stateXs.length - 1) {
        const fx = 0.5 * (x + stateXs[index + 1])
        elements.push(line('bp-motion-edge-a-' + index, x + 17, stateY, fx - 7, stateY, C.pose, 2))
        elements.push(shape('bp-motion-factor-' + index, fx - 7, stateY - 7, 14, 14, C.poseSoft, { stroke: C.pose, strokeWidth: 2, radius: 0 }))
        elements.push(line('bp-motion-edge-b-' + index, fx + 7, stateY, stateXs[index + 1] - 17, stateY, C.pose, 2))
      }
    })
    elements.push(text('bp-rail-k', railX + 18, stageY + 18, railW - 36, 18, 'WORKED S1 FACTORIZATION', 9, { color: deep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.05 }))
    elements.push(card('bp-rail-eq', railX + 18, stageY + 54, railW - 36, 116, C.mapSoft, { stroke: C.map, radius: 6 }))
    elements.push(text('bp-rail-eq-v', railX + 30, stageY + 72, railW - 60, 82, tex`\begin{aligned}p(\mathbf X,\mathcal M\mid Z,U,\mathbf b,A)&\propto p(\mathbf x_1)\prod_t f_t^{\mathrm{mot}}\\&\quad\times\prod_jp(\mathbf m_j)\prod_t f_t^{\mathrm{rad}}\end{aligned}`, 10, { fontWeight: 700, align: 'center', valign: 'middle', lineHeight: 1.35 }))
    ;['motion chain', 'radio coupling', 'map ↔ state messages'].forEach((label, index) => {
      const y = stageY + 188 + index * 49
      elements.push(card('bp-step-card-' + index, railX + 18, y, railW - 36, 38, index === 1 ? C.measurementSoft : '#FBFCFD', { stroke: index === 1 ? C.measurement : C.line, radius: 5 }))
      elements.push(text('bp-step-label-' + index, railX + 28, y + 10, railW - 56, 18, label, 10, { color: index === 1 ? C.measurementDeep : C.soft, fontFamily: SANS, fontWeight: 700, align: 'center' }))
    })
    elements.push(card('bp-fallback-result', railX + 18, stageY + 344, railW - 36, 64, C.mapDeep, { stroke: C.mapDeep, radius: 6 }))
    elements.push(text('bp-fallback-result-v', railX + 28, stageY + 360, railW - 56, 30, tex`\{b(\mathbf x_t)\}_{1:5}\quad+\quad\{b(\mathbf m_j)\}_{A,B}`, 12, { color: C.paper, fontWeight: 700, align: 'center' }))
    return elements
  }

  elements.push(text('pmbm-state-map-title', stageX + 22, stageY + 18, stageW - 44, 18, 'PMBM-SLAM · TRAJECTORY DENSITY × CONDITIONAL RFS MAP', 10, { color: deep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.1 }))
  elements.push(card('pmbm-joint-node', stageX + 165, stageY + 54, 414, 54, C.measurementSoft, { stroke: C.measurement, strokeWidth: 2, radius: 6 }))
  elements.push(text('pmbm-joint-eq', stageX + 180, stageY + 70, 384, 24, tex`f(\mathbf X,\mathcal M\mid Z,U)`, 16, { color: C.measurementDeep, fontWeight: 700, align: 'center' }))
  elements.push(text('pmbm-factor-op', stageX + 350, stageY + 112, 44, 20, tex`=`, 16, { color: C.measurementDeep, fontWeight: 700, align: 'center' }))
  elements.push(card('pmbm-state-density', stageX + 74, stageY + 142, 254, 58, C.poseSoft, { stroke: C.pose, radius: 6 }))
  elements.push(text('pmbm-state-density-eq', stageX + 88, stageY + 158, 226, 26, tex`f(\mathbf X\mid Z,U)`, 16, { color: C.poseDeep, fontWeight: 700, align: 'center' }))
  elements.push(text('pmbm-times-op', stageX + 350, stageY + 160, 44, 24, tex`\times`, 16, { color: C.measurementDeep, fontWeight: 700, align: 'center' }))
  elements.push(card('pmbm-map-density', stageX + 416, stageY + 142, 254, 58, C.measurementSoft, { stroke: C.measurement, radius: 6 }))
  elements.push(text('pmbm-map-density-eq', stageX + 430, stageY + 158, 226, 26, tex`f(\mathcal M\mid\mathbf X,Z,U)`, 15, { color: C.measurementDeep, fontWeight: 700, align: 'center' }))
  ;[0.58, 0.27, 0.15].forEach((weight, index) => {
    const x = stageX + 74 + index * 88
    elements.push(card('pmbm-particle-' + index, x, stageY + 222, 78, 52, index === 0 ? C.poseSoft : '#FBFCFD', { stroke: index === 0 ? C.pose : C.line, radius: 5 }))
    elements.push(text('pmbm-particle-v-' + index, x + 6, stageY + 232, 66, 32, tex`\mathbf X^{(${index + 1})}\\w^{(${index + 1})}=${weight.toFixed(2)}`, 9, { color: index === 0 ? C.poseDeep : C.soft, fontWeight: 700, align: 'center', lineHeight: 1.3 }))
  })
  elements.push(card('pmbm-conditional-selected', stageX + 360, stageY + 222, 310, 52, C.measurementSoft, { stroke: C.measurement, radius: 5 }))
  elements.push(text('pmbm-conditional-selected-v', stageX + 374, stageY + 234, 282, 28, tex`\mathbf X^{(1)}\Longrightarrow f_{\mathrm{PMBM}}^{(1)}(\mathcal M)`, 13, { color: C.measurementDeep, fontWeight: 700, align: 'center' }))
  elements.push(card('pmbm-ppp-node', stageX + 118, stageY + 310, 230, 74, C.mapSoft, { stroke: C.map, radius: 6 }))
  elements.push(text('pmbm-ppp-k', stageX + 132, stageY + 322, 202, 14, 'UNDETECTED · PPP', 8, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, align: 'center' }))
  elements.push(text('pmbm-ppp-eq', stageX + 132, stageY + 345, 202, 24, tex`f_{\mathrm P}^{u,(1)}(\mathcal M^u;\lambda^{u,(1)})`, 11, { color: C.mapDeep, fontWeight: 700, align: 'center' }))
  elements.push(text('pmbm-union-op', stageX + 356, stageY + 334, 36, 26, tex`\uplus`, 17, { color: C.measurementDeep, fontWeight: 700, align: 'center' }))
  elements.push(card('pmbm-mbm-node', stageX + 400, stageY + 310, 270, 74, C.measurementSoft, { stroke: C.measurement, radius: 6 }))
  elements.push(text('pmbm-mbm-k', stageX + 414, stageY + 322, 242, 14, 'DETECTED · MBM', 8, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, align: 'center' }))
  elements.push(text('pmbm-mbm-eq', stageX + 414, stageY + 344, 242, 28, tex`\sum_h w_h^{(1)}\prod_i f_{\mathrm B}^{(1,h,i)}(\mathcal M^i)`, 11, { color: C.measurementDeep, fontWeight: 700, align: 'center' }))

  elements.push(text('pmbm-rail-k', railX + 18, stageY + 18, railW - 36, 18, 'SELECT TRAJECTORY PARTICLE', 9, { color: deep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.05 }))
  ;['particle 1 · 0.58', 'particle 2 · 0.27', 'particle 3 · 0.15'].forEach((label, index) => {
    const y = stageY + 56 + index * 52
    elements.push(card('pmbm-particle-control-' + index, railX + 18, y, railW - 36, 40, index === 0 ? C.measurementSoft : '#FBFCFD', { stroke: index === 0 ? C.measurement : C.line, radius: 5 }))
    elements.push(text('pmbm-particle-control-v-' + index, railX + 28, y + 11, railW - 56, 18, label, 10, { color: index === 0 ? C.measurementDeep : C.soft, fontFamily: SANS, fontWeight: 700, align: 'center' }))
  })
  elements.push(card('pmbm-fallback-eq-card', railX + 18, stageY + 230, railW - 36, 104, C.measurementSoft, { stroke: C.measurement, radius: 6 }))
  elements.push(text('pmbm-fallback-eq-v', railX + 30, stageY + 246, railW - 60, 72, tex`f(\mathbf X,\mathcal M\mid Z,U)\approx\sum_n w^{(n)}\delta(\mathbf X-\mathbf X^{(n)})f_{\mathrm{PMBM}}^{(n)}(\mathcal M)`, 10, { fontWeight: 700, align: 'center', valign: 'middle', lineHeight: 1.3 }))
  elements.push(card('pmbm-fallback-result', railX + 18, stageY + 350, railW - 36, 58, C.measurementDeep, { stroke: C.measurementDeep, radius: 6 }))
  elements.push(text('pmbm-fallback-result-v', railX + 28, stageY + 365, railW - 56, 28, 'trajectory mixture + conditional PPP / MBM map', 11, { color: C.paper, fontFamily: SANS, fontWeight: 700, align: 'center' }))
  return elements
}

function bundleAdjustmentSlide(ctx) {
  const { regular, text, card, shape, line, C, SANS, MONO, tex, texBlock } = ctx
  const elements = [
    card('ba-graph-card', 96, 202, 430, 420, C.paper, { stroke: C.line, radius: 8 }),
    text('ba-graph-k', 118, 220, 386, 18, 'BIPARTITE OBSERVATION GRAPH · FIXED ASSOCIATIONS', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .75 }),
    text('ba-graph-sub', 118, 244, 386, 18, 'the same map entity observed from several poses couples those poses indirectly', 9, { color: C.soft, fontFamily: SANS, fontWeight: 700, align: 'center' }),
    shape('ba-bs-node', 296, 266, 28, 28, C.ink, { shape: 'ellipse', stroke: C.paper, strokeWidth: 2 }),
    text('ba-bs-label', 250, 248, 120, 16, tex`\mathbf b\;\text{known}`, 9, { color: C.ink, fontWeight: 700, align: 'center' }),
    text('ba-bs-caption', 224, 296, 172, 14, 'fixed parameter in every radio factor', 7.5, { color: C.faint, fontFamily: MONO, align: 'center' })
  ]

  const poses = [
    { x: 154, y: 338, label: tex`\mathbf T_1` },
    { x: 154, y: 416, label: tex`\mathbf T_2` },
    { x: 154, y: 494, label: tex`\mathbf T_3` },
    { x: 154, y: 572, label: tex`\mathbf T_4` }
  ]
  const maps = [
    { x: 468, y: 346, label: tex`\mathbf m_A`, color: C.map, fill: C.mapSoft },
    { x: 468, y: 460, label: tex`\mathbf m_B`, color: C.known, fill: C.knownSoft },
    { x: 468, y: 566, label: tex`\mathbf m_C`, color: C.measurement, fill: C.measurementSoft }
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
  elements.push(text('ba-objective-eq', 560, 244, 612, 92, texBlock`\begin{aligned}
    (\mathbf X^*,\mathcal M^*)&=\arg\min_{\mathbf X,\mathcal M}\sum_{(t,\ell)\in\mathcal O}\rho\!\left(\left\|\mathbf z_{t\ell}\boxminus\mathbf h_{q_{t\ell}}(\mathbf T_t,\mathbf m_{a_{t\ell}},\mathbf b)\right\|_{\Omega_{t\ell}}^2\right)\\[-.1em]
    &\qquad+\left\|\mathbf r_{\mathrm{anchor}}\right\|_{\Omega_{\mathrm{anchor}}}^{2},\qquad \Omega=\Sigma^{-1},\\[-.1em]
    \mathbf h_{\mathrm{cam}}&=\pi(\mathbf T_t\mathbf P_j),\qquad
    \mathbf h_{\mathrm{rad},q}=[L_q,\varphi_q^{\mathrm{AoA}},\varphi_q^{\mathrm{AoD}}]^{\mathsf T}.
  \end{aligned}`, 11.1, { fontWeight: 700, align: 'center', lineHeight: 1.24 }))
  elements.push(text('ba-objective-v', 572, 344, 588, 20, 'Classical BA uses image reprojection; the radio analogue replaces pixels with path length and angle residuals.', 9.5, { color: C.poseDeep, fontFamily: SANS, fontWeight: 700, align: 'center' }))

  elements.push(card('ba-schur-card', 548, 392, 636, 142, C.paper, { stroke: C.line, radius: 8 }))
  elements.push(text('ba-schur-k', 572, 408, 410, 18, 'GAUSS–NEWTON + SCHUR COMPLEMENT', 9.5, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .9 }))
  elements.push(text('ba-schur-eq', 560, 432, 612, 70, texBlock`\begin{aligned}
    \begin{bmatrix}\mathbf H_{XX}&\mathbf H_{XM}\\\mathbf H_{MX}&\mathbf H_{MM}\end{bmatrix}
    \begin{bmatrix}\Delta\mathbf X\\\Delta\mathcal M\end{bmatrix}&=-\begin{bmatrix}\mathbf g_X\\\mathbf g_M\end{bmatrix},\\[-.1em]
    (\mathbf H_{XX}-\mathbf H_{XM}\mathbf H_{MM}^{-1}\mathbf H_{MX})\Delta\mathbf X&=-(\mathbf g_X-\mathbf H_{XM}\mathbf H_{MM}^{-1}\mathbf g_M).
  \end{aligned}`, 11.2, { fontWeight: 700, align: 'center', lineHeight: 1.2 }))
  elements.push(text('ba-schur-v', 572, 506, 588, 18, 'Eliminate independent map blocks, solve the sparse pose system, then back-substitute the map update.', 9.5, { color: C.soft, fontFamily: SANS, fontWeight: 700, align: 'center' }))

  elements.push(card('ba-bridge-card', 548, 550, 636, 72, C.measurementSoft, { stroke: C.measurement, radius: 8 }))
  elements.push(text('ba-bridge-k', 572, 562, 234, 16, 'BRIDGE TO THE NEXT SLIDE', 9, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .8 }))
  elements.push(text('ba-bridge-eq', 566, 582, 600, 24, texBlock`\mathcal F_{\mathrm{GraphSLAM}}=\mathcal F_{\mathrm{BA}}\cup\mathcal F_{\mathrm{rel}}\cup\mathcal F_{\mathrm{loop}}\cup\mathcal F_{\mathrm{prior}}`, 12.5, { color: C.measurementDeep, fontWeight: 700, align: 'center' }))
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
}

function graphEquationSlide(ctx) {
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
  graph.push(text('gs-joint-bs-label', bs.x - 48, bs.y - 39, 96, 18, tex`\mathbf b\;\text{known}`, 9, { color: C.ink, fontWeight: 700, align: 'center' }))
  ;[
    [tex`\mathbf m_A`, mapA, C.map, C.mapSoft],
    [tex`\mathbf m_B`, mapB, C.known, C.knownSoft]
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
    graph.push(text('gs-joint-relative-label-' + index, factorX - 21, poseY + 12, 42, 15, tex`f^{\mathrm{rel}}`, 6.5, { color: C.faint, fontWeight: 700, align: 'center' }))
  })
  const poseLabels = [tex`\mathbf T_1`, tex`\mathbf T_2`, tex`\mathbf T_3`, tex`\mathbf T_4`, tex`\mathbf T_5`]
  poseXs.forEach((x, index) => {
    graph.push(shape('gs-joint-pose-node-' + index, x - 15, poseY - 15, 30, 30, index === 0 ? C.poseSoft : C.paper, { shape: 'ellipse', stroke: C.pose, strokeWidth: 2 }))
    graph.push(text('gs-joint-pose-label-' + index, x - 16, poseY - 8, 32, 17, poseLabels[index], 8.5, { color: C.poseDeep, fontWeight: 700, align: 'center' }))
  })
  const radioLabels = [tex`f^{\mathrm{rad}}_{1A}`, tex`f^{\mathrm{rad}}_{3A}`, tex`f^{\mathrm{rad}}_{3B}`, tex`f^{\mathrm{rad}}_{5B}`]
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
      text('gs-joint-state-eq', 650, 240, 524, 50, texBlock`\begin{aligned}
        \mathbf X&=\{\mathbf T_t\}_{t=1}^{T},&\mathbf T_t&=(\mathbf p_t,\theta_t)\in SE(2),&\mathcal M&=\{\mathbf m_j\}_{j=1}^{J}\\
        a_{t\ell}&\in\{0,1,\ldots,J\},&&q_{t\ell}\in\{\mathrm{LoS},1,2,\ldots\}\quad(a_{t\ell}>0)
      \end{aligned}`, 10.5, { fontWeight: 700, align: 'center', lineHeight: 1.3 }),
      text('gs-joint-state-v', 664, 292, 496, 12, tex`a_{t\ell}=0\;\text{means clutter / no map assignment}`, 7.5, { color: C.poseDeep, fontWeight: 700, align: 'center' }),

      card('gs-joint-factor-card', 640, 322, 544, 146, C.paper, { stroke: C.line, radius: 8 }),
      text('gs-joint-factor-k', 664, 338, 330, 16, 'ODOMETRY LIKELIHOOD, NOT A REQUIRED DYNAMICS LAW', 8.5, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .55 }),
      text('gs-joint-factor-eq', 650, 358, 524, 74, texBlock`\begin{aligned}
        p(\mathbf X,\mathcal M,A,Q\mid Z,\widetilde U,\mathbf b)\propto{}&p(\mathbf T_1)\prod_{t=2}^{T}p(\widetilde{\mathbf T}_{t-1,t}\mid\mathbf T_{t-1},\mathbf T_t)\prod_jp(\mathbf m_j)\\[-.2em]
        &\times\prod_{t,\ell}p(a_{t\ell},q_{t\ell})\,p(\mathbf z_{t\ell}\mid\mathbf T_t,\mathcal M,a_{t\ell},q_{t\ell},\mathbf b)
      \end{aligned}`, 9.2, { fontWeight: 700, align: 'center', lineHeight: 1.25 }),
      text('gs-joint-factor-v', 664, 438, 496, 18, tex`\mathbf r_t^{\mathrm{rel}}=\operatorname{Log}(\widetilde{\mathbf T}_{t-1,t}^{-1}\mathbf T_{t-1}^{-1}\mathbf T_t)\quad\text{with covariance }\Sigma_t^{\mathrm{rel}}`, 8, { color: C.soft, fontWeight: 700, align: 'center' }),

      card('gs-joint-cost-card', 640, 482, 544, 140, C.measurementSoft, { stroke: C.measurement, radius: 8 }),
      text('gs-joint-cost-k', 664, 498, 310, 16, 'COVARIANCE-WEIGHTED MAP OBJECTIVE', 8.5, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .75 }),
      text('gs-joint-cost-eq', 650, 518, 524, 62, texBlock`\begin{aligned}
        (\mathbf X^*,\mathcal M^*)&=\arg\min_{\mathbf X,\mathcal M}\;\|\mathbf r_1^{\mathrm{prior}}\|_{\Omega_1}^{2}+\sum_{t=2}^{T}\|\mathbf r_t^{\mathrm{rel}}\|_{\Omega_t^{\mathrm{rel}}}^{2}\\[-.1em]
        &\quad+\sum_{(t,\ell):a_{t\ell}>0}\rho\!\left(\|\mathbf r_{t\ell}^{\mathrm{rad}}(a_{t\ell},q_{t\ell})\|_{\Omega_{t\ell}^{\mathrm{rad}}}^{2}\right),\qquad \Omega=\Sigma^{-1}\\[-.1em]
        \mathbf r_{t\ell}^{\mathrm{rad}}(j,q)&=[c\tau,\varphi^{\mathrm{AoA}},\varphi^{\mathrm{AoD}}]^{\mathsf T}\boxminus\mathbf h_q(\mathbf T_t,\mathbf m_j,\mathbf b)
      \end{aligned}`, 8.8, { fontWeight: 700, align: 'center', lineHeight: 1.2 }),
      text('gs-joint-cost-v', 664, 584, 496, 24, 'Fixed A,Q → nonlinear least squares over associated MPCs; a=0 uses the clutter likelihood. Unknown A,Q → marginalize, maximize, or alternate association and state updates.', 7.5, { color: C.measurementDeep, fontFamily: SANS, fontWeight: 700, align: 'center', lineHeight: 1.25 }),

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
  const fallbackPoseLabels = [tex`\mathbf x_1`, tex`\mathbf x_2`, tex`\mathbf x_3`, tex`\mathbf x_4`, tex`\mathbf x_5`]
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
  ;[[stageY + 140, C.map, C.mapSoft, tex`\mathbf m_A`], [stageY + 310, C.measurement, C.measurementSoft, tex`\mathbf m_B`]].forEach((item, index) => {
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
  elements.push(text('gs-impl-eq-v', 108, 424, 1064, 66, texBlock`\begin{aligned}
    \mathbf p^*=\arg\min_{\mathbf p_{0:K-1}}{}&\|\mathbf p_0-\bar{\mathbf p}_0\|_{\Omega_0}^{2}+\sum_k\|(\mathbf p_k-\mathbf p_{k-1})-\mathbf d_k^{\mathrm{odo}}\|_{\Omega_k^{\mathrm{odo}}}^{2}\\[-.1em]
    &+\sum_k\|\mathbf p_{k-1}-2\mathbf p_k+\mathbf p_{k+1}\|_{\Omega^{\mathrm{smooth}}}^{2}+\sum_k\|\mathbf p_k-\mathbf z_k^{\mathrm{reg/LoS}}\|_{\Omega_k}^{2}\\[-.1em]
    &+\sum_{(i,j)\in\mathcal L}\|(\mathbf p_j-\mathbf p_i)-\mathbf d_{ij}^{\mathrm{loop}}\|_{\Omega_{ij}}^{2},\qquad \Omega=\Sigma^{-1}
  \end{aligned}`, 9.8, { fontWeight: 700, align: 'center', lineHeight: 1.2 }))

  elements.push(card('gs-impl-state-card', 96, 520, 344, 102, C.poseSoft, { stroke: C.pose, radius: 8 }))
  elements.push(text('gs-impl-state-k', 118, 536, 210, 16, 'STATE INSIDE THE GRAPH', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .8 }))
  elements.push(text('gs-impl-state-eq', 118, 560, 300, 24, tex`\mathbf p_k\in\mathbb R^3\quad\text{only}`, 14, { color: C.poseDeep, fontWeight: 700, align: 'center' }))
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

function methodLiveSlide(kind, ctx) {
  const { regular, liveMount, C } = ctx
  if (kind === 'bp') {
    return regular(
      's-bp-slam-live', '03 · UNKNOWN UE STATE AND MAP',
      'BP-SLAM live: walk the joint state–map factor graph',
      'Step from priors to motion and radio factors, then follow messages into marginal beliefs for the five UE states and two map features.',
      'The embedded S1 walkthrough makes the joint factorization concrete. Both the UE trajectory and map are latent. Motion factors connect consecutive UE states, and scan-level radio factors couple each state to the shared map. The route labels are conditioned only for this structural view; full multipath BP-SLAM also carries feature existence and data-association variables.',
      [...stateMapFallback('bp', ctx), liveMount()], { accent: C.map, titleSize: 31, transition: 'none' }
    )
  }
  return regular(
    's-pmbm-slam-live', '03 · UNKNOWN UE STATE AND MAP',
    'PMBM-SLAM live: open the map carried by each trajectory particle',
    'Select an S1 trajectory particle, then inspect its conditional PMBM map: undetected PPP plus detected multi-Bernoulli mixture.',
    'The embedded S1 walkthrough starts from the exact state-times-conditional-map factorization. The vehicle trajectory is represented by weighted particles. Every particle carries its own PMBM map, split into a Poisson point process for undetected features and a multi-Bernoulli mixture for detected features. Association histories remain inside the MBM rather than dominating the visual.',
    [...stateMapFallback('pmbm', ctx), liveMount()], { accent: C.measurement, titleSize: 29, transition: 'none' }
  )
}

export function appendRadioSlamSlidesAfterSection(unit, ctx) {
  if (unit.id === 'map') {
    ctx.slides.push(sharedSetupSlide(ctx))
    ctx.slides.push(methodEquationSlide('bp', ctx))
    ctx.slides.push(methodLiveSlide('bp', ctx))
    ctx.slides.push(methodEquationSlide('pmbm', ctx))
    ctx.slides.push(methodLiveSlide('pmbm', ctx))
  }
  if (unit.id === 'pose') {
    ctx.slides.push(bundleAdjustmentSlide(ctx))
    ctx.slides.push(graphEquationSlide(ctx))
    ctx.slides.push(graphLiveSlide(ctx))
    ctx.slides.push(graphIterationSlide(ctx))
  }
}

export function radioSlamLiveEntries({ slides, LIVE_BOUNDS }) {
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
