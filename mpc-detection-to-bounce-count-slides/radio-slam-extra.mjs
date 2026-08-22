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
    'Use this slide only as the controlled teaching setup for the following BP and PMBM reductions. The BS pose is known; the plotted poses, route colors, walls, and virtual anchors are simulation-truth annotations, and the measurement row index is not a semantic path label. The next two walkthroughs deliberately condition association variables so their state–map structures stay readable. Do not carry that conditioning into Section 04: the implemented GraphSLAM uses unlabeled MPC sets, estimates pose nodes only, and constructs registration and gated-LoS factors in its front end. S1 stores gain magnitude / dB rather than complex phase; the current GraphSLAM implementation discards amplitude.',
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

function graphEquationSlide(ctx) {
  const { regular, text, card, shape, line, C, SANS, MONO, tex, texBlock } = ctx
  const graph = [
    card('gs-graph-card', 96, 202, 520, 420, C.paper, { stroke: C.line, radius: 8 }),
    text('gs-graph-k', 120, 222, 472, 18, 'POSE GRAPH · ONE FIXED-MAP SOLVE', 10, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.2 }),
    card('gs-map-input', 184, 252, 344, 48, C.mapSoft, { stroke: C.map, radius: 6 }),
    text('gs-map-input-k', 200, 262, 312, 14, 'EXTERNAL INPUT · NOT A GRAPH VARIABLE', 8, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, align: 'center', letterSpacing: .8 }),
    text('gs-map-input-v', 200, 279, 312, 16, tex`\mathcal M^{(r)}\;\text{fixed during this solve}`, 10, { color: C.mapDeep, fontWeight: 700, align: 'center' })
  ]
  const poseY = 506
  const poseXs = [146, 246, 346, 446, 546]

  // Draw edges before nodes so the graph remains legible.
  graph.push(line('gs-prior-edge', 122, poseY, poseXs[0] - 15, poseY, C.pose, 2))
  poseXs.slice(0, -1).forEach((x, index) => {
    const factorX = 0.5 * (x + poseXs[index + 1])
    graph.push(line(`gs-odo-edge-a-${index}`, x + 15, poseY, factorX - 6, poseY, C.soft, 2, { opacity: .8 }))
    graph.push(line(`gs-odo-edge-b-${index}`, factorX + 6, poseY, poseXs[index + 1] - 15, poseY, C.soft, 2, { opacity: .8 }))
  })
  graph.push(line('gs-loop-edge-a', poseXs[0], poseY - 15, 340, 338, C.danger, 2, { opacity: .7 }))
  graph.push(line('gs-loop-edge-b', 352, 338, poseXs[4], poseY - 15, C.danger, 2, { opacity: .7 }))
  ;[poseXs[1], poseXs[2], poseXs[3]].forEach((x, index) => {
    graph.push(line(`gs-smooth-edge-${index}`, 346, 398, x, poseY - 15, C.known, 1.5, { opacity: .55 }))
  })
  ;[poseXs[1], poseXs[3]].forEach((x, index) => {
    graph.push(line(`gs-reg-edge-${index}`, x, 450, x, poseY - 15, C.map, 2))
  })
  graph.push(line('gs-los-edge', poseXs[2], poseY + 15, poseXs[2], 558, C.measurement, 2))

  graph.push(shape('gs-prior-factor', 110, poseY - 6, 12, 12, C.poseSoft, { stroke: C.pose, strokeWidth: 2, radius: 0 }))
  graph.push(text('gs-prior-label', 96, poseY - 27, 50, 16, 'prior', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, align: 'center' }))
  poseXs.slice(0, -1).forEach((x, index) => {
    const factorX = 0.5 * (x + poseXs[index + 1])
    graph.push(shape(`gs-odo-factor-${index}`, factorX - 6, poseY - 6, 12, 12, C.paper, { stroke: C.soft, strokeWidth: 2, radius: 0 }))
  })
  poseXs.forEach((x, index) => {
    graph.push(shape(`gs-pose-${index}`, x - 15, poseY - 15, 30, 30, index === 0 ? C.poseSoft : C.paper, { shape: 'ellipse', stroke: C.pose, strokeWidth: 2 }))
    graph.push(text(`gs-pose-label-${index}`, x - 15, poseY - 8, 30, 16, tex`p_${index}`, 10, { color: C.poseDeep, fontWeight: 700, align: 'center' }))
  })
  graph.push(shape('gs-loop-factor', 340, 332, 12, 12, C.paper, { stroke: C.danger, strokeWidth: 2, radius: 0 }))
  graph.push(text('gs-loop-label', 286, 307, 120, 16, 'verified revisit', 9, { color: C.danger, fontFamily: MONO, fontWeight: 700, align: 'center' }))
  graph.push(shape('gs-smooth-factor', 340, 392, 12, 12, C.knownSoft, { stroke: C.known, strokeWidth: 2, radius: 0 }))
  graph.push(text('gs-smooth-label', 286, 407, 120, 16, 'smoothness × K−2', 9, { color: C.knownDeep, fontFamily: MONO, fontWeight: 700, align: 'center' }))
  ;[poseXs[1], poseXs[3]].forEach((x, index) => {
    graph.push(shape(`gs-reg-factor-${index}`, x - 6, 438, 12, 12, C.mapSoft, { stroke: C.map, strokeWidth: 2, radius: 0 }))
  })
  graph.push(text('gs-reg-label', 276, 452, 140, 18, tex`\mathrm{registration}\times|\mathcal R_r|`, 9, { color: C.mapDeep, fontWeight: 700, align: 'center' }))
  graph.push(shape('gs-los-factor', poseXs[2] - 6, 558, 12, 12, C.measurementSoft, { stroke: C.measurement, strokeWidth: 2, radius: 0 }))
  graph.push(text('gs-los-label', 365, 555, 210, 18, 'accepted geometric LoS fix', 9, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700 }))
  graph.push(text('gs-graph-legend', 112, 590, 488, 16, '○ position variable · □ factor · representative factors shown', 9, { color: C.faint, fontFamily: MONO, align: 'center' }))
  graph.push(text('gs-graph-legend-2', 112, 607, 488, 14, 'map and raw MPCs stay outside the linear graph solve', 9, { color: C.faint, fontFamily: MONO, align: 'center' }))

  return regular(
    's-radio-graphslam-equations', '04 · KNOWN BS POSE, UNKNOWN UE POSE AND MAP',
    'Implemented radio GraphSLAM: a pose graph, not a VA-state graph',
    'A known-start prior and smoothness prior join derived odometry, registration, gated-LoS, and revisit factors; raw MPC labels never enter the batch graph.',
    'Show the implemented estimator rather than the former oracle-associated landmark toy. The unknowns are 3-D UE positions only. For a fixed NDT map, all graph factors are linear and one normal-equation solve gives the MAP trajectory. The map is rebuilt, scans are re-registered, and LoS is re-detected between solves. No per-return provenance, BS-identity, wall, bounce, or truth-association label is an estimator input.',
    [
      ...graph,
      card('gs-state-card', 640, 202, 544, 78, C.poseSoft, { stroke: C.pose, radius: 8 }),
      text('gs-state-k', 666, 218, 190, 16, 'ONLY GRAPH VARIABLES', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.1 }),
      text('gs-state-eq', 666, 244, 492, 22, texBlock`\mathbf p=\{\mathbf p_0,\ldots,\mathbf p_{K-1}\},\qquad \mathbf p_k\in\mathbb R^3`, 17, { fontWeight: 700, align: 'center' }),

      card('gs-cost-card', 640, 296, 544, 218, C.paper, { stroke: C.line, radius: 8 }),
      text('gs-cost-k', 666, 314, 330, 16, 'EXACT LINEAR POSE ESTIMATE · NDT MAP FIXED', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .9 }),
      text('gs-cost-eq', 660, 340, 504, 134,
        texBlock`\begin{aligned}
          \mathbf p^*=\arg\min_{\mathbf p}\;&\|\mathbf r_0\|_{\Omega_0}^2
          +\sum_k\|\mathbf r_k^{\mathrm{odo}}\|_{\Omega_k}^2
          +\sum_k\|\mathbf r_k^{\mathrm{sm}}\|_{\Omega_s}^2\\[-.15em]
          &+\sum_{k\in\mathcal R_r}\|\mathbf p_k-\hat{\mathbf p}_k^{\mathrm{reg},(r)}\|_{\Lambda_k}^2\\[-.15em]
          &+\sum_{k\in\mathcal L_r}\|\mathbf p_k-\mathbf z_k^{\mathrm{LoS}}\|_{\Gamma_k}^2
          +\sum_{(i,j)\in\mathcal C}\|(\mathbf p_j-\mathbf p_i)-\mathbf d_{ij}\|_{\Pi_{ij}}^2
        \end{aligned}`,
        11, { fontWeight: 700, align: 'center', lineHeight: 1.15 }),
      text('gs-residual-defs', 666, 480, 492, 22, `${tex`\mathbf r_k^{\mathrm{odo}}=(\mathbf p_k-\mathbf p_{k-1})-\mathbf d_k`} · ${tex`\mathbf r_k^{\mathrm{sm}}=\mathbf p_{k-1}-2\mathbf p_k+\mathbf p_{k+1}`}`, 8, { color: C.soft, fontFamily: SANS, fontWeight: 700, align: 'center' }),

      card('gs-boundary-card', 640, 530, 544, 92, C.measurementSoft, { stroke: C.measurement, radius: 8 }),
      text('gs-boundary-k', 666, 548, 242, 16, 'LOS IS GATED, NOT GIVEN', 9, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.05 }),
      text('gs-boundary-v', 666, 574, 492, 34, 'Every MPC arrives unlabeled. At most one candidate may pass range/AoA, shortest-path, and innovation gates as LoS. Other geometry-valid returns form the registration clouds.', 11, { color: C.measurementDeep, fontFamily: SANS, lineHeight: 1.35 }),
      text('gs-source', 96, 650, 1088, 17, `Between solves: rebuild ${tex`\mathcal M^{(r)}`} · re-invert scans · re-register · re-detect LoS · solve again.`, 9, { color: C.faint, fontFamily: MONO, align: 'center' })
    ], { accent: C.pose, titleSize: 31, transition: 'none' }
  )
}

function graphIterationSlide(ctx) {
  const { regular, text, card, C, SANS, MONO, tex, texBlock } = ctx
  const steps = [
    ['01', 'INITIALIZE + SOLVE ONCE', `hybrid filter → ${tex`\mathbf p_{\mathrm{FE}},\mathbf d^{\mathrm{odo}}`}<br>verify revisit → ${tex`\mathbf d^{\mathrm{loop}}`}<br>initial graph solve → ${tex`\mathbf p^{(0)}`}`],
    ['02', 'BUILD AT CURRENT POSE', `re-invert unlabeled 5-D MPCs<br>build NDT + leave-out maps`],
    ['03', 'REFRESH TWO FACTORS', `NDT-register every usable scan<br>re-detect candidate LoS fixes`],
    ['04', 'SOLVE + REPEAT', `solve the exact linear pose graph<br>${tex`\mathbf p^{(r+1)}`} · five outer passes`]
  ]
  const elements = []
  steps.forEach((step, index) => {
    const x = 96 + index * 276
    const accent = [C.faint, C.measurement, C.pose, C.map][index]
    const fill = [C.paper, C.measurementSoft, C.poseSoft, C.mapSoft][index]
    elements.push(card(`gs-iter-card-${index}`, x, 222, 244, 146, fill, { stroke: accent, strokeWidth: 2, radius: 8 }))
    elements.push(text(`gs-iter-num-${index}`, x + 20, 242, 36, 20, step[0], 11, { color: accent, fontFamily: MONO, fontWeight: 700 }))
    elements.push(text(`gs-iter-head-${index}`, x + 58, 241, 166, 20, step[1], 10, { color: accent, fontFamily: MONO, fontWeight: 700, letterSpacing: .8 }))
    elements.push(text(`gs-iter-copy-${index}`, x + 20, 286, 204, 58, step[2], 12, { color: C.soft, fontFamily: SANS, fontWeight: 700, align: 'center', lineHeight: 1.45 }))
    if (index < steps.length - 1) elements.push(text(`gs-iter-arrow-${index}`, x + 246, 278, 28, 32, '→', 22, { color: C.faint, fontFamily: SANS, fontWeight: 700, align: 'center' }))
  })

  elements.push(card('gs-iter-eq-card', 96, 394, 1088, 92, C.paper, { stroke: C.line, radius: 8 }))
  elements.push(text('gs-iter-eq-k', 120, 412, 360, 16, 'FIXED MOTION/REVISIT · REFRESH REGISTRATION/LOS', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: .85 }))
  elements.push(text('gs-iter-eq-v', 120, 438, 1040, 34,
    texBlock`\{\mathbf d^{\mathrm{odo}},\mathbf d^{\mathrm{loop}}\}\ \mathrm{fixed};\qquad (\mathbf p^{(r)},Z)\longrightarrow\{\mathcal M^{(r)},\hat{\mathbf p}^{\mathrm{reg},(r)},\mathbf z^{\mathrm{LoS},(r)}\}\longrightarrow\mathbf p^{(r+1)}`,
    15, { fontWeight: 700, align: 'center' }))

  elements.push(card('gs-standard-card', 96, 512, 526, 110, C.poseSoft, { stroke: C.pose, radius: 8 }))
  elements.push(text('gs-standard-k', 120, 530, 190, 16, 'GRAPH_SLAM', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1 }))
  elements.push(text('gs-standard-v', 120, 558, 478, 48, 'Bounce-point clouds drive the rolling/global registration front end and the final voxel map.', 13, { color: C.poseDeep, fontFamily: SANS, lineHeight: 1.4 }))
  elements.push(card('gs-va-variant-card', 658, 512, 526, 110, C.mapSoft, { stroke: C.map, radius: 8 }))
  elements.push(text('gs-va-variant-k', 682, 530, 220, 16, 'GRAPH_SLAM_VA VARIANT', 9, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1 }))
  elements.push(text('gs-va-variant-v', 682, 558, 478, 48, 'Only rolling-submap ICP/odometry switches to VA clouds. Global/batch registration, revisit verification, and the delivered map stay bounce-point; LoS gating is unchanged.', 12, { color: C.mapDeep, fontFamily: SANS, lineHeight: 1.35 }))
  elements.push(text('gs-iter-source', 96, 650, 1088, 17, 'Gauss–Newton belongs inside NDT registration; the fixed-map pose graph itself is linear.', 9, { color: C.faint, fontFamily: MONO, align: 'center' }))

  return regular(
    's-radio-graphslam-iteration', '04 · KNOWN BS POSE, UNKNOWN UE POSE AND MAP',
    'Implemented GraphSLAM alternates map building and pose-graph solves',
    'The graph is exact for a fixed map; the nonlinear dependence is handled by re-inverting and re-registering scans between solves.',
    'Walk left to right. The hybrid front end creates front-end poses and odometry increments once; start-submap revisit verification also runs once. A first prior/smoothness/odometry/revisit graph solve produces p^(0) without registration or LoS factors. Each of five outer passes then re-inverts the unlabeled measurements, rebuilds the NDT and leave-window-out maps, refreshes only registration and LoS factors, and solves the exact linear pose graph. The VA variant changes only the local odometry channel, not the graph state or delivered map.',
    elements, { accent: C.pose, titleSize: 30, transition: 'none' }
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
    ctx.slides.push(graphEquationSlide(ctx))
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
    }
  ]
  return definitions.map(entry => ({
    ...entry,
    slideIndex: slides.findIndex(slide => slide.id === entry.slide),
    inline: true, layout: 'region', bounds: LIVE_BOUNDS,
    sandbox: 'allow-scripts', hideSource: true, readyMessage: true, unloadWhenHidden: true
  }))
}
