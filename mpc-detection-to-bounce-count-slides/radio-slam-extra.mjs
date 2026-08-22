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
    text('s1-scene-k', 118, 218, 420, 18, 'COMMON PHYSICAL SCENE · HIGHLIGHTED SCAN 4', 10, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.15 })
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
  elements.push(text('s1-route-legend', 126, 590, 600, 18, `${tex`\mathrm{LoS}`} · ${tex`\mathcal W_A`} reflection · ${tex`\mathcal W_B`} reflection · faint = unfolded VA ray`, 9, { color: C.faint, fontFamily: MONO, fontWeight: 700, align: 'center' }))

  elements.push(card('s1-geometry-card', 782, 202, 402, 142, C.mapSoft, { stroke: C.map, radius: 8 }))
  elements.push(text('s1-geometry-k', 806, 220, 354, 17, 'FIXED GEOMETRY', 9, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.1 }))
  elements.push(text('s1-geometry-eq', 806, 248, 354, 82, texBlock`\begin{aligned}
    \mathbf b&=[2,\,2]^{\mathsf T}\ \mathrm m\\
    \mathcal W_A&:y=7\ \mathrm m,\qquad \mathcal W_B:x=8.5\ \mathrm m\\
    \mathbf v_A&=[2,\,12]^{\mathsf T}\ \mathrm m,\qquad \mathbf v_B=[15,\,2]^{\mathsf T}\ \mathrm m
  \end{aligned}`, 15, { fontWeight: 700, align: 'center', lineHeight: 1.4 }))

  elements.push(card('s1-poses-card', 782, 362, 402, 132, C.paper, { stroke: C.line, radius: 8 }))
  elements.push(text('s1-poses-k', 806, 380, 354, 17, 'FIVE UE POSES', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.1 }))
  elements.push(text('s1-poses-eq', 806, 406, 354, 74, texBlock`\begin{aligned}
    \mathbf x_1&=(2.8,6.2,-18^\circ)&\mathbf x_2&=(3.9,5.6,-13^\circ)\\
    \mathbf x_3&=(5.0,5.0,-6^\circ)&\mathbf x_4&=(6.1,4.3,3^\circ)\\
    \mathbf x_5&=(7.2,3.5,12^\circ)
  \end{aligned}`, 12, { fontWeight: 700, align: 'center', lineHeight: 1.45 }))

  elements.push(card('s1-data-card', 782, 512, 402, 110, C.measurementSoft, { stroke: C.measurement, radius: 8 }))
  elements.push(text('s1-data-k', 806, 530, 354, 17, 'SAME MPC REALIZATION', 9, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.05 }))
  elements.push(text('s1-data-eq', 806, 554, 354, 24, texBlock`\mathbf z_{t\ell}=(\tau_{t\ell},\varphi_{t\ell},\psi_{t\ell},\alpha_{t\ell}),\quad \alpha_{t\ell}\in\mathbb C`, 13, { fontWeight: 700, align: 'center' }))
  elements.push(text('s1-data-v', 806, 586, 354, 24, `${tex`\sigma_L=0.08\,\mathrm m`} · ${tex`\sigma_{\angle}=1.4^\circ`} · clutter at ${tex`x_2,x_4`}`, 10, { color: C.soft, fontFamily: SANS, fontWeight: 700, align: 'center' }))

  elements.push(card('s1-state-card', 96, 638, 1088, 36, C.poseDeep, { stroke: C.poseDeep, radius: 6 }))
  elements.push(text('s1-state-v', 116, 647, 1048, 18, 'BP / PMBM: poses fixed → infer map + associations   ·   GraphSLAM: infer trajectory + map from odometry + the same radio tuples', 12, { color: C.paper, fontFamily: SANS, fontWeight: 700, align: 'center' }))

  return regular(
    's-radio-slam-s1', 'SHARED EXPERIMENT',
    'Setup S1: one scene, three inference views',
    'Only the unknown-state set changes; the BS, walls, trajectory, MPC ordering, noise, and clutter stay fixed.',
    'Use this slide as the controlled experiment definition. BP-SLAM and PMBM-SLAM clamp all five UE poses, while GraphSLAM releases them and adds odometry plus a first-pose prior. All three methods consume the same deterministic radio realization. The complex MPC gain alpha is radiometric evidence; phi remains AoA.',
    elements,
    { accent: C.map, titleSize: 34, transition: 'none' }
  )
}

function methodEquationSlide(kind, ctx) {
  const { regular, text, card, shape, line, C, SANS, MONO, tex, texBlock } = ctx
  if (kind === 'bp') {
    return regular(
      's-bp-slam-equations', '03 · KNOWN BS/UE POSE, UNKNOWN MAP',
      'BP-SLAM on S1: factorize, pass messages, marginalize',
      'The five S1 UE poses are clamped; BP estimates local map-feature, existence, and association beliefs without enumerating global events.',
      'Tie the generic BP machinery to setup S1. The known inputs are the BS, all five UE poses, and the common radio tuples. The latent variables are VA or reflector states, their existence variables, and route-to-MPC labels. The output is a family of marginal map and association beliefs, not one hard route.',
      [
        card('bp-post-card', 96, 202, 1088, 112, C.mapSoft, { stroke: C.map, strokeWidth: 2, radius: 8 }),
        text('bp-post-k', 122, 218, 260, 18, 'S1 CONDITIONAL FACTORIZATION', 10, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.2 }),
        text('bp-post-eq', 122, 248, 1036, 48,
          texBlock`p(M,E,A\mid Z,\mathbf X,\mathbf b)\propto p(M,E)\prod_{t=1}^{5}\prod_{\ell}f_{t\ell}^{\mathrm{radio}}(\mathbf m_{a_{t\ell}},e_{a_{t\ell}},a_{t\ell};\mathbf z_{t\ell},\mathbf x_t,\mathbf b)`,
          18, { fontWeight: 700, align: 'center', valign: 'middle', lineHeight: 1.25 }),

        card('bp-msg-card', 96, 336, 526, 232, C.paper, { stroke: C.line, radius: 8 }),
        text('bp-msg-k', 122, 356, 470, 18, 'SUM–PRODUCT MESSAGES', 10, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.2 }),
        text('bp-msg-eq-1', 122, 382, 470, 42,
          texBlock`\mu_{v\to f}(v)\propto\prod_{g\in\mathcal N(v)\setminus f}\mu_{g\to v}(v)`,
          16, { lineHeight: 1.2 }),
        text('bp-msg-eq-2', 122, 424, 470, 60,
          texBlock`\mu_{f\to v}(v)\propto\mathop{\sum\!\big/\!\int} f(\mathcal N(f))\!\prod_{u\in\mathcal N(f)\setminus v}\!\mu_{u\to f}(u)\,du`,
          14, { lineHeight: 1.2 }),
        text('bp-msg-eq-3', 122, 486, 470, 36,
          texBlock`b(v)\propto\prod_{f\in\mathcal N(v)}\mu_{f\to v}(v)`,
          16, { lineHeight: 1.2 }),
        text('bp-msg-note', 122, 536, 470, 22, 'Loops are iterated; beliefs approximate the desired marginals.', 12, { color: C.soft, fontFamily: SANS, fontWeight: 700 }),

        card('bp-radio-card', 658, 336, 526, 232, C.measurementSoft, { stroke: C.measurement, radius: 8 }),
        text('bp-radio-k', 684, 356, 470, 18, 'RADIO + ASSOCIATION FACTOR', 10, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.2 }),
        text('bp-radio-eq', 684, 390, 470, 102,
          texBlock`\begin{aligned}
            \mathbf z_{t\ell}&=(\tau_{t\ell},\varphi_{t\ell},\psi_{t\ell},\alpha_{t\ell}),\quad \alpha_{t\ell}\in\mathbb C\\[.45em]
            f_{t\ell}^{\mathrm{radio}}&\propto\mathcal N\!\left([c\tau,\varphi,\psi]^{\mathsf T};\mathbf h_{q}(\mathbf x_t,\mathbf m_j,\mathbf b),R_{t\ell}\right)p(\alpha_{t\ell}\mid q,\mathbf m_j)
          \end{aligned}`,
          15, { fontWeight: 700, lineHeight: 1.35 }),
        text('bp-radio-note', 684, 508, 470, 44, `${tex`a=0`} represents clutter or no landmark assignment; existence variables gate whether a map feature is present.`, 12, { color: C.soft, fontFamily: SANS, lineHeight: 1.35 }),

        card('bp-return-card', 96, 590, 1088, 46, C.mapDeep, { stroke: C.mapDeep, radius: 7 }),
        text('bp-return', 120, 602, 1040, 22, 'RETURN · marginal VA / reflector states, existence probabilities, and S1 route associations', 14, { color: C.paper, fontFamily: SANS, fontWeight: 700, align: 'center' }),
        text('bp-source', 96, 650, 1088, 17, 'Equation structure: multipath-based BP-SLAM factorization and generic sum–product rules.', 9, { color: C.faint, fontFamily: MONO, align: 'center' })
      ], { accent: C.map, titleSize: 34, transition: 'none' }
    )
  }

  return regular(
    's-pmbm-slam-equations', '03 · KNOWN BS/UE POSE, UNKNOWN MAP',
    'PMBM-SLAM on S1: Poisson births, Bernoulli landmarks, global hypotheses',
    'With the same five poses and MPC realization fixed, PMBM separates never-detected map features from detected Bernoulli features and their global histories.',
    'Explain the PMBM representation using setup S1. The known inputs are the same BS, fixed trajectory, and radio tuples used by BP. The Poisson component represents undetected map objects; each detected object is Bernoulli; the mixture index h carries a compatible global association history.',
    [
      card('pmbm-post-card', 96, 202, 1088, 112, C.measurementSoft, { stroke: C.measurement, strokeWidth: 2, radius: 8 }),
      text('pmbm-post-k', 122, 218, 290, 18, 'POISSON MULTI-BERNOULLI MIXTURE', 10, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.2 }),
      text('pmbm-post-eq', 122, 246, 1036, 52,
        texBlock`f(M\mid Z,\mathbf x)=\sum_{h\in\mathcal H}w^h\!\left[f_{\mathrm P}^{u}(M^u;\lambda^u)\prod_{i=1}^{n_h}f_{\mathrm B}(M^i;r_i^h,p_i^h)\right]`,
        18, { fontWeight: 700, align: 'center', valign: 'middle' }),

      card('pmbm-poisson-card', 96, 338, 344, 224, C.paper, { stroke: C.line, radius: 8 }),
      text('pmbm-poisson-k', 120, 358, 296, 18, 'UNDETECTED · POISSON', 10, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.1 }),
      text('pmbm-poisson-eq', 120, 400, 296, 62, texBlock`f_{\mathrm P}(X)=e^{-\Lambda}\prod_{m\in X}\lambda^u(m)`, 20, { fontWeight: 700, align: 'center' }),
      text('pmbm-poisson-v', 120, 480, 296, 58, `Intensity ${tex`\lambda^u`} carries map features that may exist but have not yet produced a confirmed MPC track.`, 13, { color: C.soft, fontFamily: SANS, lineHeight: 1.4 }),

      card('pmbm-bern-card', 468, 338, 344, 224, C.paper, { stroke: C.line, radius: 8 }),
      text('pmbm-bern-k', 492, 358, 296, 18, 'DETECTED · BERNOULLI', 10, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.1 }),
      text('pmbm-bern-eq', 492, 394, 296, 82, texBlock`\begin{aligned}f_{\mathrm B}(\varnothing)&=1-r\\f_{\mathrm B}(\{m\})&=r\,p(m)\end{aligned}`, 20, { fontWeight: 700, align: 'center', lineHeight: 1.5 }),
      text('pmbm-bern-v', 492, 490, 296, 48, `${tex`r`} is landmark-existence probability; ${tex`p(m)`} is its conditional spatial density.`, 13, { color: C.soft, fontFamily: SANS, lineHeight: 1.4 }),

      card('pmbm-hyp-card', 840, 338, 344, 224, C.paper, { stroke: C.line, radius: 8 }),
      text('pmbm-hyp-k', 864, 358, 296, 18, 'GLOBAL HYPOTHESIS UPDATE', 10, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.05 }),
      text('pmbm-hyp-eq', 864, 396, 296, 76, texBlock`\begin{aligned}w^{h,\vartheta}&\propto w^h\prod_i\eta_i^{h,\vartheta(i)}\\\sum_{h,\vartheta}w^{h,\vartheta}&=1\end{aligned}`, 19, { fontWeight: 700, align: 'center', lineHeight: 1.45 }),
      text('pmbm-hyp-v', 864, 490, 296, 48, 'Murty, Gibbs, or gating/pruning retains only the most relevant compatible stories.', 13, { color: C.soft, fontFamily: SANS, lineHeight: 1.4 }),

      card('pmbm-return-card', 96, 590, 1088, 46, C.measurementDeep, { stroke: C.measurementDeep, radius: 7 }),
      text('pmbm-return', 120, 602, 1040, 22, 'RETURN · weighted global association histories + Bernoulli map components + undetected PPP intensity', 14, { color: C.paper, fontFamily: SANS, fontWeight: 700, align: 'center' }),
      text('pmbm-source', 96, 650, 1088, 17, 'The following live slide uses the identical S1 scan as BP and isolates hypothesis ranking/pruning—not a complete PMBM recursion.', 9, { color: C.faint, fontFamily: MONO, align: 'center' })
    ], { accent: C.measurement, titleSize: 32, transition: 'none' }
  )
}

function associationFallback(kind, ctx) {
  const { text, card, shape, line, C, MONO, SANS, LIVE_BOUNDS, tex } = ctx
  const x0 = LIVE_BOUNDS.x, y0 = LIVE_BOUNDS.y, w = LIVE_BOUNDS.width, h = LIVE_BOUNDS.height
  const stageX = x0 + 14, stageY = y0 + 14, stageW = 744, stageH = h - 28
  const railX = stageX + stageW + 12, railW = w - stageW - 40
  const accent = kind === 'bp' ? C.map : C.measurement
  const deep = kind === 'bp' ? C.mapDeep : C.measurementDeep
  const soft = kind === 'bp' ? C.mapSoft : C.measurementSoft
  const tracks = [[stageX + 190, stageY + 120], [stageX + 355, stageY + 220], [stageX + 520, stageY + 120]]
  const measurements = [[stageX + 255, stageY + 150], [stageX + 360, stageY + 130], [stageX + 450, stageY + 205], [stageX + 625, stageY + 250]]
  const elements = [
    card(`${kind}-fallback-bg`, x0, y0, w, h, '#F8FAFB', { stroke: C.line, radius: 0 }),
    card(`${kind}-fallback-stage`, stageX, stageY, stageW, stageH, C.paper, { stroke: C.line, radius: 6 }),
    card(`${kind}-fallback-rail`, railX, stageY, railW, stageH, C.paper, { stroke: C.line, radius: 6 }),
    text(`${kind}-fallback-title`, stageX + 22, stageY + 18, stageW - 44, 18, kind === 'bp' ? 'S1 SCAN 4 · LOOPY ROUTE ASSOCIATION' : 'S1 SCAN 4 · COMPATIBLE GLOBAL ASSIGNMENTS', 10, { color: deep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.2 })
  ]
  tracks.forEach((point, index) => {
    elements.push(shape(`${kind}-track-${index}`, point[0] - 16, point[1] - 16, 32, 32, soft, { shape: 'ellipse', stroke: accent, strokeWidth: 2 }))
    elements.push(text(`${kind}-track-label-${index}`, point[0] - 23, point[1] - 7, 46, 15, index === 0 ? tex`H_{\mathrm{LoS}}` : tex`H_${index === 1 ? 'A' : 'B'}`, 8, { color: deep, fontWeight: 700, align: 'center' }))
  })
  measurements.forEach((point, index) => {
    elements.push(line(`${kind}-zx-a-${index}`, point[0] - 6, point[1] - 6, point[0] + 6, point[1] + 6, C.ink, 2))
    elements.push(line(`${kind}-zx-b-${index}`, point[0] - 6, point[1] + 6, point[0] + 6, point[1] - 6, C.ink, 2))
    elements.push(text(`${kind}-z-label-${index}`, point[0] + 8, point[1] - 10, 34, 14, tex`z_{${index + 1}}`, 8, { color: C.soft, fontWeight: 700 }))
  })
  ;[[0,0],[0,1],[1,0],[1,1],[1,2],[2,1],[2,2]].forEach((edge, index) => {
    const a = tracks[edge[0]], b = measurements[edge[1]]
    elements.push(line(`${kind}-edge-${index}`, a[0], a[1], b[0], b[1], accent, kind === 'bp' ? 2 + (index % 3) : 2, { opacity: kind === 'bp' ? .48 : .24 }))
  })
  if (kind === 'bp') {
    elements.push(text('bp-fallback-loop', stageX + 120, stageY + 314, stageW - 240, 44, `${tex`\mu_{\mathrm{route}\to\mathrm{MPC}}\rightleftarrows\nu_{\mathrm{MPC}\to\mathrm{route}}`}<br>iterate until S1 association marginals settle`, 18, { color: C.mapDeep, fontWeight: 700, align: 'center', lineHeight: 1.4 }))
  } else {
    ;[
      [tex`h_1`, `${tex`H_A\leftrightarrow z_2`} · ${tex`H_B\leftrightarrow z_3`}`, '0.46'],
      [tex`h_2`, `${tex`H_A\leftrightarrow z_4`} · ${tex`H_B\leftrightarrow z_3`}`, '0.31'],
      [tex`h_3`, `${tex`H_A`} missed · ${tex`H_B\leftrightarrow z_4`}`, '0.14'],
      ['…', 'lower-weight stories', '0.09']
    ].forEach((row, index) => {
      const y = stageY + 292 + index * 31
      elements.push(card(`pmbm-row-${index}`, stageX + 94, y, stageW - 188, 25, index < 2 ? C.measurementSoft : '#FBFCFD', { stroke: index < 2 ? C.measurement : C.line, radius: 4 }))
      elements.push(text(`pmbm-row-h-${index}`, stageX + 108, y + 6, 44, 13, row[0], 9, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700 }))
      elements.push(text(`pmbm-row-v-${index}`, stageX + 162, y + 5, 350, 14, row[1], 9, { color: C.soft, fontFamily: SANS, fontWeight: 700 }))
      elements.push(text(`pmbm-row-w-${index}`, stageX + stageW - 150, y + 5, 42, 14, row[2], 9, { color: C.ink, fontFamily: MONO, fontWeight: 700, align: 'right' }))
    })
  }
  elements.push(text(`${kind}-rail-k`, railX + 18, stageY + 18, railW - 36, 18, `SHARED S1 · ${kind === 'bp' ? 'MESSAGE CONTROLS' : 'PRUNING'}`, 9, { color: deep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.1 }))
  ;(kind === 'bp' ? ['one sweep', 'run to fixed point', 'move S1 MPC'] : [`keep top ${tex`k`}`, 'MAP only', 'renormalize retained mass']).forEach((label, index) => {
    const y = stageY + 64 + index * 76
    elements.push(card(`${kind}-control-${index}`, railX + 18, y, railW - 36, 52, index === 0 ? soft : '#FBFCFD', { stroke: index === 0 ? accent : C.line, radius: 5 }))
    elements.push(text(`${kind}-control-v-${index}`, railX + 30, y + 16, railW - 60, 20, label, 12, { color: index === 0 ? deep : C.soft, fontFamily: SANS, fontWeight: 700, align: 'center' }))
  })
  elements.push(card(`${kind}-result-card`, railX + 18, stageY + 304, railW - 36, 106, soft, { stroke: accent, radius: 6 }))
  elements.push(text(`${kind}-result-k`, railX + 32, stageY + 322, railW - 64, 16, 'LIVE RESULT', 9, { color: deep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1 }))
  elements.push(text(`${kind}-result-v`, railX + 32, stageY + 352, railW - 64, 42, kind === 'bp' ? 'Approximate association marginals' : 'Ranked joint-event weights', 14, { fontWeight: 700, align: 'center', valign: 'middle' }))
  return elements
}

function graphEquationSlide(ctx) {
  const { regular, text, card, shape, line, C, SANS, MONO, tex, texBlock } = ctx
  const graph = [
    card('gs-graph-card', 96, 202, 406, 420, C.paper, { stroke: C.line, radius: 8 }),
    text('gs-graph-k', 120, 222, 358, 18, 'RADIO FACTOR GRAPH', 10, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.2 })
  ]
  const poseYs = [286, 350, 414, 478, 542]
  poseYs.forEach((y, index) => {
    if (index < poseYs.length - 1) {
      graph.push(line(`gs-motion-edge-${index}`, 206, y + 14, 206, poseYs[index + 1] - 14, C.soft, 2, { opacity: .65 }))
      graph.push(shape(`gs-motion-factor-${index}`, 200, 0.5 * (y + poseYs[index + 1]) - 6, 12, 12, C.paper, { stroke: C.soft, strokeWidth: 2, radius: 0 }))
    }
    graph.push(shape(`gs-pose-${index}`, 192, y - 14, 28, 28, index === 0 ? C.poseSoft : C.paper, { shape: 'ellipse', stroke: C.pose, strokeWidth: 2 }))
    graph.push(text(`gs-pose-label-${index}`, 192, y - 6, 28, 14, tex`x_${index + 1}`, 8, { color: C.poseDeep, fontWeight: 700, align: 'center' }))
  })
  ;[[390,330,tex`v_A`,C.map,C.mapSoft],[390,492,tex`v_B`,C.measurement,C.measurementSoft]].forEach((item, index) => {
    graph.push(shape(`gs-map-${index}`, item[0] - 15, item[1] - 15, 30, 30, item[4], { shape: 'ellipse', stroke: item[3], strokeWidth: 2 }))
    graph.push(text(`gs-map-label-${index}`, item[0] - 15, item[1] - 7, 30, 14, item[2], 8, { color: index === 0 ? C.mapDeep : C.measurementDeep, fontFamily: MONO, fontWeight: 700, align: 'center' }))
  })
  poseYs.forEach((y, index) => {
    const target = index % 2 === 0 ? [390,330] : [390,492]
    graph.push(line(`gs-radio-edge-${index}`, 220, y, target[0] - 16, target[1], index % 2 === 0 ? C.map : C.measurement, 2, { opacity: .42 }))
    graph.push(shape(`gs-radio-factor-${index}`, 292, y - 5, 10, 10, C.paper, { stroke: index % 2 === 0 ? C.map : C.measurement, strokeWidth: 2, radius: 0 }))
  })
  graph.push(shape('gs-prior-factor', 130, poseYs[0] - 6, 12, 12, C.poseSoft, { stroke: C.pose, strokeWidth: 2, radius: 0 }))
  graph.push(line('gs-prior-edge', 142, poseYs[0], 192, poseYs[0], C.pose, 2))
  graph.push(text('gs-graph-legend', 120, 590, 358, 18, '○ variables · □ factors · BS is fixed', 9, { color: C.faint, fontFamily: MONO, align: 'center' }))

  return regular(
    's-radio-graphslam-equations', '04 · KNOWN BS POSE, UNKNOWN UE POSE AND MAP',
    'Radio GraphSLAM on S1: optimize trajectory and VA map jointly',
    'The physical experiment stays fixed; only the five UE poses are released and connected by odometry, a first-pose prior, and the same radio tuples.',
    'Bridge the S1 geometric experiment to nonlinear least squares. The continuous state contains the five UE poses and two virtual anchors. Association A and bounce/order Q are discrete and fixed inside the illustrated solve; BP or PMBM can propose them. Alpha remains calibrated radiometric evidence for ranking hypotheses, while the Gauss-Newton geometry residual uses delay and endpoint bearings.',
    [
      ...graph,
      card('gs-state-card', 532, 202, 652, 78, C.poseSoft, { stroke: C.pose, radius: 8 }),
      text('gs-state-k', 558, 218, 170, 16, 'S1 UNKNOWNS', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.1 }),
      text('gs-state-eq', 558, 244, 600, 22, texBlock`\Theta=\{\mathbf x_{1:5},\mathbf v_A,\mathbf v_B\}\;\cdot\;A=\{a_{t\ell}\}\;\cdot\;Q=\{q_{t\ell}\}`, 18, { fontWeight: 700, align: 'center' }),

      card('gs-cost-card', 532, 296, 652, 132, C.paper, { stroke: C.line, radius: 8 }),
      text('gs-cost-k', 558, 314, 180, 16, 'MAP / NONLINEAR LEAST SQUARES', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.05 }),
      text('gs-cost-eq', 558, 344, 600, 70,
        texBlock`\begin{aligned}
          \Theta^*(A,Q)=\arg\min_{\Theta}\;&\|\mathbf r_0\|_{\Omega_0}^2+\sum_t\|\mathbf r_t^{\mathrm{mot}}\|_{\Omega_t}^2\\[-.1em]
          &+\sum_{t,\ell}\rho\!\left(\|\mathbf r_{t\ell}^{\mathrm{radio}}\|_{\Omega_{t\ell}}^2\right)
        \end{aligned}`,
        17, { fontWeight: 700, align: 'center', lineHeight: 1.45 }),

      card('gs-radio-card', 532, 444, 652, 178, C.measurementSoft, { stroke: C.measurement, radius: 8 }),
      text('gs-radio-k', 558, 462, 210, 16, 'ONE-BOUNCE VA RADIO FACTOR', 9, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.05 }),
      text('gs-radio-residual', 558, 490, 600, 38, texBlock`\mathbf r_{t\ell}^{\mathrm{radio}}=[c\tau,\,\mathrm{wrap}(\varphi),\,\mathrm{wrap}(\psi)]^{\mathsf T}-\mathbf h_{q_{t\ell}}(\mathbf x_t,\mathbf v_{a_{t\ell}},\mathbf b)`, 16, { fontWeight: 700, align: 'center' }),
      text('gs-radio-model', 558, 536, 600, 70,
        texBlock`\begin{aligned}
          \widehat L&=\|\mathbf p_t-\mathbf v_j\|\\
          P_{tj}&=\mathrm{line}(\mathbf p_t,\mathbf v_j)\cap\mathrm{bisector}(\mathbf p_{\mathrm{BS}},\mathbf v_j)\\
          \widehat\varphi&=\mathrm{wrap}(\mathrm{bearing}(P_{tj}-\mathbf p_t)-\theta_t),\quad
          \widehat\psi=\mathrm{wrap}(\mathrm{bearing}(P_{tj}-\mathbf p_{\mathrm{BS}})-\theta_{\mathrm{BS}})
        \end{aligned}`,
        14, { lineHeight: 1.45, align: 'center' }),
      text('gs-source', 96, 650, 1088, 17, `S1 geometry uses ${tex`[c\tau,\varphi,\psi]`}; calibrated ${tex`\alpha`} grades hypotheses but is not added as a Gauss–Newton geometry coordinate.`, 9, { color: C.faint, fontFamily: MONO, align: 'center' })
    ], { accent: C.pose, titleSize: 31, transition: 'none' }
  )
}

function graphFallback(ctx) {
  const { text, card, shape, line, C, MONO, SANS, LIVE_BOUNDS, tex } = ctx
  const x0 = LIVE_BOUNDS.x, y0 = LIVE_BOUNDS.y, w = LIVE_BOUNDS.width, h = LIVE_BOUNDS.height
  const stageX = x0 + 14, stageY = y0 + 14, stageW = 766, stageH = h - 28
  const railX = stageX + stageW + 12, railW = w - stageW - 40
  const poses = [[stageX + 160,stageY + 320],[stageX + 270,stageY + 285],[stageX + 382,stageY + 235],[stageX + 490,stageY + 178],[stageX + 585,stageY + 112]]
  const va1 = [stageX + 664,stageY + 82], va2 = [stageX + 100,stageY + 72], bs = [stageX + 118,stageY + 350]
  const elements = [
    card('graph-fallback-bg', x0, y0, w, h, '#F8FAFB', { stroke: C.line, radius: 0 }),
    card('graph-fallback-stage', stageX, stageY, stageW, stageH, C.paper, { stroke: C.line, radius: 6 }),
    card('graph-fallback-rail', railX, stageY, railW, stageH, C.paper, { stroke: C.line, radius: 6 }),
    text('graph-fallback-k', stageX + 22, stageY + 18, stageW - 44, 18, 'SETUP S1 · TRAJECTORY + VA MAP + MPC ROUTES', 10, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.15 })
  ]
  poses.slice(0,-1).forEach((point,index) => elements.push(line(`graph-path-${index}`, point[0], point[1], poses[index+1][0], poses[index+1][1], C.pose, 4)))
  poses.forEach((point,index) => {
    elements.push(shape(`graph-pose-${index}`, point[0]-8, point[1]-8, 16, 16, C.paper, { shape:'ellipse', stroke:C.pose, strokeWidth:3 }))
    elements.push(text(`graph-pose-label-${index}`, point[0]+10, point[1]-12, 36, 16, tex`x_${index + 1}`, 8, { color:C.poseDeep, fontWeight:700 }))
  })
  elements.push(shape('graph-bs',bs[0]-7,bs[1]-7,14,14,C.ink,{radius:0}))
  elements.push(text('graph-bs-label',bs[0]-10,bs[1]+14,62,16,'known BS',8,{color:C.ink,fontFamily:MONO,fontWeight:700}))
  ;[[va1,tex`\mathrm{VA}_1`,C.map,C.mapSoft],[va2,tex`\mathrm{VA}_2`,C.measurement,C.measurementSoft]].forEach((item,index) => {
    elements.push(shape(`graph-va-${index}`,item[0][0]-12,item[0][1]-12,24,24,item[3],{shape:'ellipse',stroke:item[2],strokeWidth:3}))
    elements.push(text(`graph-va-label-${index}`,item[0][0]-24,item[0][1]-34,48,16,item[1],9,{color:index===0?C.mapDeep:C.measurementDeep,fontFamily:MONO,fontWeight:700,align:'center'}))
  })
  const active = poses[2], p1=[stageX+510,stageY+88], p2=[stageX+190,stageY+86]
  elements.push(line('graph-route-los',bs[0],bs[1],active[0],active[1],C.measurement,4))
  elements.push(line('graph-route1-a',bs[0],bs[1],p1[0],p1[1],C.map,4))
  elements.push(line('graph-route1-b',p1[0],p1[1],active[0],active[1],C.map,4))
  elements.push(line('graph-route2-a',bs[0],bs[1],p2[0],p2[1],C.measurement,4))
  elements.push(line('graph-route2-b',p2[0],p2[1],active[0],active[1],C.measurement,4))
  ;[p1,p2].forEach((p,index)=>elements.push(shape(`graph-reflection-${index}`,p[0]-5,p[1]-5,10,10,C.paper,{shape:'ellipse',stroke:index===0?C.map:C.measurement,strokeWidth:2})))
  elements.push(text('graph-rail-k',railX+18,stageY+18,railW-36,18,'S1 GAUSS–NEWTON CONTROLS',9,{color:C.poseDeep,fontFamily:MONO,fontWeight:700,letterSpacing:1.05}))
  ;['delay only ↔ delay + angles','correct ↔ wrong association','quadratic ↔ robust loss','one step ↔ optimize'].forEach((label,index)=>{
    const y=stageY+60+index*63
    elements.push(card(`graph-control-${index}`,railX+18,y,railW-36,45,index===3?C.poseSoft:'#FBFCFD',{stroke:index===3?C.pose:C.line,radius:5}))
    elements.push(text(`graph-control-v-${index}`,railX+27,y+13,railW-54,20,label,10,{color:index===3?C.poseDeep:C.soft,fontFamily:SANS,fontWeight:700,align:'center'}))
  })
  elements.push(card('graph-result-card',railX+18,stageY+326,railW-36,82,C.poseSoft,{stroke:C.pose,radius:6}))
  elements.push(text('graph-result-k',railX+30,stageY+342,railW-60,15,'JOINT RESULT',8,{color:C.poseDeep,fontFamily:MONO,fontWeight:700,letterSpacing:1}))
  elements.push(text('graph-result-v',railX+30,stageY+370,railW-60,26,'UE trajectory + VA map',13,{fontWeight:700,align:'center'}))
  return elements
}

function methodLiveSlide(kind, ctx) {
  const { regular, liveMount, C } = ctx
  if (kind === 'bp') {
    return regular(
      's-bp-slam-live', '03 · KNOWN BS/UE POSE, UNKNOWN MAP',
      'BP-SLAM live: S1 association marginals by message passing',
      'Select one of the five shared scans, perturb an MPC, step the messages, and inspect the converged route-association marginals.',
      'The embedded lab uses setup S1: the BS, walls, virtual anchors, five fixed UE poses, deterministic radio tuples, and clutter realization are shared with PMBM and GraphSLAM. It demonstrates the Williams–Lau association engine, not every continuous BP-SLAM state update.',
      [...associationFallback('bp', ctx), liveMount()], { accent: C.map, titleSize: 31, transition: 'none' }
    )
  }
  return regular(
    's-pmbm-slam-live', '03 · KNOWN BS/UE POSE, UNKNOWN MAP',
    'PMBM-SLAM live: rank and prune S1 global hypotheses',
    'Use the identical S1 scan as BP, vary top-k pruning, and see how retained hypothesis mass changes the route marginals.',
    'The embedded lab isolates the global-hypothesis layer that a PMBM update must manage. The geometry and measurements are identical to the BP view; only the inference representation changes. It is not a complete PMBM-SLAM recursion.',
    [...associationFallback('pmbm', ctx), liveMount()], { accent: C.measurement, titleSize: 30, transition: 'none' }
  )
}

function graphLiveSlide(ctx) {
  const { regular, liveMount, C } = ctx
  return regular(
    's-radio-graphslam-live', '04 · KNOWN BS POSE, UNKNOWN UE POSE AND MAP',
    'Radio GraphSLAM live: Gauss–Newton on the same S1 MPCs',
    'Release the five UE poses, retain the same physical scene and radio realization, then compare delay-only versus delay+AoA+AoD factors.',
    'Run the S1 optimizer. The continuous variables are the same five UE poses and two virtual anchors defined on the shared setup slide. Motion factors use relative odometry; LoS and one-bounce radio factors use the same deterministic MPC tuples as the BP and PMBM pages. Association and bounce order are fixed inside one solve.',
    [...graphFallback(ctx), liveMount()], { accent: C.pose, titleSize: 30, transition: 'none' }
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
    ctx.slides.push(graphLiveSlide(ctx))
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
      src: '../bp-vs-pmbm-slides/live/?demo=hypotheses&embed=region', source: '../bp-vs-pmbm-slides/live/?demo=hypotheses',
      title: 'Section 03 · PMBM-SLAM on shared setup S1'
    },
    {
      introSlide: 's-radio-graphslam-equations', slide: 's-radio-graphslam-live',
      src: './radio-graphslam-live/?embed=region', source: './radio-graphslam-live/',
      title: 'Section 04 · radio GraphSLAM on shared setup S1'
    }
  ]
  return definitions.map(entry => ({
    ...entry,
    slideIndex: slides.findIndex(slide => slide.id === entry.slide),
    inline: true, layout: 'region', bounds: LIVE_BOUNDS,
    sandbox: 'allow-scripts', hideSource: true, readyMessage: true, unloadWhenHidden: true
  }))
}
