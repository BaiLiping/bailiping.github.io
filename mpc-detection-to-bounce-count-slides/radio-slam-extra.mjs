function methodEquationSlide(kind, ctx) {
  const { regular, text, card, shape, line, C, SANS, MONO } = ctx
  if (kind === 'bp') {
    return regular(
      's-bp-slam-equations', '03 · KNOWN BS/UE POSE, UNKNOWN MAP',
      'BP-SLAM: factorize, pass messages, marginalize',
      'Belief propagation keeps pose, landmark, existence, and association uncertainty local instead of enumerating every global assignment.',
      'Introduce BP-SLAM as sum–product on the radio-SLAM factor graph. The symbols are deliberately generic: x is the UE state, m is a map landmark such as a VA or reflector, a is a data-association label, and ψ is the radio likelihood for delay and angles. Emphasize that the output is a family of marginal beliefs, not one hard route.',
      [
        card('bp-post-card', 96, 202, 1088, 112, C.mapSoft, { stroke: C.map, strokeWidth: 2, radius: 8 }),
        text('bp-post-k', 122, 218, 220, 18, 'POSTERIOR FACTORIZATION', 10, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.2 }),
        text('bp-post-eq', 122, 248, 1036, 48,
          '<i>p</i>(<b>x</b><sub>0:T</sub>, M, A | Z,U) ∝ <i>p</i>(<b>x</b><sub>0</sub>) ∏<sub>t=1</sub><sup>T</sup> <i>p</i>(<b>x</b><sub>t</sub>|<b>x</b><sub>t−1</sub>,<b>u</b><sub>t</sub>) ∏<sub>t,ℓ</sub> ψ<sub>tℓ</sub>(<b>x</b><sub>t</sub>,<b>m</b><sub>a<sub>tℓ</sub></sub>,a<sub>tℓ</sub>;<b>z</b><sub>tℓ</sub>)',
          17, { fontWeight: 700, align: 'center', valign: 'middle', lineHeight: 1.25 }),

        card('bp-msg-card', 96, 336, 526, 232, C.paper, { stroke: C.line, radius: 8 }),
        text('bp-msg-k', 122, 356, 470, 18, 'SUM–PRODUCT MESSAGES', 10, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.2 }),
        text('bp-msg-eq', 122, 390, 470, 124,
          'μ<sub>v→f</sub>(v) ∝ ∏<sub>g∈N(v)∖f</sub> μ<sub>g→v</sub>(v)<br><br>' +
          'μ<sub>f→v</sub>(v) ∝ ∑/∫ f(N(f)) ∏<sub>u∈N(f)∖v</sub> μ<sub>u→f</sub>(u) du<br><br>' +
          '<b>b</b>(v) ∝ ∏<sub>f∈N(v)</sub> μ<sub>f→v</sub>(v)',
          16, { lineHeight: 1.24 }),
        text('bp-msg-note', 122, 526, 470, 26, 'Loops are iterated; beliefs approximate the desired marginals.', 12, { color: C.soft, fontFamily: SANS, fontWeight: 700 }),

        card('bp-radio-card', 658, 336, 526, 232, C.measurementSoft, { stroke: C.measurement, radius: 8 }),
        text('bp-radio-k', 684, 356, 470, 18, 'RADIO + ASSOCIATION FACTOR', 10, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.2 }),
        text('bp-radio-eq', 684, 390, 470, 102,
          '<b>z</b><sub>tℓ</sub> = [cτ, φ, ψ]<sup>T</sup>, &nbsp; a<sub>tℓ</sub>∈{0,1,…,J}<br><br>' +
          'ψ<sub>tℓ</sub> ∝ 𝒩(<b>z</b><sub>tℓ</sub>; <b>h</b><sub>q</sub>(<b>x</b><sub>t</sub>,<b>m</b><sub>j</sub>,<b>x</b><sub>BS</sub>), R<sub>tℓ</sub>)',
          17, { fontWeight: 700, lineHeight: 1.35 }),
        text('bp-radio-note', 684, 508, 470, 44, 'a = 0 represents clutter or no landmark assignment; existence variables gate whether a map feature is present.', 12, { color: C.soft, fontFamily: SANS, lineHeight: 1.35 }),

        card('bp-return-card', 96, 590, 1088, 46, C.mapDeep, { stroke: C.mapDeep, radius: 7 }),
        text('bp-return', 120, 602, 1040, 22, 'RETURN · marginal UE trajectory, landmark states/existence, and association probabilities', 14, { color: C.paper, fontFamily: SANS, fontWeight: 700, align: 'center' }),
        text('bp-source', 96, 650, 1088, 17, 'Equation structure: multipath-based BP-SLAM factorization and generic sum–product rules.', 9, { color: C.faint, fontFamily: MONO, align: 'center' })
      ], { accent: C.map, titleSize: 34, transition: 'none' }
    )
  }

  return regular(
    's-pmbm-slam-equations', '03 · KNOWN BS/UE POSE, UNKNOWN MAP',
    'PMBM-SLAM: Poisson births, Bernoulli landmarks, global hypotheses',
    'The map posterior separates never-detected landmarks from detected Bernoulli landmarks and keeps competing association histories explicitly weighted.',
    'Explain the PMBM representation before discussing implementation details. The Poisson component represents undetected map objects; each detected object is Bernoulli; the mixture index h carries a compatible global association history. In a practical SLAM filter the UE state is coupled to these map hypotheses, often through particles, Gaussian approximations, or conditional updates.',
    [
      card('pmbm-post-card', 96, 202, 1088, 112, C.measurementSoft, { stroke: C.measurement, strokeWidth: 2, radius: 8 }),
      text('pmbm-post-k', 122, 218, 290, 18, 'POISSON MULTI-BERNOULLI MIXTURE', 10, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.2 }),
      text('pmbm-post-eq', 122, 246, 1036, 52,
        '<i>f</i>(M|Z,<b>x</b>) = ∑<sub>h∈H</sub> w<sup>h</sup> [ <i>f</i><sup>u</sup><sub>P</sub>(M<sup>u</sup>;λ<sup>u</sup>) ∏<sub>i=1</sub><sup>n<sub>h</sub></sup> <i>f</i><sub>B</sub>(M<sup>i</sup>;r<sup>h</sup><sub>i</sub>,p<sup>h</sup><sub>i</sub>) ]',
        18, { fontWeight: 700, align: 'center', valign: 'middle' }),

      card('pmbm-poisson-card', 96, 338, 344, 224, C.paper, { stroke: C.line, radius: 8 }),
      text('pmbm-poisson-k', 120, 358, 296, 18, 'UNDETECTED · POISSON', 10, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.1 }),
      text('pmbm-poisson-eq', 120, 400, 296, 62, '<i>f</i><sub>P</sub>(X)=e<sup>−Λ</sup> ∏<sub>m∈X</sub> λ<sup>u</sup>(m)', 20, { fontWeight: 700, align: 'center' }),
      text('pmbm-poisson-v', 120, 480, 296, 58, 'Intensity λᵘ carries map features that may exist but have not yet produced a confirmed MPC track.', 13, { color: C.soft, fontFamily: SANS, lineHeight: 1.4 }),

      card('pmbm-bern-card', 468, 338, 344, 224, C.paper, { stroke: C.line, radius: 8 }),
      text('pmbm-bern-k', 492, 358, 296, 18, 'DETECTED · BERNOULLI', 10, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.1 }),
      text('pmbm-bern-eq', 492, 394, 296, 82, '<i>f</i><sub>B</sub>(∅)=1−r<br><i>f</i><sub>B</sub>({m})=r p(m)', 20, { fontWeight: 700, align: 'center', lineHeight: 1.5 }),
      text('pmbm-bern-v', 492, 490, 296, 48, 'r is landmark-existence probability; p(m) is its conditional spatial density.', 13, { color: C.soft, fontFamily: SANS, lineHeight: 1.4 }),

      card('pmbm-hyp-card', 840, 338, 344, 224, C.paper, { stroke: C.line, radius: 8 }),
      text('pmbm-hyp-k', 864, 358, 296, 18, 'GLOBAL HYPOTHESIS UPDATE', 10, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.05 }),
      text('pmbm-hyp-eq', 864, 396, 296, 76, 'w<sup>h,θ</sup> ∝ w<sup>h</sup> ∏<sub>i</sub> η<sup>h,θ(i)</sup><sub>i</sub><br>∑<sub>h,θ</sub> w<sup>h,θ</sup>=1', 19, { fontWeight: 700, align: 'center', lineHeight: 1.45 }),
      text('pmbm-hyp-v', 864, 490, 296, 48, 'Murty, Gibbs, or gating/pruning retains only the most relevant compatible stories.', 13, { color: C.soft, fontFamily: SANS, lineHeight: 1.4 }),

      card('pmbm-return-card', 96, 590, 1088, 46, C.measurementDeep, { stroke: C.measurementDeep, radius: 7 }),
      text('pmbm-return', 120, 602, 1040, 22, 'RETURN · weighted global association histories + Bernoulli map components + undetected PPP intensity', 14, { color: C.paper, fontFamily: SANS, fontWeight: 700, align: 'center' }),
      text('pmbm-source', 96, 650, 1088, 17, 'The following live slide isolates hypothesis ranking and pruning; it is not a complete PMBM state update.', 9, { color: C.faint, fontFamily: MONO, align: 'center' })
    ], { accent: C.measurement, titleSize: 32, transition: 'none' }
  )
}

function associationFallback(kind, ctx) {
  const { text, card, shape, line, C, MONO, SANS, LIVE_BOUNDS } = ctx
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
    text(`${kind}-fallback-title`, stageX + 22, stageY + 18, stageW - 44, 18, kind === 'bp' ? 'LOOPY ASSOCIATION GRAPH' : 'COMPATIBLE GLOBAL ASSIGNMENTS', 10, { color: deep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.2 })
  ]
  tracks.forEach((point, index) => {
    elements.push(shape(`${kind}-track-${index}`, point[0] - 16, point[1] - 16, 32, 32, soft, { shape: 'ellipse', stroke: accent, strokeWidth: 2 }))
    elements.push(text(`${kind}-track-label-${index}`, point[0] - 18, point[1] - 7, 36, 15, `m${index + 1}`, 9, { color: deep, fontFamily: MONO, fontWeight: 700, align: 'center' }))
  })
  measurements.forEach((point, index) => {
    elements.push(line(`${kind}-zx-a-${index}`, point[0] - 6, point[1] - 6, point[0] + 6, point[1] + 6, C.ink, 2))
    elements.push(line(`${kind}-zx-b-${index}`, point[0] - 6, point[1] + 6, point[0] + 6, point[1] - 6, C.ink, 2))
    elements.push(text(`${kind}-z-label-${index}`, point[0] + 8, point[1] - 10, 34, 14, `z${index + 1}`, 8, { color: C.soft, fontFamily: MONO, fontWeight: 700 }))
  })
  ;[[0,0],[0,1],[1,0],[1,1],[1,2],[2,1],[2,2]].forEach((edge, index) => {
    const a = tracks[edge[0]], b = measurements[edge[1]]
    elements.push(line(`${kind}-edge-${index}`, a[0], a[1], b[0], b[1], accent, kind === 'bp' ? 2 + (index % 3) : 2, { opacity: kind === 'bp' ? .48 : .24 }))
  })
  if (kind === 'bp') {
    elements.push(text('bp-fallback-loop', stageX + 120, stageY + 314, stageW - 240, 44, 'μ track→MPC  ⇄  ν MPC→track<br>iterate until association marginals settle', 19, { color: C.mapDeep, fontWeight: 700, align: 'center', lineHeight: 1.4 }))
  } else {
    ;[
      ['h₁', 'm₁↔z₁ · m₂↔z₃', '0.46'],
      ['h₂', 'm₁↔z₂ · m₂↔z₁', '0.31'],
      ['h₃', 'm₁ missed · m₂↔z₃', '0.14'],
      ['…', 'lower-weight stories', '0.09']
    ].forEach((row, index) => {
      const y = stageY + 292 + index * 31
      elements.push(card(`pmbm-row-${index}`, stageX + 94, y, stageW - 188, 25, index < 2 ? C.measurementSoft : '#FBFCFD', { stroke: index < 2 ? C.measurement : C.line, radius: 4 }))
      elements.push(text(`pmbm-row-h-${index}`, stageX + 108, y + 6, 44, 13, row[0], 9, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700 }))
      elements.push(text(`pmbm-row-v-${index}`, stageX + 162, y + 5, 350, 14, row[1], 9, { color: C.soft, fontFamily: SANS, fontWeight: 700 }))
      elements.push(text(`pmbm-row-w-${index}`, stageX + stageW - 150, y + 5, 42, 14, row[2], 9, { color: C.ink, fontFamily: MONO, fontWeight: 700, align: 'right' }))
    })
  }
  elements.push(text(`${kind}-rail-k`, railX + 18, stageY + 18, railW - 36, 18, kind === 'bp' ? 'MESSAGE CONTROLS' : 'PRUNING CONTROLS', 9, { color: deep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.1 }))
  ;(kind === 'bp' ? ['one sweep', 'run to fixed point', 'move measurement'] : ['keep top k', 'MAP only', 'renormalize retained mass']).forEach((label, index) => {
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
  const { regular, text, card, shape, line, C, SANS, MONO } = ctx
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
    graph.push(text(`gs-pose-label-${index}`, 192, y - 6, 28, 14, `x${index}`, 8, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, align: 'center' }))
  })
  ;[[390,330,'m₁',C.map,C.mapSoft],[390,492,'m₂',C.measurement,C.measurementSoft]].forEach((item, index) => {
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
    'Radio GraphSLAM: optimize trajectory and virtual-anchor map jointly',
    'The graph is standard GraphSLAM; the observation factor is radio-specific and must predict delay, endpoint bearings, association, and bounce order.',
    'This is the bridge from the Section 4 geometric families to a practical nonlinear least-squares formulation. The continuous state contains all UE poses and map parameters. Association a and bounce/order q are discrete; they can be supplied by BP/PMBM, enumerated as hypotheses, or represented with max-mixture or switchable factors. The one-bounce VA equations are shown explicitly; higher-order routes use recursive unfolding and folding.',
    [
      ...graph,
      card('gs-state-card', 532, 202, 652, 78, C.poseSoft, { stroke: C.pose, radius: 8 }),
      text('gs-state-k', 558, 218, 130, 16, 'UNKNOWNS', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.1 }),
      text('gs-state-eq', 558, 244, 600, 22, 'Θ = {<b>x</b><sub>0:T</sub>, <b>m</b><sub>1:J</sub>, b<sub>0:T</sub>}  ·  A={a<sub>tℓ</sub>}  ·  Q={q<sub>tℓ</sub>}', 18, { fontWeight: 700, align: 'center' }),

      card('gs-cost-card', 532, 296, 652, 132, C.paper, { stroke: C.line, radius: 8 }),
      text('gs-cost-k', 558, 314, 180, 16, 'MAP / NONLINEAR LEAST SQUARES', 9, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.05 }),
      text('gs-cost-eq', 558, 344, 600, 70,
        'Θ*(A,Q)=arg min<sub>Θ</sub> ‖r<sub>0</sub>‖²<sub>Ω₀</sub> + ∑<sub>t</sub> ‖r<sup>mot</sup><sub>t</sub>‖²<sub>Ω<sub>t</sub></sub><br>' +
        '+ ∑<sub>t,ℓ</sub> ρ( ‖r<sup>radio</sup><sub>tℓ</sub>‖²<sub>Ω<sub>tℓ</sub></sub> ) + ∑<sub>j</sub> ‖r<sup>map</sup><sub>j</sub>‖²',
        17, { fontWeight: 700, align: 'center', lineHeight: 1.45 }),

      card('gs-radio-card', 532, 444, 652, 178, C.measurementSoft, { stroke: C.measurement, radius: 8 }),
      text('gs-radio-k', 558, 462, 210, 16, 'ONE-BOUNCE VA RADIO FACTOR', 9, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.05 }),
      text('gs-radio-residual', 558, 490, 600, 38, '<b>r</b><sup>radio</sup><sub>tℓ</sub> = [cτ, wrap(φ), wrap(ψ)]<sup>T</sup> − <b>h</b><sub>q</sub>(<b>x</b><sub>t</sub>,<b>m</b><sub>a<sub>tℓ</sub></sub>,<b>x</b><sub>BS</sub>)', 16, { fontWeight: 700, align: 'center' }),
      text('gs-radio-model', 558, 536, 600, 70,
        'L̂ = ‖<b>p</b><sub>t</sub>−<b>v</b><sub>j</sub>‖ + cb<sub>t</sub><br>' +
        'P<sub>tj</sub> = line(<b>p</b><sub>t</sub>,<b>v</b><sub>j</sub>) ∩ bisector(<b>p</b><sub>BS</sub>,<b>v</b><sub>j</sub>)<br>' +
        'φ̂ = wrap(bearing(P<sub>tj</sub>−<b>p</b><sub>t</sub>)−θ<sub>t</sub>), &nbsp; ψ̂ = wrap(bearing(P<sub>tj</sub>−<b>p</b><sub>BS</sub>)−θ<sub>BS</sub>)',
        14, { lineHeight: 1.45, align: 'center' }),
      text('gs-source', 96, 650, 1088, 17, 'Synthesis: standard GraphSLAM objective + radio MPC/virtual-anchor measurement geometry; angle signs depend on the array convention.', 9, { color: C.faint, fontFamily: MONO, align: 'center' })
    ], { accent: C.pose, titleSize: 31, transition: 'none' }
  )
}

function graphFallback(ctx) {
  const { text, card, shape, line, C, MONO, SANS, LIVE_BOUNDS } = ctx
  const x0 = LIVE_BOUNDS.x, y0 = LIVE_BOUNDS.y, w = LIVE_BOUNDS.width, h = LIVE_BOUNDS.height
  const stageX = x0 + 14, stageY = y0 + 14, stageW = 766, stageH = h - 28
  const railX = stageX + stageW + 12, railW = w - stageW - 40
  const poses = [[stageX + 160,stageY + 320],[stageX + 270,stageY + 285],[stageX + 382,stageY + 235],[stageX + 490,stageY + 178],[stageX + 585,stageY + 112]]
  const va1 = [stageX + 664,stageY + 82], va2 = [stageX + 100,stageY + 72], bs = [stageX + 118,stageY + 350]
  const elements = [
    card('graph-fallback-bg', x0, y0, w, h, '#F8FAFB', { stroke: C.line, radius: 0 }),
    card('graph-fallback-stage', stageX, stageY, stageW, stageH, C.paper, { stroke: C.line, radius: 6 }),
    card('graph-fallback-rail', railX, stageY, railW, stageH, C.paper, { stroke: C.line, radius: 6 }),
    text('graph-fallback-k', stageX + 22, stageY + 18, stageW - 44, 18, 'TRAJECTORY + VA MAP + FOLDED MPC ROUTES', 10, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.15 })
  ]
  poses.slice(0,-1).forEach((point,index) => elements.push(line(`graph-path-${index}`, point[0], point[1], poses[index+1][0], poses[index+1][1], C.pose, 4)))
  poses.forEach((point,index) => {
    elements.push(shape(`graph-pose-${index}`, point[0]-8, point[1]-8, 16, 16, C.paper, { shape:'ellipse', stroke:C.pose, strokeWidth:3 }))
    elements.push(text(`graph-pose-label-${index}`, point[0]+10, point[1]-12, 36, 16, `x${index}`, 8, { color:C.poseDeep, fontFamily:MONO, fontWeight:700 }))
  })
  elements.push(shape('graph-bs',bs[0]-7,bs[1]-7,14,14,C.ink,{radius:0}))
  elements.push(text('graph-bs-label',bs[0]-10,bs[1]+14,62,16,'known BS',8,{color:C.ink,fontFamily:MONO,fontWeight:700}))
  ;[[va1,'VA₁',C.map,C.mapSoft],[va2,'VA₂',C.measurement,C.measurementSoft]].forEach((item,index) => {
    elements.push(shape(`graph-va-${index}`,item[0][0]-12,item[0][1]-12,24,24,item[3],{shape:'ellipse',stroke:item[2],strokeWidth:3}))
    elements.push(text(`graph-va-label-${index}`,item[0][0]-24,item[0][1]-34,48,16,item[1],9,{color:index===0?C.mapDeep:C.measurementDeep,fontFamily:MONO,fontWeight:700,align:'center'}))
  })
  const active = poses[2], p1=[stageX+510,stageY+88], p2=[stageX+190,stageY+86]
  elements.push(line('graph-route1-a',bs[0],bs[1],p1[0],p1[1],C.map,4))
  elements.push(line('graph-route1-b',p1[0],p1[1],active[0],active[1],C.map,4))
  elements.push(line('graph-route2-a',bs[0],bs[1],p2[0],p2[1],C.measurement,4))
  elements.push(line('graph-route2-b',p2[0],p2[1],active[0],active[1],C.measurement,4))
  ;[p1,p2].forEach((p,index)=>elements.push(shape(`graph-reflection-${index}`,p[0]-5,p[1]-5,10,10,C.paper,{shape:'ellipse',stroke:index===0?C.map:C.measurement,strokeWidth:2})))
  elements.push(text('graph-rail-k',railX+18,stageY+18,railW-36,18,'GAUSS–NEWTON CONTROLS',9,{color:C.poseDeep,fontFamily:MONO,fontWeight:700,letterSpacing:1.05}))
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
      'BP-SLAM live: association marginals by message passing',
      'Move an MPC inside overlapping gates, step the messages, and compare the converged marginal association probabilities.',
      'The embedded lab uses a normalized one-scan association problem. It demonstrates the Williams–Lau message updates used as the association engine in BP-style multiobject tracking and SLAM, but it does not reproduce every continuous BP-SLAM state update. Use the shared weights tab first, then BP marginals.',
      [...associationFallback('bp', ctx), liveMount()], { accent: C.map, titleSize: 31, transition: 'none' }
    )
  }
  return regular(
    's-pmbm-slam-live', '03 · KNOWN BS/UE POSE, UNKNOWN MAP',
    'PMBM-SLAM live: rank and prune global association hypotheses',
    'Inspect compatible joint assignments, vary top-k pruning, and see how retained hypothesis mass changes the marginals.',
    'The embedded lab isolates the global-hypothesis layer that a PMBM update must manage. It deliberately does not claim to be a complete PMBM-SLAM recursion: PPP birth/undetected intensity, Bernoulli state updates, and platform-state coupling are summarized on the preceding slide.',
    [...associationFallback('pmbm', ctx), liveMount()], { accent: C.measurement, titleSize: 30, transition: 'none' }
  )
}

function graphLiveSlide(ctx) {
  const { regular, liveMount, C } = ctx
  return regular(
    's-radio-graphslam-live', '04 · KNOWN BS POSE, UNKNOWN UE POSE AND MAP',
    'Radio GraphSLAM live: Gauss–Newton on MPC factors',
    'Switch delay-only versus delay+AoA+AoD, inject a wrong association, and observe how robust loss changes the joint trajectory–map solution.',
    'Run the embedded deterministic teaching example. The continuous variables are five UE poses and two virtual anchors. Motion factors use relative odometry; each one-bounce radio factor predicts unfolded path length plus UE-frame AoA and BS-frame AoD. Association and bounce order are fixed inside one solve, exactly as stated on the equation slide.',
    [...graphFallback(ctx), liveMount()], { accent: C.pose, titleSize: 30, transition: 'none' }
  )
}

export function appendRadioSlamSlidesAfterSection(unit, ctx) {
  if (unit.id === 'map') {
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
      title: 'Section 03 · BP-SLAM association marginals'
    },
    {
      introSlide: 's-pmbm-slam-equations', slide: 's-pmbm-slam-live',
      src: '../bp-vs-pmbm-slides/live/?demo=hypotheses&embed=region', source: '../bp-vs-pmbm-slides/live/?demo=hypotheses',
      title: 'Section 03 · PMBM global hypotheses'
    },
    {
      introSlide: 's-radio-graphslam-equations', slide: 's-radio-graphslam-live',
      src: './radio-graphslam-live/?embed=region', source: './radio-graphslam-live/',
      title: 'Section 04 · radio GraphSLAM Gauss–Newton lab'
    }
  ]
  return definitions.map(entry => ({
    ...entry,
    slideIndex: slides.findIndex(slide => slide.id === entry.slide),
    inline: true, layout: 'region', bounds: LIVE_BOUNDS,
    sandbox: 'allow-scripts', hideSource: true, readyMessage: true, unloadWhenHidden: true
  }))
}
