import { readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const templatePath = join(here, '..', 'target-handover-slides', 'index.html')
const outputPath = join(here, 'index.html')

const C = {
  bg: '#F7F6F1',
  paper: '#FFFFFF',
  ink: '#1B2320',
  muted: '#5B665F',
  line: '#D8DCD1',
  teal: '#0F766E',
  tealSoft: '#E2F1EE',
  violet: '#6D4FC2',
  violetSoft: '#EEE9FB',
  rose: '#B7355C',
  roseSoft: '#F8E6EC',
  blue: '#2456C8',
  blueSoft: '#E5EBFB',
  orange: '#E08607',
  orangeSoft: '#FBEDD7',
  green: '#0E9F6E',
  grey: '#8B929C'
}

const FONT = "Georgia, 'Times New Roman', serif"

function text(id, x, y, w, h, html, fontSize = 20, options = {}) {
  return {
    id,
    type: 'text',
    x, y, w, h,
    rotation: 0,
    opacity: options.opacity ?? 1,
    html,
    fontSize,
    fontFamily: options.fontFamily || FONT,
    fontWeight: options.fontWeight ?? 400,
    color: options.color || C.ink,
    align: options.align || 'left',
    valign: options.valign || 'top',
    lineHeight: options.lineHeight ?? 1.35,
    ...(options.letterSpacing !== undefined ? { letterSpacing: options.letterSpacing } : {}),
    ...(options.link ? { link: options.link } : {}),
    ...(options.fx ? { fx: options.fx } : {})
  }
}

function shape(id, x, y, w, h, fill, options = {}) {
  return {
    id,
    type: 'shape',
    shape: options.shape || 'rect',
    x, y, w, h,
    fill,
    stroke: options.stroke ?? 'none',
    strokeWidth: options.strokeWidth ?? 0,
    radius: options.radius ?? (options.shape === 'ellipse' ? 0 : 10),
    rotation: options.rotation ?? 0,
    opacity: options.opacity ?? 1,
    ...(options.link ? { link: options.link } : {}),
    ...(options.fx ? { fx: options.fx } : {})
  }
}

function line(id, x, y, w, h, color, width = 2, options = {}) {
  return {
    id,
    type: 'shape',
    shape: 'line',
    x, y, w, h,
    fill: color,
    stroke: options.stroke || 'none',
    strokeWidth: options.strokeWidth ?? width,
    radius: 0,
    rotation: options.rotation ?? 0,
    opacity: options.opacity ?? 1,
    strokeStyle: options.strokeStyle || 'solid',
    ...(options.fx ? { fx: options.fx } : {})
  }
}

function connector(id, x1, y1, x2, y2, color, width = 2, options = {}) {
  const dx = x2 - x1
  const dy = y2 - y1
  const length = Math.hypot(dx, dy)
  return line(
    id,
    (x1 + x2) / 2 - length / 2,
    (y1 + y2) / 2 - width / 2,
    length,
    width,
    color,
    0,
    { ...options, rotation: Math.atan2(dy, dx) * 180 / Math.PI }
  )
}

function footer() {
  return [
    text('footer-l', 96, 682, 720, 24, 'Extended-object MTT · Partition uncertainty · Bai Liping', 13, { color: C.muted }),
    text('footer-r', 1084, 682, 100, 24, '{{page}} / {{pages}}', 13, { color: C.muted, align: 'right' })
  ]
}

function heading(tag, title, cite = '') {
  return [
    text('htag', 96, 58, 780, 27, tag, 15, { color: C.teal, fontWeight: 700, letterSpacing: 2.2 }),
    text('htitle', 96, 92, 1040, 62, title, 38, { fontWeight: 700, lineHeight: 1.08 }),
    shape('hbar', 96, 158, 64, 5, C.teal, { radius: 0 }),
    ...(cite ? [text('hcite', 930, 62, 254, 24, cite, 14, { color: C.muted, align: 'right' })] : [])
  ]
}

function slide(id, tag, title, notes, elements, options = {}) {
  return {
    id,
    background: options.background || C.bg,
    transition: options.transition || 'morph',
    notes,
    elements: [
      ...heading(tag, title, options.cite || ''),
      ...elements,
      ...footer()
    ]
  }
}

function card(id, x, y, w, h, options = {}) {
  return shape(id, x, y, w, h, options.fill || C.paper, {
    stroke: options.stroke || C.line,
    strokeWidth: options.strokeWidth ?? 1,
    radius: options.radius ?? 12,
    fx: options.fx
  })
}

function pill(id, x, y, w, label, fill, color = '#FFFFFF') {
  return [
    shape(`${id}-bg`, x, y, w, 34, fill, { radius: 17 }),
    text(`${id}-text`, x + 10, y + 7, w - 20, 20, label, 13, { color, fontWeight: 700, align: 'center', valign: 'middle', lineHeight: 1 })
  ]
}

const INLINE_BOUNDS = { x: 96, y: 180, width: 1088, height: 480 }

function inlineMount() {
  return shape(
    'live-demo-mount',
    INLINE_BOUNDS.x,
    INLINE_BOUNDS.y,
    INLINE_BOUNDS.width,
    INLINE_BOUNDS.height,
    'rgba(255,255,255,0)',
    { opacity: 0, radius: 0 }
  )
}

function pointCloud(prefix, x, y, scale = 1) {
  const left = [[0, 18], [26, 2], [54, 14], [18, 43]]
  const right = [[164, 24], [194, 4], [214, 28], [184, 48]]
  const extra = [[116, 23], [270, 88]]
  const elements = [
    shape(`${prefix}-extent-1`, x - 14 * scale, y - 8 * scale, 92 * scale, 68 * scale, C.blueSoft, { shape: 'ellipse', stroke: C.blue, strokeWidth: 2, opacity: 0.88 }),
    shape(`${prefix}-extent-2`, x + 148 * scale, y - 5 * scale, 92 * scale, 68 * scale, C.orangeSoft, { shape: 'ellipse', stroke: C.orange, strokeWidth: 2, opacity: 0.88 })
  ]
  left.forEach(([dx, dy], index) => {
    elements.push(shape(`${prefix}-l${index}`, x + dx * scale, y + dy * scale, 14 * scale, 14 * scale, C.blue, { shape: 'ellipse', stroke: '#FFFFFF', strokeWidth: 2 }))
  })
  right.forEach(([dx, dy], index) => {
    elements.push(shape(`${prefix}-r${index}`, x + dx * scale, y + dy * scale, 14 * scale, 14 * scale, C.orange, { shape: 'ellipse', stroke: '#FFFFFF', strokeWidth: 2 }))
  })
  elements.push(shape(`${prefix}-m9`, x + extra[0][0] * scale, y + extra[0][1] * scale, 16 * scale, 16 * scale, C.green, { shape: 'ellipse', stroke: '#FFFFFF', strokeWidth: 2 }))
  elements.push(shape(`${prefix}-m10`, x + extra[1][0] * scale, y + extra[1][1] * scale, 16 * scale, 16 * scale, '#FFFFFF', { shape: 'ellipse', stroke: C.grey, strokeWidth: 2 }))
  return elements
}

const slides = []

slides.push({
  id: 's-cover',
  background: C.bg,
  transition: 'none',
  notes: 'Extended objects generate multiple measurements per scan. That makes grouping part of data association, and the number of possible set partitions explodes. This presentation uses one ten-measurement teaching scene to compare the main ways trackers control that uncertainty.',
  elements: [
    text('cover-kicker', 96, 108, 980, 28, 'INTERACTIVE BRIEFING · EXTENDED-OBJECT MULTI-TARGET TRACKING', 15, { color: C.teal, fontWeight: 700, letterSpacing: 2.3, fx: { enter: 'fade-up', order: 0 } }),
    text('cover-title', 96, 154, 1040, 152, 'Partition uncertainty in extended-object multi-target tracking', 58, { fontWeight: 700, lineHeight: 1.05, fx: { enter: 'fade-up', order: 1 } }),
    text('cover-number', 96, 342, 490, 112, '115 975', 92, { color: C.violet, fontWeight: 700, lineHeight: 1, fx: { countUp: true } }),
    text('cover-number-label', 104, 450, 430, 54, 'set partitions already at M = 10 measurements', 20, { color: C.muted, lineHeight: 1.3 }),
    ...pointCloud('cover-cloud', 730, 346, 1.3),
    text('cover-sub', 96, 550, 1060, 54, 'Candidate sets · retained mixtures · Gibbs joint search · BP marginal approximation · no explicit hard partition', 22, { color: C.muted }),
    shape('cover-rule', 96, 622, 1088, 3, C.teal, { radius: 0, fx: { loop: { type: 'dash-march' } } }),
    text('cover-link', 96, 646, 1088, 28, 'bailiping.com/eo-mtt', 15, { color: C.teal, fontWeight: 700 })
  ]
})

slides.push(slide(
  's-scope',
  'WHY GROUPING APPEARS',
  'One object can explain a cloud, not just one point',
  'Under the common point-object model, one object produces at most one measurement per scan. An extended object may produce several. That adds a grouping question before or alongside object assignment. The points in our toy scene are individual detections; the latent objects are extended ellipses.',
  [
    card('scope-point-card', 96, 198, 500, 350, { fill: '#FFFFFF' }),
    ...pill('scope-point-pill', 122, 220, 152, 'POINT OBJECT', C.ink),
    shape('scope-point-object', 188, 310, 30, 30, C.blue, { shape: 'ellipse' }),
    shape('scope-point-meas', 392, 310, 18, 18, '#FFFFFF', { shape: 'ellipse', stroke: C.blue, strokeWidth: 3 }),
    line('scope-point-arrow', 232, 325, 138, 0, C.blue, 3),
    text('scope-point-copy', 122, 398, 448, 110, 'Common point-target model:<br><b>at most one measurement</b> from an object in one scan.', 22, { lineHeight: 1.45 }),
    card('scope-extended-card', 620, 198, 564, 350, { fill: C.tealSoft, stroke: '#B9DAD4' }),
    ...pill('scope-extended-pill', 646, 220, 180, 'EXTENDED OBJECT', C.teal),
    shape('scope-extent', 700, 292, 202, 94, C.blueSoft, { shape: 'ellipse', stroke: C.blue, strokeWidth: 2 }),
    ...[[742, 326], [782, 302], [820, 337], [850, 312], [876, 345]].map(([x, y], index) => shape(`scope-em${index}`, x, y, 18, 18, C.blue, { shape: 'ellipse', stroke: '#FFFFFF', strokeWidth: 2 })),
    text('scope-extended-copy', 646, 398, 512, 110, 'One object can generate <b>several detections</b>.<br>Which detections form a cell becomes uncertain.', 22, { lineHeight: 1.45 }),
    text('scope-boundary', 96, 580, 1088, 56, '<b>Scope boundary:</b> the teaching points are detections from geometric extended objects—not point targets, and not a complete kinematic/extent filter.', 17, { color: C.muted, align: 'center' })
  ]
))

slides.push(slide(
  's-partition',
  'THE OBJECT OF INFERENCE',
  'A partition is a set of disjoint, non-empty cells',
  'A partition covers the measurement set exactly once. Cells can then be assigned to existing objects, new objects, or—in models with Poisson point clutter—individual measurements can be treated as clutter events. The partition count does not yet include those assignment labels.',
  [
    card('partition-formula-bg', 96, 198, 1088, 100, { fill: C.violetSoft, stroke: '#D9CDEF' }),
    text('partition-formula', 130, 222, 1020, 74, 'Z = {m₁,…,mₘ} &nbsp;&nbsp;→&nbsp;&nbsp; P = {C₁,…,Cₖ}<br><span style="font-size:17px;color:#5B665F">Cᵢ ≠ ∅ · Cᵢ ∩ Cⱼ = ∅ for i ≠ j · ⋃ᵢ₌₁ᵏ Cᵢ = Z</span>', 29, { align: 'center', fontWeight: 700, lineHeight: 1.25 }),
    card('partition-example', 96, 332, 1088, 244),
    text('partition-example-label', 122, 352, 1040, 28, 'ONE VALID PARTITION OF THE TEN-MEASUREMENT SCENE', 14, { color: C.muted, fontWeight: 700, letterSpacing: 1.7 }),
    card('partition-cell-1', 126, 406, 282, 104, { fill: C.blueSoft, stroke: '#B9C9F2' }),
    text('partition-cell-1-t', 146, 428, 242, 58, 'C₁ = {m₁,m₂,m₃,m₄}<br><b>object 1 candidate</b>', 20, { color: C.blue, align: 'center' }),
    card('partition-cell-2', 432, 406, 314, 104, { fill: C.orangeSoft, stroke: '#EED3A5' }),
    text('partition-cell-2-t', 452, 428, 274, 58, 'C₂ = {m₅,…,m₉}<br><b>object 2 candidate</b>', 20, { color: C.orange, align: 'center' }),
    card('partition-cell-3', 770, 406, 180, 104, { fill: '#F0F1EE' }),
    text('partition-cell-3-t', 788, 428, 144, 58, 'C₃ = {m₁₀}<br><b>clutter?</b>', 20, { color: C.grey, align: 'center' }),
    text('partition-not-yet', 970, 418, 180, 82, 'Partitioning groups.<br><b>Association labels come next.</b>', 18, { color: C.muted, align: 'center', valign: 'middle' }),
    text('partition-clutter', 96, 598, 1088, 42, 'Under the usual Poisson point-clutter model, clutter detections are normally individual clutter events—not one multi-detection clutter source.', 16, { color: C.muted, align: 'center' })
  ],
  { cite: 'Set partitions + assignments' }
))

slides.push(slide(
  's-explosion',
  'COMBINATORIAL LOAD',
  'Bell numbers turn exact enumeration into the bottleneck',
  'Bell numbers count partitions of M labeled measurements. B(10) is 115,975, B(20) is about 5.17 times ten to the thirteen, and B(30) about 8.47 times ten to the twenty-three. A log scale is necessary just to put them on one chart. Assignments and clutter labels multiply the joint space further.',
  [
    text('explosion-big', 96, 196, 344, 104, 'B(10)<br><span style="font-size:74px;color:#6D4FC2">115 975</span>', 28, { fontWeight: 700, lineHeight: 1.12 }),
    text('explosion-copy', 96, 330, 340, 196, 'B(M) counts <b>set partitions only</b>.<br><br>Inside each partition, cells still need object / birth / clutter explanations.<br><br>Exact summation becomes impossible quickly.', 20, { color: C.muted, lineHeight: 1.45 }),
    {
      id: 'bell-chart', type: 'chart', x: 472, y: 190, w: 712, h: 394, rotation: 0, opacity: 1, preset: 'bar',
      option: {
        grid: { left: 62, right: 18, top: 34, bottom: 42 },
        xAxis: { type: 'category', data: ['4', '6', '8', '10', '15', '20', '30'], axisLabel: { fontSize: 14 }, name: 'measurements M' },
        yAxis: { type: 'value', max: 25, name: 'log₁₀ B(M)', axisLabel: { fontSize: 13 } },
        series: [{ type: 'bar', data: [1.18, 2.31, 3.62, 5.06, 9.14, 13.71, 23.93], itemStyle: { color: C.violet, borderRadius: [5, 5, 0, 0] }, barWidth: 48 }],
        tooltip: { trigger: 'item', formatter: 'M={b}: log₁₀ B(M)={c}' }
      },
      fx: { enter: 'fade-up' }
    },
    text('explosion-axis-note', 472, 594, 712, 40, 'Logarithmic vertical axis · B(20) ≈ 5.17×10¹³ · B(30) ≈ 8.47×10²³', 15, { color: C.muted, align: 'center' })
  ]
))

const roadmapRows = [
  ['A', 'Generate a restricted candidate set', 'weighted update over P<sub>cand</sub>', C.violet, C.violetSoft],
  ['B1', 'Retain global hypotheses', 'mixture of partition–association histories', C.teal, C.tealSoft],
  ['B2a', 'Gibbs / stochastic search', 'high-weight <b>joint hypotheses</b>', C.teal, '#EDF5F3'],
  ['B2b', 'Belief propagation', 'approximate association <b>marginals</b>', C.rose, C.roseSoft],
  ['C', 'One new potential object per measurement', 'no explicit hard partition', C.rose, '#FBEFF3']
]
const roadmapElements = []
roadmapRows.forEach(([code, action, output, color, fill], index) => {
  const y = 190 + index * 86
  roadmapElements.push(card(`roadmap-row-${index}`, 96, y, 1088, 70, { fill, stroke: color, strokeWidth: 1 }))
  roadmapElements.push(shape(`roadmap-code-bg-${index}`, 116, y + 13, 82, 44, color, { radius: 22 }))
  roadmapElements.push(text(`roadmap-code-${index}`, 116, y + 22, 82, 26, code, 17, { color: '#FFFFFF', fontWeight: 700, align: 'center', valign: 'middle', lineHeight: 1 }))
  roadmapElements.push(text(`roadmap-action-${index}`, 224, y + 13, 456, 46, action, 21, { fontWeight: 700, valign: 'middle' }))
  roadmapElements.push(text(`roadmap-output-${index}`, 706, y + 13, 446, 46, `OUTPUT · ${output}`, 18, { color, align: 'right', valign: 'middle' }))
})
slides.push(slide(
  's-roadmap',
  'FIVE MECHANISMS',
  'Every method answers “what do we carry forward?” differently',
  'The source page groups these mechanisms into three broad viewpoints, but it is useful to separate five outputs. Most important: Gibbs and BP are not two ways to do the same thing. Gibbs searches joint hypotheses. BP approximates marginals. The last route also uses sum-product ideas but changes the model so each measurement proposes a new potential object.',
  [
    ...roadmapElements,
    text('roadmap-note', 96, 632, 1088, 30, 'B2a and B2b share the goal “avoid enumeration,” but not the representation, output, or diagnostics.', 17, { color: C.ink, fontWeight: 700, align: 'center' })
  ]
))

slides.push(slide(
  's-candidate-mechanism',
  'FAMILY A · RESTRICT THEN MARGINALIZE',
  'The heuristic chooses the support before the filter weights it',
  'Classical extended-target PHD-family implementations often replace the full partition set with a small generated candidate set. Single-linkage distance partitioning sweeps a threshold. Other clustering and splitting ideas can generate different candidates. The downstream update marginalizes only over what the generator admitted.',
  [
    card('candidate-flow-bg', 96, 194, 1088, 176, { fill: C.violetSoft, stroke: '#D9CDEF' }),
    ...pill('candidate-step-1', 122, 236, 226, '1 · CLUSTER / SPLIT', C.violet),
    text('candidate-arrow-1', 360, 236, 62, 38, '→', 32, { color: C.violet, align: 'center' }),
    ...pill('candidate-step-2', 434, 236, 238, '2 · BUILD P<sub>cand</sub>', C.violet),
    text('candidate-arrow-2', 684, 236, 62, 38, '→', 32, { color: C.violet, align: 'center' }),
    ...pill('candidate-step-3', 758, 236, 394, '3 · WEIGHTED FILTER UPDATE', C.violet),
    text('candidate-equation', 150, 304, 980, 42, 'ν(Z) ≈ Σ<sub>P∈P<sub>cand</sub></sub> ω<sub>P</sub> · update given P', 25, { align: 'center', fontWeight: 700 }),
    text('candidate-tool-label', 96, 408, 1088, 28, 'WAYS TO GENERATE CANDIDATES · EACH IMPOSES DIFFERENT BLIND SPOTS', 14, { color: C.muted, fontWeight: 700, letterSpacing: 1.7 }),
    ...pill('tool-single', 96, 454, 196, 'single linkage', C.violet),
    ...pill('tool-complete', 308, 454, 196, 'complete / average', C.blue),
    ...pill('tool-dbscan', 520, 454, 166, 'DBSCAN', C.green),
    ...pill('tool-spectral', 702, 454, 166, 'spectral', C.rose),
    ...pill('tool-split', 884, 454, 300, 'prediction / EM splits', C.teal),
    text('candidate-catch', 96, 544, 1088, 76, '<b>The catch:</b> every partition outside P<sub>cand</sub> receives exactly zero weight. The generator limits what the filter can recover.', 22, { color: C.violet, align: 'center', valign: 'middle' })
  ],
  { cite: '[25]–[28]' }
))

slides.push(slide(
  's-candidate-blindspot',
  'A SPECIFIC BLIND SPOT',
  'This single-linkage sweep can never put m₉ on the left',
  'On the source page’s ten measurements, sweeping d from 0.5 to 40 in quarter-unit steps produces nine distinct partitions. The plausible left-assignment of m9 never appears because single linkage connects m9 to the right cluster first. This is a claim about this scene and this single-linkage sweep—not every candidate generator.',
  [
    card('blind-produced', 96, 206, 516, 310, { fill: C.violetSoft, stroke: '#D9CDEF' }),
    text('blind-produced-tag', 124, 228, 460, 26, 'PRODUCED BY THE SWEEP', 14, { color: C.violet, fontWeight: 700, letterSpacing: 1.6 }),
    text('blind-produced-code', 124, 282, 460, 54, '{1,2,3,4} {5,6,7,8,9} {10}', 27, { fontFamily: 'ui-monospace, monospace', fontWeight: 700, align: 'center' }),
    text('blind-produced-copy', 130, 370, 448, 96, 'As d increases, m₉ touches the right cluster first. Single linkage then chains from that nearest connection.', 20, { color: C.muted, align: 'center' }),
    card('blind-missing', 636, 206, 548, 310, { fill: '#FFFFFF', stroke: C.rose, strokeWidth: 2 }),
    text('blind-missing-tag', 664, 228, 492, 26, 'PLAUSIBLE · BUT ABSENT', 14, { color: C.rose, fontWeight: 700, letterSpacing: 1.6 }),
    text('blind-missing-code', 664, 282, 492, 54, '{1,2,3,4,9} {5,6,7,8} {10}', 27, { fontFamily: 'ui-monospace, monospace', fontWeight: 700, align: 'center' }),
    text('blind-missing-copy', 670, 370, 480, 96, 'No threshold can undo the first connection while preserving the two compact four-point cells.', 20, { color: C.muted, align: 'center' }),
    text('blind-count', 96, 548, 1088, 70, '<span style="font-size:42px;color:#6D4FC2"><b>9</b></span> sweep candidates &nbsp;vs&nbsp; <span style="font-size:42px;color:#1B2320"><b>115 975</b></span> valid partitions', 21, { align: 'center', valign: 'middle' }),
    inlineMount()
  ],
  { cite: 'Specific to the source-page scene' }
))

slides.push(slide(
  's-pmbm',
  'FAMILY B1 · RETAIN HYPOTHESES',
  'A PMBM posterior keeps competing histories as a mixture',
  'For the assumed extended-target model, PMBM is conjugate. Each global hypothesis indexes a partition-and-association history, has a weight, and carries its own multi-Bernoulli density. The exact representation grows multiplicatively, so practical filters gate, prune, cap, and recycle.',
  [
    card('pmbm-definition', 96, 196, 430, 386, { fill: C.tealSoft, stroke: '#B9DAD4' }),
    text('pmbm-definition-tag', 124, 222, 376, 24, 'GLOBAL HYPOTHESIS', 14, { color: C.teal, fontWeight: 700, letterSpacing: 1.7 }),
    text('pmbm-definition-eq', 124, 278, 376, 72, 'partition history<br>+ association history', 27, { fontWeight: 700, align: 'center' }),
    text('pmbm-definition-plus', 124, 368, 376, 46, '+ weight + own MB density', 19, { color: C.teal, fontWeight: 700, align: 'center' }),
    text('pmbm-definition-copy', 124, 450, 376, 84, 'Nothing is collapsed yet: competing global explanations remain explicitly correlated in the mixture.', 18, { color: C.muted, align: 'center' }),
    text('pmbm-branch-title', 574, 200, 610, 28, 'HYPOTHESIS COUNT WITHOUT MANAGEMENT', 14, { color: C.muted, fontWeight: 700, letterSpacing: 1.6 }),
    ...[4, 12, 36, 108].flatMap((value, index) => {
      const x = 584 + index * 150
      return [
        shape(`pmbm-count-bg-${index}`, x, 278, 96, 96, index === 0 ? C.teal : '#FFFFFF', { shape: 'ellipse', stroke: C.teal, strokeWidth: 2 }),
        text(`pmbm-count-${index}`, x, 307, 96, 38, String(value), 31, { color: index === 0 ? '#FFFFFF' : C.teal, fontWeight: 700, align: 'center', valign: 'middle', lineHeight: 1 }),
        ...(index < 3 ? [text(`pmbm-arrow-${index}`, x + 100, 304, 46, 42, '×3 →', 19, { color: C.muted, align: 'center' })] : [])
      ]
    }),
    text('pmbm-time', 574, 392, 610, 30, 'scan t &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; t+1 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; t+2 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; t+3', 15, { color: C.muted, align: 'center' }),
    card('pmbm-management', 574, 456, 610, 126, { fill: '#FFFFFF' }),
    text('pmbm-management-copy', 602, 480, 554, 78, '<b>Tractability comes from management:</b><br>gating · pruning · capping · recycling', 22, { align: 'center', valign: 'middle', lineHeight: 1.45 }),
    text('pmbm-bottom', 96, 612, 1088, 36, 'Principled representation does not remove combinatorics—it decides what must be approximated later.', 18, { color: C.muted, align: 'center' })
  ],
  { cite: 'PMBM [29]' }
))

slides.push(slide(
  's-pmb-projection',
  'PMB APPROXIMATION',
  'Projection keeps selected marginals, not the global mixture',
  'A PMB approximation turns the mixture into one multi-Bernoulli density using track-oriented or variational methods. The displayed m9 probabilities are illustrative values from the source-page teaching example. Selected marginal association uncertainty survives, while global correlations between hypotheses are discarded.',
  [
    card('projection-mixture', 96, 204, 420, 326, { fill: C.tealSoft, stroke: '#B9DAD4' }),
    text('projection-mixture-tag', 122, 228, 368, 25, 'PMBM MIXTURE', 14, { color: C.teal, fontWeight: 700, letterSpacing: 1.7 }),
    ...[0.46, 0.31, 0.14, 0.09].flatMap((value, index) => {
      const y = 280 + index * 52
      return [
        card(`projection-h-${index}`, 126, y, 72, 34, { fill: '#FFFFFF', radius: 6 }),
        text(`projection-h-t-${index}`, 126, y + 7, 72, 20, `H${index + 1}`, 13, { color: C.teal, fontWeight: 700, align: 'center' }),
        shape(`projection-h-bar-${index}`, 216, y + 10, value * 500, 14, C.teal, { radius: 7 }),
        text(`projection-h-v-${index}`, 436, y + 6, 50, 22, value.toFixed(2), 13, { fontFamily: 'ui-monospace, monospace', align: 'right' })
      ]
    }),
    text('projection-arrow', 528, 322, 100, 60, '→', 54, { color: C.teal, align: 'center' }),
    card('projection-pmb', 640, 204, 544, 326, { fill: '#FFFFFF' }),
    text('projection-pmb-tag', 670, 228, 484, 25, 'ONE PMB DENSITY · m₉ MARGINAL', 14, { color: C.muted, fontWeight: 700, letterSpacing: 1.5 }),
    ...[
      ['object 2', 0.46, C.orange], ['object 1', 0.31, C.blue], ['new object', 0.14, C.green], ['clutter', 0.09, C.grey]
    ].flatMap(([label, value, color], index) => {
      const y = 282 + index * 48
      return [
        text(`projection-m-label-${index}`, 674, y, 126, 26, label, 16, { color }),
        shape(`projection-m-track-${index}`, 810, y + 5, 270, 12, '#E9ECE6', { radius: 6 }),
        shape(`projection-m-bar-${index}`, 810, y + 5, value * 270, 12, color, { radius: 6 }),
        text(`projection-m-value-${index}`, 1090, y - 1, 56, 24, value.toFixed(2), 14, { fontFamily: 'ui-monospace, monospace', align: 'right' })
      ]
    }),
    card('projection-preserve', 96, 558, 516, 70, { fill: C.tealSoft, stroke: '#B9DAD4' }),
    text('projection-preserve-t', 116, 578, 476, 34, '<b>Preserves:</b> selected marginal association information', 17, { color: C.teal, align: 'center' }),
    card('projection-drop', 636, 558, 548, 70, { fill: C.roseSoft, stroke: '#E8B9C9' }),
    text('projection-drop-t', 656, 578, 508, 34, '<b>Drops:</b> global hypothesis branching and correlations', 17, { color: C.rose, align: 'center' }),
    inlineMount()
  ],
  { cite: 'PMB approximations [30]' }
))

slides.push(slide(
  's-direct-fork',
  'FAMILY B2 · A CRITICAL FORK',
  'Avoiding enumeration does not imply the same approximation',
  'Granström and colleagues use stochastic methods to search the joint partition-association space. Xia and colleagues use a factor graph and belief propagation to approximate association marginals. Calling both direct approximate inference is useful, but treating them as interchangeable is wrong.',
  [
    card('fork-source', 392, 194, 496, 84, { fill: C.ink, stroke: C.ink }),
    text('fork-source-t', 412, 214, 456, 44, 'DO NOT ENUMERATE THE FULL JOINT SPACE', 20, { color: '#FFFFFF', fontWeight: 700, align: 'center', valign: 'middle' }),
    connector('fork-left-line', 640, 278, 346, 372, C.teal, 4),
    connector('fork-right-line', 640, 278, 934, 372, C.rose, 4),
    card('fork-gibbs', 96, 372, 500, 218, { fill: C.tealSoft, stroke: C.teal, strokeWidth: 2 }),
    text('fork-gibbs-tag', 124, 396, 444, 26, 'GIBBS / STOCHASTIC OPTIMIZATION · [31]', 14, { color: C.teal, fontWeight: 700, letterSpacing: 1.1 }),
    text('fork-gibbs-title', 124, 446, 444, 52, 'Searches high-weight<br><b>joint hypotheses</b>', 27, { align: 'center', fontWeight: 700, lineHeight: 1.2 }),
    text('fork-gibbs-output', 124, 526, 444, 34, 'Output: sampled / optimized partitions + associations', 16, { color: C.muted, align: 'center' }),
    card('fork-bp', 684, 372, 500, 218, { fill: C.roseSoft, stroke: C.rose, strokeWidth: 2 }),
    text('fork-bp-tag', 712, 396, 444, 26, 'BELIEF PROPAGATION · [32]', 14, { color: C.rose, fontWeight: 700, letterSpacing: 1.3 }),
    text('fork-bp-title', 712, 446, 444, 52, 'Approximates association<br><b>marginals</b>', 27, { align: 'center', fontWeight: 700, lineHeight: 1.2 }),
    text('fork-bp-output', 712, 526, 444, 34, 'Output: soft beliefs—without enumerating hypotheses', 16, { color: C.muted, align: 'center' }),
    text('fork-bottom', 96, 616, 1088, 36, '<b>Different objects of approximation → different diagnostics:</b> mixing and mode coverage vs loopy-BP marginal bias.', 18, { align: 'center' })
  ]
))

slides.push(slide(
  's-gibbs',
  'B2a · STOCHASTIC JOINT SEARCH',
  'Gibbs sampling searches beyond fixed candidates',
  'The stochastic route proposes changes to partition and association variables and favors higher-weight joint explanations. With an irreducible chain, the correct stationary distribution, adequate mixing, and enough samples, it can explore outside a fixed heuristic set. A finite run can still miss modes. The histogram is explicitly schematic.',
  [
    card('gibbs-process', 96, 198, 510, 394, { fill: C.tealSoft, stroke: '#B9DAD4' }),
    text('gibbs-process-tag', 124, 222, 454, 24, 'ONE TEACHING MOVE', 14, { color: C.teal, fontWeight: 700, letterSpacing: 1.7 }),
    ...[
      ['1', 'choose one measurement'], ['2', 'propose existing cell / new cell / clutter'], ['3', 'sample using relative joint weight'], ['4', 'record the visited joint hypothesis']
    ].flatMap(([number, label], index) => {
      const y = 270 + index * 66
      return [
        shape(`gibbs-step-bg-${index}`, 126, y, 38, 38, C.teal, { shape: 'ellipse' }),
        text(`gibbs-step-n-${index}`, 126, y + 8, 38, 22, number, 16, { color: '#FFFFFF', fontWeight: 700, align: 'center' }),
        text(`gibbs-step-t-${index}`, 184, y + 5, 390, 30, label, 19, { valign: 'middle' })
      ]
    }),
    card('gibbs-hist', 636, 198, 548, 264),
    text('gibbs-hist-tag', 664, 222, 492, 24, 'SCHEMATIC VISIT HISTOGRAM', 14, { color: C.muted, fontWeight: 700, letterSpacing: 1.5 }),
    ...[
      ['{1–4}{5–9}', 0.44], ['{1–4,9}{5–8}', 0.29], ['{1–4}{5–8}{9}', 0.18], ['other visited', 0.09]
    ].flatMap(([label, value], index) => {
      const y = 274 + index * 44
      return [
        text(`gibbs-h-label-${index}`, 664, y, 170, 25, label, 14, { fontFamily: 'ui-monospace, monospace' }),
        shape(`gibbs-h-track-${index}`, 842, y + 4, 252, 12, '#E9ECE6', { radius: 6 }),
        shape(`gibbs-h-bar-${index}`, 842, y + 4, value * 252, 12, C.teal, { radius: 6 }),
        text(`gibbs-h-value-${index}`, 1102, y - 2, 44, 24, `${Math.round(value * 100)}%`, 13, { fontFamily: 'ui-monospace, monospace', align: 'right' })
      ]
    }),
    card('gibbs-assumptions', 636, 482, 548, 110, { fill: '#FFF8FA', stroke: '#E8B9C9' }),
    text('gibbs-assumptions-t', 660, 503, 500, 72, '<b>Needed for interpretation:</b> irreducibility · target distribution · burn-in · mixing · enough samples.<br>Finite time can miss modes.', 17, { color: C.rose, align: 'center', valign: 'middle' }),
    text('gibbs-boundary', 96, 618, 1088, 32, 'The bars above illustrate a mechanism; they are not a calibrated posterior from [31].', 16, { color: C.muted, align: 'center' })
  ],
  { cite: 'Granström et al. [31]' }
))

const factorElements = []
;[[296, 292, 'x₁', C.blue], [460, 292, 'x₂', C.orange]].forEach(([x, y, label, color], index) => {
  factorElements.push(shape(`bp-x-${index}`, x, y, 64, 64, '#FFFFFF', { shape: 'ellipse', stroke: color, strokeWidth: 3 }))
  factorElements.push(text(`bp-x-t-${index}`, x, y + 17, 64, 30, label, 22, { color, fontWeight: 700, align: 'center' }))
})
;[[216, 'a₁'], [330, 'a₂'], [444, 'a₉'], [558, 'a₁₀']].forEach(([x, label], index) => {
  factorElements.push(shape(`bp-a-${index}`, x, 446, 54, 54, '#FFFFFF', { shape: 'ellipse', stroke: C.ink, strokeWidth: 2 }))
  factorElements.push(text(`bp-a-t-${index}`, x, 461, 54, 28, label, 18, { fontWeight: 700, align: 'center' }))
  factorElements.push(connector(`bp-edge-l-${index}`, 328, 356, x + 27, 446, C.blue, 1.5, { opacity: 0.55 }))
  factorElements.push(connector(`bp-edge-r-${index}`, 492, 356, x + 27, 446, C.orange, 1.5, { opacity: 0.55 }))
  factorElements.push(shape(`bp-y-${index}`, x, 542, 54, 54, '#FFFFFF', { shape: 'ellipse', stroke: C.green, strokeWidth: 2 }))
  factorElements.push(text(`bp-y-t-${index}`, x, 557, 54, 28, `y${['₁', '₂', '₉', '₁₀'][index]}`, 18, { color: C.green, fontWeight: 700, align: 'center' }))
  factorElements.push(connector(`bp-y-edge-${index}`, x + 27, 500, x + 27, 542, C.green, 1.5, { opacity: 0.65, strokeStyle: 'dashed' }))
})
slides.push(slide(
  's-bp',
  'FAMILY C · SCHEMATIC SPA',
  'SPA passes messages—it does not sample partitions',
  'This diagram shows the Family C [33,34] factor graph: each measurement has an association variable and a measurement-born potential object. Sum-product messages approximate association marginals without explicitly enumerating partition hypotheses. Loopy SPA may bias marginals, and this schematic omits kinematic and extent inference. Trajectory-PMB BP [32] is a distinct factor-graph tracker that also approximates association marginals; it is not the method drawn here.',
  [
    card('bp-graph-bg', 96, 196, 650, 420, { fill: '#FFFFFF' }),
    text('bp-graph-tag', 122, 218, 598, 24, 'SCHEMATIC FACTOR-GRAPH VIEW', 14, { color: C.muted, fontWeight: 700, letterSpacing: 1.5 }),
    ...factorElements,
    text('bp-graph-caption', 122, 604, 598, 26, 'x: legacy objects · a: associations · y: measurement-born potential objects', 14, { color: C.muted, align: 'center' }),
    card('bp-output', 776, 196, 408, 190, { fill: C.roseSoft, stroke: '#E8B9C9' }),
    text('bp-output-tag', 802, 220, 356, 24, 'WHAT BP RETURNS', 14, { color: C.rose, fontWeight: 700, letterSpacing: 1.6 }),
    text('bp-output-main', 802, 270, 356, 64, 'p(aᵢ = object k)<br><b>soft marginals</b>', 28, { align: 'center', fontWeight: 700, lineHeight: 1.25 }),
    card('bp-not-output', 776, 406, 408, 118, { fill: '#FFFFFF', stroke: C.line }),
    text('bp-not-output-t', 800, 428, 360, 72, '<b>Not returned:</b><br>a sampled list of global partition hypotheses', 19, { color: C.muted, align: 'center', valign: 'middle' }),
    card('bp-caveat', 776, 544, 408, 72, { fill: C.tealSoft, stroke: '#B9DAD4' }),
    text('bp-caveat-t', 798, 554, 364, 56, 'Family C [33,34] shown · trajectory-PMB BP [32] is distinct and is not the graph drawn here.', 14, { color: C.teal, fontWeight: 700, align: 'center', valign: 'middle' }),
    inlineMount()
  ],
  { cite: 'Scalable SPA EOT [33,34]' }
))

slides.push(slide(
  's-no-hard-partition',
  'FAMILY C · NO EXPLICIT HARD PARTITION',
  'Every measurement proposes a potential new object',
  'The scalable SPA formulation gives each measurement an association variable over existing objects, a new object, and clutter. Each measurement also instantiates a potential new object. Groups emerge implicitly when several association marginals favor the same object. There is no explicit partition enumeration.',
  [
    text('nohard-input', 96, 208, 226, 32, 'FOR EACH mᵢ', 15, { color: C.rose, fontWeight: 700, letterSpacing: 1.7, align: 'center' }),
    shape('nohard-meas', 168, 280, 80, 80, C.rose, { shape: 'ellipse' }),
    text('nohard-meas-t', 168, 299, 80, 42, 'mᵢ', 28, { color: '#FFFFFF', fontWeight: 700, align: 'center' }),
    text('nohard-arrow', 276, 294, 74, 44, '→', 42, { color: C.rose, align: 'center' }),
    card('nohard-assoc', 368, 218, 360, 240, { fill: C.roseSoft, stroke: '#E8B9C9' }),
    text('nohard-assoc-tag', 394, 244, 308, 24, 'ASSOCIATION VARIABLE aᵢ', 14, { color: C.rose, fontWeight: 700, letterSpacing: 1.4, align: 'center' }),
    ...pill('nohard-existing', 410, 294, 276, 'existing object', C.blue),
    ...pill('nohard-new', 410, 342, 276, 'new object', C.green),
    ...pill('nohard-clutter', 410, 390, 276, 'clutter', C.grey),
    text('nohard-plus', 744, 294, 54, 44, '+', 38, { color: C.rose, align: 'center' }),
    card('nohard-potential', 816, 218, 368, 240, { fill: C.tealSoft, stroke: '#B9DAD4' }),
    text('nohard-potential-tag', 842, 244, 316, 24, 'ONE POTENTIAL OBJECT yᵢ', 14, { color: C.teal, fontWeight: 700, letterSpacing: 1.4, align: 'center' }),
    shape('nohard-y', 948, 296, 104, 104, '#FFFFFF', { shape: 'ellipse', stroke: C.green, strokeWidth: 3 }),
    text('nohard-y-t', 948, 324, 104, 50, 'yᵢ', 30, { color: C.green, fontWeight: 700, align: 'center' }),
    card('nohard-groups', 96, 492, 1088, 112, { fill: '#FFFFFF' }),
    text('nohard-groups-t', 124, 514, 1032, 70, '<b>Groups remain implicit:</b> multiple measurements can place high marginal belief on the same object.<br>No partition variable needs to be enumerated or selected.', 21, { align: 'center', valign: 'middle', lineHeight: 1.45 }),
    text('nohard-bottom', 96, 628, 1088, 30, 'This family still tracks extended objects; [34] adds explicit geometric shape inference.', 16, { color: C.muted, align: 'center' })
  ],
  { cite: 'SPA EOT [33], geometric EOT [34]' }
))

slides.push(slide(
  's-cost',
  'COMPLEXITY LENS',
  'Name the layer before quoting the scaling',
  'This table compares where complexity moves, not end-to-end runtimes. With K_legacy legacy objects and M measurements, the SPA association layer has O(K_legacy M + M²) = O(K_total M) work per fixed message iteration because the M measurements also seed M new potential objects. State prediction, extent inference, particle operations, gating, and implementation choices add cost.',
  [
    {
      id: 'cost-table', type: 'table', x: 96, y: 196, w: 1088, h: 414, rotation: 0, opacity: 1, header: true,
      columns: [{ w: 0.8 }, { w: 1.4 }, { w: 1.35 }, { w: 1.45 }],
      rows: [
        { cells: [{ html: 'Mechanism' }, { html: 'Carries forward' }, { html: 'Pressure point' }, { html: 'Characteristic failure' }] },
        { cells: [{ html: 'Candidate set', bold: true }, { html: 'few allowed partitions' }, { html: '|P<sub>cand</sub>| updates' }, { html: 'plausible partition absent' }] },
        { cells: [{ html: 'PMBM', bold: true }, { html: 'weighted global mixture' }, { html: 'branching hypothesis count' }, { html: 'pruning discards mass / correlations' }] },
        { cells: [{ html: 'Gibbs search', bold: true }, { html: 'visited joint hypotheses' }, { html: 'moves × mixing time' }, { html: 'finite run misses modes' }] },
        { cells: [{ html: 'BP marginals', bold: true }, { html: 'soft association beliefs' }, { html: 'edges × iterations' }, { html: 'loopy marginal bias' }] },
        { cells: [{ html: 'SPA potential objects', bold: true }, { html: 'M proposal nodes + beliefs' }, { html: '≈ O((K<sub>legacy</sub>+M)M) / iter.*' }, { html: 'full particle / extent cost omitted' }] }
      ],
      style: { headerBg: C.ink, headerColor: '#FFFFFF', zebra: 'rgba(27,35,32,0.04)', borderColor: 'rgba(27,35,32,0.16)', borderWidth: 1, cellPadX: 13, cellPadY: 9, fontSize: 16, color: C.ink, radius: 10 }
    },
    text('cost-note', 96, 622, 1088, 40, '*K<sub>total</sub> = K<sub>legacy</sub> + M. This is per fixed association iteration; particle and extent work is omitted.', 15, { color: C.rose, fontWeight: 700, align: 'center' })
  ]
))

slides.push(slide(
  's-decision',
  'CHOOSING THE APPROXIMATION',
  'Ask what uncertainty must survive downstream',
  'There is no universal winner. Candidate sets are simple when geometry is decisive. PMBM mixtures preserve global ambiguity but require management. Sampling is valuable when joint modes matter and can be diagnosed. BP is attractive when marginals are enough and pairwise association structure is manageable. The per-measurement-potential-object route avoids explicit partitions entirely.',
  [
    card('decision-q1', 96, 202, 524, 116, { fill: C.violetSoft, stroke: '#D9CDEF' }),
    text('decision-q1-t', 122, 224, 472, 70, '<b>Can geometry define a trustworthy small support?</b><br><span style="color:#6D4FC2">Candidate partitions may be enough.</span>', 20, { align: 'center', valign: 'middle' }),
    card('decision-q2', 660, 202, 524, 116, { fill: C.tealSoft, stroke: '#B9DAD4' }),
    text('decision-q2-t', 686, 224, 472, 70, '<b>Must cross-track joint ambiguity survive?</b><br><span style="color:#0F766E">Retain and manage a mixture.</span>', 20, { align: 'center', valign: 'middle' }),
    card('decision-q3', 96, 342, 524, 116, { fill: '#EDF5F3', stroke: '#B9DAD4' }),
    text('decision-q3-t', 122, 364, 472, 70, '<b>Do the important joint modes need explicit search?</b><br><span style="color:#0F766E">Use sampling—with mixing diagnostics.</span>', 20, { align: 'center', valign: 'middle' }),
    card('decision-q4', 660, 342, 524, 116, { fill: C.roseSoft, stroke: '#E8B9C9' }),
    text('decision-q4-t', 686, 364, 472, 70, '<b>Are soft marginals the downstream currency?</b><br><span style="color:#B7355C">Use BP / SPA—with bias diagnostics.</span>', 20, { align: 'center', valign: 'middle' }),
    card('decision-answer', 96, 494, 1088, 116, { fill: C.ink, stroke: C.ink }),
    text('decision-answer-t', 128, 520, 1024, 68, 'The design choice is not “which clustering algorithm wins?”<br><b>It is which representation of uncertainty the next update can afford—and needs.</b>', 24, { color: '#FFFFFF', align: 'center', valign: 'middle', lineHeight: 1.35 })
  ]
))

slides.push(slide(
  's-takeaways',
  'TAKEAWAYS',
  'Four distinctions prevent most conceptual mistakes',
  'Closing summary. Bell numbers explain the computational pressure. Candidate sets restrict support. PMBM represents global ambiguity then manages it. Gibbs and BP must remain distinct: joint search versus marginal approximation. The no-hard-partition formulation makes groups implicit through association beliefs and per-measurement potential objects.',
  [
    ...[
      ['1', 'B(10) = 115 975 counts partitions only', 'assignments and clutter labels enlarge the joint space', C.violet],
      ['2', 'Candidate-set misses are generator-specific', 'the live blind spot belongs to this single-linkage sweep', C.blue],
      ['3', 'Gibbs joint search ≠ BP marginal approximation', 'mixing diagnostics and marginal-bias diagnostics answer different questions', C.teal],
      ['4', 'No hard partition does not mean no uncertainty', 'soft association beliefs and potential objects carry it instead', C.rose]
    ].flatMap(([number, title, detail, color], index) => {
      const y = 190 + index * 94
      return [
        card(`take-row-${index}`, 96, y, 1088, 78, { fill: index % 2 ? '#FFFFFF' : '#F1F2EE' }),
        shape(`take-num-bg-${index}`, 118, y + 16, 46, 46, color, { shape: 'ellipse' }),
        text(`take-num-${index}`, 118, y + 27, 46, 24, number, 18, { color: '#FFFFFF', fontWeight: 700, align: 'center' }),
        text(`take-title-${index}`, 188, y + 12, 470, 28, title, 20, { fontWeight: 700 }),
        text(`take-detail-${index}`, 672, y + 13, 484, 46, detail, 16, { color: C.muted, align: 'right', valign: 'middle' })
      ]
    }),
    text('take-refs', 96, 590, 1088, 46, 'Sources: extended-target PHD/CPHD [25–28] · PMBM/PMB [29,30] · sampling [31] · trajectory-PMB BP [32] · scalable SPA EOT [33,34]', 15, { color: C.muted, align: 'center' }),
    text('take-link', 96, 638, 1088, 28, 'Interactive source and full references · bailiping.com/eo-mtt', 16, { color: C.teal, fontWeight: 700, align: 'center' })
  ]
))

const deck = {
  format: 'bento/slides',
  version: 1,
  docId: 'eo-mtt-partition-uncertainty-deck',
  title: 'Partition uncertainty in extended-object multi-target tracking',
  readonly: true,
  meta: {
    author: 'Liping Bai',
    subject: 'Partition uncertainty in extended-object multi-target tracking',
    company: 'Chalmers University of Technology',
    source: 'bailiping.com/eo-mtt'
  },
  size: { width: 1280, height: 720 },
  theme: { background: C.bg, color: C.ink, accent: C.teal, fontFamily: FONT },
  slides
}

const inlineLiveMap = [
  {
    slide: 's-candidate-blindspot',
    slideIndex: slides.findIndex(entry => entry.id === 's-candidate-blindspot'),
    inline: true,
    layout: 'region',
    bounds: INLINE_BOUNDS,
    src: './live/?view=partition&embed=region',
    source: './live/?view=partition',
    title: 'Candidate partitions and the single-linkage blind spot',
    sandbox: 'allow-scripts',
    hideSource: true,
    readyMessage: true,
    unloadWhenHidden: true
  },
  {
    slide: 's-pmb-projection',
    slideIndex: slides.findIndex(entry => entry.id === 's-pmb-projection'),
    inline: true,
    layout: 'region',
    bounds: INLINE_BOUNDS,
    src: './live/?view=hypotheses&embed=region',
    source: './live/?view=hypotheses',
    title: 'PMBM hypothesis management and PMB projection',
    sandbox: 'allow-scripts',
    hideSource: true,
    readyMessage: true,
    unloadWhenHidden: true
  },
  {
    slide: 's-bp',
    slideIndex: slides.findIndex(entry => entry.id === 's-bp'),
    inline: true,
    layout: 'region',
    bounds: INLINE_BOUNDS,
    src: './live/?view=inference&embed=region',
    source: './live/?view=inference',
    title: 'Gibbs joint search versus schematic per-measurement SPA',
    sandbox: 'allow-scripts',
    hideSource: true,
    readyMessage: true,
    unloadWhenHidden: true
  }
]

const escapedDeck = JSON.stringify(deck, null, 1).replaceAll('<', '\\u003c')
const configText = JSON.stringify(inlineLiveMap, null, 2)
let html = readFileSync(templatePath, 'utf8')
html = html.replace('<title>bento/slides</title>', '<title>Partition uncertainty in extended-object multi-target tracking | Slides</title>')
html = html.replace(
  /(<script type="application\/bento\+json" id="bento-doc">\s*)[\s\S]*?(\s*<\/script>)/,
  `$1${escapedDeck}$2`
)
html = html.replace(
  /<script type="application\/json" id="(?:bento-live-config|bento-inline-live-map)">[\s\S]*?<\/script>/,
  `<script type="application/json" id="bento-inline-live-map">\n${configText}\n    </script>`
)
html = html.replaceAll('../assets/bento-live.css', '../assets/bento-inline-live.css')
html = html.replaceAll('../assets/bento-live.js', '../assets/bento-inline-live.js')
writeFileSync(outputPath, html)
console.log(`Wrote ${outputPath} with ${slides.length} regular slides and ${inlineLiveMap.length} inline demos.`)
