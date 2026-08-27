import { readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

const buildPath = resolve('mpc-detection-to-bounce-count-slides/build.mjs')
let source = readFileSync(buildPath, 'utf8')
let changed = false

const importLine = "import { appendRadioSlamSlidesAfterSection, radioSlamLiveEntries } from './radio-slam-extra.mjs'"
if (!source.includes(importLine)) {
  const importAnchor = "import { fileURLToPath } from 'node:url'"
  if (!source.includes(importAnchor)) throw new Error('Could not find build import anchor')
  source = source.replace(importAnchor, `${importAnchor}\n${importLine}`)
  changed = true
}

if (!source.includes('appendRadioSlamSlidesAfterSection(unit,')) {
  const loopTail = "    [...sectionLiveFallback(unit), liveMount()], { accent: unit.accent, titleSize: 31, transition: 'none' }\n  ))\n}\n\nslides.push(regular(\n  's-takeaway'"
  const patchedLoopTail = "    [...sectionLiveFallback(unit), liveMount()], { accent: unit.accent, titleSize: 31, transition: 'none' }\n  ))\n  appendRadioSlamSlidesAfterSection(unit, {\n    slides, regular, text, card, shape, line, C, SANS, MONO, LIVE_BOUNDS, liveMount\n  })\n}\n\nslides.push(regular(\n  's-takeaway'"
  if (!source.includes(loopTail)) throw new Error('Could not find section loop tail')
  source = source.replace(loopTail, patchedLoopTail)
  changed = true
}

if (!source.includes('...radioSlamLiveEntries({ slides, LIVE_BOUNDS })')) {
  const liveMapPattern = /const inlineLiveMap = sectionUnits\.map\(unit => \{([\s\S]*?)\n\}\)\n\nconst deck =/
  const match = source.match(liveMapPattern)
  if (!match) throw new Error('Could not find inline live map')
  const body = match[1]
  const replacement = `const inlineLiveMap = [\n  ...sectionUnits.map(unit => {${body}\n  }),\n  ...radioSlamLiveEntries({ slides, LIVE_BOUNDS })\n]\n\nconst deck =`
  source = source.replace(liveMapPattern, replacement)
  changed = true
}

if (!source.includes("'s-pose-double-math'")) {
  const introLine = "  slides.push(regular(introId, unit.section, unit.title, unit.introSubtitle, unit.note, sectionIntroElements(unit), { accent: unit.accent, titleSize: 35, transition: 'none' }))"
  const insertAnchor = `${introLine}\n  slides.push(regular(`
  if (!source.includes(insertAnchor)) throw new Error('Could not find section intro/live insertion anchor')

  const poseDoubleMathBlock = [
    "  if (unit.id === 'pose') {",
    "    slides.push(regular(",
    "      's-pose-double-math', unit.section, 'Corner ×2 without LoS: noisy joint pose–map estimation',",
    "      'One associated single-bounce MPC and one double-bounce MPC share wall A; the UE pose and both walls are unknown.',",
    "      'Formulate the single-snapshot corner case before opening the interactive construction. The BS pose and clock synchronization are known. Path 1 reflects on wall A; path 2 follows A→B. Both delay/global-AoD/body-AoA tuples are noisy, and there is deliberately no LoS factor. Emphasize that the local measurement Jacobian is 6×7, so a generic solution has at least one null direction; noise turns the exact continuum into a likelihood ridge rather than a unique estimate.',",
    "      [",
    "        ...labelPill('pose-double-math-pill', 96, 190, 244, '4.2 · CORNER ×2 · NO LoS', C.poseDeep, C.poseSoft),",
    "        text('pose-double-math-association', 364, 196, 820, 26, 'Associated routes: path 1 = A · path 2 = A→B · synchronized delay', 12, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, align: 'right', letterSpacing: .35 }),",
    "",
    "        card('pose-double-measurements-card', 96, 238, 330, 348, C.paper, { stroke: C.measurement, strokeWidth: 2 }),",
    "        text('pose-double-measurements-k', 118, 258, 286, 20, 'NOISY MPC DATA', 10, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.25 }),",
    "        text('pose-double-measurements-eq', 116, 286, 290, 92, texBlock`\\begin{aligned}\\tilde{\\mathbf z}_k&=[c\\tilde\\tau_k,\\,\\tilde\\psi_k^{\\rm g},\\,\\tilde\\varphi_k^{\\rm b}]^{\\mathsf T}\\\\&=\\mathbf h_k(\\mathbf x)+\\boldsymbol\\epsilon_k,\\quad k\\in\\{1,2\\},\\\\\\boldsymbol\\epsilon_k&\\sim\\mathcal N(\\mathbf 0,\\mathbf R_k).\\end{aligned}`, 14, { color: C.ink, align: 'center' }),",
    "        text('pose-double-measurements-copy', 118, 384, 286, 34, 'k = 1: A; k = 2: A→B. AoD is global; AoA remains in the UE body frame.', 11.5, { color: C.soft, fontFamily: SANS, lineHeight: 1.3 }),",
    "        card('pose-double-no-los-card', 116, 428, 290, 54, C.measurementSoft, { stroke: C.measurement }),",
    "        text('pose-double-no-los-eq', 126, 437, 102, 34, texBlock`\\mathcal Z_{\\mathrm{LoS}}=\\varnothing`, 15, { color: C.measurementDeep, align: 'center' }),",
    "        text('pose-double-no-los-copy', 238, 439, 156, 34, 'No direct range or bearing residual anchors the UE.', 11.5, { color: C.measurementDeep, fontFamily: SANS, fontWeight: 700, lineHeight: 1.25 }),",
    "        text('pose-double-state-k', 118, 504, 286, 20, 'UNKNOWN STATE · 7 DOF', 10, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.15 }),",
    "        text('pose-double-state-eq', 116, 526, 290, 48, texBlock`\\begin{aligned}\\mathbf x&=[p_x,p_y,\\theta,\\beta_A,\\rho_A,\\beta_B,\\rho_B]^{\\mathsf T},\\\\\\ell_j&=\\{\\mathbf r:\\mathbf n(\\beta_j)^{\\mathsf T}\\mathbf r=\\rho_j\\}.\\end{aligned}`, 12.8, { color: C.ink, align: 'center' }),",
    "",
    "        card('pose-double-model-card', 444, 238, 372, 348, C.paper, { stroke: C.pose, strokeWidth: 2 }),",
    "        text('pose-double-model-k', 466, 258, 328, 20, 'ASSOCIATED SPECULAR PATHS', 10, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.15 }),",
    "        text('pose-double-paths-eq', 464, 282, 332, 70, texBlock`\\begin{aligned}\\mathcal P_1&:\\;\\mathbf b\\to\\mathbf q_1\\to\\mathbf p,\\quad\\mathbf q_1\\in\\ell_A,\\\\\\mathcal P_2&:\\;\\mathbf b\\to\\mathbf q_{21}\\to\\mathbf q_{22}\\to\\mathbf p,\\quad\\mathbf q_{21}\\in\\ell_A,\\;\\mathbf q_{22}\\in\\ell_B.\\end{aligned}`, 13.7, { color: C.ink, align: 'center' }),",
    "        shape('pose-double-model-rule-1', 466, 360, 328, 1, C.line, { radius: 0 }),",
    "        text('pose-double-length-k', 466, 374, 328, 18, 'PREDICTED PATH LENGTHS', 9.5, { color: C.faint, fontFamily: MONO, fontWeight: 700, letterSpacing: 1 }),",
    "        text('pose-double-lengths-eq', 462, 394, 336, 76, texBlock`\\begin{aligned}L_1&=\\|\\mathbf q_1-\\mathbf b\\|+\\|\\mathbf p-\\mathbf q_1\\|,\\\\L_2&=\\|\\mathbf q_{21}-\\mathbf b\\|+\\|\\mathbf q_{22}-\\mathbf q_{21}\\|+\\|\\mathbf p-\\mathbf q_{22}\\|.\\end{aligned}`, 13.2, { color: C.ink, align: 'center' }),",
    "        shape('pose-double-model-rule-2', 466, 478, 328, 1, C.line, { radius: 0 }),",
    "        text('pose-double-angle-eq', 462, 490, 336, 54, texBlock`\\begin{aligned}\\psi_k^{\\rm g}&=\\angle(\\mathbf q_{k,1}-\\mathbf b),\\\\\\varphi_k^{\\rm b}&=\\operatorname{wrap}[\\angle(\\mathbf q_{k,\\rm last}-\\mathbf p)-\\theta].\\end{aligned}`, 12.5, { color: C.ink, align: 'center' }),",
    "        text('pose-double-specular', 466, 550, 328, 24, `At each hit: ${tex`\\widehat{\\mathbf d}^{+}=(\\mathbf I-2\\mathbf n\\mathbf n^{\\mathsf T})\\widehat{\\mathbf d}^{-}`} · forward segments only.`, 10.8, { color: C.poseDeep, fontFamily: SANS, fontWeight: 700, align: 'center' }),",
    "",
    "        card('pose-double-estimation-card', 834, 238, 350, 348, C.paper, { stroke: C.poseDeep, strokeWidth: 2 }),",
    "        text('pose-double-estimation-k', 856, 258, 306, 20, 'CONSTRAINED MAXIMUM LIKELIHOOD', 10, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1 }),",
    "        text('pose-double-estimation-eq', 852, 286, 314, 86, texBlock`\\begin{aligned}r_{k,L}&=c\\tilde\\tau_k-L_k(\\mathbf x),\\\\r_{k,\\psi}&=\\operatorname{wrap}(\\tilde\\psi_k^{\\rm g}-\\psi_k^{\\rm g}(\\mathbf x)),\\\\r_{k,\\varphi}&=\\operatorname{wrap}(\\tilde\\varphi_k^{\\rm b}-\\varphi_k^{\\rm b}(\\mathbf x)),\\\\\\widehat{\\mathbf x}&=\\arg\\min_{\\mathbf x}\\sum_{k=1}^{2}\\mathbf r_k^{\\mathsf T}\\mathbf R_k^{-1}\\mathbf r_k.\\end{aligned}`, 11.2, { color: C.ink, align: 'center' }),",
    "        text('pose-double-constraints', 856, 382, 306, 70, 'Subject to: shared wall A, wall incidence, equal-angle reflection, positive residual length, and the ordered route A→B.', 12.5, { color: C.soft, fontFamily: SANS, lineHeight: 1.4 }),",
    "        card('pose-double-rank-card', 854, 468, 310, 94, C.poseSoft, { stroke: C.pose }),",
    "        text('pose-double-rank-k', 870, 480, 278, 16, 'LOCAL OBSERVABILITY', 9.5, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1 }),",
    "        text('pose-double-rank-eq', 864, 500, 290, 48, texBlock`\\dim\\mathbf x=7,\\quad\\dim(\\mathbf z_1,\\mathbf z_2)=6\\;\\Longrightarrow\\;\\dim\\ker\\mathbf J\\ge 1`, 13.2, { color: C.poseDeep, align: 'center' }),",
    "        text('pose-double-rank-copy', 856, 566, 306, 18, 'Generic result: a pose–map likelihood ridge, not one point.', 11.5, { color: C.poseDeep, fontFamily: SANS, fontWeight: 700, align: 'center' }),",
    "",
    "        card('pose-double-conclusion-card', 96, 606, 1088, 48, C.poseDeep, { stroke: C.poseDeep }),",
    "        text('pose-double-conclusion', 118, 616, 1044, 28, 'Noise thickens the remaining null curve. On the next slide, select Corner ×2 to visualize feasible slices and route rejection.', 15, { color: C.paper, fontFamily: SANS, fontWeight: 700, align: 'center', valign: 'middle' })",
    "      ], { accent: C.pose, titleSize: 32, transition: 'none' }",
    "    ))",
    "  }"
  ].join('\n')

  source = source.replace(insertAnchor, `${introLine}\n${poseDoubleMathBlock}\n  slides.push(regular(`)
  changed = true
}

function replaceSlideNotation(from, to) {
  if (source.includes(from)) {
    source = source.replace(from, to)
    changed = true
  }
}

if (source.includes("'s-pose-double-math'")) {
  replaceSlideNotation(
    'Both delay/AoD/body-AoA tuples are noisy',
    'Both delay/global-AoD/body-AoA tuples are noisy'
  )
  replaceSlideNotation(
    String.raw`\tilde\psi_k,\,\tilde\varphi_k^{\rm b}`,
    String.raw`\tilde\psi_k^{\rm g},\,\tilde\varphi_k^{\rm b}`
  )
  replaceSlideNotation(
    "'k = 1 is the single-bounce MPC; k = 2 is the double-bounce MPC.', 12, { color: C.soft, fontFamily: SANS, lineHeight: 1.35 }",
    "'k = 1: A; k = 2: A→B. AoD is global; AoA remains in the UE body frame.', 11.5, { color: C.soft, fontFamily: SANS, lineHeight: 1.3 }"
  )
  replaceSlideNotation(
    String.raw`\psi_k&=\angle(\mathbf q_{k,1}-\mathbf b)-\theta_B`,
    String.raw`\psi_k^{\rm g}&=\angle(\mathbf q_{k,1}-\mathbf b)`
  )
  replaceSlideNotation(
    "text('pose-double-estimation-eq', 852, 286, 314, 86, texBlock`\\widehat{\\mathbf x}=\\arg\\min_{\\mathbf x}\\sum_{k=1}^{2}\\left\\|\\operatorname{wrap}_{\\angle}\\!\\left(\\tilde{\\mathbf z}_k-\\mathbf h_k(\\mathbf x)\\right)\\right\\|_{\\mathbf R_k^{-1}}^{2}`, 13.7, { color: C.ink, align: 'center' })",
    "text('pose-double-estimation-eq', 852, 286, 314, 86, texBlock`\\begin{aligned}r_{k,L}&=c\\tilde\\tau_k-L_k(\\mathbf x),\\\\r_{k,\\psi}&=\\operatorname{wrap}(\\tilde\\psi_k^{\\rm g}-\\psi_k^{\\rm g}(\\mathbf x)),\\\\r_{k,\\varphi}&=\\operatorname{wrap}(\\tilde\\varphi_k^{\\rm b}-\\varphi_k^{\\rm b}(\\mathbf x)),\\\\\\widehat{\\mathbf x}&=\\arg\\min_{\\mathbf x}\\sum_{k=1}^{2}\\mathbf r_k^{\\mathsf T}\\mathbf R_k^{-1}\\mathbf r_k.\\end{aligned}`, 11.2, { color: C.ink, align: 'center' })"
  )
  replaceSlideNotation(
    'Noise thickens the remaining null curve. The next interactive slide visualizes feasible slices and rejects non-forward or negative-length routes.',
    'Noise thickens the remaining null curve. On the next slide, select Corner ×2 to visualize feasible slices and route rejection.'
  )
}

for (const marker of [
  importLine,
  'appendRadioSlamSlidesAfterSection(unit,',
  '...radioSlamLiveEntries({ slides, LIVE_BOUNDS })',
  "'s-pose-double-math'",
  'Corner ×2 without LoS: noisy joint pose–map estimation',
  String.raw`\tilde\psi_k^{\rm g}`,
  String.raw`r_{k,L}&=c\tilde\tau_k-L_k`,
  String.raw`\dim\ker\mathbf J\ge 1`
]) {
  if (!source.includes(marker)) throw new Error(`Patch validation failed: ${marker}`)
}

if (changed) {
  writeFileSync(buildPath, source)
  console.log(`Patched ${buildPath}`)
} else {
  console.log(`No changes needed in ${buildPath}`)
}
