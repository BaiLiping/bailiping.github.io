import { readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const templatePath = join(here, '..', 'frame-registration-slides', 'index.html')
const outputPath = join(here, 'index.html')

const C = {
  bg: '#F4F6F8', paper: '#FFFFFF', ink: '#16222E', soft: '#51606E', faint: '#8A97A3', line: '#D7DEE5',
  measurement: '#E8720C', measurementDeep: '#B45607', measurementSoft: '#FCEBDA',
  known: '#7C4DBE', knownDeep: '#5D3691', knownSoft: '#F0EAF9',
  map: '#0E8F7E', mapDeep: '#0A6B5E', mapSoft: '#E3F2EF',
  pose: '#1874B8', poseDeep: '#0D568C', poseSoft: '#E8F2FA',
  ue: '#2CA02C', danger: '#C22F2F'
}

const SERIF = "Georgia, 'Times New Roman', serif"
const SANS = "Arial, Helvetica, sans-serif"
const MONO = "Menlo, Consolas, monospace"
const LIVE_BOUNDS = { x: 96, y: 180, width: 1088, height: 480 }

function text(id, x, y, w, h, html, fontSize = 22, options = {}) {
  return {
    id, type: 'text', x, y, w, h, rotation: 0, opacity: options.opacity ?? 1, html, fontSize,
    fontFamily: options.fontFamily || SERIF, fontWeight: options.fontWeight ?? 400,
    color: options.color || C.ink, align: options.align || 'left', valign: options.valign || 'top',
    lineHeight: options.lineHeight ?? 1.3,
    ...(options.letterSpacing === undefined ? {} : { letterSpacing: options.letterSpacing }),
    ...(options.link ? { link: options.link } : {}), ...(options.fx ? { fx: options.fx } : {})
  }
}

function shape(id, x, y, w, h, fill, options = {}) {
  return {
    id, type: 'shape', shape: options.shape || 'rect', x, y, w, h, fill,
    stroke: options.stroke ?? 'none', strokeWidth: options.strokeWidth ?? 0,
    radius: options.radius ?? (options.shape === 'ellipse' ? 0 : 9), rotation: options.rotation ?? 0,
    opacity: options.opacity ?? 1, ...(options.link ? { link: options.link } : {}),
    ...(options.fx ? { fx: options.fx } : {})
  }
}

function line(id, x1, y1, x2, y2, color, width = 2, options = {}) {
  const dx = x2 - x1, dy = y2 - y1, length = Math.hypot(dx, dy)
  return shape(id, (x1 + x2 - length) / 2, (y1 + y2 - width) / 2, length, width, color, {
    radius: 0, rotation: Math.atan2(dy, dx) * 180 / Math.PI, opacity: options.opacity ?? 1,
    ...(options.fx ? { fx: options.fx } : {})
  })
}

function card(id, x, y, w, h, fill = C.paper, options = {}) {
  return shape(id, x, y, w, h, fill, { stroke: options.stroke || C.line, strokeWidth: options.strokeWidth ?? 1, radius: options.radius ?? 10 })
}

function image(id, x, y, w, h, src, options = {}) {
  return {
    id, type: 'image', x, y, w, h, src, fit: options.fit || 'contain',
    radius: options.radius ?? 0, rotation: options.rotation ?? 0, opacity: options.opacity ?? 1,
    ...(options.alt ? { alt: options.alt } : {}), ...(options.fx ? { fx: options.fx } : {})
  }
}

function footer(section) {
  return [
    text('footer-left', 96, 684, 820, 18, `MPC detection → bounce count · ${section} · Bai Liping`, 11, { color: C.faint, fontFamily: SANS }),
    text('footer-right', 1080, 684, 104, 18, '{{page}} / {{pages}}', 11, { color: C.faint, fontFamily: SANS, align: 'right' })
  ]
}

function regular(id, section, title, subtitle, notes, elements, options = {}) {
  const accent = options.accent || C.pose
  return {
    id, background: options.background || C.bg, transition: options.transition || 'morph', notes,
    elements: [
      text('section-label', 96, 28, 770, 22, section.toUpperCase(), 12, { color: accent, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.8 }),
      text('deck-label', 920, 30, 264, 20, 'RADIO GEOMETRY · INTERACTIVE', 10, { color: C.faint, fontFamily: MONO, fontWeight: 700, align: 'right', letterSpacing: 1 }),
      shape('top-rule', 96, 66, 1088, 1, C.line, { radius: 0 }),
      text('slide-title', 96, 88, 1088, 54, title, options.titleSize || 37, { fontWeight: 700, lineHeight: 1.08, fx: { enter: 'fade-up', order: 0 } }),
      ...(subtitle ? [text('slide-subtitle', 96, 146, 1088, 30, subtitle, 16, { color: C.soft, fontFamily: SANS, lineHeight: 1.35, fx: { enter: 'fade-up', order: 1 } })] : []),
      ...elements, ...footer(section)
    ]
  }
}

function labelPill(id, x, y, w, label, color, fill) {
  return [
    shape(`${id}-bg`, x, y, w, 32, fill, { stroke: color, strokeWidth: 1, radius: 16 }),
    text(`${id}-text`, x + 8, y + 7, w - 16, 18, label, 11, { color, fontFamily: MONO, fontWeight: 700, align: 'center', valign: 'middle', lineHeight: 1 })
  ]
}

const cases = [
  {
    id: 'known-los', section: '02 · KNOWN BS/UE POSE AND MAP', number: '2.1', title: 'Line of sight: confirm the zero-bounce route',
    short: 'LoS', mode: 'known', caseId: 'los', bounces: 0, accent: C.known, deep: C.knownDeep, soft: C.knownSoft,
    premise: 'With the map and both poses fixed, the direct segment is a complete geometric hypothesis—not merely the shortest delay.',
    known: 'finite map · BS pose · UE pose · both headings', data: 'delay L · global AoA φ · global AoD ψ',
    method: 'Check map visibility, direct distance, and the two opposing bearing rays.',
    verdict: 'Accept only when all gates agree; then bounce count = 0.',
    observe: 'Perturb any measured quantity and watch the direct-path gate fail.',
    note: 'Start with the clean direct path. Emphasize that AoA and AoD confirmation uses the known array headings and that a clear finite-map segment is still required.'
  },
  {
    id: 'known-single', section: '02 · KNOWN BS/UE POSE AND MAP', number: '2.2', title: 'Single bounce: mirror once, fold once',
    short: 'Single', mode: 'known', caseId: 'single', bounces: 1, accent: C.known, deep: C.knownDeep, soft: C.knownSoft,
    premise: 'A mapped wall turns a one-bounce route into a direct line to one virtual anchor.',
    known: 'wall A · finite support · BS/UE poses', data: 'delay L₁ · global AoA φ · global AoD ψ',
    method: 'Mirror the BS in wall A, then fold UE→VA¹ back through that wall.',
    verdict: 'The folded point must lie on the finite wall and predict the complete tuple.',
    observe: 'Move the UE or perturb the tuple; the mapped route either remains coherent or fails.',
    note: 'Explain the image-source method as a deterministic map operation. No one-bounce prefix is needed because the candidate wall is already known.'
  },
  {
    id: 'known-double', section: '02 · KNOWN BS/UE POSE AND MAP', number: '2.3', title: 'Double bounce: test an ordered wall pair',
    short: 'Corner ×2', mode: 'known', caseId: 'double', bounces: 2, accent: C.known, deep: C.knownDeep, soft: C.knownSoft,
    premise: 'Wall order is part of the hypothesis: A→B and B→A are different unfolded routes.',
    known: 'finite walls A and B · BS/UE poses', data: 'one two-bounce MPC tuple',
    method: 'Mirror BS through A then B; fold UE→VA² through B then A.',
    verdict: 'Accept A→B only when both points, every leg, and the tuple are feasible.',
    observe: 'Step through the two mirrors and two folds; watch order determine the points.',
    note: 'Stress that the bounce count is the length of an accepted ordered wall sequence. Finite support and occlusion checks matter as much as angle parity.'
  },
  {
    id: 'known-triple', section: '02 · KNOWN BS/UE POSE AND MAP', number: '2.4', title: 'Triple bounce: extend the image-source ladder',
    short: 'Corner ×3', mode: 'known', caseId: 'triple', bounces: 3, accent: C.known, deep: C.knownDeep, soft: C.knownSoft,
    premise: 'A third mapped reflector adds one mirror and one fold—not a new inference principle.',
    known: 'finite walls A, B, C · BS/UE poses', data: 'one three-bounce MPC tuple',
    method: 'Build VA¹→VA²→VA³, then fold the final straight line through C→B→A.',
    verdict: 'All three reflection points must be ordered, on-wall, visible, and measurement-consistent.',
    observe: 'Advance through the ladder and see how one invalid wall point rejects the full route.',
    note: 'Use the recursive visual rhythm: mirror forward in hypothesis order, then fold backward through the same walls.'
  },
  {
    id: 'known-corridor2', section: '02 · KNOWN BS/UE POSE AND MAP', number: '2.5', title: 'Known corridor: a two-bounce route is still testable',
    short: 'Corridor ×2', mode: 'known', caseId: 'corridor', bounces: 2, accent: C.known, deep: C.knownDeep, soft: C.knownSoft,
    premise: 'Parallel walls create angle symmetries, but the known map fixes their locations and finite support.',
    known: 'corridor walls R and L · BS/UE poses', data: 'delay and two global bearings',
    method: 'Mirror through R→L and fold back to the two mapped wall segments.',
    verdict: 'The map resolves the geometric slide: only a feasible R→L route is accepted.',
    observe: 'Drag the UE and watch the known walls keep the two reflection points pinned.',
    note: 'Contrast this with the unknown-map corridor later. Parallelism alone is not a problem when the wall offsets are supplied by the map.'
  },
  {
    id: 'known-corridor3', section: '02 · KNOWN BS/UE POSE AND MAP', number: '2.6', title: 'Known corridor: repeated walls and path parity',
    short: 'Corridor ×3', mode: 'known', caseId: 'corridor3', bounces: 3, accent: C.known, deep: C.knownDeep, soft: C.knownSoft,
    premise: 'A valid ordered sequence may revisit a wall: R→L→R is a legitimate three-bounce candidate.',
    known: 'corridor walls R and L · repeated support', data: 'one odd-order MPC tuple',
    method: 'Climb a three-rung image ladder, then fold through R→L→R.',
    verdict: 'Parity is a clue; the finite-map route and full tuple provide the decision.',
    observe: 'Follow the repeated-wall fold and compare its odd-order angle signature.',
    note: 'Point out that angle parity cannot by itself distinguish every odd-bounce path from a lower-order path. The map performs the decisive feasibility test.'
  },
  {
    id: 'unknown-single', section: '03 · KNOWN BS/UE POSE, UNKNOWN MAP', number: '3.1', title: 'Single bounce: infer a VA, point, and wall',
    short: 'Single', mode: 'map', caseId: 'usingle', bounces: 1, accent: C.map, deep: C.mapDeep, soft: C.mapSoft,
    premise: 'Known poses convert both array bearings to the global frame even when no wall map exists.',
    known: 'BS/UE positions · both headings', data: 'delay L · global AoA φ · global AoD ψ',
    method: 'Walk the full reverse-AoA length to VA, intersect the delay ellipse, then bisect BS↔VA.',
    verdict: 'The measurement constructs one incidence point and one supporting wall line.',
    observe: 'Move the UE or perturb a bearing; VA, ellipse intersection, and inferred wall move together.',
    note: 'Make the estimator boundary explicit: faint reference geometry explains the synthetic answer but is not an input.'
  },
  {
    id: 'unknown-double', section: '03 · KNOWN BS/UE POSE, UNKNOWN MAP', number: '3.2', title: 'Double bounce: peel with a one-bounce prefix',
    short: 'Corner ×2', mode: 'map', caseId: 'udouble', bounces: 2, accent: C.map, deep: C.mapDeep, soft: C.mapSoft,
    premise: 'A higher-order MPC alone leaves a VA family; an associated prefix path supplies the first anchor.',
    known: 'BS/UE poses · associated one-bounce prefix', data: 'path-1 and path-2 tuples',
    method: 'Recover VA¹ from path 1, locate P₂, subtract the last leg, then infer P₁ and wall B.',
    verdict: 'Positive residual length and ordered forward geometry certify the two-bounce construction.',
    observe: 'Step through the peel and see exactly where the lower-order prefix enters.',
    note: 'Do not imply that path 2 identifies both walls alone. The associated one-bounce prefix is an explicit extra input.'
  },
  {
    id: 'unknown-triple', section: '03 · KNOWN BS/UE POSE, UNKNOWN MAP', number: '3.3', title: 'Triple bounce: recurse through nested prefixes',
    short: 'Corner ×3', mode: 'map', caseId: 'utriple', bounces: 3, accent: C.map, deep: C.mapDeep, soft: C.mapSoft,
    premise: 'Each recovered prefix VA becomes the focus needed to peel one more bounce.',
    known: 'BS/UE poses · one- and two-bounce prefixes', data: 'three associated MPC tuples',
    method: 'Build VA¹ and VA² from prefixes, then peel path 3 through three delay ellipses.',
    verdict: 'The recursion returns three ordered incidence points and three inferred walls.',
    observe: 'Advance to the final rung and inspect how each derived residual length is spent.',
    note: 'Explain the anchor ladder as data association plus geometry. Without correct prefix association the recursive construction is not licensed.'
  },
  {
    id: 'unknown-corridor2', section: '03 · KNOWN BS/UE POSE, UNKNOWN MAP', number: '3.4', title: 'Unknown corridor: the crossing degenerates',
    short: 'Corridor ×2', mode: 'map', caseId: 'ucorridor', bounces: 2, accent: C.map, deep: C.mapDeep, soft: C.mapSoft,
    premise: 'When the two walls are parallel, the candidate lines can coincide instead of crossing transversely.',
    known: 'BS/UE poses · associated prefix', data: 'one- and two-bounce tuples',
    method: 'Run the same peeling construction and compare the two candidate UE lines.',
    verdict: 'Coincident lines retain a corridor slide; the data does not determine both wall offsets.',
    observe: 'Watch AoA/AoD parity lock while the inferred corridor explanation remains non-unique.',
    note: 'This is the first clear observability warning: successful algebra does not guarantee a unique wall map.'
  },
  {
    id: 'unknown-corridor3', section: '03 · KNOWN BS/UE POSE, UNKNOWN MAP', number: '3.5', title: 'Unknown corridor: more bounces preserve the slide',
    short: 'Corridor ×3', mode: 'map', caseId: 'ucorridor3', bounces: 3, accent: C.map, deep: C.mapDeep, soft: C.mapSoft,
    premise: 'A third corridor bounce adds path-order checks but no transverse information.',
    known: 'BS/UE poses · two nested prefixes', data: 'one-, two-, and three-bounce tuples',
    method: 'Peel twice and compare the third candidate line with the first.',
    verdict: 'Parity rejects impossible routes yet leaves the same wall-offset null direction.',
    observe: 'Compare odd and even angle signatures while the corridor slide survives.',
    note: 'Separate feasibility from observability. Higher order can prune bad candidates without making the surviving corridor family unique.'
  },
  {
    id: 'unknown-ambiguity', section: '03 · KNOWN BS/UE POSE, UNKNOWN MAP', number: '3.6', title: 'Two-wall ambiguity: a composite VA hides wall rotation',
    short: 'Ambiguity', mode: 'map', caseId: 'uestimate', bounces: 2, accent: C.map, deep: C.mapDeep, soft: C.mapSoft,
    premise: 'A composite virtual anchor identifies an ordered reflection transform—not always the two individual walls.',
    known: 'multiple known UE poses · fixed BS pose', data: 'geometric MPC tuples over motion',
    method: 'Rotate both intersecting walls together and recompute the physical incidence points.',
    verdict: 'The VA and radio geometry stay fixed while the walls move: one true null mode remains.',
    observe: 'Move the common-wall rotation and watch delay/AoA/AoD remain unchanged.',
    note: 'This ambiguity belongs only to the unknown-map regime. Finite support, a separate prefix, or calibrated radiometry can add information.'
  },
  {
    id: 'pose-single', section: '04 · KNOWN BS POSE, UNKNOWN UE POSE AND MAP', number: '4.1', title: 'Single bounce: one path leaves a two-parameter family',
    short: 'Single', mode: 'pose', caseId: 'usingleu', bounces: 1, accent: C.pose, deep: C.poseDeep, soft: C.poseSoft,
    premise: 'The BS gives a global AoD, but UE AoA is body-frame data until a heading hypothesis is supplied.',
    known: 'BS position + heading · synchronized delay', data: 'L · global AoD ψ · body AoA φbody',
    method: 'Walk AoD to mirrored-UE endpoint E; hypothesize bounce P and heading θ; infer UE and wall.',
    verdict: 'Every feasible (P, θ) is coherent; their union fills the BS-centred delay disk.',
    observe: 'Move P and θ independently and watch the UE candidate and wall pivot together.',
    note: 'Do not promote the body-frame AoA into the map frame without θ. The slider is a candidate slice, not a compass measurement.'
  },
  {
    id: 'pose-double', section: '04 · KNOWN BS POSE, UNKNOWN UE POSE AND MAP', number: '4.2', title: 'Double bounce: prune the joint family by path order',
    short: 'Corner ×2', mode: 'pose', caseId: 'udoubleu', bounces: 2, accent: C.pose, deep: C.poseDeep, soft: C.poseSoft,
    premise: 'The first path hypothesizes UE and wall A; the second path must survive a forward reflected-ray test.',
    known: 'BS pose · associated path pair', data: 'two delays · global AoDs · body AoAs',
    method: 'Choose (P¹, θ), infer wall A, then strip path 2 against that wall and its remaining delay.',
    verdict: 'Invalid slices turn red; the feasible subset still retains a joint pose–map ambiguity.',
    observe: 'Sweep heading until a forward intersection reverses or the delay budget becomes negative.',
    note: 'The second path adds constraints, but it does not reveal heading by itself. Preserve the feasible family rather than reporting one arbitrary point.'
  },
  {
    id: 'pose-triple', section: '04 · KNOWN BS POSE, UNKNOWN UE POSE AND MAP', number: '4.3', title: 'Triple bounce: run the recursive feasibility test',
    short: 'Corner ×3', mode: 'pose', caseId: 'utripleu', bounces: 3, accent: C.pose, deep: C.poseDeep, soft: C.poseSoft,
    premise: 'Every candidate heading rotates all body-frame AoAs together and changes the entire three-wall construction.',
    known: 'BS pose · associated three-path ladder', data: 'three delays · global AoDs · body AoAs',
    method: 'Infer wall A, require the path-2 prefix, then trace path 3 through two forward wall hits.',
    verdict: 'Only slices with ordered positive segments at every rung remain feasible.',
    observe: 'Change θ and see one rejected prefix prevent the third wall from being constructed.',
    note: 'Use red states as physically meaningful rejection, not numerical failure. Infinite-line crossings behind a ray origin are not bounces.'
  },
  {
    id: 'pose-corridor2', section: '04 · KNOWN BS POSE, UNKNOWN UE POSE AND MAP', number: '4.4', title: 'Unknown-pose corridor: heading masquerades as wall tilt',
    short: 'Corridor ×2', mode: 'pose', caseId: 'ucorridoru', bounces: 2, accent: C.pose, deep: C.poseDeep, soft: C.poseSoft,
    premise: 'At the reference heading the candidate lines coincide; another heading wedges the inferred walls.',
    known: 'BS pose · associated two-path data', data: 'delays · global AoDs · body AoAs',
    method: 'Hypothesize wall R from path 1, strip path 2, and compare its UE line with line 1.',
    verdict: 'Feasible headings re-dress the same corridor slide; infeasible headings are rejected.',
    observe: 'Sweep θ and see the corridor become a wedge that still closes on the sliding UE.',
    note: 'Heading uncertainty and map tilt are coupled. A corridor double bounce supplies feasibility but not the missing transverse anchor.'
  },
  {
    id: 'pose-corridor3', section: '04 · KNOWN BS POSE, UNKNOWN UE POSE AND MAP', number: '4.5', title: 'Unknown-pose corridor: parity seals the null direction',
    short: 'Corridor ×3', mode: 'pose', caseId: 'ucorridor3u', bounces: 3, accent: C.pose, deep: C.poseDeep, soft: C.poseSoft,
    premise: 'The third path must pass two ordered wall hits, yet its surviving line still closes on the same UE family.',
    known: 'BS pose · associated three-path data', data: 'three delays · global AoDs · body AoAs',
    method: 'Build wall R and wall L hypotheses, then strip path 3 twice and compare line 3 with line 1.',
    verdict: 'More bounces prune slices but do not remove the corridor slide.',
    observe: 'Compare the final candidate lines at θ = 0 and at a feasible nonzero heading.',
    note: 'An off-axis wall, LoS interval, second anchor, or other independent factor is required to close this null direction.'
  },
  {
    id: 'pose-rank-point', section: '04 · KNOWN BS POSE, UNKNOWN UE POSE AND MAP', number: '4.6a', title: 'Rank test: nonparallel wall families give a point',
    short: 'Rank · point', mode: 'pose', caseId: 'uestimateu', figure: 'point', bounces: 1, accent: C.pose, deep: C.poseDeep, soft: C.poseSoft,
    premise: 'A deliberately stronger sensor supplies displacement vectors directly in the global frame.',
    known: 'BS pose · global displacement vectors', data: 'single-bounce E points over multiple poses',
    method: 'Use n̂ ∝ ΔE − o for wall directions, then solve the two-family offset system.',
    verdict: 'Nonparallel normals make the 2×2 system full rank and select one intersection.',
    observe: 'Move the two family members, then solve and watch the residual collapse to a point.',
    note: 'State the stronger assumption clearly. Ordinary wheel or IMU odometry is body-frame data and would keep heading inside a nonlinear graph.'
  },
  {
    id: 'pose-rank-line', section: '04 · KNOWN BS POSE, UNKNOWN UE POSE AND MAP', number: '4.6b', title: 'Rank test: parallel walls return a line',
    short: 'Rank · line', mode: 'pose', caseId: 'uestimateu', figure: 'corridor', bounces: 2, accent: C.pose, deep: C.poseDeep, soft: C.poseSoft,
    premise: 'The same global-displacement special case becomes rank deficient when both wall normals are parallel.',
    known: 'BS pose · global displacement vectors', data: 'same-wall E sequences in a corridor',
    method: 'Build the two offset families and inspect the singular values of the cross-family system.',
    verdict: 'Rank one fixes one combination; walls and trajectory slide along the surviving null vector.',
    observe: 'Press solve, then sweep the null direction instead of forcing a unique answer.',
    note: 'Optimization should report a line of solutions here. The smallest singular value is an observability diagnostic, not a nuisance to hide.'
  }
]

const sectionUnits = [
  {
    id: 'known', mode: 'known', section: '02 · KNOWN BS/UE POSE AND MAP', title: 'Known BS/UE pose and map',
    accent: C.known, deep: C.knownDeep, soft: C.knownSoft, defaultCase: 'los', defaultBounces: 0,
    introSubtitle: 'Known geometry turns inference into route validation.',
    premise: 'The bounce count is the length of the mapped wall sequence that reproduces the measured MPC.',
    known: 'finite wall map · BS/UE positions · both array headings',
    method: 'Mirror through the selected walls, fold to incidence points, and compare the complete MPC tuple.',
    verdict: 'The accepted route’s wall-sequence length is its bounce count.',
    flow: [['KNOWN', 'BS/UE poses + map'], ['DO', 'enumerate → mirror → fold'], ['RETURN', 'accepted sequence length']],
    watch: 'Next: test LoS through three-bounce routes with the same map-aware feasibility rule.',
    observe: 'Switch among all six mapped candidates and compare how the same map-aware feasibility test scales from LoS to three bounces.',
    note: 'Introduce the known-map regime as enumeration plus validation. The next slide consolidates LoS, single, double, triple, and both corridor cases behind one tile selector.',
    tiles: [
      ['2.1', 'LoS', '0 bounce'], ['2.2', 'Single', '1 bounce'], ['2.3', 'Corner ×2', '2 bounces'],
      ['2.4', 'Corner ×3', '3 bounces'], ['2.5', 'Corridor ×2', 'R→L'], ['2.6', 'Corridor ×3', 'R→L→R']
    ]
  },
  {
    id: 'map', mode: 'map', section: '03 · KNOWN BS/UE POSE, UNKNOWN MAP', title: 'Known BS/UE pose, unknown map',
    accent: C.map, deep: C.mapDeep, soft: C.mapSoft, defaultCase: 'usingle', defaultBounces: 1,
    introSubtitle: 'Known poses let the measurements bootstrap the missing map.',
    premise: 'Construct virtual anchors and walls from MPCs, then peel higher bounce orders using associated prefix paths.',
    known: 'BS/UE positions · both array headings · associated prefix paths for higher orders',
    method: 'Construct a VA and wall from one path; use associated prefixes to peel higher orders.',
    verdict: 'Retain the constructed bounce order—and any unresolved wall family—without inventing uniqueness.',
    flow: [['KNOWN', 'BS/UE poses'], ['DO', 'MPCs → VA → wall'], ['RETURN', 'order or wall family']],
    watch: 'Next: compare VA construction, recursive peeling, and the corridor ambiguity.',
    observe: 'Use one tile bar to compare identifiable single-bounce geometry, recursive corners, corridor degeneracy, and the two-wall null mode.',
    note: 'Emphasize the estimator boundary: reference walls explain the synthetic answer but are not inputs. The next slide keeps all six unknown-map constructions in one live workspace.',
    tiles: [
      ['3.1', 'Single', 'one wall'], ['3.2', 'Corner ×2', 'anchor ladder'], ['3.3', 'Corner ×3', 'full recursion'],
      ['3.4', 'Corridor ×2', 'coincident lines'], ['3.5', 'Corridor ×3', 'parity'], ['3.6', 'Ambiguity', 'wall rotation']
    ]
  },
  {
    id: 'pose', mode: 'pose', section: '04 · KNOWN BS POSE, UNKNOWN UE POSE AND MAP', title: 'Known BS pose, unknown UE pose and map',
    accent: C.pose, deep: C.poseDeep, soft: C.poseSoft, defaultCase: 'usingleu', defaultBounces: 1,
    introSubtitle: 'The measurements may support a family rather than one solution.',
    premise: 'Estimate UE pose and walls jointly, then use feasibility and rank to decide whether the answer is a point or a family.',
    known: 'BS position + heading · synchronized delays · associated path ladder',
    method: 'Hypothesize heading and a first bounce; keep ordered positive-length routes and inspect rank.',
    verdict: 'Report the feasible family unless an independent factor makes the system full rank.',
    flow: [['KNOWN', 'BS pose'], ['DO', 'joint UE + wall hypotheses'], ['RETURN', 'point or family from rank']],
    watch: 'Next: compare recursive feasibility with the final point-versus-line rank test.',
    observe: 'Move through five recursive cases, then use the Rank tile’s point/line switch to compare full-rank and corridor outcomes.',
    note: 'Do not promote body-frame AoA to the global frame without a heading hypothesis. The next slide consolidates every §4 construction, including both rank-test outcomes.',
    tiles: [
      ['4.1', 'Single', '2D family'], ['4.2', 'Corner ×2', 'feasible subset'], ['4.3', 'Corner ×3', 'recursive test'],
      ['4.4', 'Corridor ×2', 'continuum'], ['4.5', 'Corridor ×3', 'null direction'], ['4.6', 'Rank test', 'point or line']
    ]
  }
]

function introElements(item) {
  return [
    ...labelPill('case-pill', 96, 198, 150, `${item.number} · ${item.short}`, item.deep, item.soft),
    text('premise', 96, 250, 720, 84, item.premise, 27, { fontWeight: 700, lineHeight: 1.2 }),
    card('known-card', 96, 370, 336, 154, item.soft, { stroke: item.accent }),
    text('known-k', 120, 392, 286, 20, 'KNOWN', 11, { color: item.deep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.4 }),
    text('known-v', 120, 426, 286, 74, item.known, 18, { lineHeight: 1.4 }),
    card('data-card', 454, 370, 336, 154, C.measurementSoft, { stroke: C.measurement }),
    text('data-k', 478, 392, 286, 20, 'MPC DATA', 11, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.4 }),
    text('data-v', 478, 426, 286, 74, item.data, 18, { lineHeight: 1.4 }),
    card('method-card', 812, 198, 372, 326, C.paper, { stroke: C.line }),
    text('method-k', 840, 226, 316, 20, 'CONSTRUCTION', 11, { color: item.deep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.4 }),
    text('method-v', 840, 260, 316, 92, item.method, 21, { fontWeight: 700, lineHeight: 1.3 }),
    shape('method-rule', 840, 372, 316, 1, C.line, { radius: 0 }),
    text('verdict-k', 840, 394, 316, 20, 'DECISION', 11, { color: item.deep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.4 }),
    text('verdict-v', 840, 426, 316, 72, item.verdict, 18, { color: C.soft, lineHeight: 1.35 }),
    card('observe-card', 96, 550, 1088, 82, item.deep, { stroke: item.deep }),
    text('observe-k', 124, 569, 152, 20, 'ON THE NEXT SLIDE', 10, { color: '#DDEEFF', fontFamily: MONO, fontWeight: 700, letterSpacing: 1.2 }),
    text('observe-v', 292, 565, 858, 42, item.observe, 18, { color: C.paper, fontWeight: 700, valign: 'middle' })
  ]
}

function pathPoints(item) {
  const presets = {
    0: [],
    1: [[455, 280]],
    2: [[390, 286], [620, 300]],
    3: [[350, 292], [505, 250], [665, 310]]
  }
  return presets[Math.min(3, item.bounces)]
}

function liveFallback(item) {
  const x0 = LIVE_BOUNDS.x, y0 = LIVE_BOUNDS.y, w = LIVE_BOUNDS.width, h = LIVE_BOUNDS.height
  const stageX = x0 + 14, stageY = y0 + 14, stageW = 752, stageH = h - 28
  const railX = stageX + stageW + 12, railW = w - stageW - 40
  const bs = [stageX + 120, stageY + 285], ue = [stageX + 620, stageY + 170]
  const points = pathPoints(item).map(([x, y]) => [stageX + x - 96, stageY + y - 180])
  const nodes = [bs, ...points, ue]
  const elements = [
    card('fallback-bg', x0, y0, w, h, '#F8FAFB', { stroke: C.line, radius: 0 }),
    card('fallback-stage', stageX, stageY, stageW, stageH, C.paper, { stroke: C.line, radius: 6 }),
    card('fallback-rail', railX, stageY, railW, stageH, C.paper, { stroke: C.line, radius: 6 }),
    line('wall-top', stageX + 84, stageY + 78, stageX + 680, stageY + 78, C.faint, 4, { opacity: item.mode === 'known' ? .72 : .28 }),
    line('wall-right', stageX + 680, stageY + 78, stageX + 680, stageY + 360, C.faint, 4, { opacity: item.mode === 'known' ? .72 : .28 }),
    shape('bs-mark', bs[0] - 7, bs[1] - 7, 14, 14, C.ink, { radius: 0 }),
    text('bs-label', bs[0] - 15, bs[1] + 14, 60, 18, 'BS', 11, { color: C.ink, fontFamily: MONO, fontWeight: 700 }),
    shape('ue-mark', ue[0] - 8, ue[1] - 8, 16, 16, C.ue, { shape: 'ellipse', stroke: C.paper, strokeWidth: 2 }),
    text('ue-label', ue[0] + 12, ue[1] - 9, 90, 18, item.mode === 'pose' ? 'UE?' : 'UE', 11, { color: C.ue, fontFamily: MONO, fontWeight: 700 })
  ]
  nodes.slice(0, -1).forEach((node, index) => elements.push(line(`path-${index}`, node[0], node[1], nodes[index + 1][0], nodes[index + 1][1], C.soft, 4)))
  points.forEach((point, index) => {
    elements.push(shape(`bounce-${index}`, point[0] - 6, point[1] - 6, 12, 12, C.map, { shape: 'ellipse', stroke: C.paper, strokeWidth: 2 }))
    elements.push(text(`bounce-label-${index}`, point[0] + 10, point[1] - 19, 70, 18, `P${index + 1}`, 10, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700 }))
  })
  elements.push(text('fallback-path-label', stageX + 30, stageY + 24, 690, 24, `${item.short} · ${item.bounces} bounce${item.bounces === 1 ? '' : 's'} · deterministic initial state`, 12, { color: item.deep, fontFamily: MONO, fontWeight: 700 }))
  elements.push(text('rail-head', railX + 18, stageY + 20, railW - 36, 20, 'INTERACTIVE CONTROLS', 10, { color: item.deep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.2 }))
  ;['delay L = cτ', 'arrival angle φ', 'departure angle ψ'].forEach((label, index) => {
    const y = stageY + 60 + index * 72
    elements.push(text(`control-label-${index}`, railX + 18, y, railW - 36, 18, label, 12, { color: C.soft, fontFamily: SANS, fontWeight: 700 }))
    elements.push(shape(`control-track-${index}`, railX + 18, y + 27, railW - 36, 5, C.line, { radius: 3 }))
    elements.push(shape(`control-thumb-${index}`, railX + 130 + index * 18, y + 20, 18, 18, index === 0 ? C.measurement : item.accent, { shape: 'ellipse' }))
  })
  elements.push(card('result-card', railX + 18, stageY + 290, railW - 36, 114, item.soft, { stroke: item.accent, radius: 6 }))
  elements.push(text('result-k', railX + 34, stageY + 308, railW - 68, 18, 'EXPECTED RESULT', 10, { color: item.deep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1 }))
  elements.push(text('result-v', railX + 34, stageY + 338, railW - 68, 52, item.verdict, 15, { fontWeight: 700, lineHeight: 1.3 }))
  return elements
}

function liveMount() {
  return shape('live-demo-mount', LIVE_BOUNDS.x, LIVE_BOUNDS.y, LIVE_BOUNDS.width, LIVE_BOUNDS.height, 'rgba(255,255,255,0)', { opacity: 0, radius: 0 })
}

function sectionIntroElements(unit) {
  return [
    card('section-idea-card', 96, 206, 1088, 214, unit.soft, { stroke: unit.accent, strokeWidth: 2 }),
    text('section-idea-k', 126, 232, 200, 20, 'KEY IDEA', 11, { color: unit.deep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.5 }),
    text('section-idea-v', 126, 278, 1028, 108, unit.premise, 32, { fontWeight: 700, lineHeight: 1.22, valign: 'middle' }),
    ...unit.flow.flatMap((item, index) => {
      const widths = [300, 330, 322], xs = [96, 464, 862], x = xs[index], width = widths[index]
      return [
        card(`section-flow-card-${index}`, x, 458, width, 108, index === 2 ? unit.deep : C.paper, { stroke: index === 2 ? unit.deep : unit.accent, strokeWidth: 1, radius: 8 }),
        text(`section-flow-k-${index}`, x + 22, 480, width - 44, 18, item[0], 10, { color: index === 2 ? '#DDEEFF' : unit.deep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.3 }),
        text(`section-flow-v-${index}`, x + 22, 510, width - 44, 38, item[1], 18, { color: index === 2 ? C.paper : C.ink, fontWeight: 700, lineHeight: 1.22, valign: 'middle' })
      ]
    }),
    text('section-flow-arrow-1', 396, 488, 68, 42, '→', 34, { color: unit.accent, align: 'center', valign: 'middle' }),
    text('section-flow-arrow-2', 794, 488, 68, 42, '→', 34, { color: unit.accent, align: 'center', valign: 'middle' }),
    shape('section-watch-rule', 96, 594, 1088, 1, C.line, { radius: 0 }),
    text('section-watch', 96, 610, 1088, 30, unit.watch, 16, { color: unit.deep, fontFamily: SANS, fontWeight: 700, align: 'center' })
  ]
}

function sectionLiveFallback(unit) {
  const x0 = LIVE_BOUNDS.x, y0 = LIVE_BOUNDS.y, w = LIVE_BOUNDS.width, h = LIVE_BOUNDS.height
  const tabsX = x0 + 14, tabsY = y0 + 8, tabsW = w - 28, tabW = tabsW / unit.tiles.length
  const stageX = x0 + 14, stageY = y0 + 70, stageW = 752, stageH = h - 84
  const railX = stageX + stageW + 12, railW = w - stageW - 40
  const bs = [stageX + 115, stageY + 255], ue = [stageX + 630, stageY + 135]
  const bounceSets = {
    0: [],
    1: [[stageX + 430, stageY + 82]],
    2: [[stageX + 300, stageY + 80], [stageX + 590, stageY + 94]],
    3: [[stageX + 245, stageY + 92], [stageX + 440, stageY + 55], [stageX + 620, stageY + 120]]
  }
  const points = bounceSets[Math.min(3, unit.defaultBounces)]
  const nodes = [bs, ...points, ue]
  const elements = [
    card('fallback-bg', x0, y0, w, h, '#F8FAFB', { stroke: C.line, radius: 0 })
  ]
  unit.tiles.forEach((tile, index) => {
    const x = tabsX + index * tabW
    elements.push(card(`fallback-tab-${index}`, x, tabsY, tabW - 1, 52, index === 0 ? unit.soft : C.paper, { stroke: index === 0 ? unit.accent : C.line, radius: 2 }))
    elements.push(text(`fallback-tab-num-${index}`, x + 7, tabsY + 6, tabW - 14, 12, tile[0], 8, { color: unit.deep, fontFamily: MONO, fontWeight: 700 }))
    elements.push(text(`fallback-tab-name-${index}`, x + 7, tabsY + 20, tabW - 14, 17, tile[1], 12, { fontWeight: 700 }))
    elements.push(text(`fallback-tab-detail-${index}`, x + 7, tabsY + 38, tabW - 14, 10, tile[2], 7, { color: C.faint, fontFamily: MONO }))
  })
  elements.push(card('fallback-stage', stageX, stageY, stageW, stageH, C.paper, { stroke: C.line, radius: 6 }))
  elements.push(card('fallback-rail', railX, stageY, railW, stageH, C.paper, { stroke: C.line, radius: 6 }))
  elements.push(line('wall-top', stageX + 82, stageY + 62, stageX + 676, stageY + 62, C.faint, 4, { opacity: unit.mode === 'known' ? .72 : .25 }))
  elements.push(line('wall-right', stageX + 676, stageY + 62, stageX + 676, stageY + 330, C.faint, 4, { opacity: unit.mode === 'known' ? .72 : .25 }))
  nodes.slice(0, -1).forEach((node, index) => elements.push(line(`fallback-path-${index}`, node[0], node[1], nodes[index + 1][0], nodes[index + 1][1], C.soft, 4)))
  points.forEach((point, index) => {
    elements.push(shape(`fallback-bounce-${index}`, point[0] - 6, point[1] - 6, 12, 12, C.map, { shape: 'ellipse', stroke: C.paper, strokeWidth: 2 }))
    elements.push(text(`fallback-bounce-label-${index}`, point[0] + 9, point[1] - 18, 68, 16, `P${index + 1}`, 9, { color: C.mapDeep, fontFamily: MONO, fontWeight: 700 }))
  })
  elements.push(shape('fallback-bs', bs[0] - 7, bs[1] - 7, 14, 14, C.ink, { radius: 0 }))
  elements.push(text('fallback-bs-label', bs[0] - 15, bs[1] + 13, 52, 16, 'BS', 10, { color: C.ink, fontFamily: MONO, fontWeight: 700 }))
  elements.push(shape('fallback-ue', ue[0] - 8, ue[1] - 8, 16, 16, C.ue, { shape: 'ellipse', stroke: C.paper, strokeWidth: 2 }))
  elements.push(text('fallback-ue-label', ue[0] + 12, ue[1] - 9, 70, 16, unit.mode === 'pose' ? 'UE?' : 'UE', 10, { color: C.ue, fontFamily: MONO, fontWeight: 700 }))
  elements.push(text('fallback-stage-label', stageX + 24, stageY + 18, 700, 18, `${unit.tiles[0][0]} · ${unit.tiles[0][1]} · select any tile to replace this construction`, 10, { color: unit.deep, fontFamily: MONO, fontWeight: 700 }))
  elements.push(text('fallback-rail-k', railX + 18, stageY + 18, railW - 36, 18, 'ACTIVE CASE CONTROLS', 9, { color: unit.deep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.1 }))
  ;['delay L = cτ', unit.mode === 'pose' ? 'body-frame AoA φ' : 'arrival angle φ', 'departure angle ψ'].forEach((label, index) => {
    const y = stageY + 54 + index * 61
    elements.push(text(`fallback-control-label-${index}`, railX + 18, y, railW - 36, 16, label, 11, { color: C.soft, fontFamily: SANS, fontWeight: 700 }))
    elements.push(shape(`fallback-control-track-${index}`, railX + 18, y + 24, railW - 36, 5, C.line, { radius: 3 }))
    elements.push(shape(`fallback-control-thumb-${index}`, railX + 120 + index * 22, y + 18, 17, 17, index === 0 ? C.measurement : unit.accent, { shape: 'ellipse' }))
  })
  elements.push(card('fallback-result-card', railX + 18, stageY + 252, railW - 36, 132, unit.soft, { stroke: unit.accent, radius: 6 }))
  elements.push(text('fallback-result-k', railX + 34, stageY + 270, railW - 68, 16, 'SECTION DECISION', 9, { color: unit.deep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1 }))
  elements.push(text('fallback-result-v', railX + 34, stageY + 298, railW - 68, 74, unit.verdict, 14, { fontWeight: 700, lineHeight: 1.3 }))
  return elements
}

const slides = []

slides.push({
  id: 's-cover', background: C.bg, transition: 'none',
  notes: 'Open with the conversion problem: an MPC tuple is not yet a bounce count. The presentation moves through three knowledge regimes and makes each geometric construction operable on the following live slide.',
  elements: [
    text('cover-kicker', 96, 70, 980, 28, 'INTERACTIVE BRIEFING · RADIO MULTIPATH GEOMETRY', 14, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 2.2, fx: { enter: 'fade-up', order: 0 } }),
    text('cover-title', 96, 128, 1088, 128, 'MPC detection<br><span style="color:#0A6B5E">→ bounce count</span>', 64, { fontWeight: 700, lineHeight: 1.02, fx: { enter: 'fade-up', order: 1 } }),
    text('cover-sub', 96, 286, 900, 58, 'Turn delay, AoA, AoD, and path loss into a physically valid route—under progressively weaker geometric knowledge.', 23, { color: C.soft, fontFamily: SANS, lineHeight: 1.35 }),
    ...[
      ['01', 'Measurement', 'τ · φ · ψ · path loss', C.measurement, C.measurementSoft],
      ['02', 'Known map', 'test the route directly', C.known, C.knownSoft],
      ['03', 'Unknown map', 'bootstrap VAs and walls', C.map, C.mapSoft],
      ['04', 'Unknow UE&Map', 'retain the joint family', C.pose, C.poseSoft]
    ].flatMap((item, index) => {
      const x = 96 + index * 276
      return [
        card(`cover-card-${index}`, x, 408, 252, 150, item[4], { stroke: item[3] }),
        text(`cover-num-${index}`, x + 20, 430, 46, 24, item[0], 14, { color: item[3], fontFamily: MONO, fontWeight: 700 }),
        text(`cover-head-${index}`, x + (index === 3 ? 14 : 20), 470, index === 3 ? 224 : 210, 30, item[1], index === 3 ? 19 : 23, { fontWeight: 700 }),
        text(`cover-copy-${index}`, x + 20, 510, 210, 26, item[2], 13, { color: C.soft, fontFamily: SANS })
      ]
    }),
    shape('cover-rule', 96, 612, 1088, 3, C.pose, { radius: 0, fx: { loop: { type: 'dash-march' } } }),
    text('cover-link', 96, 640, 1088, 22, 'bailiping.com/mpc-detection-to-bounce-count', 13, { color: C.poseDeep, fontFamily: MONO, fontWeight: 700 })
  ]
})

slides.push(regular(
  's-measurement', '01 · MEASUREMENT', 'One resolved path gives four observables',
  'Geometry starts with delay and bearings; power helps distinguish otherwise similar routes.',
  'Define the resolved path tuple. Delay becomes path length, AoA and AoD are local until headings are known, and path loss remains a calibrated radiometric observation rather than a direct bounce counter.',
  [
    card('tuple-card', 96, 214, 690, 300, C.paper, { stroke: C.measurement, strokeWidth: 2 }),
    text('tuple', 132, 252, 618, 72, '(τ, φ, ψ, PL)', 52, { color: C.measurementDeep, fontWeight: 700, align: 'center' }),
    text('tuple-map', 132, 350, 618, 116, 'delay τ → L = cτ<br>AoA φ → arrival bearing at the UE<br>AoD ψ → departure bearing at the BS<br>PL → path loss', 21, { lineHeight: 1.55 }),
    card('boundary-card', 820, 214, 364, 300, C.measurementSoft, { stroke: C.measurement }),
    text('boundary-k', 848, 242, 308, 20, 'IMPORTANT BOUNDARY', 11, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.2 }),
    text('boundary-v', 848, 282, 308, 178, 'An MPC tuple constrains a route.<br><br>It does <b>not</b> name the walls, their order, or the bounce count by itself.', 24, { fontWeight: 700, lineHeight: 1.35 }),
    text('measurement-foot', 96, 560, 1088, 60, 'The rest of the deck asks what becomes identifiable as map and pose knowledge are removed.', 20, { color: C.soft, fontFamily: SANS, align: 'center' })
  ], { accent: C.measurement }
))

slides.push(regular(
  's-pdp', '01 · MEASUREMENT', 'The PDP separates paths; calibrated power grades them',
  'Delay resolves candidate routes into peaks. Relative power adds evidence about their physical plausibility.',
  'Read the power-delay profile from left to right. Each resolved peak contributes one tuple, while calibrated path loss can penalize implausible material, roughness, interaction-count, or blockage hypotheses. Power supports geometry; it does not replace it.',
  [
    card('pdp-chart', 96, 214, 690, 330, C.paper, { stroke: C.line }),
    text('pdp-k', 124, 236, 320, 20, 'POWER–DELAY PROFILE', 11, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.3 }),
    line('pdp-axis-y', 146, 278, 146, 488, C.soft, 2),
    line('pdp-axis-x', 146, 488, 744, 488, C.soft, 2),
    ...[0, 1, 2, 3].flatMap(index => {
      const y = 488 - index * 58
      return [
        line(`pdp-grid-${index}`, 146, y, 744, y, C.line, 1, { opacity: index === 0 ? 0 : .7 }),
        text(`pdp-db-${index}`, 102, y - 8, 34, 16, `${-30 + index * 10}`, 10, { color: C.faint, fontFamily: MONO, align: 'right' })
      ]
    }),
    ...[
      [226, 148, 'LoS', C.measurement],
      [330, 72, 'MPC 1', C.known],
      [438, 108, 'MPC 2', C.map],
      [548, 52, 'MPC 3', C.pose],
      [664, 28, 'noise', C.faint]
    ].flatMap((peak, index) => [
      line(`pdp-stem-${index}`, peak[0], 486, peak[0], 486 - peak[1], peak[3], index === 4 ? 2 : 4),
      shape(`pdp-peak-${index}`, peak[0] - 6, 480 - peak[1], 12, 12, peak[3], { shape: 'ellipse', stroke: C.paper, strokeWidth: 2 }),
      text(`pdp-label-${index}`, peak[0] - 39, 458 - peak[1], 78, 20, peak[2], 10, { color: peak[3], fontFamily: MONO, fontWeight: 700, align: 'center' })
    ]),
    text('pdp-y-label', 146, 260, 220, 16, 'relative power (dB) ↑', 10, { color: C.faint, fontFamily: MONO }),
    text('pdp-x-label', 560, 502, 184, 18, 'excess delay τ →', 11, { color: C.faint, fontFamily: MONO, align: 'right' }),
    card('power-card', 820, 214, 364, 330, C.measurementSoft, { stroke: C.measurement }),
    text('power-k', 848, 238, 308, 20, 'WHY PATH LOSS HELPS', 11, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.2 }),
    ...[
      ['01', 'interactions', 'more bounces usually spend more power'],
      ['02', 'materials', 'reflection loss depends on the surface'],
      ['03', 'roughness', 'diffuse scattering weakens the specular path'],
      ['04', 'blockage', 'occlusion can remove an otherwise valid route']
    ].flatMap((factor, index) => {
      const y = 280 + index * 60
      return [
        text(`power-num-${index}`, 848, y, 30, 18, factor[0], 10, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700 }),
        text(`power-head-${index}`, 888, y - 2, 106, 20, factor[1], 15, { fontWeight: 700 }),
        text(`power-copy-${index}`, 994, y - 2, 162, 36, factor[2], 12, { color: C.soft, fontFamily: SANS, lineHeight: 1.3 })
      ]
    }),
    text('pdp-foot', 96, 576, 1088, 42, 'Geometry proposes the route. Calibrated radiometry helps rank competing, geometrically valid routes.', 20, { color: C.measurementDeep, fontWeight: 700, align: 'center' })
  ], { accent: C.measurement }
))

slides.push(regular(
  's-pdp-motion', '01 · MEASUREMENT', 'Watch the delay profile evolve as the UE moves',
  'Every path peak traces changing route length, visibility, and gain along the UE trajectory.',
  'Let the animation run. Point out how the first arrival shifts with geometric range, while reflected paths drift, appear, and disappear as their route lengths and visibility conditions change. The measurements are not a frozen fingerprint of the environment.',
  [
    card('pdp-motion-card', 96, 200, 1088, 438, C.paper, { stroke: C.measurement, strokeWidth: 2 }),
    text('pdp-motion-k', 118, 211, 320, 18, 'ANIMATED POWER–DELAY PROFILE', 10, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, letterSpacing: 1.2 }),
    image(
      'pdp-motion-gif', 118, 235, 1044, 377,
      '../mpc-detection-to-bounce-count/pdp-dynamics.gif',
      { fit: 'contain', radius: 4, alt: 'Power-delay profile evolving as the user equipment moves' }
    ),
    text('pdp-motion-caption', 118, 615, 1044, 16, 'UE motion → delays drift · gains change · paths enter and leave visibility', 11, { color: C.measurementDeep, fontFamily: MONO, fontWeight: 700, align: 'center' })
  ], { accent: C.measurement, transition: 'none' }
))

slides.push(regular(
  's-roadmap', 'TEACHING MAP', 'Three knowledge regimes, one recurring question',
  'Every section asks which route hypotheses remain physically feasible.',
  'Use this as the table of contents. Each topic link goes to an introductory slide, and the next slide activates the corresponding live geometry automatically.',
  [
    ...[
      ['02', 'Known BS/UE pose + map', C.known, C.knownSoft, 's-known'],
      ['03', 'Known BS/UE pose, unknown map', C.map, C.mapSoft, 's-map'],
      ['04', 'Known BS pose, unknown UE + map', C.pose, C.poseSoft, 's-pose']
    ].flatMap((item, index) => {
      const y = 226 + index * 130
      const target = sectionUnits.some(entry => `s-${entry.id}` === item[4]) ? item[4] : undefined
      return [
        card(`road-card-${index}`, 96, y, 1088, 104, item[3], { stroke: item[2], strokeWidth: 2 }),
        text(`road-num-${index}`, 124, y + 32, 66, 40, item[0], 28, { color: item[2], fontFamily: MONO, fontWeight: 700, valign: 'middle' }),
        text(`road-head-${index}`, 216, y + 27, 920, 50, item[1], 28, { fontWeight: 700, valign: 'middle', link: target })
      ]
    })
  ], { accent: C.pose }
))

for (const unit of sectionUnits) {
  const introId = `s-${unit.id}`
  const liveId = `${introId}-live`
  slides.push(regular(introId, unit.section, unit.title, unit.introSubtitle, unit.note, sectionIntroElements(unit), { accent: unit.accent, titleSize: 35, transition: 'none' }))
  slides.push(regular(
    liveId, unit.section, `${unit.title}: all cases`, `Experiment · ${unit.observe}`,
    `Consolidated live experiment for ${unit.title}. Select any case tile, use its controls, then press Escape to return focus to Bento or Page Up to revisit the section concept slide.`,
    [...sectionLiveFallback(unit), liveMount()], { accent: unit.accent, titleSize: 31, transition: 'none' }
  ))
}

slides.push(regular(
  's-takeaway', 'TAKEAWAY', 'Bounce count is a model-selection result',
  'The same MPC tuple means different things under different map and pose knowledge.',
  'Close on the hierarchy: direct route testing when geometry is known, recursive landmark construction when only poses are known, and joint pose-map families when the UE heading is also unknown.',
  [
    ...[
      ['Known map', 'test ordered routes', 'finite-wall feasibility + tuple match', C.known, C.knownSoft],
      ['Unknown map', 'construct landmarks', 'VA walk + prefix-path peeling', C.map, C.mapSoft],
      ['Unknown UE + map', 'retain families', 'heading slices + rank/null diagnostics', C.pose, C.poseSoft]
    ].flatMap((item, index) => {
      const x = 96 + index * 368
      return [
        card(`take-card-${index}`, x, 230, 344, 274, item[4], { stroke: item[3], strokeWidth: 2 }),
        text(`take-k-${index}`, x + 24, 256, 296, 24, item[0].toUpperCase(), 11, { color: item[3], fontFamily: MONO, fontWeight: 700, letterSpacing: 1.2 }),
        text(`take-head-${index}`, x + 24, 304, 296, 70, item[1], 30, { fontWeight: 700 }),
        text(`take-copy-${index}`, x + 24, 400, 296, 62, item[2], 17, { color: C.soft, fontFamily: SANS, lineHeight: 1.4 })
      ]
    }),
    text('take-final', 96, 554, 1088, 60, 'Never manufacture a unique route when the geometry supports a family.', 27, { color: C.poseDeep, fontWeight: 700, align: 'center' })
  ], { accent: C.pose }
))

function liveUrl(unit, embed) {
  const params = new URLSearchParams({ section: unit.mode === 'known' ? 'known' : unit.mode === 'map' ? 'unknown-map' : 'unknown-pose-map', case: unit.defaultCase })
  if (embed) params.set('embed', 'section')
  return `../mpc-detection-to-bounce-count/?${params.toString()}`
}

const inlineLiveMap = sectionUnits.map(unit => {
  const introSlide = `s-${unit.id}`, slide = `${introSlide}-live`
  return {
    introSlide, slide, slideIndex: slides.findIndex(entry => entry.id === slide), inline: true, layout: 'region', bounds: LIVE_BOUNDS,
    src: liveUrl(unit, true), source: liveUrl(unit, false), title: `${unit.section} · consolidated cases`,
    sandbox: 'allow-scripts', hideSource: true, readyMessage: true, unloadWhenHidden: true
  }
})

const deck = {
  format: 'bento/slides', version: 1, docId: 'mpc-detection-to-bounce-count-deck',
  title: 'MPC Detection to Bounce Count', readonly: true,
  meta: { author: 'Bai Liping', subject: 'Multipath geometry, virtual anchors, incidence points, and bounce-count inference', company: 'bailiping.com', source: 'bailiping.com/mpc-detection-to-bounce-count' },
  size: { width: 1280, height: 720 }, theme: { background: C.bg, color: C.ink, accent: C.pose, fontFamily: SERIF }, slides
}

const serializedDeck = JSON.stringify(deck, null, 1).replaceAll('<', '\\u003c')
const serializedMap = JSON.stringify(inlineLiveMap, null, 2).replaceAll('<', '\\u003c')
let html = readFileSync(templatePath, 'utf8')
html = html.replace('<title>bento/slides</title>', '<title>MPC Detection to Bounce Count | Interactive Slides</title>')
html = html.replace(/(<script type="application\/bento\+json" id="bento-doc">\s*)[\s\S]*?(\s*<\/script>)/, `$1${serializedDeck}$2`)
html = html.replace(/<script type="application\/json" id="(?:bento-live-config|bento-inline-live-map)">[\s\S]*?<\/script>/, `<script type="application/json" id="bento-inline-live-map">\n${serializedMap}\n    </script>`)
html = html.replaceAll('../assets/bento-live.css', '../assets/bento-inline-live.css')
html = html.replaceAll('../assets/bento-live.js', '../assets/bento-inline-live.js')

if (!html.includes('"docId": "mpc-detection-to-bounce-count-deck"')) throw new Error('Bento document replacement failed')
if (!html.includes('id="bento-inline-live-map"')) throw new Error('Inline-live map replacement failed')
writeFileSync(outputPath, html)
console.log(`Wrote ${outputPath} with ${slides.length} regular slides and ${inlineLiveMap.length} consolidated paired inline demo.`)
