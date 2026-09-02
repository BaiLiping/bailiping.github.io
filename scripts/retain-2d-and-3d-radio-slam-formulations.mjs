import { readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

const pagePath = resolve('mpc-detection-to-bounce-count/index.html')
const twoDSourcePath = resolve('scripts/add-radio-slam-problem-formulation.mjs')
const threeDSourcePath = resolve('scripts/upgrade-radio-slam-formulation-3d.mjs')

let html = readFileSync(pagePath, 'utf8')
const originalHtml = html

function extractRawTemplate(source, declaration, followingMarker) {
  const startMarker = `const ${declaration} = String.raw\``
  const start = source.indexOf(startMarker)
  if (start < 0) throw new Error(`Could not find template declaration: ${declaration}`)
  const contentStart = start + startMarker.length
  const end = source.indexOf(`\`\n\n${followingMarker}`, contentStart)
  if (end < 0) throw new Error(`Could not find end of template: ${declaration}`)
  return source.slice(contentStart, end)
}

function replaceRequired(text, before, after, label) {
  if (!text.includes(before)) throw new Error(`Could not find ${label}`)
  return text.replace(before, after)
}

function canonicalTwoDSection() {
  const source = readFileSync(twoDSourcePath, 'utf8')
  let section = extractRawTemplate(source, 'problemFormulationSection', 'insertBefore(')

  section = replaceRequired(
    section,
    '<!-- ============ 05 unified radio-SLAM problem formulation ============ -->',
    '<!-- ============ 05 simplified 2D radio-SLAM problem formulation ============ -->',
    '2D section marker'
  )
  section = replaceRequired(
    section,
    '<section class="sec companion-section" id="problem-formulation">',
    '<section class="sec companion-section" id="problem-formulation-2d" data-formulation-dimension="2d-simplified">',
    '2D section id'
  )
  section = replaceRequired(
    section,
    '<h2><span class="no">05</span>A complete radio-SLAM problem formulation</h2>',
    '<h2><span class="no">05</span>A simplified 2D radio-SLAM problem formulation</h2>',
    '2D title'
  )
  section = replaceRequired(
    section,
    '<p class="lede">This section collects the problem in one place. The base-station pose is known. The continuous unknowns are the UE trajectory and a persistent radio map. Each detected MPC also has a discrete explanation: clutter, LoS, one bounce on one surface, two ordered bounces on two surfaces, and so on.</p>',
    '<p class="lede">This is a pedagogical planar cross-section of the same radio-SLAM problem. It keeps only an \(x\)-\(y\) position, one heading angle, line-segment reflectors, and scalar bearings so that the LoS, one-bounce, and two-bounce geometry can be read without 3D manifold notation. The full 3D formulation in Section 06 is the model intended for implementation.</p>',
    '2D introductory scope'
  )
  section = replaceRequired(
    section,
    '<div class="accuracy"><strong>Recommended representation for this page.</strong> Use <em>physical wall variables</em> as the map state and compute virtual anchors and bounce points deterministically inside each radio factor. This preserves finite-wall support, AoD, visibility, and a shared physical surface across different BSs. A directly optimized VA is an excellent compact state for a one-bounce path, but a single composite-VA point is generally insufficient to identify the individual walls of a multi-bounce path.</div>',
    '<div class="accuracy"><strong>Scope of the simplification.</strong> This section assumes that all relevant ray segments and reflector normals lie in one plane. It is not a competing physical model: use it for intuition and for the 2D interactive diagrams above, then use the \(SE(3)\), planar-reflector, and spherical-direction model in Section 06 for the actual estimator.</div>',
    '2D scope callout'
  )

  const idUpdates = [
    ['#formulation-state', '#formulation-2d-state'],
    ['#formulation-hypothesis', '#formulation-2d-hypothesis'],
    ['#formulation-measurement', '#formulation-2d-measurement'],
    ['#formulation-map', '#formulation-2d-map'],
    ['id="formulation-state"', 'id="formulation-2d-state"'],
    ['id="formulation-hypothesis"', 'id="formulation-2d-hypothesis"'],
    ['id="formulation-measurement"', 'id="formulation-2d-measurement"'],
    ['id="formulation-map"', 'id="formulation-2d-map"']
  ]
  for (const [before, after] of idUpdates) section = section.replaceAll(before, after)

  section = section.replace(
    'aria-label="Radio SLAM problem-formulation subsections"',
    'aria-label="Simplified two-dimensional radio SLAM formulation subsections"'
  )
  section = section.replace(
    '<a href="#formulation-2d-state"><span>5.1</span><strong>Continuous state</strong><small>UE · known BS · wall · VA</small></a>',
    '<a href="#formulation-2d-state"><span>5.1</span><strong>Simplified state</strong><small>SE(2) · line wall · VA</small></a>'
  )
  section = section.replace(
    '<h3 class="subh"><span class="no">5.1</span>Continuous state: UE trajectory, known BS, and radio map</h3>',
    '<h3 class="subh"><span class="no">5.1</span>Continuous state in the simplified 2D model</h3>'
  )
  section = section.replace(
    '<p class="lede">The drawings on this page use a planar \\(SE(2)\\) model. The same construction extends to \\(SE(3)\\) by replacing headings with rotation matrices and wall segments with finite planes.</p>',
    '<p class="lede">The state below matches the interactive drawings: the UE and BS move or point in one plane, and each reflector is represented by a line segment. This removes elevation, roll, and pitch only to make the geometry easier to see.</p>'
  )
  section = section.replace(
    '<h3 class="subh"><span class="no">5.3</span>Unified measurement model for delay, AoA, AoD, and path loss</h3>',
    '<h3 class="subh"><span class="no">5.3</span>Simplified 2D measurement model</h3>'
  )
  section = section.replace(
    '<h3 class="subh"><span class="no">5.4</span>Factor graph, unknown association, and the joint MAP problem</h3>',
    '<h3 class="subh"><span class="no">5.4</span>Simplified 2D factor graph and joint MAP problem</h3>'
  )
  section = section.replace(
    '<div class="accuracy"><strong>3D extension.</strong> Replace \\(\\theta_t\\) and \\(\\theta_s\\) by \\(\\mathbf R_t,\\mathbf R_s\\in SO(3)\\). A finite rectangular surface needs a center \\(\\mathbf c_j\\), unit normal \\(\\mathbf n_j\\in\\mathbb S^2\\), an in-plane orientation, length, and width. Center, normal, length, and width alone do not determine rotation about the normal.</div>',
    '<div class="accuracy"><strong>Embedding into the full model.</strong> Embed every 2D point in \(\mathbb R^3\), replace the scalar headings by rotations in \(SO(3)\), replace line reflectors by planes or finite planar patches, and replace scalar bearings by directions on \(\mathbb S^2\). Section 06 carries out exactly this lift.</div>'
  )

  return section.trim() + '\n\n'
}

function canonicalThreeDSection() {
  const source = readFileSync(threeDSourcePath, 'utf8')
  let section = extractRawTemplate(source, 'formulation3D', 'replaceBlock(')

  section = section.replace(
    '<!-- ============ 05 unified 3D radio-SLAM problem formulation ============ -->',
    '<!-- ============ 06 full 3D radio-SLAM problem formulation ============ -->'
  )
  section = section.replace(
    '<section class="sec companion-section" id="problem-formulation" data-formulation-dimension="3d">',
    '<section class="sec companion-section" id="problem-formulation-3d" data-formulation-dimension="3d">'
  )
  section = section.replace(
    '<h2><span class="no">05</span>A complete 3D radio-SLAM problem formulation</h2>',
    '<h2><span class="no">06</span>The full 3D radio-SLAM problem formulation</h2>'
  )

  const idUpdates = [
    ['#formulation-state', '#formulation-3d-state'],
    ['#formulation-hypothesis', '#formulation-3d-hypothesis'],
    ['#formulation-measurement', '#formulation-3d-measurement'],
    ['#formulation-map', '#formulation-3d-map'],
    ['id="formulation-state"', 'id="formulation-3d-state"'],
    ['id="formulation-hypothesis"', 'id="formulation-3d-hypothesis"'],
    ['id="formulation-measurement"', 'id="formulation-3d-measurement"'],
    ['id="formulation-map"', 'id="formulation-3d-map"'],
    ['<span>5.1</span>', '<span>6.1</span>'],
    ['<span>5.2</span>', '<span>6.2</span>'],
    ['<span>5.3</span>', '<span>6.3</span>'],
    ['<span>5.4</span>', '<span>6.4</span>'],
    ['<span class="no">5.1</span>', '<span class="no">6.1</span>'],
    ['<span class="no">5.2</span>', '<span class="no">6.2</span>'],
    ['<span class="no">5.3</span>', '<span class="no">6.3</span>'],
    ['<span class="no">5.4</span>', '<span class="no">6.4</span>']
  ]
  for (const [before, after] of idUpdates) section = section.replaceAll(before, after)

  section = section.replace(
    'The formal estimation problem is three-dimensional.',
    'This is the formal estimation problem used for the radio-SLAM system; the preceding 2D section is only a readable planar specialization.'
  )

  return section.trim() + '\n\n'
}

function findGraphSectionStart(document) {
  const sectionIndex = document.indexOf('<section class="sec companion-section" id="bistatic-graphslam">')
  if (sectionIndex < 0) throw new Error('Could not find the bistatic GraphSLAM section')
  const commentIndex = document.lastIndexOf('<!-- ============', sectionIndex)
  return commentIndex >= 0 ? commentIndex : sectionIndex
}

function removeExistingFormulationsAndInsertBoth(document, twoD, threeD) {
  const graphStart = findGraphSectionStart(document)
  const candidateIds = [
    'id="problem-formulation"',
    'id="problem-formulation-2d"',
    'id="problem-formulation-3d"'
  ]
  const candidates = candidateIds
    .map(id => document.indexOf(id))
    .filter(index => index >= 0 && index < graphStart)
  if (candidates.length === 0) {
    return document.slice(0, graphStart) + twoD + threeD + document.slice(graphStart)
  }
  const firstSection = Math.min(...candidates)
  const firstComment = document.lastIndexOf('<!-- ============', firstSection)
  const replaceStart = firstComment >= 0 ? firstComment : document.lastIndexOf('<section', firstSection)
  if (replaceStart < 0) throw new Error('Could not find the beginning of the existing formulation section')
  return document.slice(0, replaceStart) + twoD + threeD + document.slice(graphStart)
}

function normalizeTopNavigation(document) {
  const start = document.indexOf('  <nav class="topnav">')
  const end = document.indexOf('  </nav>', start)
  if (start < 0 || end < 0) throw new Error('Could not isolate the top navigation')
  let nav = document.slice(start, end + '  </nav>'.length)
  nav = nav.replace(/^\s*<a href="#(?:problem-formulation|problem-formulation-2d|problem-formulation-3d|bistatic-graphslam)">.*?<\/a>\s*$/gm, '')
  nav = nav.replace(/\n{3,}/g, '\n')
  const anchor = '    <a href="#unknown-pose-map">Unknown UE + map</a>'
  if (!nav.includes(anchor)) throw new Error('Could not find the navigation insertion point')
  nav = nav.replace(
    anchor,
    anchor + '\n' +
    '    <a href="#problem-formulation-2d">Simplified 2D</a>\n' +
    '    <a href="#problem-formulation-3d">Full 3D</a>\n' +
    '    <a href="#bistatic-graphslam">GraphSLAM bridge</a>'
  )
  return document.slice(0, start) + nav + document.slice(end + '  </nav>'.length)
}

function renumberGraphSlamSection(document) {
  document = document.replace(
    /<!-- ============ (?:05|06|07) bistatic radio to GraphSLAM ============ -->/,
    '<!-- ============ 07 bistatic radio to GraphSLAM ============ -->'
  )
  document = document.replace(
    /<h2><span class="no">(?:05|06|07)<\/span>Bistatic radio SLAM as GraphSLAM<\/h2>/,
    '<h2><span class="no">07</span>Bistatic radio SLAM as GraphSLAM</h2>'
  )
  for (let i = 1; i <= 4; i += 1) {
    for (const oldSection of [5, 6, 7]) {
      document = document.replaceAll(
        `<span>${oldSection}.${i}</span>`,
        `<span>7.${i}</span>`
      )
      document = document.replaceAll(
        `<span class="no">${oldSection}.${i}</span>`,
        `<span class="no">7.${i}</span>`
      )
    }
  }
  return document
}

const twoD = canonicalTwoDSection()
const threeD = canonicalThreeDSection()
html = removeExistingFormulationsAndInsertBoth(html, twoD, threeD)
html = renumberGraphSlamSection(html)
html = normalizeTopNavigation(html)

const required = [
  'id="problem-formulation-2d" data-formulation-dimension="2d-simplified"',
  'A simplified 2D radio-SLAM problem formulation',
  'id="formulation-2d-state"',
  'id="formulation-2d-hypothesis"',
  'id="formulation-2d-measurement"',
  'id="formulation-2d-map"',
  'id="problem-formulation-3d" data-formulation-dimension="3d"',
  'The full 3D radio-SLAM problem formulation',
  'id="formulation-3d-state"',
  'id="formulation-3d-hypothesis"',
  'id="formulation-3d-measurement"',
  'id="formulation-3d-map"',
  '<h2><span class="no">07</span>Bistatic radio SLAM as GraphSLAM</h2>',
  '<a href="#problem-formulation-2d">Simplified 2D</a>',
  '<a href="#problem-formulation-3d">Full 3D</a>'
]
for (const value of required) {
  if (!html.includes(value)) throw new Error(`Dual-formulation validation failed: ${value}`)
}

for (const id of [
  'problem-formulation-2d',
  'formulation-2d-state',
  'formulation-2d-hypothesis',
  'formulation-2d-measurement',
  'formulation-2d-map',
  'problem-formulation-3d',
  'formulation-3d-state',
  'formulation-3d-hypothesis',
  'formulation-3d-measurement',
  'formulation-3d-map'
]) {
  const count = html.split(`id="${id}"`).length - 1
  if (count !== 1) throw new Error(`Expected exactly one id=${id}; found ${count}`)
}

if (html.includes('id="problem-formulation"')) {
  throw new Error('Legacy unsuffixed problem-formulation id remains')
}

if (html !== originalHtml) {
  writeFileSync(pagePath, html)
  console.log('Kept the simplified 2D formulation and added the full 3D formulation beside it.')
} else {
  console.log('The simplified 2D and full 3D formulations are already synchronized.')
}
