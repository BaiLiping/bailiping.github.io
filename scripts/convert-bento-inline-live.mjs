import { readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const bounds = { x: 96, y: 180, width: 1088, height: 480 }

const decks = [
  {
    path: 'frame-registration-slides/index.html',
    demos: [
      {
        state: 'state-ransac-live',
        slide: 's-rung2',
        title: 'RANSAC: consensus over trust',
        source: './live/?demo=ransac',
        src: './live/?demo=ransac&embed=region'
      },
      {
        state: 'state-icp-live',
        slide: 's-rung3',
        title: 'ICP: initialization chooses the basin',
        source: './live/?demo=icp',
        src: './live/?demo=icp&embed=region'
      },
      {
        state: 'state-ndt-live',
        slide: 's-rung4',
        title: 'NDT: a smooth score with a finite basin',
        source: './live/?demo=ndt',
        src: './live/?demo=ndt&embed=region'
      }
    ]
  },
  {
    path: 'target-handover-slides/index.html',
    demos: [
      {
        state: 'state-handover-rule-live',
        slide: 's-criterion',
        title: 'Point-target handover: gates, timing, and messages',
        source: './live/',
        src: './live/?embed=region'
      },
      {
        state: 'state-handover-timeline-live',
        slide: 's-animation',
        title: 'Point-target handover timeline',
        source: './live/?view=timeline',
        src: './live/?view=timeline&embed=region'
      }
    ]
  }
]

function mountMarker() {
  return {
    id: 'live-demo-mount',
    type: 'shape',
    shape: 'rect',
    x: bounds.x,
    y: bounds.y,
    w: bounds.width,
    h: bounds.height,
    fill: 'rgba(255,255,255,0)',
    stroke: 'none',
    strokeWidth: 0,
    radius: 0,
    rotation: 0,
    opacity: 0
  }
}

function convertDeck(definition) {
  const path = join(root, definition.path)
  let html = readFileSync(path, 'utf8')
  const documentMatch = html.match(
    /<script type="application\/bento\+json" id="bento-doc">\s*([\s\S]*?)\s*<\/script>/
  )
  if (!documentMatch) throw new Error(`Missing Bento document in ${definition.path}`)

  const doc = JSON.parse(documentMatch[1])
  const removedStates = new Set(definition.demos.map(demo => demo.state))
  doc.slides = doc.slides.filter(slide => !removedStates.has(slide.id))

  for (const demo of definition.demos) {
    const slide = doc.slides.find(entry => entry.id === demo.slide)
    if (!slide) throw new Error(`Missing parent slide ${demo.slide} in ${definition.path}`)
    slide.elements = slide.elements
      .filter(element => element.id !== 'live-demo-mount' && !removedStates.has(element.link))
      .concat(mountMarker())
    const note = `Interactive region: ${demo.title}. The complete static slide remains underneath for print and offline use.`
    if (!slide.notes.includes(note)) slide.notes = `${slide.notes} ${note}`
  }

  const liveMap = definition.demos.map(demo => ({
    slide: demo.slide,
    slideIndex: doc.slides.findIndex(slide => slide.id === demo.slide),
    inline: true,
    layout: 'region',
    bounds,
    src: demo.src,
    source: demo.source,
    title: demo.title,
    sandbox: 'allow-scripts',
    hideSource: true,
    readyMessage: true,
    unloadWhenHidden: true
  }))

  const serializedDoc = JSON.stringify(doc, null, 1).replaceAll('<', '\\u003c')
  const serializedMap = JSON.stringify(liveMap, null, 2).replaceAll('<', '\\u003c')
  html = html.replace(
    /(<script type="application\/bento\+json" id="bento-doc">\s*)[\s\S]*?(\s*<\/script>)/,
    `$1${serializedDoc}$2`
  )
  html = html.replace(
    /<script type="application\/json" id="(?:bento-live-config|bento-inline-live-map)">[\s\S]*?<\/script>/,
    `<script type="application/json" id="bento-inline-live-map">\n${serializedMap}\n    </script>`
  )
  html = html.replaceAll('../assets/bento-live.css', '../assets/bento-inline-live.css')
  html = html.replaceAll('../assets/bento-live.js', '../assets/bento-inline-live.js')

  if (doc.slides.some(slide => slide.stateOf)) {
    throw new Error(`State slides remain in ${definition.path}`)
  }
  if (!html.includes('id="bento-inline-live-map"')) {
    throw new Error(`Inline map replacement failed in ${definition.path}`)
  }

  writeFileSync(path, html)
  console.log(`Converted ${definition.path}: ${doc.slides.length} regular slides, ${liveMap.length} inline demos.`)
}

decks.forEach(convertDeck)
