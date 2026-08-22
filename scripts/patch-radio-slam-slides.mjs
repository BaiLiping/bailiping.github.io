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

for (const marker of [
  importLine,
  'appendRadioSlamSlidesAfterSection(unit,',
  '...radioSlamLiveEntries({ slides, LIVE_BOUNDS })'
]) {
  if (!source.includes(marker)) throw new Error(`Patch validation failed: ${marker}`)
}

if (changed) {
  writeFileSync(buildPath, source)
  console.log(`Patched ${buildPath}`)
} else {
  console.log(`No changes needed in ${buildPath}`)
}
