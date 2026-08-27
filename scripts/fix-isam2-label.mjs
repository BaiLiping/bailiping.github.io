import { readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

const sourcePath = resolve('mpc-detection-to-bounce-count-slides/radio-slam-extra.mjs')
let source = readFileSync(sourcePath, 'utf8')

const before = 'SAME POSTERIOR AS BATCH BA / GRAPHSAM'
const after = 'SAME POSTERIOR AS BATCH BA / GRAPHSLAM'

if (source.includes(before)) {
  source = source.replaceAll(before, after)
  writeFileSync(sourcePath, source)
  console.log(`Corrected iSAM2 bridge label in ${sourcePath}`)
} else if (source.includes(after)) {
  console.log(`iSAM2 bridge label already correct in ${sourcePath}`)
} else {
  throw new Error('Could not find the iSAM2 batch-objective label')
}
