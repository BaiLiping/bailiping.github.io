import { readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

const path = resolve('scripts/upgrade-radio-slam-formulation-3d.mjs')
let source = readFileSync(path, 'utf8')
const original = source

source = source.replaceAll(
  '\\mathbf r^{\\mathrm{rad}}_{ts\\ell}(h)',
  '\\mathbf r^{\\mathrm{rad}}_{ts\\ell}(\\mathbf x_t,\\mathcal M,h)'
)
source = source.replaceAll(
  '\\mathbf r^{\\mathrm{rad}}_{ts\\ell}(a_{ts\\ell})',
  '\\mathbf r^{\\mathrm{rad}}_{ts\\ell}(\\mathbf x_t,\\mathcal M,a_{ts\\ell})'
)

for (const required of [
  '\\mathbf r^{\\mathrm{rad}}_{ts\\ell}(\\mathbf x_t,\\mathcal M,h)\\sim\\mathcal N',
  '\\|\\mathbf r^{\\mathrm{rad}}_{ts\\ell}(\\mathbf x_t,\\mathcal M,a_{ts\\ell})\\|',
  '\\mathbf r^{\\mathrm{rad}}_{ts\\ell}(\\mathbf x_t,\\mathcal M,h);\\mathbf0'
]) {
  if (!source.includes(required)) throw new Error(`Residual-notation validation failed: ${required}`)
}

if (source.includes('\\mathbf r^{\\mathrm{rad}}_{ts\\ell}(h)')) {
  throw new Error('Legacy state-suppressed 3D residual notation remains')
}

if (source !== original) {
  writeFileSync(path, source)
  console.log('Normalized all 3D residual references to show UE state, complete map, and path hypothesis.')
} else {
  console.log('The 3D residual notation is already fully map-conditioned.')
}
