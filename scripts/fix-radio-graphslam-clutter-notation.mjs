import { readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

const sourcePath = resolve('mpc-detection-to-bounce-count-slides/radio-slam-extra.mjs')
let source = readFileSync(sourcePath, 'utf8')

function replaceRequired(before, after, label) {
  if (source.includes(after)) return
  if (!source.includes(before)) throw new Error(`Could not find ${label}`)
  source = source.replace(before, after)
}

replaceRequired(
  String.raw`a_{t\ell}&\in\{0,1,\ldots,J\},&&q_{t\ell}\in\{\mathrm{LoS},1,2,\ldots\}`,
  String.raw`a_{t\ell}&\in\{0,1,\ldots,J\},&&q_{t\ell}\in\{\mathrm{LoS},1,2,\ldots\}\quad(a_{t\ell}>0)`,
  'path-class domain conditioned on a non-clutter association'
)

replaceRequired(
  String.raw`p(\mathbf z_{t\ell}\mid\mathbf T_t,\mathbf m_{a_{t\ell}},q_{t\ell},\mathbf b)`,
  String.raw`p(\mathbf z_{t\ell}\mid\mathbf T_t,\mathcal M,a_{t\ell},q_{t\ell},\mathbf b)`,
  'measurement likelihood that handles clutter without indexing m_0'
)

replaceRequired(
  String.raw`\sum\nolimits_{t,\ell}\rho\!\left(\|\mathbf r_{t\ell}^{\mathrm{rad}}(a_{t\ell},q_{t\ell})\|_{\Omega_{t\ell}^{\mathrm{rad}}}^{2}\right)`,
  String.raw`\sum\nolimits_{(t,\ell):a_{t\ell}>0}\rho\!\left(\|\mathbf r_{t\ell}^{\mathrm{rad}}(a_{t\ell},q_{t\ell})\|_{\Omega_{t\ell}^{\mathrm{rad}}}^{2}\right)`,
  'geometric radio residual restricted to associated MPCs'
)

replaceRequired(
  'Fixed A,Q → ordinary nonlinear least squares. Unknown A,Q → marginalize, maximize, or alternate association and continuous-state updates.',
  'Fixed A,Q → nonlinear least squares over associated MPCs; a=0 uses the clutter likelihood. Unknown A,Q → marginalize, maximize, or alternate association and state updates.',
  'clutter-likelihood explanation'
)

for (const marker of [
  String.raw`p(\mathbf z_{t\ell}\mid\mathbf T_t,\mathcal M,a_{t\ell},q_{t\ell},\mathbf b)`,
  String.raw`\sum\nolimits_{(t,\ell):a_{t\ell}>0}`,
  'a=0 uses the clutter likelihood'
]) {
  if (!source.includes(marker)) throw new Error(`Clutter-notation validation failed: ${marker}`)
}

writeFileSync(sourcePath, source)
console.log('Corrected the GraphSLAM clutter-association notation.')
