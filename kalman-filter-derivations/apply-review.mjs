import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
const here = dirname(fileURLToPath(import.meta.url));

// Idempotent migration. Refuse unexpected source rather than silently doing nothing.
function replaceOnce(source, before, after, label) {
  if (source.includes(after)) return source;
  if (!source.includes(before)) throw new Error(`Review migration: cannot locate ${label}`);
  if (source.indexOf(before) !== source.lastIndexOf(before)) throw new Error(`Ambiguous migration: ${label}`);
  return source.replace(before, after);
}

const deckPath = resolve(here, 'bento-deck.mjs');
let deck = await readFile(deckPath, 'utf8');
const importLine = "import { applyMathReview } from './math-review.mjs';\n";
if (!deck.includes(importLine)) deck = importLine + deck;
deck = replaceOnce(deck, 'export const deck = {',
  'applyMathReview(slides, { tex, texBlock, mathLines, mathParagraphs, muted, equationSheetSlide, C });\n\nexport const deck = {',
  'review hook');
await writeFile(deckPath, deck);

const appPath = resolve(here, 'live/app.js');
let app = await readFile(appPath, 'utf8');
app = replaceOnce(app,
  'Change the two Gaussian sources. The common posterior is shown once; an agreement check confirms that four derivations reach it.',
  'Change the two Gaussian sources. Four equivalent formulas agree numerically; this is a consistency check, not four independent derivations.',
  'scalar experiment description');
app = replaceOnce(app,
  'The gain is nearly aligned with the measurement normal; the prior supplies little cross-coordinate coupling.',
  'For this measurement direction, the gain is nearly parallel to the measurement normal. Alignment alone does not imply weak prior correlation.',
  'geometry interpretation');
app = replaceOnce(app,
  'for (let k = 0; k < LP.length; k += 1) value += W[row][k] * W[column][k]',
  'for (let k = 0; k < LP.length; k += 1) value = ops.round(value + ops.round(W[row][k] * W[column][k]))',
  'QR covariance reconstruction rounding');
app = replaceOnce(app,
  'const trueState = Array.from({ length: n }, () => normal(rng))\n    const z = H.map((row, i) => row.reduce((sum, value, j) => sum + value * trueState[j], 0) + Math.sqrt(Math.max(R[i][i], 1e-12)) * .18 * normal(rng))',
  '// Draw from the same Gaussian model used by every algebraic route.\n    const xi = Array.from({ length: n }, () => normal(rng))\n    const eta = Array.from({ length: m }, () => normal(rng))\n    const trueState = full.addVec(priorMean, full.matVec(LP, xi))\n    const z = full.addVec(full.matVec(H, trueState), full.matVec(LR, eta))',
  'consistent Gaussian measurement generation');
app = replaceOnce(app,
  'Increase conditioning and lower simulated significant digits. Exact identities can separate numerically even though the target estimator is unchanged.',
  'Toy rounding, not IEEE emulation. QR starts from covariance factors; other paths start from matrices. The reference is native double, not exact truth.',
  'arithmetic disclosure');
app = replaceOnce(app,
  'MAXIMUM DIFFERENCE FROM FULL-PRECISION COVARIANCE UPDATE',
  'DIFFERENCE FROM NATIVE-DOUBLE REFERENCE',
  'reference label');
app = replaceOnce(app,
  "[mathHtml(String.raw`\\lambda_{\\min}`), diagnostics.minEigen, tone(diagnostics.minEigen, 'eigen')]",
  "[mathHtml(String.raw`\\lambda_{\\min}(P_s)`), diagnostics.minEigen, tone(diagnostics.minEigen, 'eigen')]",
  'symmetric eigenvalue label');
app = replaceOnce(app,
  '<div class="eq-layout">\n          <div class="eq-grid" id="eq-grid"></div>',
  '<div class="eq-layout">\n          <p class="review-numerics-note">P_s = (P + Pᵀ)/2. QR symmetry is enforced by mirroring. Δ is the largest entrywise difference across the toy mean and covariance.</p>\n          <div class="eq-grid" id="eq-grid"></div>',
  'diagnostic explanation');
await writeFile(appPath, app);

const cssPath = resolve(here, 'live/styles.css');
let css = await readFile(cssPath, 'utf8');
if (!css.includes('.review-numerics-note')) {
  css += '\n/* Mathematical-review disclosure; compact enough for the embedded region. */\n.review-numerics-note { margin: 0; padding: 0 2px; font-size: 10px; line-height: 1.35; color: var(--soft, #b9c4d6); }\n';
  await writeFile(cssPath, css);
}
console.log('Applied mathematical review and numerical-demo corrections.');
