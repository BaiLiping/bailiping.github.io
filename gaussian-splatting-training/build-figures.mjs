import {mkdirSync, writeFileSync} from 'node:fs';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import assert from 'node:assert/strict';
import {createTraining, evaluateTraining, stepTraining} from './live/model.mjs';

// Scientific figures, computed from the same model used by the live laboratory.
// No external libraries, image downloads, or pre-rendered training snapshots.
const here = dirname(fileURLToPath(import.meta.url));
const output = resolve(here, 'figures');
mkdirSync(output, {recursive: true});

const C = {
  cream: '#F5F3ED', white: '#FFFDFA', ink: '#182D33', muted: '#60747A',
  teal: '#16736E', rust: '#B44F37', line: '#D6DEDC', pale: '#E4F0ED',
};
const esc = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const fmt = value => value.toFixed(6);
const short = value => Number(value.toFixed(3));
const text = (x, y, value, size = 15, fill = C.ink, extra = '') =>
  `<text x="${x}" y="${y}" fill="${fill}" font-family="Arial, Helvetica, sans-serif" font-size="${size}" ${extra}>${esc(value)}</text>`;
const svg = (width, height, title, description, content) =>
  `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title description"><title id="title">${esc(title)}</title><desc id="description">${esc(description)}</desc><rect width="${width}" height="${height}" fill="${C.cream}"/>${content}</svg>\n`;

function pixelImage(image, size, x, y, width, id) {
  const cell = width / size;
  const rects = [];
  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      const at = 3 * (row * size + col);
      const rgb = Array.from(image.slice(at, at + 3), value => Math.round(255 * Math.max(0, Math.min(1, value))));
      rects.push(`<rect x="${short(x + col * cell)}" y="${short(y + row * cell)}" width="${cell}" height="${cell}" fill="rgb(${rgb.join(',')})"/>`);
    }
  }
  return `<defs><clipPath id="${id}"><rect x="${x}" y="${y}" width="${width}" height="${width}" rx="7"/></clipPath></defs><g clip-path="url(#${id})" shape-rendering="crispEdges">${rects.join('')}</g><rect x="${x}" y="${y}" width="${width}" height="${width}" rx="7" fill="none" stroke="${C.line}"/>`;
}

const training = createTraining({size: 32});
const initial = evaluateTraining(training);
const after = stepTraining(training, {steps: 200});
assert.equal(training.iteration, 200);
assert.ok(after.loss < initial.loss * 0.01, 'The figure must show actual convergence.');
assert.ok(after.image.every(Number.isFinite));

const columns = [
  {x: 20, title: 'Target image', image: training.target, caption: 'Fixed supervision', detail: '32 × 32 RGB pixels', accent: C.ink},
  {x: 201, title: 'Initial splats', image: initial.image, caption: `Loss ${fmt(initial.loss)}`, detail: `SSIM ${fmt(initial.ssim)}`, accent: C.rust},
  {x: 382, title: 'After 200 steps', image: after.image, caption: `Loss ${fmt(after.loss)}`, detail: `SSIM ${fmt(after.ssim)}`, accent: C.teal},
];
const trainingBody = columns.map((column, i) =>
  text(column.x + 80, 30, column.title, 17, column.accent, 'text-anchor="middle" font-weight="700"') +
  pixelImage(column.image, 32, column.x, 54, 160, `image-${i}`) +
  text(column.x + 80, 243, column.caption, 16, column.accent, 'text-anchor="middle" font-weight="700"') +
  text(column.x + 80, 266, column.detail, 13, C.muted, 'text-anchor="middle"')
).join('') +
  `<line x1="20" y1="283" x2="542" y2="283" stroke="${C.line}"/>` +
  text(281, 306, 'Five splats, with L = 0.8 L1 + 0.2 (1 − SSIM)', 15, C.ink, 'text-anchor="middle"');
const trainingSVG = svg(562, 322, 'Actual Gaussian image fitting over 200 Adam steps',
  `Three vector images show the fixed target, initial five-splat render, and fitted render. All are 32 by 32 RGB images. The computed loss falls from ${fmt(initial.loss)} to ${fmt(after.loss)}. SSIM increases from ${fmt(initial.ssim)} to ${fmt(after.ssim)}.`,
  trainingBody);
writeFileSync(resolve(output, 'training.svg'), trainingSVG);

// One scalar L1 update. The target exists only at p=1; there is no asserted
// target profile at the other pixel positions shown on the horizontal axis.
const p = 1, target = 0.8, sigma = 1, amplitude = 0.8, mu0 = 0, rate = 0.1;
const profile = (position, center) => amplitude * Math.exp(-0.5 * ((position - center) / sigma) ** 2);
const before = profile(p, mu0);
const derivative = Math.sign(before - target) * before * (p - mu0) / sigma ** 2;
const mu1 = mu0 - rate * derivative;
const afterPixel = profile(p, mu1);
const errorBefore = Math.abs(before - target), errorAfter = Math.abs(afterPixel - target);
assert.ok(Math.abs(before - 0.4852245277701068) < 1e-14);
assert.ok(Math.abs(afterPixel - 0.5087501167461241) < 1e-14);
assert.ok(errorAfter < errorBefore);

const plot = {x: 54, y: 89, w: 326, h: 207, xmin: -2.3, xmax: 2.3, ymax: 0.9};
const px = x => plot.x + (x - plot.xmin) / (plot.xmax - plot.xmin) * plot.w;
const py = y => plot.y + plot.h * (1 - y / plot.ymax);
const curve = center => Array.from({length: 301}, (_, i) => {
  const position = plot.xmin + (plot.xmax - plot.xmin) * i / 300;
  return `${i ? 'L' : 'M'}${short(px(position))},${short(py(profile(position, center)))}`;
}).join(' ');

let plotBody = text(22, 29, 'One pixel pulls the center right', 22, C.ink, 'font-weight="700"');
plotBody += `<line x1="25" y1="57" x2="51" y2="57" stroke="${C.rust}" stroke-width="2.7" stroke-dasharray="6 4"/>`;
plotBody += text(58, 62, 'Before', 13, C.rust);
plotBody += `<line x1="128" y1="57" x2="154" y2="57" stroke="${C.teal}" stroke-width="2.7"/>`;
plotBody += text(161, 62, 'After one step', 13, C.teal);
plotBody += `<circle cx="288" cy="57" r="4" fill="${C.ink}"/>`;
plotBody += text(299, 62, 'Target pixel', 13, C.ink);
plotBody += `<rect x="${plot.x}" y="${plot.y}" width="${plot.w}" height="${plot.h}" fill="${C.white}"/>`;
for (const value of [0, 0.4, 0.8]) {
  plotBody += `<line x1="${plot.x}" y1="${short(py(value))}" x2="${plot.x + plot.w}" y2="${short(py(value))}" stroke="${C.line}"/>`;
  plotBody += text(plot.x - 9, short(py(value) + 5), value.toFixed(1), 13, C.muted, 'text-anchor="end"');
}
for (const value of [-2, -1, 0, 1, 2]) {
  plotBody += `<line x1="${short(px(value))}" y1="${plot.y + plot.h}" x2="${short(px(value))}" y2="${plot.y + plot.h + 4}" stroke="${C.muted}"/>`;
  plotBody += text(short(px(value)), plot.y + plot.h + 20, value, 13, value === 1 ? C.ink : C.muted, `text-anchor="middle"${value === 1 ? ' font-weight="700"' : ''}`);
}
plotBody += text(plot.x, 82, 'Intensity C', 12, C.muted);
plotBody += text(plot.x + plot.w, plot.y + plot.h + 20, 'p', 14, C.ink, 'text-anchor="end" font-style="italic"');
plotBody += `<line x1="${short(px(p))}" y1="${short(py(target))}" x2="${short(px(p))}" y2="${plot.y + plot.h}" stroke="${C.muted}" stroke-dasharray="3 4"/>`;
plotBody += `<path d="${curve(mu0)}" fill="none" stroke="${C.rust}" stroke-width="2.7" stroke-dasharray="6 4"/><path d="${curve(mu1)}" fill="none" stroke="${C.teal}" stroke-width="2.7"/>`;
plotBody += `<circle cx="${short(px(p))}" cy="${short(py(target))}" r="4.2" fill="${C.ink}"/><circle cx="${short(px(p))}" cy="${short(py(before))}" r="4" fill="${C.white}" stroke="${C.rust}" stroke-width="2"/><circle cx="${short(px(p))}" cy="${short(py(afterPixel))}" r="3.3" fill="${C.teal}"/>`;
plotBody += text(411, 91, 'At p = 1', 15, C.ink, 'font-weight="700"');
const readouts = [
  {label: 'Target', value: target, y: 121, color: C.ink},
  {label: 'Before', value: before, y: 184, color: C.rust},
  {label: 'After', value: afterPixel, y: 247, color: C.teal},
];
for (const readout of readouts) {
  plotBody += text(411, readout.y, readout.label, 13, readout.color);
  plotBody += text(411, readout.y + 23, fmt(readout.value), 21, readout.color, 'font-weight="700"');
}
plotBody += text(54, 337, 'The target pixel stays fixed. Only μ moves.', 13, C.muted);
plotBody += `<rect x="22" y="351" width="518" height="76" rx="8" fill="${C.white}" stroke="${C.line}"/>`;
plotBody += text(37, 376, `∂L/∂μ = ${fmt(derivative)}     η = 0.1`, 17, C.ink);
plotBody += text(37, 400, `μ₀ = 0     μ₁ = ${fmt(mu1)}`, 17, C.teal, 'font-weight="700"');
plotBody += text(37, 420, `L1 error: ${fmt(errorBefore)} before, ${fmt(errorAfter)} after`, 13, C.muted);
plotBody += text(281, 444, 'C(p; μ) = 0.8 exp[−(p − μ)² / 2],  σ = 1', 12, C.muted, 'text-anchor="middle"');
const pixelSVG = svg(562, 448, 'A scalar L1 update moves a Gaussian center toward a bright target pixel',
  `The two curves are C(p; mu)=0.8 exp(-(p-mu)^2/2). At pixel p=1, the fixed target intensity is 0.8. The initial center is zero and prediction ${fmt(before)}. The L1 center derivative is ${fmt(derivative)}. One step of size 0.1 moves the center to ${fmt(mu1)}, increasing the prediction to ${fmt(afterPixel)} and decreasing absolute error from ${fmt(errorBefore)} to ${fmt(errorAfter)}. The plotted displacement is not exaggerated.`,
  plotBody);
writeFileSync(resolve(output, 'pixel-step.svg'), pixelSVG);

for (const [name, source] of [['training.svg', trainingSVG], ['pixel-step.svg', pixelSVG]]) {
  assert.ok(!/NaN|Infinity|undefined/.test(source), `${name} contains invalid numerical output.`);
  assert.ok(source.includes('<title') && source.includes('<desc'), `${name} must have accessible explanatory text.`);
}
assert.equal((trainingSVG.match(/fill="rgb\(/g) || []).length, 3 * 32 * 32);
console.log(JSON.stringify({
  figures: ['figures/training.svg', 'figures/pixel-step.svg'],
  training: {steps: training.iteration, initialLoss: initial.loss, finalLoss: after.loss, finalSSIM: after.ssim},
  scalarStep: {before, derivative, rate, mu1, after: afterPixel, errorBefore, errorAfter},
}, null, 2));
