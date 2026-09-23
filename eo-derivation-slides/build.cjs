'use strict';

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {createRequire} = require('node:module');
const sourceDir = path.resolve(__dirname, '../eo-derivation/source');
const mathRequire = createRequire(path.join(sourceDir, 'package.json'));
const {mathjax} = mathRequire('mathjax-full/js/mathjax.js');
const {TeX} = mathRequire('mathjax-full/js/input/tex.js');
const {SVG} = mathRequire('mathjax-full/js/output/svg.js');
const {liteAdaptor} = mathRequire('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = mathRequire('mathjax-full/js/handlers/html.js');
const {AllPackages} = mathRequire('mathjax-full/js/input/tex/AllPackages.js');
const content = require('./content.cjs');
const appendix = require('./appendix.cjs');
const appendixExcerpt = fs.readFileSync(path.join(sourceDir, 'appendix-I-original.tex'));
if (crypto.createHash('sha256').update(appendixExcerpt).digest('hex') !== appendix.provenance.excerptSHA256) throw new Error('Appendix I source excerpt changed; review its transcription before rebuilding.');
const articlePath = path.resolve(__dirname, '../eo-derivation/index.html');
const article = fs.readFileSync(articlePath);
const sourceHash = crypto.createHash('sha256').update(article).digest('hex');
const expectedSourceHash = 'f7131e7eb0d02bc49e00f9156bb9c7abb48a8b0a276f4c72e08bf75416d89db7';
if (sourceHash !== expectedSourceHash) {
  throw new Error('The restored article has changed. Review content.cjs against it and update expectedSourceHash before rebuilding.');
}

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const mathdoc = mathjax.document('', {
  InputJax: new TeX({packages: AllPackages, macros: {
    ub: ['\\underline{\\boldsymbol{#1}}', 1],
    ob: ['\\overline{\\boldsymbol{#1}}', 1],
    tb: ['\\tilde{\\boldsymbol{#1}}', 1]
  }}),
  OutputJax: new SVG({fontCache: 'none'})
});
const esc = s => String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const mathCache = new Map();
function math(tex, display = true) {
  const key = display + '|' + tex;
  if (!mathCache.has(key)) {
    const html = adaptor.outerHTML(mathdoc.convert(tex, {display}));
    if (/data-mjx-error|<merror|Unknown command/.test(html)) throw new Error('Invalid equation: ' + tex);
    mathCache.set(key, '<span class="' + (display ? 'display-math' : 'inline-math') + '" role="math" aria-label="' + esc(tex) + '">' + html + '</span>');
  }
  return mathCache.get(key);
}
function render(html) {
  const parts = html.split('$');
  if (parts.length % 2 !== 1) throw new Error('Unpaired inline math delimiter: ' + html);
  return parts.map((part, i) => i % 2 ? math(part, false) : part).join('');
}
function equation(key) {
  if (!content.equations[key]) throw new Error('Unknown equation ' + key);
  return '<div class="eq" data-equation="' + key + '">' + math(content.equations[key]) + '</div>';
}
const figures = [];
function figure(slide) {
  if (!slide.figure) return '';
  const file = path.resolve(__dirname, '../eo-derivation/assets', slide.figure);
  const bytes = fs.readFileSync(file);
  figures.push({file:slide.figure, sha256:crypto.createHash('sha256').update(bytes).digest('hex')});
  const mime = path.extname(file) === '.svg' ? 'image/svg+xml' : 'image/png';
  return '<figure class="source-figure"><img src="data:' + mime + ';base64,' + bytes.toString('base64') + '" alt="' + esc(slide.alt || slide.caption) + '"><figcaption>' + esc(slide.caption) + '</figcaption></figure>';
}
function refs(keys) {
  return keys.map(key => {
    const ref = content.references[key];
    return '<a href="' + esc(ref.url) + '" title="' + esc(ref.title) + '">' + key + '</a>';
  }).join('');
}
const ids = new Set();
const slides = content.slides.map((s, i) => {
  if (ids.has(s.id)) throw new Error('Duplicate slide ID: ' + s.id);
  ids.add(s.id);
  if (!s.notes.trim()) throw new Error('Missing speaker notes: ' + s.id);
  const scenario = /^S([1-4])/.exec(s.chapter);
  const colorClass = s.section === 'grbp' ? ' scenario-grbp' : scenario ? ' scenario-' + scenario[1] : '';
  return '<section class="slide ' + esc(s.kind) + colorClass + (i ? '' : ' active') + '" id="' + esc(s.id) + '" aria-label="Slide ' + (i+1) + ': ' + esc(s.title) + '">' +
    '<div class="slide-top"><span class="kicker">' + esc(s.chapter) + '</span><span class="mark">JOINT PDF FACTORIZATION</span></div>' +
    '<h1>' + esc(s.title) + '</h1><div class="slide-content"><div class="panel left">' + render(s.left) + '</div>' +
    '<div class="panel right">' + s.equations.map(equation).join('') + render(s.right) + figure(s) + '</div></div>' +
    '<footer class="slide-footer"><span><a href="../eo-derivation/#' + esc(s.section) + '">Corresponding derivation</a> · ' + refs(s.refs) + '</span>' +
    '<span class="page-num">' + String(i+1).padStart(2,'0') + ' / ' + content.slides.length + '</span></footer>' +
    '<aside class="speaker-notes">' + esc(s.notes) + '</aside></section>';
}).join('\n');
const css = fs.readFileSync(path.join(sourceDir,'style.css'),'utf8') + '\n' + fs.readFileSync(path.join(__dirname,'deck.css'),'utf8');
const runtime = fs.readFileSync(path.join(__dirname,'runtime.js'),'utf8');
const html = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' +
  '<meta name="color-scheme" content="light"><meta name="referrer" content="no-referrer">' +
  '<link rel="canonical" href="https://bailiping.com/eo-derivation-slides/"><meta name="eo-source-sha256" content="' + sourceHash + '"><link rel="icon" href="data:,"><title>' + esc(content.title) + '</title><style>' + css + '</style></head>' +
  '<body class="deck"><main class="stage" aria-label="Presentation">' + slides + '</main>' +
  '<nav class="deck-controls" aria-label="Presentation controls"><button data-prev aria-label="Previous slide">←</button><span class="counter" aria-live="polite"></span>' +
  '<button data-next aria-label="Next slide">→</button><button data-overview>Overview · O</button><button data-notes>Notes · N</button>' +
  '<button data-full>Full screen · F</button><a href="slides.pdf">PDF</a><button data-print>Print</button></nav>' +
  '<dialog class="notes-dialog"><div class="dialog-heading"><h2>Speaker notes</h2><button data-close-notes aria-label="Close speaker notes">Close · Esc</button></div><p></p></dialog>' +
  '<dialog class="overview-dialog"><div class="dialog-heading"><h2>Scenarios and GrBP</h2><button data-close-overview aria-label="Close overview">Close · Esc</button></div><div class="overview-grid"></div></dialog>' +
  '<script>' + runtime + '</script></body></html>';
fs.writeFileSync(path.join(__dirname,'index.html'), html);
fs.writeFileSync(path.join(__dirname,'build.json'), JSON.stringify({
  article:'../eo-derivation/index.html',articleSHA256:sourceHash,sourceSnapshot:'2026-09-15 / 2d2dccc (S1–S4); 2026-09-18 Appendix I addition (GrBP)',
  appendix:appendix.provenance,
  slides:content.slides.length,renderedExpressions:mathCache.size,embeddedFigures:figures,
  externalRuntimeDependencies:0,articleModifiedByBuild:false
},null,2) + '\n');
if (!fs.readFileSync(articlePath).equals(article)) throw new Error('The build modified the restored article.');
console.log('Built ' + content.slides.length + ' slides; embedded ' + figures.length + ' source graphs and ' + mathCache.size + ' math expressions. Restored article unchanged.');
