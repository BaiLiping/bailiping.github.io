import {readFileSync,writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {dirname,resolve} from 'node:path';
import assert from 'node:assert/strict';
import {deck,labs} from './deck.mjs';
const here=dirname(fileURLToPath(import.meta.url)),safe=x=>JSON.stringify(x,null,1).replaceAll('<','\\u003c');
const ids=deck.slides.map(s=>s.id);assert.equal(new Set(ids).size,ids.length);
for(const s of deck.slides){
 assert.ok(s.notes);assert.equal(new Set(s.elements.map(e=>e.id)).size,s.elements.length);
 for(const e of s.elements){assert.ok(e.x>=0&&e.y>=0&&e.x+e.w<=1280&&e.y+e.h<=720,`Out of bounds: ${s.id}/${e.id}`);if(e.link&&!e.link.includes(':'))assert.ok(ids.includes(e.link));}
}
assert.equal(labs.length,5);
const template=readFileSync(resolve(here,'../mpc-detection-to-bounce-count-slides/index.html'),'utf8');
const scripts=[...template.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)];
const runtime=scripts.filter(m=>/id="bento-rt(?:-css)?"/.test(m[1])||m[2].includes("new DecompressionStream('deflate-raw')")).map(m=>m[0]).join('\n');
assert.ok(runtime.includes('bento-rt-css')&&runtime.includes('DecompressionStream'));
const license=template.match(/<!--\s*NOTICE — bento\/slides[\s\S]*?-->/)?.[0];assert.ok(license);
const routes=Object.fromEntries(deck.slides.map((s,i)=>[s.id,i]));
const math=`<link rel="stylesheet" href="../grassmannian/math.css"><script>window.MathJax={tex:{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]},svg:{fontCache:'local'},startup:{typeset:false}};</script><script defer src="../mpc-detection-to-bounce-count-slides/assets/vendor/mathjax-3.2.2-tex-svg-full.js"></script><script defer src="../assets/mathjax-dynamic.js"></script>`;
// The local dynamic typesetter reads the literal TeX delimiters in our source.
const metadata=`<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="only light"><meta name="grassmannian-revision" content="2026-10-03-affine-association"><title>Grassmannian Manifold | Bai Liping</title><link rel="canonical" href="https://bailiping.com/grassmannian-slides/"><meta name="description" content="${deck.slides.length} interactive Grassmannian manifold slides with five live experiments, including affine Grassmannian data association from Lusk and How (2022)."><link rel="icon" href="../grassmannian/figures/hero.svg">`;
const route=`<script>(()=>{const routes=${JSON.stringify(routes)};function route(){const id=location.hash.replace(/^#\\/?/,'');if(Object.hasOwn(routes,id))history.replaceState(null,'',location.pathname+location.search+'#/'+routes[id]);}addEventListener('hashchange',route);route();})();</script>`;
const dialog=`<dialog id="lab-dialog" aria-labelledby="lab-title"><header><button id="lab-back" type="button">← Back to slides <span>Esc</span></button><h2 id="lab-title"></h2><a id="lab-full" href="../grassmannian/">Full lab ↗</a></header><div id="lab-stage"><p id="lab-status" role="status">Loading the experiment…</p></div></dialog>`;
const fallback='<noscript><main style="max-width:800px;margin:60px auto;padding:24px;font:22px/1.5 system-ui"><h1>Grassmannian manifold</h1><p>Read the complete presentation in the <a href="print.html">static slide edition</a>, or download the <a href="grassmannian.pdf">PDF</a>. The five live experiments require JavaScript.</p></main></noscript>';
writeFileSync(resolve(here,'index.html'),`<!doctype html><html lang="en"><head>${metadata}\n${license}\n<link rel="stylesheet" href="deck.css">${math}${route}\n<script type="application/bento+json" id="bento-doc">${safe(deck)}</script></head><body><div id="app"></div>${fallback}${dialog}<script type="application/json" id="grassmann-labs">${safe(labs)}</script>${runtime}<script defer src="deck-ui.js"></script></body></html>\n`);

function escape(s){return String(s).replaceAll('&','&amp;').replaceAll('"','&quot;').replaceAll('<','&lt;');}
function el(e){
 if(e.id.startsWith('try-'))return '';
 const pos=`left:${e.x}px;top:${e.y}px;width:${e.w}px;height:${e.h}px;`;
 if(e.type==='shape')return `<div style="${pos}background:${e.fill};border:${e.strokeWidth||0}px solid ${e.stroke};border-radius:${e.radius||0}px"></div>`;
 if(e.type==='image')return `<img style="${pos}object-fit:contain" src="${escape(e.src)}" alt="${escape(e.alt||'')}">`;
 if(e.type==='text')return `<div data-print-el="${e.id}" style="${pos}font-family:${escape(e.fontFamily)};font-size:${e.fontSize}px;font-weight:${e.fontWeight};line-height:${e.lineHeight};color:${e.color};text-align:${e.align};">${e.link?`<a href="${e.link.includes(':')?escape(e.link):'#'+escape(e.link)}">${e.html}</a>`:e.html}</div>`;
 throw Error('Unsupported print element '+e.type);
}
const pages=deck.slides.map(s=>`<article class="print-page" id="${s.id}"><div class="print-canvas" style="background:${s.background}">${s.elements.map(el).join('')}</div></article>`).join('\n');
const printCss=`*{box-sizing:border-box}body{margin:0;background:#dfe4df;color:#213d36}.print-toolbar{margin:20px auto;text-align:center;font:15px system-ui}.print-toolbar a{color:#137b69;margin:0 16px}.print-page{width:calc(1280px * var(--preview-scale,1));height:calc(720px * var(--preview-scale,1));margin:20px auto;position:relative;overflow:hidden}.print-canvas{position:relative;width:1280px;height:720px;transform:scale(var(--preview-scale,1));transform-origin:top left}.print-canvas>div,.print-canvas>img{position:absolute}.print-canvas a{color:inherit;text-decoration:none}.math-display{display:block;min-height:0;margin:.5em 0;line-height:1.1}.math-display:first-child{margin-top:0}mjx-container[display=true]{margin:.4em 0!important}@page{size:1280px 720px;margin:0}@media print{body{background:white}.print-toolbar{display:none}.print-page{width:1280px;height:720px;margin:0;break-after:page;page-break-after:always}.print-page:last-child{break-after:auto;page-break-after:auto}.print-canvas{transform:none}}`;
writeFileSync(resolve(here,'print.html'),`<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex"><title>Grassmannian Manifold · Printable Slides</title><style>${printCss}</style>${math}</head><body><div class="print-toolbar"><a href="./">← Interactive deck</a><a href="grassmannian.pdf">Download PDF</a><button onclick="window.print()">Print slides</button></div>${pages}<script>function resize(){document.documentElement.style.setProperty('--preview-scale',Math.min(1,(innerWidth-24)/1280));}addEventListener('resize',resize);resize();</script></body></html>\n`);
writeFileSync(resolve(here,'deck.json'),JSON.stringify(deck,null,2)+'\n');
console.log(`Built ${deck.slides.length} slides, ${labs.length} labs, and a static print edition.`);
