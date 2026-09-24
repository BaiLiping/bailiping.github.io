import {readFileSync,writeFileSync} from 'node:fs';
import {dirname,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import assert from 'node:assert/strict';
import {deck,inlineLiveMap} from './bento-deck.mjs';
const here=dirname(fileURLToPath(import.meta.url));
const safe=x=>JSON.stringify(x,null,1).replaceAll('<','\\u003c');
const ids=deck.slides.map(s=>s.id);
assert.equal(new Set(ids).size,ids.length);
for(const s of deck.slides){
 assert.ok(s.notes,'Missing slide notes');
 assert.equal(new Set(s.elements.map(e=>e.id)).size,s.elements.length,'Duplicate element ID');
 for(const e of s.elements){
  assert.ok(e.x>=0&&e.y>=0&&e.x+e.w<=1280&&e.y+e.h<=720,`Outside slide: ${s.id}/${e.id}`);
  if(e.link&&!e.link.includes(':'))assert.ok(ids.includes(e.link),'Invalid slide target '+e.link);
 }
}
for(const e of inlineLiveMap){
 assert.ok(ids.includes(e.slide)&&ids.includes(e.introSlide));
 assert.equal(ids.indexOf(e.slide),e.slideIndex);
 assert.equal(ids.indexOf(e.introSlide),e.slideIndex-1);
}
let html=readFileSync(resolve(here,'../mpc-detection-to-bounce-count-slides/index.html'),'utf8');
html=html.replace(/(<script type="application\/bento\+json" id="bento-doc">\s*)[\s\S]*?(\s*<\/script>)/,(_,a,b)=>a+safe(deck)+b);
html=html.replace(/(<script type="application\/json" id="bento-inline-live-map">\s*)[\s\S]*?(\s*<\/script>)/,(_,a,b)=>a+safe(inlineLiveMap)+b);
html=html.replace(/<title>[\s\S]*?<\/title>/,'<title>Cramér–Rao Bounds for Radio Measurements | Bai Liping</title>');
html=html.replace(/<link rel="canonical"[^>]*>/,'<link rel="canonical" href="https://bailiping.com/radio-cramer-rao-slides/">');
html=html.replace(/<meta name="description"[^>]*>\s*/,'');
html=html.replace('./assets/vendor/mathjax-3.2.2-tex-svg-full.js','../mpc-detection-to-bounce-count-slides/assets/vendor/mathjax-3.2.2-tex-svg-full.js');
const routes=Object.fromEntries(deck.slides.map((s,i)=>[s.id,i]));
html=html.replace('</head>',`<meta name="description" content="${deck.slides.length} interactive Bento slides and ${inlineLiveMap.length} live experiments explaining radio Cramér–Rao bounds, 384-port coded pilots, known channels, receiver clock bias, AoA, AoD and path gain.">
<meta name="radio-crb-revision" content="2026-09-24-coded-pilots-known-channel">
<style id="radio-crb-layout">
.bento-slide .math-display{height:auto!important;min-height:0;margin:.6em 0;line-height:1.1}
.bento-slide .math-display:first-child{margin-top:.1em}
.bento-slide .math-display:last-child{margin-bottom:.1em}
.bento-slide .math-inline{white-space:normal}
.bento-slide .bento-text-inner{overflow:visible;text-rendering:optimizeLegibility}
.bento-slide a:focus-visible{outline:3px solid #1874B8;outline-offset:3px}
</style>
<script>(()=>{const routes=${JSON.stringify(routes)};function route(){let r;try{r=decodeURIComponent(location.hash.replace(/^#\\/?/,''));}catch{return;}if(Object.hasOwn(routes,r))history.replaceState(null,'',location.pathname+location.search+'#/'+routes[r]);}addEventListener('hashchange',route);route();})();</script>
</head>`);
writeFileSync(resolve(here,'index.html'),html);
writeFileSync(resolve(here,'deck.json'),JSON.stringify(deck,null,2)+'\n');
writeFileSync(resolve(here,'live-demos.json'),JSON.stringify(inlineLiveMap,null,2)+'\n');
console.log(`Built ${deck.slides.length} slides and ${inlineLiveMap.length} live labs.`);
