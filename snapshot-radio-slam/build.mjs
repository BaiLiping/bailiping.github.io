import {readFileSync,writeFileSync} from 'node:fs';
import {dirname,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import assert from 'node:assert/strict';
import {deck,inlineLiveMap} from './bento-deck.mjs';

const here=dirname(fileURLToPath(import.meta.url));
const safe=value=>JSON.stringify(value,null,1).replaceAll('<','\\u003c');
const ids=deck.slides.map(slide=>slide.id);
assert.equal(deck.slides.length,25,'Keep the lesson focused on its 25-slide learning sequence.');
assert.equal(new Set(ids).size,ids.length,'Slide identifiers must be unique.');
for(const slide of deck.slides){
 assert.ok(slide.notes?.includes('Sources:'),'Every slide needs notes and source links.');
 assert.equal(new Set(slide.elements.map(element=>element.id)).size,slide.elements.length,'Duplicate element identifier.');
 for(const element of slide.elements){
  assert.ok(element.x>=0&&element.y>=0&&element.x+element.w<=1280&&element.y+element.h<=720,`Outside canvas: ${slide.id}/${element.id}`);
  if(element.link&&!element.link.includes(':'))assert.ok(ids.includes(element.link),'Invalid internal target '+element.link);
  if(element.id.endsWith('-body'))assert.ok(element.fontSize>=20,'Main panel text must remain readable.');
 }
}
assert.equal(inlineLiveMap.length,3);
for(const lab of inlineLiveMap){
 assert.ok(ids.includes(lab.slide)&&ids.includes(lab.introSlide));
 assert.equal(ids.indexOf(lab.slide),lab.slideIndex);
 assert.equal(ids.indexOf(lab.introSlide),lab.slideIndex-1,'Each lab must immediately follow its introduction.');
 assert.equal(lab.src,`./live/?lab=${new URLSearchParams(lab.src.split('?')[1]).get('lab')}&embed=1`);
 assert.deepEqual(lab.bounds,{x:72,y:180,width:1136,height:475});
}

// Reuse the same native Bento runtime as neighboring public decks. Preserve
// its license and its existing keyboard, touch, presenter and iframe support.
let html=readFileSync(resolve(here,'../mpc-detection-to-bounce-count-slides/index.html'),'utf8');
assert.ok(html.includes('id="bento-doc"')&&html.includes('id="bento-inline-live-map"'));
html=html.replace(/(<script type="application\/bento\+json" id="bento-doc">\s*)[\s\S]*?(\s*<\/script>)/,(_,a,b)=>a+safe(deck)+b);
html=html.replace(/(<script type="application\/json" id="bento-inline-live-map">\s*)[\s\S]*?(\s*<\/script>)/,(_,a,b)=>a+safe(inlineLiveMap)+b);
html=html.replace(/<title>[\s\S]*?<\/title>/,'<title>Snapshot Radio SLAM: Initial Pose and Clock | Bai Liping</title>');
html=html.replace(/<link rel="canonical"[^>]*>/,'<link rel="canonical" href="https://bailiping.com/snapshot-radio-slam/">');
html=html.replace(/<meta name="description"[^>]*>\s*/,'');
html=html.replace('./assets/vendor/mathjax-3.2.2-tex-svg-full.js','../mpc-detection-to-bounce-count-slides/assets/vendor/mathjax-3.2.2-tex-svg-full.js');
const routes=Object.fromEntries(deck.slides.map((slide,index)=>[slide.id,index]));
html=html.replace('</head>',`<meta name="description" content="25 Bento slides and three interactive labs explaining snapshot radio SLAM, initial UE pose and clock estimation, conditional least squares, SVD, orientation search, robust consensus and QAIC.">
<meta name="snapshot-radio-slam-revision" content="2026-09-25-unified-angle-delay-v2">
<style id="snapshot-radio-slam-layout">
.bento-slide .math-display{height:auto!important;min-height:0;margin:.55em 0;line-height:1.1}
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
console.log(`Built ${deck.slides.length} slides and ${inlineLiveMap.length} live laboratories.`);
