import {withAuthoring,authoringDocument} from '../scripts/authoring-build.mjs';
import {readFileSync,writeFileSync,existsSync} from 'node:fs';
import {dirname,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import assert from 'node:assert/strict';
import {deck,inlineLiveMap,PAPER,BASE} from './bento-deck.mjs';

const here=dirname(fileURLToPath(import.meta.url));
const safe=x=>JSON.stringify(x,null,1).replaceAll('<','\\u003c');
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const ids=deck.slides.map(s=>s.id);
assert.equal(ids.length,39);
assert.equal(new Set(ids).size,ids.length);
for(const s of deck.slides){
 assert.ok(s.notes.includes('Sources:'));
 assert.equal(new Set(s.elements.map(e=>e.id)).size,s.elements.length);
 for(const e of s.elements){
  assert.ok(e.x>=0&&e.y>=0&&e.x+e.w<=1280&&e.y+e.h<=720,`Outside slide: ${s.id}/${e.id}`);
  if(e.link&&!e.link.includes(':')&&!e.link.startsWith('.'))assert.ok(ids.includes(e.link),`Unknown slide ${e.link}`);
  if(e.type==='image')assert.ok(existsSync(resolve(here,e.src)),`Missing figure ${e.src}`);
 }
}
assert.equal(inlineLiveMap.length,3);
for(const l of inlineLiveMap){assert.equal(ids.indexOf(l.slide),l.slideIndex);assert.equal(ids.indexOf(l.introSlide),l.slideIndex-1);}

// Preserve the existing native Bento renderer and third-party license notices.
// Only project-owned content, routes and metadata change.
let html=readFileSync(resolve(here,'../mpc-detection-to-bounce-count-slides/index.html'),'utf8');
html=html.replace(/(<script type="application\/bento\+json" id="bento-doc">\s*)[\s\S]*?(\s*<\/script>)/,(_,a,b)=>a+safe(deck)+b);
html=html.replace(/(<script type="application\/json" id="bento-inline-live-map">\s*)[\s\S]*?(\s*<\/script>)/,(_,a,b)=>a+safe(inlineLiveMap)+b);
html=html.replace(/<title>[\s\S]*?<\/title>/,'<title>Gaussian Splatting · Loss, Training &amp; Joint Pose | Bai Liping</title>');
html=html.replace(/<link rel="canonical"[^>]*>/,`<link rel="canonical" href="${BASE}">`);
html=html.replace(/<meta name="description"[^>]*>\s*/,'');
html=html.replace('./assets/vendor/mathjax-3.2.2-tex-svg-full.js','../mpc-detection-to-bounce-count-slides/assets/vendor/mathjax-3.2.2-tex-svg-full.js');
const routes=Object.fromEntries(deck.slides.map((s,i)=>[s.id,i]));
const externalLinksScript=`(()=>{
 const external=el=>{const url=el?.dataset.link;return url&&/^https?:\\/\\//i.test(url)?url:null;};
 const decorate=()=>document.querySelectorAll('[data-link]').forEach(el=>{if(external(el)){el.setAttribute('role','link');el.setAttribute('tabindex','0');}});
 document.addEventListener('click',event=>{
  const el=event.target.closest?.('[data-link]');const url=external(el);if(!url)return;
  event.preventDefault();event.stopPropagation();
  if(event.ctrlKey||event.metaKey||event.shiftKey)window.open(url,'_blank','noopener');else location.assign(url);
 },true);
 document.addEventListener('keydown',event=>{if(event.key==='Enter'&&external(event.target)){event.preventDefault();event.target.click();}},true);
 new MutationObserver(decorate).observe(document.documentElement,{childList:true,subtree:true});decorate();
})();`;
const equationCSS=`.bento-slide .math-display{height:auto!important;min-height:0;margin:.48em 0;line-height:1.08}.bento-slide .math-display:first-child{margin-top:.1em}.bento-slide .math-display:last-child{margin-bottom:.1em}.bento-slide .bento-text-inner{overflow:visible;text-rendering:optimizeLegibility}.bento-slide p{margin:.5em 0}.bento-slide p:first-child{margin-top:0}.bento-slide p:last-child{margin-bottom:0}.bento-slide .math-inline{white-space:normal}.bento-slide [data-link][role="link"]{cursor:pointer}.bento-slide [data-link][role="link"]:focus-visible{outline:3px solid #16736E;outline-offset:3px}`;
html=html.replace('</head>',`<meta name="description" content="39 slides and three computed labs explaining Gaussian-splatting losses, image gradients, multi-view geometry and joint UE pose–splat estimation.">
<meta name="gaussian-training-revision" content="2026-11-05-joint-live-v5">
<style id="gaussian-training-layout">${equationCSS}</style>
<script id="gaussian-training-external-links">${externalLinksScript}</script>
<script>(()=>{const routes=${JSON.stringify(routes)};function route(){let r;try{r=decodeURIComponent(location.hash.replace(/^#\\/?/,''));}catch{return;}if(Object.hasOwn(routes,r))history.replaceState(null,'',location.pathname+location.search+'#/'+routes[r]);}addEventListener('hashchange',route);route();})();</script>
</head>`);
writeFileSync(resolve(here,'index.html'),withAuthoring(html,import.meta.url));
const renderedDeck=authoringDocument(deck,import.meta.url);
writeFileSync(resolve(here,'deck.json'),JSON.stringify(renderedDeck,null,2)+'\n');
writeFileSync(resolve(here,'live-demos.json'),JSON.stringify(inlineLiveMap,null,2)+'\n');

// Expanded reading mode comes from the same slide content and notes.
const contents=renderedDeck.slides.map((s,i)=>`<a href="#${s.id}"><span>${String(i+1).padStart(2,'0')}</span> ${esc(s.lessonTitle)}</a>`).join('');
const excluded=new Set(['section','home','rule','heading','subtitle','sources','read','contents','number','open-lab']);
const localHref=url=>url.startsWith(BASE)?'./'+url.slice(BASE.length):url;
const sections=renderedDeck.slides.map((s,i)=>{
 const lab=inlineLiveMap.find(l=>l.slide===s.id);
 const items=s.elements.filter(e=>!excluded.has(e.id)&&(e.type==='text'||e.type==='image'));
 const body=items.map(e=>{
  const content=e.link?`<a href="${esc(e.link.includes(':')?localHref(e.link):'#'+e.link)}">${e.html}</a>`:e.html;
  return e.type==='image'?`<figure><img src="${esc(e.src)}" alt="${esc(e.alt)}" loading="lazy"></figure>`:e.id.endsWith('-label')?`<h3>${content}</h3>`:`<div class="lesson-copy">${content}</div>`;
 }).join('');
 const [notes,refs]=s.notes.split('\n\nSources:\n');
 const sources=refs.split('\n').map(line=>{const split=line.lastIndexOf(' — ');return `<a href="${esc(line.slice(split+3))}">${esc(line.slice(0,split))}</a>`;}).join(' · ');
 return `<article id="${s.id}"><p class="chapter">${String(i+1).padStart(2,'0')} / ${renderedDeck.slides.length}</p><h2>${esc(s.lessonTitle)}</h2><p class="intro">${esc(s.lessonSubtitle)}</p><div class="slide-content">${body}</div>${lab?`<p><a class="lab-link" href="${esc(lab.source)}">Open this interactive laboratory</a></p>`:''}<section class="notes"><h3>Explanation and assumptions</h3><p>${esc(notes)}</p></section><p class="source">${sources}</p><a class="back" href="./#/${i}">Open slide ${i+1}</a></article>`;
}).join('\n');
const study=`<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gaussian Splatting · Loss, Training and Joint Pose Notes | Bai Liping</title><meta name="description" content="Expanded derivations for original 3DGS and MonoGS losses, pose and splat gradients, SE(3) updates, differentiability and gauge freedoms."><link rel="canonical" href="${BASE}study.html"><link rel="stylesheet" href="../assets/deck-theme.css"><style>
:root{color-scheme:light;--ink:#182D33;--muted:#60747A;--teal:#16736E;--line:#D6DEDC}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:80px}body{margin:0;background:#F5F3ED;color:var(--ink);font:18px/1.65 Arial,Helvetica,sans-serif}a{color:var(--teal);text-underline-offset:3px}a:focus-visible{outline:3px solid #B44F37;outline-offset:3px}header{position:sticky;top:0;z-index:2;border-bottom:1px solid var(--line);background:#fffdfaee;padding:14px max(20px,calc((100vw - 1100px)/2));display:flex;justify-content:space-between;gap:18px;font-size:16px}main{width:min(1100px,calc(100% - 40px));margin:auto;padding:28px 0 70px}h1,h2{font-family:'Source Serif 4',Georgia,serif;line-height:1.2}h1{font-size:clamp(32px,5vw,48px)}h2{font-size:32px;margin:0 0 12px}h3{font:600 15px/1.4 'IBM Plex Mono',monospace;color:var(--teal);margin:24px 0 12px}p{margin:.8em 0}.intro{color:var(--muted)}nav{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 24px;margin:28px 0}nav a{padding:8px 0;border-bottom:1px solid var(--line);font-size:16px;text-decoration:none}nav span{font-variant-numeric:tabular-nums;color:#B44F37;margin-right:10px}article{background:#FFFDFA;border:1px solid var(--line);border-radius:10px;padding:30px 28px;margin:28px 0;overflow:hidden}.chapter{font:14px 'IBM Plex Mono',monospace;color:#B44F37}.lesson-copy{margin:10px 0}.lesson-copy span[style]{font-size:inherit!important}.math-display{display:block;overflow-x:auto;padding:10px 0}.slide-content{overflow-x:auto}.notes{background:#E4F0ED;border-left:4px solid var(--teal);padding:2px 20px 12px;margin-top:28px}.notes p{font-size:17px}.source,.back{font-size:15px}.source a{overflow-wrap:anywhere}figure{margin:22px 0}img{display:block;max-width:100%;height:auto}.lab-link{display:inline-block;background:var(--teal);color:white;padding:11px 18px;border-radius:7px;text-decoration:none}@media(max-width:640px){body{font-size:17px}nav{grid-template-columns:1fr}article{padding:24px 18px}main{width:calc(100% - 24px)}h2{font-size:27px}header{font-size:14px}.notes{padding-inline:14px}}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}@media print{header,nav,.back{display:none}article{break-inside:avoid;box-shadow:none}.notes{background:white}body{background:white;font-size:11pt}h2{font-size:20pt}}
</style><script>window.MathJax={tex:{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]},svg:{fontCache:'local'}};</script><script defer src="../mpc-detection-to-bounce-count-slides/assets/vendor/mathjax-3.2.2-tex-svg-full.js"></script></head><body><header><a href="./">Presentation</a><a href="../#group-random-thoughts">Random Thoughts</a><a href="${PAPER}">Original paper</a></header><main><h1>How Pixels Teach Gaussian Splats</h1><p class="intro">Expanded notes, equations and source links for the ${renderedDeck.slides.length}-slide lesson. The live demos compute image losses and parameter updates in your browser.</p><p><a class="lab-link" href="#joint-start">Joint pose + splats</a> <a class="lab-link" href="./live/?lab=training">Train the splats</a> <a class="lab-link" href="./live/?lab=multiview">Test camera constraints</a> <a class="lab-link" href="./live/joint.html">Optimize pose + splats</a></p><nav aria-label="Lesson contents">${contents}</nav>${sections}</main></body></html>`;
writeFileSync(resolve(here,'study.html'),study);
console.log(`Built ${renderedDeck.slides.length} slides, ${inlineLiveMap.length} labs and reading notes.`);

