import fs from 'node:fs';
import {prepareAnimations} from './prepare-animations.mjs';
import {buildSlides,theme} from './deck.mjs';
prepareAnimations();
const source=fs.readFileSync(new URL('../et-handover/index.html',import.meta.url),'utf8');
// A classic script needs no same-origin sandbox permission. Keep the pure model
// as an ES module for source-level checks, and bundle its exports for the viewer.
const model=fs.readFileSync(new URL('./live/model.mjs',import.meta.url),'utf8').replaceAll('export ','');
const controls=fs.readFileSync(new URL('./live/app.mjs',import.meta.url),'utf8').replace(/^import[^\n]+\n/,'');
fs.writeFileSync(new URL('./live/app.js',import.meta.url),'(()=>{\n'+model+'\n'+controls+'\n})();\n');
// Reuse the site's licensed Bento runtime verbatim; only the authored document changes.
const notice=source.match(/<!--\s*NOTICE[\s\S]*?-->/)?.[0]||'';
const runtimeBlocks=[...source.matchAll(/<script id="bento-rt(?:-css)?" type="bento\/deflate-b64">[\s\S]*?<\/script>/g)].map(m=>m[0]);
const loader=source.match(/<script>\s*\(async \(\) => \{[\s\S]*?<\/script>/)?.[0];
if(runtimeBlocks.length!==2||!loader||!notice)throw new Error('Licensed Bento shell unavailable');
const authored=buildSlides();const demos=authored.filter(s=>s.__demo).map(s=>s.__demo);const slides=authored.map(({__demo,...s})=>s);
const doc={format:'bento/slides',version:1,docId:'bai-liping-nuscenes-tracking',title:'Multitarget Tracking on nuScenes',readonly:true,meta:{subject:'Random finite set tracking and original nuScenes experiment analysis',author:'Bai Liping'},size:{width:1280,height:720},theme,slides};
const json=x=>JSON.stringify(x,null,1).replaceAll('<','\\u003c');
const html=`<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="only light"><title>Multitarget Tracking on nuScenes — Presentation</title><meta name="description" content="A 20-slide research presentation of RFS tracking, archived nuScenes results, and original visual failure analysis."><link rel="canonical" href="https://bailiping.com/multitarget-tracking/presentation.html"><link rel="icon" href="/favicon.ico">${notice}<script type="application/bento+json" id="bento-doc">${json(doc)}</script><link rel="stylesheet" href="../assets/bento-inline-live.css"><link rel="stylesheet" href="./presentation.css"></head><body><div id="bento-splash" aria-hidden="true">Multitarget tracking on nuScenes</div><div id="app"></div><noscript><main><h1>Multitarget tracking on nuScenes</h1><p>This slide viewer requires JavaScript. The complete project narrative, results, and original figures are available on the <a href="./">project page</a>.</p></main></noscript>${runtimeBlocks.join('\n')}${loader}<script type="application/json" id="bento-inline-live-map">${json(demos)}</script><script src="../assets/bento-inline-live.js"></script><script src="./presentation-links.js"></script></body></html>`;
fs.writeFileSync(new URL('./presentation.html',import.meta.url),html);
console.log(`Presentation built: ${slides.length} regular slides; ${demos.length} inline sequence viewer.`);
