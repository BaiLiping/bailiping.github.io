import {withAuthoring} from '../scripts/authoring-build.mjs';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import config from './deck.mjs';
import {preset,solveScene} from '../jpda/math.mjs';
import {sceneSVG} from '../jpda/scene.mjs';
const here=path.dirname(fileURLToPath(import.meta.url));
fs.mkdirSync(path.join(here,'assets'),{recursive:true});
for(const name of ['shared','ambiguous'])fs.writeFileSync(path.join(here,`assets/${name}.svg`),sceneSVG(solveScene(preset(name)),{showPDA:true,id:'static-'+name}));
const doc={format:'bento/slides',version:1,docId:config.docId,title:config.title,readonly:true,meta:{subject:config.description},size:{width:1280,height:720},theme:{background:config.paper,color:config.ink,accent:config.green,fontFamily:config.fontFamily},slides:config.build()};
// A static, source-generated copy makes browser printing independent of the
// presentation runtime. It also preserves the explanatory demo snapshots.
const escape=s=>String(s).replaceAll('&','&amp;').replaceAll('"','&quot;').replaceAll('<','&lt;');
const printSlides=doc.slides.map((slide,index)=>`<section class="jpda-print-slide" aria-label="Slide ${index+1}">${slide.elements.map(e=>{
 const bounds=`position:absolute;left:${e.x}px;top:${e.y}px;width:${e.w}px;height:${e.h}px;`;
 if(e.type==='shape')return `<div style="${bounds}background:${e.fill};border-radius:${e.radius||0}px"></div>`;
 if(e.type==='image')return `<img src="${escape(e.src)}" alt="${escape(e.alt)}" style="${bounds}object-fit:contain">`;
 const html=e.html.replace('{{page:2}}',String(index+1).padStart(2,'0')).replace('{{pages:2}}',String(doc.slides.length).padStart(2,'0'));
 return `<div style="${bounds}font-family:${e.fontFamily};font-size:${e.fontSize}px;font-weight:${e.fontWeight};line-height:${e.lineHeight};color:${e.color};text-align:${e.align}">${html}</div>`;
}).join('')}</section>`).join('\n');
let html=fs.readFileSync(path.join(here,'../et-handover/index.html'),'utf8')
 .replace(/\s*<!-- eo-host-start -->[\s\S]*?<!-- eo-host-end -->/g,'')
 .replace(/\s*<link[^>]+inline-live\.css[^>]*>/g,'')
 .replace(/\s*<meta name="description" content="[^"]*"\s*\/?>/g,'')
 .replace(/\s*<link rel="canonical"[^>]*>/g,'')
 .replace(/\s*<script type="application\/json" id="companion-demo-map">[\s\S]*?<\/script>\s*/g,'')
 .replace(/\s*<script src="\.\/inline-live\.js(?:\?[^"]*)?"><\/script>\s*/g,'');
const block=/(<script type="application\/bento\+json" id="bento-doc">\s*)[\s\S]*?(\s*<\/script>)/;
if(!block.test(html))throw new Error('Missing Bento runtime template');
html=html.replace(block,(_,open,close)=>open+JSON.stringify(doc,null,1).replaceAll('<','\\u003c')+close)
 .replace(/<title>[\s\S]*?<\/title>/,`<title>${config.title}</title>`)
 .replace('</head>',`<meta name="description" content="${config.description}">
    <link rel="canonical" href="https://bailiping.com/jpda-slides/">
    <link rel="stylesheet" href="../et-handover/assets/materials.css">
    <link rel="stylesheet" href="./style.css">
    <script src="../et-handover/assets/math-config.js"></script>
    <script defer src="../et-handover/assets/vendor/mathjax-3.2.2-tex-svg-full.js"></script>
    <script defer src="../et-handover/assets/mathjax-dynamic.js"></script>
    <script defer src="./deck.js"></script>
  </head>`)
 .replace('</body>',`<div id="jpda-print" aria-hidden="true">${printSlides}</div>
    <dialog id="jpda-dialog" aria-labelledby="jpda-dialog-title">
      <div class="demo-toolbar"><button type="button" id="demo-back">← Back to slides</button><h2 id="jpda-dialog-title">JPDA interactive lab</h2><a id="demo-full" href="/jpda/">Open full lab →</a></div>
      <p id="demo-loading" role="status">Loading the association lab…</p><div id="demo-frame"></div>
    </dialog>
  </body>`);
fs.writeFileSync(path.join(here,'index.html'),withAuthoring(html, import.meta.url));
console.log(`${config.title}: ${doc.slides.length} slides; two lazy interactive entry points.`);
