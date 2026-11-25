import {withAuthoring, authoringDocument} from '../scripts/authoring-build.mjs';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import config from './deck.mjs';
const here=path.dirname(fileURLToPath(import.meta.url));
const doc=authoringDocument({format:'bento/slides',version:1,docId:config.docId,title:config.title,readonly:true,meta:{subject:config.description},size:{width:1280,height:720},theme:{background:config.paper,color:config.ink,accent:config.green,fontFamily:config.fontFamily},slides:config.build()}, import.meta.url);
const escape=s=>String(s).replaceAll('&','&amp;').replaceAll('"','&quot;').replaceAll('<','&lt;');
const printSlides=doc.slides.map((slide,index)=>`<section class="crb-print-slide" aria-label="Slide ${index+1}">${slide.elements.map(e=>{
 const bounds=`position:absolute;left:${e.x}px;top:${e.y}px;width:${e.w}px;height:${e.h}px;`;
 if(e.type==='shape')return `<div style="${bounds}background:${e.fill};border-radius:${e.radius||0}px"></div>`;
 if(e.type==='image')return `<img src="${escape(e.src)}" alt="${escape(e.alt)}" style="${bounds}object-fit:contain">`;
 const html=e.html.replace('{{page:2}}',String(index+1).padStart(2,'0')).replace('{{pages:2}}',String(doc.slides.length).padStart(2,'0'));
 return `<div style="${bounds}font-family:${e.fontFamily};font-size:${e.fontSize}px;font-weight:${e.fontWeight};line-height:${e.lineHeight};color:${e.color};text-align:${e.align}">${html}</div>`;
}).join('')}</section>`).join('\n');
// Preserve the checked-in runtime. A packaged copy can rebuild from its own HTML.
const target=path.join(here,'index.html');
const current=fs.readFileSync(target,'utf8');
let html=(current.includes('<!-- crb-host-start -->')?current:fs.readFileSync(path.join(here,'../et-handover/index.html'),'utf8'))
 .replace(/\s*<!-- crb-host-start -->[\s\S]*?<!-- crb-host-end -->/g,'')
 .replace(/\s*<!-- crb-body-start -->[\s\S]*?<!-- crb-body-end -->/g,'')
 .replace(/\s*<!-- eo-host-start -->[\s\S]*?<!-- eo-host-end -->/g,'')
 .replace(/\s*<link[^>]+inline-live\.css[^>]*>/g,'')
 .replace(/\s*<meta name="description" content="[^"]*"\s*\/?>/g,'')
 .replace(/\s*<link rel="canonical"[^>]*>/g,'')
 .replace(/\s*<script type="application\/json" id="companion-demo-map">[\s\S]*?<\/script>\s*/g,'')
 .replace(/\s*<script src="\.\/inline-live\.js(?:\?[^"]*)?"><\/script>\s*/g,'');
const pattern=/(<script type="application\/bento\+json" id="bento-doc">\s*)[\s\S]*?(\s*<\/script>)/;
if(!pattern.test(html))throw new Error('Missing Bento runtime template');
const aliases=Object.fromEntries(doc.slides.map((s,i)=>[s.id.slice(2),i]));
const normalize=`(()=>{const aliases=${JSON.stringify(aliases)};function route(){const key=location.hash.slice(1);if(Object.hasOwn(aliases,key))location.replace('#/'+aliases[key]);}route();addEventListener('hashchange',route);})();`;
html=html.replace(pattern,(_,a,b)=>a+JSON.stringify(doc,null,1).replaceAll('<','\\u003c')+b)
 .replace(/<title>[\s\S]*?<\/title>/,`<title>${config.title}</title>`)
 .replace('</head>',`<!-- crb-host-start --><script>${normalize}</script><meta name="description" content="${escape(config.description)}"><link rel="canonical" href="https://bailiping.com/cramer-rao-bound/"><style>${fs.readFileSync(path.join(here,'deck.css'),'utf8')}</style><!-- crb-host-end --></head>`)
 .replace('</body>',`<!-- crb-body-start --><div id="crb-print" aria-hidden="true">${printSlides}</div>
 <script>${fs.readFileSync(path.join(here,'deck.js'),'utf8')}</script><!-- crb-body-end --></body>`);
fs.writeFileSync(path.join(here,'index.html'),withAuthoring(html, import.meta.url));
console.log(`${config.title}: ${doc.slides.length} native Bento slides, three compact labs, complete static print layout.`);
