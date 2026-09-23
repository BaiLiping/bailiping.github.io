'use strict';
// Adapt Drawings/graph_drawing.tex (EO_Writing_Long_Version, d0a380a):
// two representative rows + ellipses on each side of the shared psi factor.
// Black edges are dependencies; colored arrows between cards are processing flow.
const fs = require('node:fs');
const path = require('node:path');
const {createRequire} = require('node:module');
const mathRequire = createRequire(path.resolve(__dirname,'package.json'));
const {mathjax}=mathRequire('mathjax-full/js/mathjax.js');
const {TeX}=mathRequire('mathjax-full/js/input/tex.js');
const {SVG}=mathRequire('mathjax-full/js/output/svg.js');
const {liteAdaptor}=mathRequire('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler}=mathRequire('mathjax-full/js/handlers/html.js');
const {AllPackages}=mathRequire('mathjax-full/js/input/tex/AllPackages.js');
const adaptor=liteAdaptor();RegisterHTMLHandler(adaptor);
const doc=mathjax.document('',{InputJax:new TeX({packages:AllPackages}),OutputJax:new SVG({fontCache:'none'})});
const C={ink:'#16273e',muted:'#596d80',rule:'#ccd8e3',wash:'#f2f6fa',prior:'#b96815',evidence:'#087f68',blue:'#2766b1'};
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
const t=String.raw;
const cache=new Map();
function math(tex,x,y,size=18,color=C.ink){
  if(!cache.has(tex)){
    const s=adaptor.outerHTML(doc.convert(tex,{display:false}));
    if(/data-mjx-error|merror/.test(s))throw new Error(tex);
    const svg=s.slice(s.indexOf('<svg'),s.lastIndexOf('</svg>')+6);
    cache.set(tex,svg);
  }
  let svg=cache.get(tex);
  const exW=Number(svg.match(/width="([\d.]+)ex"/)[1]);
  const exH=Number(svg.match(/height="([\d.]+)ex"/)[1]);
  const w=exW*size*.5,h=exH*size*.5;
  svg=svg.replace(/width="[^"]+"/,`width="${w}"`).replace(/height="[^"]+"/,`height="${h}"`).replace(/style="[^"]*"/,'');
  return `<g role="math" aria-label="${esc(tex)}" fill="${color}" color="${color}">${svg.replace('<svg ',`<svg x="${x-w/2}" y="${y-h/2}" `)}</g>`;
}
const label=(s,x,y,size=16,color=C.ink,anchor='middle',weight=400)=>`<text x="${x}" y="${y}" font-family="Arial,Helvetica,sans-serif" font-size="${size}" fill="${color}" text-anchor="${anchor}" font-weight="${weight}">${esc(s)}</text>`;
const rect=(x,y,w,h,fill=C.wash,stroke=C.rule,r=8)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}"/>`;
const flow=(d,color='blue')=>`<path d="${d}" fill="none" stroke="${C[color]}" stroke-width="2" marker-end="url(#${color})"/>`;
const edge=(d)=>`<path data-edge="dependency" d="${d}" fill="none" stroke="${C.ink}" stroke-width="1.35"/>`;
function graph({id,x,y,w,h=115,title,station='',prior='f',large=false,legacyCount}){
  const size=large?24:15.5,rad=large?28:17;
  const rowY=large?[84,203]:[40,92],mid=(rowY[0]+rowY[1])/2;
  const xs=[.075,.205,.34,.455,.54,.635,.77,.93].map(v=>v*w);
  const nt=legacyCount || (station?`n_{${station}}^t`:'n^t'),ng=station?`n_{${station}}^g`:'n^g';
  let s=rect(0,0,w,h)+(title?label(title,16,large?29:18,large?17:14,C.ink,'start',700):'');
  if(large){s+=label('LEGACY TARGETS',w*.26,29,13,C.muted)+label('NEWBORN CANDIDATES',w*.79,29,13,C.muted);}
  const fw=large?70:62,fh=large?43:29;
  let nodes='';
  function factor(name,tex,cx,cy,width=fw){nodes+=`<g data-node="factor" data-name="${name}">${rect(cx-width/2,cy-fh/2,width,fh,'#fff',C.ink,3)}${math(tex,cx,cy,size)}</g>`;}
  function variable(name,tex,cx,cy){nodes+=`<g data-node="variable" data-name="${name}"><circle cx="${cx}" cy="${cy}" r="${rad}" fill="#fff" stroke="${C.ink}"/>${math(tex,cx,cy,size)}</g>`;}
  factor('psi',t`\boldsymbol\psi`,xs[4],mid,large?62:44);
  rowY.forEach((yy,i)=>{
    const ti=i?nt:'1',gj=i?ng:'1';
    factor('prior-'+i,t`${prior}(\cdot)`,xs[0],yy,large?110:97);
    variable('legacy-'+i,t`\underline{\boldsymbol y}^{${ti}}`,xs[1],yy);
    factor('legacy-factor-'+i,t`\underline l(\cdot)`,xs[2],yy,large?72:46);
    variable('target-association-'+i,`a^{${ti}}`,xs[3],yy);
    variable('group-association-'+i,`b^{${gj}}`,xs[5],yy);
    factor('newborn-factor-'+i,t`\overline l(\cdot)`,xs[6],yy,large?72:46);
    variable('newborn-'+i,t`\overline{\boldsymbol y}^{${gj}}`,xs[7],yy);
    for(const [a,b] of [[0,1],[1,2],[2,3],[5,6],[6,7]])s+=edge(`M${xs[a]},${yy}H${xs[b]}`);
    s+=edge(`M${xs[3]},${yy}L${xs[4]},${mid}L${xs[5]},${yy}`);
  });
  // Dots explicitly denote independent legacy and group count ranges.
  for(const col of [0,1,2,3,5,6,7])s+=math(t`\vdots`,xs[col],mid,large?25:16,C.muted);
  s+=nodes;
  return `<g id="${id}" data-factor-graph="multi-target" data-legacy-range="${esc(nt)}" data-group-range="${esc(ng)}" transform="translate(${x},${y})">${s}</g>`;
}
function wrap(name,title,description,body,w=1200,h=450){
  let defs='<defs>';
  for(const col of ['prior','blue','evidence'])defs+=`<marker id="${col}" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0L8,4L0,8Z" fill="${C[col]}"/></marker>`;
  defs+='</defs>';
  const svg=`<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-labelledby="title desc"><title id="title">${esc(title)}</title><desc id="desc">${esc(description)}</desc>${defs}${body}</svg>\n`;
  fs.writeFileSync(path.resolve(__dirname,'../assets',name+'.svg'),svg);
}
const desc='Adapted from the manuscript factor graph. First and last legacy target chains and first and last group/newborn chains are shown with ellipses. All association variables connect to the shared consistency factor, which abbreviates the product of pairwise consistency factors. Newborn density is included in the newborn factor. Dark edges are dependencies; colored inter-card arrows are processing flow.';
let s=graph({id:'local',x:8,y:5,w:1104,h:250,large:true});
s+=math(t`\boldsymbol\psi(\boldsymbol a,\boldsymbol b)=\prod_{i=1}^{n^t}\prod_{j=1}^{n^g}\Psi_{i,j}(a^i,b^j)`,560,294,20);
s+=label('One prior and likelihood factor per legacy target; one newborn factor per measurement group.',560,329,15,C.muted);
wrap('local_grbp','GrBP factor graph with multiple legacy targets and measurement groups',desc,s,1120,340);
const ys=[8,143,278],x=160,w=880,h=118,mid=66;
s='';
ys.forEach((y,i)=>{
 const bs=i+1;
 s+=graph({id:'bs-'+bs,x,y,w,h,title:`BS ${bs} · independent GrBP`,station:bs,prior:`f_{${bs}}`});
 s+=math(`\\mathcal F_{${bs},k}^{-}`,57,y+mid,23,C.prior)+flow(`M98,${y+mid}H${x}`,'prior');
 s+=flow(`M${x+w},${y+mid}H1101`)+math(`\\mathcal F_{${bs},k}^{+}`,1150,y+mid,23,C.blue);
});
s+=label('Each BS has its own legacy and group counts. Time index k is suppressed inside the graphs.',600,436,16,C.muted);
wrap('distributed','Distributed GrBP: a full multi-target graph at every base station',desc,s);
s='';
ys.forEach((y,i)=>{s+=graph({id:'stage-'+(i+1),x,y,w,h,title:`Stage ${i+1} · BS ${i+1} groups`,station:i+1,prior:i?`\\widetilde f_{(${i})}`:'f'});});
s+=math(t`\mathcal F_k^-`,57,ys[0]+mid,23,C.prior)+flow(`M98,${ys[0]+mid}H${x}`,'prior');
for(let i=0;i<2;i++){
 const yy=ys[i]+h+9;
 s+=flow(`M${x+w},${ys[i]+mid}H1080V${yy}H120V${ys[i+1]+mid}H${x}`);
 s+=rect(330,yy-10,550,20,'#fff','#fff',0)+label(`Updated beliefs from stage ${i+1}, including newborn proposals`,605,yy+5,14,C.blue);
}
s+=flow(`M${x+w},${ys[2]+mid}H1101`)+math(t`\mathcal F_k^+`,1150,ys[2]+mid,23,C.blue);
s+=label('Each stage associates all incoming tracks with its own groups. The previous posterior becomes the next prior.',600,436,16,C.muted);
wrap('centralized','Sequential GrBP: full multi-target association at every update stage',desc,s);
s='';
const px=183,pw=778,py=[8,143,278],cy=209;
py.forEach((y,i)=>{s+=graph({id:'bs-'+(i+1),x:px,y,w:pw,h,title:`BS ${i+1} · common predicted legacy set`,station:i+1,prior:'f',legacyCount:'n^t'});});
s+=rect(8,171,112,76)+label('Common',64,195,17,C.prior)+math(t`\mathcal F_k^-`,64,227,22,C.prior);
for(const y of py)s+=flow(`M120,${cy}H151V${y+mid}H${px}`,'prior');
s+=rect(1054,165,138,91)+label('Legacy-track',1123,190,15)+label('combiner',1123,211,15)+math(t`f(\cdot)\prod_s\gamma_s^i`,1123,238,16);
for(let i=0;i<3;i++){
 const yy=py[i]+mid;
 s+=flow(`M${px+pw},${yy}H1008V${cy}H1054`,'evidence');
 s+=math(`\\{\\gamma_${i+1}^{i}\\}_i`,1005,yy-16,17,C.evidence);
}
s+=flow('M1123,165V103')+math(t`\{\widetilde f_k^i\}_i`,1123,80,22,C.blue);
s+=flow('M64,247V415H1123V256','prior');
s+=rect(370,402,530,25,'#fff','#fff',0)+label('Common prior enters the combiner once',635,421,16,C.prior);
s+=label('Local newborn candidates remain separate; the combiner receives legacy likelihood messages for every matched track.',600,447,15,C.muted);
wrap('centralized_parallel','Parallel GrBP: full local association graphs and per-track likelihood combination',desc,s);
console.log('Built 4 multi-target GrBP factor-graph diagrams.');
