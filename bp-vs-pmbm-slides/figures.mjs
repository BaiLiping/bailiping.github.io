import fs from 'node:fs';
import {createRequire} from 'node:module';
const require=createRequire(new URL('../eo-derivation/source/package.json',import.meta.url));
const {mathjax}=require('mathjax-full/js/mathjax.js'),{TeX}=require('mathjax-full/js/input/tex.js'),{SVG}=require('mathjax-full/js/output/svg.js'),{liteAdaptor}=require('mathjax-full/js/adaptors/liteAdaptor.js'),{RegisterHTMLHandler}=require('mathjax-full/js/handlers/html.js'),{AllPackages}=require('mathjax-full/js/input/tex/AllPackages.js');
const adaptor=liteAdaptor();RegisterHTMLHandler(adaptor);const mj=mathjax.document('',{InputJax:new TeX({packages:AllPackages}),OutputJax:new SVG({fontCache:'none'})});
export const C={ink:'#16273e',muted:'#596d80',green:'#087f68',blue:'#2766b1',orange:'#b96815',purple:'#7854a3',line:'#d8e1e9',wash:'#f2f6fa'};
const escape=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
const text=(s,x,y,size=18,color=C.ink,anchor='middle',weight=400)=>`<text x="${x}" y="${y}" fill="${color}" font-family="Arial,Helvetica,sans-serif" font-size="${size}" text-anchor="${anchor}" font-weight="${weight}">${escape(s)}</text>`;
const line=(x1,y1,x2,y2,color=C.line,width=2,dash='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}" ${dash?`stroke-dasharray="${dash}"`:''}/>`;
const rect=(x,y,w,h,fill=C.wash,stroke='none',r=8)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}"/>`;
const circle=(x,y,r,color=C.green)=>`<circle cx="${x}" cy="${y}" r="${r}" fill="#fff" stroke="${color}" stroke-width="2"/>`;
function math(tex,x,y,size=22,color=C.ink){const out=adaptor.outerHTML(mj.convert(tex,{display:false}));if(/data-mjx-error/.test(out))throw Error(tex);let svg=out.slice(out.indexOf('<svg'),out.lastIndexOf('</svg>')+6);const w=Number(svg.match(/width="([\d.]+)ex"/)[1])*size*.5,h=Number(svg.match(/height="([\d.]+)ex"/)[1])*size*.5;svg=svg.replace(/width="[^"]+"/,`width="${w}"`).replace(/height="[^"]+"/,`height="${h}"`).replace(/style="[^"]*"/,'').replace('<svg ',`<svg x="${x-w/2}" y="${y-h/2}" `);return `<g role="math" aria-label="${escape(tex)}" color="${color}">${svg}</g>`;}
function write(name,title,body,w,h){fs.writeFileSync(new URL(`./assets/${name}.svg`,import.meta.url),`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="${escape(title)}">${body}</svg>\n`);}
export function buildFigures(L){
 let s='';
 // The same ambiguity appears on the cover and in the constraint lesson.
 const tY=[96,252],zY=[70,220,332];
 for(let i=0;i<2;i++)for(let j=0;j<2;j++)s+=line(105,tY[i],460,zY[j],i?C.blue:C.green,2,'6 6');
 tY.forEach((y,i)=>{s+=`<ellipse cx="105" cy="${y}" rx="58" ry="40" fill="${i?C.blue:C.green}" fill-opacity=".06"/>`+circle(105,y,22,i?C.blue:C.green)+text(`Track ${i+1}`,105,y+62,18,i?C.blue:C.green);});
 zY.forEach((y,j)=>{s+=line(450,y-10,470,y+10,C.ink,3)+line(450,y+10,470,y-10,C.ink,3)+text(`Detection ${j+1}`,535,y+6,17,C.muted);});
 s+=text('Ambiguous matches',280,24,17,C.muted)+text('Clutter or a new target?',380,386,17,C.muted);
 write('cover','Two predicted tracks competing for detections',s,650,410);
 s='';
 const cards=[[[0,1],'Compatible',C.green],[[1,0],'Compatible',C.blue],[[0,0],'Double claim',C.orange]];
 cards.forEach(([a,title,color],k)=>{const x=k*360;s+=rect(x+4,4,344,304,'#fff',C.line)+text(title,x+176,43,22,color,'middle',700);a.forEach((j,i)=>{s+=line(x+82,108+i*110,x+270,108+j*110,color,3);});for(let i=0;i<2;i++){s+=circle(x+82,108+i*110,20,color)+text(`T${i+1}`,x+82,114+i*110,17,color);s+=rect(x+246,84+i*110,48,48,C.wash)+math(`z_${i+1}`,x+270,108+i*110,22);}s+=text(k===2?'One detection cannot serve both tracks.':'One detection per assigned track.',x+176,282,15,C.muted);});
 write('constraints','Two legal assignments and one illegal double claim',s,1080,312);
 s='';let factors='';const ay=[82,208,334],by=[60,158,256,354];
 for(let i=0;i<3;i++)for(let j=0;j<4;j++)if(L[i][j+1]>0){const f=.30+.18*i,fx=156+(600-156)*f,fy=ay[i]+(by[j]-ay[i])*f;s+=line(156,ay[i],600,by[j],C.line,2);factors+=rect(fx-24,fy-15,48,30,'#fff',C.muted,3)+math(String.raw`\psi_{${i+1}${j+1}}`,fx,fy,16);}
 s+=factors;
 for(let i=0;i<3;i++){s+=line(50,ay[i],132,ay[i],C.muted)+rect(7,ay[i]-18,65,36,C.wash,C.muted,3)+math(`w_${i+1}`,40,ay[i],23)+circle(144,ay[i],23,C.green)+math(`a_${i+1}`,144,ay[i],25,C.green);}
 for(let j=0;j<4;j++)s+=circle(623,by[j],23,C.blue)+math(`b_${j+1}`,623,by[j],25,C.blue);
 s+=text('Tracks',144,18,17,C.green)+text('Measurements',605,18,17,C.blue)+text('Gated graph for the shared example',332,408,16,C.muted);
 write('factor-graph','Three track variables and four measurement variables coupled by seven active pairwise consistency factors',s,675,425);
 s='';
 const factorial=n=>n<2?1:n*factorial(n-1),count=n=>Array.from({length:n+1},(_,k)=>factorial(n)**2/(factorial(k)*factorial(n-k)**2)).reduce((a,b)=>a+b,0);
 const x=n=>78+(n-1)*82,y=v=>340-Math.log10(v)*47;
 for(let power=0;power<=6;power++){const yy=y(10**power);s+=line(70,yy,664,yy,C.line,1)+math(`10^{${power}}`,44,yy,16,C.muted);}
 for(let n=1;n<=8;n++){s+=text(String(n),x(n),372,16,C.muted);if(n>1){s+=line(x(n-1),y(count(n-1)),x(n),y(count(n)),C.orange,3)+line(x(n-1),y((n-1)**2),x(n),y(n*n),C.green,3);}s+=circle(x(n),y(count(n)),4,C.orange)+circle(x(n),y(n*n),4,C.green);}
 s+=text('Number of tracks = number of measurements',370,408,17,C.muted)+text('Compatible assignments',280,24,18,C.orange)+text('BP pair weights',558,24,18,C.green);
 write('scaling','Combinatorial assignment count versus quadratic pair count on a logarithmic axis',s,710,425);
}
