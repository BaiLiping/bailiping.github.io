(function(){
'use strict';
const M=window.ISAMSummary,$=id=>document.getElementById(id),fmt=v=>Math.abs(v)<1e-9?'0.00':v.toFixed(2);
if(new URLSearchParams(location.search).get('embed')==='region')document.body.classList.add('embedded');
const t=(x,y,s,size=12,attrs='')=>`<text x="${x}" y="${y}" font-size="${size}" ${attrs}>${s}</text>`;
function draw(){
  const state=M.solve(Number($('dp-z').value));
  $('dp-z-value').value=state.z.toFixed(1);$('dp-s').textContent=fmt(state.s);$('dp-u').textContent=fmt(state.u);
  $('dp-tree').innerHTML=`<path d="M294 56H436" stroke="#a6b2a5" stroke-width="2"/><path d="M426 51L436 56L426 61" fill="none" stroke="#a6b2a5" stroke-width="2"/><rect x="25" y="18" width="269" height="77" rx="10" fill="#f5e8df" stroke="#a94f2a"/><rect x="436" y="18" width="279" height="77" rx="10" fill="#e7f0ea" stroke="#2f6b4f"/>`+t(158,41,'Root: s | ∅',14,'text-anchor="middle"')+t(158,67,`s* = ${fmt(state.s)}`,18,'text-anchor="middle" font-weight="700"')+t(575,41,'Child: u | s',14,'text-anchor="middle"')+t(575,67,`u*(s*) = ${fmt(state.u)}`,18,'text-anchor="middle" font-weight="700"')+t(366,45,'use s*',11,'text-anchor="middle"')+t(158,113,'external evidence changes',11,'text-anchor="middle"')+t(575,113,'same conditional / recovery rule',11,'text-anchor="middle"');
  // Fixed axes make the unchanging green summary visually unchanging too.
  const xy=(s,c)=>[44+(s+1)*650/6,151-c*125/24];let svg='';
  for(let c=0;c<=24;c+=6){const [x,y]=xy(-1,c);svg+=`<path d="M${x} ${y}H694" stroke="#d8ded7"/>`+t(34,y+3,c,10,'text-anchor="end"');}
  for(let s=-1;s<=5;s++){const [x,y]=xy(s,0);svg+=t(x,y+15,s,10,'text-anchor="middle"');}
  const curve=(fn,color,dash='')=>{const d=Array.from({length:121},(_,i)=>{const s=-1+i/20;return `${i?'L':'M'}${xy(s,fn(s)).join(' ')}`;}).join(' ');return `<path d="${d}" fill="none" stroke="${color}" stroke-width="2.5" ${dash?`stroke-dasharray="${dash}"`:''}/>`;};
  svg+=curve(M.summary,'#2f6b4f')+curve(s=>M.external(s,state.z),'#a94f2a','6 4')+curve(s=>M.summary(s)+M.external(s,state.z),'#496e87');
  const [x,y]=xy(state.s,state.cost);svg+=`<path d="M${x} 151V${y}" stroke="#496e87" stroke-dasharray="3 3"/><circle cx="${x}" cy="${y}" r="4.5" fill="#496e87" stroke="white"/>`+t(x+9,y-9,'s* = '+fmt(state.s),12)+t(14,15,'cost',10)+t(728,175,'s',12,'text-anchor="end"');
  $('dp-plot').innerHTML=svg;
  $('dp-status').innerHTML=`<strong>Same cached summary, new estimates.</strong> At z = ${fmt(state.z)}, the root chooses s* = ${fmt(state.s)}; the unchanged rule u*(s) gives u* = ${fmt(state.u)}. Reusing a subtree does not freeze its variables.`;
  window.isamDPState=state;
}
$('dp-z').addEventListener('input',draw);$('dp-reset').onclick=()=>{$('dp-z').value=1;draw();};$('dp-example').onclick=()=>{$('dp-z').value=4;draw();};
$('dp-prev').onclick=()=>parent.postMessage({type:'bento-inline-nav',direction:-1},'*');$('dp-next').onclick=()=>parent.postMessage({type:'bento-inline-nav',direction:1},'*');
draw();
})();
