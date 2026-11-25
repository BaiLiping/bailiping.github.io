// Deterministic vector fallbacks, computed from the same model as the live labs.
import M from './model.js';
const green='#2F6B4F',blue='#496E87',rust='#A94F2A',gray='#9ca79b',ink='#203129';
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;');
const label=(s,x,y,size=14,color=ink)=>`<text x="${x}" y="${y}" font-family="Arial,sans-serif" font-size="${size}" fill="${color}">${esc(s)}</text>`;
function plot(x,y,w,h,bounds){
 const [a,b,c,d]=bounds,X=v=>x+(v-a)*w/(b-a),Y=v=>y+h-(v-c)*h/(d-c);let grid='';
 for(let i=0;i<5;i++)grid+=`<path d="M${x+i*w/4},${y}v${h} M${x},${y+i*h/4}h${w}" stroke="#e1e6de" fill="none"/>`;
 const line=(p,color=green,dash='')=>`<polyline points="${p.filter(q=>q[1]!==null).map(q=>X(q[0]).toFixed(2)+','+Y(q[1]).toFixed(2)).join(' ')}" fill="none" stroke="${color}" stroke-width="2" ${dash?'stroke-dasharray="'+dash+'"':''}/>`;
 const dot=(q,color=blue,r=3)=>`<circle cx="${X(q[0])}" cy="${Y(q[1])}" r="${r}" fill="${color}"/>`;
 return {grid,line,dot};
}
let mission;
export function fallback(id){
 let picture='',values=[],title='',caption='';
 if(['whole-run','optimize'].includes(id)){
  mission??=M.makeMission();const p=plot(75,50,640,335,[-18,18,-6,23]);picture=p.grid;const k=id==='whole-run'?24:48;
  if(id==='whole-run')for(let f=0;f<=k;f++)mission.scans[f].forEach((q,j)=>{if(mission.accepted[f].has(j))picture+=p.dot(M.transform(mission.poses[f],q),'#b1c9b5',1.5);});
  picture+=p.line(mission.truth,gray,'5 5')+p.line(mission.poses.slice(0,k+1),green);
  title=id==='whole-run'?'A scan becomes part of the map':'A loop adds competing evidence';
  values=id==='whole-run'?['FRAME 24 / 48','Heading bias: 0.35° / frame','Green: composed ICP poses','Gray: simulator reference','Play, step, or scrub the sequence.']:['INITIAL ODOMETRY','49 poses · 48 sequential edges','1 supplied loop: frame 0 ↔ 48',`Initial objective: ${M.graphCost(mission.poses,M.graphEdges(mission,.15)).toFixed(2)}`,'Step the solver and re-place scans.'];
  caption='Synthetic planar detections · world coordinates in metres · truth is evaluation only';
 }else if(id==='range'){
  const v=M.rangeSpectrum(),p=plot(70,50,660,325,[0,12,-70,5]);picture=p.grid+p.line(v.spectrum.filter(q=>q[0]<=12));for(const r of [6,6.45])picture+=p.line([[r,-70],[r,5]],rust,'4 4');title='Bandwidth sets the range scale';values=['BANDWIDTH 0.45 GHz','Reflector separation: 0.45 m',`Nominal resolution: ${v.resolution.toFixed(3)} m`,'Hann window enabled','256 complex samples → DFT'];caption='Normalized spectrum · range 0–12 m · power −70 to 5 dB';
 }else if(id==='cfar'){
  const v=M.cfar(),p=plot(70,50,660,325,[0,127,0,Math.max(...v.power)*1.12]);picture=p.grid+p.line(v.power.map((x,i)=>[i,x]),blue)+p.line(v.thresholds.map((x,i)=>[i,x]));v.hits.forEach(k=>picture+=p.dot([k,v.power[k]],rust,5));title='Compare each cell to its background';values=['DESIGN FALSE-ALARM RATE: 0.001','16 training cells · 4 guard cells',`${v.hits.length} cells pass the threshold`,'Green: adaptive threshold','Orange: detected cells'];caption='Fixed synthetic power slice · range bins 0–127';
 }else if(id==='frames'){
  const q=[6*Math.cos(M.rad(30)),3],t=[1.5,-1,M.rad(35)],m=M.transform(t,q);for(const [i,b]of [[0,[-2,9,-5.5,5.5]],[1,[-12,12,-12,12]]]){const p=plot(60+i*360,65,300,300,b);picture+=p.grid+(i?p.line([t,m]):p.line([[0,0],q],blue))+p.dot(i?m:q,blue,7)+p.dot(i?t:[0,0],green,5);}title='One measurement, two coordinate frames';values=['ASSUMED POSE','Position: (1.5, −1.0) m','Heading: 35°',`Local point: (${q[0].toFixed(2)}, 3.00) m`,`World point: (${m[0].toFixed(2)}, ${m[1].toFixed(2)}) m`];caption='Left: radar coordinates · right: world coordinates · metres';
 }else if(id==='velocity'){
  const v=M.velocity(),p=plot(70,50,660,325,[-65,65,-6,3]);picture=p.grid;v.data.forEach((q,i)=>picture+=p.dot([M.deg(q.a),q.d],v.inliers.includes(i)?blue:rust));picture+=p.line(Array.from({length:131},(_,i)=>[i-65,-v.v[0]*Math.cos(M.rad(i-65))-v.v[1]*Math.sin(M.rad(i-65))]));title='One velocity explains the static returns';values=['RANSAC + LEAST SQUARES',`Forward velocity: ${v.v[0].toFixed(2)} m/s`,`Lateral velocity: ${v.v[1].toFixed(2)} m/s`,`${v.inliers.length} / 80 returns retained`,'Moving returns: 25%'];caption='Bearing −65° to 65° · radial velocity in m/s';
 }else{
  const v=M.makeICP(),p=plot(210,42,340,340,[-8,8,-8,8]);picture=p.grid;v.target.forEach(q=>picture+=p.dot(q,gray,4));v.source.forEach(q=>picture+=p.dot(q,green,2.8));title='Associate, then update the rigid pose';values=['INITIAL TRANSFORM: IDENTITY','Reference: gray · current: green','Initial heading: 0°','Correspondence gate: 2.0 m','Step 1: find tentative matches'];caption='Gated point-to-point ICP · fixed noise and clutter · coordinates in metres';
 }
 return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1136 475"><rect x=".5" y=".5" width="1135" height="474" rx="14" fill="#FFFEFB" stroke="#D8DED7"/><rect x="776" y="1" width="359" height="473" rx="14" fill="#E7F0EA"/>${label(title,25,28,17)}${picture}${label(caption,25,425,12,'#66756E')}${label('PRINT VIEW · deterministic initial state',25,451,11,'#66756E')}${values.map((s,i)=>label(s,800,74+i*53,i===0?14:15,i===0?green:ink)).join('')}${label('Advance from the concept slide',800,394,13,green)}${label('to use the live experiment.',800,416,13,green)}</svg>`;
}
