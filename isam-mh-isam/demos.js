/* Deterministic SVG experiments. All scores and tree highlights are computed. */
(function(){
'use strict';
const E=window.ISAMTeaching,$=id=>document.getElementById(id);
const colors={green:'#2f6b4f',coral:'#a94f2a',blue:'#4d6f8b',amber:'#986b22',grey:'#a2ada5',line:'#dce2db'};
const fmt=x=>Math.abs(x)<1e-8?'0.00':x.toFixed(2);
const line=(x1,y1,x2,y2,stroke,w=1,extra='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}" stroke-width="${w}" ${extra}/>`;
const text=(x,y,t,size=13,extra='')=>`<text x="${x}" y="${y}" font-size="${size}" ${extra}>${t}</text>`;
function plotBase(bounds,W=620,equal=true,xLabel='x [m]'){
  const [xmin,xmax,ymin,ymax]=bounds,H=300,pad=34,scale=Math.min((W-2*pad)/(xmax-xmin),(H-2*pad)/(ymax-ymin));
  const sx=equal?scale:(W-2*pad)/(xmax-xmin),sy=equal?scale:(H-2*pad)/(ymax-ymin);
  const ox=(W-sx*(xmax-xmin))/2,oy=(H-sy*(ymax-ymin))/2;
  const xy=p=>[ox+(p[0]-xmin)*sx,H-oy-(p[1]-ymin)*sy];let s='';
  for(let x=Math.ceil(xmin);x<=xmax;x++){const a=xy([x,ymin]),b=xy([x,ymax]);s+=line(...a,...b,colors.line)+text(a[0],a[1]+17,x,10,'text-anchor="middle"');}
  for(let y=Math.ceil(ymin);y<=ymax;y++){const a=xy([xmin,y]),b=xy([xmax,y]);s+=line(...a,...b,colors.line)+text(a[0]-12,a[1]+3,y,10,'text-anchor="end"');}
  s+=text(W-18,H-9,xLabel,10,'text-anchor="end"')+text(12,15,'y [m]',10);
  return {s,xy,scale};
}
function path(points,xy,color,width=3,dash='',opacity=1){if(!points.length)return '';const d=points.map((p,i)=>`${i?'L':'M'}${xy(p).join(' ')}`).join(' ');return `<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" opacity="${opacity}" ${dash?`stroke-dasharray="${dash}"`:''}/>`;}
function dots(points,xy,color,labels=false){return points.map((p,i)=>{const [x,y]=xy(p);return `<circle cx="${x}" cy="${y}" r="4" fill="${color}" stroke="white" stroke-width="1.5"/>`+(labels?text(x+8,y-8,`p${i}`,11):'');}).join('');}
function initQR(){
  let model=new E.IncrementalDemo(Number($('qr-noise').value));
  function draw(){const s=model.snapshot(),p=plotBase([-.5,5.7,-.5,5.3],380);let svg=p.s+path(E.truth,p.xy,colors.grey,2,'5 4');
    const odom=[[0,0]];for(let k=1;k<=model.n;k++){const z=E.odometry(k,model.sigma).z;odom.push(odom[k-1].map((v,q)=>v+z[q]));}
    svg+=path(odom,p.xy,colors.coral,2,'5 3')+path(s.points,p.xy,colors.green)+dots(s.points,p.xy,colors.green);
    s.points.forEach((point,i)=>{if(model.closed&&i===8)return;const [x,y]=p.xy(point);svg+=text(x+7,y-8,model.closed&&i===0?'p0 / p8':`p${i}`,11);});
    svg+=text(458,28,'Current R',14);const cell=19,x0=424,y0=47,max=Math.max(1,...model.qr.R.flat().map(Math.abs));
    for(let i=0;i<model.n;i++)for(let j=0;j<model.n;j++){const v=Math.abs(model.qr.R[i][j]);svg+=`<rect x="${x0+j*cell}" y="${y0+i*cell}" width="16" height="16" rx="2" fill="${v<1e-12?'#eef1ea':colors.green}" opacity="${v<1e-12?1:.3+.7*v/max}"/>`;}
    svg+=text(424,224,`${model.n} unknown poses`,11)+text(424,245,'Shared by x and y solves',11);
    if(model.closed){const last=p.xy(s.points.at(-1));svg+=`<circle cx="${last[0]}" cy="${last[1]}" r="12" fill="none" stroke="${colors.coral}" stroke-width="2"/>`;}
    $('qr-plot').innerHTML=svg;$('qr-poses').textContent=model.n+1;$('qr-error').textContent=fmt(s.error);$('qr-inc').textContent=s.rows;$('qr-batch').textContent=s.batchRows;$('qr-difference').textContent=s.difference.toExponential(2)+' m';
    $('qr-next').disabled=model.n>=8;$('qr-close').disabled=model.n!==8||model.closed;
    $('qr-status').textContent=model.closed?'The loop constraint corrected the whole path. Reusing the old factorization gives the same linear least-squares answer as rebuilding it.':model.n===8?'The path should return to the origin, but odometry has drifted. Press “Close loop.”':`Odometry ${model.n}/8. A tree of measurements can have zero residual and still drift away from the synthetic truth.`;
    $('qr-noise-value').value=model.sigma.toFixed(2);window.isamDemoState.qr=model;
  }
  $('qr-next').onclick=()=>{model.next();draw();};$('qr-close').onclick=()=>{model.close();draw();};
  const reset=()=>{model=new E.IncrementalDemo(Number($('qr-noise').value));draw();};$('qr-reset').onclick=reset;$('qr-noise').oninput=reset;draw();
}
function initTree(){
  function draw(){
    const order=$('tree-order').value,event=$('tree-event').value,t=E.symbolic(E.graphEdges,E.orders[order]);
    let vars=event==='append'?[8]:event==='local'?[6,8]:event==='long'?[0,8]:t.cliques.filter(c=>[...c.F,...c.S].includes(2)).flatMap(c=>c.F);
    const a=E.affected(t,vars),positions={},depths={};let leaf=0,maxDepth=0;
    function locate(id,depth){depths[id]=depth;maxDepth=Math.max(maxDepth,depth);const c=t.cliques[id];if(!c.children.length)positions[id]=leaf++;else{c.children.forEach(k=>locate(k,depth+1));positions[id]=c.children.reduce((s,k)=>s+positions[k],0)/c.children.length;}}
    t.cliques.filter(c=>c.parent===null).forEach(c=>locate(c.id,0));
    const xy=id=>[leaf===1?310:105+positions[id]*410/(leaf-1),27+depths[id]*264/Math.max(1,maxDepth)];let s='';
    for(const c of t.cliques){if(c.parent!==null)s+=line(...xy(c.parent),...xy(c.id),colors.grey,2);}
    for(const c of t.cliques){const [x,y]=xy(c.id),on=a.ids.has(c.id),orphan=a.orphans.some(q=>q.id===c.id);s+=`<rect x="${x-93}" y="${y-17}" width="186" height="34" rx="8" fill="${on?'#f7e9df':'#e8eee5'}" stroke="${on?colors.coral:colors.green}" stroke-width="${orphan?3:1.5}"/>`+text(x,y+5,`${c.F.join(', ')} | ${c.S.length?c.S.join(', '):'∅'}`,14,'text-anchor="middle"');}
    $('tree-plot').innerHTML=s;$('tree-affected').textContent=a.ids.size+' / '+t.cliques.length;$('tree-reused').textContent=t.cliques.length-a.ids.size;$('tree-fill').textContent=t.fills.length;
    const cell=14,x0=48,y0=31;let matrix=text(42,13,'R: block sparsity before the update',12);const original=new Set(E.graphEdges.map(e=>e.slice().sort((x,y)=>x-y).join(',')));
    for(let i=0;i<9;i++){matrix+=text(x0+i*cell+6,y0-7,t.order[i],9,'text-anchor="middle"')+text(x0-12,y0+i*cell+10,t.order[i],9,'text-anchor="end"');for(let j=0;j<9;j++){const u=t.order[i],v=t.order[j],on=i===j||t.sep[u].includes(v);let fill='#f0f1ec';if(on)fill=i===j||original.has([u,v].sort((a,b)=>a-b).join(','))?colors.green:colors.amber;matrix+=`<rect x="${x0+j*cell}" y="${y0+i*cell}" width="12" height="12" rx="1" fill="${fill}"/>`;}}
    matrix+=text(218,50,'Green: original structure',13)+text(218,75,'Amber: elimination fill-in',13)+text(218,105,`${t.nonzeros} structural nonzero blocks`,13)+text(218,131,`${a.orphans.length} orphan subtree(s) to reattach`,13);
    $('tree-matrix').innerHTML=matrix;
    $('tree-status').textContent=`${a.variables} of 9 existing frontal variables belong to the affected top. ${order==='recentFirst'&&event==='append'?'Putting the newest pose first makes this new odometry touch every old clique.':'Unmarked conditionals can be reused, although their state estimates may change during back-substitution.'}`;
    window.isamDemoState.tree={tree:t,affected:a,event};
  }
  $('tree-order').onchange=draw;$('tree-event').onchange=draw;draw();
}
const modeName=h=>h?h.split('').map(x=>x==='0'?'+':'−').join(''):'∅';
const branchColor=id=>({'00':colors.green,'01':colors.blue,'10':colors.coral,'11':colors.amber,'0':colors.green,'1':colors.coral,'':colors.green}[id]);
function initMH(){
  let model;
  const names=['0 · Anchored start','1 · First ambiguous displacement','2 · Second ambiguous displacement','3 · Endpoint constraint','4 · Mid-trajectory constraint'];
  const details=['p₀ is fixed; one unambiguous displacement reaches p₁.','Both modes fit equally well. There is no residual redundancy yet.','Two binary choices produce four candidates before capacity pruning.','p₄ − p₀ ≈ (4, 0) is added. Opposite-sign branches can agree at the endpoint.','p₂ − p₀ ≈ (2, 1.2) identifies which intermediate path is supported.'];
  function draw(){
    const base=plotBase([-.3,4.5,-2.8,2.8],620,false,'pose index');let s=base.s;
    for(const h of model.last.filter(h=>h.status!=='retained'))s+=path(h.points,base.xy,colors.grey,1.5,'4 4',.65);
    for(const h of model.live)s+=path(h.points,base.xy,branchColor(h.id),3)+dots(h.points,base.xy,branchColor(h.id));
    const observation=(point,label)=>{const [x,y]=base.xy(point);return `<rect x="${x-7}" y="${y-7}" width="14" height="14" fill="white" stroke="${colors.coral}" stroke-width="2"/>`+text(x+12,y-10,label,11);};
    if(model.stage>=3)s+=observation([4,0],'endpoint');if(model.stage>=4)s+=observation([2,1.2],'midpoint');
    if(!model.live.length)s+=text(310,285,'No surviving trajectories',15,'text-anchor="middle"');
    $('mh-plot').innerHTML=s;$('mh-event').textContent=names[model.stage];$('mh-detail').textContent=details[model.stage];
    const positions={'':[48,90],'0':[228,47],'1':[228,137],'00':[433,21],'01':[433,67],'10':[433,113],'11':[433,159]};let ts='';
    const nodes=Object.keys(model.history).sort((a,b)=>a.length-b.length||a.localeCompare(b));
    for(const id of nodes){if(id){const par=id.slice(0,-1);ts+=line(...positions[par],...positions[id],colors.grey,1.4);}}
    for(const id of nodes){const [x,y]=positions[id],alive=model.live.some(h=>h.id.startsWith(id)),color=alive?branchColor(id):colors.grey;ts+=`<circle cx="${x}" cy="${y}" r="17" fill="${alive?'#fffdfa':'#eef0ec'}" stroke="${color}" stroke-width="2" ${alive?'':'stroke-dasharray="3 3"'}/>`+text(x,y+4,modeName(id),11,'text-anchor="middle"');}
    ts+=text(80,170,'L₀',11)+text(270,170,'L₁',11)+text(482,170,'L₂',11);$('mh-tree').innerHTML=ts;
    $('mh-table').innerHTML=model.last.length?model.last.map(h=>`<tr><td><span class="dot" style="background:${branchColor(h.id)}"></span>${modeName(h.id)}</td><td>${fmt(h.cost)}</td><td>${h.dof}</td><td>${h.status}</td></tr>`).join(''):'<tr><td colspan="4">No active branch to extend.</td></tr>';
    const status=$('mh-status');status.classList.toggle('warning',!model.live.length);
    status.textContent=!model.live.length?'No hypotheses survive. The needed branch was discarded earlier. Raise the cap and reset; there is no automatic resurrection.':`${model.live.length} retained ${model.live.length===1?'hypothesis':'hypotheses'}. `+(model.stage<3?'Wait for later evidence before committing.':model.stage===3?'The endpoint alone need not determine the intermediate path.':model.live.length===1?'The retained path matches the intermediate observation.':'With these settings, multiple alternatives still survive.');
    $('mh-next').disabled=model.stage>=4||!model.live.length;$('mh-sigma-value').value=model.sigma.toFixed(2);window.isamDemoState.mh=model;
  }
  function reset(){model=new E.HypothesisDemo(Number($('mh-cap').value),Number($('mh-sigma').value),$('mh-gate').checked);draw();}
  $('mh-next').onclick=()=>{model.next();draw();};$('mh-reset').onclick=reset;$('mh-cap').onchange=reset;$('mh-gate').onchange=reset;$('mh-sigma').oninput=reset;reset();
}
window.initISAMDemos=()=>{window.isamDemoState={};initQR();initTree();initMH();};
})();
