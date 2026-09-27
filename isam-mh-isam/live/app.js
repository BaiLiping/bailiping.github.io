/* Deterministic live renderers. Model and numerical work live in ../engine.js. */
(()=>{
'use strict';
const E=window.ISAMTeaching;
const $=id=>document.getElementById(id);
const params=new URLSearchParams(location.search);
const mode=['qr','tree','mh'].includes(params.get('demo'))?params.get('demo'):'qr';
const embedded=params.get('embed')==='region';
document.body.classList.toggle('embedded',embedded);
const colors={green:'#2f6b4f',rust:'#a94f2a',blue:'#496e87',amber:'#986b22',grey:'#a2ada5',line:'#dce2db',soft:'#e7f0ea',warm:'#f5e8df',panel:'#fffefb'};
const fmt=x=>Math.abs(x)<1e-8?'0.00':x.toFixed(2);
const line=(x1,y1,x2,y2,stroke=colors.line,w=1,extra='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}" stroke-width="${w}" ${extra}/>`;
const tx=(x,y,value,size=13,extra='')=>`<text x="${x}" y="${y}" font-size="${size}" ${extra}>${value}</text>`;
const metric=(id,label)=>`<div class="metric"><span>${label}</span><strong id="${id}">—</strong></div>`;
const nav='<div class="mobile-nav"><button data-nav="-1">← Previous slide</button><button data-nav="1">Next slide →</button></div>';
let state=null,step=()=>{};

function plotBase(bounds,W=620,equal=true,xLabel='x [m]'){
  const [xmin,xmax,ymin,ymax]=bounds,H=300,pad=34,scale=Math.min((W-2*pad)/(xmax-xmin),(H-2*pad)/(ymax-ymin));
  const sx=equal?scale:(W-2*pad)/(xmax-xmin),sy=equal?scale:(H-2*pad)/(ymax-ymin);
  const ox=(W-sx*(xmax-xmin))/2,oy=(H-sy*(ymax-ymin))/2;
  const xy=p=>[ox+(p[0]-xmin)*sx,H-oy-(p[1]-ymin)*sy];let s='';
  for(let x=Math.ceil(xmin);x<=xmax;x++){const a=xy([x,ymin]),b=xy([x,ymax]);s+=line(...a,...b)+tx(a[0],a[1]+17,x,10,'text-anchor="middle"');}
  for(let y=Math.ceil(ymin);y<=ymax;y++){const a=xy([xmin,y]),b=xy([xmax,y]);s+=line(...a,...b)+tx(a[0]-12,a[1]+3,y,10,'text-anchor="end"');}
  s+=tx(W-18,H-9,xLabel,10,'text-anchor="end"')+tx(12,15,'y [m]',10);
  return {s,xy,scale};
}
function path(points,xy,color,width=3,dash='',opacity=1){if(!points.length)return '';const d=points.map((p,i)=>`${i?'L':'M'}${xy(p).join(' ')}`).join(' ');return `<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" opacity="${opacity}" ${dash?`stroke-dasharray="${dash}"`:''}/>`;}
function dots(points,xy,color,labels=false){return points.map((p,i)=>{const [x,y]=xy(p);return `<circle cx="${x}" cy="${y}" r="4" fill="${color}" stroke="white" stroke-width="1.5"/>`+(labels?tx(x+8,y-8,`p${i}`,11):'');}).join('');}

function setupQR(){
  $('app').innerHTML=`<div class="lab qr"><section class="controls"><div class="eyebrow">Live 01 · Incremental square root</div><h1>Close the loop.<br>Move the history.</h1><p>Add the eight deterministic odometry factors, then add one closure. The graph is translation-only so the factor model remains linear.</p><label class="field" for="qr-noise">Drift scale <output id="qr-noise-value">0.24</output></label><input id="qr-noise" type="range" min="0.08" max="0.50" step="0.02" value="0.24"><div class="buttons"><button class="primary" id="qr-next">Add odometry</button><button id="qr-close">Close loop</button></div><div class="buttons"><button id="qr-reset">Reset</button></div><div class="param">Fixed p0 · 2-D translation factors<br>Loop measurement: p8 − p0 = (0, 0)</div><p class="hint">Rows count vector constraints, each containing two scalar residuals. They are bookkeeping—not measured runtime.</p>${nav}</section><section class="stage"><div class="chart"><h2>Truth, dead reckoning, optimized trajectory, and current triangular factor</h2><svg id="qr-plot" role="img" aria-label="Ground truth, odometry trajectory, optimized trajectory, and square-root matrix"></svg></div><div><div class="metrics">${metric('qr-poses','poses')}${metric('qr-error','whitened error')}${metric('qr-inc','incremental rows')}${metric('qr-batch','batch rows')}</div><div class="message"><div class="legend"><span><i class="dot" style="background:${colors.grey}"></i>synthetic truth</span><span><i class="dot" style="background:${colors.rust}"></i>odometry only</span><span><i class="dot" style="background:${colors.green}"></i>optimized</span></div><div id="qr-status" aria-live="polite"></div><span class="hint">Batch / incremental maximum coordinate difference: <b id="qr-difference">0</b></span></div></div></section></div>`;
  let model;
  function reset(){model=new E.IncrementalDemo(Number($('qr-noise').value));draw();}
  function draw(){
    const snapshot=model.snapshot(),p=plotBase([-.5,5.7,-.5,5.3],510);let svg=p.s+path(E.truth,p.xy,colors.grey,2,'5 4');
    const odom=[[0,0]];for(let k=1;k<=model.n;k++){const z=E.odometry(k,model.sigma).z;odom.push(odom[k-1].map((v,q)=>v+z[q]));}
    svg+=path(odom,p.xy,colors.rust,2,'5 3')+path(snapshot.points,p.xy,colors.green)+dots(snapshot.points,p.xy,colors.green);
    snapshot.points.forEach((point,i)=>{if(model.closed&&i===8)return;const [x,y]=p.xy(point);svg+=tx(x+7,y-8,model.closed&&i===0?'p0 / p8':`p${i}`,11);});
    svg+=tx(602,27,'Current R',14);const cell=20,x0=548,y0=43,max=Math.max(1,...model.qr.R.flat().map(Math.abs));
    for(let i=0;i<model.n;i++)for(let j=0;j<model.n;j++){const value=Math.abs(model.qr.R[i][j]);svg+=`<rect x="${x0+j*cell}" y="${y0+i*cell}" width="17" height="17" rx="2" fill="${value<1e-12?'#eef1ea':colors.green}" opacity="${value<1e-12?1:.3+.7*value/max}"/>`;}
    svg+=tx(548,226,`${model.n} unknown poses`,11)+tx(548,247,'shared by x and y solves',11);
    if(model.closed){const last=p.xy(snapshot.points.at(-1));svg+=`<circle cx="${last[0]}" cy="${last[1]}" r="12" fill="none" stroke="${colors.rust}" stroke-width="2"/>`;}
    $('qr-plot').setAttribute('viewBox','0 0 760 300');$('qr-plot').innerHTML=svg;
    $('qr-poses').textContent=model.n+1;$('qr-error').textContent=fmt(snapshot.error);$('qr-inc').textContent=snapshot.rows;$('qr-batch').textContent=snapshot.batchRows;$('qr-difference').textContent=snapshot.difference.toExponential(2)+' m';
    $('qr-next').disabled=model.n>=8;$('qr-close').disabled=model.n!==8||model.closed;$('qr-noise-value').value=model.sigma.toFixed(2);
    $('qr-status').textContent=model.closed?'The closure corrected the whole path. Incremental and batch QR agree to numerical precision.':model.n===8?'The path should return to the origin, but odometry has drifted. Close the loop.':`Odometry ${model.n}/8. A measurement tree can fit with zero residual and still drift from truth.`;
    state=model;
  }
  step=()=>{model.n<8?model.next():model.close();draw();};
  $('qr-next').addEventListener('click',()=>{model.next();draw();});$('qr-close').addEventListener('click',()=>{model.close();draw();});$('qr-reset').addEventListener('click',reset);$('qr-noise').addEventListener('input',reset);reset();
}

function setupTree(){
  $('app').innerHTML=`<div class="lab tree"><section class="controls"><div class="eyebrow">Live 02 · Symbolic elimination</div><h1>Ordering changes<br>the awakened top.</h1><p>The graph has nine existing scalar variable blocks. All cliques and fill-in edges are computed from the selected elimination order.</p><label class="field" for="tree-order">Ordering</label><select id="tree-order"><option value="chronological">Chronological / recent last</option><option value="balanced">Branched toy ordering</option><option value="recentFirst">Recent first</option></select><label class="field" for="tree-event">Update</label><select id="tree-event"><option value="append">New odometry (8 → 9)</option><option value="local">Short loop (6 ↔ 8)</option><option value="long">Long loop (0 ↔ 8)</option><option value="relin">Relinearize x2</option></select><div class="param">Pre-update Bayes tree<br>Green: reusable · rust: rebuild</div><p class="hint">Try Recent first + new odometry, then Chronological. These are teaching orders—not CCOLAMD.</p>${nav}</section><section class="stage"><div class="split"><div class="chart"><h2>Bayes tree · each box is F | S</h2><svg id="tree-plot" role="img" aria-label="Bayes tree with affected cliques and reusable subtrees"></svg></div><div class="chart"><h2>Triangular block sparsity</h2><svg id="tree-matrix" role="img" aria-label="Symbolic sparsity pattern and elimination fill-in"></svg></div></div><div><div class="metrics">${metric('tree-affected','cliques rebuilt')}${metric('tree-reused','cliques reused')}${metric('tree-fill','fill-in edges')}${metric('tree-frontals','frontals rebuilt')}</div><div class="message" id="tree-status" aria-live="polite"></div></div></section></div>`;
  function draw(){
    const order=$('tree-order').value,event=$('tree-event').value,tree=E.symbolic(E.graphEdges,E.orders[order]);
    const vars=event==='append'?[8]:event==='local'?[6,8]:event==='long'?[0,8]:tree.cliques.filter(c=>[...c.F,...c.S].includes(2)).flatMap(c=>c.F);
    const affected=E.affected(tree,vars),positions={},depths={};let leaf=0,maxDepth=0;
    function locate(id,depth){depths[id]=depth;maxDepth=Math.max(maxDepth,depth);const c=tree.cliques[id];if(!c.children.length)positions[id]=leaf++;else{c.children.forEach(k=>locate(k,depth+1));positions[id]=c.children.reduce((sum,k)=>sum+positions[k],0)/c.children.length;}}
    tree.cliques.filter(c=>c.parent===null).forEach(c=>locate(c.id,0));
    const xy=id=>[leaf===1?300:90+positions[id]*420/Math.max(1,leaf-1),27+depths[id]*254/Math.max(1,maxDepth)];let svg='';
    for(const c of tree.cliques){if(c.parent!==null)svg+=line(...xy(c.parent),...xy(c.id),colors.grey,2);}
    for(const c of tree.cliques){const [x,y]=xy(c.id),on=affected.ids.has(c.id),orphan=affected.orphans.some(q=>q.id===c.id);svg+=`<rect x="${x-78}" y="${y-17}" width="156" height="34" rx="8" fill="${on?colors.warm:colors.soft}" stroke="${on?colors.rust:colors.green}" stroke-width="${orphan?3:1.5}"/>`+tx(x,y+5,`${c.F.join(', ')} | ${c.S.length?c.S.join(', '):'∅'}`,13,'text-anchor="middle"');}
    $('tree-plot').setAttribute('viewBox','0 0 600 300');$('tree-plot').innerHTML=svg;
    const cell=22,x0=40,y0=35,original=new Set(E.graphEdges.map(edge=>edge.slice().sort((a,b)=>a-b).join(',')));let matrix=tx(38,16,'R block pattern before update',11);
    for(let i=0;i<9;i++){matrix+=tx(x0+i*cell+9,y0-7,tree.order[i],9,'text-anchor="middle"')+tx(x0-10,y0+i*cell+14,tree.order[i],9,'text-anchor="end"');for(let j=0;j<9;j++){const u=tree.order[i],v=tree.order[j],on=i===j||tree.sep[u].includes(v);let fill='#f0f1ec';if(on)fill=i===j||original.has([u,v].sort((a,b)=>a-b).join(','))?colors.green:colors.amber;matrix+=`<rect x="${x0+j*cell}" y="${y0+i*cell}" width="19" height="19" rx="2" fill="${fill}"/>`;}}
    matrix+=tx(40,254,'green: graph structure',10)+tx(40,273,'amber: elimination fill-in',10)+tx(40,292,`${tree.nonzeros} structural nonzero blocks`,10);
    $('tree-matrix').setAttribute('viewBox','0 0 260 300');$('tree-matrix').innerHTML=matrix;
    $('tree-affected').textContent=affected.ids.size+' / '+tree.cliques.length;$('tree-reused').textContent=tree.cliques.length-affected.ids.size;$('tree-fill').textContent=tree.fills.length;$('tree-frontals').textContent=affected.variables+' / 9';
    $('tree-status').textContent=order==='recentFirst'&&event==='append'?'Putting the newest pose first makes a new odometry edge touch every old clique.':'Unmarked conditionals are reusable, although their state estimates may still change during back-substitution.';
    state={tree,affected,event,order};
  }
  step=draw;$('tree-order').addEventListener('change',draw);$('tree-event').addEventListener('change',draw);draw();
}

const modeName=id=>id?id.split('').map(x=>x==='0'?'+':'−').join(''):'∅';
const branchColor=id=>({'00':colors.green,'01':colors.blue,'10':colors.rust,'11':colors.amber,'0':colors.green,'1':colors.rust,'':colors.green}[id]||colors.grey);
function setupMH(){
  $('app').innerHTML=`<div class="lab mh"><section class="controls"><div class="eyebrow">Live 03 · Branch and prune</div><h1>Wait for evidence.<br>Then prune.</h1><p>Two ambiguous displacements create four candidates. An endpoint measurement and then a midpoint measurement disambiguate them.</p><label class="field" for="mh-cap">Keep at most</label><select id="mh-cap"><option value="4">4 hypotheses</option><option value="2">2 hypotheses</option><option value="1">1 hypothesis</option></select><label class="check"><input id="mh-gate" type="checkbox" checked>95% chi-square gate</label><label class="field" for="mh-sigma">Later-measurement scale <output id="mh-sigma-value">0.22</output></label><input id="mh-sigma" type="range" min="0.14" max="0.80" step="0.02" value="0.22"><div class="buttons"><button id="mh-next" class="primary">Next event</button><button id="mh-reset">Reset</button></div><div class="param" id="mh-event">0 · Anchored start</div><p class="hint">Try cap = 1 from the start. Early ties are broken before any evidence distinguishes them.</p>${nav}</section><section class="stage"><div class="chart"><h2>Optimized trajectory per retained mode sequence</h2><svg id="mh-plot" role="img" aria-label="Optimized y-position versus pose index for retained mode assignments"></svg></div><div class="split"><div class="chart"><h2>Hypo-tree · dashed means pruned</h2><svg id="mh-tree" role="img" aria-label="Hypothesis prefix tree with active and pruned branches"></svg></div><div class="card" style="padding:10px 12px"><h2>Residual consistency</h2><div class="table-wrap"><table><thead><tr><th>Modes</th><th>error</th><th>DoF</th><th>decision</th></tr></thead><tbody id="mh-table"></tbody></table></div><p class="hint" id="mh-detail"></p><div class="message" id="mh-status" aria-live="polite"></div></div></div></section></div>`;
  const names=['0 · Anchored start','1 · First ambiguous displacement','2 · Second ambiguous displacement','3 · Endpoint constraint','4 · Mid-trajectory constraint'];
  const details=['One fixed pose and one unambiguous displacement.','Both modes fit equally well; there is no residual redundancy yet.','Two binary choices produce four candidates before pruning.','The endpoint supports two opposite-sign paths.','The midpoint identifies the supported intermediate path.'];
  let model;
  function reset(){model=new E.HypothesisDemo(Number($('mh-cap').value),Number($('mh-sigma').value),$('mh-gate').checked);draw();}
  function draw(){
    const base=plotBase([-.3,4.5,-2.8,2.8],760,false,'pose index');let svg=base.s;
    for(const h of model.last.filter(h=>h.status!=='retained'))svg+=path(h.points,base.xy,colors.grey,1.5,'4 4',.65);
    for(const h of model.live)svg+=path(h.points,base.xy,branchColor(h.id),3)+dots(h.points,base.xy,branchColor(h.id));
    const observation=(point,label)=>{const [x,y]=base.xy(point);return `<rect x="${x-7}" y="${y-7}" width="14" height="14" fill="white" stroke="${colors.rust}" stroke-width="2"/>`+tx(x+12,y-10,label,11);};
    if(model.stage>=3)svg+=observation([4,0],'endpoint');if(model.stage>=4)svg+=observation([2,1.2],'midpoint');if(!model.live.length)svg+=tx(380,286,'No surviving trajectories',15,'text-anchor="middle"');
    $('mh-plot').setAttribute('viewBox','0 0 760 300');$('mh-plot').innerHTML=svg;$('mh-event').textContent=names[model.stage];$('mh-detail').textContent=details[model.stage];
    const positions={'':[44,92],'0':[192,48],'1':[192,136],'00':[360,21],'01':[360,67],'10':[360,113],'11':[360,159]};let treeSvg='';const ids=Object.keys(model.history).sort((a,b)=>a.length-b.length||a.localeCompare(b));
    for(const id of ids){if(id){const parent=id.slice(0,-1);treeSvg+=line(...positions[parent],...positions[id],colors.grey,1.4);}}
    for(const id of ids){const [x,y]=positions[id],alive=model.live.some(h=>h.id.startsWith(id)),color=alive?branchColor(id):colors.grey;treeSvg+=`<circle cx="${x}" cy="${y}" r="17" fill="${alive?colors.panel:'#eef0ec'}" stroke="${color}" stroke-width="2" ${alive?'':'stroke-dasharray="3 3"'}/>`+tx(x,y+4,modeName(id),11,'text-anchor="middle"');}
    treeSvg+=tx(70,177,'L0',10)+tx(207,177,'L1',10)+tx(374,177,'L2',10);$('mh-tree').setAttribute('viewBox','0 0 410 185');$('mh-tree').innerHTML=treeSvg;
    $('mh-table').innerHTML=model.last.length?model.last.map(h=>`<tr><td><i class="dot" style="background:${branchColor(h.id)}"></i>${modeName(h.id)}</td><td>${fmt(h.cost)}</td><td>${h.dof}</td><td>${h.status}</td></tr>`).join(''):'<tr><td colspan="4">No active branch to extend.</td></tr>';
    const status=$('mh-status');status.classList.toggle('warning',!model.live.length);status.textContent=!model.live.length?'No hypotheses survive. Raise the cap and reset; deleted branches are not automatically restored.':`${model.live.length} retained ${model.live.length===1?'hypothesis':'hypotheses'}. `+(model.stage<3?'Wait for later evidence before committing.':model.stage===3?'The endpoint alone does not determine the intermediate path.':model.live.length===1?'The retained path matches the midpoint observation.':'Several alternatives remain with these settings.');
    $('mh-next').disabled=model.stage>=4||!model.live.length;$('mh-sigma-value').value=model.sigma.toFixed(2);state=model;
  }
  step=()=>{model.next();draw();};$('mh-next').addEventListener('click',step);$('mh-reset').addEventListener('click',reset);$('mh-cap').addEventListener('change',reset);$('mh-gate').addEventListener('change',reset);$('mh-sigma').addEventListener('input',reset);reset();
}

try{if(!E)throw new Error('Teaching engine unavailable');({qr:setupQR,tree:setupTree,mh:setupMH})[mode]();}
catch(error){$('app').innerHTML='<p class="error">The experiment could not initialize. Please reload the page.</p>';console.error(error);}
document.querySelectorAll('[data-nav]').forEach(button=>button.addEventListener('click',()=>parent.postMessage({type:'bento-inline-nav',direction:Number(button.dataset.nav)},'*')));
window.ISAMLab={getState:()=>{if(!state)return null;return mode==='tree'?{order:state.order,event:state.event,cliques:state.tree.cliques.length,affected:state.affected.ids.size}:{mode,stage:state.stage??state.n??0,live:state.live?.map(h=>h.id)};},step};
})();
