(()=>{
 'use strict';
 const BP=window.TrackingBP,G=window.TrackingGraph,L=window.TrackingLesson,app=document.querySelector('#app');
 const state={...BP.defaults,index:0,j:0,m:0,entry:1,kind:'legacy',playing:false};
 let model=BP.run(state),steps=BP.steps(state.iterations),timer;
 const R=String.raw,cache=new Map(),T=latex=>`<span class="math-tex math-inline" data-latex="${latex.replaceAll('&','&amp;').replaceAll('"','&quot;')}">${cache.get(latex)||`\\(${latex}\\)`}</span>`;
 function cacheMath(){app.querySelectorAll('[data-latex]').forEach(e=>{if(e.querySelector('mjx-container'))cache.set(e.dataset.latex,e.innerHTML);});}
 app.innerHTML=`<div class="lab"><div class="stepbar"><button id="prev" aria-label="Previous message">←</button><select id="step" aria-label="Message step"></select><button class="primary" id="next">Next message →</button><span id="progress" role="status" aria-live="polite"></span><button id="play">Play</button><button id="reset">Reset</button></div><div class="phases" aria-label="Jump to a stage">${L.groupNames.map((n,i)=>`<button data-group="${i}"><span>${i+1}</span> ${n}</button>`).join('')}<label class="iterations">Sweeps <select id="iterations" aria-label="Association iterations">${[1,2,3,5,8,12].map(n=>`<option>${n}</option>`).join('')}</select></label></div><div class="workspace"><div class="scene"><div class="graph-legend"><span><i class="out"></i>Message now</span><span><i class="in"></i>Inputs used</span><span>Click a node to inspect its track or measurement</span></div><div id="graph"></div><div id="measurement-scene" class="measurement-scene"></div><div id="beliefs" class="beliefs"></div></div><aside class="inspector"><div class="selection"><label>Track <select id="j" aria-label="Legacy target to inspect"><option value="0">1</option><option value="1">2</option></select></label><label>Measurement <select id="m" aria-label="Measurement to inspect"><option value="0">1</option><option value="1">2</option></select></label><span id="source"></span></div><h2 id="message-title"></h2><p id="description"></p><div class="formula" id="formula"></div><div class="entry-controls"><span id="domain"></span><span id="entry-buttons"></span><div id="belief-kind"><button data-kind="legacy">Legacy</button><button data-kind="new">New</button></div></div><div id="calculation"></div><div id="vector" class="vector"></div><p id="note"></p></aside></div><div class="parameters">${[['z2','Measurement 2 position',-.2,1.2,.01],['pD','Detection probability',.05,.99,.01],['birth','Expected new detections',0,1,.01]].map(([key,label,min,max,step])=>`<label class="parameter" for="${key}"><span>${label}<output id="${key}-value"></output></span><input type="range" id="${key}" min="${min}" max="${max}" step="${step}" value="${state[key]}"></label>`).join('')}</div><div class="status"><span>Fig. 4 topology · 2 legacy candidates, 2 measurements · two-cell teaching model</span><span id="check"></span></div></div>`;
 app.querySelector('.scene').append(app.querySelector('#note'));
 function stop(){clearInterval(timer);state.playing=false;app.querySelector('#play').textContent='Play';}
 function choose(index){stop();state.index=Math.max(0,Math.min(steps.length-1,index));state.entry=1;render();}
 function recalculate(){const previous=steps[state.index],previousID=previous.id,round=previous.iteration;model=BP.run(state);steps=BP.steps(state.iterations);const same=steps.findIndex(s=>s.id===previousID&&s.iteration===round);state.index=same>=0?same:steps.findIndex(s=>s.id==='nu');render();}
 app.querySelector('#prev').onclick=()=>choose(state.index-1);app.querySelector('#next').onclick=()=>choose(state.index+1);
 app.querySelector('#step').onchange=e=>choose(Number(e.target.value));
 app.querySelectorAll('[data-group]').forEach(b=>b.onclick=()=>choose(steps.findIndex(s=>s.group===Number(b.dataset.group))));
 for(const key of ['j','m'])app.querySelector('#'+key).onchange=e=>{state[key]=Number(e.target.value);render();};
 app.querySelectorAll('[data-kind]').forEach(b=>b.onclick=()=>{state.kind=b.dataset.kind;render();});
 app.querySelector('#iterations').onchange=e=>{stop();state.iterations=Number(e.target.value);recalculate();};
 for(const key of ['z2','pD','birth'])app.querySelector('#'+key).oninput=e=>{stop();state[key]=Number(e.target.value);recalculate();};
 app.querySelector('#play').onclick=()=>{if(state.playing)return stop();if(state.index===steps.length-1)state.index=0;state.playing=true;app.querySelector('#play').textContent='Pause';timer=setInterval(()=>{state.index++;state.entry=1;if(state.index>=steps.length-1){state.index=steps.length-1;stop();}render();},2600);render();};
 app.querySelector('#reset').onclick=()=>{stop();Object.assign(state,BP.defaults,{index:0,j:0,m:0,entry:1,kind:'legacy'});model=BP.run(state);steps=BP.steps(state.iterations);render();};
 function scene(){
  const pos=x=>36+(x+.2)/1.4*608;
  return `<svg viewBox="0 0 682 44" role="img" aria-label="Spatial cells at 0 and 1; measurement 1 at 0.3 and measurement 2 at ${state.z2.toFixed(2)}"><line x1="36" y1="27" x2="644" y2="27" stroke="#d6e0e7" stroke-width="2"/>${[0,1].map((x,i)=>`<circle cx="${pos(x)}" cy="27" r="4" fill="#203446"/><text x="${pos(x)}" y="43" text-anchor="middle">${i?'R = 1':'L = 0'}</text>`).join('')}${model.z.map((x,m)=>`<path d="M ${pos(x)-4} 22 L ${pos(x)+4} 22 L ${pos(x)} 29 z" fill="${m?'#c16023':'#126caa'}"/><text x="${pos(x)}" y="${Math.abs(model.z[0]-model.z[1])<.12?(m?17:7):15}" text-anchor="middle" fill="${m?'#c16023':'#126caa'}">z${m+1} = ${x.toFixed(2)}</text>`).join('')}</svg>`;
 }
 function cards(step){
  const final=step.id==='belief';
  return [0,1,2,3].map(i=>{
   const old=i<2,n=i%2,v=old?(final?model.legacy[n]:model.alpha[n]):final||step.id==='zeta'?model.newTargets[n]:null;
   return `<button class="belief-card" data-candidate="${i}" aria-label="Inspect ${old?'legacy':'new'} target ${n+1} belief"><div>${old?'Legacy':'New'} ${n+1}<strong>${v?(100*(1-v[0])).toFixed(1)+'%':'—'}</strong></div><div class="belief-track"><i style="width:${v?100*(1-v[0]):0}%"></i></div><small>${v?(final||!old?'BP existence':'predicted existence'):'after association update'}</small></button>`;
  }).join('');
 }
 function render(){
  cacheMath();const step=steps[state.index],lesson=L.explain(model,step,state,T);
  app.querySelector('#step').innerHTML=steps.map((s,i)=>`<option value="${i}">${i+1}. ${s.title}</option>`).join('');app.querySelector('#step').value=state.index;
  app.querySelector('#progress').textContent=`${state.index+1} / ${steps.length}`;
  app.querySelector('#prev').disabled=state.index===0;app.querySelector('#next').disabled=state.index===steps.length-1;
  app.querySelectorAll('[data-group]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.group)===step.group)));
  app.querySelector('#iterations').value=state.iterations;app.querySelector('#j').value=state.j;app.querySelector('#m').value=state.m;
  app.querySelector('#graph').innerHTML=G.svg({step,j:state.j,m:state.m,kind:state.kind},T);
  app.querySelectorAll('[data-node]').forEach(n=>{function select(){if(n.dataset.j!==undefined)state.j=Number(n.dataset.j);if(n.dataset.m!==undefined)state.m=Number(n.dataset.m);render();}n.onclick=select;n.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}};});
  app.querySelector('#measurement-scene').innerHTML=scene();app.querySelector('#beliefs').innerHTML=cards(step);app.querySelector('#beliefs').hidden=step.id!=='belief';
  app.querySelectorAll('[data-candidate]').forEach(b=>b.onclick=()=>{const n=Number(b.dataset.candidate);state.kind=n<2?'legacy':'new';state[n<2?'j':'m']=n%2;choose(steps.length-1);});
  app.querySelector('#source').textContent=lesson.source;app.querySelector('#message-title').textContent=lesson.title;app.querySelector('#description').textContent=lesson.description;
  app.querySelector('#formula').innerHTML=lesson.equation;app.querySelector('#calculation').innerHTML=lesson.table;app.querySelector('#note').textContent=lesson.note;
  app.querySelector('#domain').textContent=step.id==='belief'?'Inspect':lesson.domain+':';
  app.querySelector('#entry-buttons').hidden=step.id==='belief';app.querySelector('#belief-kind').hidden=step.id!=='belief';
  app.querySelector('#entry-buttons').innerHTML=lesson.labels.map((n,i)=>`<button data-entry="${i}" aria-label="Inspect output entry ${i}" aria-pressed="${i===state.entry}">${T(n)}</button>`).join('');
  app.querySelectorAll('[data-entry]').forEach(b=>b.onclick=()=>{state.entry=Number(b.dataset.entry);render();});
  app.querySelectorAll('[data-kind]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.kind===state.kind)));
  app.querySelector('#vector').innerHTML=`<span class="vector-label">${lesson.normalized?'Probability':'Message'}<small>[${lesson.labels.map(n=>n==='\\varnothing'?'absent':n).join(', ')}]</small></span>${lesson.vector.map((v,i)=>`<div class="${i===state.entry?'selected':''}"><span>${v.label}</span><strong>${v.value}</strong></div>`).join('')}`;
  for(const key of ['z2','pD','birth']){app.querySelector('#'+key).value=state[key];app.querySelector('#'+key+'-value').textContent=state[key].toFixed(2);}
  app.querySelector('#check').textContent=step.id==='belief'?`Exact check · largest state error ${(100*model.maxError).toFixed(2)} pp`:step.group===2?`Sweep ${step.iteration||0} · all 4 pairs update in parallel`:'Only the orange path is being inspected';
  app.dataset.step=step.id;app.dataset.index=state.index;window.typesetDynamicMath?.();
 }
 addEventListener('bento-live-visibility',e=>{if(e.detail.paused)stop();});document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
 render();
})();
