(() => {
  'use strict';
  const V = window.VIMath, $ = id => document.getElementById(id);
  const query = new URLSearchParams(location.search), embedded = query.has('embed');
  const mode = query.get('demo') === 'em' ? 'em' : 'meanfield';
  if (embedded) document.body.classList.add('embedded');
  let timer = null, render, step, state;
  const green = '#2f6b4f', rust = '#a94f2a', blue = '#496e87', ink = '#203129', muted = '#66756e', rule = '#d8ded7';
  const f = (n, digits = 3) => Number(n).toFixed(digits);
  const tx = (x, y, s, fill = muted, size = 10, anchor = 'middle') => `<text x="${x}" y="${y}" fill="${fill}" font-size="${size}" font-family="system-ui,sans-serif" text-anchor="${anchor}">${s}</text>`;
  const line = (x1,y1,x2,y2,color=rule,width=1,dash='') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}" ${dash ? `stroke-dasharray="${dash}"` : ''}/>`;
  function path(points, color, width = 2, dash = '') {
    return `<path d="${points.map((p,i) => (i ? 'L' : 'M') + p.map(v => f(v,2)).join(',')).join(' ')}" fill="none" stroke="${color}" stroke-width="${width}" ${dash ? `stroke-dasharray="${dash}"` : ''}/>`;
  }
  const svg = label => `<svg role="img" aria-label="${label}"></svg>`;
  const metric = (id,label) => `<div class="metric"><span>${label}</span><strong id="${id}">—</strong></div>`;
  const nav = '<div class="mobile-nav"><button data-nav="-1">← Previous slide</button><button data-nav="1">Next slide →</button></div>';
  function pause() { clearInterval(timer); timer = null; const b = $('run'); if (b) b.textContent = 'Run'; }
  function toggleRun() { if (timer) return pause(); $('run').textContent = 'Pause'; timer = setInterval(step, 260); }
  function bind(id, event, fn) { $(id).addEventListener(event, fn); }
  function graph(target, series, colors, label) {
    const w = 670, h = 100, left = 56, right = 16, top = 9, bottom = 23;
    const values = series.flat(); let lo = Math.min(...values), hi = Math.max(...values);
    if (hi-lo < .05) { lo -= .025; hi += .025; }
    const pad = .07*(hi-lo); lo -= pad; hi += pad;
    const xx = i => left + i*(w-left-right)/Math.max(1,series[0].length-1);
    const yy = v => top + (hi-v)*(h-top-bottom)/(hi-lo);
    let s = '';
    for(let j=0;j<3;j++) { const val=lo+(hi-lo)*j/2, y=yy(val); s+=line(left,y,w-right,y)+tx(left-7,y+3,f(val,2),muted,9,'end'); }
    series.forEach((arr,k) => { s += path(arr.map((v,i)=>[xx(i),yy(v)]),colors[k],2,k ? '5 3' : ''); const end=arr.length-1; s+=`<circle cx="${xx(end)}" cy="${yy(arr[end])}" r="3" fill="${colors[k]}"/>`; });
    s += tx(left,h-5,'0',muted,9)+tx(w-right,h-5,String(series[0].length-1),muted,9)+tx((w+left)/2,h-5,label,muted,9);
    target.setAttribute('viewBox',`0 0 ${w} ${h}`); target.innerHTML=s;
  }
  function mfSetup() {
    $('app').innerHTML=`<div class="lab meanfield"><section class="controls"><div class="eyebrow">Live 01 · Fixed posterior</div><h1>Independence has<br>a geometric cost.</h1><label class="field" for="rho">Target correlation ρ <output id="rho-value"></output></label><input id="rho" type="range" min="-.95" max=".95" step=".05" value=".8"><div class="buttons"><button id="independent">ρ = 0</button><button id="correlated">ρ = 0.9</button></div><p>Target marginal variances are both 1. The variational density is a product of two Gaussians.</p><div class="buttons"><button id="reset">Start CAVI from offset</button></div><div class="buttons"><button id="step" class="primary">Update q₁</button><button id="run">Run</button></div><div class="buttons"><button id="fit">Show optimum</button></div><div class="param" id="mf-params"></div><p class="hint">CAVI uses the newest other mean:<br>m₁ ← ρm₂; then m₂ ← ρm₁.<br>Each updated variance is 1 − ρ².</p>${nav}</section><section class="stack"><div class="plots"><div class="chart"><h2>Exact target vs. factorized q</h2>${svg('Exact Gaussian target, current mean-field distribution, and product of exact marginals')}</div><div class="chart"><h2>Marginal variances</h2>${svg('Comparison of the true and variational marginal variances')}</div></div><div class="metrics">${metric('mf-kl','KL(q ∥ p), nats')}${metric('mf-best','Best possible KL')}${metric('mf-updates','Factor updates')}</div><div class="message"><div class="legend"><span><i class="swatch target"></i>Exact target p</span><span><i class="swatch approx"></i>Current q</span><span><i class="swatch marginal"></i>Product of marginals</span></div><div id="mf-message" class="status"></div><span class="hint">Contours have Mahalanobis radius 2; they are not 95% contours. Normalized-target ELBO = −KL.</span></div></section></div>`;
    state={rho:.8,mean:[0,0],variance:[.36,.36],updates:0,next:0,history:[],message:'Optimal mean-field shown. Start CAVI from an offset to watch each factor update.'};
    function reset(optimal=false) {
      pause(); state.mean=optimal?[0,0]:[-1.5,1.5]; state.variance=optimal?[1-state.rho**2,1-state.rho**2]:[1,1]; state.updates=0;state.next=0;state.history=[state.mean.slice()];
      state.message=optimal?'At the reverse-KL optimum, both variances are 1 − ρ². The exact marginal variances remain 1.':'Initial q has displaced means and unit variances. Update one factor while keeping the other fixed.';
      render();
    }
    step=()=>{
      const oldKL=V.gaussianKL(state.rho,state.mean,state.variance), j=state.next;
      Object.assign(state,V.cavi(state.rho,state.mean,state.variance,j)); state.next=1-j;state.updates++;state.history.push(state.mean.slice());
      const kl=V.gaussianKL(state.rho,state.mean,state.variance);
      state.message=`Updated q${j===0?'₁':'₂'}; the other factor stayed fixed. KL decreased by ${f(Math.max(0,oldKL-kl),5)} nats.`;
      render();if(state.updates>=160 || (state.updates>2 && Math.abs(kl + .5*Math.log(1-state.rho**2))<1e-9)) pause();
    };
    render=()=>{
      $('rho-value').textContent=f(state.rho,2);$('rho').value=state.rho;
      $('mf-kl').textContent=f(V.gaussianKL(state.rho,state.mean,state.variance));$('mf-best').textContent=f(-.5*Math.log(1-state.rho**2));$('mf-updates').textContent=state.updates;
      $('step').textContent=state.next===0?'Update q₁':'Update q₂';$('mf-params').textContent=`m = (${f(state.mean[0])}, ${f(state.mean[1])})`; $('mf-message').textContent=state.message;
      const plots=document.querySelectorAll('.plots svg'), w=360,h=255,cx=180,cy=126,scale=27;
      let s='';
      for(let i=-4;i<=4;i++){s+=line(cx+i*scale,cy-4*scale,cx+i*scale,cy+4*scale)+line(cx-4*scale,cy+i*scale,cx+4*scale,cy+i*scale);if(i%2===0)s+=tx(cx+i*scale,cy+4*scale+14,i,muted,9)+tx(cx-4*scale-8,cy-i*scale+3,i,muted,9,'end');}
      s+=tx(cx+4*scale+16,cy+4*scale+14,'z₁',ink,11)+tx(cx-4*scale-8,cy-4*scale-7,'z₂',ink,11);
      function contour(mean,v,rho,color,dash='') { const points=[];for(let i=0;i<=100;i++){const a=2*Math.PI*i/100, x=mean[0]+2*Math.sqrt(v[0])*Math.cos(a),y=mean[1]+2*Math.sqrt(v[1])*(rho*Math.cos(a)+Math.sqrt(1-rho*rho)*Math.sin(a));points.push([cx+scale*x,cy-scale*y]);}return path(points,color,2.3,dash); }
      s+=contour([0,0],[1,1],0,blue,'4 4')+contour([0,0],[1,1],state.rho,green)+contour(state.mean,state.variance,0,rust);
      s+=path(state.history.map(m=>[cx+scale*m[0],cy-scale*m[1]]),rust,1,'3 3');s+=`<circle cx="${cx+scale*state.mean[0]}" cy="${cy-scale*state.mean[1]}" r="3.5" fill="${rust}"/>`;
      plots[0].setAttribute('viewBox',`0 0 ${w} ${h}`);plots[0].innerHTML=s;
      let b='';const bh=172,base=211,top=26,max=Math.max(1.1,...state.variance)*1.15;
      [0,.5,1].forEach(v=>{const y=base-v/max*bh;b+=line(40,y,300,y)+tx(31,y+3,f(v,1),muted,10,'end');});
      [0,1].forEach(j=>{const x=75+j*128;[[1,green],[state.variance[j],rust]].forEach(([v,c],k)=>{const hei=v/max*bh;b+=`<rect x="${x+k*34}" y="${base-hei}" width="26" height="${hei}" rx="3" fill="${c}"/>`+tx(x+k*34+13,base-hei-7,f(v,2),c,10);});b+=tx(x+31,233,'Var(z'+(j+1)+')',ink,11);});
      plots[1].setAttribute('viewBox','0 0 335 255');plots[1].innerHTML=b;
    };
    bind('rho','input',()=>{state.rho=Number($('rho').value);reset(true);});bind('independent','click',()=>{state.rho=0;reset(true);});bind('correlated','click',()=>{state.rho=.9;reset(true);});bind('reset','click',()=>reset(false));bind('fit','click',()=>reset(true));bind('step','click',step);bind('run','click',toggleRun);reset(true);
  }
  function emSetup() {
    $('app').innerHTML=`<div class="lab em"><section class="controls"><div class="eyebrow">Live 02 · Latent-variable learning</div><h1>Soft assignments.<br>Sharper parameters.</h1><label class="field" for="separation">Data separation <output id="sep-value">3.50</output></label><input id="separation" type="range" min=".5" max="6" step=".25" value="3.5"><label class="field" for="initialization">Initial means</label><select id="initialization"><option value="spread">Separated: −0.5, +0.5</option><option value="poor">Both on the left</option><option value="identical">Identical: 0, 0</option></select><label class="check"><input type="checkbox" id="learn">Learn variances (minimum 0.09)</label><div class="buttons"><button id="step" class="primary">Next: E-step</button><button id="run">Run</button></div><div class="buttons"><button id="reset">Reset fit</button><button id="new-data">New data</button></div><div class="phase" id="em-phase">INITIAL q IS UNIFORM</div><div class="param" id="em-params"></div><p class="hint" id="data-info"></p><p class="hint">Default: known variance 0.64.<br>Changing data or model resets the objective history. E-step colours the rug; M-step moves the density.</p>${nav}</section><section class="stack"><div class="chart"><h2>Observed data & current mixture density</h2>${svg('Observed-data histogram, weighted Gaussian component densities, mixture density, and soft assignment rug')}</div><div class="chart"><h2>Likelihood and ELBO · nats per observation</h2>${svg('Log likelihood and evidence lower bound across successive E and M half-steps')}</div><div class="message"><div class="legend"><span><i class="swatch target"></i>Log likelihood: <b id="ll"></b></span><span><i class="swatch approx"></i>ELBO: <b id="elbo"></b></span><span>KL gap: <b id="gap"></b></span></div><div id="em-message"></div></div></section></div>`;
    state={seed:7,separation:3.5,data:[],model:null,q:[],next:'E',updates:0,history:[],learn:false,kind:'spread',message:''};
    function reset(regenerate=false) {
      pause();state.separation=Number($('separation').value);state.kind=$('initialization').value;state.learn=$('learn').checked;
      if(regenerate||!state.data.length)state.data=V.sampleData(state.seed,state.separation);
      state.model=V.initialModel(state.kind,state.separation);state.q=state.data.map(()=>[.5,.5]);state.next='E';state.updates=0;state.history=[V.metrics(state.data,state.model,state.q)];
      state.message='Start with a uniform q. The E-step replaces it by exact posterior responsibilities and closes the gap.';
      render();
    }
    step=()=>{
      const before=V.metrics(state.data,state.model,state.q),phase=state.next;
      if(phase==='E') {state.q=V.expectation(state.data,state.model);state.next='M';state.message='E-step: responsibilities changed; parameters and likelihood stayed fixed. The ELBO now touches the likelihood.';}
      else {state.model=V.maximization(state.data,state.q,state.model,state.learn,.09);state.next='E';state.message='M-step: parameters changed; q stayed fixed. The bound rose, and its gap to the likelihood may reopen.';}
      state.updates++;const after=V.metrics(state.data,state.model,state.q);state.history.push(after);render();
      if(state.updates>=120 || (phase==='M' && state.updates>2 && Math.abs(after.likelihood-before.likelihood)<1e-8)) {
        pause();state.message+=' Automatic run stopped at numerical convergence or its iteration limit.';$('em-message').textContent=state.message;
      }
    };
    render=()=>{
      const met=state.history[state.history.length-1],n=state.data.length;
      $('sep-value').textContent=f(state.separation,2);$('step').textContent='Next: '+state.next+'-step';$('em-phase').textContent=state.updates?'COMPLETED '+(state.next==='E'?'M':'E')+' · HALF-STEP '+state.updates:'INITIAL q IS UNIFORM';
      $('ll').textContent=f(met.likelihood/n);$('elbo').textContent=f(met.elbo/n);$('gap').textContent=f(Math.max(0,met.gap/n),5);$('em-message').textContent=state.message;
      $('em-params').innerHTML=`π = (${f(state.model.weight[0],2)}, ${f(state.model.weight[1],2)})<br>μ = (${f(state.model.mean[0],2)}, ${f(state.model.mean[1],2)})<br>σ² = (${f(state.model.variance[0],2)}, ${f(state.model.variance[1],2)})`;
      $('data-info').textContent=`Synthetic data · N = ${n} · seed ${state.seed}. Generating weights: 0.4 / 0.6; standard deviation: 0.8.`;
      const plots=document.querySelectorAll('.stack .chart svg');
      const w=700,h=207,left=43,right=16,top=9,base=162;
      let lo=Math.min(...state.data,...state.model.mean.map((m,k)=>m-3*Math.sqrt(state.model.variance[k])))-.5,hi=Math.max(...state.data,...state.model.mean.map((m,k)=>m+3*Math.sqrt(state.model.variance[k])))+.5;
      const xx=x=>left+(x-lo)*(w-left-right)/(hi-lo),binCount=30,bw=(hi-lo)/binCount,bins=Array(binCount).fill(0);state.data.forEach(x=>bins[Math.min(binCount-1,Math.max(0,Math.floor((x-lo)/bw)))]++);
      const component=[[],[]],total=[];let max=Math.max(...bins)/(n*bw);
      for(let i=0;i<=220;i++){const x=lo+(hi-lo)*i/220;let sum=0;for(let k=0;k<2;k++){const v=state.model.weight[k]*Math.exp(V.logNormal(x,state.model.mean[k],state.model.variance[k]));component[k].push([x,v]);sum+=v;}total.push([x,sum]);max=Math.max(max,sum);}
      max*=1.15;const yy=y=>base-y/max*(base-top);let s='';
      for(let j=0;j<=3;j++){const v=max*j/3,y=yy(v);s+=line(left,y,w-right,y)+tx(left-6,y+3,f(v,2),muted,9,'end');}
      bins.forEach((b,i)=>{const x=xx(lo+i*bw),y=yy(b/n/bw);s+=`<rect x="${x+.5}" y="${y}" width="${(w-left-right)/binCount-1}" height="${base-y}" fill="#d8ded7" opacity=".8"/>`;});
      component.forEach((arr,k)=>s+=path(arr.map(([x,y])=>[xx(x),yy(y)]),k?rust:green,2));s+=path(total.map(([x,y])=>[xx(x),yy(y)]),ink,2.5,'5 3');
      for(let j=0;j<=6;j++){const x=lo+(hi-lo)*j/6;s+=tx(xx(x),h-8,f(x,1),muted,10);}
      state.data.forEach((x,i)=>{const r=state.q[i][0],a=[47,107,79],b=[169,79,42],color=`rgb(${a.map((v,k)=>Math.round(r*v+(1-r)*b[k])).join(',')})`;s+=line(xx(x),base+8,xx(x),base+17,color,2);});
      s+=tx(w-right,top+7,'Dashed: mixture · Rug: stored q(z)',muted,9,'end');plots[0].setAttribute('viewBox',`0 0 ${w} ${h}`);plots[0].innerHTML=s;
      graph(plots[1],[state.history.map(t=>t.likelihood/n),state.history.map(t=>t.elbo/n)],[green,rust],'E / M half-steps (q or θ updated)');
    };
    bind('step','click',step);bind('run','click',toggleRun);bind('reset','click',()=>reset(false));bind('new-data','click',()=>{state.seed++;reset(true);});bind('separation','input',()=>reset(true));bind('initialization','change',()=>reset(false));bind('learn','change',()=>reset(false));reset(true);
  }
  try { mode==='em'?emSetup():mfSetup(); } catch(error) {$('app').innerHTML='<p class="error">The experiment could not initialize. Please reload the page.</p>';console.error(error);return;}
  document.querySelectorAll('[data-nav]').forEach(b=>b.addEventListener('click',()=>parent.postMessage({type:'bento-inline-nav',direction:Number(b.dataset.nav)},'*')));
  addEventListener('message',event=>{if(event.source!==parent)return;if(event.data?.type==='bento-live-pause')pause();});
  addEventListener('pagehide',pause);document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
  addEventListener('keydown',event=>{if(!embedded || /INPUT|SELECT|TEXTAREA|BUTTON/.test(event.target.tagName))return;if(event.key==='ArrowLeft'||event.key==='ArrowRight'){event.preventDefault();parent.postMessage({type:'bento-inline-nav',direction:event.key==='ArrowLeft'?-1:1},'*');}});
  window.VILab={getState:()=>JSON.parse(JSON.stringify(state)),step,pause};
  if(embedded)parent.postMessage({type:'bento-inline-ready'},'*');
})();
