(() => {
  'use strict';
  const BP = window.FactorBP, G = window.FactorGraph, app = document.getElementById('app');
  const mode = new URLSearchParams(location.search).get('demo') === 'factor' ? 'factor' : 'schedule';
  const state = { ...BP.defaults, phase: 3, selected: 'C>x3', output: 1, playing: false, view:'message', variable:'x3' };
  let model = BP.run(state), timer;
  const mathCache=new Map();
  const R = String.raw, M = s => `<span class="math-tex math-inline" data-latex="${s.replaceAll('&','&amp;').replaceAll('"','&quot;')}">${mathCache.get(s)||`\\(${s}\\)`}</span>`;
  function cacheMath(){app.querySelectorAll('.math-tex[data-latex]').forEach(e=>{if(e.querySelector('mjx-container'))mathCache.set(e.dataset.latex,e.innerHTML);});}
  const fmt = x => Number(x.toFixed(4)).toString();
  const mu = (from,to) => R`\mu_{${G.tex(from)}\to ${G.tex(to)}}`;
  const names = ['','Leaf messages','Towards the centre','Across the central edge','Out towards the leaves','All marginals available'];
  function parameters() {
    return `<div class="parameters">${[['a',R`f_A(1)`,'Prior weight at the first variable',.05,.95],['b',R`f_B(1)`,'Prior weight at the second variable',.05,.95],['q',R`q`,'XOR agreement in the central factor',.5,.99]].map(([key,label,hint,min,max]) => `<div class="parameter"><label for="${key}"><span>${M(label)} · ${key==='q'?'XOR agreement':'probability of 1'}</span><output id="${key}-value">${state[key].toFixed(2)}</output></label><input id="${key}" type="range" min="${min}" max="${max}" step="0.01" value="${state[key]}" aria-label="${hint}"></div>`).join('')}</div>`;
  }
  function status() {
    return `<div class="status"><span>Binary values and factor tables are teaching choices; graph and schedule follow Figs. 1 & 7.</span><span id="verification" class="success"></span></div>`;
  }
  function mount() {
    if (mode==='schedule') {
      app.innerHTML = `<div class="lab schedule-lab"><div class="stepbar"><button id="prev" aria-label="Previous message step">←</button>${[1,2,3,4,5].map(n=>`<button class="phase-number" data-phase="${n}" aria-label="Message step ${n}">${n}</button>`).join('')}<button id="next" class="primary">Next step →</button><span class="phase-title" id="phase-title" role="status" aria-live="polite"></span><button id="play">Play</button><button id="reset">Reset</button></div><div class="schedule-workspace"><div class="schedule-scene"><p class="phase-story" id="phase-story"></p><div class="graph-zone" id="graph"></div><div class="beliefs-label">Belief at 1 · click a variable to see its received messages</div><div class="beliefs" id="beliefs"></div></div><aside class="schedule-inspector"><div class="inspect-tabs" role="group" aria-label="Calculation to inspect"><button data-view="message">Message calculation</button><button data-view="belief">Variable belief</button></div><div class="inspect-toolbar"><select id="message" aria-label="Message to inspect"></select><select id="variable" aria-label="Variable belief to inspect">${BP.variables.map(v=>`<option value="${v}">${v}</option>`).join('')}</select><div class="output-choice" id="output-choice"><span id="output-label"></span><button data-retained="0" aria-label="Compute entry at zero">0</button><button data-retained="1" aria-label="Compute entry at one">1</button></div></div><div class="message-equation" id="equation"></div><div id="calculation"></div><div class="vector" id="vector"></div><p class="message-note" id="message-note"></p></aside></div>${parameters()}${status()}</div>`;
      app.querySelector('#prev').onclick=()=>changePhase(state.phase-1);
      app.querySelector('#next').onclick=()=>changePhase(state.phase+1);
      app.querySelectorAll('[data-phase]').forEach(b=>b.onclick=()=>changePhase(Number(b.dataset.phase)));
      app.querySelector('#message').onchange=e=>{state.selected=e.target.value;updateSchedule();};
      app.querySelector('#variable').onchange=e=>{state.variable=e.target.value;updateSchedule();};
      app.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{state.view=b.dataset.view;updateSchedule();});
      app.querySelectorAll('[data-retained]').forEach(b=>b.onclick=()=>{state.output=Number(b.dataset.retained);updateSchedule();});
      app.querySelector('#play').onclick=()=>{if(state.playing)stop();else{if(state.phase===5)changePhase(1);state.playing=true;app.querySelector('#play').textContent='Pause';timer=setInterval(()=>{changePhase(state.phase+1,false);if(state.phase===5)stop();},1800);}};
    } else {
      app.innerHTML=`<div class="lab"><div class="stepbar"><span class="phase-title">One entry of a factor message</span><div class="factor-state"><span>Hold ${M(R`x_3`)} fixed:</span><button data-output="0">0</button><button data-output="1">1</button></div><button id="reset">Reset</button></div><div class="factor-workspace"><div><table class="factor-table"><thead><tr>${[R`(x_1,x_2)`,R`f_A(x_1)`,R`f_B(x_2)`,R`f_C(x_1,x_2,x_3)`,R`\text{product}`].map(s=>`<th>${M(s)}</th>`).join('')}</tr></thead><tbody id="terms"></tbody><tfoot id="term-sum"></tfoot></table><div class="factor-table-note">Each row fixes the output variable and sums out one assignment of the other two variables.</div></div><aside class="factor-detail"><h2>Multiply, then sum</h2><div class="factor-formula">${M(R`\mu_{f_C\to x_3}(x_3)`)}<br>${M(R`=\sum_{x_1,x_2}f_A(x_1)f_B(x_2)`)}<br>${M(R`\qquad\cdot f_C(x_1,x_2,x_3)`)}</div><div class="factor-bars" id="factor-bars"></div><p>The outgoing message is a function of ${M(R`x_3`)}. It excludes the message ${M(R`\mu_{x_3\to f_C}`)} arriving from the recipient.</p></aside></div>${parameters()}${status()}</div>`;
      app.querySelectorAll('[data-output]').forEach(b=>b.onclick=()=>{state.output=Number(b.dataset.output);updateFactor();});
    }
    app.querySelector('#reset').onclick=()=>{stop();Object.assign(state,BP.defaults,{phase:1,selected:'A>x1',output:1,view:'message',variable:'x3'});model=BP.run(state);for(const key of ['a','b','q'])app.querySelector('#'+key).value=state[key];update();};
    for(const key of ['a','b','q'])app.querySelector('#'+key).addEventListener('input',e=>{stop();state[key]=Number(e.target.value);model=BP.run(state);update();});
    update();
  }
  function stop(){clearInterval(timer);state.playing=false;const b=app.querySelector('#play');if(b)b.textContent='Play';}
  function changePhase(n,pause=true){if(pause)stop();state.phase=Math.max(1,Math.min(5,n));state.selected=BP.key(...BP.schedule[state.phase-1][0]);updateSchedule();}
  function updateSchedule(){
    cacheMath();
    const Lesson=window.ScheduleLesson;
    app.querySelector('#phase-title').textContent=`${state.phase}/5 · ${names[state.phase]}`;
    app.querySelector('#phase-story').textContent=Lesson.stories[state.phase];
    app.querySelectorAll('[data-phase]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.phase)===state.phase)));
    app.querySelector('#prev').disabled=state.phase===1;app.querySelector('#next').disabled=state.phase===5;
    const available=Object.values(model.messages).filter(m=>m.phase<=state.phase);
    const select=app.querySelector('#message');select.innerHTML=available.map(m=>`<option value="${BP.key(m.from,m.to)}">Step ${m.phase} · ${m.from.startsWith('x')?m.from:'f'+m.from} → ${m.to.startsWith('x')?m.to:'f'+m.to}</option>`).join('');select.value=state.selected;
    const m=model.messages[state.selected],beliefMode=state.view==='belief';
    app.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.view===state.view)));
    app.querySelector('#message').hidden=beliefMode;app.querySelector('#variable').hidden=!beliefMode;app.querySelector('#variable').value=state.variable;
    app.querySelector('#output-choice').hidden=beliefMode;
    app.querySelector('#output-label').innerHTML=M(`${G.tex(m.variable)}=`);
    app.querySelectorAll('[data-retained]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.retained)===state.output)));
    app.querySelector('#graph').innerHTML=G.svg(model,state.phase,state.selected,M,{explain:true,belief:beliefMode?state.variable:null})+`<div class="graph-legend"><span><i class="legend-color input"></i>Used here</span><span><i class="legend-color inspected"></i>Outgoing</span><span>Vectors: [at 0, at 1]</span></div>`;
    app.querySelectorAll('[data-message]').forEach(g=>{const choose=()=>{state.selected=g.dataset.message;state.view='message';updateSchedule();};g.onclick=choose;g.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose();}};});
    const chooseVariable=v=>{state.variable=v;state.view='belief';updateSchedule();};
    app.querySelectorAll('[data-variable]').forEach(g=>{g.onclick=()=>chooseVariable(g.dataset.variable);g.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();chooseVariable(g.dataset.variable);}};});
    if(!beliefMode){
      app.querySelector('#equation').innerHTML=Lesson.equation(m,M);
      app.querySelector('#calculation').innerHTML=Lesson.calculation(m,state.output,M);
      app.querySelector('#vector').innerHTML=[0,1].map(v=>`<div class="${v===state.output?'selected-entry':''}"><small>${M(`${G.tex(m.variable)}=${v}`)}</small><strong>${fmt(m.values[v])}</strong></div>`).join('');
      app.querySelector('#message-note').innerHTML=Lesson.reason(m,M);
    }else{
      const b=BP.beliefAt(model,state.variable,state.phase),x=G.tex(state.variable);
      app.querySelector('#equation').innerHTML=M(`${b.complete?'p':'b'}(${x}=1)=`+R`\frac{${fmt(b.raw[1])}}{${fmt(b.raw[0])}+${fmt(b.raw[1])}}=${b.belief[1].toFixed(4)}`);
      app.querySelector('#calculation').innerHTML=Lesson.beliefCalculation(b,M);
      app.querySelector('#vector').innerHTML=[0,1].map(v=>`<div><small>${M(`${b.complete?'p':'b'}(${x}=${v})`)}</small><strong>${b.belief[v].toFixed(4)}</strong></div>`).join('');
      app.querySelector('#message-note').innerHTML=b.complete?`Exact marginal. Independent enumeration gives ${M(`p(${x}=1)=${model.exact.beliefs[state.variable][1].toFixed(4)}`)}.`:`Partial belief: multiply only messages received so far. The full marginal will be ${M(`p(${x}=1)=${model.exact.beliefs[state.variable][1].toFixed(4)}`)}.`;
    }
    app.querySelector('#beliefs').innerHTML=BP.variables.map(v=>{const b=BP.beliefAt(model,v,state.phase);return `<button data-belief="${v}" class="${b.complete?'':'pending'}" aria-pressed="${beliefMode&&state.variable===v}" aria-label="Inspect ${b.complete?'exact':'partial'} belief at ${v}"><div class="belief-head">${M(G.tex(v))}<strong>${b.belief[1].toFixed(4)}</strong></div><div class="belief-track"><div class="belief-fill" style="width:${100*b.belief[1]}%"></div></div><div class="belief-exact">${b.complete?'exact':`partial · ${b.received.length}/${b.received.length+b.missing.length}`}</div></button>`;}).join('');
    app.querySelectorAll('[data-belief]').forEach(b=>b.onclick=()=>chooseVariable(b.dataset.belief));
    verification();window.typesetDynamicMath?.();
  }
  function updateFactor(){
    cacheMath();
    const m=model.messages['C>x3'];
    app.querySelectorAll('[data-output]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.output)===state.output)));
    app.querySelector('#terms').innerHTML=m.terms[state.output].map(row=>`<tr><td>${M(`(${row.assignment.x1},${row.assignment.x2})`)}</td><td>${fmt(row.incoming[0])}</td><td>${fmt(row.incoming[1])}</td><td>${fmt(row.factorValue)}</td><td class="term">${fmt(row.weight)}</td></tr>`).join('');
    app.querySelector('#term-sum').innerHTML=`<tr><td colspan="4">${M(`${mu('C','x3')}(${state.output})`)} · sum of the four rows</td><td class="term">${fmt(m.values[state.output])}</td></tr>`;
    app.querySelector('#factor-bars').innerHTML=[0,1].map(v=>`<div class="output-row ${v===state.output?'selected':''}">${M(R`x_3=${v}`)}<div class="belief-track"><div class="belief-fill" style="width:${m.values[v]*100}%"></div></div><strong>${fmt(m.values[v])}</strong></div>`).join('');
    verification();window.typesetDynamicMath?.();
  }
  function verification(){const completed=mode==='factor'||state.phase>=5;app.querySelector('#verification').innerHTML=completed?`BP vs all 32 configurations · max error ${model.maxError<1e-12?M(R`<10^{-12}`):model.maxError.toExponential(1)}`:`${Object.values(model.messages).filter(m=>m.phase<=state.phase).length} / 18 directed messages sent`;}
  function update(){for(const key of ['a','b','q'])app.querySelector('#'+key+'-value').textContent=state[key].toFixed(2);if(mode==='factor')updateFactor();else updateSchedule();}
  window.addEventListener('bento-live-visibility',e=>{if(e.detail.paused)stop();});document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  mount();
})();
