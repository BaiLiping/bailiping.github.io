/* Coded-pilot, unknown-channel bound calculator. All calculations are local. */
(() => {
  'use strict';
  const API = window.RadioProjectCRB;
  const controls = document.getElementById('controls');
  const results = document.getElementById('results');
  if (!API) {
    results.innerHTML='<div class="alert"><strong>The calculation model did not load.</strong>Please reload the laboratory.</div>';
    return;
  }
  const COLORS={blue:'#1874B8',teal:'#0A6B5E',orange:'#E8720C',ink:'#16222E',muted:'#586674',line:'#DCE3E9'};
  const STORAGE='bailiping-radio-project-crb-v1';
  let saved={};
  try{saved=JSON.parse(sessionStorage.getItem(STORAGE)||'{}')||{};}catch(_){/* Storage can be disabled in embedded content. */}
  let coded={...API.defaults,...saved.coded};
  let result=null;
  let timer=null;
  const local={cell:[3,6],raw:false};
  const $=selector=>document.querySelector(selector);
  const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const num=(v,d=3)=>{
    if(v===Infinity||v===-Infinity)return '∞';
    if(!Number.isFinite(v))return '—';
    if(v===0)return '0';
    if(Math.abs(v)<.001||Math.abs(v)>=1e6)return v.toExponential(d-1).replace('e+','e');
    return Number(v.toPrecision(d)).toLocaleString('en-US',{maximumFractionDigits:9});
  };
  const db=v=>Number.isFinite(v)?Number(v.toFixed(1)).toLocaleString('en-US'):'—';
  const count=v=>Math.round(v).toLocaleString('en-US');
  const duration=v=>!Number.isFinite(v)?'—':v<.001?`${num(v*1e6)} μs`:v<1?`${num(v*1e3)} ms`:`${num(v)} s`;
  function save(){try{sessionStorage.setItem(STORAGE,JSON.stringify({coded}));}catch(_){/* Calculations remain usable without persistence. */}}
  function numberField(name,label,min,max,step='any'){
    return `<label>${label}<input type="number" name="${name}" value="${esc(coded[name])}" min="${min}" max="${max}" step="${step}"></label>`;
  }
  function renderControls(){
    controls.innerHTML=`<h1 class="project-control-title">Project observation design</h1><p class="project-control-note">One BS → one UE · array elements below.<br>Default: 384 TX, 32 RX, 50 coded symbols.</p>
    <form id="setup" class="project-controls" autocomplete="off">
      <fieldset><legend>Antenna elements</legend><div class="input-grid">
        ${numberField('txY','BS along y',1,64,1)}${numberField('txZ','BS along z',1,64,1)}
        ${numberField('rxY','UE along y',1,64,1)}${numberField('rxZ','UE along z',1,64,1)}
      </div><p class="control-hint" id="array-count"></p></fieldset>
      <fieldset><legend>Coded sounding</legend><div class="input-grid">
        ${numberField('symbols','Symbols S',1,1024,1)}${numberField('tones','Subcarriers N',1,65536,1)}
        ${numberField('bandwidthMHz','Bandwidth B [MHz]',.001,5000)}${numberField('pilotSeed','QPSK seed',0,4294967295,1)}
      </div><p class="control-hint">One seeded QPSK X is reused across all tones. This is an illustrative X, not the saved NumPy pilot matrix.</p></fieldset>
      <fieldset class="wide"><legend>Power &amp; noise</legend><div class="input-grid">
        ${numberField('txPowerDbm','Total BS power [dBm]',-50,80,.5)}${numberField('nfDb','UE noise figure [dB]',0,30,.5)}
        ${numberField('lossDb','Effective path loss [dB]',0,240,.5)}${numberField('fcGHz','Carrier [GHz]',.01,300,.1)}
      </div><p class="control-hint">Conducted BS power is shared across TX ports and tones. Effective path loss describes a calibrated representative scalar path.</p></fieldset>
      <details><summary>Directions &amp; element spacing</summary><fieldset><legend>Local array-frame angles</legend><div class="input-grid">
        ${numberField('txAz','AoD azimuth [°]',-90,90,1)}${numberField('txEl','AoD elevation [°]',-90,90,1)}
        ${numberField('rxAz','AoA azimuth [°]',-90,90,1)}${numberField('rxEl','AoA elevation [°]',-90,90,1)}
        ${numberField('dLambda','Spacing d / λ',.001,5,.05)}
      </div><p class="control-hint">Arrays lie in yz. Broadside is +x. Azimuth turns toward +y.</p></fieldset></details>
    </form><p class="assumption-note">The estimator sees only noisy I/Q and the pilots X. All seven path parameters are unknown to it. The path set here is only where the CRB is evaluated. Delay is the apparent delay τg + b. No campus ray-tracing realization is loaded.</p>`;
    updateControlReadouts();
  }
  function updateControlReadouts(){
    if($('#array-count'))$('#array-count').innerHTML=`<strong>${count(coded.txY*coded.txZ)} TX</strong> · <strong>${count(coded.rxY*coded.rxZ)} RX</strong> · ${count(coded.symbols)} symbols.`;
  }
  function heading(eyebrow,title,subtitle){return `<div class="view-heading"><div><div class="eyebrow">${eyebrow}</div><h2>${title}</h2><p class="subtitle">${subtitle}</p></div></div>`;}
  function metric(label,value,unit,note,color='',extra=''){
    return `<article class="metric ${color} ${extra} ${value===Infinity?'unidentified':''}"><div class="metric-label">${label}</div><div class="metric-value">${num(value)}<span class="unit">${unit}</span></div><div class="metric-note">${value===Infinity?'Unidentifiable in this model':note}</div></article>`;
  }
  function invalid(r){return `<div class="alert"><strong>Please adjust the setup.</strong>${(r.errors||['The model could not be evaluated.']).map(esc).join('<br>')}</div>`;}
  const PARAMS=['Apparent delay','AoA azimuth','AoA elevation','AoD azimuth','AoD elevation','Path loss','Phase'];
  const SHORT=['τ ns','aᴿ °','eᴿ °','aᵀ °','eᵀ °','ℓ dB','φ rad'];
  const UNITS=['ns','deg','deg','deg','deg','dB','rad'];
  function rankNumber(r){return typeof r.rank==='number'?r.rank:(r.rank?.rank??r.rank?.value??r.parameters?.filter(p=>p.identifiable).length??0);}
  function heatColor(v,max){
    if(!Number.isFinite(v))return ['#EEF1F4','#7D8791'];
    const t=Math.min(1,local.raw?Math.log1p(Math.abs(v))/Math.max(Math.log1p(max),1e-15):Math.abs(v));
    const target=v<0?[232,114,12]:[24,116,184],base=[247,249,251];
    return [`rgb(${base.map((a,i)=>Math.round(a+(target[i]-a)*t)).join(',')})`,t>.65?'#fff':COLORS.ink];
  }
  function fisherTable(r){
    const matrix=local.raw?r.fim:r.correlation;
    const max=Math.max(...r.fim.flat().filter(Number.isFinite).map(Math.abs),1);
    return `<table class="matrix-table" aria-label="Full coded-pilot Fisher information matrix"><thead><tr><th></th>${SHORT.map(s=>`<th scope="col">${s}</th>`).join('')}</tr></thead><tbody>${matrix.map((row,i)=>`<tr><th scope="row">${SHORT[i]}</th>${row.map((v,j)=>{
      const [bg,fg]=heatColor(v,max);
      const text=local.raw?num(v,2):(Number.isFinite(v)?v.toFixed(2):'—');
      return `<td><button data-cell="${i},${j}" class="${local.cell[0]===i&&local.cell[1]===j?'selected':''}" style="background:${bg};color:${fg}" aria-label="${PARAMS[i]} and ${PARAMS[j]}, ${local.raw?'Fisher information':'normalized derivative similarity'} ${Number.isFinite(v)?v:'undefined'}">${text}</button></td>`;
    }).join('')}</tr>`).join('')}</tbody></table>`;
  }
  function codedView(r){
    const b=r.bounds,Nt=coded.txY*coded.txZ,Nr=coded.rxY*coded.rxZ,pilotRank=r.pilot?.rank??Math.min(Nt,coded.symbols),rank=rankNumber(r);
    const noise=r.signal.noiseDbm??r.signal.noiseBandwidthDbm;
    const time=r.grid.usefulTimeS??coded.symbols/(coded.bandwidthMHz*1e6/coded.tones);
    const [i,j]=local.cell;
    return heading('Your coded observation design · illustrative path','Structured bounds with fewer symbols than TX ports.','These are seven-parameter marginal √CRBs, using a seeded QPSK matrix X reused on every tone. The actual saved pilot matrix and campus channel realization are not loaded.')
      + `<div class="metric-grid project-metrics">
        ${metric('Apparent delay τg + b',b.tauNs,'ns','Contains receiver clock bias b')}
        ${metric('AoA · azimuth',b.rxAzDeg,'°','Arrival at the UE','teal')}
        ${metric('AoA · elevation',b.rxElDeg,'°','Arrival at the UE','teal')}
        ${metric('AoD · azimuth',b.txAzDeg,'°','Departure at the BS')}
        ${metric('AoD · elevation',b.txElDeg,'°','Departure at the BS')}
        ${metric('Effective path loss ℓ',b.lossDb,'dB','Unknown scalar path gain','orange')}
        ${metric('Path phase φ',b.phaseRad,'rad','Unknown reference phase','orange')}
        <article class="metric rank-card"><div class="metric-label">Pilot rank / TX ports</div><div class="metric-value">${count(pilotRank)} <small>/ ${count(Nt)}</small></div><div class="metric-note">Structured FIM rank: ${rank} / 7</div></article>
      </div>
      <div class="calculation-strip project-strip"><div><div class="small-label">Aggregate SNR Γ</div><div class="small-value">${db(r.signal.snrTotalDb??10*Math.log10(r.signal.gamma))} dB</div></div><div><div class="small-label">Receiver noise across B</div><div class="small-value">${db(noise)} dBm</div></div><div><div class="small-label">Useful duration S / Δf</div><div class="small-value">${duration(time)}</div></div><div><div class="small-label">Complex IQ observations</div><div class="small-value">${count(Nr*coded.symbols*coded.tones)}</div></div></div>
      <div class="insight project-insight"><strong>Two different identifiability questions.</strong> ${pilotRank<Nt?`Rank-${pilotRank} pilots cannot recover an unrestricted ${Nr} × ${Nt} channel matrix on each tone.`:`The pilot matrix has full TX row rank.`} The structured parameter Jacobian has rank ${rank} of 7.</div>
      <p class="provenance"><strong>Model scope:</strong> representative isolated far-field path, apparent delay coordinate, spatially white receiver noise. The seed produces this demo’s illustrative pilot matrix. Changing it changes illumination and parameter coupling. Actual project bounds require the saved X and channel/path realization.</p>
      ${r.warnings?.length?`<div class="insight orange">${r.warnings.map(esc).join('<br>')}</div>`:''}
      <hr class="section-divider"><div class="section-heading"><div><h3>The full Fisher matrix</h3><p>Coded pilots can couple AoD, path gain, and phase.</p></div><div class="segmented"><button data-matrix="normalized" class="${local.raw?'':'active'}">Normalized</button><button data-matrix="raw" class="${local.raw?'active':''}">Actual J</button></div></div>
      <div class="project-fisher"><div class="matrix-panel">${fisherTable(r)}<div class="matrix-legend"><span>${local.raw?'negative':'−1'}</span><div class="color-scale"></div><span>${local.raw?'positive':'+1'}</span></div><p class="footnote">${local.raw?'Actual entries carry reciprocal row/column units. Colors show signed logarithmic magnitude.':'Jᵢⱼ / √(JᵢᵢJⱼⱼ) compares whitened signal derivatives. It is not the correlation of estimation errors.'}</p></div>
      <div class="rank-explanation"><h3>${PARAMS[i]} × ${PARAMS[j]}</h3><div class="small-label field-gap">J<sub>${i+1},${j+1}</sub> [1 / (${UNITS[i]} · ${UNITS[j]})]</div><div class="selected-number">${num(r.fim[i][j],7)}</div><div class="small-label">Normalized derivative similarity</div><div class="selected-number">${num(r.correlation[i][j],5)}</div><div class="formula-box"><div class="formula">J = 2 Re(Dᴴ C<sub>n</sub>⁻¹ D)<br>σ<sub>i</sub> ≥ √[(J⁻¹)<sub>ii</sub>]</div></div><p class="footnote">The engine differentiates the coded signal and inverts the whole matrix. It does not impose the block zeros of the balanced orthogonal-pilot example. A singular parameter direction receives an infinite bound.</p></div></div>
      <p class="model-note">Every parameter above is unknown to the estimator, including the path gain and phase. The CRB is evaluated at the representative path on the left. Delay is apparent delay τg + b: the channel alone cannot separate the clock bias from the path delays. Geometry across several paths can, as the slides after this lab explain.</p>`;
  }
  function render(){
    updateControlReadouts();
    result=API.compute(coded);
    save();
    if(!result.valid){results.innerHTML=invalid(result);return;}
    results.innerHTML=codedView(result);
  }
  function schedule(){clearTimeout(timer);timer=setTimeout(render,60);}
  controls.addEventListener('submit',event=>event.preventDefault());
  controls.addEventListener('input',event=>{
    const t=event.target,key=t.name;if(!key||!(key in coded))return;
    if(t.value===''||!Number.isFinite(+t.value))return;coded[key]=+t.value;
    updateControlReadouts();schedule();
  });
  $('#reset').addEventListener('click',()=>{
    coded={...API.defaults};renderControls();render();
  });
  results.addEventListener('click',event=>{
    const t=event.target.closest('button');if(!t)return;
    if(t.dataset.cell){local.cell=t.dataset.cell.split(',').map(Number);const scroll=results.scrollTop;render();results.scrollTop=scroll;results.querySelector(`[data-cell="${local.cell.join(',')}"]`)?.focus({preventScroll:true});}
    if(t.dataset.matrix){local.raw=t.dataset.matrix==='raw';const scroll=results.scrollTop;render();results.scrollTop=scroll;}
  });
  renderControls();render();
})();
