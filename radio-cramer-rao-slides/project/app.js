/* Project-design and known-channel demonstrations. All calculations are local. */
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
  const CLOCK_DEFAULTS={tones:3300,bandwidthMHz:400,snrDb:30,spectralTilt:0,occupiedFraction:1,unknownPhase:true};
  const STORAGE='bailiping-radio-project-crb-v1';
  let saved={};
  try{saved=JSON.parse(sessionStorage.getItem(STORAGE)||'{}')||{};}catch(_){/* Storage can be disabled in embedded content. */}
  let coded={...API.defaults,...saved.coded};
  let known={...CLOCK_DEFAULTS,...saved.known};
  let lab=new URLSearchParams(location.search).get('lab')==='known'?'known':'coded';
  let result=null;
  let timer=null;
  const local={cell:[3,6],raw:false,delayNs:12,clockNs:8,referenceNs:20};
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
  function save(){try{sessionStorage.setItem(STORAGE,JSON.stringify({coded,known}));}catch(_){/* Calculations remain usable without persistence. */}}
  function numberField(name,label,min,max,step='any'){
    const state=lab==='coded'?coded:known;
    return `<label>${label}<input type="number" name="${name}" value="${esc(state[name])}" min="${min}" max="${max}" step="${step}"></label>`;
  }
  function renderControls(){
    if(lab==='coded'){
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
      </form><p class="assumption-note">Representative isolated far-field path. Delay is the apparent delay τg + b. Clock is not estimated separately. The seven-parameter Fisher matrix includes unknown effective path gain and phase. No campus ray-tracing realization is loaded.</p>`;
    }else{
      controls.innerHTML=`<h1 class="project-control-title">Known geometry, unknown clock</h1><p class="project-control-note">Hᵍ is known. A clock offset b still changes the received signal.</p>
      <form id="setup" class="project-controls" autocomplete="off">
        <fieldset><legend>Observed frequency grid</legend><div class="input-grid">
          ${numberField('tones','Subcarriers N',1,32768,1)}${numberField('bandwidthMHz','Bandwidth B [MHz]',.001,5000)}
        </div><p class="control-hint">Baseband offsets fₖ = (k − ⌊N/2⌋)B/N. This is the project’s frequency grid.</p></fieldset>
        <fieldset><legend>Available signal energy</legend>${numberField('snrDb','Aggregate SNR Γ [dB]',-100,150,.5)}<p class="control-hint">Γ is held fixed as the spectral shape changes. The weights describe the known Hᵍ and pilot spectrum after noise whitening.</p></fieldset>
        <fieldset class="wide"><legend>Where is the energy?</legend>
          <label class="range-label" for="spectral-tilt">Spectral tilt <output id="tilt-value">${num(known.spectralTilt)}</output></label>
          <input id="spectral-tilt" name="spectralTilt" type="range" min="-3" max="3" step=".05" value="${known.spectralTilt}"><div class="range-endpoints"><span>Lower frequencies</span><span>Higher frequencies</span></div>
          <label class="range-label field-gap" for="occupied-fraction">Occupied fraction <output id="occupied-value">${num(known.occupiedFraction*100)}%</output></label>
          <input id="occupied-fraction" name="occupiedFraction" type="range" min=".01" max="1" step=".01" value="${known.occupiedFraction}"><div class="range-endpoints"><span>Narrow spectrum</span><span>Full band</span></div>
        </fieldset>
        <fieldset class="wide"><legend>Is the common phase known?</legend><label class="wide-label">Selected clock bound<select name="unknownPhase"><option value="true" ${known.unknownPhase?'selected':''}>Unknown phase · eliminate nuisance</option><option value="false" ${known.unknownPhase?'':'selected'}>Known baseband reference phase</option></select></label><p class="control-hint">Both bounds stay visible. Knowing the reference phase here does not provide an absolute carrier-frequency timing reference.</p></fieldset>
      </form><p class="assumption-note">The spectrum is illustrative and normalized to the chosen Γ. No channel tensor is loaded. Scroll the result panel for four meanings of “known channel” and the geometric-delay / clock ambiguity.</p>`;
    }
    updateControlReadouts();
  }
  function updateControlReadouts(){
    if(lab==='coded'&&$('#array-count'))$('#array-count').innerHTML=`<strong>${count(coded.txY*coded.txZ)} TX</strong> · <strong>${count(coded.rxY*coded.rxZ)} RX</strong> · ${count(coded.symbols)} symbols.`;
    if(lab==='known'){
      if($('#tilt-value'))$('#tilt-value').textContent=num(known.spectralTilt);
      if($('#occupied-value'))$('#occupied-value').textContent=`${num(known.occupiedFraction*100)}%`;
    }
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
    const knownPhase=r.boundsKnownPhase?.tauNs;
    const knownGain=r.boundsKnownGainPhase?.txAzDeg;
    return heading('Your coded observation design · illustrative path','Structured bounds with fewer symbols than TX ports.','These are seven-parameter marginal √CRBs, using a seeded QPSK matrix X reused on every tone. The actual saved pilot matrix and campus channel realization are not loaded.')
      + `<div class="metric-grid project-metrics">
        ${metric('Apparent delay τg + b',b.tauNs,'ns','Geometric delay if clock known')}
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
      <div class="comparison-row"><div class="comparison-card"><strong>With known effective path phase</strong>Apparent-delay √CRB: <span class="mono">${num(knownPhase)} ns</span>.<br>The calibrated scalar phase is supplied exactly. Holding this nuisance parameter known can only retain or improve local information.</div><div class="comparison-card"><strong>With known effective complex path gain</strong>AoD azimuth √CRB: <span class="mono">${num(knownGain)}°</span>.<br>The calibrated scalar gain and phase are supplied exactly. This does not supply the complete geometric channel Hᵍ.</div></div>
      <p class="model-note">The seven-parameter engine estimates apparent delay τg + b. Interpreting it as geometric delay assumes a known clock offset b. If geometric delay and clock are both unknown, their phase derivatives coincide. The next laboratory separates that issue from whether H is called “known.”</p>`;
  }
  function knownView(r){
    const b=r.bounds,ratio=b.unknownPhaseNs/b.knownPhaseNs;
    const rms=r.rmsFrequencyHz??Math.sqrt(r.secondMomentHz2);
    return heading('A calibrated geometric channel still leaves synchronization','Known Hᵍ does not mean the clock is known.','The known channel and pilots set the frequency weights. Estimate a residual clock b from noisy IQ, with the common baseband phase either known or unknown.')
      + `<div class="known-metrics">
        ${metric('Clock b · phase known',b.knownPhaseNs,'ns','Uses weighted mean-square frequency','',known.unknownPhase?'':'selected-bound')}
        ${metric('Clock b · phase unknown',b.unknownPhaseNs,'ns','Uses centered weighted bandwidth','teal',known.unknownPhase?'selected-bound':'')}
        ${metric('Cost of an unknown phase',ratio,'×','Ratio of clock √CRBs','orange')}
      </div><div class="known-top"><div class="canvas-card spectrum-card"><canvas id="spectrum" role="img" aria-label="Known-channel weighted frequency spectrum and its weighted mean"></canvas></div><div class="moment-card"><div><div class="small-label">Weighted frequency mean μ<sub>f</sub></div><div class="small-value">${num(r.meanFrequencyHz/1e6,5)} MHz</div></div><div><div class="small-label">Centered RMS bandwidth β</div><div class="small-value">${num(r.betaHz/1e6,5)} MHz</div></div><div><div class="small-label">Uncentered RMS frequency</div><div class="small-value">${num(rms/1e6,5)} MHz</div></div><div class="formula">∑ w<sub>k</sub> = 1<br>β² = ∑ w<sub>k</sub>(f<sub>k</sub> − μ<sub>f</sub>)²<br>${count(r.grid.activeTones??known.tones)} active tones · Γ = ${db(known.snrDb)} dB</div></div></div>
      <div class="insight project-insight"><strong>Try a strong spectral tilt.</strong> When phase is unknown, a common phase change absorbs the information at the weighted frequency mean. Concentrating energy into a narrow band leaves less spread for estimating the clock, even at fixed Γ.</div>
      <div class="formula-box field-gap"><div class="clock-formula mono">Phase known: σ<sub>b</sub> ≥ [8π²Γ ∑ w<sub>k</sub>f<sub>k</sub>²]⁻¹ᐟ²<br>Phase unknown: σ<sub>b</sub> ≥ [8π²Γ ∑ w<sub>k</sub>(f<sub>k</sub> − μ<sub>f</sub>)²]⁻¹ᐟ²</div></div>
      <p class="model-note">The equations above produce seconds before conversion to ns. f<sub>k</sub> is a <strong>baseband offset</strong>, using the project’s actual grid convention. A known phase here is a known complex reference phase in that model. It is not an absolute carrier-frequency synchronization assumption. The weighted spectral envelope is illustrative.</p>
      <hr class="section-divider"><div class="section-heading"><div><h3>Four meanings of “the channel is known”</h3><p>The observation and the remaining unknowns determine which CRB applies.</p></div></div>
      <div class="known-meanings">
        <article class="meaning-card"><div class="meaning-label">01 · Simulator truth</div><h3>H is known by the generator.</h3><p>The simulator evaluates H(η), generates Y = H(η)X + N, and records the true parameters. An estimator given Y and X still has unknown η and noise.</p><p><strong>Use an unknown-parameter CRB</strong> evaluated at that truth. The coded-pilot laboratory illustrates this setting.</p><a href="?lab=coded" data-switch="coded">Inspect the coded observation model →</a></article>
        <article class="meaning-card active"><div class="meaning-label">02 · Known geometric Hᵍ</div><h3>The residual clock remains unknown.</h3><p>With calibrated geometry and gain, Y<sub>k</sub> = e<sup>−j2πfₖb</sup>Hᵍ<sub>k</sub>X + N<sub>k</sub>. The same X is reused on all tones. Changing b still changes the data distribution.</p><p><strong>Use the clock bound above.</strong> If a common phase is also unknown, remove its nuisance information by centering the weighted frequencies.</p></article>
        <article class="meaning-card"><div class="meaning-label">03 · Exact effective H</div><h3>Exact data still need a unique interpretation.</h3><p>An exact supplied H<sup>eff</sup> has no observation noise for its identifiable functions. If geometry is also known, it may identify the clock locally.</p><p>When geometric delay and clock are both unknown, an isolated path reveals their sum. <strong>Exact H<sup>eff</sup> does not separate that sum</strong> without extra information. The sliders below show why.</p></article>
        <article class="meaning-card"><div class="meaning-label">04 · Noisy estimated CSI</div><h3>Ĥ has an error covariance.</h3><p>Write Ĥ = H<sup>eff</sup> + E and retain C<sub>H</sub> = Cov(vec E). Derivatives of H<sup>eff</sup> must be weighted by this covariance.</p><p><strong>Use J = 2 Re(GᴴC<sub>H</sub>⁻¹G)</strong> for parameter-independent complex Gaussian CSI error. An unrestricted per-tone channel cannot be uniquely recovered from 50 symbols and 384 TX ports without additional structure.</p></article>
      </div>
      <section class="ambiguity-panel" aria-label="Geometric delay and clock ambiguity"><div class="section-heading"><div><h3>Different delay and clock. The same effective phase.</h3><p>For one path, the phase ramp depends on τ<sub>g</sub> + b.</p></div></div>
      <div class="ambiguity-controls"><div class="ambiguity-control"><label for="geometric-delay">Geometric delay τ<sub>g</sub> [ns]<input id="geometric-delay" type="number" min="0" max="50" step=".1" value="${local.delayNs}"></label><input id="geometric-slider" type="range" min="0" max="50" step=".1" value="${local.delayNs}" aria-label="Geometric delay in nanoseconds"></div><div class="ambiguity-control clock"><label for="clock-delay">Clock offset b [ns]<input id="clock-delay" type="number" min="-50" max="50" step=".1" value="${local.clockNs}"></label><input id="clock-slider" type="range" min="-50" max="50" step=".1" value="${local.clockNs}" aria-label="Clock offset in nanoseconds"></div></div>
      <div class="ambiguity-status"><span>Effective delay τ<sub>g</sub> + b = <span id="effective-delay" class="sum">${num(local.delayNs+local.clockNs)} ns</span></span><button class="small-button" data-transfer>Keep sum fixed: shift 5 ns</button><button class="reset" data-reference>Set current curve as reference</button></div><div class="canvas-card phase-card"><canvas id="phase" role="img" aria-label="Effective phase versus baseband frequency for the current delay sum and the saved reference"></canvas></div><p id="phase-status" class="footnote"></p>
      <div class="insight orange">If τ<sub>g</sub> and b are both unknown, ∂μ/∂τ<sub>g</sub> = ∂μ/∂b. Their Fisher columns are identical, so neither has a finite marginal bound without an additional constraint. Knowing geometric Hᵍ fixes τ<sub>g</sub> and removes this specific ambiguity.</div><p class="model-note">This plot shows unwrapped phase for interpretation. The measured phase is modulo 2π, and a uniformly spaced frequency grid also has delay aliases. The bounds above describe local precision around one valid solution.</p></section>`;
  }
  function render(){
    updateControlReadouts();
    result=lab==='coded'?API.compute(coded):API.clock(known);
    save();
    if(!result.valid){results.innerHTML=invalid(result);return;}
    results.innerHTML=lab==='coded'?codedView(result):knownView(result);
    if(lab==='known'){drawSpectrum();drawPhase();}
  }
  function schedule(){clearTimeout(timer);timer=setTimeout(render,60);}
  function switchLab(next,updateURL=true){
    lab=next==='known'?'known':'coded';
    for(const button of document.querySelectorAll('[data-lab]')){
      const active=button.dataset.lab===lab;button.setAttribute('aria-selected',String(active));button.tabIndex=active?0:-1;
    }
    results.setAttribute('aria-labelledby',`tab-${lab}`);
    if(updateURL){try{const url=new URL(location.href);url.searchParams.set('lab',lab);history.replaceState(null,'',url);}catch(_){/* Restricted iframe URL. */}}
    controls.scrollTop=0;results.scrollTop=0;renderControls();render();
  }
  function surface(id){
    const canvas=document.getElementById(id);if(!canvas)return null;
    const box=canvas.getBoundingClientRect(),w=Math.max(10,box.width),h=Math.max(10,box.height),dpr=Math.min(devicePixelRatio||1,2);
    canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);
    const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);ctx.lineJoin='round';ctx.lineCap='round';
    return {canvas,ctx,w,h};
  }
  function drawSpectrum(){
    const s=surface('spectrum');if(!s||!result?.valid)return;
    const {ctx,w,h}=s,weights=result.weights;
    if(!weights?.length)return;
    const large=w>550&&h>250,font=large?12:10;
    const margin={left:48,right:18,top:32,bottom:38},pw=w-margin.left-margin.right,ph=h-margin.top-margin.bottom;
    const fmin=-known.bandwidthMHz/2,fmax=known.bandwidthMHz/2,N=weights.length;
    const ymax=Math.max(1,...weights.map(p=>p.weight*N))*1.12;
    const x=f=>margin.left+(f/1e6-fmin)/(fmax-fmin)*pw,y=v=>margin.top+(1-v/ymax)*ph;
    const tickN=pw<400?4:6;
    ctx.font=`${font}px Arial`;ctx.fillStyle=COLORS.muted;ctx.strokeStyle='#EAF0F4';ctx.lineWidth=1;
    for(let i=0;i<=tickN;i++){
      const f=fmin+(fmax-fmin)*i/tickN,px=x(f*1e6);ctx.beginPath();ctx.moveTo(px,margin.top);ctx.lineTo(px,margin.top+ph);ctx.stroke();ctx.textAlign='center';ctx.fillText(num(f,3),px,h-21);
    }
    for(let i=0;i<=3;i++){
      const value=ymax*i/3;ctx.beginPath();ctx.moveTo(margin.left,y(value));ctx.lineTo(w-margin.right,y(value));ctx.stroke();ctx.textAlign='right';ctx.fillText(num(value,2),margin.left-7,y(value)+3);
    }
    ctx.beginPath();ctx.moveTo(x(weights[0].frequencyHz),y(0));
    const stride=Math.max(1,Math.floor(N/Math.max(pw*2,1)));
    for(let i=0;i<N;i+=stride){const p=weights[i];ctx.lineTo(x(p.frequencyHz),y(p.weight*N));}
    const last=weights[N-1];ctx.lineTo(x(last.frequencyHz),y(last.weight*N));ctx.lineTo(x(last.frequencyHz),y(0));ctx.closePath();ctx.fillStyle='rgba(24,116,184,.14)';ctx.fill();
    ctx.beginPath();let first=true;for(let i=0;i<N;i+=stride){const p=weights[i];first?ctx.moveTo(x(p.frequencyHz),y(p.weight*N)):ctx.lineTo(x(p.frequencyHz),y(p.weight*N));first=false;}ctx.strokeStyle=COLORS.blue;ctx.lineWidth=2;ctx.stroke();
    const meanX=x(result.meanFrequencyHz);ctx.beginPath();ctx.moveTo(meanX,margin.top);ctx.lineTo(meanX,margin.top+ph);ctx.strokeStyle=COLORS.orange;ctx.lineWidth=1.6;ctx.setLineDash([5,4]);ctx.stroke();ctx.setLineDash([]);
    const zeroX=x(0);ctx.beginPath();ctx.moveTo(zeroX,margin.top);ctx.lineTo(zeroX,margin.top+ph);ctx.strokeStyle='#8292A0';ctx.lineWidth=1;ctx.setLineDash([2,4]);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle=COLORS.blue;ctx.textAlign='left';ctx.font=`${font}px Arial`;ctx.fillText('Known-channel spectral energy',margin.left,17);ctx.textAlign='right';ctx.fillStyle=COLORS.orange;ctx.fillText('μf',Math.min(w-margin.right,Math.max(margin.left+20,meanX+12)),margin.top+15);
    ctx.textAlign='center';ctx.fillStyle=COLORS.muted;ctx.fillText('Baseband frequency offset fₖ [MHz]',margin.left+pw/2,h-5);
    ctx.save();ctx.translate(12,margin.top+ph/2);ctx.rotate(-Math.PI/2);ctx.fillText('Relative energy N × wₖ',0,0);ctx.restore();ctx.textAlign='left';
  }
  function drawPhase(){
    const s=surface('phase');if(!s||!result?.valid)return;const {ctx,w,h}=s;
    const margin={left:49,right:18,top:30,bottom:38},pw=w-margin.left-margin.right,ph=h-margin.top-margin.bottom;
    const sum=local.delayNs+local.clockNs,reference=local.referenceNs,B=known.bandwidthMHz;
    const fmin=-B/2,fmax=B/2,maxPhase=Math.max(.1,Math.abs(sum)*B/2e3,Math.abs(reference)*B/2e3)*1.2;
    const x=f=>margin.left+(f-fmin)/(fmax-fmin)*pw,y=phase=>margin.top+ph/2-phase/maxPhase*ph/2;
    const font=w>550&&h>250?12:10;ctx.font=`${font}px Arial`;ctx.lineWidth=1;
    for(let i=0;i<=4;i++){const f=fmin+B*i/4;ctx.strokeStyle='#EAF0F4';ctx.beginPath();ctx.moveTo(x(f),margin.top);ctx.lineTo(x(f),margin.top+ph);ctx.stroke();ctx.textAlign='center';ctx.fillStyle=COLORS.muted;ctx.fillText(num(f),x(f),h-20);}
    for(let i=-2;i<=2;i++){const p=maxPhase*i/2;ctx.beginPath();ctx.moveTo(margin.left,y(p));ctx.lineTo(w-margin.right,y(p));ctx.stroke();ctx.textAlign='right';ctx.fillText(num(p,2),margin.left-7,y(p)+3);}
    for(const [delay,color,dash,width] of [[reference,'#7C8A96',[5,5],2],[sum,COLORS.blue,[],2.5]]){
      ctx.beginPath();ctx.moveTo(x(fmin),y(-fmin*delay/1e3));ctx.lineTo(x(fmax),y(-fmax*delay/1e3));ctx.strokeStyle=color;ctx.setLineDash(dash);ctx.lineWidth=width;ctx.stroke();ctx.setLineDash([]);
    }
    ctx.textAlign='left';ctx.fillStyle=COLORS.blue;ctx.fillText(`— current sum ${num(sum)} ns`,margin.left,16);ctx.fillStyle='#6B7883';ctx.textAlign='right';ctx.fillText(`-- reference ${num(reference)} ns`,w-margin.right,16);
    ctx.textAlign='center';ctx.fillStyle=COLORS.muted;ctx.fillText('Baseband frequency [MHz]',margin.left+pw/2,h-5);
    ctx.save();ctx.translate(12,margin.top+ph/2);ctx.rotate(-Math.PI/2);ctx.fillText('Unwrapped phase / 2π',0,0);ctx.restore();ctx.textAlign='left';
    const equal=Math.abs(sum-reference)<1e-9;
    const status=$('#phase-status');if(status)status.innerHTML=equal?'<strong>The two phase curves coincide exactly.</strong> Different geometric delay and clock values can produce this same sum.':`The current sum differs from the reference by <strong>${num(sum-reference)} ns</strong>, so its phase slope changes. Use “Keep sum fixed” to change both parameters without changing this curve.`;
  }
  controls.addEventListener('submit',event=>event.preventDefault());
  controls.addEventListener('input',event=>{
    const t=event.target,key=t.name,state=lab==='coded'?coded:known;if(!key||!(key in state))return;
    if(key==='unknownPhase')state[key]=t.value==='true';
    else{if(t.value===''||!Number.isFinite(+t.value))return;state[key]=+t.value;}
    updateControlReadouts();schedule();
  });
  document.querySelector('.lab-tabs').addEventListener('click',event=>{const tab=event.target.closest('[data-lab]');if(tab)switchLab(tab.dataset.lab);});
  document.querySelector('.lab-tabs').addEventListener('keydown',event=>{
    if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;event.preventDefault();event.stopPropagation();
    switchLab(event.key==='Home'?'coded':event.key==='End'?'known':lab==='coded'?'known':'coded');$(`#tab-${lab}`).focus();
  });
  $('#reset').addEventListener('click',()=>{
    if(lab==='coded')coded={...API.defaults};else known={...CLOCK_DEFAULTS};
    local.delayNs=12;local.clockNs=8;local.referenceNs=20;renderControls();render();
  });
  results.addEventListener('click',event=>{
    const a=event.target.closest('[data-switch]');if(a){event.preventDefault();switchLab(a.dataset.switch);return;}
    const t=event.target.closest('button');if(!t)return;
    if(t.dataset.cell){local.cell=t.dataset.cell.split(',').map(Number);const scroll=results.scrollTop;render();results.scrollTop=scroll;results.querySelector(`[data-cell="${local.cell.join(',')}"]`)?.focus({preventScroll:true});}
    if(t.dataset.matrix){local.raw=t.dataset.matrix==='raw';const scroll=results.scrollTop;render();results.scrollTop=scroll;}
    if(t.hasAttribute('data-transfer')){
      const delta=local.delayNs<=45&&local.clockNs>=-45?5:-5;
      local.delayNs+=delta;local.clockNs-=delta;syncAmbiguity();drawPhase();
    }
    if(t.hasAttribute('data-reference')){local.referenceNs=local.delayNs+local.clockNs;drawPhase();}
  });
  function syncAmbiguity(){
    if(!$('#geometric-delay'))return;
    $('#geometric-delay').value=Number(local.delayNs.toFixed(2));$('#geometric-slider').value=local.delayNs;
    $('#clock-delay').value=Number(local.clockNs.toFixed(2));$('#clock-slider').value=local.clockNs;
    $('#effective-delay').textContent=`${num(local.delayNs+local.clockNs)} ns`;
  }
  results.addEventListener('input',event=>{
    const t=event.target;if(!['geometric-delay','geometric-slider','clock-delay','clock-slider'].includes(t.id)||t.value==='')return;
    if(t.id.startsWith('geometric'))local.delayNs=Math.min(50,Math.max(0,+t.value));else local.clockNs=Math.min(50,Math.max(-50,+t.value));
    syncAmbiguity();drawPhase();
  });
  let resizePending=false;
  const redraw=()=>{if(lab!=='known'||resizePending)return;resizePending=true;requestAnimationFrame(()=>{resizePending=false;drawSpectrum();drawPhase();});};
  window.addEventListener('resize',redraw);
  window.addEventListener('bento-live-visibility',event=>{if(!event.detail?.paused)redraw();});
  if(typeof ResizeObserver!=='undefined')new ResizeObserver(redraw).observe(results);
  switchLab(lab,false);
})();
