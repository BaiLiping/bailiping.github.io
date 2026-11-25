/* Interactive views for the exact single-path Fisher-information model. */
(() => {
  'use strict';
  const API = window.RadioCRB;
  const results = document.getElementById('results');
  if (!API) {
    results.innerHTML = '<div class="error-fallback"><h2>The model did not load</h2><p>Please reload this page to load the local calculation engine.</p></div>';
    return;
  }
  const C = 299792458;
  const DEG = Math.PI / 180;
  const COLORS = { ink:'#16222E', muted:'#586674', blue:'#1874B8', teal:'#0A6B5E', orange:'#E8720C', line:'#DCE3E9' };
  const LABS = ['calculator', 'geometry', 'fisher', 'bandwidth', 'multipath'];
  const STORAGE = 'bailiping-radio-crb-v1';
  let saved = null;
  try { saved = JSON.parse(sessionStorage.getItem(STORAGE) || 'null'); } catch (_) { /* Iframe storage can be unavailable. */ }
  let state = { ...API.defaults, ...(saved && typeof saved === 'object' ? saved : {}) };
  let lab = new URLSearchParams(location.search).get('lab') || 'calculator';
  if (!LABS.includes(lab)) lab = 'calculator';
  let current;
  let pending = false;
  const local = { array:'rx', yaw:-.65, pitch:.36, matrix:'correlation', cell:[1,2], separation:.5, geometryScale:1 };
  const form = document.getElementById('setup');
  const $ = selector => document.querySelector(selector);
  const input = name => form.elements.namedItem(name);
  const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const number = (v, digits=3) => {
    if (v === Infinity || v === -Infinity) return '∞';
    if (!Number.isFinite(v)) return '—';
    if (v === 0) return '0';
    if (Math.abs(v) < .001 || Math.abs(v) >= 100000) return v.toExponential(digits - 1).replace(/e\+/, 'e');
    return Number(v.toPrecision(digits)).toLocaleString('en-US', {maximumFractionDigits:8});
  };
  const db = v => Number.isFinite(v) ? Number(v.toFixed(1)).toLocaleString('en-US') : '—';
  const integer = v => Math.round(v).toLocaleString('en-US');
  const duration = v => v < .001 ? `${number(v*1e6)} μs` : v < 1 ? `${number(v*1e3)} ms` : `${number(v)} s`;
  const finite = v => Number.isFinite(v) && v >= 0;
  const persist = () => { try { sessionStorage.setItem(STORAGE, JSON.stringify(state)); } catch (_) { /* Keep interactive without storage. */ } };

  function synchronizeInputs() {
    for (const key of Object.keys(API.defaults)) {
      const field = input(key);
      if (field) field.value = state[key];
    }
    synchronizeDerived();
  }
  function synchronizeDerived() {
    const K = +state.tones;
    if (state.gridMode === 'spacing') state.bandwidthMHz = K * +state.spacingKHz / 1000;
    else state.spacingKHz = +state.bandwidthMHz * 1000 / K;
    input('bandwidthMHz').disabled = state.gridMode === 'spacing';
    input('spacingKHz').disabled = state.gridMode !== 'spacing';
    const derivedGrid = state.gridMode === 'spacing' ? 'bandwidthMHz' : 'spacingKHz';
    input(derivedGrid).value = +Number(state[derivedGrid]).toPrecision(9);
    const lambdaMm = C / (+state.fcGHz * 1e9) * 1000;
    if (state.spacingMode === 'physical') state.dLambda = +state.spacingMm / lambdaMm;
    else state.spacingMm = +state.dLambda * lambdaMm;
    input('dLambda').disabled = state.spacingMode === 'physical';
    input('spacingMm').disabled = state.spacingMode !== 'physical';
    const derivedSpacing = state.spacingMode === 'physical' ? 'dLambda' : 'spacingMm';
    input(derivedSpacing).value = +Number(state[derivedSpacing]).toPrecision(8);
    $('#link-fields').hidden = state.mode !== 'link';
    $('#snr-field').hidden = state.mode === 'link';
    $('#snr-label').textContent = state.mode === 'total' ? 'Aggregate SNR Γ [dB]' : 'SNR per RX / tone / symbol [dB]';
    $('#antenna-count').innerHTML = `<strong>${integer(state.txY*state.txZ)} TX</strong> and <strong>${integer(state.rxY*state.rxZ)} RX</strong> ports. Counts are elements, not separate BSs or UEs.`;
    $('#energy-hint').textContent = state.mode === 'total'
      ? 'The total information-bearing signal energy is fixed. Adding observations alone does not add energy.'
      : state.mode === 'perTone'
        ? 'ρ is the received SNR per RX element, per tone, per symbol, summed over the pilot TX energy.'
        : 'Power is total conducted BS power across all TX elements and tones. Effective path loss is attenuation of this calibrated scalar path, including propagation and element gains. Noise uses 290 K.';
  }
  function schedule() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; render(); });
  }
  function usePreset(name) {
    state = { ...API.defaults };
    if (name === 'large') Object.assign(state, {txY:24,txZ:16,symbols:384});
    if (name === 'ula') Object.assign(state, {txY:16,txZ:1,rxY:8,rxZ:1,symbols:16});
    if (name === 'broadside') Object.assign(state, {txAz:0,txEl:0,rxAz:0,rxEl:0});
    synchronizeInputs();
    persist();
    render();
  }
  function switchLab(next, updateURL=true) {
    lab = LABS.includes(next) ? next : 'calculator';
    for (const button of document.querySelectorAll('[data-lab]')) {
      const active = button.dataset.lab === lab;
      button.id = `tab-${button.dataset.lab}`;
      button.setAttribute('aria-controls', 'results');
      button.setAttribute('aria-selected', active ? 'true' : 'false');
      button.tabIndex = active ? 0 : -1;
    }
    if (updateURL) {
      try { const url = new URL(location.href); url.searchParams.set('lab', lab); history.replaceState(null, '', url); } catch (_) { /* Restricted iframe. */ }
    }
    results.scrollTop = 0;
    results.setAttribute('aria-labelledby', `tab-${lab}`);
    render();
  }
  function heading(eyebrow, title, subtitle, tag='') {
    return `<div class="view-heading"><div><div class="eyebrow">${eyebrow}</div><h2>${title}</h2><p class="subtitle">${subtitle}</p></div>${tag ? `<span class="status-pill ${tag.includes('unidentifiable') ? 'warning' : ''}">${tag}</span>` : ''}</div>`;
  }
  function alertHTML(r) {
    if (!r.valid) {
      const messages = r.errors?.length ? r.errors : ['This setup does not satisfy the model assumptions.'];
      return `<div class="alert"><strong>This setup cannot support the stated sounding model.</strong><ul class="warning-list">${messages.map(m=>`<li>${esc(m)}</li>`).join('')}</ul>${state.symbols < state.txY*state.txZ ? `<button class="reset" data-fix-training style="margin-top:9px">Use ${integer(state.txY*state.txZ)} symbols for ${integer(state.txY*state.txZ)} TX ports</button>` : ''}</div>`;
    }
    return '';
  }
  function warningsHTML(r) {
    if (!r.warnings?.length) return '';
    return `<div class="insight orange">${r.warnings.map(esc).join('<br>')}</div>`;
  }
  function metric(label, value, unit, note, color='') {
    return `<article class="metric ${color} ${value===Infinity?'unidentified':''}"><div class="metric-label">${label}</div><div class="metric-value">${number(value)}<span class="unit">${unit}</span></div><div class="metric-note">${value===Infinity?'Not jointly identifiable here':note}</div></article>`;
  }
  function calculator(r) {
    const b = r.bounds;
    const missing = r.parameters.filter(p=>!p.identifiable).length;
    const count = state.rxY*state.rxZ*state.tones*state.symbols;
    return heading('Single-path measurement precision', 'How small could the uncertainty be?', 'Each number is √CRB: a lower bound on a locally unbiased estimator’s marginal standard deviation.', missing ? `${missing} unidentifiable` : '7 parameters · joint FIM')
      + `<div class="metric-grid">
        ${metric('Delay τ',b.tauNs,'ns','Frequency slope','')}
        ${metric('AoA · azimuth',b.rxAzDeg,'°','Arrival at the UE','teal')}
        ${metric('AoA · elevation',b.rxElDeg,'°','Arrival at the UE','teal')}
        ${metric('AoD · azimuth',b.txAzDeg,'°','Departure at the BS','')}
        ${metric('AoD · elevation',b.txElDeg,'°','Departure at the BS','')}
        ${metric('Effective path loss ℓ',b.lossDb,'dB','Calibrated scalar path gain','orange')}
        ${metric('Path phase φ',b.phaseRad,'rad','At array / frequency center','orange')}
        ${metric('Path-length equivalent cτ',b.tauNs*C/1e9,'m','One-way / bistatic · known clock','teal')}
      </div>
      <div class="calculation-strip">
        <div><div class="small-label">Aggregate SNR Γ</div><div class="small-value">${db(r.signal.snrTotalDb)} dB</div></div>
        <div><div class="small-label">RMS bandwidth β</div><div class="small-value">${number(r.grid.betaHz/1e6)} MHz</div></div>
        <div><div class="small-label">Complex RX observations</div><div class="small-value">${integer(count)}</div></div>
        <div><div class="small-label">Useful coherent duration L/Δf</div><div class="small-value">${duration(r.grid.usefulTimeS)}</div></div>
      </div>
      <p class="footnote"><span class="mono">Γ = N<sub>r</sub>K Lρ</span>. Here ρ = <strong>${db(r.signal.snrPerToneDb)} dB</strong> per RX / tone / symbol. Δf = <strong>${number(r.grid.spacingHz/1e3,5)} kHz</strong>, occupied tone span = <strong>${number(r.grid.spanHz/1e6,6)} MHz</strong>.${state.mode==='link'?` Receiver noise: <strong>${db(r.signal.noiseDbm)} dBm</strong> across B, <strong>${db(r.signal.noisePerToneDbm)} dBm</strong> per tone.`:''}</p>
      <div class="insight"><strong>Try this:</strong> double B with aggregate Γ fixed. The delay bound halves. Double an array’s side length to improve its angular aperture. Increase L at fixed Γ and observe that the bounds stay unchanged.</div>
      ${warningsHTML(r)}
      <p class="footnote">The arrival and departure angle pairs are inverted jointly. The delay bound uses the spread of the tones around their mean because the unknown path phase absorbs their common phase. Interpreting cτ as geometric path length assumes the clock offset is known. It is not a UE position bound.</p>`;
  }
  function geometry(r) {
    const tx = local.array === 'tx';
    const group = tx ? 'tx' : 'rx';
    const azKey = tx ? 'txAzDeg' : 'rxAzDeg';
    const elKey = tx ? 'txElDeg' : 'rxElDeg';
    const identifiable = finite(r.bounds[azKey]) && finite(r.bounds[elKey]);
    return heading('Spatial phase gradients', 'The array measures a direction through phase.', 'Rotate the view, then change the angles or array shape. The orange arrow is expressed in this array’s local frame.')
      + `<div class="inline-controls geometry-controls"><div class="segmented" aria-label="Array to inspect"><button data-array="rx" class="${tx?'':'active'}">UE · AoA</button><button data-array="tx" class="${tx?'active':''}">BS · AoD</button></div><label>View yaw <input id="camera-yaw" type="range" min="-180" max="180" step="1" value="${Math.round(local.yaw/DEG)}" aria-label="Camera yaw"></label><label>Tilt <input id="camera-pitch" type="range" min="-70" max="70" step="1" value="${Math.round(local.pitch/DEG)}" aria-label="Camera tilt"></label><button class="reset" data-camera-reset>Reset view</button></div>
      <div class="canvas-card geometry-card"><canvas id="geometry-canvas" aria-label="Rotatable three-dimensional array and local angular uncertainty contour" role="img"></canvas><div class="canvas-hint">Drag to rotate · ${integer(r.geometry[group].count)} elements<br>Array plane: yz · broadside: +x</div><div id="geometry-badge" class="canvas-badge">${identifiable ? '1σ local covariance contour' : 'Joint angle pair is unidentifiable'}</div></div>
      <div class="geometry-footer"><div class="compact-box"><strong>${tx?'AoD at BS':'AoA at UE'} · marginal bounds</strong>Azimuth: <span class="mono">${number(r.bounds[azKey])}°</span> &nbsp; Elevation: <span class="mono">${number(r.bounds[elKey])}°</span><br>d = ${number(r.geometry.spacingM*1000)} mm = ${number(r.geometry.dLambda)}λ, f<sub>c</sub> = ${number(state.fcGHz)} GHz.</div><div class="compact-box"><strong>Read the contour locally</strong>The ellipse uses the joint 2 × 2 angular covariance and its correlation. Its display is scaled by the stated factor. It is a Mahalanobis-radius-1 contour, enclosing about 39.3% for a 2D Gaussian.</div></div>
      <p class="footnote">A yz array observes direction components u<sub>y</sub> and u<sub>z</sub>. A ULA supplies only one component, so generic azimuth and elevation cannot both be recovered. Planar geometry also has a front/back ambiguity unless a hemisphere is known.</p>`;
  }

  const PARAM_SHORT = ['τ ns','aᴿ °','eᴿ °','aᵀ °','eᵀ °','ℓ dB','φ rad'];
  const PARAM_NAMES = ['delay','AoA azimuth','AoA elevation','AoD azimuth','AoD elevation','path loss','path phase'];
  const PARAM_UNITS = ['ns','deg','deg','deg','deg','dB','rad'];
  function heatColor(v, raw=false, max=1) {
    if (!Number.isFinite(v)) return {bg:'#F0F2F4',fg:'#7A858E'};
    const magnitude = raw ? Math.log1p(Math.abs(v))/Math.max(Math.log1p(max),1e-12) : Math.abs(v);
    const t = Math.min(1, Math.max(0,magnitude));
    const target = v < 0 ? [232,114,12] : [24,116,184];
    const base = [247,249,251];
    return {bg:`rgb(${base.map((b,i)=>Math.round(b+(target[i]-b)*t)).join(',')})`,fg:t>.64?'#fff':COLORS.ink};
  }
  function fisher(r) {
    const normalized = local.matrix === 'correlation';
    const matrix = normalized ? r.correlation : r.fim;
    const max = Math.max(...r.fim.flat().filter(Number.isFinite).map(Math.abs),1e-12);
    const [i,j] = local.cell;
    const value = r.fim[i][j];
    const corr = r.correlation[i][j];
    const angle = i===3 || i===4 || j===3 || j===4 ? 3 : 1;
    const jaa = r.fim[angle][angle], jee=r.fim[angle+1][angle+1], jae=r.fim[angle][angle+1];
    const azBound = r.bounds[angle===1?'rxAzDeg':'txAzDeg'];
    const rows = matrix.map((row,a)=>`<tr><th scope="row">${PARAM_SHORT[a]}</th>${row.map((v,b)=>{
      const col=heatColor(v,!normalized,max);
      const val=normalized?(Number.isFinite(v)?Number(v.toFixed(2)).toFixed(2):'—'):number(v,2);
      return `<td><button data-cell="${a},${b}" class="${a===i&&b===j?'selected':''}" style="background:${col.bg};color:${col.fg}" aria-label="${PARAM_NAMES[a]} and ${PARAM_NAMES[b]}: ${normalized?'normalized derivative inner product':'Fisher information'} ${Number.isFinite(v)?v:'undefined'}">${val}</button></td>`;
    }).join('')}</tr>`).join('');
    return heading('Differentiate → compare → invert', 'Information is a matrix, not seven separate numbers.', 'A matrix entry compares two changes in the noiseless IQ signal. Select any cell to inspect its value.')
      + `<div class="inline-controls"><div class="segmented"><button data-matrix="correlation" class="${normalized?'active':''}">Normalized derivative inner products</button><button data-matrix="fim" class="${normalized?'':'active'}">Actual FIM J</button></div></div>
      <div class="fisher-layout"><div class="matrix-panel"><table class="matrix-table" aria-label="Seven-parameter Fisher information matrix"><thead><tr><th></th>${PARAM_SHORT.map(s=>`<th scope="col">${s}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table><div class="matrix-legend"><span>${normalized?'−1':'negative'}</span><div class="color-scale"></div><span>${normalized?'+1':'positive'}</span></div><p class="footnote">${normalized?'Jᵢⱼ / √(JᵢᵢJⱼⱼ). This is correlation between signal derivatives, not correlation between estimation errors.':'Units depend on the row and column. Colors use a signed logarithmic magnitude scale because these entries have different units.'} A dash means a derivative has zero norm.</p><p class="footnote"><strong>Why these zero blocks?</strong> Centered frequencies and array centroids remove phase couplings. Balanced orthogonal pilots separate the TX and RX angle blocks for this isolated path. General pilots and overlapping paths need not have these zeros.</p></div>
      <div class="matrix-detail"><div><h3>Selected: ${PARAM_NAMES[i]} × ${PARAM_NAMES[j]}</h3><div class="detail-label">J<sub>${i+1},${j+1}</sub> [1 / (${PARAM_UNITS[i]} · ${PARAM_UNITS[j]})]</div><div class="selected-number">${number(value,6)}</div><div class="detail-label">Normalized derivative inner product</div><div class="selected-number">${number(corr,5)}</div><div class="formula-box" style="margin-top:10px"><div class="formula">J = 2 Re(Dᴴ C<sub>n</sub>⁻¹ D)<br>D = ∂μ / ∂ηᵀ<br><small>Complex Gaussian noise, known covariance.</small></div></div></div>
      <ol class="step-list"><li><strong>Collect signal energy</strong>Γ = N<sub>r</sub> K L ρ = <span class="mono">${number(r.signal.gamma,5)}</span> (${db(r.signal.snrTotalDb)} dB).</li><li><strong>Delay information after unknown phase</strong>β = <span class="mono">${number(r.grid.betaHz/1e6,5)} MHz</span>.<br>J<sub>ττ</sub> = 8π²Γβ² × 10⁻¹⁸<br>= <span class="mono">${number(r.fim[0][0],5)} ns⁻²</span>.</li><li><strong>Invert the ${angle===1?'AoA':'AoD'} pair together</strong><span class="mono">J = [${number(jaa)} ${number(jae)}<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${number(jae)} ${number(jee)}]</span><br>σ<sub>az</sub> ≥ √[J<sub>el,el</sub> / det(J)]<br>= <span class="mono">${number(azBound)}°</span>.</li></ol></div></div>
      <div class="insight">Why not use 1 / J<sub>ii</sub>? That would hold the other parameters known. The diagonal of J⁻¹ accounts for their joint estimation. For a singular direction the marginal bound is infinite, not zero.</div>`;
  }

  function energyHeld(r) {
    if (state.mode === 'total') return `Aggregate Γ = ${db(r.signal.snrTotalDb)} dB, independent of the number of observations.`;
    if (state.mode === 'perTone') return `Per-RX / tone / symbol ρ = ${db(r.signal.snrPerToneDb)} dB. Γ grows with K or L.`;
    return `Total conducted power ${db(state.txPowerDbm)} dBm, path loss ${db(state.lossDb)} dB and noise figure ${db(state.nfDb)} dB. Noise grows with B.`;
  }
  function bandwidth(r) {
    const modeNote = state.mode==='total'
      ? 'At fixed aggregate energy-to-noise ratio, στ is approximately proportional to 1/B. More tones at fixed B approach a finite bandwidth limit.'
      : state.mode==='perTone'
        ? 'At fixed ρ, more tones provide more total signal energy: Γ = Nr K L ρ. An improving K curve therefore includes extra energy.'
        : 'At fixed conducted power, increasing B raises receiver noise and shortens the useful record when K and L are fixed. The delay slope is approximately B⁻¹ᐟ².';
    return heading('Separate aperture from energy', 'What does a wider frequency grid actually buy?', 'Each point recomputes the same joint Fisher matrix. Both plots sweep B = KΔf and preserve the displayed SNR convention.')
      + `<div class="formula-box"><div class="small-label">Held fixed in these sweeps</div><div style="font-size:11px;line-height:1.5">${energyHeld(r)}</div></div>
      <div class="sweep-grid"><div class="canvas-card plot-card"><div class="plot-title">Sweep B · hold K = ${integer(state.tones)}, L = ${integer(state.symbols)}</div><canvas id="bandwidth-plot" role="img" aria-label="Calculated delay standard-deviation bound versus bandwidth"></canvas></div><div class="canvas-card plot-card"><div class="plot-title">Sweep K · hold B = ${number(r.grid.bandwidthHz/1e6)} MHz, L = ${integer(state.symbols)}</div><canvas id="tones-plot" role="img" aria-label="Calculated delay standard-deviation bound versus subcarrier count"></canvas></div></div>
      <div class="sweep-notes"><p><strong>Bandwidth sweep.</strong> Δf = B/K changes with B. The useful duration L/Δf therefore changes too. Center-frequency phase carries no separate delay information when the path phase is unknown.</p><p><strong>Tone-count sweep.</strong> Δf = B/K shrinks as K grows. More tones also lengthen the useful record. With K = 1 the unknown path phase and delay are inseparable, so the delay bound is infinite.</p></div>
      <div class="insight">${modeNote} Current useful duration: <strong>${duration(r.grid.usefulTimeS)}</strong>. This model assumes coherence across that entire record, excluding CP and gaps.</div>`;
  }
  function overlap(r, normalizedDelay) {
    const K=+state.tones;
    const q=Math.PI*normalizedDelay/K;
    let c;
    if (Math.abs(Math.sin(q))<1e-12) c=1;
    else c=Math.min(1,Math.abs(Math.sin(K*q)/(K*Math.sin(q))));
    const independent = Math.max(0,1-c*c);
    return { correlation:c, independent, penalty:independent<1e-14?Infinity:1/Math.sqrt(independent), condition:c>1-1e-14?Infinity:(1+c)/(1-c), delayNs:normalizedDelay/r.grid.bandwidthHz*1e9 };
  }
  function multipath(r) {
    const o=overlap(r,local.separation);
    return heading('Two complex gains · known delays', 'When two paths overlap, their gains become coupled.', 'This deliberately small example uses two paths with the same array direction. Their delays are known, and both complex gains are unknown.', 'Gain-identifiability toy')
      + `<div class="separation-controls"><label for="separation-slider">Path separation Δτ</label><input id="separation-slider" type="range" min="0" max="6" step="0.005" value="${local.separation}" aria-label="Delay separation from zero to six over bandwidth"><input id="separation-ns" type="number" min="0" max="${6/r.grid.bandwidthHz*1e9}" step="any" value="${Number(o.delayNs.toPrecision(6))}" aria-label="Path delay separation in nanoseconds"><span class="mono">ns</span></div>
      <div class="multipath-layout"><div class="canvas-card multipath-plot"><canvas id="overlap-plot" role="img" aria-label="Uniform-tone waveform similarity versus delay separation"></canvas></div><div class="multipath-stats"><div class="mini-stat"><div class="metric-label">Waveform similarity |c|</div><div class="metric-value">${number(o.correlation)}</div><div class="metric-note">1 = identical signatures</div></div><div class="mini-stat"><div class="metric-label">Remaining gain information 1 − |c|²</div><div class="metric-value" style="color:var(--teal)">${number(o.independent)}</div><div class="metric-note">After treating the other gain as unknown</div></div><div class="mini-stat"><div class="metric-label">Gain √CRB inflation</div><div class="metric-value" style="color:var(--orange)">${number(o.penalty)}<span class="unit">×</span></div><div class="metric-note">Relative to orthogonal path signatures</div></div></div></div>
      <div class="formula-box" style="margin-top:10px"><div class="formula">c(Δτ) = (1/K) ∑<sub>k</sub> exp(−j2πf<sub>k</sub>Δτ) &nbsp; · &nbsp; gain information factor = 1 − |c|²</div></div>
      <p class="footnote">The gain Gram matrix is [[1, c], [c*, 1]]. Its inverse gives the displayed inflation. This is not the full multipath delay/angle CRB. Different arrival or departure directions can add separation. Finite subcarrier spacing makes the delay signature periodic, with period 1/Δf.</p>`;
  }

  function render() {
    synchronizeDerived();
    current = API.compute(state);
    persist();
    if (!current.valid) {
      results.innerHTML=heading('Check the sounding setup','The pilot assumptions need to hold.','Balanced orthogonal training requires sufficient symbols to distinguish all transmitting ports.')+alertHTML(current)+'<div class="formula-box"><div class="formula large">rank(X) = N<sub>t</sub> ⇒ L ≥ N<sub>t</sub></div><p class="footnote">Here X is the TX-by-symbol pilot matrix on each tone. The counts on the left describe antenna elements of one BS and one UE.</p></div>';
      return;
    }
    const renderers={calculator,geometry,fisher,bandwidth,multipath};
    results.innerHTML=renderers[lab](current);
    if(lab==='geometry') initializeGeometry();
    if(lab==='bandwidth') drawSweeps();
    if(lab==='multipath') drawOverlap();
  }

  function canvasContext(id) {
    const canvas=document.getElementById(id);
    if(!canvas) return null;
    const rect=canvas.getBoundingClientRect();
    const width=Math.max(10,rect.width), height=Math.max(10,rect.height);
    const scale=Math.min(devicePixelRatio||1,2);
    canvas.width=Math.round(width*scale); canvas.height=Math.round(height*scale);
    const ctx=canvas.getContext('2d');
    ctx.scale(scale,scale);
    ctx.clearRect(0,0,width,height);
    ctx.lineJoin='round'; ctx.lineCap='round';
    return {canvas,ctx,width,height};
  }
  function project(v,w,h,scale) {
    const cy=Math.cos(local.yaw),sy=Math.sin(local.yaw),cp=Math.cos(local.pitch),sp=Math.sin(local.pitch);
    const xx=cy*v[0]-sy*v[1],yy=sy*v[0]+cy*v[1];
    const zz=cp*v[2]-sp*xx;
    const depth=sp*v[2]+cp*xx;
    return [w*.46+yy*scale,h*.57-zz*scale,depth];
  }
  function line3(ctx,a,b,w,h,scale,color,width=1,dash=[]) {
    const pa=project(a,w,h,scale),pb=project(b,w,h,scale);
    ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.moveTo(pa[0],pa[1]);ctx.lineTo(pb[0],pb[1]);ctx.stroke();ctx.setLineDash([]);
    return [pa,pb];
  }
  function arrow(ctx,a,b,w,h,scale,color,label) {
    const [pa,pb]=line3(ctx,a,b,w,h,scale,color,2.2);
    const angle=Math.atan2(pb[1]-pa[1],pb[0]-pa[0]);
    ctx.fillStyle=color;ctx.beginPath();ctx.moveTo(pb[0],pb[1]);ctx.lineTo(pb[0]-9*Math.cos(angle-.42),pb[1]-9*Math.sin(angle-.42));ctx.lineTo(pb[0]-9*Math.cos(angle+.42),pb[1]-9*Math.sin(angle+.42));ctx.closePath();ctx.fill();
    ctx.font='11px Arial';ctx.fillText(label,pb[0]+8,pb[1]-8);
  }
  function initializeGeometry() {
    drawGeometry();
    const canvas=$('#geometry-canvas');
    canvas.style.touchAction='none';
    canvas.style.cursor='grab';
    let drag=null;
    canvas.addEventListener('pointerdown',event=>{drag={x:event.clientX,y:event.clientY,yaw:local.yaw,pitch:local.pitch};canvas.setPointerCapture(event.pointerId);canvas.style.cursor='grabbing';});
    canvas.addEventListener('pointermove',event=>{
      if(!drag)return;
      local.yaw=drag.yaw+(event.clientX-drag.x)*.008;
      local.pitch=Math.max(-1.22,Math.min(1.22,drag.pitch-(event.clientY-drag.y)*.008));
      $('#camera-yaw').value=Math.max(-180,Math.min(180,local.yaw/DEG));
      $('#camera-pitch').value=local.pitch/DEG;
      drawGeometry();
    });
    const finish=()=>{drag=null;canvas.style.cursor='grab';};
    canvas.addEventListener('pointerup',finish);canvas.addEventListener('pointercancel',finish);
  }
  function drawGeometry() {
    const surface=canvasContext('geometry-canvas');if(!surface||!current?.valid)return;
    const {ctx,width:w,height:h}=surface,r=current;
    const name=local.array, isTx=name==='tx', array=r.geometry[name];
    const az=state[`${name}Az`]*DEG,el=state[`${name}El`]*DEG;
    const direction=[Math.cos(el)*Math.cos(az),Math.cos(el)*Math.sin(az),Math.sin(el)];
    const dAz=[-Math.cos(el)*Math.sin(az),Math.cos(el)*Math.cos(az),0];
    const dEl=[-Math.sin(el)*Math.cos(az),-Math.sin(el)*Math.sin(az),Math.cos(el)];
    const size=Math.max(array.ny-1,array.nz-1,1);
    const unit=1.65/size;
    const scale=Math.min(w/5.9,h/3.65);
    const corners=[[0,-(array.ny-1)*unit/2,-(array.nz-1)*unit/2],[0,(array.ny-1)*unit/2,-(array.nz-1)*unit/2],[0,(array.ny-1)*unit/2,(array.nz-1)*unit/2],[0,-(array.ny-1)*unit/2,(array.nz-1)*unit/2]];
    ctx.beginPath();corners.forEach((v,i)=>{const p=project(v,w,h,scale);i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]);});ctx.closePath();ctx.fillStyle='rgba(24,116,184,.055)';ctx.fill();ctx.strokeStyle='#B9D4E7';ctx.lineWidth=1;ctx.stroke();
    const points=[];
    for(let y=0;y<array.ny;y++)for(let z=0;z<array.nz;z++)points.push(project([0,(y-(array.ny-1)/2)*unit,(z-(array.nz-1)/2)*unit],w,h,scale));
    points.sort((a,b)=>a[2]-b[2]);
    const dot=Math.min(4.8,Math.max(1.3,scale*unit*.24));
    for(const p of points){ctx.beginPath();ctx.arc(p[0],p[1],dot,0,Math.PI*2);ctx.fillStyle=isTx?COLORS.blue:COLORS.teal;ctx.fill();}
    const axisLength=1.2;
    for(const [v,label] of [[[axisLength,0,0],'+x'],[[0,axisLength,0],'+y'],[[0,0,axisLength],'+z']]){
      const [,p]=line3(ctx,[0,0,0],v,w,h,scale,'#93A0AA',1,[3,3]);ctx.font='10px Arial';ctx.fillStyle=COLORS.muted;ctx.fillText(label,p[0]+4,p[1]+3);
    }
    const rayLength=2.45, endpoint=direction.map(x=>rayLength*x);
    const idx=isTx?3:1,cov=r.covariance;
    const aa=cov[idx][idx],bb=cov[idx+1][idx+1],ab=cov[idx][idx+1];
    let magnification=1;
    if(finite(aa)&&finite(bb)&&Number.isFinite(ab)){
      const trace=aa+bb,discriminant=Math.sqrt(Math.max(0,(aa-bb)**2+4*ab**2));
      const largest=Math.sqrt(Math.max(0,(trace+discriminant)/2))*DEG;
      magnification=Math.min(10000,Math.max(.000001,.17/Math.max(largest,1e-12)));
      magnification=Number(magnification.toPrecision(2));
      const a=Math.sqrt(Math.max(aa,0)),b=a>0?ab/a:0,c=Math.sqrt(Math.max(0,bb-b*b));
      const samples=[];
      for(let k=0;k<=80;k++){
        const t=2*Math.PI*k/80,da=a*Math.cos(t)*DEG*magnification,de=(b*Math.cos(t)+c*Math.sin(t))*DEG*magnification;
        samples.push(endpoint.map((v,j)=>v+rayLength*(da*dAz[j]+de*dEl[j])));
      }
      ctx.beginPath();const p0=project([0,0,0],w,h,scale);ctx.moveTo(p0[0],p0[1]);for(const p of samples){const q=project(p,w,h,scale);ctx.lineTo(q[0],q[1]);}ctx.closePath();ctx.fillStyle='rgba(10,107,94,.075)';ctx.fill();
      ctx.beginPath();samples.forEach((p,k)=>{const q=project(p,w,h,scale);k?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]);});ctx.strokeStyle=COLORS.teal;ctx.lineWidth=2;ctx.fillStyle='rgba(10,107,94,.17)';ctx.fill();ctx.stroke();
      $('#geometry-badge').innerHTML=`1σ local covariance contour<br><strong>Angular display ×${number(magnification)}</strong>`;
    }else{
      $('#geometry-badge').innerHTML='Joint angle pair is unidentifiable<br><strong>No finite 2D covariance contour</strong>';
    }
    arrow(ctx,[0,0,0],endpoint,w,h,scale,COLORS.orange,isTx?'AoD':'AoA');
    const origin=project([0,0,0],w,h,scale);ctx.beginPath();ctx.arc(origin[0],origin[1],3,0,2*Math.PI);ctx.fillStyle=COLORS.ink;ctx.fill();
    ctx.font='10px Arial';ctx.fillStyle=COLORS.muted;ctx.fillText(`Azimuth ${number(state[`${name}Az`])}° · elevation ${number(state[`${name}El`])}°`,12,h-12);
    if(w>=600){ctx.textAlign='right';ctx.fillText('Schematic aperture · exact local angle covariance',w-12,h-12);ctx.textAlign='left';}
  }

  function logspace(a,b,n){return Array.from({length:n},(_,i)=>Math.exp(Math.log(a)+(Math.log(b)-Math.log(a))*i/(n-1)));}
  function ticksLog(min,max){const vals=[];for(let p=Math.floor(Math.log10(min));p<=Math.ceil(Math.log10(max));p++)for(const m of [1,2,5]){const x=m*10**p;if(x>=min*.999&&x<=max*1.001)vals.push(x);}return vals;}
  function plotLog(id,points,options){
    const surface=canvasContext(id);if(!surface)return;const {ctx,width:w,height:h}=surface;
    const margin={l:48,r:16,t:14,b:37},pw=w-margin.l-margin.r,ph=h-margin.t-margin.b;
    if(pw<20||ph<20)return;
    const finitePoints=points.filter(p=>p.x>0&&p.y>0&&Number.isFinite(p.y));
    if(!finitePoints.length){ctx.fillStyle=COLORS.muted;ctx.font='12px Arial';ctx.fillText('No finite delay bound in this setup.',20,50);return;}
    const xmin=Math.min(...points.map(p=>p.x)),xmax=Math.max(...points.map(p=>p.x));
    let ymin=Math.min(...finitePoints.map(p=>p.y)),ymax=Math.max(...finitePoints.map(p=>p.y));
    if(ymax/ymin<1.2){ymin/=1.8;ymax*=1.8;}else{ymin/=1.2;ymax*=1.2;}
    const x=v=>margin.l+(Math.log(v)-Math.log(xmin))/(Math.log(xmax)-Math.log(xmin))*pw;
    const y=v=>margin.t+ph-(Math.log(v)-Math.log(ymin))/(Math.log(ymax)-Math.log(ymin))*ph;
    const xt=ticksLog(xmin,xmax),yt=ticksLog(ymin,ymax);
    ctx.font='9px Arial';ctx.lineWidth=1;
    const xstep=Math.ceil(xt.length/Math.max(3,Math.floor(pw/50)));
    xt.forEach((v,i)=>{if(i%xstep)return;ctx.strokeStyle='#EDF1F4';ctx.beginPath();ctx.moveTo(x(v),margin.t);ctx.lineTo(x(v),margin.t+ph);ctx.stroke();ctx.textAlign='center';ctx.fillStyle=COLORS.muted;ctx.fillText(number(v,2),x(v),h-21);});
    const ystep=Math.ceil(yt.length/Math.max(3,Math.floor(ph/25)));
    yt.forEach((v,i)=>{if(i%ystep)return;ctx.strokeStyle='#EDF1F4';ctx.beginPath();ctx.moveTo(margin.l,y(v));ctx.lineTo(w-margin.r,y(v));ctx.stroke();ctx.textAlign='right';ctx.fillStyle=COLORS.muted;ctx.fillText(number(v,2),margin.l-6,y(v)+3);});
    ctx.strokeStyle='#A7B2BC';ctx.beginPath();ctx.moveTo(margin.l,margin.t);ctx.lineTo(margin.l,margin.t+ph);ctx.lineTo(w-margin.r,margin.t+ph);ctx.stroke();
    ctx.save();ctx.translate(10,margin.t+ph/2);ctx.rotate(-Math.PI/2);ctx.textAlign='center';ctx.fillStyle=COLORS.muted;ctx.font='10px Arial';ctx.fillText('√CRB delay [ns]',0,0);ctx.restore();
    ctx.textAlign='center';ctx.fillStyle=COLORS.muted;ctx.font='10px Arial';ctx.fillText(options.xlabel,margin.l+pw/2,h-5);
    ctx.beginPath();ctx.strokeStyle=options.color||COLORS.blue;ctx.lineWidth=2.4;let started=false;
    points.forEach(p=>{if(!(p.y>0&&Number.isFinite(p.y))){started=false;return;}if(started)ctx.lineTo(x(p.x),y(p.y));else{ctx.moveTo(x(p.x),y(p.y));started=true;}});ctx.stroke();
    const mark=options.current;
    if(mark&&mark.x>=xmin&&mark.x<=xmax&&mark.y>0&&Number.isFinite(mark.y)){
      ctx.beginPath();ctx.arc(x(mark.x),y(mark.y),4.5,0,2*Math.PI);ctx.fillStyle=COLORS.orange;ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.stroke();
      ctx.fillStyle=COLORS.orange;ctx.font='bold 10px Arial';ctx.textAlign='right';ctx.fillText(`Now: ${number(mark.y)} ns`,w-margin.r,12);
    }
    ctx.textAlign='left';
  }
  function drawSweeps(){
    if(!current?.valid)return;
    const base={...state,gridMode:'bandwidth',bandwidthMHz:current.grid.bandwidthHz/1e6};
    const lowB=Math.max(.001,Math.min(20,base.bandwidthMHz/4));
    const highB=Math.max(1600,base.bandwidthMHz*2);
    const bValues=logspace(lowB,highB,75);
    const bPoints=API.sweep(base,'bandwidthMHz',bValues).map(p=>({x:p.value,y:p.result.bounds?.tauNs??Infinity}));
    const maxK=Math.max(6600,state.tones*2);
    const kValues=[...new Set(logspace(2,maxK,90).map(Math.round))];
    const kPoints=API.sweep(base,'tones',kValues).map(p=>({x:p.value,y:p.result.bounds?.tauNs??Infinity}));
    plotLog('bandwidth-plot',bPoints,{xlabel:'Nominal bandwidth B [MHz] · log scale',color:COLORS.blue,current:{x:base.bandwidthMHz,y:current.bounds.tauNs}});
    plotLog('tones-plot',kPoints,{xlabel:'Subcarrier count K · log scale',color:COLORS.teal,current:{x:state.tones,y:current.bounds.tauNs}});
  }
  function drawOverlap(){
    const surface=canvasContext('overlap-plot');if(!surface||!current?.valid)return;const {ctx,width:w,height:h}=surface;
    const margin={l:43,r:15,t:27,b:38},pw=w-margin.l-margin.r,ph=h-margin.t-margin.b;
    const x=v=>margin.l+v/6*pw,y=v=>margin.t+(1-v)*ph;
    ctx.font='10px Arial';ctx.lineWidth=1;
    for(let i=0;i<=4;i++){const v=i/4;ctx.strokeStyle='#EAF0F4';ctx.beginPath();ctx.moveTo(margin.l,y(v));ctx.lineTo(w-margin.r,y(v));ctx.stroke();ctx.fillStyle=COLORS.muted;ctx.textAlign='right';ctx.fillText(number(v,2),margin.l-7,y(v)+3);}
    for(let i=0;i<=6;i++){ctx.strokeStyle='#EDF1F4';ctx.beginPath();ctx.moveTo(x(i),margin.t);ctx.lineTo(x(i),margin.t+ph);ctx.stroke();ctx.fillStyle=COLORS.muted;ctx.textAlign='center';ctx.fillText(number(i/current.grid.bandwidthHz*1e9,3),x(i),h-20);}
    for(const [key,color] of [['correlation',COLORS.blue],['independent',COLORS.teal]]){
      ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=2.4;
      for(let n=0;n<=500;n++){const t=6*n/500,o=overlap(current,t);n?ctx.lineTo(x(t),y(o[key])):ctx.moveTo(x(t),y(o[key]));}ctx.stroke();
    }
    const o=overlap(current,local.separation);ctx.strokeStyle=COLORS.orange;ctx.lineWidth=1.5;ctx.setLineDash([4,3]);ctx.beginPath();ctx.moveTo(x(local.separation),margin.t);ctx.lineTo(x(local.separation),margin.t+ph);ctx.stroke();ctx.setLineDash([]);
    for(const key of ['correlation','independent']){ctx.beginPath();ctx.arc(x(local.separation),y(o[key]),4,0,2*Math.PI);ctx.fillStyle=COLORS.orange;ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.stroke();}
    ctx.textAlign='left';ctx.font='10px Arial';ctx.fillStyle=COLORS.blue;ctx.fillText('— waveform similarity |c|',margin.l,13);ctx.fillStyle=COLORS.teal;ctx.fillText('— information 1 − |c|²',Math.max(margin.l+150,w*.53),13);
    ctx.textAlign='center';ctx.fillStyle=COLORS.muted;ctx.fillText('Delay separation Δτ [ns]',margin.l+pw/2,h-5);ctx.textAlign='left';
  }

  form.addEventListener('submit',event=>event.preventDefault());
  form.addEventListener('input',event=>{
    const target=event.target,key=target.name;
    if(!key||!Object.prototype.hasOwnProperty.call(API.defaults,key))return;
    if(key==='mode' && current?.valid && target.value!=='link'){
      state.snrDb=target.value==='total'?current.signal.snrTotalDb:current.signal.snrPerToneDb;
      input('snrDb').value=Number(state.snrDb.toPrecision(8));
    }
    if(target.type==='number'){
      if(target.value===''||!Number.isFinite(target.valueAsNumber))return;
      state[key]=target.valueAsNumber;
    }else state[key]=target.value;
    synchronizeDerived();schedule();
  });
  $('#preset').addEventListener('change',event=>usePreset(event.target.value));
  $('#reset').addEventListener('click',()=>{usePreset('baseline');$('#preset').value='baseline';});
  $('.lab-tabs').addEventListener('click',event=>{const button=event.target.closest('[data-lab]');if(button)switchLab(button.dataset.lab);});
  $('.lab-tabs').addEventListener('keydown',event=>{
    if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
    event.preventDefault();event.stopPropagation();const index=LABS.indexOf(lab);
    const next=event.key==='Home'?0:event.key==='End'?LABS.length-1:(index+(event.key==='ArrowRight'?1:-1)+LABS.length)%LABS.length;
    switchLab(LABS[next]);document.querySelector(`[data-lab="${LABS[next]}"]`).focus();
  });
  results.addEventListener('click',event=>{
    const target=event.target.closest('button');if(!target)return;
    if(target.hasAttribute('data-fix-training')){state.symbols=state.txY*state.txZ;synchronizeInputs();render();}
    if(target.dataset.array){local.array=target.dataset.array;render();}
    if(target.hasAttribute('data-camera-reset')){local.yaw=-.65;local.pitch=.36;render();}
    if(target.dataset.matrix){local.matrix=target.dataset.matrix;render();}
    if(target.dataset.cell){local.cell=target.dataset.cell.split(',').map(Number);render();results.querySelector(`[data-cell="${local.cell.join(',')}"]`)?.focus({preventScroll:true});}
  });
  results.addEventListener('input',event=>{
    const t=event.target;
    if(t.id==='camera-yaw'){local.yaw=+t.value*DEG;drawGeometry();}
    if(t.id==='camera-pitch'){local.pitch=+t.value*DEG;drawGeometry();}
    if(t.id==='separation-slider'||t.id==='separation-ns'){
      if(t.value==='')return;
      const raw=t.id==='separation-slider'?+t.value:+t.value*current.grid.bandwidthHz/1e9;
      local.separation=Math.max(0,Math.min(6,raw));
      const o=overlap(current,local.separation);
      if(t.id!=='separation-slider')$('#separation-slider').value=local.separation;
      if(t.id!=='separation-ns')$('#separation-ns').value=Number(o.delayNs.toPrecision(6));
      const stats=document.querySelectorAll('.multipath-stats .metric-value');
      stats[0].innerHTML=number(o.correlation);stats[1].innerHTML=number(o.independent);stats[2].innerHTML=number(o.penalty)+'<span class="unit">×</span>';
      drawOverlap();
    }
  });
  let resizePending=false;
  const redraw=()=>{if(resizePending)return;resizePending=true;requestAnimationFrame(()=>{resizePending=false;if(lab==='geometry')drawGeometry();if(lab==='bandwidth')drawSweeps();if(lab==='multipath')drawOverlap();});};
  window.addEventListener('resize',redraw);
  window.addEventListener('bento-live-visibility',event=>{if(!event.detail?.paused)redraw();});
  if(typeof ResizeObserver!=='undefined')new ResizeObserver(redraw).observe(results);
  synchronizeInputs();switchLab(lab,false);
})();
