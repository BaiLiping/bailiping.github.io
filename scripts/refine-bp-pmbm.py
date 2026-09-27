"""Idempotent, scoped correction of the September 2026 BP/PMBM lesson.

Preserves the article's layout and the existing Bento slide renderer. The shared
model and its tests are separate, inspectable JavaScript files. Once the marker
is present this migration does not overwrite subsequent editorial changes.
"""
from pathlib import Path
import re
ROOT = Path(__file__).resolve().parents[1]

def replace(s, old, new):
    if old not in s:
        raise ValueError(f"Audit anchor missing: {old[:120]!r}")
    return s.replace(old, new)

def article():
    path=ROOT/'bp-vs-pmbm/index.html'
    s=path.read_text()
    if 'bp-pmbm-audit-2026-09-07' in s:
        return
    s=replace(s,'<main>','<main data-audit="bp-pmbm-audit-2026-09-07">')
    s=replace(s,'<a href="/">Home</a>','<a href="/bp-vs-pmbm-slides/">Slides</a> · <a href="/">Home</a>')
    s=replace(s,'which never enumerates events and instead passes messages until the <em>marginal</em> association probabilities emerge','an inference algorithm that approximates association marginals using messages instead of enumerating events')
    s=replace(s,'which represents ambiguity as a weighted mixture of global hypotheses and manages that mixture with hypothesis selection and pruning','a posterior family whose detected-target part retains a weighted mixture of compatible global hypotheses')
    s=replace(s,'Drag anything in Fig.&nbsp;1 — every figure below recomputes.','The numerical figures recompute when you drag Fig.&nbsp;1; the across-scan diagram is schematic.')
    a=s.index('<div class="accuracy"><strong>Model boundary.')
    b=s.index('<!-- ============ §1',a)
    s=s[:a]+'''<div class="accuracy"><strong>BP is an algorithm; PMBM is a posterior family.</strong> They are not mutually exclusive alternatives. Here the numerical comparison is <em>association BP versus exhaustive summation of the same one-scan assignment posterior</em>, with a separate explanation of how that posterior occurs inside PMBM. The benchmark has three certainly existing point targets, constant detection probability, positive Poisson clutter intensity, and no undetected-target PPP. It is not a complete PMBM filter or an empirical comparison of tracking accuracy over time. See §3 for uncertain existence, newly detected targets, and the correct PMBM weights.</div>

'''+s[b:]
    s=replace(s,'making their marginal-association comparison exact for the toy model','so exhaustive enumeration is an exact reference for this benchmark; BP remains approximate on a loopy graph')
    s=replace(s,'<h4>Sensor model</h4>','''<button class="preset" id="treePreset">Separated tracks (acyclic)</button>
        <button class="preset" id="symmetricPreset">Symmetric ambiguity (loopy)</button>
        <h4>Sensor model</h4>''')
    a=s.index('        <div class="ctl">\n          <label>Toy birth')
    b=s.index('        <div class="statline"',a)
    s=s[:a]+'''        <label class="hint"><input type="checkbox" id="gateToggle" checked> Restrict pairs to the 99% gate</label>
'''+s[b:]
    s=replace(s,'step="0.05" value="-4.301"','step="0.001" value="-4.30103"')
    s=replace(s,'with z₄ off in the clutter','with z₄ initially outside every gate')
    s=replace(s,'i.e. the only associations either method will consider.','i.e. the pairs retained when gating is enabled. The gate is a likelihood truncation approximation, not a probability that an association is correct.')
    s=replace(s,' For display only, its mass is later split by the toy score β̃ = λ̃<sub>b</sub>/(λ̃<sub>b</sub>+λ<sub>c</sub>).',' No target-birth model is used in this benchmark.')
    s=replace(s,'or clutter/new','or unassigned to an existing track')
    s=replace(s,'Loopy sum–product on this bipartite graph is <em>provably convergent</em> (Williams &amp; Lau, 2014), costs O(n·m) per sweep, and its fixed point delivers approximate <strong>marginal</strong> association probabilities — no hypothesis is ever written down.',
      'For this specific Williams–Lau association construction, finite nonnegative weights and strictly positive missed/unassigned weights give a convergence guarantee. The guarantee is not a theorem about arbitrary loopy BP. Our dense implementation costs O(n·m) per sweep and O(T·n·m) for T sweeps. Its fixed point gives approximate <strong>marginal</strong> probabilities without listing global assignments. Convergence and accuracy are different questions <a href="#ref-bp">[1]</a>.')
    s=replace(s,'Converged ⏭','Final iterate ⏭')
    s=replace(s,'Convergence — max message change per full sweep','Convergence diagnostic — max |log ν(new) − log ν(old)| per sweep')
    s=replace(s,'p(b<sub>j</sub> = ·) — measurement-oriented marginals; for illustration, unassigned mass splits into <b>⊕ possible birth</b> vs ∅ clutter by the toy score β̃','p(b<sub>j</sub> = ·) — measurement beliefs update after μ; ∅ means unassigned (clutter in this benchmark). Track and measurement beliefs agree on each edge only at a fixed point.')
    s=replace(s,'State beliefs b(x<sub>i</sub>) — Gaussian mixtures','Single-track state beliefs b(x<sub>i</sub>) — Gaussian mixtures')
    s=replace(s,'(95 % ellipses; opacity ∝ weight; prior','(each ellipse is a component’s 95% contour, not a 95% contour of the whole mixture; opacity ∝ weight; prior')
    s=replace(s,'<span class="m">p(a<sub>i</sub>=j) ∝','<span class="m">q(a<sub>i</sub>=j) ∝')
    s=replace(s,'<span class="m">p(a<sub>i</sub>=∅) ∝','<span class="m">q(a<sub>i</sub>=∅) ∝')
    s=replace(s,'<div class="eqnote">t = 0 is the uncoupled start:', '''<div class="eq"><span class="m">q(b<sub>j</sub>=i) = μ<sub>i→j</sub> / (1 + Σ<sub>l</sub> μ<sub>l→j</sub>), &nbsp; q(b<sub>j</sub>=∅) = 1 / (1 + Σ<sub>l</sub> μ<sub>l→j</sub>)</span><span class="num">(3b) measurement-side normalization</span></div>
        <div class="eqnote">Here q denotes a BP belief; p denotes an exact probability when the distinction matters. The consistency factor is Ψ<sub>ij</sub> = 1{(a<sub>i</sub>=j) ⇔ (b<sub>j</sub>=i)}. Thus p(a,b|Z) ∝ ∏<sub>i</sub>ℓ<sub>i,aᵢ</sub> ∏<sub>ij</sub>Ψ<sub>ij</sub>. The scalar messages are ratios, not probabilities: μ may exceed one.</div>
        <div class="eqnote">t = 0 is the uncoupled start:''')
    s=replace(s,'and only then do the marginals, edge thicknesses and Δ move.','measurement beliefs update after μ; track beliefs, edge thicknesses, and Δ update after ν.')
    s=replace(s,'The Poisson multi-Bernoulli mixture is a <em>conjugate</em> multi-object posterior under the standard point-target model:','Under independent point-target detections, Poisson clutter, independent survival/motion, and Poisson birth, the PMBM family is conjugate:')
    s=replace(s,'Each global hypothesis is one compatible association story.','Each global hypothesis selects mutually compatible local histories. Bernoulli components are conditionally independent within that hypothesis, but the mixture generally couples them. “Detected” includes potentially existing targets, not only certain objects <a href="#ref-pmbm">[4]</a>.')
    # The valid set-partition density introduced in the earlier site audit is retained.
    s=replace(s,'<div class="eqnote">Exact inference for this benchmark carries every valid A.', '''<div class="eqnote">For a PPP, f<sup>ppp</sup>(U)=exp(−Λ)∏<sub>x∈U</sub>λ<sup>u</sup>(x), with Λ=∫λ<sup>u</sup>(x)dx. A Bernoulli has f(∅)=1−r, f({x})=r p(x), and zero density for larger sets.</div>
    <div class="eqnote">Exact inference for this benchmark carries every valid A.''')
    s=replace(s,'then prune and recycle.','then prune or apply optional reductions such as recycling (which changes the representation).')
    insert='''
  <details class="accuracy" id="full-pmbm" open>
    <summary><strong>The actual PMBM association update — one predicted parent hypothesis</strong></summary>
    <p>Let r<sub>i</sub> be existence probability, p<sub>i</sub>(x) the normalized single-target density, λ<sup>u</sup>(x) the predicted undetected-target PPP intensity, c(z) the clutter intensity, and g(z|x) the measurement likelihood. The superscript “−” denotes the predicted parent. Detection probability may depend on x. The following equations are before gating:</p>
    <div class="eq">ρ<sub>i0</sub> = 1 − r<sub>i</sub> + r<sub>i</sub> ∫(1−p<sub>D</sub>(x))p<sub>i</sub>(x)dx</div>
    <div class="eq">ρ<sub>ij</sub> = r<sub>i</sub> ∫p<sub>D</sub>(x)g(z<sub>j</sub>|x)p<sub>i</sub>(x)dx</div>
    <div class="eq">e<sub>j</sub> = ∫p<sub>D</sub>(x)g(z<sub>j</sub>|x)λ<sup>u</sup>(x)dx, &nbsp; q<sub>j</sub> = c(z<sub>j</sub>) + e<sub>j</sub></div>
    <div class="eq">w<sub>h,A</sub><sup>+</sup> ∝ w<sub>h</sub><sup>−</sup> ∏<sub>i</sub>ρ<sub>i,aᵢ</sub> ∏<sub>j unassigned</sub>q<sub>j</sub></div>
    <p>After factoring out the common ∏<sub>j</sub>q<sub>j</sub>, use <strong>ℓ<sub>i0</sub>=ρ<sub>i0</sub> and ℓ<sub>ij</sub>=ρ<sub>ij</sub>/q<sub>j</sub></strong> in exactly the same assignment solver. The q<sub>j</sub> here is evidence, not the BP belief q(a). Our displayed ℓ<sub>i0</sub>=1−P<sub>D</sub> and clutter-only denominator require r<sub>i</sub>=1 and e<sub>j</sub>=0. With multiple predicted parents, their weights and local histories must also be retained and updated.</p>
    <div class="eq">r<sub>j,new</sub> = e<sub>j</sub>/q<sub>j</sub>, &nbsp; p<sub>j,new</sub>(x) = p<sub>D</sub>(x)g(z<sub>j</sub>|x)λ<sup>u</sup>(x)/e<sub>j</sub></div>
    <p>This existence is <em>conditional on z<sub>j</sub> being unassigned to existing tracks</em>. Its marginal existence is P(unassigned z<sub>j</sub>|Z)·e<sub>j</sub>/q<sub>j</sub>. The density is needed only when e<sub>j</sub>&gt;0. “New” means <em>newly detected</em>; it need not mean physically born in the current scan. Clutter and a new target are the two existence outcomes of one Bernoulli, not necessarily two separately enumerated global events.</p>
    <div class="eq">r<sub>i,miss</sub><sup>+</sup> = r<sub>i</sub>∫(1−p<sub>D</sub>(x))p<sub>i</sub>(x)dx / ρ<sub>i0</sub>, &nbsp; λ<sup>u,+</sup>(x) = (1−p<sub>D</sub>(x))λ<sup>u</sup>(x)</div>
    <p>A detected existing-target branch has existence one. Its state density is proportional to p<sub>D</sub>(x)g(z<sub>j</sub>|x)p<sub>i</sub>(x); a missed branch’s state density is proportional to (1−p<sub>D</sub>(x))p<sub>i</sub>(x). Zero-evidence branches are omitted. These are conditional state updates; a complete filter also needs prediction, history management, and a specified estimator <a href="#ref-pmbm">[4]</a>.</p>
  </details>
'''
    pos=s.index('  <div class="fig">',s.index('<section class="sec pm"'))
    s=s[:pos]+insert+s[pos:]
    s=replace(s,'Every valid assignment hypothesis in the normalized benchmark, ranked by eq. 5.','The highest-ranked assignment hypotheses in the normalized benchmark, ranked by eq. 5 (remaining rows are summarized if necessary).')
    s=replace(s,' The ⊕ badge uses β̃ only as a visual birth-versus-clutter split; a real PMBM Bernoulli uses measurement-dependent PPP evidence.',' ∅ means the measurement is unassigned, hence clutter in this benchmark.')
    s=replace(s,'Slide k to see how truncation discards normalized assignment mass.','Slide k to see both discarded mass and the renormalized top-k marginals. Exact marginals continue to use all events.')
    s=replace(s,'<div class="matwrap"><table class="mat" id="exheat"></table></div>','''<div class="matwrap"><table class="mat" id="exheat"></table></div>
      <div class="heatnote" style="margin-top:18px">Renormalized top-k marginals — a different approximation from BP</div>
      <div class="matwrap"><table class="mat" id="topkheat"></table></div>
      <div class="eqnote" id="topkRO"></div>''')
    s=replace(s,'<h5>BP marginals (converged)</h5>','<h5 id="cmpBPTitle">BP final-iterate marginals</h5>')
    s=replace(s,'Exact / PMBM marginals','Exact assignment marginals')
    s=replace(s,'The difference compounds over time. PMBM branches every scan and prunes (dashed = discarded); identity ambiguity can stay unresolved until later evidence settles it. A BP-based tracker instead re-negotiates marginals within each scan and carries a single belief set forward — O(n·m) flat cost, but modes merged at every step (the JPDA-style coalescence risk).',
      'Schematic, not a multi-scan experiment. Retained PMBM hypotheses can preserve competing explanations; pruning loses discarded histories. A track-oriented marginal filter may instead carry a product of single-track beliefs forward. That projection discards inter-track dependence even when each belief stays multimodal. Gaussian moment matching is a further, separate approximation. Neither operation is inherent to BP. The dense association subproblem costs O(T·n·m), not constant time; trajectory formulations are needed when identity/history is the explicit inference object.')
    s=replace(s,'by a single multi-Bernoulli using marginal association probabilities','by a single multi-Bernoulli for detected targets, while retaining the undetected-target PPP, using marginal association probabilities')
    s=replace(s,'It does not mean that every BP/SPA tracker is simply a PMBM filter with its hypotheses removed.','It does not mean that every BP/SPA tracker is simply a PMBM filter with its hypotheses removed. TOMB/P (track-oriented) and MOMB/P (measurement-oriented) use different groupings and can have different coalescence behavior <a href="#ref-marginal">[2]</a>.')
    s=replace(s,'<th class="b">Belief propagation (SPA / BP tracker)</th>','<th class="b">Association BP inside a marginal tracker</th>')
    s=replace(s,'Marginal association probabilities p(a<sub>i</sub>=j); one belief per track','Approximate association marginals q(a<sub>i</sub>=j); any downstream state representation is a separate design choice')
    s=replace(s,'Hypothesis generation (Murty’s k-best or Gibbs) + weight update (5) + pruning, recycling, merging','Compatible child generation (e.g. ranked assignment or sampling), PMBM evidence/state updates, and optional hypothesis reduction')
    s=replace(s,'<code>O(T·n·m)</code>, T ≈ tens of sweeps; embarrassingly scalable, message-parallel','<code>O(T·n·m)</code> for this dense association solver; T depends on weights and tolerance. Sparse implementations can exploit active edges. State updates cost extra.')
    s=replace(s,'<code>O(k·N³)</code> per Murty call plus hypothesis bookkeeping; combinatorial pressure managed, not removed','Depends on retained parents, gated subproblems, children requested, state integration, and the assignment/sampling implementation; no universal cubic per-scan cost.')
    s=replace(s,'Marginalized each scan in many BP-based filters; efficient, but projection can merge modes and promote coalescence','Track-oriented factorization discards dependence; moment matching may additionally merge modes. These are tracker choices, not a requirement of BP.')
    s=replace(s,'JPDA’s soft spirit, made scalable and convergent','Related to JPDA marginalization; convergence is specific to this association graph and assumptions')
    s=replace(s,'MHT’s hypothesis spirit, made Bayes-exact via RFS conjugacy','Related to MHT-style association histories, with an explicit RFS existence and undetected-object model')
    for needle,anchor in [('https://arxiv.org/abs/1209.6299','ref-bp'),('https://arxiv.org/abs/1203.2995','ref-marginal'),('https://doi.org/10.1109/JPROC.2018.2789427','ref-meyer'),('https://arxiv.org/abs/1703.04264','ref-pmbm')]:
        s=replace(s,f'<p><a href="{needle}"',f'<p id="{anchor}"><a href="{needle}"')
    a=s.index('  <p class="footnote">Model notes')
    b=s.index('\n</section>',a)
    s=s[:a]+'''  <p class="footnote">Benchmark assumptions: one scan; three independent, certainly existing targets; one measurement at most per target and one owner at most per measurement; constant P<sub>D</sub>; homogeneous Poisson clutter; no PPP of undetected targets. Observation model z=x+v with R=60I and Gaussian priors P<sub>i</sub>=S<sub>i</sub>−R. Coordinates are arbitrary 2D units, covariances have squared units, and clutter intensity has inverse-area units. Clutter intensity is not the expected clutter count; over area V that count is λ<sub>c</sub>V.</p>
  <p class="footnote">Gating is optional. With gating on, pairs outside χ²<sub>2</sub>(0.99)=9.210340… are set to zero and the remaining assignment distribution is renormalized. “Exact” means exact for that restricted distribution, not for the original untruncated sensor likelihood. The demo retains ℓ<sub>i0</sub>=1−P<sub>D</sub>. A genuinely censored-detection model instead needs a consistently normalized in-gate likelihood and a no-in-gate-detection probability such as 1−P<sub>D</sub>P<sub>G</sub>. Simply drawing a 99% gate does not perform that model correction.</p>
  <p class="footnote">The solver checks both log-message change and agreement of the two edge-belief views, to tolerance 10<sup>−10</sup>, with a 10,000-sweep safety cap. A cap is reported as a cap, not convergence. These stopping diagnostics do not bound BP’s error relative to exhaustive enumeration. The exact problem has at most Σ<sub>k=0</sub><sup>3</sup>C(3,k)C(4,k)k! = 73 assignments.</p>
  <div class="foot">Shared, dependency-free JavaScript solver for the article and slides · all computation stays in your browser · references are linked above.</div>'''+s[b:]
    # Core functions are now shared with slides and build-time figures.
    s=replace(s,'<script>\n','<script src="./association-model.js"></script>\n<script>\n')
    s=replace(s,'const GATE=9.21','const GATE=AssociationModel.GATE') if 'const GATE=9.21' in s else s
    s=replace(s,'b:1e-5,','gated:true,') if 'b:1e-5,' in s else s
    a=s.index('function buildL(){');b=s.index('function compute(){',a)
    s=s[:a]+'''function buildL(){return AssociationModel.buildWeights(state,state.gated!==false);}
function margFrom(L,nu){return AssociationModel.targetMarginals(L,nu);}
function runBP(L,maxIter=10000,tol=1e-10){return AssociationModel.bp(L,{maxIterations:maxIter,tolerance:tol}).history;}
function enumHyps(L){return AssociationModel.enumerate(L);}
function exactMarg(ev,n,m){return AssociationModel.eventMarginals(ev,n,m).marginals;}
'''+s[b:]
    s=replace(s,'const hist=runBP(L);','const solve=AssociationModel.bp(L);\n  const hist=solve.history;')
    s=replace(s,'return {L,gate,n,m,hist,ev,ex,bpFinal,sweeps,sweepDeltas,maxErr,pairs,beliefComps};','return {L,gate,n,m,hist,ev,ex,bpFinal,sweeps,sweepDeltas,maxErr,pairs,beliefComps,solve,graph:AssociationModel.topology(L)};')
    s=re.sub(r'^const beta = .*\n','',s,flags=re.M)
    a=s.index('function renderStat1(){');b=s.index('function renderLMat(){',a)
    s=s[:a]+'''function renderStat1(){
  $('stat1').innerHTML=`n = <b>${R.n}</b> tracks · m = <b>${R.m}</b> measurements<br>`+
    `positive-weight edges: <b>${R.graph.edges}</b>; independent cycles: <b>${R.graph.cycles}</b><br>`+
    `valid assignments: <b>${R.ev.length}</b><br>BP: <b>${R.sweeps}</b> sweeps; `+
    (R.solve.converged?'tolerance met':'iteration cap reached')+`<br>log-message change: ${R.solve.delta.toExponential(1)}`;
  $('bpEnd').textContent=R.solve.converged?'Final iterate ⏭':'Capped iterate ⏭';
}
'''+s[b:]
    a=s.index('function renderBMarg(){');b=s.index('\nfunction ',a+10)
    s=s[:a]+'''function renderBMarg(){
  const st=curState();
  if(!st.bmarg){$('bjheat').innerHTML='<tr><td>Measurement beliefs appear after the first μ half-step.</td></tr>';return;}
  let h='<tr><th></th><th>∅ unassigned</th>';
  for(let i=0;i<R.n;i++)h+=`<th>T${i+1}</th>`;h+='</tr>';
  st.bmarg.forEach((row,j)=>{h+=`<tr><td class="rowh">z${j+1}</td>`;row.forEach(v=>h+=heatCell(v,'31,119,180'));h+='</tr>';});
  $('bjheat').innerHTML=h;
}
'''+s[b:]
    # Remove the old, post-hoc birth/clutter badge and its slider handler.
    s=re.sub(r"const bj=beta\(\);",'',s)
    s=s.replace("const {ev}=R, b=beta();", "const {ev}=R;")
    s=s.replace("const ev=R.ev, b=beta();", "const ev=R.ev;")
    s=re.sub(r"\$\('bbSlider'\)\.addEventListener\('input',[\s\S]*?renderBMarg\(\); renderStat1\(\); renderHyps\(\); \}\);",'',s)
    a=s.index('function renderHyps(){');b=s.index('function renderExact()',a)
    # Rebuild this short renderer instead of preserving its misleading toy Bernoulli.
    s=s[:a]+'''function renderHyps(){
  const ev=R.ev;state.k=Math.min(state.k,ev.length);
  $('kSlider').max=ev.length;$('kSlider').value=state.k;$('kOut').textContent=state.k;
  const top=AssociationModel.eventMarginals(ev,R.n,R.m,state.k);
  $('kRO').textContent=`retained ${(100*top.mass).toFixed(2)}% · discarded ${(100*top.discardedMass).toFixed(2)}%`;
  let h='<tr><th>rank</th>';for(let j=0;j<R.m;j++)h+=`<th>z${j+1}</th>`;h+='<th>missed tracks</th><th>all-event probability</th></tr>';
  const SHOW=25;
  ev.slice(0,SHOW).forEach((e,r)=>{
    h+=`<tr class="${r<state.k?'kept':'cut'}"><td>${r+1}</td>`;
    for(let j=0;j<R.m;j++){const i=e.a.indexOf(j);h+=i<0?'<td><span class="chip nul">∅</span></td>':`<td><span class="chip" style="--c:${TCOL[i]}">T${i+1}</span></td>`;}
    const missed=e.a.map((j,i)=>j<0?`T${i+1}`:null).filter(Boolean).join(', ')||'—';
    h+=`<td>${missed}</td><td>${(100*e.p).toFixed(3)}%</td></tr>`;
  });
  if(ev.length>SHOW)h+=`<tr><td colspan="${R.m+3}">${ev.length-SHOW} further hypotheses (included in all exact calculations).</td></tr>`;
  $('hypTable').innerHTML=h;$('hypCount2').textContent=`all ${ev.length}`;
  $('topkheat').innerHTML=heatTable(top.marginals,'232,114,12');
  const error=100*AssociationModel.maxDifference(top.marginals,R.ex);
  $('topkRO').textContent=`Top-k maximum marginal error: ${error.toFixed(3)} pp. Discarded mass ε = ${(100*top.discardedMass).toFixed(3)}%. For restriction and renormalization of this discrete assignment posterior, total variation is ε, so every marginal error is at most ε (in probability units). The tail is known here only because all hypotheses were enumerated; a production k-best solver generally does not know it.`;
}
'''+s[b:]
    a=s.index('  const verdict = R.maxErr');b=s.index('\n}',a)
    s=s[:a]+'''  $('cmpBPTitle').textContent=R.solve.converged?'BP marginals (stopping tolerance met)':'BP marginals (iteration cap; unfinished)';
  const topology=R.graph.acyclic?'The positive-weight association graph is acyclic.':'The positive-weight association graph is cyclic.';
  $('errRO').textContent=`Maximum absolute BP–exact difference: ${(100*R.maxErr).toFixed(3)} pp. ${topology} Graph structure is computed from edges, not inferred from a small error. Edge-belief consistency residual: ${R.solve.dualResidual.toExponential(2)}. `+(R.solve.converged?'Stopping tolerance met; loopy approximation error may remain.':'Iteration cap reached: this comparison includes unfinished iteration error.');'''+s[b:]
    s=replace(s,'BP tracker — marginalize &amp; move on','Marginal-filter example — project &amp; move on')
    s=replace(s,"$('kSlider').addEventListener('input',e=>{ state.k=+e.target.value; renderHyps(); });",'''$('kSlider').addEventListener('input',e=>{ state.k=+e.target.value; renderHyps(); });
$('gateToggle').addEventListener('change',e=>{state.gated=e.target.checked;refresh(true);});
$('treePreset').addEventListener('click',()=>{
  state.T=clone(DEFAULT.T);state.Z=clone(DEFAULT.Z);
  state.T.forEach((t,i)=>{t.x=120+220*i;t.y=210;state.Z[i]={x:t.x+5,y:t.y-3};});
  state.Z[3]={x:660,y:45};state.gated=true;$('gateToggle').checked=true;refresh(true);
});
$('symmetricPreset').addEventListener('click',()=>{
  state.T=clone(DEFAULT.T);state.T.forEach(t=>{t.x=320;t.y=210;t.S=[[500,0],[0,500]];});
  state.Z=[{x:305,y:200},{x:320,y:225},{x:335,y:200},{x:560,y:60}];
  state.gated=true;$('gateToggle').checked=true;refresh(true);
});''')
    s=s.replace("'converged'", "(R.solve.converged?'tolerance met':'capped')")
    s=replace(s,'/* boot */','''window.BPAssociationAudit = {snapshot:()=>({L:R.L,exact:R.ex,bp:R.bpFinal,iterations:R.sweeps,converged:R.solve.converged,graph:R.graph,delta:R.solve.delta})};
/* boot */''')
    if 'beta()' in s or "$('bbSlider')" in s:
        raise ValueError('A stale toy-birth reference survived the migration.')
    s=s.replace('const GATE = 9.21, BELIEF_CHI2 = 5.99', 'const GATE = AssociationModel.GATE, BELIEF_CHI2 = -2*Math.log(0.05)')
    start=s.index('const DEFAULT = {',s.index("'use strict'"))
    end=s.index('const clone =',start)
    s=s[:start]+'const DEFAULT = AssociationModel.DEFAULT;\n'+s[end:]
    s=s.replace('PD: 0.90, c: 5e-5, b: 1e-5,', 'PD: DEFAULT.PD, c: DEFAULT.c, gated: true,')
    s=s.replace('p(a<sub>i</sub> = ·) — BP marginals', 'q(a<sub>i</sub> = ·) — BP beliefs')
    s=s.replace('p(b<sub>j</sub> = ·) — measurement-oriented marginals', 'q(b<sub>j</sub> = ·) — measurement-oriented beliefs')
    s=s.replace('max Δ = ${st.delta.toExponential(1)}', 'max |Δ log ν| = ${st.delta.toExponential(1)}')
    s=s.replace('complete association factor graph', 'association consistency graph')
    s=s.replace('p(b<sub>j</sub> = ·) — measurement beliefs','q(b<sub>j</sub> = ·) — measurement beliefs')
    s=s.replace('Poisson clutter density', 'Poisson clutter intensity')
    s=s.replace('<td class="rl">track birth</td>','<td class="rl">new detections</td>')
    s=s.replace('<style>\n:root{','<style>\n#full-pmbm{max-width:none;font-size:14px;line-height:1.6;padding:18px 20px}\n#full-pmbm .eq{display:block;white-space:nowrap;overflow-x:auto;line-height:1.8;padding:4px 0}\n:root{',1)
    path.write_text(s)

def deck():
    path=ROOT/'bp-vs-pmbm-slides/build-deck.mjs'
    s=path.read_text()
    if '// bp-pmbm-audit-2026-09-07' in s:return
    s=replace(s,'import fs from "node:fs";', 'import fs from "node:fs";\nimport Model from "../bp-vs-pmbm/association-model.js";\n// bp-pmbm-audit-2026-09-07')
    start=s.index('function matrixWeight(')
    end=s.index('function fmtWeight(',start)
    s=s[:start]+'''const benchmarkTracks = Model.DEFAULT.T;
const benchmarkMeasurements = Model.DEFAULT.Z;
const {L} = Model.buildWeights(Model.DEFAULT);
const bpResult = Model.bp(L, {history:false});
if (!bpResult.converged) throw new Error("Default benchmark did not meet the BP stopping tolerance.");
const bp = bpResult.marginals;
const events = Model.enumerate(L);
const exact = Model.eventMarginals(events, L.length, L[0].length-1).marginals;
const maxBpError = Model.maxDifference(bp,exact);
const topFiveMass = Model.eventMarginals(events,L.length,L[0].length-1,5).mass;

'''+s[end:]
    pairs=[
      ('Two inference representations','An inference algorithm and a posterior family'),
      ('The deck contrasts two inference representations, not two incompatible measurement models.','BP is an inference algorithm; PMBM is a posterior family. The two can be combined, rather than being mutually exclusive filters.'),
      ('The live computation is exact for a normalized one-scan assignment model—not for an entire PMBM filter update.','Exact enumeration is the reference for a one-scan model; BP approximates it. Neither demo is a full PMBM tracker.'),
      ('• existing tracks may be missed','• certain existing tracks may be missed'),
      ('• measurement-specific PPP birth evidence','• measurement-specific new-detection evidence'),
      ('Change geometry, Pᴅ, or clutter density and the entire inference problem changes.','Certain targets, zero undetected PPP. Gating truncates pair weights; it is not a corrected sensor model.'),
      ('The specific association construction converges; each sweep touches every track–measurement edge.','Positive miss/unassigned weights: this association BP converges. A small residual does not imply exact marginals.'),
      ('The missed-detection column is one minus detection probability.','For certain existence and no gate correction, the missed-detection column is one minus detection probability.'),
      ('convergence guarantee for this association model','positive miss/unassigned weights; not a guarantee for arbitrary loopy BP'),
      ('conjugate under its assumed model','conjugate under independent point detections, Poisson clutter/birth, and independent survival/motion'),
      ('PPP birth evidence','PPP new-detection evidence'),
      ('Retain the head. Quantify the tail.','Retain the head. Inspect the truncation error.'),
      ('Practical joint-hypothesis filters generate selected high-weight children, then prune, recycle, merge, or cap.','Top-k restriction changes marginals after renormalization. Full tail mass is available only in this exhaustive demo.'),
      ('Marginalize now, or preserve alternative histories until later evidence resolves them.','A product-of-marginals projection discards dependence. Gaussian moment matching is a separate approximation.'),
      ('Marginalization can merge modes and contribute to coalescence, while PMBM-style global hypotheses preserve alternative association histories and then prune them under computational pressure.','Marginalization itself need not merge modes. A track-product projection loses cross-track dependence; optional Gaussian moment matching can additionally merge modes. These are filter choices, not requirements of BP. Retained hypotheses preserve alternatives only until pruning removes them. A trajectory posterior is needed when trajectories are the explicit state.'),
      ('BP-BASED TRACKER','MARGINAL-FILTER EXAMPLE'),
      ('identity ambiguity can survive','retained alternatives can survive'),
      ('Approximate a PMBM-style mixture by one multi-Bernoulli using marginal association probabilities.','TOMB/P approximates the detected-object MBM by one MB and retains the undetected-object PPP.'),
      ('The bridge is precise: TOMB/P uses marginal association probabilities to approximate a mixture by a single multi-Bernoulli.','TOMB/P uses track-oriented marginal association probabilities to approximate the MBM by one MB while retaining the PPP. MOMB/P uses a different grouping.'),
      ('PMBM-STYLE MIXTURE','DETECTED-OBJECT MBM'),
      ('A specific bridge—not a universal equivalence between BP/SPA trackers and PMBM.','PPP retained on both sides. This is a specific projection, not a universal BP = PMBM equivalence.'),
      ('• accuracy outweighs flat cost','• joint dependence is worth its cost'),
      ('• pruning mass can be monitored','• hypothesis growth fits the budget'),
      ('BP returns approximate marginals.','Association BP approximates marginals.'),
      ('PPP-driven birth evidence','PPP-driven new-detection evidence'),
      ('One Association Problem, Two Philosophies','One Association Problem, Two Views'),
    ]
    for old,new in pairs:s=s.replace(old,new)
    # Keep all diagram endpoints at their specified locations (rotation is about the rectangle centre).
    old='''graphElements.push(rect("graph-edge-" + i + "-" + j, x1, y1, Math.hypot(x2 - x1, y2 - y1), 2, TRACKS[i], {'''
    new='''const length = Math.hypot(x2-x1,y2-y1);
  graphElements.push(rect("graph-edge-" + i + "-" + j, (x1+x2-length)/2, (y1+y2)/2-1, length, 2, TRACKS[i], {'''
    s=replace(s,old,new)
    s=s.replace('const rightYs = [250, 345, 440, 535];','const rightYs = [285, 375, 465, 555];')
    s=s.replace('04 · FACTOR GRAPH','04 · PAIRWISE CONSISTENCY GRAPH')
    s=s.replace('marginal p(aᵢ=j)','BP belief q(aᵢ=j)')
    s=s.replace('projected into marginals','not represented by marginals alone')
    s=s.replace('preserved across tracks and hypotheses','encoded by the retained mixture')
    # Ensure this critical distinction is visible, not merely in speaker notes.
    s=s.replace('≠ a complete PMBM posterior update','BP = inference algorithm; PMBM = posterior family. They can be combined.')
    extra=r'''
const pmbmWeightsSlide = regular(
  "s-pmbm-weights", "PMBM · ACTUAL ASSOCIATION EVIDENCE",
  "What changes in a full PMBM update?",
  "Condition on one predicted parent hypothesis. Integrate states first; then solve the same assignment structure.",
  "r_i is Bernoulli existence, p_i(x) is its conditional state density, lambda^u is the undetected PPP intensity, c(z) is clutter intensity, and g(z|x) is the likelihood. All equations are before gating. The parent's hypothesis weight multiplies each child; all parents and histories must be included in a complete PMBM. New means newly detected, not necessarily physically born this scan. New existence below is conditional on the measurement not being assigned to an existing track. See the article for state densities and both set-partition sums in the PMBM density.",
  [
    rect("pw-left",72,226,554,326,WHITE,{stroke:LINE,strokeWidth:1,radius:12}),
    text("pw-head",98,244,500,28,"EXISTING BERNOULLI COMPONENTS",14,BP_DEEP,{fontWeight:700}),
    text("pw-old",98,289,500,162,
      "ρᵢ₀ = 1 − rᵢ + rᵢ ∫ (1−pᴅ(x)) pᵢ(x) dx<br><br>ρᵢⱼ = rᵢ ∫ pᴅ(x) g(zⱼ|x) pᵢ(x) dx<br><br>ℓᵢ₀ = ρᵢ₀ &nbsp;;&nbsp; ℓᵢⱼ = ρᵢⱼ / qⱼ",21,INK,{lineHeight:1.45}),
    text("pw-explanation",98,480,500,50,"The toy weights require rᵢ = 1 and zero PPP evidence. A full update also changes existence and state density.",17,SOFT,{lineHeight:1.35}),
    rect("pw-right",654,226,554,326,PM_WASH,{stroke:PM,strokeWidth:1,radius:12}),
    text("pw-head2",680,244,500,28,"UNASSIGNED-MEASUREMENT EVIDENCE",14,PM_DEEP,{fontWeight:700}),
    text("pw-new",680,289,500,162,
      "eⱼ = ∫ pᴅ(x) g(zⱼ|x) λᵘ(x) dx<br><br>qⱼ = c(zⱼ) + eⱼ<br><br>rⱼ,new = eⱼ / qⱼ",21,INK,{lineHeight:1.45}),
    text("pw-condition",680,480,500,54,"Conditional on zⱼ being unassigned. Marginal existence = P(unassigned zⱼ | Z) · eⱼ/qⱼ.",17,SOFT,{lineHeight:1.35}),
    text("pw-weight",98,573,1080,60,"Child weight ∝ parent weight × ∏<sub>i</sub> ρ<sub>i,aᵢ</sub> × ∏<sub>j unassigned</sub> q<sub>j</sub>.<br>Divide out common ∏<sub>all j</sub> q<sub>j</sub> to obtain the normalized ℓ weights above.",20,INK,{fontWeight:700,align:"center",lineHeight:1.4})
  ],{sectionColor:PM_DEEP}
);
slides.splice(slides.findIndex(s=>s.id==="s-pmbm")+1,0,pmbmWeightsSlide);

const dependenceSlide = regular(
  "s-dependence", "MARGINALS · WHAT IS LOST?",
  "Exact marginals do not determine the joint posterior.",
  "Two certain tracks, two measurements, no misses in this illustrative distribution: only two legal joint events.",
  "This is an illustrative discrete posterior, not a call to the positive-miss BP demo. The exact joint puts probability one half on each permutation. Each marginal is uniform. Multiplying those exact marginals creates four events with weight one quarter, including two illegal double claims. Thus loss of dependence comes from a product projection, even with exact marginals. It is not caused by BP convergence error, and does not require moment matching a state density.",
  [
    rect("dep-left",72,234,536,290,PM_WASH,{stroke:PM,strokeWidth:1,radius:12}),
    text("dep-left-head",104,262,470,28,"EXACT JOINT: TWO COMPATIBLE EVENTS",14,PM_DEEP,{fontWeight:700}),
    text("dep-left-body",104,310,470,180,"P(a₁=1, a₂=2) = 1/2<br><br>P(a₁=2, a₂=1) = 1/2<br><br>No event assigns one measurement twice.",24,INK,{lineHeight:1.4}),
    rect("dep-right",672,234,536,290,WHITE,{stroke:BP,strokeWidth:1,radius:12}),
    text("dep-right-head",704,262,470,28,"PRODUCT OF EXACT MARGINALS",14,BP_DEEP,{fontWeight:700}),
    text("dep-right-body",704,310,470,180,"P(aᵢ=1) = P(aᵢ=2) = 1/2<br><br>∏ᵢ P(aᵢ) gives four events of 1/4.<br><br>Two are illegal double claims: total 1/2.",23,INK,{lineHeight:1.4}),
    text("dep-note",106,562,1070,66,"Marginalization ≠ product projection ≠ Gaussian moment matching.<br>BP only estimates marginals here; it does not instruct you to sample assignments independently.",22,INK,{fontWeight:700,align:"center",lineHeight:1.4})
  ]
);
slides.splice(slides.findIndex(s=>s.id==="s-time"),0,dependenceSlide);

// Small, clickable primary references are present on every teaching slide.
const references = [
 {text:"Williams & Lau (2014), §§III–IV · association BP",url:"https://arxiv.org/abs/1209.6299"},
 {text:"García-Fernández et al. (2018), §III · PMBM",url:"https://arxiv.org/abs/1703.04264"},
 {text:"Williams (2015), §§III–V · TOMB/P and MOMB/P",url:"https://arxiv.org/abs/1203.2995"}
];
for(const slide of slides){
 if(slide.id==="s-cover")continue;
 const r=slide.id.includes("pmbm")||slide.id.includes("joint")?references[1]:["s-time","s-bridge","s-dependence","s-decision"].includes(slide.id)?references[2]:references[0];
 slide.elements.push(text("primary-reference",72,661,1040,17,r.text,10,SOFT,{fontFamily:SANS,link:r.url}));
}
'''
    s=replace(s,'const doc = {',extra+'\nconst doc = {')
    s=replace(s,'html = html.replace("<title>bento/slides</title>", "<title>BP × PMBM — data association slides | Bai Liping</title>");',r'html = html.replace(/<title>[^<]*<\/title>/, "<title>BP × PMBM — data association slides | Bai Liping</title>");')
    path.write_text(s)

if __name__=='__main__':
    article()
    deck()
