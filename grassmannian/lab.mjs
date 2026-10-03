import * as M from './model.mjs';
import * as A from './association.mjs';
import * as V from './visuals.mjs';
const $=id=>document.getElementById(id),params=new URLSearchParams(location.search);
const lab=['basis','angles','geodesic','pca','association'].includes(params.get('lab'))?params.get('lab'):'basis';
document.body.dataset.lab=lab;
const state={phi:45,tilt:28,a:25,b:60,spin:0,theta:65,t:.4,flip:false,lambda:4,angle:55,translation:7,noise:0,rho:3,epsilon:.16,sigma:.05,mode:'shifted',clutter:true};
let view={yaw:.6,pitch:.5},timer=0,run=false,pca=M.pcaData(),Q=M.initialPcaBasis(),history=[M.pcaScore(Q,pca.C)],iterations=0,result;
const title={basis:'One plane. Many bases.',angles:'Measure the difference between subspaces.',geodesic:'Move along a shortest path.',pca:'Learn a plane from a point cloud.',association:'Match lines and planes without a pose guess.'};
$('title').textContent=title[lab];
function slider(id,label,min,max,step,format=x=>x+'°'){
 const div=document.createElement('div');div.className='control';div.innerHTML=`<label for="${id}">${label}<output id="${id}-out"></output></label><input id="${id}" type="range" min="${min}" max="${max}" step="${step}" value="${state[id]}">`;
 div.querySelector('input').addEventListener('input',e=>{state[id]=Number(e.target.value);if(id==='lambda')resetPca();draw();});
 div.dataset.control=id;div.format=format;$('controls').append(div);
}
function buttons(items){const div=document.createElement('div');div.className='buttons';for(const [id,text,fn,primary]of items){const b=document.createElement('button');b.id=id;b.textContent=text;if(primary)b.className='primary';b.addEventListener('click',fn);div.append(b);}$('controls').append(div);}
function hint(s){const p=document.createElement('p');p.className='hint';p.textContent=s;$('controls').append(p);}
function preset(s){stop();Object.assign(state,s);draw();}
function metrics(items){$('metrics').innerHTML=items.map(([v,t])=>`<div class="metric"><strong>${v}</strong><span>${t}</span></div>`).join('');}
function matrix(A){return A.map(r=>r.map(x=>(Math.abs(x)<5e-6?0:x).toFixed(3).padStart(7)).join(' ')).join('\n');}
function stop(){run=false;clearTimeout(timer);timer=0;if($('play'))$('play').textContent=lab==='pca'?'Run descent':'Animate';}
function resetPca(){stop();pca=M.pcaData(state.lambda);Q=M.initialPcaBasis();history=[M.pcaScore(Q,pca.C)];iterations=0;}
function stepPca(){const r=M.pcaStep(Q,pca.C);Q=r.Q;iterations++;history.push(r.score);if(r.gradient<1e-7||iterations>=160)stop();}
function tick(){if(!run||document.hidden)return;if(lab==='pca'){stepPca();}else{state.t=Math.min(1,state.t+.008);if(state.t>=1)stop();}draw();if(run)timer=setTimeout(tick,lab==='pca'?95:25);}
function play(){if(run){stop();return;}if(lab==='geodesic'&&state.t>=1)state.t=0;run=true;$('play').textContent='Pause';tick();}
if(lab==='basis'){
 slider('phi','Rotate the basis',0,180,1);slider('tilt','Tilt the plane',-60,60,1);
 buttons([['quarter','Quarter turn',()=>preset({phi:90})],['flip','Flip both vectors',()=>preset({phi:180})],['reset','Reset',()=>preset({phi:45,tilt:28})]]);
 hint('Rotate the basis: its columns move, but the plane, projector, and projected vector stay fixed. Tilting the plane changes the subspace.');
}else if(lab==='angles'){
 slider('a','Rotation in (e₁,e₃)',0,90,1);slider('b','Rotation in (e₂,e₄)',0,90,1);slider('spin','Rotate Y’s basis',0,180,1);
 buttons([['same','Same',()=>preset({a:0,b:0})],['shared','Shared direction',()=>preset({a:0,b:60})],['orthogonal','Orthogonal',()=>preset({a:90,b:90})],['reset','Reset',()=>preset({a:25,b:60,spin:0})]]);
 hint('The two slices define Gr(2,4). Changing Y’s basis leaves its singular values and both distances unchanged.');$('view-help').textContent='The two coordinate slices show independent rotations in R⁴.';
}else if(lab==='geodesic'){
 slider('theta','Endpoint principal angle',0,90,1);slider('t','Position along path',0,1,.01,x=>Number(x).toFixed(2));
 buttons([['play','Animate',play,true],['flip','Flip endpoint sign',()=>{state.flip=!state.flip;draw();}],['cut','Cut locus: 90°',()=>preset({theta:90,t:.5})],['reset','Reset',()=>preset({theta:65,t:.4,flip:false})]]);
 hint('This is Gr(1,3). The path is on the space of unoriented lines; opposite sphere points are identified.');
}else if(lab==='pca'){
 slider('lambda','Second covariance eigenvalue',1,6,.1,x=>Number(x).toFixed(1));
 buttons([['step','One step',()=>{stop();stepPca();draw();},true],['play','Run descent',play],['reset','Reset',()=>{resetPca();draw();}],['tie','Close the eigenvalue gap',()=>{state.lambda=1;resetPca();draw();}]]);
 hint('Minimize −tr(QᵀCQ) using a horizontal gradient, QR retraction, and backtracking. The covariance is computed from the plotted centered cloud.');
}else{
 $('graph').hidden=false;
 const div=document.createElement('div');div.className='control wide';div.innerHTML='<label for="mode">Pairwise descriptor</label><select id="mode"><option value="shifted">Shifted affine · paper</option><option value="raw">Raw affine · no pair shift</option><option value="directions">Directions only · ignore offsets</option></select>';
 div.querySelector('select').addEventListener('change',e=>{state.mode=e.target.value;draw();});$('controls').append(div);
 slider('angle','Scan rotation',-150,150,5);slider('translation','Translation parameter s',0,18,.5,x=>Number(x).toFixed(1)+' m');
 slider('noise','Position / angle noise',0,.15,.01,x=>Number(x).toFixed(2));slider('rho','Length scale ρ',.5,12,.5,x=>Number(x).toFixed(1)+' m');
 slider('epsilon','Consistency threshold ε',.02,.4,.01,x=>Number(x).toFixed(2)+' rad');
 buttons([['clean','Clean + clutter',()=>preset({noise:0,mode:'shifted',rho:3,epsilon:.16})],['noisy','Noisy + clutter',()=>preset({noise:.04,mode:'shifted',rho:3,epsilon:.16})],['reset','Reset',()=>preset({angle:55,translation:7,noise:0,mode:'shifted',rho:3,epsilon:.16})]]);
 hint('σ = 0.05 rad. All same-type candidates are scored using geometry. Truth labels are used only to evaluate the selected matches.');
}
function draw(){
 document.querySelectorAll('[data-control]').forEach(d=>{const id=d.dataset.control;$(id).value=state[id];$(id+'-out').textContent=d.format(state[id]);});
 const ctx=V.setupCanvas($('scene'));
 if(lab==='basis'){
  const Q=M.planeBasis(M.rad(state.tilt)),Y=M.mul(Q,M.rotation(M.rad(state.phi))),P=M.projector(Q),err=M.norm(M.add(P,M.projector(Y),-1));
  V.basisPicture(ctx,Q,Y,view);metrics([[M.norm(M.add(Q,Y,-1)).toFixed(3),'Basis difference ‖Q − Y‖F'],[err.toExponential(1),'Projector difference ‖QQᵀ − YYᵀ‖F'],['2','dim Gr(2,3) = 2(3 − 2)']]);
  $('observation').textContent=`At ${state.phi}°, the basis matrices differ, yet they represent the same plane. Rotating or flipping a basis is a change of coordinates within one Grassmannian point.`;
  $('detail').innerHTML=`<div class="matrix-pair"><pre>Q =\n${matrix(Q)}</pre><pre>Y = QO =\n${matrix(Y)}</pre><pre>P = QQᵀ = YYᵀ\n${matrix(P)}</pre></div><p>The colored finite patch is a display window onto an infinite plane through the origin. Orthonormality residual: ${M.orthError(Y).toExponential(2)}.</p>`;
  result={Q,Y,projectorError:err};
 }else if(lab==='angles'){
  result=M.anglePair(M.rad(state.a),M.rad(state.b),M.rad(state.spin));V.anglesPicture(ctx,M.rad(state.a),M.rad(state.b));
  metrics([[result.angles.map(x=>M.deg(x).toFixed(1)+'°').join(' · '),'Sorted principal angles θ₁ ≤ θ₂'],[result.geodesic.toFixed(3)+' rad','Canonical geodesic distance ‖θ‖₂'],[result.projection.toFixed(3),'Projection distance ‖sin θ‖₂']]);
  $('observation').textContent=`Rotations of ${state.a}° and ${state.b}° produce two principal angles, sorted by magnitude. The angle-to-distance calculation uses radians. A ${state.spin}° basis rotation changes no subspace distance.`;
  $('detail').innerHTML=`<pre>QᵀY =\n${matrix(M.mul(M.transpose(result.Q),result.Y))}</pre><p>Singular values: ${result.angles.map(x=>Math.cos(x).toFixed(6)).join(', ')}. Projection distance equals ‖QQᵀ − YYᵀ‖F / √2. In Gr(2,3), two planes always share at least one direction, so one principal angle must be zero; R⁴ lets this experiment vary both.</p>`;
 }else if(lab==='geodesic'){
  const u=[1,0,0],a=M.rad(state.theta),v=[Math.cos(a),Math.sin(a)*.8,Math.sin(a)*.6].map(x=>x*(state.flip?-1:1));
  result=M.lineGeodesic(u,v,state.t);V.geodesicPicture(ctx,u,v,state.t,view);
  metrics([[state.t.toFixed(2),'Interpolation parameter t'],[(state.t*result.theta).toFixed(3)+' rad','Distance from start along this minimizing path'],[Math.abs(M.dot(result.point,result.point)-1).toExponential(1),'Unit-norm residual']]);
  $('observation').textContent=state.theta===90?'At 90°, the minimizing path is not unique. This demo displays one branch; an endpoint sign flip may select the other.':state.theta===0?'Both endpoints represent the same line. A sign flip still leaves a zero-length path.':`A ${state.theta}° endpoint separation traces a constant-speed path between unoriented lines. Flipping the endpoint vector leaves this shortest subspace path unchanged.`;
  $('detail').innerHTML=`<p>After sign alignment, q(t) = u cos(tθ) + w sin(tθ), where w = (v − u cos θ)/sin θ. The zero-angle case is handled separately. The geodesic distance from the start is tθ for 0 ≤ t ≤ 1. No animation runs until you press Animate.</p><pre>q(t) = [${result.point.map(x=>x.toFixed(5)).join(', ')}]</pre>`;
 }else if(lab==='pca'){
  const score=M.pcaScore(Q,pca.C),loss=10+state.lambda-score;
  V.pcaPicture(ctx,pca,Q,history,view);metrics([[score.toFixed(4),'Captured variance · optimum '+pca.optimum.toFixed(1)],[loss.toFixed(4),'Reconstruction loss · optimum 1.0'],[String(iterations),'Gradient steps · eigenvalue gap '+(state.lambda-1).toFixed(1)]]);
  $('observation').textContent=state.lambda===1?'λ₂ = λ₃: more than one plane is globally optimal. Objective convergence cannot identify a unique two-dimensional subspace.':`A positive boundary gap (λ₂ − λ₃ = ${(state.lambda-1).toFixed(1)}) makes the leading two-dimensional eigenspace unique. QR keeps the iterate on the manifold; the line search accepts only sufficient objective improvement.`;
  $('detail').innerHTML=`<p>96 deterministic centered samples, using orthogonal Fourier columns, have covariance eigenvalues (9, ${state.lambda.toFixed(1)}, 1); this is a synthetic teaching cloud, not Gaussian data. We use C = XᵀX / N. Gradient: −2(I − QQᵀ)CQ. Initial step 0.15; Armijo coefficient 10⁻⁴. The spectral solution provides the comparison optimum.</p><pre>C =\n${matrix(pca.C)}\n\n‖grad f‖F = ${M.norm(M.pcaGradient(Q,pca.C)).toExponential(3)}\n‖QᵀQ − I‖F = ${M.orthError(Q).toExponential(3)}</pre>`;
  result={score,loss,iterations,Q,orthError:M.orthError(Q),gap:state.lambda-1};
 }else{
  $('mode').value=state.mode;const scene=A.makeScene(state);result=A.associate(scene,state);
  V.associationPicture(ctx,scene,result,view);V.graphPicture(V.setupCanvas($('graph')),result);
  metrics([[String(result.candidates.length),'Same-type candidate matches'],[String(result.matches.length),'Selected one-to-one matches'],[result.correct+' / 6','Ground-truth objects correctly recovered'],[result.density.toFixed(3),'Density uᵀMu / uᵀu · Eq. (9)']]);
  const names=result.matches.map(({i,j})=>`A${i+1} ↔ B${j+1}`).join('; ');
  $('observation').textContent=`${names||'No matches'}. ${state.mode==='shifted'?'Each pair is shifted to its first stored anchor before embedding.':state.mode==='raw'?'Raw affine distances change with the scan origin; compare with the shifted descriptor.':'Directions alone discard the offsets that distinguish parallel objects.'}`;
  const edge=result.selected.length>1?result.pairs.find(p=>p.u===result.selected[0]&&p.v===result.selected[1]):result.pairs[0];
  $('detail').innerHTML=`<p><a href="https://arxiv.org/pdf/2205.08556" target="_top">Lusk &amp; How (2022)</a>: affine embedding (4), pair consistency (8), and the discrete weighted density objective (9). The demo exhaustively evaluates all feasible cliques of 24 vertices. The paper uses the CLIPPER relaxation for larger graphs.</p><p>Example edge: dA = ${edge.d1.toFixed(6)}, dB = ${edge.d2.toFixed(6)}, discrepancy = ${edge.c.toFixed(6)} rad, weight = ${edge.w.toFixed(6)}. Candidate reuse of an object is forbidden. The graph counts the diagonal ones in its density.</p><p>Controlled synthetic scene: three lines, three planes, and one distractor of each type in scan B. Anchors transform with the objects; positions are perturbed by at most the noise setting per coordinate (m), and orientations by at most half that setting (rad). The translation is (s, −0.35s, 0.2s), where s is the slider. Scan B is centered for display only. Display patches are finite windows of infinite objects.</p><p>Mixed-dimensional comparisons use only min(k₁+1,k₂+1) principal angles, as in the paper. This is a dissimilarity across dimensions: containment can yield zero. This demo stops at association; it does not extract lidar features, run the paper’s pose estimator, or reproduce its KITTI results. Defaults ρ = 3 m, ε = 0.16 rad, σ = 0.05 rad are teaching settings (paper: 40 m, 0.2 rad, 0.02 rad).</p>`;
 }
 window.grassmannLab={lab,state:{...state},result,running:run};
}
let drag;
$('scene').addEventListener('pointerdown',e=>{if(lab==='angles')return;drag=[e.clientX,e.clientY,view.yaw,view.pitch];$('scene').setPointerCapture(e.pointerId);});
$('scene').addEventListener('pointermove',e=>{if(!drag)return;view.yaw=drag[2]+(e.clientX-drag[0])*.008;view.pitch=M.clamp(drag[3]+(e.clientY-drag[1])*.008,-1.2,1.2);draw();});
$('scene').addEventListener('pointerup',()=>drag=null);
$('scene').addEventListener('pointercancel',()=>drag=null);
$('scene').addEventListener('keydown',e=>{if(lab==='angles')return;if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','r','R'].includes(e.key)){e.preventDefault();e.stopPropagation();if(e.key.toLowerCase()==='r')view={yaw:.6,pitch:.5};else if(e.key==='ArrowLeft')view.yaw-=.1;else if(e.key==='ArrowRight')view.yaw+=.1;else view.pitch=M.clamp(view.pitch+(e.key==='ArrowUp'?.1:-.1),-1.2,1.2);draw();}});
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&window.parent!==window){e.preventDefault();window.parent.postMessage({type:'grassmann-close'},location.origin);}});
window.addEventListener('message',e=>{if(e.origin===location.origin&&e.source===parent&&e.data?.type==='grassmann-pause'){stop();draw();}});
document.addEventListener('visibilitychange',()=>{if(document.hidden){stop();draw();}});
window.addEventListener('pagehide',stop);
new ResizeObserver(()=>draw()).observe($('scene'));
draw();parent.postMessage({type:'grassmann-ready',lab},location.origin);
