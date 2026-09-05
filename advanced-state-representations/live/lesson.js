/* UI owns canonical controls; models and SVG figures are independent of the DOM. */
'use strict';
const F=window.ASRFigures;
const params=new URLSearchParams(location.search);
const demo=Object.hasOwn(F.defaults,params.get('demo'))?params.get('demo'):'tangent';
const embedded=params.get('embed')==='region';
document.documentElement.classList.toggle('embedded',embedded);
const config={
 tangent:{name:'A real tangent, not an arbitrary line',intro:'Start at the marked rotation. Increase the step and compare the straight displacement with the valid group update.',controls:[['theta','Starting angle (deg)',-120,120,5],['delta','Tangent increment (rad)',-1.2,1.2,.05]]},
 manifold:{name:'Does the update remain a rotation?',intro:'Repeat the same increment. Check both determinant and orthogonality, not just the direction of one arrow.',controls:[['delta','Increment (deg)',-20,20,1],['n','Number of updates',0,24,1]]},
 optimize:{name:'Watch the optimizer actually solve',intro:'The numbered correspondences stay fixed. Advance one iteration and compare residual length, cost, and group validity.',controls:[['angle','Initial orientation (deg)',-100,100,5],['steps','Gauss–Newton iteration',0,8,1]]},
 adjoint:{name:'Same motion, different coordinates',intro:'The correctly converted left update overlaps the right update. The coral frame uses the same numbers on the wrong side.',controls:[['theta','Starting orientation (deg)',-90,90,5],['tx','Starting x-position (m)',-.5,1.8,.1],['delta','Angular increment (deg)',-60,60,5]]},
 spline:{name:'One coefficient, one support interval',intro:'Select a coefficient and move it. The shaded region is its mathematical support, not an arbitrary neighborhood.',controls:[['degree','Spline degree',[[1,'Linear (degree 1)'],[3,'Cubic (degree 3)']]],['t','Query time (s)',0,6,.05],['selected','Moved coefficient index',0,8,1],['shift','Coefficient displacement (m)',-.8,.8,.1]]},
 gp:{name:'Motion prior, measurements, uncertainty',intro:'Compare position random walk with a position–velocity prior. Then inspect the query variance and matrix structure.',controls:[
  ['kind','Motion prior',[['rw','Random walk: state = position'],['cv','Constant velocity: state = [p, v]']]],
  ['q','Process spectral density q',.05,1,.05],
  ['sigma','Measurement standard deviation (m)',.05,.5,.05],
  ['t','Query time (s)',0,6,.05],
  ['matrix','Show matrix',[['info','Posterior information'],['cov','Posterior covariance']]]
 ]},
 pose:{name:'Valid poses can follow different paths',intro:'Compare group interpolation with a valid rotation/translation split and an invalid entrywise matrix blend.',controls:[['angle','End orientation (deg)',-175,180,5],['u','Fraction of interval',0,1,.05]]}
};
const cfg=config[demo],state={...F.defaults[demo]},app=document.getElementById('app');
document.title=cfg.name+' · Advanced state representations';
const controls=cfg.controls.map(([key,label,min,max,step])=>Array.isArray(min)?`<label class="control" for="${key}"><span>${label}</span><select id="${key}" name="${key}">${min.map(([v,l])=>`<option value="${v}" ${String(state[key])===String(v)?'selected':''}>${l}</option>`).join('')}</select></label>`:`<label class="control" for="${key}"><span>${label}<output id="${key}-value" for="${key}"></output></span><input id="${key}" name="${key}" type="range" min="${min}" max="${max}" step="${step}" value="${state[key]}"></label>`).join('');
app.innerHTML=`<section class="lab" aria-label="${cfg.name}"><aside><h1>${cfg.name}</h1><p class="intro">${cfg.intro}</p><div class="controls">${controls}</div><div class="actions"><button id="reset" type="button">Reset</button>${demo==='optimize'?'<button id="step" type="button">One iteration →</button>':''}${demo==='pose'?'<button id="collapse" type="button">180° halfway</button>':''}</div><p class="keys">Page Up / Down: slides. Escape: return focus.<br>Arrow keys adjust the focused slider.</p></aside><div class="result"><div id="figure"></div><div id="metrics" aria-live="polite" aria-atomic="true"></div><p id="caption"></p></div></section>`;
let scheduled=false;
function draw(){scheduled=false;const r=F.render(demo,state);document.getElementById('figure').innerHTML=r.svg;document.getElementById('metrics').innerHTML=r.metrics.map(([label,value])=>`<div class="metric"><strong>${value}</strong><span>${label}</span></div>`).join('');document.getElementById('caption').textContent=r.caption;for(const [key] of cfg.controls){const out=document.getElementById(key+'-value');if(out)out.textContent=Number.isInteger(state[key])?state[key]:Number(state[key]).toFixed(2);}const step=document.getElementById('step');if(step)step.disabled=state.steps>=8;window.ASRLab={demo,state:{...state},metrics:r.metrics};document.documentElement.dataset.ready='true';}
function requestDraw(){if(!scheduled){scheduled=true;requestAnimationFrame(draw);}}
for(const [key] of cfg.controls){document.getElementById(key).addEventListener('input',e=>{state[key]=typeof F.defaults[demo][key]==='number'?Number(e.target.value):e.target.value;if(key==='angle'&&demo==='optimize'){state.steps=0;document.getElementById('steps').value=0;}requestDraw();});}
function sync(){for(const [key] of cfg.controls)document.getElementById(key).value=state[key];draw();}
document.getElementById('reset').addEventListener('click',()=>{Object.assign(state,F.defaults[demo]);sync();});
document.getElementById('step')?.addEventListener('click',()=>{state.steps=Math.min(8,state.steps+1);sync();});
document.getElementById('collapse')?.addEventListener('click',()=>{state.angle=180;state.u=.5;sync();});
// The site adapter handles navigation and lifecycle. No ongoing animation or timer.
draw();
window.parent.postMessage({type:'bento-inline-ready'},'*');
