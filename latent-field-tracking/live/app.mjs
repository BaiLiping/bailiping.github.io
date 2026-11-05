import {covarianceLab,LearningLab} from './model.mjs';
const $=id=>document.getElementById(id),C={teal:'#16736e',rust:'#b44f37',gray:'#84908f',ink:'#182d33',line:'#d6dedc'};
let lab=new URLSearchParams(location.search).get('lab')==='learning'?'learning':'covariance',cov,learning=new LearningLab(),timer=null,busy=false;
const f=(x,n=2)=>Number.isFinite(x)?x.toFixed(n):'—';
const metric=(name,value)=>`<div class="metric"><span>${name}</span><strong>${value}</strong></div>`;
const peakScale=a=>{const m=Math.max(...a);return a.map(x=>x/m);};
function ticks(a,b,n=4){return Array.from({length:n+1},(_,i)=>a+(b-a)*i/n);}
function number(x){if(Math.abs(x)<.000001)return'0';return Math.abs(x)>=100?x.toFixed(0):Math.abs(x)>=10?x.toFixed(0):Math.abs(x)>=1?x.toFixed(1).replace(/\.0$/,''):x.toFixed(2).replace(/0$/,'');}
function chart(id,{series,xmin,xmax,ymin=0,ymax,xlabel='',ylabel='',markers=[],threshold=null,empty=''}){
 const el=$(id),vb=el.getAttribute('viewBox').split(' ').map(Number),w=vb[2],h=vb[3],m={l:43,r:14,t:12,b:31},pw=w-m.l-m.r,ph=h-m.t-m.b;
 if(!Number.isFinite(ymax)||ymax<=ymin)ymax=ymin+1;
 const X=x=>m.l+(x-xmin)/(xmax-xmin)*pw,Y=y=>m.t+(ymax-y)/(ymax-ymin)*ph;
 let html=`<rect x="${m.l}" y="${m.t}" width="${pw}" height="${ph}" fill="#fffdfa"/>`;
 for(const y of ticks(ymin,ymax,3))html+=`<line x1="${m.l}" y1="${Y(y)}" x2="${w-m.r}" y2="${Y(y)}" stroke="${C.line}" stroke-width=".8"/><text x="${m.l-7}" y="${Y(y)+4}" text-anchor="end">${number(y)}</text>`;
 for(const x of ticks(xmin,xmax,4))html+=`<text x="${X(x)}" y="${h-15}" text-anchor="middle">${number(x)}</text>`;
 html+=`<text x="${w-m.r}" y="${h-1}" text-anchor="end" style="font-size:11px">${xlabel}</text>`;
 if(ylabel)html+=`<text x="${m.l+3}" y="10" style="font-size:10px">${ylabel}</text>`;
 if(threshold!==null)html+=`<line x1="${m.l}" y1="${Y(threshold)}" x2="${w-m.r}" y2="${Y(threshold)}" stroke="${C.rust}" stroke-dasharray="3 4" opacity=".6"/><text x="${w-m.r-3}" y="${Y(threshold)-4}" text-anchor="end" style="font-size:10px;fill:${C.rust}">threshold</text>`;
 for(const marker of markers){html+=`<line x1="${X(marker.x)}" y1="${m.t}" x2="${X(marker.x)}" y2="${h-m.b}" stroke="${marker.color||C.gray}" stroke-dasharray="3 4" opacity=".5"/>`;if(marker.label)html+=`<text x="${X(marker.x)+3}" y="${m.t+12+(marker.row||0)*13}" style="font-size:11px;fill:${marker.color||C.gray}">${marker.label}</text>`;}
 for(const s of series){if(!s.x.length)continue;const path=s.x.map((x,i)=>(i?'L':'M')+X(x).toFixed(2)+','+Y(Math.max(ymin,Math.min(ymax,s.y[i]))).toFixed(2)).join(' ');html+=`<path d="${path}" fill="none" stroke="${s.color}" stroke-width="${s.width||2.2}"${s.dash?' stroke-dasharray="'+s.dash+'"':''} stroke-linejoin="round" stroke-linecap="round"/>`;if(s.x.length===1)html+=`<circle cx="${X(s.x[0])}" cy="${Y(s.y[0])}" r="3" fill="${s.color}"/>`;}
 if(empty)html+=`<text x="${m.l+pw/2}" y="${m.t+ph/2}" text-anchor="middle" style="font-size:14px;fill:${C.ink}">${empty}</text>`;
 el.innerHTML=html;
}
function heatmap(data){const n=data.length,max=Math.max(...data.flat().map(Math.abs)),left=44,top=15,cell=23;let html='';for(let i=0;i<n;i++){html+=`<text x="${left-10}" y="${top+i*cell+16}" text-anchor="end">${i+1}</text><text x="${left+i*cell+12}" y="${top+n*cell+15}" text-anchor="middle">${i+1}</text>`;for(let j=0;j<n;j++){const z=data[i][j]/max,base=z>=0?[22,115,110]:[180,79,55],rgb=base.map(v=>Math.round(255+(v-255)*Math.abs(z)));html+=`<rect x="${left+j*cell}" y="${top+i*cell}" width="${cell-1}" height="${cell-1}" fill="rgb(${rgb})"><title>Sensor ${i+1}, ${j+1}: ${f(data[i][j],3)}</title></rect>`;}}html+=`<text x="113" y="188" text-anchor="middle" style="font-size:10px">rust −${f(max)} · white 0 · teal +${f(max)}</text>`;$('matrix-chart').innerHTML=html;}
function seed(id){const n=Math.max(1,Math.min(9999,Math.round(Number($(id).value)||7)));$(id).value=n;return n;}
function calculateCovariance(){cov=covarianceLab({separation:Number($('separation').value),snr:Number($('snr').value),snapshots:Number($('snapshots').value),seed:seed('cov-seed')});renderCovariance();}
function renderCovariance(){if(!cov)return;const threshold=Number($('threshold').value),peaks=[];for(let i=1;i<cov.beam.length-1;i++)if(cov.beam[i]>=threshold&&cov.beam[i]>cov.beam[i-1]&&cov.beam[i]>=cov.beam[i+1])peaks.push(cov.angles[i]);
 $('separation-value').value=cov.separation+'°';$('snr-value').value=cov.snr+' dB';$('threshold-value').value=f(threshold);$('frame-label').textContent=cov.snapshots+' independent snapshot'+(cov.snapshots===1?'':'s');
 chart('angle-chart',{series:[{x:cov.angles,y:cov.intensityPosterior.map(()=>1),color:C.gray,dash:'5 4',width:1.7},{x:cov.angles,y:cov.beam,color:C.rust},{x:cov.angles,y:peakScale(cov.posterior),color:C.teal,width:2.8}],xmin:-50,xmax:50,ymin:0,ymax:1.08,xlabel:'candidate bearing (degrees)',threshold,markers:[{x:cov.trueAngles[0],label:'known θ₁',row:0},{x:cov.trueAngles[1],label:'true θ₂',row:1,color:C.teal}]});
 heatmap(cov.sampleRe);
 $('cov-metrics').innerHTML=metric('Full-likelihood MAP',f(cov.mapAngle,1)+'°')+metric('True second bearing',f(cov.trueAngles[1],1)+'°')+metric('Beam peaks over threshold',String(peaks.length))+metric('Posterior spread',f(cov.posteriorSD,1)+'°');
 $('angle-note').textContent=(peaks.length?'Beam candidates: '+peaks.map(x=>f(x,1)+'°').join(', ')+'. ':'No beam peak exceeds this threshold. ')+'Full likelihood conditions on the known first source. Curves have separate peak normalization.';
}
function renderLearning(){const s=learning.getState(),t=s.trace,x=t.map(p=>p.frame),fieldX=s.field.map(p=>p.x);$('frame-label').textContent='Frame '+s.frame+' · '+s.trainingCount+' training updates';$('learn-snr-value').value=s.snr+' dB';
 chart('track-chart',{series:[{x,y:t.map(p=>p.truth),color:C.gray,dash:'4 3'},{x,y:t.map(p=>p.fixed),color:C.rust,width:1.8},{x,y:t.map(p=>p.learned),color:C.teal,width:2}],xmin:0,xmax:Math.max(80,s.frame),ymin:-1,ymax:1,xlabel:'frame',empty:s.frame?'':'Press Play, Step or Run to begin'});
 const max=Math.max(...s.field.flatMap(p=>[p.medianTruth,p.medianLearned,p.medianFixed]));chart('field-chart',{series:[{x:fieldX,y:s.field.map(p=>p.medianTruth),color:C.gray,dash:'4 3'},{x:fieldX,y:s.field.map(p=>p.medianFixed),color:C.rust,width:1.6},{x:fieldX,y:s.field.map(p=>p.medianLearned),color:C.teal,width:2.5}],xmin:-.85,xmax:.85,ymin:0,ymax:Math.ceil(max*1.06),xlabel:'context x',markers:[{x:s.context,color:C.teal}]});
 const maxPDF=Math.max(...s.pdfLearned,...s.pdfFixed,...s.pdfTruth)*1.08;chart('density-chart',{series:[{x:s.powerGrid,y:s.pdfTruth,color:C.gray,dash:'4 3'},{x:s.powerGrid,y:s.pdfFixed,color:C.rust,width:1.5},{x:s.powerGrid,y:s.pdfLearned,color:C.teal,width:2.5}],xmin:0,xmax:14,ymin:0,ymax:maxPDF,xlabel:'signal power γ'});
 $('context-label').textContent='previous estimate c = '+f(s.context)+' · σ = '+f(s.sigma);
 $('learn-metrics').innerHTML=metric('Learned-prior RMSE',s.frame?f(s.rmseLearned,3):'—')+metric('Fixed-prior RMSE',s.frame?f(s.rmseFixed,3):'—')+metric('Current true x',f(s.xTruth,3))+metric('Learned estimate x',f(s.xLearned,3));
 $('learning-note').textContent=s.frame?(s.learning?'Posterior E[log γ] has updated the shared mean. ':'Learning is off: both filters use the same prior. ')+'RMSE uses all '+s.frame+' synthetic frames. Truth only evaluates the result.':'No observations yet. The two priors start identical. The gray field and density are hidden truth, shown only for comparison.';
}
function stop(){if(timer){clearInterval(timer);timer=null;}$('play').textContent='▶ Play';$('play').setAttribute('aria-pressed','false');}
function next(){learning.step();renderLearning();}
function resetLearning(){stop();learning.reset({snr:Number($('learn-snr').value),seed:seed('learn-seed'),learning:$('learning-on').checked});renderLearning();}
function setLab(name){stop();lab=name;document.querySelectorAll('[data-lab]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.lab===lab)));for(const name of ['covariance','learning']){$(name+'-controls').hidden=name!==lab;$(name+'-view').hidden=name!==lab;$(name+'-explanation').hidden=name!==lab;}
 $('result-title').textContent=lab==='covariance'?'What survives in the array data?':'Posterior estimates teach a shared power prior';$('scope').textContent=lab==='covariance'?'Synthetic Gaussian array example. Known first bearing and powers. These are computed curves, not paper benchmark results.':'Known-present one-target teaching reduction. Conditional lognormal affine flow, fixed width, regularized mean fit. No paper benchmark reproduction.';
 $('expand').href='./?lab='+lab;const url=new URL(location.href);url.searchParams.set('lab',lab);history.replaceState(null,'',url.pathname+url.search);if(lab==='covariance'){if(!cov)calculateCovariance();else renderCovariance();}else renderLearning();
}
for(const id of ['separation','snr'])$(id).addEventListener('input',calculateCovariance);
for(const id of ['snapshots','cov-seed'])$(id).addEventListener('change',calculateCovariance);
$('threshold').addEventListener('input',renderCovariance);$('resample').onclick=()=>{$('cov-seed').value=seed('cov-seed')%9999+1;calculateCovariance();};
$('cov-reset').onclick=()=>{$('separation').value=14;$('snr').value=8;$('snapshots').value=32;$('threshold').value=.6;$('cov-seed').value=7;calculateCovariance();};
$('learn-snr').addEventListener('input',resetLearning);$('learn-seed').addEventListener('change',resetLearning);$('learning-on').addEventListener('change',resetLearning);$('learn-reset').onclick=resetLearning;
$('step').onclick=()=>{stop();next();};$('play').onclick=()=>{if(timer){stop();return;}$('play').textContent='Ⅱ Pause';$('play').setAttribute('aria-pressed','true');next();timer=setInterval(next,100);};
$('run').onclick=async()=>{if(busy)return;busy=true;stop();$('run').disabled=true;$('run').textContent='Computing…';const steps=Number($('passes').value)*80;await new Promise(resolve=>requestAnimationFrame(resolve));for(let i=0;i<steps;i++)learning.step();renderLearning();$('run').disabled=false;$('run').textContent='Run selected traversals';busy=false;};
for(const b of document.querySelectorAll('[data-lab]'))b.onclick=()=>setLab(b.dataset.lab);
window.addEventListener('bento-live-visibility',e=>{if(e.detail?.paused)stop();});document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});window.addEventListener('pagehide',stop);
setLab(lab);
