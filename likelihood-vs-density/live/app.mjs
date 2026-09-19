import {normal,binomial,betaPdf,coin,integrate} from '../math.mjs';
const params=new URLSearchParams(location.search),mode=['coin','gaussian','prior'].includes(params.get('demo'))?params.get('demo'):'coin';
document.documentElement.classList.toggle('embed',params.has('embed'));
const $=id=>document.getElementById(id),fmt=x=>Math.abs(x)<.001&&x!==0?x.toExponential(2):Number(x.toFixed(3)).toString();
$('demo').value=mode;$('demo').onchange=e=>{location.search='?demo='+e.target.value};
const defs={coin:[['theta','Candidate θ',.01,.99,.01,.5],['n','Number of flips',1,30,1,10],['k','Observed heads',0,10,1,7]],gaussian:[['theta','Candidate θ',-3,3,.1,1],['y','Observed y',-3,3,.1,1.5],['gain','Gain b',.5,3,.1,2],['sigma','Noise σ',.3,2,.1,.7]],prior:[['n','Number of flips',1,30,1,10],['k','Observed heads',0,10,1,7],['alpha','Prior α',1,12,1,1],['beta','Prior β',1,12,1,1]]}[mode];
$('controls').innerHTML=defs.map(([id,label,min,max,step,value])=>`<div class="control"><label for="${id}">${label}<output id="${id}-value" for="${id}">${value}</output></label><input id="${id}" type="range" min="${min}" max="${max}" step="${step}" value="${value}"></div>`).join('')+'<button id="reset" type="button">Reset</button>';
const read=()=>Object.fromEntries(defs.map(([id])=>[id,Number($(id).value)]));
const metric=(label,value)=>`<div class="metric">${label}<b>${value}</b></div>`;
/** Independent SVG charts; axes report actual heights, not unit-peak rescaling. */
function chart(title,xlabel,ylabel,lo,hi,fn,{bars=false,mark=null,chosen=null}={}){
 const W=500,H=255,L=55,R=16,T=25,B=43,w=W-L-R,h=H-T-B,pts=bars?Array.from({length:hi-lo+1},(_,i)=>[lo+i,fn(lo+i)]):Array.from({length:401},(_,i)=>{const x=lo+(hi-lo)*i/400;return[x,fn(x)]});
 const ymax=Math.max(...pts.map(p=>p[1]),1e-12)*1.14,x=v=>L+w*(v-lo)/(hi-lo||1),y=v=>T+h*(1-v/ymax),ink='#2f6b4f',alt='#a94f2a';
 let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${title}. Horizontal axis ${xlabel}; vertical axis ${ylabel}."><text x="${L}" y="13" font-size="12" fill="#66756e">${ylabel}</text>`;
 for(let i=0;i<=2;i++){const v=ymax*i/2;s+=`<path d="M${L},${y(v)}H${W-R}" stroke="#e0e5df"/><text x="${L-7}" y="${y(v)+4}" text-anchor="end" font-size="11" fill="#66756e">${fmt(v)}</text>`;}
 s+=`<path d="M${L},${T}V${T+h}H${W-R}" fill="none" stroke="#66756e"/>`;
 if(bars){const bw=Math.min(w/(hi-lo+2)*.68,36);for(const [v,p]of pts)s+=`<rect x="${x(v)-bw/2}" y="${y(p)}" width="${bw}" height="${T+h-y(p)}" fill="${v===chosen?alt:ink}" opacity=".82"/>`;}
 else s+=`<path d="${pts.map(([a,b],i)=>(i?'L':'M')+x(a).toFixed(2)+','+y(b).toFixed(2)).join(' ')}" fill="none" stroke="${ink}" stroke-width="2.7"/>`;
 if(mark!==null){const v=fn(mark);s+=`<path d="M${x(mark)},${T+h}V${y(v)}" stroke="${alt}" stroke-dasharray="4 4"/><circle cx="${x(mark)}" cy="${y(v)}" r="4" fill="${alt}"/>`;}
 for(const v of [lo,(hi+lo)/2,hi])s+=`<text x="${x(v)}" y="${T+h+17}" text-anchor="middle" font-size="11" fill="#66756e">${fmt(v)}</text>`;
 s+=`<text x="${L+w/2}" y="${H-5}" text-anchor="middle" font-size="13" fill="#203129">${xlabel}</text></svg>`;
 return `<section class="plot"><h2>${title}</h2>${s}</section>`;
}
function render(){
 if($('n')){$('k').max=$('n').value;if(+$('k').value>+$('n').value)$('k').value=$('n').value;}
 const v=read();for(const [id]of defs)$(id+'-value').textContent=fmt(v[id]);
 $('plots').classList.toggle('three',mode==='prior');
 if(mode==='coin'){
  const {n,k,theta}=v,c=coin(n,k),pmf=j=>binomial(n,j,theta),like=t=>binomial(n,k,t),sum=Array.from({length:n+1},(_,i)=>pmf(i)).reduce((a,b)=>a+b,0);
  $('plots').innerHTML=chart(`PMF: θ fixed at ${fmt(theta)}`,'possible head count K','probability mass',0,n,pmf,{bars:true,chosen:k})+chart(`Likelihood: K fixed at ${k}`,'candidate parameter θ','likelihood value',0,1,like,{mark:theta});
  $('metrics').innerHTML=metric('Sum over K',fmt(sum))+metric('Area over θ',fmt(c.area))+metric('MLE θ',fmt(c.mle));
  $('caption').textContent='The orange bar is the observed count; the orange point evaluates its likelihood at the selected θ. Area = 1/(n+1), not one. No prior is used.';
 }else if(mode==='gaussian'){
  const {theta,y:obs,gain:b,sigma:s}=v,mean=b*theta,ml=obs/b,density=y=>normal(y,mean,s),like=t=>normal(obs,b*t,s),yl=Math.min(mean,obs)-4*s,yh=Math.max(mean,obs)+4*s,tl=Math.min(theta,ml)-4*s/b,th=Math.max(theta,ml)+4*s/b;
  $('plots').innerHTML=chart(`Density: θ fixed at ${fmt(theta)}`,'possible measurement y','density in y',yl,yh,density,{mark:obs})+chart(`Likelihood: y fixed at ${fmt(obs)}`,'candidate state θ','likelihood value',tl,th,like,{mark:theta});
  $('metrics').innerHTML=metric('Full density area in y','1')+metric('Full likelihood area in θ',fmt(1/b))+metric('MLE θ = y/b',fmt(ml));
  $('caption').textContent='Model: Y | θ ∼ N(bθ, σ²). Both orange points have the same height. Areas are exact over the full real line; finite plotting windows omit tails. No prior is used.';
 }else{
  const {n,k,alpha:a,beta:b}=v,c=coin(n,k,a,b),prior=t=>betaPdf(t,a,b),like=t=>binomial(n,k,t),posterior=t=>betaPdf(t,c.alpha,c.beta);
  $('plots').innerHTML=chart(`Prior: Beta(${a}, ${b})`,'parameter θ','density in θ',0,1,prior)+chart(`Likelihood: ${k} heads / ${n}`,'parameter θ','likelihood value',0,1,like)+chart(`Posterior: Beta(${c.alpha}, ${c.beta})`,'parameter θ','density in θ',0,1,posterior);
  $('metrics').innerHTML=metric('Likelihood area',fmt(c.area))+metric('Posterior area','1')+metric('Posterior mean',fmt(c.mean))+metric('Evidence P(K=k)',fmt(integrate(t=>prior(t)*like(t),0,1,2000)));
  $('caption').textContent='Each plot has its own labeled vertical scale. Prior and posterior integrate to one; the raw likelihood is not normalized in θ. Changing α or β does not change the likelihood.';
 }
}
for(const [id]of defs)$(id).addEventListener('input',render);
$('reset').onclick=()=>{for(const[id,,min,max,,value]of defs){$(id).max=max;$(id).min=min;$(id).value=value;}render();};
render();
if(parent!==window){
 const target=location.origin==='null'?'*':location.origin;
 parent.postMessage({type:'bento-inline-ready'},target);
 addEventListener('keydown',event=>{
  if(event.target.closest('input,select,textarea,button')||event.ctrlKey||event.altKey||event.metaKey)return;
  const direction=['ArrowRight','PageDown',' '].includes(event.key)?1:['ArrowLeft','PageUp'].includes(event.key)?-1:0;
  if(direction){event.preventDefault();parent.postMessage({type:'bento-inline-nav',direction},target);}
 });
}
