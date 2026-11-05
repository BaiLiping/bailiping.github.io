/* Presentation and SVG labs: runs from file:// with no network. */
(() => {
  'use strict';
  const M = window.CRBMath;
  const $ = id => document.getElementById(id);
  const C = {ink:'#16273e',muted:'#596d80',line:'#d8e1e9',teal:'#087f68',tealsoft:'#edf7f3',blue:'#2766b1',bluesoft:'#edf3fa',rust:'#b96815',rustsoft:'#fff4e7',paper:'#ffffff'};
  const esc = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt = (x,d=3) => !Number.isFinite(x) ? '∞' : x !== 0 && Math.abs(x) < 0.001 ? x.toExponential(2) : x.toFixed(d);
  const svg = (w,h,b) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" aria-hidden="true">${b}</svg>`;
  const txt=(x,y,s,opts={})=>`<text x="${x}" y="${y}" fill="${opts.color||C.muted}" font-family="Arial,Helvetica,sans-serif" font-size="${opts.size||14}" text-anchor="${opts.anchor||'start'}"${opts.weight?' font-weight="'+opts.weight+'"':''}>${esc(s)}</text>`;
  const line=(x1,y1,x2,y2,color=C.line,width=1,dash='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}"${dash?' stroke-dasharray="'+dash+'"':''}/>`;
  const circle=(x,y,r,color,stroke='none',w=1)=>`<circle cx="${x}" cy="${y}" r="${r}" fill="${color}" stroke="${stroke}" stroke-width="${w}"/>`;
  const path=(pts,color,width=2,fill='none',dash='')=>`<path d="${pts.map((p,i)=>(i?'L':'M')+p[0].toFixed(2)+','+p[1].toFixed(2)).join(' ')}" stroke="${color}" stroke-width="${width}" fill="${fill}"${dash?' stroke-dasharray="'+dash+'"':''} stroke-linejoin="round"/>`;
  const pdf=(x,mu,sd)=>Math.exp(-0.5*((x-mu)/sd)**2)/(sd*Math.sqrt(2*Math.PI));
  function plot(w,h,xlo,xhi,ymax,opts={}) {
    const l=52,r=w-18,t=opts.top||25,b=h-39;
    const X=x=>l+(x-xlo)/(xhi-xlo)*(r-l),Y=y=>b-y/ymax*(b-t);
    let out='';
    [0,.5,1].forEach(k=>{out+=line(l,Y(k*ymax),r,Y(k*ymax),C.line);out+=txt(l-8,Y(k*ymax)+4,fmt(k*ymax,2),{size:11,anchor:'end'});});
    for(let i=0;i<=4;i++) {let x=xlo+(xhi-xlo)*i/4;out+=txt(X(x),b+20,fmt(x,1),{size:12,anchor:'middle'});}
    out+=txt(l,13,opts.yLabel||'Density',{size:12})+txt((l+r)/2,h-1,opts.xLabel||'Estimate',{size:12,anchor:'middle'});
    return {X,Y,l,r,t,b,w,h,xlo,xhi,ymax,out};
  }
  function density(P,mu,sd,color,{fill=false,dash='',width=2.5}={}){
    if(!(sd>0))return '';
    const lo=Math.max(P.xlo,mu-5*sd),hi=Math.min(P.xhi,mu+5*sd);
    if(hi<lo)return '';
    const pts=Array.from({length:361},(_,i)=>{let x=lo+(hi-lo)*i/360;return [P.X(x),P.Y(pdf(x,mu,sd))];});
    let out='';
    if(fill)out+=path([[P.X(lo),P.b],...pts,[P.X(hi),P.b]],'none',0,color+'22');
    return out+path(pts,color,width,'none',dash);
  }
  function staticCharts(){
    document.querySelectorAll('[data-chart]').forEach(el=>{
      const name=el.dataset.chart;let b='';
      if(name==='hero'){
        const nrm=M.normalGenerator(1945),W=535,H=470,mid=292;
        b+=`<rect x="10" y="7" width="510" height="450" rx="20" fill="${C.tealsoft}"/>`;
        b+=txt(39,43,'SAME TRUTH · LESS SPREAD',{size:11,color:C.teal,weight:700});
        b+=line(mid,68,mid,348,C.teal,1.4,'5 5');
        [4,16,64].forEach((n,i)=>{const y=111+102*i;b+=line(115,y,482,y,'#b9cfc6');b+=txt(38,y-6,'n = '+n,{size:17,color:C.ink,serif:true});b+=txt(38,y+16,'SD '+fmt(2/Math.sqrt(n),2),{size:11});for(let j=0;j<130;j++){const x=mid+(2/Math.sqrt(n))*65*nrm();const yy=y+7*nrm();if(x>110&&x<490)b+=circle(x,yy,2.6,i===1?C.blue:C.teal);}});
        b+=txt(mid,375,'True μ',{size:15,anchor:'middle',color:C.teal});
        b+=txt(269,422,'Var(X̄) = σ² / n',{size:36,serif:true,color:C.ink,anchor:'middle'});
        el.innerHTML=svg(W,H,b);return;
      }
      if(name==='overlap'){
        const W=535,H=285,X=x=>45+(x+4)/8*465;
        [[1.45,117,'Noisier observations'],[.42,264,'Cleaner observations']].forEach(([sd,base,label])=>{
          b+=txt(45,base-99,label,{size:14,color:C.ink});b+=line(45,base,510,base);
          [0,1].forEach((mu,k)=>{let pts=Array.from({length:251},(_,i)=>{let x=-4+8*i/250;return[X(x),base-92*pdf(x,mu,sd)];});b+=path([[45,base],...pts,[510,base]],'none',0,(k?C.rust:C.teal)+'1b')+path(pts,k?C.rust:C.teal,2.5);b+=txt(X(mu),base+16,k?'θ + δ':'θ',{size:12,anchor:'middle',color:k?C.rust:C.teal});});
        });el.innerHTML=svg(W,H,b);return;
      }
      if(name==='likelihood'){
        const W=520,H=215,l=35,r=498,base=174,X=x=>l+(x+3)/6*(r-l);
        b+=txt(l,17,'Likelihood / maximum',{size:12})+line(l,base,r,base);
        [1.4,.48].forEach((sd,k)=>{let pts=Array.from({length:251},(_,i)=>{let x=-3+6*i/250;return[X(x),base-131*Math.exp(-.5*(x/sd)**2)];});b+=path(pts,k?C.teal:C.blue,3);});
        b+=txt(X(0),201,'Candidate parameter θ',{anchor:'middle',size:13})+txt(320,78,'Sharper → more information',{color:C.teal,size:12,anchor:'middle'});
        el.innerHTML=svg(W,H,b);return;
      }
      if(name==='scaling'){
        const W=535,H=350,l=57,r=510,t=25,base=286,X=n=>l+Math.log2(n)/7*(r-l),Y=y=>base-(Math.log2(y)+6)/8*(base-t);
        [.03125,.125,.5,2,4].forEach(y=>{b+=line(l,Y(y),r,Y(y));b+=txt(l-9,Y(y)+4,String(y),{size:11,anchor:'end'});});
        [1,4,16,64,128].forEach(n=>b+=txt(X(n),307,n,{size:12,anchor:'middle'}));
        const values=Array.from({length:200},(_,i)=>Math.pow(2,7*i/199));
        b+=path(values.map(n=>[X(n),Y(4/n)]),C.teal,3)+path(values.map(n=>[X(n),Y(2/Math.sqrt(n))]),C.rust,3);
        [4,16,64].forEach(n=>{b+=circle(X(n),Y(4/n),4,C.teal);b+=circle(X(n),Y(2/Math.sqrt(n)),4,C.rust);});
        b+=txt(l,12,'Bound · log scale',{size:12})+txt((l+r)/2,344,'Independent samples n · log scale',{size:12,anchor:'middle'});
        b+=txt(310,93,'SD: 2 / √n',{color:C.rust,size:16})+txt(310,223,'Variance: 4 / n',{color:C.teal,size:16});
        el.innerHTML=svg(W,H,b);return;
      }
      if(name==='bias-cloud'){
        const P=plot(535,208,-3,3,1.15);b=P.out+density(P,0,.8,C.teal,{fill:true})+density(P,1.1,.37,C.rust,{fill:true});b+=line(P.X(0),P.t,P.X(0),P.b,C.ink,1.4,'4 4');b+=txt(P.X(0)+8,P.t+12,'Truth',{color:C.ink,size:12});
        el.innerHTML=svg(P.w,P.h,b);return;
      }
      if(name==='ellipse'){
        const W=510,H=330,cx=250,cy=155,ang=-.55;
        const point=(a,z)=>[cx+a*Math.cos(ang)-z*Math.sin(ang),cy+a*Math.sin(ang)+z*Math.cos(ang)];
        b+=line(45,cy,469,cy)+line(cx,29,cx,281);
        let pts=Array.from({length:181},(_,i)=>point(174*Math.cos(i*Math.PI/90),53*Math.sin(i*Math.PI/90)));
        b+=path(pts,C.teal,2.5,C.teal+'20');
        let a=point(-200,0),d=point(200,0);b+=line(...a,...d,C.teal,1.7,'5 4');
        a=point(0,-77);d=point(0,77);b+=line(...a,...d,C.blue,2);
        b+=circle(cx,cy,4,C.ink)+txt(44,20,'Covariance-bound ellipse',{size:15,color:C.ink});
        b+=txt(260,293,'Weak information → long error axis',{size:15,anchor:'middle',color:C.teal});
        b+=txt(260,318,'Strong information → short error axis',{size:15,anchor:'middle',color:C.blue});
        el.innerHTML=svg(W,H,b);return;
      }
      if(name==='range-direction'){
        const W=515,H=283,ax=151,ay=144,px=280,py=144;
        b+=`<circle cx="${ax}" cy="${ay}" r="129" fill="none" stroke="${C.line}" stroke-width="2"/>`;
        b+=line(ax,ay,px,py,C.blue,2,'5 5')+line(px,64,px,229,C.muted,2,'5 4')+line(px,py,393,py,C.teal,3);
        b+=`<path d="M393,144 l-12,-6 v12 Z" fill="${C.teal}"/>`;
        b+=circle(ax,ay,9,C.blue)+circle(px,py,8,C.rust);
        b+=txt(ax,ay+29,'Anchor',{size:15,color:C.blue,anchor:'middle'})+txt(px-4,py-15,'Target',{size:15,color:C.rust,anchor:'end'});
        b+=txt(330,121,'Radial sensitivity',{color:C.teal,size:14})+txt(300,239,'Tangent: zero first derivative',{size:13});
        b+=txt(30,22,'A constant-range circle',{size:13});el.innerHTML=svg(W,H,b);return;
      }
    });
  }
  // ----- Lab 1: actual repeated independent Gaussian datasets -----
  const mc={n:16,sigma:2,mu:1,seed:42,estimator:'mean',values:[],last:null,normal:null};
  function simulate(count){
    count=Math.min(count,20000-mc.values.length);
    for(let i=0;i<count;i++) {mc.last=M.gaussianTrial(mc.n,mc.sigma,mc.mu,mc.normal);mc.values.push(mc.last[mc.estimator]);}
    renderMC();
  }
  function replayMC(){mc.values=[];mc.normal=M.normalGenerator(mc.seed);simulate(2000);}
  function resetMC(){mc.n=16;mc.sigma=2;mc.seed=42;mc.estimator='mean';$('mc-n').value=16;$('mc-sigma').value=2;$('mc-seed').value=42;$('mc-estimator').value='mean';replayMC();}
  function renderMC(){
    const v=mc.sigma**2/mc.n,exact=mc.estimator==='mean'?v:mc.sigma**2,sd=Math.sqrt(exact),lo=mc.mu-4*sd,hi=mc.mu+4*sd;
    const bins=37,counts=Array(bins).fill(0),dx=(hi-lo)/bins;let tails=0;
    mc.values.forEach(x=>{let k=Math.floor((x-lo)/dx);if(k<0||k>=bins)tails++;else counts[k]++;});
    const hist=counts.map(c=>c/(mc.values.length*dx));
    const ymax=Math.max(...hist,1/Math.sqrt(2*Math.PI*v),1/Math.sqrt(2*Math.PI*exact))*1.14;
    const P=plot(755,265,lo,hi,ymax);let b=P.out;
    hist.forEach((y,i)=>b+=`<rect x="${P.X(lo+i*dx)+.4}" y="${P.Y(y)}" width="${Math.max(0,P.X(lo+dx)-P.X(lo)-.8)}" height="${P.b-P.Y(y)}" fill="${C.blue}" opacity=".56" rx="1"/>`);
    b+=density(P,mc.mu,Math.sqrt(v),C.teal,{width:3})+density(P,mc.mu,sd,C.rust,{dash:'6 4',width:2.2});
    b+=line(P.X(mc.mu),P.t,P.X(mc.mu),P.b,C.ink,1,'3 4')+txt(P.X(mc.mu)+8,P.t+13,'True μ = 1',{size:12,color:C.ink});
    $('mc-chart').innerHTML=svg(755,265,b);
    const st=M.statistics(mc.values,mc.mu);
    $('mc-emp').textContent=fmt(st.variance,4);$('mc-exact').textContent=fmt(exact,4);$('mc-crb').textContent=fmt(v,4);
    $('mc-n-out').textContent=mc.n;$('mc-sigma-out').textContent=fmt(mc.sigma,1);
    $('mc-status').textContent=`${st.count.toLocaleString()} trials · seed ${mc.seed} · empirical bias ${fmt(st.bias,4)} · ${tails} estimates outside plot. Each trial draws ${mc.n} new measurements.`;
    $('mc-ratio').textContent=`Exact efficiency: ${fmt(100*v/exact,mc.estimator==='mean'?0:1)}%. SD floor: ${fmt(Math.sqrt(v),3)}. ${mc.estimator==='mean'?'The sample mean attains the bound.':'Unused measurements do not reduce this estimator’s variance.'}`;
    $('mc-one').disabled=$('mc-more').disabled=mc.values.length>=20000;
  }
  ['mc-n','mc-sigma'].forEach(id=>$(id).addEventListener('input',()=>{mc.n=+$('mc-n').value;mc.sigma=+$('mc-sigma').value;replayMC();}));
  $('mc-estimator').addEventListener('change',()=>{mc.estimator=$('mc-estimator').value;replayMC();});
  $('mc-seed').addEventListener('change',()=>{mc.seed=Math.min(999999,Math.max(1,Math.trunc(+$('mc-seed').value)||42));$('mc-seed').value=mc.seed;replayMC();});
  $('mc-one').addEventListener('click',()=>simulate(1));$('mc-more').addEventListener('click',()=>simulate(1000));$('mc-reset').addEventListener('click',resetMC);
  // ----- Lab 2: analytic bias/variance trade-off; alpha=0 is a point mass -----
  const bias={alpha:.55,mu:.35};
  function renderBias(){
    const d=M.shrinkage(bias.alpha,bias.mu,2,16),sd=.5*bias.alpha;
    const ymax=(sd>0?Math.max(pdf(d.expectation,d.expectation,sd),pdf(bias.mu,bias.mu,.5)):pdf(bias.mu,bias.mu,.5))*1.13;
    const P=plot(755,224,-4.6,4.6,ymax);let b=P.out+density(P,bias.mu,.5,C.teal,{fill:true});
    if(sd>0)b+=density(P,d.expectation,sd,C.rust,{fill:true});
    else {const x=P.X(0);b+=line(x,P.b,x,P.t+8,C.rust,3)+`<path d="M${x},${P.t+8} l-5,10 h10 Z" fill="${C.rust}"/>`;b+=txt(x+12,P.t+17,'Point mass at 0',{color:C.rust,size:13});}
    b+=line(P.X(bias.mu),P.t,P.X(bias.mu),P.b,C.ink,1.3,'4 4')+txt(Math.min(P.r-40,P.X(bias.mu)+7),P.t+13,'True μ',{color:C.ink,size:12});
    $('bias-chart').innerHTML=svg(755,224,b);
    const W=755,H=108,l=115,r=731,max=Math.max(.25,d.mse)*1.3,X=x=>l+x/max*(r-l);let bars='';
    bars+=txt(10,49,'Total MSE',{size:13,color:C.ink});
    bars+=`<rect x="${l}" y="28" width="${X(d.variance)-l}" height="29" fill="${C.blue}"/>`;
    bars+=`<rect x="${X(d.variance)}" y="28" width="${X(d.mse)-X(d.variance)}" height="29" fill="${C.rust}"/>`;
    bars+=line(X(.25),20,X(.25),74,C.teal,2,'4 3')+txt(X(.25),13,'Unbiased CRB 0.25',{size:12,anchor:'middle',color:C.teal});
    bars+=line(l,74,r,74)+txt(l,90,'0',{size:11})+txt(r,90,fmt(max,2),{size:11,anchor:'end'});
    bars+=txt(115,107,'Blue: variance',{color:C.blue,size:12})+txt(261,107,'Rust: squared bias',{color:C.rust,size:12});
    $('bias-bars').innerHTML=svg(W,H,bars);
    $('bias-var').textContent=fmt(d.variance,4);$('bias-sq').textContent=fmt(d.bias*d.bias,4);$('bias-mse').textContent=fmt(d.mse,4);
    $('bias-alpha-out').textContent=fmt(bias.alpha,2);$('bias-mu-out').textContent=fmt(bias.mu,2);
    const equal=Math.abs(d.mse-.25)<1e-10;
    $('bias-verdict').textContent=equal?'MSE equals the unbiased CRB.':d.mse<.25?'MSE is below 0.25 here. Bias changes which bound applies.':'Bias now outweighs the variance reduction: MSE exceeds 0.25.';
    $('bias-verdict').classList.toggle('warn',d.mse>.25+1e-10);
    $('bias-bound-note').textContent=`Biased variance bound = α² / Iₙ = ${fmt(d.biasedVarianceBound,4)}. This estimator attains that bound, too.`;
  }
  function setBias(alpha,mu){bias.alpha=alpha;bias.mu=mu;$('bias-alpha').value=alpha;$('bias-mu').value=mu;renderBias();}
  ['bias-alpha','bias-mu'].forEach(id=>$(id).addEventListener('input',()=>{bias.alpha=+$('bias-alpha').value;bias.mu=+$('bias-mu').value;renderBias();}));
  document.querySelectorAll('[data-bias-preset]').forEach(btn=>btn.addEventListener('click',()=>{const v=btn.dataset.biasPreset;if(v==='near')setBias(.55,.35);if(v==='far')setBias(.55,2);if(v==='unbiased')setBias(1,bias.mu);if(v==='constant')setBias(0,bias.mu);}));
  $('bias-reset').addEventListener('click',()=>setBias(.55,.35));
  // ----- Lab 3: Fisher geometry and exact Schur elimination of a common offset -----
  const presets={surround:[[-4,-3],[4,-3],[4,3],[-4,3]],cluster:[[-4,-1.3],[-4.8,-.3],[-4.4,.65],[-3.9,1.3]],collinear:[[-5,0],[-2.5,0],[2.5,0],[5,0]]};
  const geom={anchors:presets.surround.map(p=>p.slice()),target:[0,0],sigma:.6,unknownBias:false,selected:4,result:null};
  const gx=x=>380+40*x,gy=y=>180-40*y;
  function selectedPoint(){return geom.selected===4?geom.target:geom.anchors[geom.selected];}
  function syncGeomInputs(){const p=selectedPoint();$('geom-selected').value=geom.selected;$('geom-x').value=p[0];$('geom-y').value=p[1];$('geom-x-out').textContent=fmt(p[0],2);$('geom-y-out').textContent=fmt(p[1],2);$('geom-sigma-out').textContent=fmt(geom.sigma,2);}
  function renderGeom(){
    const f=M.rangeFisher(geom.anchors,geom.target,geom.sigma,geom.unknownBias);geom.result=f;
    const W=760,H=365;let b=`<defs><clipPath id="geometry-clip"><rect x="40" y="20" width="680" height="320"/></clipPath></defs>`;
    b+=`<rect x="40" y="20" width="680" height="320" rx="8" fill="${C.paper}"/>`;
    for(let x=-8;x<=8;x+=2){b+=line(gx(x),20,gx(x),340,C.line);b+=txt(gx(x),356,x,{size:11,anchor:'middle'});}
    for(let y=-4;y<=4;y+=2){b+=line(40,gy(y),720,gy(y),C.line);b+=txt(30,gy(y)+4,y,{size:11,anchor:'end'});}
    b+=txt(733,357,'x (m)',{size:11,anchor:'end'})+txt(42,13,'y (m)',{size:11});
    b+='<g clip-path="url(#geometry-clip)">';
    geom.anchors.forEach(a=>b+=line(gx(a[0]),gy(a[1]),gx(geom.target[0]),gy(geom.target[1]),C.blue,1.3,'5 5'));
    let outside=false;
    if(!f.invalid&&f.rank===2){
      const r1=1/Math.sqrt(f.hi),r2=1/Math.sqrt(f.lo),c=Math.cos(f.angle),s=Math.sin(f.angle);
      const points=Array.from({length:361},(_,i)=>{let t=i*Math.PI/180;let x=geom.target[0]+r1*Math.cos(t)*c-r2*Math.sin(t)*s,y=geom.target[1]+r1*Math.cos(t)*s+r2*Math.sin(t)*c;if(Math.abs(x)>8.5||Math.abs(y)>4)outside=true;return[gx(x),gy(y)];});
      b+=path(points,C.teal,2.5,C.teal+'25');
    }else if(!f.invalid){
      const c=Math.cos(f.angle+Math.PI/2),s=Math.sin(f.angle+Math.PI/2),p=geom.target;
      b+=line(gx(p[0]-30*c),gy(p[1]-30*s),gx(p[0]+30*c),gy(p[1]+30*s),C.rust,2.2,'8 5');
    }
    b+='</g>';
    [...geom.anchors,geom.target].forEach((p,i)=>{
      const x=gx(p[0]),y=gy(p[1]),col=i===4?C.rust:C.blue,name=i===4?'P':'A'+(i+1);
      b+=`<g data-point="${i}" tabindex="0" role="button" aria-label="Move ${i===4?'target P':'anchor '+name}; arrow keys adjust coordinates" style="cursor:grab">`;
      b+=circle(x,y,20,'transparent');if(geom.selected===i)b+=circle(x,y,15,'none',col,1.5);
      if(i===4)b+=`<path d="M${x},${y-9} L${x+9},${y} L${x},${y+9} L${x-9},${y} Z" fill="${col}"/>`;
      else b+=`<rect x="${x-7}" y="${y-7}" width="14" height="14" rx="3" fill="${col}"/>`;
      b+=txt(x+18,y-13,name,{size:15,color:col,weight:600})+'</g>';
    });
    $('geometry-chart').innerHTML=svg(W,H,b).replace('aria-hidden="true"','role="group" aria-label="Draggable range geometry"');
    const label=geom.unknownBias?'Jₑ':'J';
    $('geom-matrix').textContent=f.invalid?'Derivative unavailable':`${label} [m⁻²] = [ ${fmt(f.a,2).padStart(6)}  ${fmt(f.b,2).padStart(6)} ]\n           [ ${fmt(f.b,2).padStart(6)}  ${fmt(f.c,2).padStart(6)} ]`;
    $('geom-status').textContent=f.invalid?f.reason:f.rank<2?`Rank ${f.rank}/2 · no finite ordinary 2D CRB.`:`PEB = ${fmt(f.peb,3)} m · condition ${fmt(f.condition,1)}`;
    $('geom-status').classList.toggle('warn',f.invalid||f.rank<2||f.condition>1000);
    $('geom-eigen').textContent=f.invalid?'Move the target or anchor apart.':`Information eigenvalues: ${fmt(f.hi,3)}, ${fmt(f.lo,3)} m⁻². ${geom.unknownBias?'Unknown offset eliminated by Schur complement.':'Common offset is known.'}`;
    $('geometry-caption').textContent=f.invalid?'The range derivative is undefined at coincidence; this demo flags distances below 0.08 m.':f.rank<2?'Dashed direction: locally unobserved. No pseudoinverse is presented as a finite full-state bound.':outside?'Unit-Mahalanobis CRB ellipse extends beyond the view (clipped). Not a confidence region.':'Unit-Mahalanobis CRB ellipse; not a confidence region. Geometry is evaluated at the true target.';
    syncGeomInputs();
  }
  function geomPreset(name){geom.anchors=presets[name].map(p=>p.slice());geom.target=[0,0];geom.selected=4;renderGeom();}
  document.querySelectorAll('[data-geom-preset]').forEach(b=>b.addEventListener('click',()=>geomPreset(b.dataset.geomPreset)));
  $('geom-sigma').addEventListener('input',()=>{geom.sigma=+$('geom-sigma').value;renderGeom();});
  $('geom-bias').addEventListener('change',()=>{geom.unknownBias=$('geom-bias').checked;renderGeom();});
  $('geom-selected').addEventListener('change',()=>{geom.selected=+$('geom-selected').value;renderGeom();});
  ['geom-x','geom-y'].forEach((id,k)=>$(id).addEventListener('input',()=>{selectedPoint()[k]=+$(id).value;renderGeom();}));
  $('geom-reset').addEventListener('click',()=>{geom.sigma=.6;geom.unknownBias=false;$('geom-sigma').value=.6;$('geom-bias').checked=false;geomPreset('surround');});
  let dragging=false;
  $('geometry-chart').addEventListener('pointerdown',e=>{const hit=e.target.closest('[data-point]');if(!hit)return;geom.selected=+hit.dataset.point;dragging=true;$('geometry-chart').setPointerCapture(e.pointerId);e.preventDefault();renderGeom();});
  $('geometry-chart').addEventListener('pointermove',e=>{if(!dragging)return;const plotSVG=$('geometry-chart').querySelector('svg');const matrix=plotSVG.getScreenCTM();if(!matrix)return;const local=new DOMPoint(e.clientX,e.clientY).matrixTransform(matrix.inverse());const x=local.x,y=local.y;const p=selectedPoint();p[0]=Math.max(-5.5,Math.min(5.5,(x-380)/40));p[1]=Math.max(-3.5,Math.min(3.5,(180-y)/40));renderGeom();});
  ['pointerup','pointercancel','lostpointercapture'].forEach(ev=>$('geometry-chart').addEventListener(ev,()=>dragging=false));
  $('geometry-chart').addEventListener('keydown',e=>{const hit=e.target.closest('[data-point]');if(!hit||!e.key.startsWith('Arrow'))return;geom.selected=+hit.dataset.point;const p=selectedPoint();const d=e.shiftKey?.5:.1;if(e.key==='ArrowLeft')p[0]-=d;if(e.key==='ArrowRight')p[0]+=d;if(e.key==='ArrowUp')p[1]+=d;if(e.key==='ArrowDown')p[1]-=d;p[0]=Math.max(-5.5,Math.min(5.5,p[0]));p[1]=Math.max(-3.5,Math.min(3.5,p[1]));e.preventDefault();e.stopPropagation();renderGeom();$('geometry-chart').querySelector(`[data-point="${geom.selected}"]`).focus();});
  const state=()=>({mc:{n:mc.n,sigma:mc.sigma,seed:mc.seed,estimator:mc.estimator,...M.statistics(mc.values,mc.mu)},bias:{...bias,...M.shrinkage(bias.alpha,bias.mu,2,16)},geometry:{anchors:geom.anchors.map(p=>p.slice()),target:geom.target.slice(),sigma:geom.sigma,unknownBias:geom.unknownBias,...geom.result}});
  // Compact labs use only the three experiment panels, without the full guide.
  if(document.body.dataset.mode==='lab'){
    const params=new URLSearchParams(location.search);
    const inline=params.get('embed')==='slide';
    const embedded=params.get('embed')==='1'||inline;
    document.body.classList.toggle('embedded',embedded);
    document.body.classList.toggle('inline-slide',inline);
    const names=['gaussian-lab','bias-lab','geometry-lab'];
    let active=names.includes(params.get('lab'))?params.get('lab'):'gaussian-lab';
    function show(name){
      active=names.includes(name)?name:'gaussian-lab';
      document.querySelectorAll('[data-lab-panel]').forEach(el=>{el.hidden=el.dataset.labPanel!==active;});
      document.querySelectorAll('[data-lab]').forEach(el=>el.setAttribute('aria-pressed',String(el.dataset.lab===active)));
    }
    document.querySelectorAll('[data-lab]').forEach(button=>button.addEventListener('click',()=>show(button.dataset.lab)));
    replayMC();renderBias();renderGeom();show(active);
    window.CRBDeck={go:show,state:()=>({current:active,...state()})};
    addEventListener('keydown',event=>{
      if(!embedded)return;
      const send=data=>parent.postMessage(data,location.protocol==='file:'?'*':location.origin);
      if(event.key==='Escape'){event.preventDefault();send({type:inline?'crb-overview':'crb-back'});}
      if(!inline||event.target.closest('input,select,textarea,button,[data-point]')||event.ctrlKey||event.metaKey||event.altKey)return;
      const direction=['ArrowLeft','PageUp'].includes(event.key)?-1:['ArrowRight','PageDown'].includes(event.key)?1:0;
      if(direction){event.preventDefault();send({type:'crb-nav',direction});}
    });
    parent.postMessage({type:'crb-ready'},location.protocol==='file:'?'*':location.origin);
    return;
  }
  // ----- Full interactive guide -----
  const slides=Array.from(document.querySelectorAll('.slide'));
  const meta=JSON.parse($('slide-meta').textContent);
  let current=0,reading=document.body.dataset.mode==='guide'||matchMedia('(max-width:700px)').matches;
  document.body.classList.toggle('reading',reading);
  function fit(){if(reading)return;const r=document.querySelector('.stage-wrap').getBoundingClientRect();const scale=Math.min((r.width-30)/1280,(r.height-20)/720,1.35);$('stage').style.transform=`scale(${Math.max(.1,scale)})`;}
  function go(i,updateHash=true){
    if(typeof i==='string')i=slides.findIndex(s=>s.dataset.slideId===i);
    current=Math.max(0,Math.min(slides.length-1,Number.isFinite(i)?i:0));
    slides.forEach((s,j)=>{s.classList.toggle('active',j===current);s.setAttribute('aria-hidden',reading?'false':String(j!==current));});
    $('counter').textContent=`${String(current+1).padStart(2,'0')} / ${slides.length}`;
    $('progress').style.width=(current+1)/slides.length*100+'%';$('prev').disabled=current===0;$('next').disabled=current===slides.length-1;
    $('notes-title').textContent=`${current+1}. ${meta[current].title}`;$('notes-copy').textContent=meta[current].notes;
    document.querySelectorAll('[data-toc]').forEach(b=>b.classList.toggle('current',+b.dataset.toc===current));
    if(updateHash){history.replaceState(null,'','#'+slides[current].dataset.slideId);if(reading)slides[current].scrollIntoView({behavior:'auto',block:'start'});}
    document.title=`${meta[current].title} · Cramér–Rao Bound | Bai Liping`;fit();
  }
  function route(){let h;try{h=decodeURIComponent(location.hash.replace(/^#\/?/,''));}catch{h='';}if(/^\d+$/.test(h))go(+h,false);else go(h||'overview',false);}
  function toggleRead(){reading=!reading;document.body.classList.toggle('reading',reading);$('read-mode').textContent=reading?'Slide view':'Read view';go(current,false);fit();if(reading)slides[current].scrollIntoView({block:'start'});else window.scrollTo(0,0);}
  function toggleNotes(){$('notes').classList.toggle('open');$('notes-button').setAttribute('aria-expanded',String($('notes').classList.contains('open')));}
  async function fullscreen(){try{if(document.fullscreenElement)await document.exitFullscreen();else if(document.documentElement.requestFullscreen)await document.documentElement.requestFullscreen();}catch{$('fullscreen').textContent='Fullscreen unavailable';}}
  $('toc-grid').innerHTML=meta.map((s,i)=>`<button data-toc="${i}"><span class="toc-num">${String(i+1).padStart(2,'0')}</span><span>${esc(s.title)}</span></button>`).join('');
  $('toc-grid').addEventListener('click',e=>{const b=e.target.closest('[data-toc]');if(b){$('overview-dialog').close();go(+b.dataset.toc);}});
  document.addEventListener('click',e=>{const b=e.target.closest('[data-go]');if(b)go(b.dataset.go);});
  $('prev').addEventListener('click',()=>go(current-1));$('next').addEventListener('click',()=>go(current+1));
  $('outline').addEventListener('click',()=>$('overview-dialog').showModal());$('toc-close').addEventListener('click',()=>$('overview-dialog').close());
  $('notes-button').addEventListener('click',toggleNotes);$('notes-close').addEventListener('click',toggleNotes);$('read-mode').addEventListener('click',toggleRead);$('fullscreen').addEventListener('click',fullscreen);
  $('print-deck').addEventListener('click',()=>window.print());
  document.addEventListener('keydown',e=>{
    if(e.target.closest('input,select,textarea,[data-point]')||$('overview-dialog').open)return;
    if(['ArrowRight','PageDown'].includes(e.key)){e.preventDefault();go(current+1);}
    if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();go(current-1);}
    if(e.key==='Home'){e.preventDefault();go(0);}if(e.key==='End'){e.preventDefault();go(slides.length-1);}
    if(e.key.toLowerCase()==='n')toggleNotes();if(e.key.toLowerCase()==='o')$('overview-dialog').showModal();if(e.key.toLowerCase()==='f')fullscreen();
    if(e.key==='Escape'){$('notes').classList.remove('open');$('notes-button').setAttribute('aria-expanded','false');}
  });
  addEventListener('resize',fit);addEventListener('hashchange',route);addEventListener('fullscreenchange',fit);
  staticCharts();replayMC();renderBias();renderGeom();route();$('read-mode').textContent=reading?'Slide view':'Read view';
  window.CRBDeck={go,meta,renderAll:()=>{staticCharts();renderMC();renderBias();renderGeom();},state:()=>({current,mc:{n:mc.n,sigma:mc.sigma,seed:mc.seed,estimator:mc.estimator,...M.statistics(mc.values,mc.mu)},bias:{...bias,...M.shrinkage(bias.alpha,bias.mu,2,16)},geometry:{anchors:geom.anchors.map(p=>p.slice()),target:geom.target.slice(),sigma:geom.sigma,unknownBias:geom.unknownBias,...geom.result}})};
})();
