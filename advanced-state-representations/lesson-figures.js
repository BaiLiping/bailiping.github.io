/* The same deterministic SVG renderer supplies live views AND print fallbacks. */
(function(root,factory){const api=factory(typeof module==='object'&&module.exports?require('./lesson-model.js'):root.ASRMath);if(typeof module==='object'&&module.exports)module.exports=api;else root.ASRFigures=api;})(typeof globalThis!=='undefined'?globalThis:this,function(M){
'use strict';
const C={ink:'#182D33',muted:'#60747A',rule:'#D6DEDC',teal:'#16736E',coral:'#C2573F',blue:'#3F6F91',amber:'#A87820',paper:'#FFFDFA',soft:'#E4F0ED'};
const rad=Math.PI/180;
const defaults={tangent:{theta:35,delta:.7},manifold:{delta:12,n:8},optimize:{angle:-55,steps:0},adjoint:{theta:45,tx:1,delta:35},spline:{degree:3,t:2.6,selected:4,shift:.7},gp:{kind:'cv',q:.3,sigma:.2,t:2.4,matrix:'info'},pose:{angle:120,u:.5}};
const fmt=(n,d=3)=>Math.abs(n)<1e-10?'0':Math.abs(n)>9999||Math.abs(n)<.0001?n.toExponential(2):n.toFixed(d);
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
const text=(x,y,s,c=C.ink,size=14,anchor='start')=>`<text x="${x}" y="${y}" fill="${c}" font-size="${size}" text-anchor="${anchor}">${esc(s)}</text>`;
const line=(x,y,X,Y,c=C.rule,w=1,dash='')=>`<line x1="${x}" y1="${y}" x2="${X}" y2="${Y}" stroke="${c}" stroke-width="${w}" ${dash?`stroke-dasharray="${dash}"`:''}/>`;
const dot=(x,y,c=C.teal,r=5)=>`<circle cx="${x}" cy="${y}" r="${r}" fill="${c}" stroke="white" stroke-width="1.5"/>`;
const path=(pts,c=C.teal,w=3,dash='')=>`<polyline points="${pts.map(p=>p.join(',')).join(' ')}" fill="none" stroke="${c}" stroke-width="${w}" stroke-linejoin="round" ${dash?`stroke-dasharray="${dash}"`:''}/>`;
function arrow(a,b,c=C.teal,w=3){const d=Math.atan2(b[1]-a[1],b[0]-a[0]),s=8;return line(...a,...b,c,w)+`<path d="M${b} L${[b[0]-s*Math.cos(d-.45),b[1]-s*Math.sin(d-.45)]} L${[b[0]-s*Math.cos(d+.45),b[1]-s*Math.sin(d+.45)]} Z" fill="${c}"/>`;}
function plot(x,y,w,h,x0,x1,y0,y1,equal=false){if(equal){const sy=h/(y1-y0),sx=w/(x1-x0),s=Math.min(sx,sy);x+=(w-s*(x1-x0))/2;y+=(h-s*(y1-y0))/2;w=s*(x1-x0);h=s*(y1-y0);}const P=p=>[x+(p[0]-x0)*w/(x1-x0),y+h-(p[1]-y0)*h/(y1-y0)];return {P,x,y,w,h,x0,x1,y0,y1};}
function axes(a,xname='x (m)',yname='y (m)',ticks=4){let out='';for(let k=0;k<=ticks;k++){const x=a.x0+(a.x1-a.x0)*k/ticks,y=a.y0+(a.y1-a.y0)*k/ticks,[X]=a.P([x,0]),[,Y]=a.P([0,y]);out+=line(X,a.y,X,a.y+a.h)+line(a.x,Y,a.x+a.w,Y)+text(X,a.y+a.h+17,fmt(x,1),C.muted,11,'middle')+text(a.x-8,Y+4,fmt(y,1),C.muted,11,'end');}return out+text(a.x+a.w/2,a.y+a.h+34,xname,C.muted,12,'middle')+text(a.x,a.y-8,yname,C.muted,12);}
function frame(a,T,c=C.teal,length=.5,dash=''){const p=[T[0][2],T[1][2]],o=a.P(p),x=a.P(M.apply2(T,[length,0])),y=a.P(M.apply2(T,[0,length]));return dot(...o,c,3)+arrow(o,x,c,3)+line(...o,...y,c,3,dash)+dot(...y,c,2);}
const samples=(n,f)=>Array.from({length:n+1},(_,i)=>f(i/n));
function wrap(body,title){return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 360" role="img" aria-label="${esc(title)}" style="font-family:Arial,sans-serif;background:${C.paper}"><title>${esc(title)}</title>${body}</svg>`;}
const gpCache=new Map();
function render(demo,state={}){if(!defaults[demo])throw Error('Unknown demonstration');const s={...defaults[demo],...state};let b='',metrics=[],caption='',title='';
if(demo==='tangent'){
 const r=M.tangent(s.theta*rad,s.delta),a=plot(85,40,630,265,-1.9,1.9,-1.65,1.65,true),P=a.P;
 b=axes(a,'first column: x','first column: y')+path(samples(120,u=>P([Math.cos(2*Math.PI*u),Math.sin(2*Math.PI*u)])),C.blue,3);
 const ends=[-1.5,1.5].map(t=>P(r.p.map((v,i)=>v+t*r.v[i])));b+=line(...ends[0],...ends[1],C.coral,2,'6 4');
 b+=arrow(P(r.p),P(r.linear),C.coral)+path(samples(50,u=>P([Math.cos(s.theta*rad+u*s.delta),Math.sin(s.theta*rad+u*s.delta)])),C.teal,5);
 for(const [p,l,c,dy] of [[r.p,'current',C.ink,17],[r.linear,'linear step',C.coral,-10],[r.exact,'Exp step',C.teal,-10]]){const q=P(p);b+=dot(...q,c)+text(q[0]+8,q[1]+dy,l,c,13);}
 b+=text(15,18,'BLUE: valid rotations    DASHED: tangent through the current point',C.muted,13);
 metrics=[['tangent · radius',fmt(r.dot)],['linear norm',fmt(Math.hypot(...r.linear))],['Exp norm',fmt(Math.hypot(...r.exact))]];
 caption='Change the increment. The straight tangent step leaves the circle; the exponential follows the circle. The diagram is SO(2), not a picture of SO(3).';title='Exact unit circle, tangent and retraction';
}else if(demo==='manifold'){
 const r=M.rotation(s.delta*rad,s.n),lim=Math.max(1.4,r.radius*1.15),a=plot(60,45,475,265,-lim,lim,-lim,lim,true),P=a.P;
 b=axes(a,'first column: x','first column: y')+path(samples(120,u=>P([Math.cos(u*2*Math.PI),Math.sin(u*2*Math.PI)])),C.blue,1.5,'5 3')+path(r.path.map(q=>P(q.naive)),C.coral,2)+path(r.path.map(q=>P(q.exact)),C.teal,3);
 for(const [A,c] of [[r.naive,C.coral],[r.exact,C.teal]])for(let j=0;j<2;j++)b+=arrow(P([0,0]),P(A.map(row=>row[j])),c,3);
 b+=text(20,18,'TEAL: exact composition    CORAL: first-order matrix accumulation',C.muted,13)+text(535,92,'This is approximation error,',C.ink,16)+text(535,117,'not just floating-point drift.',C.ink,16)+text(535,172,'The shortcut both stretches',C.muted,15)+text(535,197,'and accumulates angle bias.',C.muted,15);
 metrics=[['exact determinant',fmt(M.valid2(r.exact).det)],['shortcut determinant',fmt(M.valid2(r.naive).det)],['shortcut orthogonality error',fmt(M.valid2(r.naive).error)],['shortcut angle bias (deg)',fmt((r.naiveAngle-r.exactAngle)/rad)]];
 caption='Use smaller steps or more updates. Each shortcut multiplies length by sqrt(1 + delta²); exact group composition preserves a unit orthonormal frame.';title='Repeated rotations and loss of group constraints';
}else if(demo==='adjoint'){
 const r=M.sides(s.theta*rad,s.tx,s.delta*rad),a=plot(65,45,490,265,-2,4,-1.8,2.2,true),P=a.P;
 b=axes(a)+frame(a,M.pose(),C.blue,.7)+frame(a,r.T,C.muted,.65)+frame(a,r.wrong,C.coral,.7)+frame(a,r.left,C.amber,.7,'3 2')+frame(a,r.right,C.teal,.7);
 b+=text(10,18,'GRAY: start   TEAL / AMBER: same pose   CORAL: wrong side, unchanged coordinates',C.muted,12)+text(545,98,'Right increment (body)',C.teal,16)+text(545,124,r.xi.map(v=>fmt(v,2)).join(', '),C.ink,16)+text(545,180,'Left increment (world)',C.amber,16)+text(545,206,r.eta.map(v=>fmt(v,2)).join(', '),C.ink,16);
 metrics=[['correct side-switch error',fmt(r.error)],['unchanged-vector error',fmt(r.wrongError)]];
 caption='Change the starting orientation or translation. A side switch requires the adjoint. Position enters the translational coordinates whenever the increment also rotates.';title='Actual body and world perturbations with matching handedness';
}else if(demo==='optimize'){
 const r=M.fit(s.angle*rad,s.steps),a=plot(60,48,490,265,-2,2.8,-1.9,2.3,true),P=a.P;
 b=axes(a);r.estimated.forEach((p,i)=>{const q=P(p),z=P(r.measurements[i]);b+=line(...q,...z,C.muted,1,'4 3')+dot(...q,C.teal,6)+line(z[0]-6,z[1],z[0]+6,z[1],C.coral,3)+line(z[0],z[1]-6,z[0],z[1]+6,C.coral,3)+text(z[0]+8,z[1]-8,String(i+1),C.coral,12);});b+=frame(a,r.T,C.teal,.45);
 const hist=M.fit(s.angle*rad,8).history,max=Math.max(hist[0].cost,.01),h=plot(590,80,170,195,0,8,0,max);b+=axes(h,'iteration','cost',2)+path(hist.map((v,i)=>h.P([i,v.cost])),C.muted,2)+dot(...h.P([s.steps,r.cost]),C.teal,6)+text(10,18,'TEAL: transformed points    CORAL +: fixed measurements    DASHED: residuals',C.muted,13);
 metrics=[['iteration',s.steps],['cost = half squared residual',fmt(r.cost)],['det R',fmt(M.valid2(r.T).det)],['orthogonality error',fmt(M.valid2(r.T).error)]];
 caption='Step through the solve. Correspondences are fixed; this is pose fitting, not a full SLAM or data-association system. A right tangent update keeps every iterate valid.';title='Actual Gauss–Newton pose fitting';
}else if(demo==='spline'){
 const r=M.spline(s.degree,s.t,s.selected,s.shift),a=plot(60,37,710,124,0,6,-1.4,2.4),q=plot(60,219,710,96,0,6,0,1),P=a.P;
 const x0=P([r.support[0],0])[0],x1=P([r.support[1],0])[0];b=`<rect x="${x0}" y="${a.y}" width="${x1-x0}" height="${a.h}" fill="${C.soft}"/>`+axes(a,'','p(t) (m)',3);
 b+=path(samples(160,u=>P([6*u,M.spline(s.degree,6*u,s.selected,0).value])),C.muted,2,'5 3')+path(samples(160,u=>P([6*u,M.spline(s.degree,6*u,s.selected,s.shift).value])),C.teal,3)+dot(...P([s.t,r.value]),C.coral,5)+axes(q,'time (s)','basis weight',3);
 for(let i=0;i<9;i++)b+=path(samples(180,u=>q.P([6*u,M.basis(9,s.degree,6*u)[i]])),i===s.selected?C.coral:r.active.includes(i)?C.teal:C.rule,i===s.selected?3:1.6);
 const X=P([s.t,0])[0];b+=line(X,32,X,315,C.coral,1.5,'4 3')+text(15,16,'TEAL: edited curve    DASHED: reference    SHADED: moved coefficient support',C.muted,12)+text(62,197,'Below: actual basis functions. Selected basis is coral; active others are teal.',C.muted,12);
 metrics=[['active coefficient indices',r.active.join(', ')],['sum of basis weights',fmt(r.N.reduce((a,b)=>a+b,0))],['p(t) (m)',fmt(r.value)],['dp/dt (m/s)',fmt(r.velocity)]];
 caption='Move one coefficient: only its support interval changes. At most degree + 1 bases are nonzero; fewer may be active at a knot. Endpoint derivatives are one-sided.';title='Exact B-spline basis functions and local control influence';
}else if(demo==='gp'){
 const key=[s.kind,s.q,s.sigma].join('|');if(!gpCache.has(key)){if(gpCache.size>10)gpCache.clear();gpCache.set(key,M.posterior(s.kind,s.q,s.sigma));}const post=gpCache.get(key),r=M.queryGP(post,s.t),grid=samples(120,u=>{const t=6*u,v=M.queryGP(post,t);return {t,...v};}),low=Math.min(-.8,...grid.map(v=>v.mean-1.96*Math.sqrt(v.variance))),high=Math.max(1.8,...grid.map(v=>v.mean+1.96*Math.sqrt(v.variance))),a=plot(55,45,490,265,0,6,low-.1,high+.1),P=a.P;
 b=axes(a,'time (s)','position (m)',3);const upper=grid.map(v=>P([v.t,v.mean+1.96*Math.sqrt(v.variance)])),lower=grid.map(v=>P([v.t,v.mean-1.96*Math.sqrt(v.variance)])).reverse();b+=`<polygon points="${[...upper,...lower].map(p=>p.join(',')).join(' ')}" fill="${C.soft}"/>`+path(grid.map(v=>P([v.t,v.mean])),C.teal,3);M.observations.forEach(o=>{const p=P([o.t,o.z]);b+=dot(...p,C.coral,5);});const p=P([s.t,r.mean]);b+=line(p[0],45,p[0],310,C.amber,2,'4 3')+dot(...p,C.amber,5);
 const A=s.matrix==='info'?post.info:post.cov,n=A.length,size=180/n,max=Math.max(...A.flat().map(Math.abs));for(let i=0;i<n;i++)for(let j=0;j<n;j++){const v=Math.abs(A[i][j]),opacity=v<max*1e-12?0:.22+.78*Math.sqrt(v/max);b+=`<rect x="${592+j*size}" y="${76+i*size}" width="${size-.7}" height="${size-.7}" fill="${C.teal}" opacity="${opacity}"/>`;}
 b+=text(590,55,s.matrix==='info'?'Posterior information':'Posterior covariance',C.ink,14)+text(590,279,s.kind==='cv'?'2 × 2 blocks: position, velocity':'1 × 1 blocks: position',C.muted,12)+text(590,302,'Shade = absolute entry magnitude',C.muted,11)+text(12,17,'SHADED: pointwise 95% latent-position band    CORAL: noisy observations',C.muted,12);
 metrics=[['query mean (m)',fmt(r.mean)],['support variance (m²)',fmt(r.supportVariance,5)],['bridge variance (m²)',fmt(r.bridge,5)],['total variance (m²)',fmt(r.variance,5)]];
 caption='The exact model includes every measurement time in the Markov chain. Querying adds the conditional bridge variance. Switch information/covariance: sparse precision does not mean sparse covariance.';title='Exact continuous-time Gaussian process posterior and uncertainty';
}else if(demo==='pose'){
 const r=M.interpolate(s.angle*rad,s.u),a=plot(65,50,470,260,-1,3,-1,2.5,true),P=a.P;
 b=axes(a)+path(samples(80,u=>{const T=M.interpolate(s.angle*rad,u).group;return P([T[0][2],T[1][2]]);}),C.teal,3)+path([P([0,0]),P([2,1])],C.blue,2,'5 3')+frame(a,r.T0,C.muted,.4)+frame(a,r.T1,C.muted,.4)+frame(a,r.matrix,C.coral,.6)+frame(a,r.split,C.blue,.6)+frame(a,r.group,C.teal,.6)+text(12,18,'TEAL: constant twist    BLUE: valid split model    CORAL: raw matrix blend',C.muted,13)+text(545,100,'Two valid models can follow',C.ink,16)+text(545,125,'different translation paths.',C.ink,16)+text(545,186,'At 180° and halfway, the',C.coral,15)+text(545,211,'matrix-blended frame collapses.',C.coral,15);
 metrics=[['constant-twist det R',fmt(M.valid2(r.group).det)],['split-model det R',fmt(M.valid2(r.split).det)],['matrix-blend det R',fmt(M.valid2(r.matrix).det)],['matrix orthogonality error',fmt(M.valid2(r.matrix).error)]];
 caption='Neither valid model is universally best. Constant-twist SE(2) interpolation is not generally a straight translation; linear blending of rotation-matrix entries is not a valid pose model.';title='Group interpolation versus matrix blending and split interpolation';
}
return {svg:wrap(b,title),metrics,caption,title,state:s};}
return {C,defaults,fmt,render};
});
