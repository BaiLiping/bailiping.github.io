/* Radar SLAM teaching models. All distances are metres and angles radians.
 * Ground truth is used only by the synthetic observation generator and displays.
 * Registration and graph solvers receive measurements, never landmark IDs.
 */
(function(root){
'use strict';
const TAU=2*Math.PI, rad=x=>x*Math.PI/180, deg=x=>x*180/Math.PI;
const wrap=a=>Math.atan2(Math.sin(a),Math.cos(a));
function rng(seed){return()=>{seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return ((t^t>>>14)>>>0)/4294967296;};}
function normal(r){return Math.sqrt(-2*Math.log(Math.max(1e-12,r())))*Math.cos(TAU*r());}
function transform(t,p){const c=Math.cos(t[2]),s=Math.sin(t[2]);return [c*p[0]-s*p[1]+t[0],s*p[0]+c*p[1]+t[1]];}
function inverse(t){const c=Math.cos(t[2]),s=Math.sin(t[2]);return [-c*t[0]-s*t[1],s*t[0]-c*t[1],-t[2]];}
function compose(a,b){return [...transform(a,b),wrap(a[2]+b[2])];}
function relative(a,b){return compose(inverse(a),b);}
function associate(src,dst,t,gate=2,robust=true){
 const pairs=[];
 src.forEach((p,i)=>{const q=transform(t,p);let best=gate*gate,j=-1;dst.forEach((d,k)=>{const dd=(q[0]-d[0])**2+(q[1]-d[1])**2;if(dd<best){best=dd;j=k;}});if(j>=0)pairs.push({i,j,p:q,q:dst[j],d:Math.sqrt(best),w:1});});
 if(robust&&pairs.length){const ds=pairs.map(x=>x.d).sort((a,b)=>a-b),scale=Math.max(.08,1.5*ds[Math.floor(ds.length/2)]);pairs.forEach(p=>p.w=Math.min(1,scale/Math.max(p.d,1e-9)));}
 return pairs;
}
function alignPairs(pairs){
 if(pairs.length<3)return null;
 let sw=0,px=0,py=0,qx=0,qy=0;
 for(const p of pairs){sw+=p.w;px+=p.w*p.p[0];py+=p.w*p.p[1];qx+=p.w*p.q[0];qy+=p.w*p.q[1];}
 px/=sw;py/=sw;qx/=sw;qy/=sw;
 let c=0,s=0,spread=0;for(const p of pairs){const x=p.p[0]-px,y=p.p[1]-py,u=p.q[0]-qx,v=p.q[1]-qy;c+=p.w*(x*u+y*v);s+=p.w*(x*v-y*u);spread+=p.w*(x*x+y*y);}
 if(spread<1e-10)return null;
 const a=Math.atan2(s,c),co=Math.cos(a),si=Math.sin(a);
 return [qx-co*px+si*py,qy-si*px-co*py,a];
}
function icp(src,dst,init=[0,0,0],iterations=30,gate=2,robust=true){
 let t=init.slice(),count=0;
 for(let n=0;n<iterations;n++){const pairs=associate(src,dst,t,gate,robust),d=alignPairs(pairs);if(!d)break;t=compose(d,t);count++;if(Math.hypot(d[0],d[1])+Math.abs(d[2])<1e-6)break;}
 const pairs=associate(src,dst,t,gate,robust),rmse=pairs.length?Math.sqrt(pairs.reduce((s,p)=>s+p.d*p.d,0)/pairs.length):Infinity;
 return {t,pairs,rmse,count};
}
function makeMission(biasDeg=.35){
 const r=rng(429),world=[];
 for(let i=0;i<95;i++)world.push([-13+26*r(),-5+26*r()]);
 const truth=[],scans=[],poses=[[0,0,0]],edges=[],quality=[0],accepted=[];const N=48;
 for(let k=0;k<=N;k++){
  const a=TAU*k/N,t=[8*Math.sin(a),8*(1-Math.cos(a)),wrap(a)];truth.push(t);
  const r=rng(1000+k),scan=[];
  for(const p of world){const q=transform(inverse(t),p),d=Math.hypot(...q);if(d<15&&d>1&&r()>.12){const rr=d+.06*normal(r),aa=Math.atan2(q[1],q[0])+rad(.17)*normal(r);scan.push([rr*Math.cos(aa),rr*Math.sin(aa)]);}}
  for(let j=0;j<3;j++)scan.push([26*r()-13,26*r()-13]);
  scans.push(scan);
  if(k){const last=edges.at(-1),initial=last?last.z:[.8,0,.08];const fit=icp(scan,scans[k-1],initial,40,1.7,true);const z=fit.t.slice();z[2]=wrap(z[2]+rad(biasDeg));edges.push({i:k-1,j:k,z,sigma:[.12,.12,rad(1.8)],loop:false});poses.push(compose(poses.at(-1),z));quality.push(fit.rmse);accepted.push(new Set(fit.pairs.filter(p=>p.d<.45).map(p=>p.i)));}else accepted.push(new Set(scan.map((_,i)=>i)));
 }
 // This teaching example supplies candidate (0,N). It does not implement place retrieval.
 const loop=icp(scans[N],scans[0],[0,0,0],50,2,true);
 return {world,truth,scans,poses,edges,quality,accepted,loop,N,biasDeg};
}
function makeICP(noise=.06){
 const r=rng(77),target=[];
 for(let i=0;i<32;i++){const a=TAU*i/32,rr=5+.7*Math.sin(3*a)+.35*Math.cos(7*a);target.push([rr*Math.cos(a),rr*Math.sin(a)]);}
 for(let i=0;i<13;i++)target.push([-2+4*r(),-2+4*r()]);
 const truth=[1.3,-.7,rad(13)],source=target.map(q=>transform(inverse(truth),[q[0]+noise*normal(r),q[1]+noise*normal(r)]));
 for(let i=0;i<9;i++)source.push([14*r()-7,14*r()-7]);
 return {source,target,truth};
}
function rangeSpectrum(BGHz=.45,separation=.45,hann=true){
 const N=256,B=BGHz*1e9,c=299792458,r1=6,r2=r1+separation,r=rng(121);let re=[],im=[];
 for(let n=0;n<N;n++){const a=TAU*(2*B*r1/c)*n/N,b=TAU*(2*B*r2/c)*n/N+.7;re.push(Math.cos(a)+.8*Math.cos(b)+.07*normal(r));im.push(Math.sin(a)+.8*Math.sin(b)+.07*normal(r));}
 const spectrum=[];let max=1e-20;
 for(let k=0;k<N/2;k++){let a=0,b=0;for(let n=0;n<N;n++){const w=hann?.5-.5*Math.cos(TAU*n/(N-1)):1,co=Math.cos(TAU*k*n/N),si=Math.sin(TAU*k*n/N);a+=w*(re[n]*co+im[n]*si);b+=w*(im[n]*co-re[n]*si);}const p=a*a+b*b;max=Math.max(max,p);spectrum.push([k*c/(2*B),p]);}
 return {re,im,spectrum:spectrum.map(p=>[p[0],Math.max(-70,10*Math.log10(Math.max(1e-20,p[1])/max))]),resolution:c/(2*B),r1,r2};
}
function cfar(exponent=3,edge=false,seed=12){
 const r=rng(seed),n=128,guard=2,train=8,N=2*train,alpha=N*(Math.pow(10**(-exponent),-1/N)-1),power=[];
 for(let i=0;i<n;i++)power.push(-Math.log(Math.max(r(),1e-12))*(edge&&i>=74?7:1));
 const targets=[32,62,103];[24,12,33].forEach((p,i)=>power[targets[i]]+=p);
 const thresholds=Array(n).fill(null),hits=[];
 for(let k=guard+train;k<n-guard-train;k++){let sum=0;for(let j=guard+1;j<=guard+train;j++)sum+=power[k-j]+power[k+j];thresholds[k]=alpha*sum/N;if(power[k]>thresholds[k])hits.push(k);}
 return {power,thresholds,hits,targets,alpha};
}
function fitVelocity(data,indices=null){
 const set=indices||data.map((_,i)=>i);let a=0,b=0,c=0,d=0,e=0;
 for(const i of set){const p=data[i],x=Math.cos(p.a),y=Math.sin(p.a);a+=x*x;b+=x*y;c+=y*y;d-=x*p.d;e-=y*p.d;}
 const det=a*c-b*b,disc=Math.sqrt((a-c)**2+4*b*b),hi=(a+c+disc)/2,lo=(a+c-disc)/2;
 if(det<1e-10)return {v:[NaN,NaN],condition:Infinity};
 return {v:[(c*d-b*e)/det,(a*e-b*d)/det],condition:hi/Math.max(lo,1e-12)};
}
function velocity(vx=4,vy=1,movers=.25,aperture=130,robust=true){
 const r=rng(919),data=[];
 for(let i=0;i<80;i++){const a=rad(aperture)*((i+.4)/80-.5),moving=r()<movers;data.push({a,d:-vx*Math.cos(a)-vy*Math.sin(a)+.09*normal(r)+(moving?2.2+2*r():0),moving});}
 let inliers=data.map((_,i)=>i);
 if(robust){let best=[],err=Infinity;for(let n=0;n<220;n++){const i=Math.floor(r()*80),j=Math.floor(r()*80);if(i===j)continue;const f=fitVelocity(data,[i,j]);if(!Number.isFinite(f.v[0]))continue;const ids=[],res=[];data.forEach((p,k)=>{const e=Math.abs(p.d+f.v[0]*Math.cos(p.a)+f.v[1]*Math.sin(p.a));if(e<.28){ids.push(k);res.push(e);}});const cost=res.reduce((s,e)=>s+e*e,0);if(ids.length>best.length||(ids.length===best.length&&cost<err)){best=ids;err=cost;}}
 if(best.length>=3)inliers=best;
 }
 const fit=fitVelocity(data,inliers);return {data,inliers,...fit,truth:[vx,vy]};
}
function residual(a,b,z){const c=Math.cos(a[2]),s=Math.sin(a[2]),dx=b[0]-a[0],dy=b[1]-a[1];return [c*dx+s*dy-z[0],-s*dx+c*dy-z[1],wrap(b[2]-a[2]-z[2])];}
function graphEdges(m,sigma=.12,falseLoop=false){return [...m.edges,{i:0,j:m.N,z:falseLoop?[4,-3,rad(24)]:m.loop.t.slice(),sigma:[sigma,sigma,rad(1.2)],loop:true}];}
function graphCost(poses,edges,robust=false){let cost=0;for(const e of edges){const r=residual(poses[e.i],poses[e.j],e.z),norm=Math.hypot(...r.map((x,k)=>x/e.sigma[k]));cost+=robust&&e.loop&&norm>3?3*norm-4.5:.5*norm*norm;}return cost;}
function solveLinear(A,b){
 const n=b.length,M=A.map((row,i)=>[...row,b[i]]);
 for(let k=0;k<n;k++){let p=k;for(let i=k+1;i<n;i++)if(Math.abs(M[i][k])>Math.abs(M[p][k]))p=i;if(Math.abs(M[p][k])<1e-14)return null;[M[p],M[k]]=[M[k],M[p]];for(let i=k+1;i<n;i++){const f=M[i][k]/M[k][k];if(!f)continue;for(let j=k;j<=n;j++)M[i][j]-=f*M[k][j];}}
 const x=Array(n).fill(0);for(let i=n-1;i>=0;i--){let v=M[i][n];for(let j=i+1;j<n;j++)v-=M[i][j]*x[j];x[i]=v/M[i][i];}return x;
}
function graphStep(poses,edges,robust=false){
 const n=(poses.length-1)*3,H=Array.from({length:n},()=>Array(n).fill(0)),g=Array(n).fill(0);
 for(const e of edges){const a=poses[e.i],b=poses[e.j],c=Math.cos(a[2]),s=Math.sin(a[2]),dx=b[0]-a[0],dy=b[1]-a[1],r=residual(a,b,e.z),norm=Math.hypot(...r.map((v,k)=>v/e.sigma[k])),w=robust&&e.loop?Math.min(1,3/Math.max(norm,1e-12)):1;
 const A=[[-c,-s,-s*dx+c*dy],[s,-c,-c*dx-s*dy],[0,0,-1]],B=[[c,s,0],[-s,c,0],[0,0,1]],blocks=[[e.i,A],[e.j,B]].filter(x=>x[0]>0);
 for(let row=0;row<3;row++){const weight=w/(e.sigma[row]**2),items=[];for(const [node,J]of blocks)for(let k=0;k<3;k++)items.push([(node-1)*3+k,J[row][k]]);for(const [i,x]of items){g[i]+=weight*x*r[row];for(const [j,y]of items)H[i][j]+=weight*x*y;}}
 }
 const old=graphCost(poses,edges,robust);
 for(const lambda of [1e-5,.001,.1,10,1000]){const A=H.map((row,i)=>row.map((v,j)=>v+(i===j?lambda*Math.max(1,H[i][i]):0))),d=solveLinear(A,g.map(x=>-x));if(!d)continue;const trial=poses.map((p,i)=>i?[p[0]+d[(i-1)*3],p[1]+d[(i-1)*3+1],wrap(p[2]+d[(i-1)*3+2])]:p.slice()),cost=graphCost(trial,edges,robust);if(cost<=old+1e-10)return {poses:trial,cost,old,lambda,accepted:true,step:Math.hypot(...d)};}
 return {poses,cost:old,old,lambda:0,accepted:false,step:0};
}
const api={rad,deg,wrap,rng,normal,transform,inverse,compose,relative,associate,alignPairs,icp,makeMission,makeICP,rangeSpectrum,cfar,velocity,residual,graphEdges,graphCost,graphStep};
if(typeof module!=='undefined')module.exports=api;root.RadarModel=api;
})(typeof window!=='undefined'?window:globalThis);
