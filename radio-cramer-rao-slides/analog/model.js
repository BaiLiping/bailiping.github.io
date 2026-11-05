/* Analog BS/UE beam-sweep CRB. Unit-modulus array responses, unit-norm RF weights.
 * The seven real parameters are apparent delay [ns], four angles [deg], loss [dB], phase [rad].
 * Static isolated path; all beam pairs are separate coherent acquisitions, one OFDM symbol each.
 * This ideal phase-shifter codebook is illustrative, not a measured acquisition codebook.
 */
(function(root,factory){if(typeof module==='object'&&module.exports)module.exports=factory();else root.RadioAnalogCRB=factory();})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const pi=Math.PI,rad=pi/180;
const defaults={nty:24,ntz:16,nry:8,nrz:4,bsAz:10,bsEl:10,ueAz:8,ueEl:4,azMin:-60,azMax:60,elMin:-30,elMax:30,K:3300,BMHz:400,fcGHz:27.2,spacing:.5,powerDbm:30,nfDb:9,lossDb:120,rxAz:-20,rxEl:10,txAz:15,txEl:5,schedule:'full'};
const labels=['Apparent delay','AoA azimuth','AoA elevation','AoD azimuth','AoD elevation','Effective attenuation','Reference phase'];
const units=['ns','deg','deg','deg','deg','dB','rad'];
const mul=(a,b)=>[a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]];
const scale=(a,b)=>[a[0]*b,a[1]*b];
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1];
const zeros=n=>Array.from({length:n},()=>Array(n).fill(0));
function lin(a,b,n){return Array.from({length:n},(_,i)=>n===1?(a+b)/2:a+(b-a)*i/(n-1));}
function grid(na,ne,c){return lin(c.azMin,c.azMax,na).flatMap(a=>lin(c.elMin,c.elMax,ne).map(e=>[a,e]));}
function config(input={}){
 const c={...defaults,...input};
 for(const k of Object.keys(defaults))if(typeof defaults[k]==='number'&&!Number.isFinite(c[k]))throw new Error(k+' must be finite.');
 for(const k of ['nty','ntz','nry','nrz','bsAz','bsEl','ueAz','ueEl','K'])if(!Number.isInteger(c[k])||c[k]<1)throw new Error(k+' must be a positive integer.');
 if(c.nty*c.ntz>2048||c.nry*c.nrz>2048||c.bsAz*c.bsEl>1000||c.ueAz*c.ueEl>1000||c.K>100000)throw new Error('Configuration exceeds the teaching calculator size limit.');
 if(c.BMHz<=0||c.fcGHz<=0||c.spacing<=0)throw new Error('Bandwidth, carrier and spacing must be positive.');
 if(c.azMin>c.azMax||c.elMin>c.elMax||c.elMin<-90||c.elMax>90)throw new Error('Invalid beam coverage.');
 if(!['full','fixed','bs-only','ue-only'].includes(c.schedule))throw new Error('Unknown beam-pair schedule.');
 return c;
}
function steering(ny,nz,az,el,d=.5){
 const a=az*rad,e=el*rad,uy=Math.cos(e)*Math.sin(a),uz=Math.sin(e);
 const ya=Math.cos(e)*Math.cos(a)*rad,ye=-Math.sin(e)*Math.sin(a)*rad,ze=Math.cos(e)*rad;
 const v=[],da=[],de=[];
 for(let y=0;y<ny;y++)for(let z=0;z<nz;z++){
  const ry=2*pi*d*(y-(ny-1)/2),rz=2*pi*d*(z-(nz-1)/2),p=ry*uy+rz*uz;
  const q=[Math.cos(p),Math.sin(p)];v.push(q);da.push(scale([-q[1],q[0]],ry*ya));de.push(scale([-q[1],q[0]],ry*ye+rz*ze));
 }
 return {v,da,de};
}
function response(ny,nz,az,el,beam,d=.5,tx=false){
 const a=steering(ny,nz,az,el,d),b=steering(ny,nz,beam[0],beam[1],d).v,norm=Math.sqrt(ny*nz),out=[];
 for(const vals of [a.v,a.da,a.de]){
  let re=0,im=0;
  for(let i=0;i<vals.length;i++){re+=(b[i][0]*vals[i][0]+b[i][1]*vals[i][1])/norm;im+=(b[i][0]*vals[i][1]-b[i][1]*vals[i][0])/norm;}
  out.push([re,tx?-im:im]);
 }
 return out;
}
function eigenSym(matrix){
 const n=matrix.length,a=matrix.map(r=>r.slice()),v=zeros(n);for(let i=0;i<n;i++)v[i][i]=1;
 for(let step=0;step<200*n*n;step++){
  let p=0,q=1,m=0;for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)if(Math.abs(a[i][j])>m){m=Math.abs(a[i][j]);p=i;q=j;}
  if(m<1e-14)break;
  const angle=.5*Math.atan2(2*a[p][q],a[q][q]-a[p][p]),c=Math.cos(angle),s=Math.sin(angle),ap=a[p][p],aq=a[q][q],b=a[p][q];
  for(let k=0;k<n;k++)if(k!==p&&k!==q){const x=a[k][p],y=a[k][q];a[k][p]=a[p][k]=c*x-s*y;a[k][q]=a[q][k]=s*x+c*y;}
  a[p][p]=c*c*ap-2*s*c*b+s*s*aq;a[q][q]=s*s*ap+2*s*c*b+c*c*aq;a[p][q]=a[q][p]=0;
  for(let k=0;k<n;k++){const x=v[k][p],y=v[k][q];v[k][p]=c*x-s*y;v[k][q]=s*x+c*y;}
 }
 return {values:a.map((r,i)=>r[i]),vectors:v};
}
function marginal(J){
 const n=J.length,scales=J.map((r,i)=>Math.sqrt(Math.max(0,r[i]))),corr=J.map((r,i)=>r.map((x,j)=>scales[i]&&scales[j]?x/scales[i]/scales[j]:0));
 const e=eigenSym(corr),tol=Math.max(1,...e.values)*1e-10,good=e.values.map(x=>x>tol),rank=good.filter(Boolean).length;
 const estimable=scales.map((s,i)=>s>0&&e.values.reduce((t,_,k)=>t+(good[k]?0:e.vectors[i][k]**2),0)<1e-8);
 const covariance=J.map((r,i)=>r.map((_,j)=>estimable[i]&&estimable[j]?e.values.reduce((t,x,k)=>t+(good[k]?e.vectors[i][k]*e.vectors[j][k]/x:0),0)/scales[i]/scales[j]:NaN));
 return {rank,corr,covariance,bounds:scales.map((_,i)=>estimable[i]?Math.sqrt(Math.max(0,covariance[i][i])):Infinity),estimable,eigenvalues:e.values};
}
function evaluate(input={}){
 const c=config(input),bs=grid(c.bsAz,c.bsEl,c),ue=grid(c.ueAz,c.ueEl,c),tr=bs.map(b=>response(c.nty,c.ntz,c.txAz,c.txEl,b,c.spacing,true)),rr=ue.map(b=>response(c.nry,c.nrz,c.rxAz,c.rxEl,b,c.spacing));
 const best=v=>v.reduce((j,x,i)=>dot(x[0],x[0])>dot(v[j][0],v[j][0])?i:j,0),bt=best(tr),br=best(rr);
 let pairs=[];for(let m=0;m<bs.length;m++)for(let n=0;n<ue.length;n++)if(c.schedule==='full'||(c.schedule==='fixed'&&m===bt&&n===br)||(c.schedule==='bs-only'&&n===br)||(c.schedule==='ue-only'&&m===bt))pairs.push([m,n]);
 const B=c.BMHz*1e6,df=B/c.K,mean=c.K%2?0:-df/2,beta2=df*df*(c.K*c.K-1)/12,mom=[c.K,c.K*mean,c.K*(beta2+mean*mean)];
 const P=10**((c.powerDbm-30)/10),noise=10**((-174+c.nfDb-30)/10)*df,weight=10**(-c.lossDb/10)*(P/c.K)/noise;
 if(!Number.isFinite(weight)||weight<=0)throw new Error('The power/noise configuration is outside numeric range.');
 const J=zeros(7);let energy=0;
 for(const [m,n] of pairs){
  const t=tr[m],r=rr[n],g=mul(r[0],t[0]);energy+=dot(g,g);
  const D=[scale([g[1],-g[0]],2*pi*1e-9),mul(r[1],t[0]),mul(r[2],t[0]),mul(r[0],t[1]),mul(r[0],t[2]),scale(g,-Math.log(10)/20),[-g[1],g[0]]];
  for(let i=0;i<7;i++)for(let j=0;j<=i;j++)J[i][j]+=2*weight*dot(D[i],D[j])*mom[(i===0?1:0)+(j===0?1:0)];
 }
 for(let i=0;i<7;i++)for(let j=0;j<i;j++)J[j][i]=J[i][j];
 const result=marginal(J),Gamma=weight*c.K*energy;
 return {...result,J,c,bs,ue,pairs,noise,df,beta:Math.sqrt(beta2),Gamma,snrDb:10*Math.log10(Gamma),symbols:pairs.length,samples:pairs.length*c.K,usefulMs:1000*pairs.length/df,energyJ:P*pairs.length/df,bestPair:[bt,br],labels,units};
}
return {defaults,labels,units,config,grid,steering,response,eigenSym,marginal,evaluate};
});
