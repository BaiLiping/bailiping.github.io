// Original educational reductions of Eqs. (1)–(4) and (21), Wei et al. (2026).
// No author data, pretrained weights, source code, or benchmark claims.
const PI = Math.PI;
export const linspace = (a,b,n) => Array.from({length:n},(_,i)=>a+(b-a)*i/(n-1));
export function rng(seed=7) {
  let s=Number(seed)>>>0;
  return ()=>{s+=0x6D2B79F5;let t=s;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296;};
}
function normal(r){return Math.sqrt(-2*Math.log(Math.max(r(),1e-15)))*Math.cos(2*PI*r());}
const dot = (a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
export function steering(u,m=6){const d=Math.sqrt(m),k=Array.from({length:m},(_,i)=>i);return {re:k.map(k=>Math.cos(PI*k*u)/d),im:k.map(k=>Math.sin(PI*k*u)/d)};}
export function covarianceMatrix(sources,noise,m=6){
  const re=Array.from({length:m},(_,i)=>Array.from({length:m},(_,j)=>i===j?noise:0));
  const im=Array.from({length:m},()=>Array(m).fill(0));
  for(const {u,power} of sources){const a=steering(u,m);for(let i=0;i<m;i++)for(let j=0;j<m;j++){
    re[i][j]+=power*(a.re[i]*a.re[j]+a.im[i]*a.im[j]);
    im[i][j]+=power*(a.im[i]*a.re[j]-a.re[i]*a.im[j]);
  }}return {re,im};
}
function cholesky(matrix){
  const n=matrix.re.length,re=Array.from({length:n},()=>Array(n).fill(0)),im=re.map(row=>row.slice());let logdet=0;
  for(let i=0;i<n;i++)for(let j=0;j<=i;j++){
    let sr=matrix.re[i][j],si=matrix.im[i][j];
    for(let k=0;k<j;k++){sr-=re[i][k]*re[j][k]+im[i][k]*im[j][k];si-=im[i][k]*re[j][k]-re[i][k]*im[j][k];}
    if(i===j){if(!(sr>0))throw new Error('Covariance is not positive definite');re[i][j]=Math.sqrt(sr);logdet+=Math.log(sr);}
    else{re[i][j]=sr/re[j][j];im[i][j]=si/re[j][j];}
  }return{re,im,logdet};
}
export function complexGaussianLogLikelihood(samples,matrix){
  const L=cholesky(matrix),n=matrix.re.length;let ll=-samples.length*(n*Math.log(PI)+L.logdet);
  for(const z of samples){const yr=[],yi=[];for(let i=0;i<n;i++){
    let r=z.re[i],v=z.im[i];for(let j=0;j<i;j++){r-=L.re[i][j]*yr[j]-L.im[i][j]*yi[j];v-=L.re[i][j]*yi[j]+L.im[i][j]*yr[j];}
    yr[i]=r/L.re[i][i];yi[i]=v/L.re[i][i];ll-=yr[i]*yr[i]+yi[i]*yi[i];
  }}return ll;
}
export function normalizeLogWeights(logw){const mx=Math.max(...logw);const w=logw.map(v=>Math.exp(v-mx)),sum=w.reduce((a,b)=>a+b,0);if(!(sum>0)||!Number.isFinite(sum))throw new Error('Invalid posterior weights');return w.map(v=>v/sum);}
function sampleSignal(sources,noise,m,n,random){return Array.from({length:n},()=>{
  const re=Array.from({length:m},()=>Math.sqrt(noise/2)*normal(random)),im=Array.from({length:m},()=>Math.sqrt(noise/2)*normal(random));
  for(const {u,power}of sources){const a=steering(u,m),cr=normal(random)*Math.sqrt(power/2),ci=normal(random)*Math.sqrt(power/2);for(let j=0;j<m;j++){re[j]+=a.re[j]*cr-a.im[j]*ci;im[j]+=a.re[j]*ci+a.im[j]*cr;}}return{re,im};
});}
function projectedPower(z,a){let r=0,i=0;for(let j=0;j<a.re.length;j++){r+=a.re[j]*z.re[j]+a.im[j]*z.im[j];i+=a.re[j]*z.im[j]-a.im[j]*z.re[j];}return r*r+i*i;}
export function covarianceLab({separation=14,snr=8,snapshots=32,seed=7}={}){
  separation=Number(separation);snr=Number(snr);snapshots=Math.max(1,Math.round(snapshots));
  const sensors=6,noise=10**(-snr/10),trueAngles=[-separation/2,separation/2],sources=trueAngles.map(t=>({u:Math.sin(t*PI/180),power:1}));
  const samples=sampleSignal(sources,noise,sensors,snapshots,rng(seed));
  const angles=linspace(-50,50,201),beamRaw=angles.map(t=>samples.reduce((s,z)=>s+projectedPower(z,steering(Math.sin(t*PI/180),sensors)),0)/snapshots);
  const beamMax=Math.max(...beamRaw),beam=beamRaw.map(x=>x/beamMax);
  const logLikelihood=angles.map(t=>complexGaussianLogLikelihood(samples,covarianceMatrix([sources[0],{u:Math.sin(t*PI/180),power:1}],noise,sensors)));
  const posterior=normalizeLogWeights(logLikelihood),mi=posterior.indexOf(Math.max(...posterior));
  const covariance=covarianceMatrix(sources,noise,sensors),sampleRe=Array.from({length:sensors},()=>Array(sensors).fill(0)),sampleIm=sampleRe.map(row=>row.slice());
  for(let i=0;i<sensors;i++)for(let j=0;j<sensors;j++)for(const z of samples){sampleRe[i][j]+=(z.re[i]*z.re[j]+z.im[i]*z.im[j])/snapshots;sampleIm[i][j]+=(z.im[i]*z.re[j]-z.re[i]*z.im[j])/snapshots;}
  const intensityPosterior=Array(angles.length).fill(1/angles.length);
  const mean=dot(angles,posterior),sd=Math.sqrt(dot(angles.map(x=>(x-mean)**2),posterior));
  return{angles,beam,beamRaw,posterior,intensityPosterior,mapAngle:angles[mi],trueAngles,covarianceRe:covariance.re,covarianceIm:covariance.im,sampleRe,sampleIm,samples,noise,sensors,snapshots,seed,snr,separation,logLikelihood,phaseInformation:Math.max(...logLikelihood)-Math.min(...logLikelihood),posteriorPeak:posterior[mi],posteriorMean:mean,posteriorSD:sd};
}

const X=linspace(-.92,.92,81),T=linspace(-5,4.5,49),G=T.map(Math.exp),STEP=1.7/80;
const priorMu=Math.log(2),priorSigma=1.1;
const basis=c=>[1,c,Math.sin(PI*c),Math.cos(2*PI*c)];
const trueMu=x=>Math.log(2)+1.1*Math.sin(PI*x)+.4*Math.cos(2*PI*x);
const trueSigma=.45;
function reflect(x){while(x>.85||x<-.85){if(x>.85)x=1.7-x;if(x<-.85)x=-1.7-x;}return x;}
function normalLog(x,mu,sigma){return-.5*((x-mu)/sigma)**2-Math.log(sigma)-.5*Math.log(2*PI);}
function lognormalPDF(g,mu,sigma){return Math.exp(normalLog(Math.log(g),mu,sigma))/g;}
function solve(A,b){
  const n=b.length,M=A.map((r,i)=>[...r,b[i]]);
  for(let k=0;k<n;k++){let p=k;for(let i=k+1;i<n;i++)if(Math.abs(M[i][k])>Math.abs(M[p][k]))p=i;[M[k],M[p]]=[M[p],M[k]];const d=M[k][k];if(Math.abs(d)<1e-12)throw new Error('Singular regression');for(let j=k;j<=n;j++)M[k][j]/=d;for(let i=0;i<n;i++)if(i!==k){const f=M[i][k];for(let j=k;j<=n;j++)M[i][j]-=f*M[k][j];}}
  return M.map(r=>r[n]);
}
const meanOf=p=>dot(X,p);
function predict(p,direction){const out=Array(X.length).fill(0),sigma=.035;for(let i=0;i<X.length;i++){
  const m=reflect(X[i]+direction*STEP),q=X.map(x=>Math.exp(-.5*((x-m)/sigma)**2)),s=q.reduce((a,b)=>a+b,0);for(let j=0;j<X.length;j++)out[j]+=p[i]*q[j]/s;
}return out;}
function filter(predicted,samples,noise,mu,sigma){
  const n=samples[0].re.length,L=samples.length,energy=samples.reduce((s,z)=>s+dot(z.re,z.re)+dot(z.im,z.im),0);
  const projection=X.map(u=>samples.reduce((s,z)=>s+projectedPower(z,steering(u,n)),0));
  const powerPrior=T.map(t=>normalLog(t,mu,sigma)); // equal Δlogγ grid, Gaussian density in logγ; no 1/γ.
  const logw=[];for(let i=0;i<X.length;i++)for(let j=0;j<T.length;j++){
    const g=G[j],logdet=(n-1)*Math.log(noise)+Math.log(noise+g),q=energy/noise-g*projection[i]/(noise*(noise+g));
    logw.push(Math.log(Math.max(predicted[i],1e-300))+powerPrior[j]-L*logdet-q);
  }
  const w=normalizeLogWeights(logw),px=Array(X.length).fill(0),pt=Array(T.length).fill(0);
  for(let i=0;i<X.length;i++)for(let j=0;j<T.length;j++){const v=w[i*T.length+j];px[i]+=v;pt[j]+=v;}
  return{px,pt,elog:dot(T,pt),elog2:dot(T.map(t=>t*t),pt),logPowerBoundaryMass:pt[0]+pt.at(-1)};
}
export class LearningLab {
  constructor(options={}){this.reset(options);}
  reset(options={}){
    this.options={snr:6,seed:7,learning:true,...this.options,...options};this.random=rng(this.options.seed);this.frame=0;this.cycle=0;this.trainingCount=0;this.trace=[];this.sumErrorLearned=0;this.sumErrorFixed=0;
    this.noise=2*10**(-this.options.snr/10);this.beta=[priorMu,0,0,0];
    // Tikhonov regularization about the context-free initial mean, not truth.
    this.A=Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>i===j?12:0));this.b=[12*priorMu,0,0,0];
    this.pLearned=normalizeLogWeights(X.map(x=>normalLog(x,-.7,.18)));this.pFixed=this.pLearned.slice();
    this.xTruth=-.85;this.xLearned=meanOf(this.pLearned);this.xFixed=this.xLearned;this.context=this.xLearned;this.mu=priorMu;this.sigma=priorSigma;this.samplePower=0;this.boundaryMass=0;
    return this.getState();
  }
  step(){
    // The generator knows the hidden trajectory and field. Estimators receive only samples, noise and motion direction.
    const direction=Math.floor(this.frame/80)%2===0?1:-1;
    this.frame++;this.cycle=Math.floor((this.frame-1)/80)+1;this.xTruth=reflect(-.85+this.frame*STEP);
    const gamma=Math.exp(trueMu(this.xTruth)+trueSigma*normal(this.random));
    const samples=sampleSignal([{u:this.xTruth,power:gamma}],this.noise,4,3,this.random);
    this.samplePower=samples.reduce((s,z)=>s+dot(z.re,z.re)+dot(z.im,z.im),0)/3;
    this.updateFromSamples(samples,direction);
    this.sumErrorLearned+=(this.xLearned-this.xTruth)**2;this.sumErrorFixed+=(this.xFixed-this.xTruth)**2;
    this.trace.push({frame:this.frame,truth:this.xTruth,learned:this.xLearned,fixed:this.xFixed});
    return this.getState();
  }
  updateFromSamples(samples,direction){
    const predLearned=predict(this.pLearned,direction),predFixed=predict(this.pFixed,direction);
    this.context=meanOf(this.pLearned); // frozen preceding posterior mean, never hidden xTruth
    const phi=basis(this.context),mu=this.options.learning?dot(phi,this.beta):priorMu;
    // Same fixed spread in both filters isolates learning the context-dependent mean.
    this.sigma=priorSigma;
    this.mu=mu;
    const qLearned=filter(predLearned,samples,this.noise,mu,this.sigma),qFixed=filter(predFixed,samples,this.noise,priorMu,priorSigma);
    this.pLearned=qLearned.px;this.pFixed=qFixed.px;this.xLearned=meanOf(this.pLearned);this.xFixed=meanOf(this.pFixed);this.boundaryMass=qLearned.logPowerBoundaryMass;
    if(this.options.learning){
      // Exact regularized M-step for μθ(c)=φ(c)ᵀβ with fixed per-step σ.
      // Known existence gives weight one. Posterior targets and contexts stay frozen.
      for(let i=0;i<4;i++){this.b[i]+=phi[i]*qLearned.elog;for(let j=0;j<4;j++)this.A[i][j]+=phi[i]*phi[j];}
      this.beta=solve(this.A,this.b);this.trainingCount++;
    }
    return{posteriorLogPower:qLearned.pt,expectedLogPower:qLearned.elog,context:this.context};
  }
  getState(){
    const powerGrid=linspace(.02,14,200),mu=this.options.learning?dot(basis(this.context),this.beta):priorMu;
    const currentSigma=priorSigma;
    return{frame:this.frame,cycle:this.cycle,xTruth:this.xTruth,xLearned:this.xLearned,xFixed:this.xFixed,context:this.context,mu,sigma:currentSigma,priorMu,priorSigma,truthMu:trueMu(this.context),truthSigma:trueSigma,trace:this.trace.slice(),
      field:linspace(-.85,.85,101).map(x=>({x,truth:Math.exp(trueMu(x)+trueSigma**2/2),learned:Math.exp((this.options.learning?dot(basis(x),this.beta):priorMu)+currentSigma**2/2),fixed:Math.exp(priorMu+priorSigma**2/2),medianTruth:Math.exp(trueMu(x)),medianLearned:Math.exp(this.options.learning?dot(basis(x),this.beta):priorMu),medianFixed:Math.exp(priorMu)})),
      powerGrid,pdfLearned:powerGrid.map(g=>lognormalPDF(g,mu,currentSigma)),pdfFixed:powerGrid.map(g=>lognormalPDF(g,priorMu,priorSigma)),pdfTruth:powerGrid.map(g=>lognormalPDF(g,trueMu(this.context),trueSigma)),
      positionGrid:X.slice(),posteriorLearned:this.pLearned.slice(),posteriorFixed:this.pFixed.slice(),samplePower:this.samplePower,rmseLearned:this.frame?Math.sqrt(this.sumErrorLearned/this.frame):0,rmseFixed:this.frame?Math.sqrt(this.sumErrorFixed/this.frame):0,
      trainingCount:this.trainingCount,seed:this.options.seed,snr:this.options.snr,learning:this.options.learning,noise:this.noise,beta:this.beta.slice(),boundaryMass:this.boundaryMass,snapshots:3,sensors:4};
  }
}
