// Parametric JPDA, one scan, independent Gaussian predicted target states.
// Fortmann et al. (1983), (3.18)-(3.20); Bar-Shalom et al. (2009), (38)-(51).
// No finite validation gate: P_G=1. Positive assignments must be injective.
export const add = (a,b) => a.map((row,i)=>row.map((x,j)=>x+b[i][j]));
export const scale = (a,s) => a.map(row=>row.map(x=>x*s));
export const transpose = a => a[0].map((_,j)=>a.map(row=>row[j]));
export const multiply = (a,b) => a.map(row=>b[0].map((_,j)=>row.reduce((v,x,k)=>v+x*b[k][j],0)));
export const mv = (a,v) => a.map(row=>row.reduce((s,x,i)=>s+x*v[i],0));
export const outer = v => v.map(x=>v.map(y=>x*y));
export const zero = () => [[0,0],[0,0]];
export const eye = s => [[s,0],[0,s]];
export const det = a => a[0][0]*a[1][1]-a[0][1]*a[1][0];
export const inverse = a => scale([[a[1][1],-a[0][1]],[-a[1][0],a[0][0]]],1/det(a));
const finitePair = v => Array.isArray(v)&&v.length===2&&v.every(Number.isFinite);
const positiveCovariance = a => Array.isArray(a)&&a.length===2&&a.every(finitePair)&&Math.abs(a[0][1]-a[1][0])<1e-10&&a[0][0]>0&&det(a)>0;
const symmetrize = a => [[a[0][0],(a[0][1]+a[1][0])/2],[(a[0][1]+a[1][0])/2,a[1][1]]];
export function gaussianLogDensity(z,mean,covariance) {
  const innovation=z.map((x,i)=>x-mean[i]);
  const weighted=mv(inverse(covariance),innovation);
  return -Math.log(2*Math.PI)-0.5*Math.log(det(covariance))-0.5*innovation.reduce((s,x,i)=>s+x*weighted[i],0);
}
export function normalizeLogs(values) {
  const largest=Math.max(...values);
  if(!Number.isFinite(largest)) throw new Error('No feasible event has positive probability under these settings.');
  const relative=values.map(value=>Math.exp(value-largest));
  const total=relative.reduce((a,b)=>a+b,0);
  return {probabilities:relative.map(value=>value/total),logNormalizer:largest+Math.log(total)};
}
export function enumerateAssignments(targetCount,measurementCount) {
  if(!Number.isInteger(targetCount)||targetCount<1||targetCount>4||!Number.isInteger(measurementCount)||measurementCount<0||measurementCount>8) throw new Error('Use 1–4 targets and 0–8 measurements.');
  const events=[];
  function visit(assignment,used) {
    if(assignment.length===targetCount) {events.push(assignment);return;}
    visit([...assignment,0],used);
    for(let j=1;j<=measurementCount;j++) if(!used.has(j)) visit([...assignment,j],new Set([...used,j]));
  }
  visit([],new Set());
  return events;
}
export function momentUpdate(track,measurements,noise,beta) {
  const S=add(track.cov,noise),K=multiply(track.cov,inverse(S));
  const corrected=symmetrize(add(track.cov,scale(multiply(multiply(K,S),transpose(K)),-1)));
  const branches=[{mean:[...track.mean],cov:track.cov},...measurements.map(z=>{
    const shift=mv(K,z.map((x,i)=>x-track.mean[i]));
    return {mean:track.mean.map((x,i)=>x+shift[i]),cov:corrected};
  })];
  const mean=[0,1].map(axis=>branches.reduce((sum,branch,j)=>sum+beta[j]*branch.mean[axis],0));
  let covariance=zero(),spread=zero(),base=zero();
  branches.forEach((branch,j)=>{
    const contribution=scale(outer(branch.mean.map((x,i)=>x-mean[i])),beta[j]);
    spread=add(spread,contribution);
    base=add(base,scale(branch.cov,beta[j]));
  });
  covariance=symmetrize(add(base,spread));
  return {mean,cov:covariance,S,K,corrected,branches,spread:symmetrize(spread),base:symmetrize(base)};
}
export function solveJPDA({tracks,measurements,noise,lambda}) {
  if(!Array.isArray(tracks)||!Array.isArray(measurements)||!measurements.every(finitePair)||!positiveCovariance(noise)||!Number.isFinite(lambda)||lambda<=0) throw new Error('Invalid measurements, measurement covariance, or clutter intensity.');
  if(!tracks.every(t=>finitePair(t.mean)&&positiveCovariance(t.cov)&&Number.isFinite(t.pd)&&t.pd>=0&&t.pd<=1)) throw new Error('Each track needs a finite mean, positive covariance, and detection probability in [0,1].');
  const assignments=enumerateAssignments(tracks.length,measurements.length);
  const logRatios=tracks.map(track=>measurements.map(z=>Math.log(track.pd)+gaussianLogDensity(z,track.mean,add(track.cov,noise))-Math.log(lambda)));
  const misses=tracks.map(track=>Math.log1p(-track.pd));
  const logWeights=assignments.map(a=>a.reduce((sum,j,t)=>sum+(j===0?misses[t]:logRatios[t][j-1]),0));
  const normalized=normalizeLogs(logWeights);
  const events=assignments.map((assignment,i)=>({id:assignment.join('-'),assignment,logWeight:logWeights[i],probability:normalized.probabilities[i]}));
  const beta=tracks.map(()=>Array(measurements.length+1).fill(0));
  events.forEach(event=>event.assignment.forEach((j,t)=>{beta[t][j]+=event.probability;}));
  const clutter=measurements.map((_,j)=>Math.max(0,1-beta.reduce((sum,row)=>sum+row[j+1],0)));
  const independent=logRatios.map((row,t)=>normalizeLogs([misses[t],...row]).probabilities);
  const updates=tracks.map((track,t)=>momentUpdate(track,measurements,noise,beta[t]));
  const pdaUpdates=tracks.map((track,t)=>momentUpdate(track,measurements,noise,independent[t]));
  const map=events.reduce((best,event)=>event.probability>best.probability?event:best,events[0]);
  const entropy=-events.reduce((sum,e)=>sum+(e.probability>0?e.probability*Math.log(e.probability):0),0);
  return {tracks,measurements,noise,lambda,events,beta,clutter,updates,independent,pdaUpdates,map,entropy,logRatios,logNormalizer:normalized.logNormalizer};
}
export const PRESETS={
  separated:{label:'Separated returns',separation:6,pd:0.9,lambda:0.02,sigma:0.65,measurements:[[-3.2,0.4],[3.1,-0.3],[0,3.6]]},
  ambiguous:{label:'Overlapping targets',separation:1.2,pd:0.9,lambda:0.02,sigma:0.65,measurements:[[-0.9,0.4],[0.9,-0.4],[0,2.8]]},
  shared:{label:'One shared return',separation:2,pd:0.9,lambda:0.02,sigma:0.65,measurements:[[0,0.25]]},
  missed:{label:'A missed detection',separation:5,pd:0.85,lambda:0.02,sigma:0.65,measurements:[[-2.4,0.3],[0,4]]}
};
export const preset = key => structuredClone(PRESETS[key]||PRESETS.shared);
export function sceneModel(state) {
  return {tracks:[{mean:[-state.separation/2,0],cov:eye(1.21),pd:state.pd},{mean:[state.separation/2,0],cov:eye(1.21),pd:state.pd}],measurements:state.measurements,noise:eye(state.sigma**2),lambda:state.lambda};
}
export const solveScene = state => solveJPDA(sceneModel(state));
