/* RadarSplat teaching renderer. Independent implementation of paper §§3.5, 8.4.
 * Browser approximations: small scene, coarse range grid, normalized analytic
 * antenna kernels, first-order SH slice, no multipath source map or preprocessing.
 * All displayed powers are relative, not calibrated watts. */
export const TAU = 2 * Math.PI;
export const rad = d => d * Math.PI / 180;
export const deg = r => r * 180 / Math.PI;
export const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
export const wrap = a => ((a + Math.PI) % TAU + TAU) % TAU - Math.PI;
export const CONFIG = { H: 400, W: 240, Q: 4, rmin: 2.5, rmax: 50 };
export const DEFAULT_SENSOR = { x: 0, y: 0, z: 1.5, yaw: 0, azWidth: 1.8, elWidth: 12, leakage: .17, falloff: true };
export const INITIAL_SCENE = [
 {name:'Roadside reflector', x:20, y:6, z:1.5, sx:.8, sy:.35, sz:.45, yaw:25, pitch:0, rho:1.2, alpha:.85, eta:.1, directional:.45, facing:-145},
 {name:'Wall segment', x:29, y:-11, z:1.8, sx:3, sy:.35, sz:1.2, yaw:65, pitch:0, rho:1.8, alpha:.9, eta:.05, directional:.3, facing:160},
 {name:'Elevated return', x:13, y:-5, z:5, sx:.6, sy:.6, sz:.6, yaw:0, pitch:20, rho:1.5, alpha:.8, eta:.1, directional:.1, facing:160},
 {name:'Noise-like component', x:-13, y:18, z:1.5, sx:1.5, sy:.7, sz:.5, yaw:-25, pitch:0, rho:1, alpha:.08, eta:.87, directional:0, facing:0}
];
export const cloneScene = () => INITIAL_SCENE.map(g=>({...g}));
export function transpose(a) { return a[0].map((_,i)=>a.map(row=>row[i])); }
export function multiply(a,b) { return a.map(row=>b[0].map((_,j)=>row.reduce((s,v,k)=>s+v*b[k][j],0))); }
export function rotation(yaw,pitch=0) {
 const a=rad(yaw),b=rad(pitch),c=Math.cos(a),s=Math.sin(a),u=Math.cos(b),v=Math.sin(b);
 return [[c*u,-s,c*v],[s*u,c,s*v],[-v,0,u]];
}
export function covariance(g) {
 const R=rotation(g.yaw,g.pitch);
 return multiply(multiply(R,[[g.sx*g.sx,0,0],[0,g.sy*g.sy,0],[0,0,g.sz*g.sz]]),transpose(R));
}
export function project(g,sensor=DEFAULT_SENSOR) {
 const R=rotation(-sensor.yaw),d=[[g.x-sensor.x],[g.y-sensor.y],[g.z-sensor.z]];
 const [x,y,z]=multiply(R,d).map(r=>r[0]),h=Math.hypot(x,y),r=Math.hypot(h,z);
 if(r<1e-6 || h<1e-6) return null;
 const theta=Math.atan2(y,x),phi=Math.atan2(z,h);
 const J=[[x/r,y/r,z/r],[-y/(h*h),x/(h*h),0],[-x*z/(r*r*h),-y*z/(r*r*h),h/(r*r)]];
 const world=covariance(g),local=multiply(multiply(R,world),transpose(R));
 const spherical=multiply(multiply(J,local),transpose(J));
 const view=Math.atan2(sensor.y-g.y,sensor.x-g.x);
 // Constant + x/y direction terms are a restricted real degree-one SH model.
 const horizontal=Math.hypot(sensor.x-g.x,sensor.y-g.y)/r;
 const rho=Math.max(0,g.rho*(1+g.directional*horizontal*Math.cos(view-rad(g.facing))));
 const gain=Math.exp(-4*Math.log(2)*(deg(phi)/sensor.elWidth)**2);
 return {x,y,z,r,theta,phi,J,world,local,spherical,rho,gain, sigma:rho*Math.min(g.alpha+g.eta,1)};
}
export function returnWeight(g,p,mode) {
 return mode==='target'?p.rho*g.alpha:mode==='noise'?p.rho*g.eta:mode==='occupancy'?g.alpha:p.sigma;
}
export function azimuthConvolve(input,H,W,Q,width,enabled=true) {
 const out=new Float32Array(H*W),HH=H*Q,da=360/HH;
 const weights=Array.from({length:2*Q},(_,k)=>Math.exp(-4*Math.log(2)*(((k-Q+.5)*da)/width)**2));
 const sum=weights.reduce((s,x)=>s+x,0);
 for(let a=0;a<H;a++) {
  if(!enabled) { const src=(a*Q+Math.floor(Q/2))*W; out.set(input.subarray(src,src+W),a*W); continue; }
  for(let k=0;k<weights.length;k++) {
   const src=((a*Q+Math.floor(Q/2)+k-Q+HH)%HH)*W,w=weights[k]/sum,dst=a*W;
   for(let n=0;n<W;n++) out[dst+n]+=input[src+n]*w;
  }
 }
 return out;
}
export function rangeConvolve(input,H,W,sigma,dr) {
 if(sigma<=0) return input.slice();
 const radius=Math.ceil(3*sigma/dr),kernel=Array.from({length:2*radius+1},(_,k)=>Math.exp(-.5*((k-radius)*dr/sigma)**2));
 const sum=kernel.reduce((s,x)=>s+x,0); const out=new Float32Array(input.length);
 for(let a=0;a<H;a++)for(let n=0;n<W;n++) {
  let v=0;for(let k=-radius;k<=radius;k++)if(n+k>=0&&n+k<W)v+=input[a*W+n+k]*kernel[k+radius]/sum;
  out[a*W+n]=v;
 }
 return out;
}
export function render(scene,sensor,mode='total',config=CONFIG) {
 const {H,W,Q,rmin,rmax}=config,HH=H*Q,dr=(rmax-rmin)/W,da=TAU/HH;
 const geometry=new Float32Array(HH*W),power=new Float32Array(HH*W);
 const projections=scene.map(g=>project(g,sensor));
 scene.forEach((g,i)=>{
  const p=projections[i];if(!p)return;
  // Marginalize elevation: retain the (range,azimuth) covariance block.
  // Pixel integration is approximated by a small variance floor, avoiding aliasing.
  const A=p.spherical[0][0]+dr*dr/12,B=p.spherical[0][1],C=p.spherical[1][1]+da*da/12;
  const det=A*C-B*B;if(!(det>0))return;
  const sr=Math.sqrt(A),st=Math.sqrt(C),ac=(p.theta+Math.PI)/da-.5;
  const ar=Math.min(Math.ceil(4*st/da),Math.floor(HH/2)-1),nr=Math.ceil(4*sr/dr),nc=(p.r-rmin)/dr-.5;
  const weight=returnWeight(g,p,mode)*p.gain*p.gain;
  for(let aa=Math.floor(ac)-ar;aa<=Math.ceil(ac)+ar;aa++) {
   const a=(aa%HH+HH)%HH,dt=wrap(-Math.PI+(a+.5)*da-p.theta);
   for(let n=Math.max(0,Math.floor(nc)-nr);n<=Math.min(W-1,Math.ceil(nc)+nr);n++) {
    const r=rmin+(n+.5)*dr,d=r-p.r,mahal=(C*d*d-2*B*d*dt+A*dt*dt)/det;
    if(mahal>16)continue;
    const value=Math.exp(-.5*mahal),idx=a*W+n;
    geometry[idx]+=value;power[idx]+=value*weight*(sensor.falloff?(10/r)**4:1);
   }
  }
 });
 const footprint=azimuthConvolve(geometry,H,W,Q,sensor.azWidth,false);
 const elevation=azimuthConvolve(power,H,W,Q,sensor.azWidth,false);
 const azimuth=azimuthConvolve(power,H,W,Q,sensor.azWidth,true);
 const final=rangeConvolve(azimuth,H,W,sensor.leakage,dr);
 return {footprint,elevation,azimuth,final,projections,config,dr};
}
export function peaks(profile,threshold) {
 const out=[];
 for(let i=1;i<profile.length-1;i++)if(profile[i]>=threshold&&profile[i]>profile[i-1]&&profile[i]>=profile[i+1])out.push(i);
 return out;
}
// Independent 1-D inverse problem. This is not the full paper training algorithm.
export const TRAIN_TARGET={r:21,s:.65,rho:1.2,alpha:.85,eta:.1};
export const TRAIN_INITIAL={r:18.8,s:1.8,rho:.7,alpha:.1,eta:.1};
export const TRAIN_GRID=Array.from({length:180},(_,i)=>10+(i+.5)*25/180);
export function trainProfile(p,occupancy=false) {
 const a=occupancy?p.alpha:p.rho*Math.min(p.alpha+p.eta,1);
 const arr=Float32Array.from(TRAIN_GRID,r=>a*Math.exp(-.5*((r-p.r)/p.s)**2)*(10/r)**4);
 return rangeConvolve(arr,1,arr.length,.17,25/180);
}
const targetPower=trainProfile(TRAIN_TARGET),targetOcc=trainProfile(TRAIN_TARGET,true);
export function trainLoss(p,occWeight=1) {
 const pred=trainProfile(p),occ=trainProfile(p,true);
 let a=0,b=0,sa=0,sb=0;
 for(let i=0;i<pred.length;i++) {a+=(pred[i]-targetPower[i])**2;b+=(occ[i]-targetOcc[i])**2;sa+=targetPower[i]**2;sb+=targetOcc[i]**2;}
 return a/sa+occWeight*b/sb+4*Math.max(0,p.alpha+p.eta-1)**2;
}
export function trainStep(p,occWeight=1) {
 const keys=['r','s','rho','alpha','eta'],eps=[.01,.005,.005,.002,.002],pre=[4,.65,.35,.18,.18];
 const grad=keys.map((key,i)=>{const a={...p,[key]:p[key]+eps[i]},b={...p,[key]:p[key]-eps[i]};return (trainLoss(a,occWeight)-trainLoss(b,occWeight))/(2*eps[i]);});
 const before=trainLoss(p,occWeight);let best={...p},loss=before,rate=.6,accepted=false;
 for(let trial=0;trial<14;trial++,rate*=.5) {
  const q={...p}; keys.forEach((k,i)=>q[k]-=rate*pre[i]*grad[i]);
  q.r=clamp(q.r,10.5,34.5);q.s=clamp(q.s,.2,3);q.rho=clamp(q.rho,.05,3);q.alpha=clamp(q.alpha,.01,.99);q.eta=clamp(q.eta,.01,.99);
  const candidate=trainLoss(q,occWeight);
  if(candidate<loss) {best=q;loss=candidate;accepted=true;break;}
 }
 return {p:best,loss,grad,rate:accepted?rate:0};
}
