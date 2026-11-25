/** Small, deterministic mathematical models for the Splat-LOAM teaching labs.
 * Renderer: 3-D planar Gaussians, spherical rays, exact plane intersections.
 * Tracker: 2-D geometric point-to-line analogue only, not the paper's full tracker.
 */
export const DEG = Math.PI / 180;
export const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
export const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
export const add = (a, b) => a.map((v, i) => v + b[i]);
export const sub = (a, b) => a.map((v, i) => v - b[i]);
export const mul = (a, s) => a.map(v => v * s);
export const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
export const norm = a => Math.sqrt(dot(a, a));
export const normalize = a => mul(a, 1 / norm(a));

export function sphericalRay(azimuth, elevation, yaw = 0) {
  const a = azimuth + yaw, c = Math.cos(elevation);
  return [c * Math.cos(a), c * Math.sin(a), Math.sin(elevation)];
}

export function makeSplat({id, center, tangentU, tangentV, scaleU, scaleV, opacity, color}) {
  const u = normalize(tangentU), v0 = sub(tangentV, mul(u, dot(tangentV, u))), v = normalize(v0);
  return {id, center:[...center], u, v, normal:normalize(cross(u,v)), scaleU, scaleV, opacity, color};
}

export function makeRenderScene({tilt = 25, scale = 1.15, opacity = 0.82} = {}) {
  const a = tilt * DEG;
  return [
    makeSplat({id:'A',center:[5,0,0],tangentU:[-Math.sin(a),Math.cos(a),0],tangentV:[0,0,1],scaleU:scale,scaleV:scale*0.85,opacity,color:'#c66b3d'}),
    makeSplat({id:'B',center:[8,0.6,0],tangentU:[0,1,0],tangentV:[0,0,1],scaleU:3.2,scaleV:1.9,opacity:0.9,color:'#167d8d'}),
    makeSplat({id:'C',center:[5.8,-3.4,-0.1],tangentU:[1,0,0],tangentV:[0,0,1],scaleU:1.5,scaleV:1.35,opacity:0.88,color:'#647d65'})
  ];
}

/** Ray direction must be unit length so t is metric range. No projection approximation. */
export function intersectSplat(ray, splat, origin = [0,0,0]) {
  const denominator = dot(splat.normal, ray);
  if (Math.abs(denominator) < 1e-9) return null;
  const range = dot(splat.normal, sub(splat.center, origin)) / denominator;
  if (!(range > 0) || !Number.isFinite(range)) return null;
  const point = add(origin, mul(ray, range));
  const local = sub(point, splat.center);
  const u = dot(local,splat.u) / splat.scaleU, v = dot(local,splat.v) / splat.scaleV;
  const gaussian = Math.exp(-0.5*(u*u + v*v));
  const alpha = clamp(splat.opacity*gaussian,0,1);
  return {id:splat.id, range, point, u, v, gaussian, alpha, color:splat.color};
}

/** D is deliberately unnormalised, matching the paper's front-to-back range sum. */
export function compositeHits(rawHits) {
  const ordered = rawHits.filter(Boolean).sort((a,b) => a.range - b.range);
  let transmission=1, depth=0, opacity=0;
  const hits = ordered.map(hit => {
    const weight=transmission*hit.alpha;
    const contribution=weight*hit.range;
    const result={...hit, transmission, weight, contribution};
    depth+=contribution; opacity+=weight; transmission*=1-hit.alpha;
    return result;
  });
  return {depth, expectedDepth:opacity>1e-6?depth/opacity:null, opacity, transmission, hits};
}
export function renderRay(ray, splats) { return compositeHits(splats.map(s => intersectSplat(ray,s))); }

/** Angular coordinates in degrees at the centre of a sensor-grid cell. */
export function pixelAngles(image, col, row) {
  return {
    azimuth:image.azMin+(col+0.5)/image.width*(image.azMax-image.azMin),
    elevation:image.elMax-(row+0.5)/image.height*(image.elMax-image.elMin)
  };
}

/** The cell containing an angular direction, clamped to this cropped field of view. */
export function pixelForAngles(image, azimuth, elevation) {
  return {
    col:clamp(Math.floor((azimuth-image.azMin)/(image.azMax-image.azMin)*image.width),0,image.width-1),
    row:clamp(Math.floor((image.elMax-elevation)/(image.elMax-image.elMin)*image.height),0,image.height-1)
  };
}

export function rangeImage(splats, {width=144,height=64,yaw=0,azMin=-55,azMax=55,elMin=-24,elMax=24}={}) {
  const depth=new Float32Array(width*height), expectedDepth=new Float32Array(width*height), opacity=new Float32Array(width*height);
  const image={width,height,depth,expectedDepth,opacity,azMin,azMax,elMin,elMax,yaw,
    footprintSigma:3,footprintCounts:Object.fromEntries(splats.map(s=>[s.id,0]))};
  for(let y=0;y<height;y++) for(let x=0;x<width;x++) {
    const angles=pixelAngles(image,x,y);
    const ray=sphericalRay(angles.azimuth*DEG,angles.elevation*DEG,yaw*DEG);
    const hits=splats.map(s=>intersectSplat(ray,s));
    // A teaching footprint only: the Gaussian has infinite support. This count
    // ignores opacity and occlusion and is not a number of detected returns.
    for(const hit of hits) if(hit && hit.u*hit.u+hit.v*hit.v<=image.footprintSigma**2) image.footprintCounts[hit.id]++;
    const r=compositeHits(hits), i=y*width+x;
    depth[i]=r.depth; expectedDepth[i]=r.expectedDepth ?? NaN; opacity[i]=r.opacity;
  }
  return image;
}

export function makeMap(kind='corner') {
  const line=(a,b)=>{const tangent=normalize(sub(b,a));return {a,b,normal:[-tangent[1],tangent[0]]};};
  return kind==='corridor'
    ? [line([-5,-2.4],[7,-2.4]),line([-5,2.4],[7,2.4])]
    : [line([4,-3],[4,3]),line([-3,3],[4,3]),line([-3,-2.5],[1.7,-2.5])];
}
function randomGenerator(seed) {
  let x=seed>>>0;
  return ()=>{x=(1664525*x+1013904223)>>>0;return (x+0.5)/4294967296;};
}
export function transformPoint(point,pose) {
  const c=Math.cos(pose[2]),s=Math.sin(pose[2]);
  return [c*point[0]-s*point[1]+pose[0],s*point[0]+c*point[1]+pose[1]];
}
/** Truth only generates observations; registerStep never receives it. */
export function makeScan(map, {noise=0.015, seed=37, truth=[0,0,0], countPerLine=42, trim=0.12}={}) {
  const rand=randomGenerator(seed), scan=[];
  for(const line of map) for(let i=0;i<countPerLine;i++) {
    const f=trim+(1-2*trim)*(i+0.5)/countPerLine;
    const world=add(line.a,mul(sub(line.b,line.a),f));
    const e=noise*Math.sqrt(-2*Math.log(rand()))*Math.cos(2*Math.PI*rand());
    const noisy=add(world,mul(line.normal,e));
    const relative=sub(noisy,truth.slice(0,2)),c=Math.cos(truth[2]),s=Math.sin(truth[2]);
    scan.push([c*relative[0]+s*relative[1],-s*relative[0]+c*relative[1]]);
  }
  return scan;
}
function closestOnSegment(p,l) {
  const d=sub(l.b,l.a), f=clamp(dot(sub(p,l.a),d)/dot(d,d),0,1);
  return add(l.a,mul(d,f));
}
function huber(r,delta) {const a=Math.abs(r); return a<=delta ? 0.5*r*r : delta*(a-0.5*delta);}
export function correspondences(scan,map,pose,maxDistance=1.8) {
  const c=Math.cos(pose[2]),s=Math.sin(pose[2]), out=[];
  for(const q of scan) {
    const p=transformPoint(q,pose);
    let best=null, d2=Infinity;
    for(const line of map) {
      const nearest=closestOnSegment(p,line), delta=sub(p,nearest), d=dot(delta,delta);
      if(d<d2){d2=d;best={line,nearest};}
    }
    if(!best || d2>maxDistance*maxDistance) continue;
    const n=best.line.normal, residual=dot(n,sub(p,best.line.a));
    const dtheta=[-s*q[0]-c*q[1],c*q[0]-s*q[1]];
    out.push({q,p,nearest:best.nearest,normal:n,residual,jacobian:[n[0],n[1],dot(n,dtheta)]});
  }
  return out;
}
export function evaluateRegistration(scan,map,pose,delta=0.12) {
  const matches=correspondences(scan,map,pose);
  return {matches, count:matches.length,
    rmse:matches.length?Math.sqrt(matches.reduce((v,m)=>v+m.residual*m.residual,0)/matches.length):Infinity,
    cost:matches.length?matches.reduce((v,m)=>v+huber(m.residual,delta),0)/matches.length:Infinity};
}
function solve3(A,b) {
  const a=A.map((r,i)=>[...r,b[i]]);
  for(let i=0;i<3;i++) {
    let k=i;
    for(let j=i+1;j<3;j++) if(Math.abs(a[j][i])>Math.abs(a[k][i])) k=j;
    [a[i],a[k]]=[a[k],a[i]];
    if(Math.abs(a[i][i])<1e-14) return null;
    const p=a[i][i];for(let j=i;j<4;j++)a[i][j]/=p;
    for(let r=0;r<3;r++) if(r!==i) {const f=a[r][i];for(let j=i;j<4;j++)a[r][j]-=f*a[i][j];}
  }
  return a.map(r=>r[3]);
}
/** Robust geometric Gauss–Newton with current-pose nearest-line matches and backtracking.
 * No truth pose, observation labels, or hard-coded convergence targets enter the update.
 */
export function registerStep(scan,map,pose,{huberDelta=0.12,damping=1e-5}={}) {
  const before=evaluateRegistration(scan,map,pose,huberDelta), H=Array.from({length:3},()=>[0,0,0]), g=[0,0,0];
  for(const m of before.matches) {
    const w=Math.abs(m.residual)<=huberDelta?1:huberDelta/Math.abs(m.residual);
    for(let i=0;i<3;i++) {g[i]+=w*m.jacobian[i]*m.residual;for(let j=0;j<3;j++) H[i][j]+=w*m.jacobian[i]*m.jacobian[j];}
  }
  const information=H.map(r=>[...r]);
  for(let i=0;i<3;i++)H[i][i]+=damping;
  const direction=solve3(H,g.map(v=>-v))||[0,0,0];
  const translation=Math.hypot(direction[0],direction[1]);
  let stepScale=Math.min(1,translation>0?0.65/translation:1,Math.abs(direction[2])>0?(8*DEG)/Math.abs(direction[2]):1);
  let next=[...pose],after=before,accepted=false;
  for(let backtrack=0;backtrack<12;backtrack++) {
    const candidate=pose.map((v,i)=>v+stepScale*direction[i]), e=evaluateRegistration(scan,map,candidate,huberDelta);
    if(e.count>=0.95*before.count && e.cost<=before.cost+1e-14) {next=candidate;after=e;accepted=true;break;}
    stepScale*=0.5;
  }
  const step=next.map((v,i)=>v-pose[i]);
  return {pose:next,before,after,step,accepted,information,unobservableX:information[0][0]<1e-9};
}
export function poseError(pose,truth=[0,0,0]) {
  return {translation:Math.hypot(pose[0]-truth[0],pose[1]-truth[1]), yaw:Math.abs(Math.atan2(Math.sin(pose[2]-truth[2]),Math.cos(pose[2]-truth[2])))/DEG};
}

