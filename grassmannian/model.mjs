// Small real-matrix engine shared by the article, slide labs, and numerical tests.
// Matrices are row-major arrays. Bases have orthonormal columns.
export const clamp = (x, a = -1, b = 1) => Math.max(a, Math.min(b, x));
export const rad = d => d * Math.PI / 180;
export const deg = r => r * 180 / Math.PI;
export const zeros = (n, m) => Array.from({length:n}, () => Array(m).fill(0));
export const eye = n => Array.from({length:n}, (_,i) => Array.from({length:n}, (_,j) => +(i===j)));
export const transpose = A => A[0].map((_,j) => A.map(r => r[j]));
export const mul = (A,B) => A.map(r => B[0].map((_,j) => r.reduce((s,v,k) => s + v*B[k][j],0)));
export const add = (A,B,s=1) => A.map((r,i) => r.map((v,j) => v+s*B[i][j]));
export const scale = (A,s) => A.map(r => r.map(x => x*s));
export const norm = A => Math.hypot(...A.flat());
export const dot = (a,b) => a.reduce((s,x,i) => s+x*b[i],0);
export const unit = v => { const n=Math.hypot(...v); if(n<1e-14) throw Error('Zero direction'); return v.map(x=>x/n); };
export const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
export const projector = Q => mul(Q,transpose(Q));
export const orthError = Q => norm(add(mul(transpose(Q),Q),eye(Q[0].length),-1));
export const rotation = a => [[Math.cos(a),-Math.sin(a)],[Math.sin(a),Math.cos(a)]];
export const planeBasis = a => [[Math.cos(a),0],[0,1],[Math.sin(a),0]];

export function qr(A) {
  // Twice-reorthogonalized modified Gram-Schmidt; these labs use tiny matrices.
  const cols=[];
  for (const column of transpose(A)) {
    let v=[...column];
    for(let pass=0;pass<2;pass++) for(const q of cols){const c=dot(q,v);v=v.map((x,i)=>x-c*q[i]);}
    if(Math.hypot(...v)<1e-13) throw Error('Basis lost rank');
    cols.push(unit(v));
  }
  return transpose(cols);
}

export function eigenSymmetric(A) {
  const n=A.length, D=A.map(r=>[...r]), V=eye(n);
  for(let it=0;it<100*n*n;it++) {
    let p=0,q=0,largest=0;
    for(let i=0;i<n;i++) for(let j=i+1;j<n;j++) if(Math.abs(D[i][j])>largest){largest=Math.abs(D[i][j]);p=i;q=j;}
    if(largest<1e-14*Math.max(1,norm(D))) break;
    const a=.5*Math.atan2(2*D[p][q],D[q][q]-D[p][p]), c=Math.cos(a),s=Math.sin(a);
    const app=D[p][p],aqq=D[q][q],apq=D[p][q];
    for(let k=0;k<n;k++) if(k!==p&&k!==q){const kp=D[k][p],kq=D[k][q];D[k][p]=D[p][k]=c*kp-s*kq;D[k][q]=D[q][k]=s*kp+c*kq;}
    D[p][p]=c*c*app-2*c*s*apq+s*s*aqq;
    D[q][q]=s*s*app+2*c*s*apq+c*c*aqq;D[p][q]=D[q][p]=0;
    for(let k=0;k<n;k++){const vp=V[k][p],vq=V[k][q];V[k][p]=c*vp-s*vq;V[k][q]=s*vp+c*vq;}
  }
  const order=Array.from({length:n},(_,i)=>i).sort((a,b)=>D[b][b]-D[a][a]);
  return {values:order.map(i=>D[i][i]),vectors:V.map(r=>order.map(i=>r[i]))};
}

export function principalAngles(Q,Y) {
  const C=mul(transpose(Q),Y);
  // Only min(k1,k2) singular values: mixed-dimensional affine objects need this.
  const gram=C.length<C[0].length?mul(C,transpose(C)):mul(transpose(C),C);
  const {values}=eigenSymmetric(gram);
  return values.map(x=>Math.acos(clamp(Math.sqrt(Math.max(0,x)),0,1)));
}
export function distances(Q,Y) {
  const angles=principalAngles(Q,Y);
  return {angles,geodesic:Math.hypot(...angles),projection:Math.hypot(...angles.map(Math.sin)),spectral:Math.sin(angles.at(-1))};
}
export function anglePair(a,b,spin=0) {
  const Q=[[1,0],[0,1],[0,0],[0,0]];
  const Y=mul([[Math.cos(a),0],[0,Math.cos(b)],[Math.sin(a),0],[0,Math.sin(b)]],rotation(spin));
  return {Q,Y,...distances(Q,Y)};
}
export function horizontal(Q,A) { return add(A,mul(Q,mul(transpose(Q),A)),-1); }

export function exponential(Q,D,t=1) {
  // Q + (QV(cos(tΣ)-I) + U sin(tΣ))Vᵀ, omitting zero singular directions.
  if(norm(mul(transpose(Q),D))>1e-8) throw Error('Expected a horizontal tangent');
  const {values,vectors}=eigenSymmetric(mul(transpose(D),D));
  let out=Q.map(r=>[...r]);
  for(let j=0;j<values.length;j++) {
    const sigma=Math.sqrt(Math.max(0,values[j]));
    if(sigma<1e-12)continue;
    const v=vectors.map(r=>r[j]),qv=Q.map(r=>dot(r,v)),u=D.map(r=>dot(r,v)/sigma);
    const c=Math.cos(t*sigma)-1,s=Math.sin(t*sigma);
    out=out.map((r,i)=>r.map((x,k)=>x+(qv[i]*c+u[i]*s)*v[k]));
  }
  return out;
}
export function lineGeodesic(u,v,t) {
  const a=unit(u),raw=unit(v),sign=dot(a,raw)<0?-1:1,b=raw.map(x=>x*sign);
  const theta=Math.acos(clamp(dot(a,b),0,1));
  if(theta<1e-10)return {point:a,aligned:b,theta};
  const w=b.map((x,i)=>(x-a[i]*Math.cos(theta))/Math.sin(theta));
  return {point:a.map((x,i)=>x*Math.cos(t*theta)+w[i]*Math.sin(t*theta)),aligned:b,theta};
}

export function pcaData(lambda2=4) {
  // Deterministic, centered teaching cloud with EXACT sample covariance eigenvalues
  // (9, lambda2, 1), using orthogonal Fourier columns over 96 samples. Not Gaussian.
  const E=qr([[1,.2,-.7],[.7,1,.3],[.3,-.5,1]]), lambdas=[9,lambda2,1];
  const points=Array.from({length:96},(_,i)=>{
    const z=lambdas.map((x,j)=>Math.sqrt(2*x)*Math.cos(2*Math.PI*(j+1)*i/96));
    return E.map(r=>dot(r,z));
  });
  const C=scale(mul(transpose(points),points),1/points.length);
  return {points,C,E,lambdas,optimum:9+lambda2,target:E.map(r=>r.slice(0,2))};
}
export function pcaScore(Q,C) { const T=mul(transpose(Q),mul(C,Q)); return T.reduce((s,r,i)=>s+r[i],0); }
export const pcaGradient = (Q,C) => scale(horizontal(Q,mul(C,Q)),-2);
export function pcaStep(Q,C,initialStep=.15) {
  const G=pcaGradient(Q,C),g2=norm(G)**2,before=pcaScore(Q,C);
  if(g2<1e-20)return {Q,score:before,step:0,gradient:Math.sqrt(g2)};
  let step=initialStep,next;
  for(let i=0;i<45;i++) {
    next=qr(add(Q,G,-step));
    if(pcaScore(next,C)>=before+1e-4*step*g2) return {Q:next,score:pcaScore(next,C),step,gradient:norm(pcaGradient(next,C))};
    step*=.5;
  }
  return {Q,score:before,step:0,gradient:Math.sqrt(g2)};
}
export const initialPcaBasis = () => qr([[.2,.9],[.9,-.3],[.7,.5]]);
