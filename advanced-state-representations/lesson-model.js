/* Pure, deterministic mathematics shared by the lesson, live labs and tests.
 * Conventions: column vectors; T maps body to world; twists are translation-first.
 * Dense linear algebra is intentional for these tiny teaching problems, not a
 * claim about production sparse-solver performance. No external dependencies.
 */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.ASRMath=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const zeros=(n,m=n)=>Array.from({length:n},()=>Array(m).fill(0));
const eye=n=>zeros(n).map((r,i)=>r.map((_,j)=>+(i===j)));
const tr=A=>A[0].map((_,j)=>A.map(r=>r[j]));
const mm=(A,B)=>A.map(r=>B[0].map((_,j)=>r.reduce((s,v,k)=>s+v*B[k][j],0)));
const mv=(A,x)=>A.map(r=>r.reduce((s,v,k)=>s+v*x[k],0));
const add=(A,B)=>A.map((r,i)=>r.map((v,j)=>v+B[i][j]));
const scale=(A,s)=>A.map(r=>r.map(v=>v*s));
const sub=(A,B)=>add(A,scale(B,-1));
const norm=A=>Math.hypot(...A.flat());
function chol(A){const n=A.length,L=zeros(n);for(let i=0;i<n;i++)for(let j=0;j<=i;j++){let s=A[i][j];for(let k=0;k<j;k++)s-=L[i][k]*L[j][k];if(i===j){if(!(s>0))throw Error('Matrix is not positive definite');L[i][j]=Math.sqrt(s);}else L[i][j]=s/L[j][j];}return L;}
function solveL(L,b){const n=b.length,y=Array(n),x=Array(n);for(let i=0;i<n;i++){let s=b[i];for(let j=0;j<i;j++)s-=L[i][j]*y[j];y[i]=s/L[i][i];}for(let i=n-1;i>=0;i--){let s=y[i];for(let j=i+1;j<n;j++)s-=L[j][i]*x[j];x[i]=s/L[i][i];}return x;}
const solve=(A,b)=>solveL(chol(A),b);
function inverse(A){const L=chol(A);return tr(eye(A.length).map(e=>solveL(L,e)));}
const sinc=x=>Math.abs(x)<1e-4?1-x*x/6+x**4/120:Math.sin(x)/x;
const cosc=x=>Math.abs(x)<1e-4?x/2-x**3/24+x**5/720:(1-Math.cos(x))/x;
const rot=a=>[[Math.cos(a),-Math.sin(a)],[Math.sin(a),Math.cos(a)]];
const J2=[[0,-1],[1,0]];
const pose=(a=0,x=0,y=0)=>[[Math.cos(a),-Math.sin(a),x],[Math.sin(a),Math.cos(a),y],[0,0,1]];
function exp2(x){const [a,b,w]=x,s=sinc(w),c=cosc(w);return pose(w,s*a-c*b,c*a+s*b);}
function log2(T){const w=Math.atan2(T[1][0],T[0][0]),s=sinc(w),c=cosc(w),d=s*s+c*c;return [(s*T[0][2]+c*T[1][2])/d,(-c*T[0][2]+s*T[1][2])/d,w];}
function inv2(T){const a=T[0][0],b=T[1][0],x=T[0][2],y=T[1][2];return [[a,b,-a*x-b*y],[-b,a,b*x-a*y],[0,0,1]];}
const adj2=T=>[[T[0][0],T[0][1],T[1][2]],[T[1][0],T[1][1],-T[0][2]],[0,0,1]];
const apply2=(T,p)=>[T[0][0]*p[0]+T[0][1]*p[1]+T[0][2],T[1][0]*p[0]+T[1][1]*p[1]+T[1][2]];
const skew=v=>[[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]];
function exp3(v){const a=Math.hypot(...v),W=skew(v),b=Math.abs(a)<1e-4?.5-a*a/24+a**4/720:(1-Math.cos(a))/(a*a);return add(add(eye(3),scale(W,sinc(a))),scale(mm(W,W),b));}
function jl3(v){const a=Math.hypot(...v),W=skew(v),b=a<1e-4?.5-a*a/24+a**4/720:(1-Math.cos(a))/(a*a),c=a<1e-4?1/6-a*a/120+a**4/5040:(a-Math.sin(a))/(a**3);return add(add(eye(3),scale(W,b)),scale(mm(W,W),c));}
const jr3=v=>jl3(v.map(x=>-x));
function expSE3(x){const R=exp3(x.slice(3)),p=mv(jl3(x.slice(3)),x.slice(0,3));return [...R.map((r,i)=>[...r,p[i]]),[0,0,0,1]];}
function adj3(T){const R=T.slice(0,3).map(r=>r.slice(0,3)),p=T.slice(0,3).map(r=>r[3]),S=mm(skew(p),R);return [...R.map((r,i)=>[...r,...S[i]]),...R.map(r=>[0,0,0,...r])];}
const det2=R=>R[0][0]*R[1][1]-R[0][1]*R[1][0];
const valid2=T=>{const R=T.slice(0,2).map(r=>r.slice(0,2));return {det:det2(R),error:norm(sub(mm(tr(R),R),eye(2)))};};
function tangent(theta,delta){const p=[Math.cos(theta),Math.sin(theta)],v=[-Math.sin(theta),Math.cos(theta)];return {p,v,linear:p.map((x,i)=>x+delta*v[i]),exact:[Math.cos(theta+delta),Math.sin(theta+delta)],dot:p[0]*v[0]+p[1]*v[1]};}
function rotation(delta,n){let exact=eye(2),naive=eye(2),E=rot(delta),N=add(eye(2),scale(J2,delta));const path=[];for(let k=0;k<=n;k++){path.push({exact:exact.map(r=>r[0]),naive:naive.map(r=>r[0])});if(k<n){exact=mm(exact,E);naive=mm(naive,N);}}return {exact,naive,path,radius:(1+delta*delta)**(n/2),exactAngle:n*delta,naiveAngle:n*Math.atan(delta)};}
function sides(theta,tx,delta){const T=pose(theta,tx,.6),xi=[.7,.2,delta],eta=mv(adj2(T),xi),right=mm(T,exp2(xi)),left=mm(exp2(eta),T),wrong=mm(exp2(xi),T);return {T,xi,eta,right,left,wrong,error:norm(sub(right,left)),wrongError:norm(sub(right,wrong))};}
const points=[[-1,-.5],[.9,-.6],[.8,.8],[-.7,1]];
const truth=pose(35*Math.PI/180,.8,.5);
const measurements=points.map(p=>apply2(truth,p));
function residual(T){return points.flatMap((p,i)=>apply2(T,p).map((x,j)=>x-measurements[i][j]));}
function jacobian(T){const R=T.slice(0,2).map(r=>r.slice(0,2));return points.flatMap(p=>{const v=mv(R,[-p[1],p[0]]);return [[...R[0],v[0]],[...R[1],v[1]]];});}
const cost=T=>residual(T).reduce((s,v)=>s+v*v,0)/2;
function fit(angle,steps){let T=pose(angle,-.5,-.4);const history=[{T,cost:cost(T),alpha:0}];for(let i=0;i<steps;i++){const r=residual(T),J=jacobian(T),A=add(mm(tr(J),J),scale(eye(3),1e-8)),d=solve(A,mv(tr(J),r).map(v=>-v));let alpha=1,U=T;while(alpha>=1/4096){U=mm(T,exp2(d.map(x=>alpha*x)));if(cost(U)<=cost(T)+1e-13)break;alpha/=2;}T=U;history.push({T,cost:cost(T),alpha});}return {T,history,points,measurements,estimated:points.map(p=>apply2(T,p)),cost:cost(T)};}
function knots(n,p){if(n<=p||p<0)throw Error('Invalid spline order');const spans=n-p;return [...Array(p+1).fill(0),...Array.from({length:spans-1},(_,i)=>i+1),...Array(p+1).fill(spans)].map(x=>6*x/spans);}
function basis(n,p,t,U=knots(n,p)){if(t<0||t>6)throw Error('Spline query outside [0,6]');if(t===6)return Array.from({length:n},(_,i)=>+(i===n-1));let N=Array.from({length:U.length-1},(_,i)=>+(U[i]<=t&&t<U[i+1]));for(let d=1;d<=p;d++)N=Array.from({length:U.length-d-1},(_,i)=>{const a=U[i+d]-U[i],b=U[i+d+1]-U[i+1];return (a?(t-U[i])*N[i]/a:0)+(b?(U[i+d+1]-t)*N[i+1]/b:0);});return N.slice(0,n);}
function derivative(n,p,t,U=knots(n,p)){if(p===0)return Array(n).fill(0);let q=t===6?6-1e-10:t;const N=basis(n+1,p-1,q,U);return Array.from({length:n},(_,i)=>{const a=U[i+p]-U[i],b=U[i+p+1]-U[i+1];return (a?p*N[i]/a:0)-(b?p*N[i+1]/b:0);});}
const coeff=[.1,.9,-.4,1.0,.2,-.6,.7,1.2,.4];
function spline(p,t,selected,shift){const n=coeff.length,U=knots(n,p),c=coeff.map((x,i)=>x+(i===selected?shift:0)),N=basis(n,p,t,U),D=derivative(n,p,t,U);return {U,c,N,value:N.reduce((s,w,i)=>s+w*c[i],0),velocity:D.reduce((s,w,i)=>s+w*c[i],0),support:[U[selected],U[selected+p+1]],active:N.flatMap((w,i)=>w>1e-12?[i]:[])};}
const observations=[{t:.35,z:.3},{t:1.1,z:1.05},{t:1.8,z:.45},{t:3,z:-.35},{t:4.2,z:.8},{t:5.4,z:1.5}];
function dynamics(kind,h,q){if(h<0||!(q>0))throw Error('Invalid process interval/variance');return kind==='rw'?{F:[[1]],Q:[[q*h]]}:{F:[[1,h],[0,1]],Q:scale([[h**3/3,h*h/2],[h*h/2,h]],q)};}
function posterior(kind='cv',q=.3,sigma=.2,extra=[]){if(!['rw','cv'].includes(kind)||q<=0||sigma<=0)throw Error('Invalid GP parameters');const times=[...new Set([0,6,...observations.map(o=>o.t),...extra])].sort((a,b)=>a-b);if(times[0]<0||times.at(-1)>6)throw Error('State time outside [0,6]');const d=kind==='rw'?1:2,n=times.length*d,prior=zeros(n),info=zeros(n),eta=Array(n).fill(0);for(let i=0;i<d;i++)prior[i][i]=1;for(let k=0;k<times.length-1;k++){const {F,Q}=dynamics(kind,times[k+1]-times[k],q),W=inverse(Q),B=F.map((r,i)=>[...r.map(v=>-v),...eye(d)[i]]),G=mm(mm(tr(B),W),B);for(let i=0;i<2*d;i++)for(let j=0;j<2*d;j++)prior[k*d+i][k*d+j]+=G[i][j];}for(let i=0;i<n;i++)for(let j=0;j<n;j++)info[i][j]=prior[i][j];for(const o of observations){const k=times.indexOf(o.t)*d;info[k][k]+=1/(sigma*sigma);eta[k]+=o.z/(sigma*sigma);}const L=chol(info),mean=solveL(L,eta),cov=tr(eye(n).map(e=>solveL(L,e)));return {kind,q,sigma,times,d,prior,info,mean,cov};}
function queryGP(post,t){if(t<post.times[0]||t>post.times.at(-1))throw Error('Query outside GP interval');const {times,d,q,kind,mean,cov}=post,n=mean.length;let k=times.findIndex(s=>s===t);if(k>=0)return {mean:mean[k*d],variance:cov[k*d][k*d],bridge:0,supportVariance:cov[k*d][k*d],weights:mean.map((_,i)=>+(i===k*d)),interval:[t,t]};k=times.findIndex(s=>s>t)-1;const h=times[k+1]-times[k],u=t-times[k],{F:Fh,Q:Qh}=dynamics(kind,h,q),{F:Fu,Q:Qu}=dynamics(kind,u,q),Fr=dynamics(kind,h-u,q).F,Psi=mm(mm(Qu,tr(Fr)),inverse(Qh)),Lambda=sub(Fu,mm(Psi,Fh)),V=sub(Qu,mm(mm(Psi,Qh),tr(Psi))),w=Array(n).fill(0);for(let j=0;j<d;j++){w[k*d+j]=Lambda[0][j];w[(k+1)*d+j]=Psi[0][j];}const mu=w.reduce((s,v,i)=>s+v*mean[i],0),sv=mv(cov,w).reduce((s,v,i)=>s+v*w[i],0),bridge=Math.max(0,V[0][0]);return {mean:mu,variance:sv+bridge,bridge,supportVariance:sv,weights:w,interval:[times[k],times[k+1]]};}
function kernel(kind,q,s,t){const m=Math.min(s,t),M=Math.max(s,t);return kind==='rw'?1+q*m:1+s*t+q*(m*m*M/2-m**3/6);}
function denseQuery(kind,q,sigma,t){const K=observations.map(a=>observations.map(b=>kernel(kind,q,a.t,b.t)+(a===b?sigma*sigma:0))),v=observations.map(o=>kernel(kind,q,t,o.t)),L=chol(K),m=solveL(L,observations.map(o=>o.z)),s=solveL(L,v);return {mean:v.reduce((a,b,i)=>a+b*m[i],0),variance:kernel(kind,q,t,t)-v.reduce((a,b,i)=>a+b*s[i],0)};}
function interpolate(angle,u){const T0=pose(),T1=pose(angle,2,1),xi=log2(mm(inv2(T0),T1)),group=mm(T0,exp2(xi.map(v=>u*v))),matrix=add(scale(T0,1-u),scale(T1,u)),split=pose(u*angle,2*u,u);return {T0,T1,xi,group,matrix,split};}
return {zeros,eye,tr,mm,mv,add,scale,sub,norm,chol,solve,inverse,rot,J2,pose,exp2,log2,inv2,adj2,apply2,skew,exp3,jl3,jr3,expSE3,adj3,valid2,tangent,rotation,sides,points,measurements,residual,jacobian,cost,fit,knots,basis,derivative,coeff,spline,observations,dynamics,posterior,queryGP,kernel,denseQuery,interpolate};
});
