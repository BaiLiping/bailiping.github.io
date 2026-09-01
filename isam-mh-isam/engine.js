/* Numerical and symbolic teaching models. No GTSAM/MH-iSAM2 dependency. */
(function(root){
'use strict';
class SquareRoot {
  constructor(n=0){this.n=n;this.R=Array.from({length:n},()=>Array(n).fill(0));this.d=[Array(n).fill(0),Array(n).fill(0)];this.tail=[0,0];this.rotations=0;this.rows=0;}
  grow(){for(const row of this.R)row.push(0);this.n++;this.R.push(Array(this.n).fill(0));this.d.forEach(v=>v.push(0));}
  add(row, rhs){
    if(row.length!==this.n||rhs.length!==2||![...row,...rhs].every(Number.isFinite))throw new Error('Invalid whitened row');
    const a=row.slice(), b=rhs.slice();this.rows++;
    for(let k=0;k<this.n;k++){
      if(Math.abs(a[k])<1e-14)continue;
      const norm=Math.hypot(this.R[k][k],a[k]),c=this.R[k][k]/norm,s=a[k]/norm;
      for(let j=k;j<this.n;j++){const u=this.R[k][j],v=a[j];this.R[k][j]=c*u+s*v;a[j]=-s*u+c*v;}
      for(let q=0;q<2;q++){const u=this.d[q][k],v=b[q];this.d[q][k]=c*u+s*v;b[q]=-s*u+c*v;}
      this.rotations++;
    }
    this.tail[0]+=b[0]*b[0];this.tail[1]+=b[1]*b[1];
  }
  solve(){
    const out=this.d.map(v=>v.slice());
    for(let i=this.n-1;i>=0;i--){if(Math.abs(this.R[i][i])<1e-12)throw new Error('Rank-deficient graph: add an anchor and connect all variables');for(let q=0;q<2;q++){for(let j=i+1;j<this.n;j++)out[q][i]-=this.R[i][j]*out[q][j];out[q][i]/=this.R[i][i];}}
    return [[0,0],...Array.from({length:this.n},(_,i)=>[out[0][i],out[1][i]])];
  }
}
function factorRow(n,f){if(!(f.sigma>0)||f.i<0||f.j>n||f.i===f.j)throw new Error('Invalid factor');const a=Array(n).fill(0);if(f.i)a[f.i-1]-=1/f.sigma;if(f.j)a[f.j-1]+=1/f.sigma;return [a,f.z.map(v=>v/f.sigma)];}
function batch(n,factors){const qr=new SquareRoot(n);for(const f of factors)qr.add(...factorRow(n,f));return {points:qr.solve(),qr};}
function error(points,factors){return factors.reduce((s,f)=>s+f.z.reduce((t,z,q)=>t+((points[f.j][q]-points[f.i][q]-z)/f.sigma)**2,0),0);}
const truth=[[0,0],[2,0],[4,0],[4,2],[4,4],[2,4],[0,4],[0,2],[0,0]];
function odometry(k,sigma){return {i:k-1,j:k,z:truth[k].map((v,q)=>v-truth[k-1][q]+sigma*(q===0?.75+.25*Math.sin(k*1.7):.42+.2*Math.cos(k*1.3))),sigma};}
class IncrementalDemo{
  constructor(sigma=.24){this.sigma=sigma;this.n=0;this.closed=false;this.qr=new SquareRoot();this.factors=[];this.batchRows=0;this.batchRotations=0;}
  append(f,grow=false){if(grow){this.qr.grow();this.n++;}this.factors.push(f);this.qr.add(...factorRow(this.n,f));const check=batch(this.n,this.factors);this.batchRows+=this.factors.length;this.batchRotations+=check.qr.rotations;return this.snapshot();}
  next(){if(this.n>=8)return this.snapshot();return this.append(odometry(this.n+1,this.sigma),true);}
  close(){if(this.n!==8||this.closed)return this.snapshot();this.closed=true;return this.append({i:0,j:8,z:[0,0],sigma:.12});}
  snapshot(){const points=this.qr.solve(),b=batch(this.n,this.factors).points;return {points,error:error(points,this.factors),difference:Math.max(0,...points.flatMap((p,i)=>p.map((v,q)=>Math.abs(v-b[i][q])))),rows:this.factors.length,batchRows:this.batchRows,rotations:this.qr.rotations,batchRotations:this.batchRotations};}
}
const graphEdges=[[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[1,3],[3,6]];
const orders={chronological:[0,1,2,3,4,5,6,7,8],balanced:[0,2,4,6,1,3,5,7,8],recentFirst:[8,7,6,5,4,3,2,1,0]};
function symbolic(edges=graphEdges,order=orders.chronological){
  const n=order.length,rank=Object.fromEntries(order.map((v,i)=>[v,i]));
  if(new Set(order).size!==n)throw new Error('Invalid elimination order');
  const adj=Array.from({length:n},()=>new Set()),sep={},fills=[];for(const [a,b]of edges){adj[a].add(b);adj[b].add(a);}
  const alive=new Set(order);
  for(const v of order){const s=[...adj[v]].filter(x=>alive.has(x)).sort((a,b)=>rank[a]-rank[b]);sep[v]=s;for(let i=0;i<s.length;i++)for(let j=i+1;j<s.length;j++){const a=s[i],b=s[j];if(!adj[a].has(b)){adj[a].add(b);adj[b].add(a);fills.push([a,b]);}}alive.delete(v);}
  const cliques=[],owner={};
  for(const v of [...order].reverse()){
    const s=sep[v],parent=s.length?owner[s[0]]:null;
    if(parent!==null&&parent!==undefined&&s.length===cliques[parent].F.length+cliques[parent].S.length&&s.every(x=>[...cliques[parent].F,...cliques[parent].S].includes(x))){cliques[parent].F.unshift(v);owner[v]=parent;}
    else{const id=cliques.length;cliques.push({id,F:[v],S:s,parent,children:[]});owner[v]=id;if(parent!==null&&parent!==undefined)cliques[parent].children.push(id);}
  }
  return {cliques,owner,sep,order,rank,fills,nonzeros:n+Object.values(sep).reduce((s,a)=>s+a.length,0)};
}
function affected(tree,variables){const ids=new Set();for(const v of variables){let c=tree.owner[v];while(c!==null&&c!==undefined){ids.add(c);c=tree.cliques[c].parent;}}const orphans=tree.cliques.filter(c=>!ids.has(c.id)&&ids.has(c.parent));return {ids,orphans,variables:tree.cliques.filter(c=>ids.has(c.id)).reduce((s,c)=>s+c.F.length,0)};}
function mhFactors(stage,modes,sigma=.22){
  const f=[{i:0,j:1,z:[1,0],sigma:.16}];
  if(stage>=1)f.push({i:1,j:2,z:[1,modes[0]*1.2],sigma:.16});
  if(stage>=2)f.push({i:2,j:3,z:[1,0],sigma:.16},{i:3,j:4,z:[1,modes[1]*1.2],sigma:.16});
  if(stage>=3)f.push({i:0,j:4,z:[4,0],sigma});
  if(stage>=4)f.push({i:0,j:2,z:[2,1.2],sigma});
  return f;
}
const quantile95={2:5.991464547,4:9.487729037};
class HypothesisDemo{
  constructor(cap=4,sigma=.22,gate=true){this.cap=cap;this.sigma=sigma;this.gate=gate;this.stage=0;this.live=[{id:'',modes:[]}];this.history={};this.last=[];this.evaluate();}
  evaluate(){
    const n=this.stage===0?1:this.stage===1?2:4;
    const candidates=this.live.map(h=>{const factors=mhFactors(this.stage,h.modes,this.sigma),points=batch(n,factors).points,dof=2*factors.length-2*n,cost=error(points,factors);return {...h,points,cost,dof,status:'retained'};});
    const retained=[];
    // All modes here have the same dimension and covariance. The paper's
    // fewer-DoF preference is therefore inactive; tail probability order
    // is equivalent to increasing residual order. Ties use mode order.
    for(const h of candidates){if(this.gate&&h.dof>0&&h.cost>quantile95[h.dof])h.status='95% gate';else retained.push(h);}
    retained.sort((a,b)=>Math.abs(a.cost-b.cost)<1e-9?a.id.localeCompare(b.id):a.cost-b.cost);
    for(const h of retained.slice(this.cap))h.status='capacity';
    this.live=retained.slice(0,this.cap);this.last=candidates;
    for(const h of candidates)this.history[h.id]={...h,at:this.stage};
    return this;
  }
  next(){if(this.stage>=4)return this;this.stage++;if(this.stage===1||this.stage===2){this.live=this.live.flatMap(h=>[1,-1].map((v,i)=>({id:h.id+String(i),modes:[...h.modes,v]})));}return this.evaluate();}
}
const api={SquareRoot,factorRow,batch,error,truth,odometry,IncrementalDemo,graphEdges,orders,symbolic,affected,mhFactors,quantile95,HypothesisDemo};
if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.ISAMTeaching=api;
})(typeof globalThis!=='undefined'?globalThis:this);
