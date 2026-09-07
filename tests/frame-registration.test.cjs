'use strict';
// Numerical checks execute the deployed solvers, not independent lookalike implementations.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const {test} = require('node:test');
const root = path.resolve(__dirname, '..');
const article = fs.readFileSync(path.join(root, 'frame-registration/index.html'), 'utf8');
const scripts = [...article.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const noop = () => {};
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    value: '1', checked: true, textContent: '', style: {}, dataset: {},
    addEventListener: noop, getContext: () => ({}), getBoundingClientRect: () => ({width:640,height:420,left:0,top:0}),
    classList: {add:noop,remove:noop,toggle:noop},
  });
  return elements.get(id);
}
function load(prefix, names, cut='/* ---------------- boot ---------------- */') {
  let code = scripts.find(x => x.includes(prefix));
  assert.ok(code, 'Deployed script not found: ' + prefix);
  const i = code.indexOf(cut);
  assert.ok(i > 0, 'Missing test boundary');
  code = code.slice(0, i) + '\nglobalThis.model = {' + names.join(',') + '};\n})();';
  const box = {console, document:{getElementById:element,querySelectorAll:()=>[]},
    window:{matchMedia:()=>({matches:true}),addEventListener:noop},
    setTimeout:noop,clearTimeout:noop,setInterval:noop,clearInterval:noop,requestAnimationFrame:noop,
    performance:{now:()=>0}, ResizeObserver:class{observe(){}}};
  vm.runInNewContext(code, box, {timeout:5000});
  return box.model;
}
const race = load('const METHODS = [', ['weightedKabsch','applyT','composeT','makeScene','makeICP','makeSoft','makeMMD','makeRANSAC','makeGICP','makeNDT','makeBP','makePMBM','poseErr','buildNDT','ndtDeriv','ndtScore','newtonStep','featMean','mmdResidJac','land','landObjective','GATE','MAXIT']);
function close(a,b,tol=1e-8) {assert.ok(Math.abs(a-b)<=tol, `${a} != ${b} (tolerance ${tol})`);}
function objective(P,Q,T) {return P.reduce((s,p,i)=>{const a=race.applyT(T,p);return s+(a[0]-Q[i][0])**2+(a[1]-Q[i][1])**2;},0);}
test('Kabsch exactly recovers paired SE(2) motion across rotations and translations',()=>{
 const P=[[-2,-1],[.3,-1.5],[2,.4],[.7,2]];
 for(const th of [-3,-1,0,.6,2.7]) {
  const T={th,tx:1.2,ty:-.8}, Q=P.map(p=>race.applyT(T,p));
  const fit=race.weightedKabsch(P,Q,[1,2,.5,3]);
  close(fit.th,T.th);close(fit.tx,T.tx);close(fit.ty,T.ty);close(objective(P,Q,fit),0);
 }
});
test('SE(2) composition applies the rightmost transform first',()=>{
 const A={th:.4,tx:1,ty:2},B={th:-.8,tx:-.6,ty:.2},p=[3,-2];
 const x=race.applyT(race.composeT(A,B),p), y=race.applyT(A,race.applyT(B,p));
 close(x[0],y[0]);close(x[1],y[1]);
});
test('Planar proper rotations cannot exactly fit a generic reflected cloud',()=>{
 const P=[[-2,-1],[.2,0],[1,2],[3,-.5]], Q=P.map(p=>[-p[0],p[1]]);
 const T=race.weightedKabsch(P,Q,P.map(()=>1)); assert.ok(objective(P,Q,T)>1);
 close(Math.cos(T.th)**2+Math.sin(T.th)**2,1);
});
test('NDT skips coincident cells and regularizes collinear ones',()=>{
 const o={ox:0,oy:0,nGrids:1,minPts:3};
 const empty=race.buildNDT([[.1,.1],[.1,.1],[.1,.1]],1,o);assert.equal(empty.grids[0].size,0);
 const ndt=race.buildNDT([[.1,.2],[.2,.2],[.3,.2]],1,o);
 for(const cell of ndt.grids[0].values()) {assert.ok(cell.l2>0);assert.ok(Number.isFinite(cell.b11));assert.ok(Number.isFinite(cell.b22));}
});
test('NDT cell keys do not alias when coordinates exceed the original canvas',()=>{
 const pts=[[.1,.1],[.2,.2],[.3,.15],[1.1,-2047.9],[1.2,-2047.8],[1.3,-2047.85]];
 const ndt=race.buildNDT(pts,1,{ox:0,oy:0,nGrids:1,minPts:3});assert.equal(ndt.grids[0].size,2);
});
test('Random-feature mean and analytic pose Jacobian agree with finite differences',()=>{
 const P=[[-.7,.2],[.4,-1],[1,.8]],Q=[[.1,0],[.4,.5]],om=[[.3,-.8],[1.2,.5],[-.4,.2]];
 const T={th:.35,tx:.12,ty:-.08}, target=race.featMean(Q,om), z=race.mmdResidJac(P,T,om,target);
 assert.equal(z.r.length,2*om.length);assert.equal(z.J.length,z.r.length);
 const keys=['th','tx','ty'],eps=1e-6;
 for(let j=0;j<3;j++){
  const a={...T,[keys[j]]:T[keys[j]]+eps},b={...T,[keys[j]]:T[keys[j]]-eps};
  const ra=race.mmdResidJac(P,a,om,target).r,rb=race.mmdResidJac(P,b,om,target).r;
  for(let k=0;k<ra.length;k++)close(z.J[k][j],(ra[k]-rb[k])/(2*eps),1e-7);
 }
});
test('Capped NN landscape keeps a fixed denominator including unmatched points',()=>{
 race.land.mode='nn';race.land.scene={source:[[0,0],[20,20]],target:[[0,0]]};
 close(race.landObjective(0,0),race.GATE**2/2);
});
test('Gated translation-only ICP is non-increasing on the plotted capped cost',()=>{
 race.land.mode='nn';race.land.scene={source:[[-1,.3],[.2,.1],[1,.2],[20,20]],target:[[-1,0],[0,0],[1,0]]};
 let tx=.3,ty=.4;
 for(let it=0;it<8;it++){
  const before=race.landObjective(tx,ty);let dx=0,dy=0,n=0;
  for(const p of race.land.scene.source){let best=Infinity,qbest;for(const q of race.land.scene.target){const d=Math.hypot(p[0]+tx-q[0],p[1]+ty-q[1]);if(d<best){best=d;qbest=q;}}
   if(best<race.GATE){dx+=qbest[0]-p[0]-tx;dy+=qbest[1]-p[1]-ty;n++;}}
  if(!n)break;tx+=dx/n;ty+=dy/n;assert.ok(race.landObjective(tx,ty)<=before+1e-12);
 }
});
test('All nine deployed race solvers remain finite on six seeded scene settings',()=>{
 const makers=['makeICP','makeSoft','makeMMD','makeRANSAC','makeGICP','makeNDT','makeBP','makePMBM'];
 for(const setting of ['clean','noise','outliers','combo','gradient','striped']){
  const scene=race.makeScene(setting,40),T={th:Math.PI/4,tx:1,ty:.6};
  const solvers=makers.map(k=>race[k]());solvers.push(race.makeICP(true));
  for(const s of solvers){s.reset(scene,T,40);for(let i=0;i<race.MAXIT&&!s.done;i++)s.step();assert.ok([s.T.th,s.T.tx,s.T.ty].every(Number.isFinite));}
 }
});
test('RANSAC output is invariant to the supplied initial pose for a fixed seed',()=>{
 const scene=race.makeScene('combo',40);let out=[];
 for(const T of [{th:0,tx:0,ty:0},{th:2,tx:3,ty:-2}]){const s=race.makeRANSAC();s.reset(scene,T,40);for(let i=0;i<race.MAXIT&&!s.done;i++)s.step();out.push(s.T);}
 close(out[0].th,out[1].th);close(out[0].tx,out[1].tx);close(out[0].ty,out[1].ty);
});
test('Article and slide document explicitly carry the corrected model boundaries',()=>{
 assert.ok(article.includes('frame-registration-review: 2026-09-07-v1'));
 assert.ok(article.includes('r.deg < 5 && r.dist < 0.2'));
 assert.ok(article.includes('const MAXIT = 70'));
 assert.ok(!article.includes('name: \'PMBM assoc\''));
 assert.ok(article.includes("fld.hoverKey.split(',').map(Number)"));
 const html=fs.readFileSync(path.join(root,'frame-registration-slides/index.html'),'utf8');
 const doc=JSON.parse(html.match(/<script[^>]*id="bento-doc"[^>]*>([\s\S]*?)<\/script>/)[1]);
 assert.equal(doc.slides.length,20);
 const ids=doc.slides.map(s=>s.id);assert.equal(new Set(ids).size,20);
 const text=JSON.stringify(doc);
 assert.ok(text.includes('variance update is not lost'));assert.ok(text.includes('nonsingular pose Hessian'));assert.ok(text.toLowerCase().includes('translation-only direct search'));
 for(const s of doc.slides){assert.equal(new Set(s.elements.map(e=>e.id)).size,s.elements.length);for(const e of s.elements)if(e.type==='image')assert.ok(e.src.startsWith('data:image/png;base64,'));}
});
test('NDT analytic gradient and Hessian match finite differences inside fixed cells',()=>{
 const pts=[[.1,.2],[.3,.7],[.8,.15],[1.1,.9],[.6,1.2],[1.2,.4]];
 const ndt=race.buildNDT(pts,4,{ox:-1,oy:-1,nGrids:1,minPts:3});
 const T={th:.08,tx:.03,ty:-.02},d=race.ndtDeriv(ndt,pts,T),keys=['tx','ty','th'],eps=1e-5;
 for(let j=0;j<3;j++){
  const a={...T,[keys[j]]:T[keys[j]]+eps},b={...T,[keys[j]]:T[keys[j]]-eps};
  const da=race.ndtDeriv(ndt,pts,a),db=race.ndtDeriv(ndt,pts,b);
  close(d.g[j],(da.f-db.f)/(2*eps),2e-7);
  for(let k=0;k<3;k++)close(d.H[k][j],(da.g[k]-db.g[k])/(2*eps),2e-6);
 }
});
test('Accepted damped Newton steps improve the actual NDT score',()=>{
 const pts=[[.1,.2],[.3,.7],[.8,.15],[1.1,.9],[.6,1.2],[1.2,.4]];
 const ndt=race.buildNDT(pts,4,{ox:-1,oy:-1,nGrids:1,minPts:3});let T={th:.15,tx:.2,ty:-.1};
 for(let it=0;it<15;it++){const before=race.ndtScore(ndt,pts,T),r=race.newtonStep(ndt,pts,T,3);assert.ok(r.score>=before-1e-10);T=r.T;}
});
