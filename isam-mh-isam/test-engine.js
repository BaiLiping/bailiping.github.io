// Run: node isam-mh-isam/test-engine.js
'use strict';
const assert=require('node:assert/strict'),E=require('./engine.js');
function close(a,b,tol=1e-9){assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);}
for(const sigma of [.08,.24,.5]){
 const m=new E.IncrementalDemo(sigma);
 for(let k=0;k<8;k++){const s=m.next();close(s.difference,0);close(s.error,0);}
 const out=m.close();assert.equal(out.rows,9);assert.equal(out.batchRows,45);close(out.difference,0);
 close(out.error,m.qr.tail.reduce((a,b)=>a+b,0));
 assert.ok(Math.hypot(...out.points.at(-1))<.2);
 m.close();assert.equal(m.factors.length,9); // Duplicate closure is a no-op.
}
for(const [name,order]of Object.entries(E.orders)){
 const t=E.symbolic(E.graphEdges,order),frontals=t.cliques.flatMap(c=>c.F);
 assert.equal(new Set(frontals).size,9);assert.equal(frontals.length,9);
 for(const c of t.cliques){if(c.parent!==null){const parent=t.cliques[c.parent];assert.ok(c.S.every(v=>[...parent.F,...parent.S].includes(v)));}else assert.equal(c.S.length,0);}
 const a=E.affected(t,[8]);assert.equal(a.ids.size,name==='recentFirst'?6:1);
}
const m=new E.HypothesisDemo(),counts=[];for(let k=0;k<4;k++){m.next();counts.push(m.live.length);}assert.deepEqual(counts,[2,4,2,1]);assert.equal(m.live[0].id,'01');
const narrow=new E.HypothesisDemo(1);for(let k=0;k<3;k++)narrow.next();assert.equal(narrow.live.length,0);narrow.next();assert.equal(narrow.live.length,0);
const noGate=new E.HypothesisDemo(4,.22,false);for(let k=0;k<4;k++)noGate.next();assert.equal(noGate.live.length,4);
const weak=new E.HypothesisDemo(4,.80,true);for(let k=0;k<4;k++)weak.next();assert.equal(weak.live.length,2);
// Both sign-reversed paths agree at the endpoint, but not at the midpoint.
const endpoint=new E.HypothesisDemo();for(let k=0;k<3;k++)endpoint.next();assert.deepEqual(endpoint.live.map(h=>h.id),['01','10']);
assert.throws(()=>new E.SquareRoot(1).solve(),/Rank-deficient/);
console.log('PASS: QR/batch agreement, residual identity, clique invariants, delayed disambiguation, capacity loss, weak/no gating.');
