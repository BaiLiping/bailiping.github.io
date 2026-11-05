const test=require('node:test'),assert=require('node:assert/strict'),BP=require('./model.js');
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
test('Figure 7 schedule sends both directions of every edge once',()=>{
  const r=BP.run();assert.deepEqual(r.phases.map(p=>p.length),[4,4,2,4,4]);
  assert.equal(Object.keys(r.messages).length,18);
  for(const [f,x] of BP.edges){assert.ok(r.messages[BP.key(f,x)]);assert.ok(r.messages[BP.key(x,f)]);}
  assert.deepEqual(r.readyAt,{x1:4,x2:4,x3:3,x4:5,x5:5});
});
test('Teaching calculation and normalization match the independently enumerated joint',()=>{
  const r=BP.run();close(r.messages['C>x3'].values[1],.528);close(r.messages['C>x3'].values[0],.472);
  [5.664,7.920].forEach((v,i)=>close(r.raw.x3[i],v));close(r.exact.z,13.584);
  for(const x of BP.variables){close(r.raw[x][0]+r.raw[x][1],r.exact.z);r.beliefs[x].forEach((v,i)=>close(v,r.exact.beliefs[x][i]));}
});
test('Every message equals enumeration of its sender-side component',()=>{
  let seed=20260918;const random=()=>((seed=(1664525*seed+1013904223)>>>0)/2**32);
  for(let trial=0;trial<70;trial++){
    const p={a:random(),b:random(),q:random()},r=BP.run(p);
    assert.ok(r.maxError<1e-12);
    for(const m of Object.values(r.messages)){
      const seen=new Set([m.from]),queue=[m.from];
      while(queue.length){const node=queue.shift();for(const n of BP.neighbors(node)){if((node===m.from&&n===m.to)||seen.has(n))continue;seen.add(n);queue.push(n);}}
      const factors=[...seen].filter(n=>BP.scopes[n]);
      const vars=[...new Set([m.variable,...factors.flatMap(f=>BP.scopes[f])])];
      const sums=[0,0];for(const assignment of BP.assignments(vars)){const w=factors.reduce((v,f)=>v*BP.factor(f,assignment,p),1);sums[assignment[m.variable]]+=w;}
      sums.forEach((v,i)=>close(v,m.values[i]));
    }
    for(const x of BP.variables){close(r.raw[x][0]+r.raw[x][1],r.exact.z);close(r.beliefs[x][0]+r.beliefs[x][1],1);}
  }
});
test('Boundary priors and neutral XOR agree with exhaustive inference',()=>{
  for(const a of [0,.05,.5,.95,1])for(const b of [0,.05,.5,.95,1])for(const q of [.5,.85,.99,1]){const r=BP.run({a,b,q});assert.ok(r.maxError<1e-12);if(q===.5)r.messages['C>x3'].values.forEach(v=>close(v,.5));}
});
test('Partial beliefs equal inference over exactly the factors received so far',()=>{
  const r=BP.run();
  for(let phase=1;phase<=5;phase++)for(const v of BP.variables){
    const b=BP.beliefAt(r,v,phase),weights=[0,0];
    for(const assignment of BP.assignments(BP.variables))weights[assignment[v]]+=b.factors.reduce((w,f)=>w*BP.factor(f,assignment,r.p),1);
    BP.normalize(weights).forEach((n,i)=>close(n,b.belief[i]));
    assert.equal(b.complete,phase>=r.readyAt[v]);
    if(b.complete)b.belief.forEach((n,i)=>close(n,r.exact.beliefs[v][i]));
  }
  close(BP.beliefAt(r,'x3',1).belief[1],.5);
  close(BP.beliefAt(r,'x3',2).belief[1],15/27);
  close(BP.beliefAt(r,'x3',3).belief[1],7.92/13.584);
});
test('The two central messages summarize disjoint factor sets',()=>{
  assert.deepEqual(BP.senderSide('C','x3').factors,['A','B','C']);
  assert.deepEqual(BP.senderSide('x3','C').factors,['D','E']);
  assert.deepEqual(BP.senderSide('x4','D').factors,[]);
  const before=BP.run(),after=BP.run({a:.95,b:.05,q:.99});
  assert.deepEqual(before.messages['x3>C'].values,[12,15]);
  assert.deepEqual(before.messages['x3>C'].values,after.messages['x3>C'].values);
  assert.notDeepEqual(before.messages['C>x3'].values,after.messages['C>x3'].values);
});
