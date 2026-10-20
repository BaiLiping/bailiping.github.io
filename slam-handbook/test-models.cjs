/* Run with: node slam-handbook/test-models.cjs */
"use strict";
const fs=require("node:fs");
const path=require("node:path");
const vm=require("node:vm");
const assert=require("node:assert/strict");
const dir=__dirname;
const win={SLAM_CHAPTERS:[],SLAM_MODELS:{},SLAM_UTIL:{
  clamp:(x,a,b)=>Math.min(b,Math.max(a,x)),
  range:(a,b,n=80)=>Array.from({length:n+1},(_,i)=>a+(b-a)*i/n),
  line:(name,points)=>({name,points}),
  out:(metrics,insight,plot)=>({metrics,insight,plot}),
  curve:(xLabel,yLabel,...series)=>({xLabel,yLabel,series})
}};
const context=vm.createContext({window:win,console});
function load(file){vm.runInContext(fs.readFileSync(path.join(dir,file),"utf8"),context,{filename:file,timeout:20000})}
for(let i=1;i<=3;i++)load("chapters-"+i+".js");
for(let i=1;i<=18;i++)load("model-"+String(i).padStart(2,"0")+".js");
const chapters=win.SLAM_CHAPTERS.slice().sort((a,b)=>a.n-b.n);
assert.equal(chapters.length,18,"There must be 18 numbered chapters");
let checks=0,points=0;
for(let n=1;n<=18;n++){
  const chapter=chapters[n-1];
  assert.equal(chapter.n,n,"Chapters must be ordered and unique");
  assert.ok(chapter.title && chapter.goal && /^\d+–\d+$/.test(chapter.pages),"Missing title, goal or printed pages");
  assert.ok(chapter.ideas.length>=3 && chapter.checks.length>=2,"Lesson needs ideas and worked self-checks");
  assert.equal(chapter.demos.length,3,"Exactly three numerical experiments per chapter");
  for(const demo of chapter.demos){
    assert.ok(demo.key && demo.ref && demo.limit && demo.tryThis,"Missing experiment teaching metadata");
    assert.equal(typeof win.SLAM_MODELS[demo.key],"function","Missing model "+demo.key);
    for(const spec of demo.params)assert.ok(spec.min<=spec.value && spec.value<=spec.max && spec.step>0,"Invalid control bounds for "+spec.key);
    for(const which of ["default","min","max","middle"]){
      const settings={};
      for(const spec of demo.params)settings[spec.key]=which==="default"?spec.value:which==="min"?spec.min:which==="max"?spec.max:(Number(spec.min)+Number(spec.max))/2;
      const result=win.SLAM_MODELS[demo.key](settings);
      assert.ok(result && Array.isArray(result.metrics) && result.metrics.length,"No output metrics for "+demo.key);
      assert.ok(typeof result.insight==="string" && result.insight.length,"No interpretation for "+demo.key);
      assert.ok(result.plot && Array.isArray(result.plot.series) && result.plot.series.length,"No plot series for "+demo.key);
      for(const metric of result.metrics)assert.ok(!/NaN|Infinity/.test(String(metric[1])),"Non-finite metric for "+demo.key);
      for(const series of result.plot.series){
        assert.ok(series.points.length,"Empty curve "+demo.key);
        for(const pt of series.points){
          assert.ok(Number.isFinite(pt[0]) && Number.isFinite(pt[1]),"Non-finite plot point in "+demo.key);
          points++;
        }
      }
      checks++;
    }
  }
}
console.log("PASS: "+chapters.length+" chapters, "+chapters.length*3+" numerical models, "+checks+" parameter cases, "+points+" finite chart points.");
