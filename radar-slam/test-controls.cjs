// DOM-free controller tests: drive real input handlers and scheduled ticks.
// These check computation/lifecycle wiring; visual and native keyboard QA is separate.
const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
function mount(demo){
 const nodes=new Map(),presets=[],timers=new Map(),events={},frames=[];let timerId=0;
 const ctx=new Proxy({}, {get:(o,k)=>o[k]??(()=>{}),set:(o,k,v)=>(o[k]=v,true)});
 const node=(id,attrs='')=>({id,value:attrs.match(/value="([^"]*)"/)?.[1]||'',type:attrs.match(/type="([^"]*)"/)?.[1],checked:/\bchecked\b/.test(attrs),dataset:{unit:attrs.match(/data-unit="([^"]*)"/)?.[1]||''},listeners:{},clientWidth:760,clientHeight:340,addEventListener(k,f){this.listeners[k]=f;},setAttribute(){},getContext(){return ctx;},click(){this.onclick?.();},textContent:'',innerHTML:''});
 const get=id=>{if(!nodes.has(id))throw Error('Missing control: '+id);return nodes.get(id);};
 for(const id of ['lab','lab-title','insight'])nodes.set(id,node(id));
 Object.defineProperty(get('lab'),'innerHTML',{set(html){for(const match of html.matchAll(/<(\w+)\b([^>]*\bid="([^"]+)"[^>]*)>/g))nodes.set(match[3],node(match[3],match[2]));for(const m of html.matchAll(/data-preset="([^"]+)"/g)){const b=node(m[1]);b.dataset.preset=m[1];presets.push(b);}}});
 const document={getElementById:id=>nodes.get(id)||null,querySelectorAll:()=>presets,documentElement:{dataset:{},classList:{toggle(){}}},addEventListener(k,f){events[k]=f;},createElement:()=>node('download'),hidden:false};
 const sandbox={document,location:{search:'?demo='+demo+'&embed=region'},URLSearchParams,devicePixelRatio:1,ResizeObserver:class{constructor(f){frames.push(f);}observe(){}},requestAnimationFrame:f=>frames.push(f),addEventListener:(k,f)=>events[k]=f,setInterval:f=>{timers.set(++timerId,f);return timerId;},clearInterval:id=>timers.delete(id),setTimeout:()=>{},Blob,URL:{createObjectURL:()=>'',revokeObjectURL(){}},console};sandbox.window=sandbox;vm.createContext(sandbox);
 for(const f of ['model.js','live/labs.js','live/render.js','live/app.js'])vm.runInContext(fs.readFileSync(__dirname+'/'+f,'utf8'),sandbox,{filename:f});frames.forEach(f=>f());
 return {state:()=>sandbox.radarDebug.getState(),input(id,value){const el=get(id);el.type==='checkbox'?el.checked=value:el.value=String(value);el.listeners.input();},click:id=>get(id).click(),preset:name=>presets.find(b=>b.dataset.preset===name).click(),tick(n=1){for(let i=0;i<n;i++)[...timers.values()].forEach(f=>f());},pause(value){document.documentElement.dataset.bentoPaused=String(value);events['bento-live-visibility']({detail:{paused:value}});},nodes,timers};
}
test('mission controls scrub, stop at the endpoint, and pause/resume hidden animation',()=>{
 const lab=mount('whole-run');assert.equal(lab.state().frame,24);lab.preset('start');lab.click('mission-play');lab.tick(2);assert.equal(lab.state().frame,2);lab.pause(true);lab.tick(4);assert.equal(lab.state().frame,2);lab.pause(false);lab.tick();assert.equal(lab.state().frame,3);lab.preset('loop');lab.click('mission-step');assert.equal(lab.state().frame,48);lab.input('mission-bias',false);assert.equal(lab.state().bias,0);lab.click('mission-reset');assert.equal(lab.state().frame,0);
});
test('signal, detection, coordinate and velocity presets actually change model inputs',()=>{
 const range=mount('range');range.preset('merged');const low=range.state().resolution;range.preset('resolved');assert.equal(low/range.state().resolution,8);range.input('range-hann',false);range.input('range-sep',1.2);assert.equal(range.state().separation,1.2);
 const cfar=mount('cfar');cfar.preset('edge');assert.equal(cfar.state().cell,74);cfar.click('cfar-new');assert.equal(cfar.state().seed,13);cfar.click('cfar-reset');assert.equal(cfar.state().seed,12);cfar.input('cfar-cell',103);assert.equal(cfar.state().cell,103);
 const frames=mount('frames');frames.preset('identity');assert.equal(JSON.stringify(frames.state().pose),'[0,0,0]');frames.preset('quarter');assert.equal(frames.state().pose[2],Math.PI/2);
 const v=mount('velocity');v.preset('narrow');assert.ok(v.state().condition>100);v.preset('moving');v.input('velocity-robust',false);assert.equal(JSON.stringify(v.state().fit),JSON.stringify(v.state().ordinary));
});
test('ICP has distinct association/update stages and reset stops its animation',()=>{
 const lab=mount('icp');lab.click('icp-step');assert.equal(lab.state().phase,1);assert.equal(lab.state().iterations,0);lab.click('icp-step');assert.equal(lab.state().phase,0);assert.equal(lab.state().iterations,1);lab.click('icp-run');lab.tick(40);assert.ok(lab.state().history.at(-1)<lab.state().history[0]);lab.preset('poor');assert.equal(lab.state().iterations,0);assert.equal(lab.timers.size,0);lab.preset('tight');lab.click('icp-step');lab.click('icp-step');
});
test('graph animation converges, presets reset the objective, and export is available',()=>{
 const lab=mount('optimize');const initial=lab.state().cost;lab.click('graph-solve');lab.tick(25);assert.ok(lab.state().cost<initial);assert.ok(lab.state().rmse<.1);lab.preset('false');assert.equal(lab.state().iterations,0);lab.click('graph-step');assert.ok(lab.state().cost<=lab.state().initialCost);lab.preset('robust');lab.click('graph-solve');lab.pause(true);lab.tick(4);assert.equal(lab.state().iterations,0);lab.pause(false);lab.tick(25);lab.click('graph-export');lab.click('graph-reset');assert.equal(lab.state().iterations,0);
});
test('frame bridge preserves control keys and combines slide/tab visibility',()=>{
 const sent=[],events={},docEvents={},data={},parent={postMessage:m=>sent.push(m)};const window={parent,addEventListener:(k,f)=>events[k]=f,dispatchEvent:()=>{}};
 const document={hidden:false,readyState:'complete',documentElement:{dataset:data},addEventListener:(k,f)=>docEvents[k]=f};
 const c={window,document,CustomEvent:class{},requestAnimationFrame:f=>f()};vm.createContext(c);vm.runInContext(fs.readFileSync(__dirname+'/live/frame.js','utf8'),c);
 events.message({source:parent,data:{type:'bento-live-pause'}});document.hidden=true;docEvents.visibilitychange();document.hidden=false;docEvents.visibilitychange();assert.equal(data.bentoPaused,'true');
 events.message({source:parent,data:{type:'bento-live-resume'}});assert.equal(data.bentoPaused,'false');
 const key=(key,control)=>events.keydown({key,target:{closest:()=>control},preventDefault(){this.defaultPrevented=true;}});
 const n=sent.length;key('ArrowRight',true);assert.equal(sent.length,n);key('PageDown',true);assert.equal(sent.at(-1).type,'bento-inline-nav');assert.equal(sent.at(-1).direction,1);key('Escape',true);assert.equal(sent.at(-1).type,'bento-inline-focus');
});
