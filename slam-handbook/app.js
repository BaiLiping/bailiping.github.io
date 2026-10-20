/* Open, self-contained teaching interface. All experiments run locally. */
(function () {
"use strict";
const chapters=(window.SLAM_CHAPTERS||[]).slice().sort((a,b)=>a.n-b.n);
const $=id=>document.getElementById(id);
const safe=x=>String(x===undefined||x===null?"":x).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#39;");
const SOURCE="https://asrl.utias.utoronto.ca/~tdb/slam/slamhandbook.pdf";
const partNames=["I · Foundations of SLAM","II · SLAM in practice","III · From SLAM to Spatial AI"];
const part=n=>n<=6?0:n<=12?1:2,zero=n=>String(n).padStart(2,"0");
const STORAGE="slam-handbook-learning-guide-v1";
const state={n:1,done:{},notes:{},nav:false};
try{const v=JSON.parse(localStorage.getItem(STORAGE)||"{}");if(v&&typeof v==="object"){state.done=v.done||{};state.notes=v.notes||{}}}catch(e){console.warn("No readable saved progress",e)}
function store(){try{localStorage.setItem(STORAGE,JSON.stringify({done:state.done,notes:state.notes}))}catch(e){console.warn("Local storage unavailable",e)}}
const complete=()=>chapters.filter(c=>state.done[c.n]).length;
function toast(message){$("status").textContent=message;setTimeout(()=>$("status").textContent="",3500)}
function navRender(){
 const q=$("chapterSearch").value.trim().toLowerCase();let html="";
 for(let i=0;i<3;i++){
  const subset=chapters.filter(c=>part(c.n)===i&&(c.title+" "+c.goal+" "+c.n).toLowerCase().includes(q));
  if(!subset.length)continue;
  html+='<section class="nav-part"><h3>'+safe(partNames[i])+'</h3>';
  for(const c of subset){
   html+='<a href="#chapter-'+c.n+'" data-chapter="'+c.n+'" class="'+(state.n===c.n?"selected":"")+'" '+(state.n===c.n?'aria-current="page"':'')+'><span class="num">'+zero(c.n)+'</span><span>'+safe(c.title)+'</span><span class="done-mark">'+(state.done[c.n]?"✓":"")+'</span></a>';
  }html+="</section>";
 }
 $("chapterList").innerHTML=html||'<p class="empty-nav">No matching chapters.</p>';
 $("progressMini").textContent=complete()+"/"+chapters.length+" chapters studied";
}
function format(val,spec){return Number(val).toLocaleString(undefined,{maximumFractionDigits:3})+(spec.unit?" "+spec.unit:"")}
function svgPlot(p,id){
 const all=(p.series||[]).map((s,i)=>({name:s.name,color:["#226b52","#3c70aa","#b26f3a","#875d9f","#9d4f59","#597b88"][i%6],points:(s.points||[]).filter(x=>x&&Number.isFinite(Number(x[0]))&&Number.isFinite(Number(x[1]))).map(x=>[Number(x[0]),Number(x[1])])})).filter(s=>s.points.length);
 if(!all.length)return '<p role="alert">No finite curve to draw for the chosen parameters.</p>';
 const points=all.flatMap(s=>s.points),L=67,T=21,R=20,B=59,W=760,H=340,w=W-L-R,h=H-T-B;
 let xmin=Math.min(...points.map(a=>a[0])),xmax=Math.max(...points.map(a=>a[0]));
 let ymin=Math.min(...points.map(a=>a[1])),ymax=Math.max(...points.map(a=>a[1]));
 function expand(a,b){const d=b-a;return d<1e-12?[a-.5,b+.5]:[a-d*.045,b+d*.045]}
 [xmin,xmax]=expand(xmin,xmax);[ymin,ymax]=expand(ymin,ymax);
 const X=x=>L+(x-xmin)*w/(xmax-xmin),Y=y=>T+h-(y-ymin)*h/(ymax-ymin);
 const fmt=n=>Number(Math.abs(n)>100?n.toFixed(0):Math.abs(n)>10?n.toFixed(1):n.toFixed(2)).toString();
 let html='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '+W+' '+H+'" role="img" aria-labelledby="'+id+'-title '+id+'-description">'
  +'<title id="'+id+'-title">'+safe(p.yLabel)+' plotted against '+safe(p.xLabel)+'</title>'
  +'<desc id="'+id+'-description">Interactive plot of '+safe(all.map(s=>s.name).join(", "))+'. Adjust the controls for updated values.</desc>'
  +'<rect x="'+L+'" y="'+T+'" width="'+w+'" height="'+h+'" fill="#fff" stroke="#c8d8ce"/>';
 for(let j=0;j<=5;j++){
  const gx=L+j*w/5,gy=T+j*h/5;
  html+='<path d="M'+gx+' '+T+'V'+(T+h)+' M'+L+' '+gy+'H'+(L+w)+'" fill="none" stroke="#e7ece6"/>'
   +'<text x="'+gx+'" y="'+(T+h+19)+'" text-anchor="middle">'+safe(fmt(xmin+j*(xmax-xmin)/5))+'</text>'
   +'<text x="'+(L-8)+'" y="'+(gy+4)+'" text-anchor="end">'+safe(fmt(ymax-j*(ymax-ymin)/5))+'</text>';
 }
 html+='<text class="plot-label" x="'+(L+w/2)+'" y="'+(H-12)+'" text-anchor="middle">'+safe(p.xLabel)+'</text>'
  +'<text class="plot-label" transform="translate(18 '+(T+h/2)+') rotate(-90)" text-anchor="middle">'+safe(p.yLabel)+'</text>';
 for(const s of all){
  if(s.points.length>1){
   html+='<polyline fill="none" stroke="'+s.color+'" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round" points="'+s.points.map(q=>X(q[0]).toFixed(2)+","+Y(q[1]).toFixed(2)).join(" ")+'"/>';
  }
  if(s.points.length===1||s.points.length<=12)for(const q of s.points)html+='<circle cx="'+X(q[0]).toFixed(2)+'" cy="'+Y(q[1]).toFixed(2)+'" r="'+(s.points.length===1?5:2.8)+'" fill="'+s.color+'" stroke="#fff" stroke-width="1.1"/>';
 }
 html+='</svg><div class="plot-key">'+all.map(s=>'<span class="key-item"><span class="key-swatch" style="border-color:'+s.color+'"></span>'+safe(s.name)+'</span>').join("")+'</div>';
 return html;
}
function experiment(c,d,index){
 const prefix="ch"+c.n+"-e"+index;
 const controls=d.params.map(s=>{
  const id=prefix+"-"+s.key;
  return '<div class="control"><div class="row"><label for="'+safe(id)+'">'+safe(s.label)+'</label><output id="'+safe(id)+'-value" for="'+safe(id)+'">'+safe(format(s.value,s))+'</output></div>'
   +'<input id="'+safe(id)+'" data-param="'+safe(s.key)+'" type="range" min="'+safe(s.min)+'" max="'+safe(s.max)+'" step="'+safe(s.step)+'" value="'+safe(s.value)+'" aria-label="'+safe(s.label)+'">'
   +'<div class="endpoints"><span>'+safe(format(s.min,s))+'</span><span>'+safe(format(s.max,s))+'</span></div></div>';
 }).join("");
 const page=Number(c.pages.match(/^\d+/)?.[0]||0)+16;
 return '<section class="experiment" id="experiment-'+index+'" data-exp-index="'+index+'">'
  +'<div class="exp-head"><h3>'+safe(d.title)+'</h3><span class="exp-no">Experiment '+index+' of '+c.demos.length+'</span></div>'
  +'<p class="exp-question">'+safe(d.question)+'</p><div class="lab-grid">'
  +'<div class="figurebox"><div id="'+prefix+'-figure" class="plot-surface">Loading numerical model…</div></div>'
  +'<div class="controls">'+controls+'<button type="button" class="reset-exp">Reset experiment</button><p class="control-note">The calculation and chart update immediately as you move a slider.</p></div></div>'
  +'<div class="metrics" id="'+prefix+'-metrics" aria-live="polite"></div>'
  +'<p class="insight" id="'+prefix+'-insight"></p>'
  +'<p class="exp-prompt"><strong>Try this:</strong> '+safe(d.tryThis)+'</p>'
  +'<p class="exp-limit"><b>Model boundary:</b> '+safe(d.limit)+'</p>'
  +'<a class="exp-citation" href="'+SOURCE+'#page='+page+'" target="_blank" rel="noopener noreferrer">Handbook '+safe(d.ref)+' (chapter '+c.n+', printed pp. '+safe(c.pages)+') ↗</a>'
  +'</section>';
}
function recalc(c,index){
 const d=c.demos[index-1],prefix="ch"+c.n+"-e"+index,box=document.querySelector('[data-exp-index="'+index+'"]');
 if(!box)return;
 const params={};
 for(const s of d.params){
  const el=box.querySelector('[data-param="'+s.key+'"]');
  params[s.key]=Number(el.value);$(el.id+"-value").textContent=format(params[s.key],s);
 }
 try{
  const m=window.SLAM_MODELS?.[d.key];
  if(typeof m!=="function")throw Error("Numerical model "+d.key+" is unavailable");
  const r=m(params);
  if(!r || !r.plot || !Array.isArray(r.metrics))throw Error("The model returned an invalid result.");
  $(prefix+"-figure").innerHTML=svgPlot(r.plot,prefix);
  $(prefix+"-metrics").innerHTML=r.metrics.map(m=>'<div class="metric"><small>'+safe(m[0])+'</small><strong>'+safe(m[1])+'</strong></div>').join("");
  $(prefix+"-insight").textContent=r.insight||"";
 }catch(e){
  console.error("Experiment error",d.key,e);
  $(prefix+"-figure").innerHTML='<p role="alert">This numerical experiment could not compute. Try resetting its settings.</p>';
  $(prefix+"-metrics").innerHTML="";
  $(prefix+"-insight").textContent=String(e.message||e);
 }
}
function closeNav(){state.nav=false;$("chapterNav").classList.remove("open");$("navToggle").setAttribute("aria-expanded","false")}
function render(n,scroll){
 const c=chapters.find(x=>x.n===n);if(!c)return;
 state.n=n;document.title="Chapter "+n+": "+c.title+" | SLAM Handbook";
 const ideas=c.ideas.map(a=>'<section class="idea"><h3>'+safe(a.title)+'</h3><p>'+safe(a.text)+'</p><div class="formula">'+safe(a.equation)+'</div><p class="source-ref">Handbook '+safe(a.ref)+'</p></section>').join("");
 const checks=c.checks.map((q,i)=>'<details><summary>'+(i+1)+'. '+safe(q.q)+'</summary><p>'+safe(q.a)+'</p></details>').join("");
 const p=chapters.find(x=>x.n===n-1),next=chapters.find(x=>x.n===n+1),bookpage=Number(c.pages.match(/^\d+/)?.[0]||0)+16;
 $("lesson").innerHTML='<div class="lesson"><p class="chapter-meta">Part '+safe(partNames[part(n)])+' · Chapter '+zero(n)+' · pp. '+safe(c.pages)+' · '+c.demos.length+' experiments</p>'
  +'<h1>'+safe(c.title)+'</h1><p class="chapter-goal">'+safe(c.goal)+'</p>'
  +'<div class="chapter-jumps"><a href="#ideas">Read the idea</a><a href="#experiments">Experiment</a><a href="#check-yourself">Check yourself</a><a href="#notebook">Notebook</a></div>'
  +'<section class="chapter-section" id="ideas"><h2>Core ideas</h2><p>Read these original explanations with <a href="'+SOURCE+'#page='+bookpage+'" target="_blank" rel="noopener noreferrer">chapter '+n+' of the handbook ↗</a>. Each idea identifies the relevant source section and printed page range.</p><div class="ideas">'+ideas+'</div></section>'
  +'<section class="chapter-section" id="experiments"><h2>Explore the ideas</h2><p>Predict, change a parameter and compare the recomputed curves and metrics. These are simplified numerical teaching models, not production SLAM algorithms.</p>'
  +c.demos.map((d,i)=>experiment(c,d,i+1)).join("")+'</section>'
  +'<section class="chapter-section" id="check-yourself"><h2>Check your understanding</h2><p>Try to answer each question before opening its explanation.</p><div class="selfchecks">'+checks+'</div><p class="exp-limit"><b>Important qualification:</b> '+safe(c.watch)+'</p></section>'
  +'<section class="chapter-section notebook" id="notebook"><h2>Reading notebook</h2><label for="chapterNote">What did you predict, observe and learn?</label>'
  +'<textarea id="chapterNote" placeholder="My prediction…\nWhat the experiment showed…\nA question to revisit…">'+safe(state.notes[n]||"")+'</textarea>'
  +'<p id="noteStatus" class="note-status">Notes stay in this browser; export your notebook to keep a backup.</p>'
  +'<button id="completeChapter" class="complete-btn'+(state.done[n]?" done":"")+'">'+(state.done[n]?"✓ Studied — mark incomplete":"Mark chapter studied")+'</button>'
  +'<span class="note-status">'+complete()+' of '+chapters.length+' chapters studied</span></section>'
  +'<nav class="prev-next" aria-label="Previous and next chapters"><span>'+(p?'<a href="#chapter-'+p.n+'">← '+safe(p.title)+'</a>':"")+'</span><span>'+(next?'<a href="#chapter-'+next.n+'">'+safe(next.title)+' →</a>':"")+'</span></nav>'
  +'</div>';
 c.demos.forEach((d,i)=>recalc(c,i+1));
 navRender();closeNav();
 if(scroll){window.scrollTo(0,0);$("lesson").focus({preventScroll:true})}
}
function chosenChapter(){const x=location.hash.match(/^#chapter-(\d+)$/);return x?Number(x[1]):state.n}
function exportNotes(){
 const lines=["# SLAM Handbook: personal reading notebook","",new Date().toISOString(),"Studied "+complete()+" / "+chapters.length+" chapters",""];
 chapters.forEach(c=>lines.push("## "+zero(c.n)+" "+c.title+" (pp. "+c.pages+")",state.done[c.n]?"Status: studied":"Status: in progress","",String(state.notes[c.n]||"").trim()||"(No notes)",""));
 const blob=new Blob([lines.join("\n")],{type:"text/markdown;charset=utf-8"}),url=URL.createObjectURL(blob),link=document.createElement("a");
 link.href=url;link.download="slam-handbook-notebook.md";document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),2000);toast("Notebook exported");
}
function events(){
 $("chapterSearch").addEventListener("input",navRender);
 $("notesExport").addEventListener("click",exportNotes);
 $("navToggle").addEventListener("click",()=>{
  state.nav=!state.nav;$("chapterNav").classList.toggle("open",state.nav);
  $("navToggle").setAttribute("aria-expanded",String(state.nav));
  if(state.nav)$("chapterSearch").focus();
 });
 $("resetProgress").addEventListener("click",()=>{
  if(confirm("Erase all saved notes and chapter progress in this browser?")){state.done={};state.notes={};store();render(state.n,false);toast("All progress cleared")}
 });
 $("lesson").addEventListener("input",e=>{
  const input=e.target;if(input.matches('input[data-param]')){
   const exp=input.closest("[data-exp-index]");recalc(chapters.find(c=>c.n===state.n),Number(exp.dataset.expIndex));
  }else if(input.id==="chapterNote"){
   state.notes[state.n]=input.value;store();$("noteStatus").textContent="Saved to this browser.";
  }
 });
 $("lesson").addEventListener("click",e=>{
  const reset=e.target.closest(".reset-exp");
  if(reset){
   const exp=reset.closest("[data-exp-index]"),ix=Number(exp.dataset.expIndex),c=chapters.find(x=>x.n===state.n);
   c.demos[ix-1].params.forEach(s=>exp.querySelector('[data-param="'+s.key+'"]').value=s.value);
   recalc(c,ix);toast("Experiment reset");
  }
  if(e.target.id==="completeChapter"){state.done[state.n]=!state.done[state.n];store();render(state.n,false);toast(state.done[state.n]?"Chapter marked studied":"Chapter marked incomplete")}
 });
 $("chapterList").addEventListener("click",e=>{if(e.target.closest('[data-chapter="'+state.n+'"]'))closeNav()});
 window.addEventListener("hashchange",()=>{const n=chosenChapter();if(n!==state.n)render(n,true)});
 document.addEventListener("keydown",e=>{if(e.key==="Escape"&&state.nav)closeNav()});
}
function start(){
 if(chapters.length!==18||chapters.some(c=>c.ideas?.length<3||c.demos?.length<3||c.checks?.length<2)){
  $("lesson").innerHTML='<p role="alert">Chapter content is incomplete. Check the files and reload.</p>';return;
 }
 events();render(Math.min(18,Math.max(1,chosenChapter())),false);
 if("serviceWorker" in navigator)window.addEventListener("load",()=>navigator.serviceWorker.register("./service-worker.js").catch(e=>console.warn("Offline caching unavailable",e)));
}
Promise.allSettled(window.SLAM_LOADS||[]).then(start);
})();
