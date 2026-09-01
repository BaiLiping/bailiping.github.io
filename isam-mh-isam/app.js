/* Lightweight, responsive slide navigation. No build step or framework required. */
(function(){
'use strict';
const slides=window.ISAMSlides,$=id=>document.getElementById(id);let current=0;
const escapeHTML=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
$('stage').innerHTML=slides.map((s,i)=>`<section class="slide" id="slide-${s.id}" aria-labelledby="title-${s.id}"><header class="slide-head"><div class="kicker">${s.section}</div><h${i===0?'1':'2'} id="title-${s.id}">${s.title}</h${i===0?'1':'2'}><p>${s.subtitle}</p></header><div class="slide-body ${s.layout}">${s.body}</div><footer class="source"><span>${s.source}</span><span class="page">${String(i+1).padStart(2,'0')} / ${slides.length} · Bai Liping</span></footer></section>`).join('');
$('overview-links').innerHTML=slides.map((s,i)=>`<a href="#${s.id}"><small>${String(i+1).padStart(2,'0')} · ${s.lab?'LAB':s.section}</small>${s.title}</a>`).join('');
function show(index){
  current=Math.max(0,Math.min(slides.length-1,index));
  document.querySelectorAll('.slide').forEach((el,i)=>{el.classList.toggle('active',i===current);el.setAttribute('aria-hidden',String(i!==current));el.inert=i!==current;});
  const s=slides[current];$('counter').textContent=`${String(current+1).padStart(2,'0')} / ${slides.length}`;$('current-title').textContent=s.title;$('progress-fill').style.width=(current+1)/slides.length*100+'%';$('previous').disabled=current===0;$('next').disabled=current===slides.length-1;
  $('notes-content').innerHTML=`<h3>${escapeHTML(s.title)}</h3><p>${escapeHTML(s.notes)}</p><p>${s.source}</p>`;
  document.title=`${s.title} · iSAM & MH-iSAM2 | Bai Liping`;window.scrollTo(0,0);
}
function route(){let hash;try{hash=decodeURIComponent(location.hash.replace(/^#\/?/,''));}catch{hash='overview';}let index=slides.findIndex(s=>s.id===hash);if(index<0&&/^\d+$/.test(hash))index=Math.max(0,Math.min(slides.length-1,Number(hash)));show(index<0?0:index);}
function go(index){const next=Math.max(0,Math.min(slides.length-1,index));if(next===current)return;location.hash=slides[next].id;}
$('previous').onclick=()=>go(current-1);$('next').onclick=()=>go(current+1);addEventListener('hashchange',route);
$('overview-open').onclick=()=>$('overview').showModal();$('notes-open').onclick=()=>$('notes').showModal();
document.querySelectorAll('[data-close]').forEach(b=>b.onclick=()=>$(b.dataset.close).close());
$('overview-links').addEventListener('click',e=>{if(e.target.closest('a'))$('overview').close();});
async function fullscreen(){try{if(document.fullscreenElement)await document.exitFullscreen();else if(document.documentElement.requestFullscreen)await document.documentElement.requestFullscreen();}catch{ $('fullscreen').textContent='Use browser fullscreen';}}
$('fullscreen').onclick=fullscreen;addEventListener('fullscreenchange',()=>{$('fullscreen').textContent=document.fullscreenElement?'Exit fullscreen':'Fullscreen';});
addEventListener('keydown',e=>{
  if(e.ctrlKey||e.metaKey||e.altKey||e.defaultPrevented||document.querySelector('dialog[open]')||e.target.closest('input,select,textarea,[contenteditable=true]'))return;
  if(['ArrowRight','PageDown'].includes(e.key)){e.preventDefault();go(current+1);}else if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();go(current-1);}else if(e.code==='Space'&&!e.target.closest('button,a')){e.preventDefault();go(current+(e.shiftKey?-1:1));}else if(e.key==='Home'){e.preventDefault();go(0);}else if(e.key==='End'){e.preventDefault();go(slides.length-1);}else if(e.key.toLowerCase()==='o')$('overview').showModal();else if(e.key.toLowerCase()==='n')$('notes').showModal();else if(e.key.toLowerCase()==='f')fullscreen();
});
let start=null;$('stage').addEventListener('touchstart',e=>{if(e.touches.length===1&&!e.target.closest('button,input,select,a,svg,.equation,.table-wrap'))start=[e.touches[0].clientX,e.touches[0].clientY];else start=null;},{passive:true});
$('stage').addEventListener('touchend',e=>{if(!start)return;const dx=e.changedTouches[0].clientX-start[0],dy=e.changedTouches[0].clientY-start[1];start=null;if(Math.abs(dx)>90&&Math.abs(dx)>2*Math.abs(dy))go(current+(dx<0?1:-1));},{passive:true});
// All equations are typeset once, including slides not currently visible.
let attempts=0;
function typeset(){if(window.MathJax?.startup?.promise){MathJax.startup.promise.then(()=>MathJax.typesetPromise([...document.querySelectorAll('.equation')])).then(()=>{window.isamMathReady=true;}).catch(mathFailure);}else if(attempts++<150)setTimeout(typeset,200);else mathFailure();}
function mathFailure(){const note=document.createElement('p');note.className='math-warning';note.textContent='Equation renderer unavailable: LaTeX is shown. The interactive demos still work.';$('stage').prepend(note);}
window.initISAMDemos();route();typeset();window.ISAMDeck={slides,go,show,get current(){return current;}};
// Print every slide; inactive slides must not remain inert in the print tree.
addEventListener('beforeprint',()=>document.querySelectorAll('.slide').forEach(el=>{el.inert=false;el.removeAttribute('aria-hidden');}));
addEventListener('afterprint',()=>show(current));
})();
