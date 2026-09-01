/* Progressive enhancement: all material remains readable without JavaScript. */
(() => {
  'use strict';
  const body=document.body;
  const frames=Array.from(document.querySelectorAll('.frame'));
  const aliases=JSON.parse(document.getElementById('legacy-anchors').textContent);
  const slides=body.dataset.view==='slides'||/consolidated-slides\.html$/.test(location.pathname)||new URLSearchParams(location.search).get('view')==='slides';
  body.dataset.view=slides?'slides':'read';
  document.querySelectorAll('[data-view-link]').forEach(a=>{if(a.dataset.viewLink===body.dataset.view)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});
  const select=document.getElementById('slide-select');
  const previous=document.getElementById('previous-slide');
  const next=document.getElementById('next-slide');
  const pair=document.getElementById('pair-shortcut');
  let current=0;
  function targetFor(hash){
    let id;try{id=decodeURIComponent(hash.replace(/^#/,''));}catch{return frames[0];}
    id=aliases[id]||id;
    const target=document.getElementById(id);
    return target?.closest('.frame')||frames[0];
  }
  function syncViewLinks(id){
    document.querySelectorAll('[data-view-link]').forEach(a=>{a.hash=id;});
  }
  function show(index,updateHash=true){
    current=Math.max(0,Math.min(frames.length-1,index));
    frames.forEach((frame,i)=>{frame.hidden=i!==current;});
    const frame=frames[current];
    select.value=frame.id;
    previous.disabled=current===0;next.disabled=current===frames.length-1;
    pair.hidden=!frame.dataset.pair;
    if(frame.dataset.pair){pair.href='#'+frame.dataset.pair;pair.textContent=frame.classList.contains('idea')?'Equation sheet':'Key idea';}
    document.getElementById('slide-progress').style.width=`${100*(current+1)/frames.length}%`;
    document.getElementById('slide-announcement').textContent=`Slide ${current+1} of ${frames.length}. ${frame.dataset.label}${frame.classList.contains('equation-sheet')?', equation summary':''}.`;
    syncViewLinks(frame.id);
    if(updateHash&&location.hash!=='#'+frame.id){history.pushState(null,'','#'+frame.id);}
    window.scrollTo({top:0,behavior:'instant'});
    document.documentElement.dataset.slide=String(current+1);
    requestAnimationFrame(labelEquations);
  }
  if(slides){
    body.classList.add('presentation');
    document.querySelector('.slide-controls').hidden=false;
    show(frames.indexOf(targetFor(location.hash)),false);
    previous.addEventListener('click',()=>show(current-1));
    next.addEventListener('click',()=>show(current+1));
    select.addEventListener('change',()=>show(frames.indexOf(document.getElementById(select.value))));
    document.addEventListener('click',event=>{
      const a=event.target.closest('a[href^="#"]');
      if(!a)return;
      const target=targetFor(a.hash);
      if(target){event.preventDefault();show(frames.indexOf(target));}
    });
    window.addEventListener('hashchange',()=>show(frames.indexOf(targetFor(location.hash)),false));
    window.addEventListener('popstate',()=>show(frames.indexOf(targetFor(location.hash)),false));
    document.addEventListener('keydown',event=>{
      if(event.altKey||event.ctrlKey||event.metaKey||event.target.closest('input,textarea,select,button,a,summary,[contenteditable="true"]'))return;
      if(event.code==='ArrowRight'||event.code==='PageDown'||event.code==='Space'&&!event.shiftKey){event.preventDefault();show(current+1);}
      else if(event.code==='ArrowLeft'||event.code==='PageUp'||event.code==='Space'&&event.shiftKey){event.preventDefault();show(current-1);}
      else if(event.code==='Home'){event.preventDefault();show(0);}
      else if(event.code==='End'){event.preventDefault();show(frames.length-1);}
    });
  }else{
    if(location.hash&&aliases[location.hash.slice(1)]){location.hash=aliases[location.hash.slice(1)];}
    const navLinks=Array.from(document.querySelectorAll('.sidebar nav a'));
    let pending=false;
    const mark=()=>{
      let active=frames[0];for(const f of frames){if(f.getBoundingClientRect().top<150)active=f;}
      navLinks.forEach(a=>{const on=a.hash==='#'+active.id;a.classList.toggle('active',on);if(on)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current');});
      syncViewLinks(active.id);pending=false;
    };
    window.addEventListener('scroll',()=>{if(!pending){requestAnimationFrame(mark);pending=true;}},{passive:true});
    window.addEventListener('resize',mark);window.addEventListener('math-ready',mark);mark();
    document.querySelectorAll('.mobile-index a').forEach(a=>a.addEventListener('click',()=>{a.closest('details').open=false;}));
  }
  // Tell keyboard users when an equation box can be scrolled horizontally.
  function labelEquations(){
    document.querySelectorAll('.eq').forEach((e,i)=>{
      if(e.scrollWidth>e.clientWidth+1){e.tabIndex=0;e.setAttribute('role','region');e.setAttribute('aria-label',`Equation ${i+1}; scroll horizontally for the full expression`);}
      else{e.removeAttribute('tabindex');e.removeAttribute('role');e.removeAttribute('aria-label');}
    });
  }
  window.addEventListener('math-ready',labelEquations);window.addEventListener('resize',labelEquations);
  select.addEventListener('change',labelEquations);
  document.querySelectorAll('.print-sheets').forEach(button=>button.addEventListener('click',async()=>{
    if(window.MathJax?.startup?.promise)await MathJax.startup.promise;
    body.classList.add('print-equations');window.print();
  }));
  window.addEventListener('afterprint',()=>body.classList.remove('print-equations'));
  // The previous page's direct links still find the consolidated family.
  window.addEventListener('math-ready',()=>{
    if(!slides&&location.hash){targetFor(location.hash).scrollIntoView({block:'start'});}
  });
  document.documentElement.dataset.uiReady='true';
})();
