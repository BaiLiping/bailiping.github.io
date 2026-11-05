(() => {
  'use strict';
  const slides = [...document.querySelectorAll('.slide')];
  const stage = document.querySelector('.stage');
  const controls = document.querySelector('.deck-controls');
  const notes = document.querySelector('.notes-dialog');
  const overview = document.querySelector('.overview-dialog');
  const previous = document.querySelector('[data-prev]');
  const next = document.querySelector('[data-next]');
  const grid = document.querySelector('.overview-grid');
  let current = 0;
  let returnFocus = null;

  function fit() {
    const availableHeight = Math.max(120, innerHeight - controls.offsetHeight - 30);
    stage.style.top = availableHeight / 2 + 8 + 'px';
    stage.style.transform = 'translate(-50%,-50%) scale(' + Math.min((innerWidth-16)/1280, availableHeight/720) + ')';
  }
  function show(index, updateHash=true) {
    current = Math.min(slides.length-1, Math.max(0,index));
    slides.forEach((slide,i) => {
      slide.classList.toggle('active', i===current);
      slide.setAttribute('aria-hidden', String(i!==current));
      slide.inert = i!==current;
    });
    document.querySelector('.counter').textContent = (current+1) + ' / ' + slides.length;
    notes.querySelector('p').textContent = slides[current].querySelector('.speaker-notes').textContent;
    previous.disabled = current===0;
    next.disabled = current===slides.length-1;
    [...grid.children].forEach((button,i) => {
      button.classList.toggle('active',i===current);
      button.setAttribute('aria-current',i===current?'step':'false');
    });
    if (updateHash) history.replaceState(null,'','#'+slides[current].id);
  }
  function indexFromHash() {
    const hash=location.hash.slice(1), found=slides.findIndex(s=>s.id===hash);
    if(found>=0)return found;
    const oldHash=/^slide-(\d+)$/.exec(hash);
    return oldHash?Number(oldHash[1])-1:0;
  }
  function open(dialog,trigger) {
    returnFocus=trigger || document.activeElement;
    dialog.showModal();
    if(dialog===overview) grid.children[current]?.focus();
    else dialog.querySelector('button').focus();
  }
  function toggle(dialog,trigger) {
    if(dialog.open)dialog.close();else open(dialog,trigger);
  }
  [notes,overview].forEach(dialog=>dialog.addEventListener('close',()=>{
    returnFocus?.focus({preventScroll:true});
  }));
  document.querySelector('[data-notes]').onclick = e=>toggle(notes,e.currentTarget);
  document.querySelector('[data-overview]').onclick = e=>toggle(overview,e.currentTarget);
  document.querySelector('[data-close-notes]').onclick = ()=>notes.close();
  document.querySelector('[data-close-overview]').onclick = ()=>overview.close();
  previous.onclick=()=>show(current-1);
  next.onclick=()=>show(current+1);
  document.querySelector('[data-print]').onclick=()=>print();
  function fullscreen() {
    if(document.fullscreenElement)document.exitFullscreen?.();
    else document.documentElement.requestFullscreen?.().catch(()=>{});
  }
  document.querySelector('[data-full]').onclick=fullscreen;
  slides.forEach((slide,i)=>{
    const button=document.createElement('button');
    const label=document.createElement('strong');
    label.textContent=String(i+1).padStart(2,'0')+' · '+slide.querySelector('.kicker').textContent;
    button.append(label,document.createTextNode(slide.querySelector('h1').textContent));
    button.onclick=()=>{show(i);overview.close();};
    grid.append(button);
  });
  document.addEventListener('click',e=>{
    const a=e.target.closest('a[href^="#"]');
    if(!a)return;
    const i=slides.findIndex(s=>s.id===a.getAttribute('href').slice(1));
    if(i<0)return;
    e.preventDefault();show(i);
  });
  document.addEventListener('keydown',e=>{
    if(notes.open || overview.open)return;
    if(/INPUT|SELECT|TEXTAREA/.test(e.target.tagName))return;
    if(e.key===' ' && /BUTTON|A/.test(e.target.tagName))return;
    const key=e.key.toLowerCase();
    if(['arrowright','arrowdown','pagedown',' '].includes(key)){e.preventDefault();show(current+1);}
    else if(['arrowleft','arrowup','pageup'].includes(key)){e.preventDefault();show(current-1);}
    else if(key==='home'){e.preventDefault();show(0);}
    else if(key==='end'){e.preventDefault();show(slides.length-1);}
    else if(key==='n')open(notes,document.querySelector('[data-notes]'));
    else if(key==='o')open(overview,document.querySelector('[data-overview]'));
    else if(key==='f')fullscreen();
  });
  addEventListener('resize',fit);
  addEventListener('hashchange',()=>show(indexFromHash(),false));
  new ResizeObserver(fit).observe(controls);
  window.EOSlides={show,count:slides.length,get current(){return current;}};
  show(indexFromHash());fit();
})();
