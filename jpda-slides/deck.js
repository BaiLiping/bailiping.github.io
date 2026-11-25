(() => {
  'use strict';
  const doc=JSON.parse(document.getElementById('bento-doc').textContent);
  const topics=doc.slides.flatMap(slide=>slide.elements.filter(e=>e.type==='text'&&/class="(?:deck-link|try-live)"/.test(e.html)).map(e=>{
    const template=document.createElement('template');template.innerHTML=e.html;
    const source=template.content.querySelector('a');
    return {slide:slide.id,element:e.id,href:source.getAttribute('href'),label:source.firstElementChild.textContent,live:source.classList.contains('try-live')};
  }));
  const dialog=document.getElementById('jpda-dialog'),mount=document.getElementById('demo-frame'),loading=document.getElementById('demo-loading');
  let returnFocus=null,frame=null,loadTimer=null;
  function cleanup(){
    clearTimeout(loadTimer);if(frame){frame.src='about:blank';frame.remove();frame=null;}mount.replaceChildren();
    document.documentElement.classList.remove('demo-open');
    const trigger=returnFocus;returnFocus=null;trigger?.focus({preventScroll:true});
  }
  function close(){if(dialog.open)dialog.close();cleanup();}
  function open(topic,trigger){
    returnFocus=trigger;
    const url=new URL(topic.href,location.href);url.searchParams.set('embed','1');
    document.getElementById('demo-full').href=topic.href;
    document.getElementById('jpda-dialog-title').textContent=topic.label.includes('overlapping')?'JPDA · overlapping targets':'JPDA · one shared return';
    loading.textContent='Loading the association lab…';loading.hidden=false;
    frame=document.createElement('iframe');frame.title='Interactive two-target JPDA association lab';frame.style.visibility='hidden';
    frame.setAttribute('sandbox','allow-scripts allow-same-origin');
    dialog.showModal();document.documentElement.classList.add('demo-open');
    mount.replaceChildren(frame);frame.src=url.href;
    document.getElementById('demo-back').focus();
    loadTimer=setTimeout(()=>{loading.textContent='The lab is taking longer to load. You can return to the slides or open the full lab above.';},12000);
  }
  function sync(){
    for(const topic of topics){
      const selector=`.bento-slide[data-slide-id="${CSS.escape(topic.slide)}"] [data-el-id="${CSS.escape(topic.element)}"] .bento-text-inner`;
      for(const inner of document.querySelectorAll(selector)){
        if(inner.querySelector('a[href]'))continue;
        const a=document.createElement('a');a.className=topic.live?'try-live':'deck-link';a.href=topic.href;a.target='_self';
        const label=document.createElement('span');label.textContent=topic.label;
        const arrow=document.createElement('span');arrow.className='reference-arrow';arrow.setAttribute('aria-hidden','true');arrow.textContent='→';a.append(label,arrow);
        if(topic.live){a.setAttribute('aria-haspopup','dialog');a.setAttribute('aria-controls','jpda-dialog');}
        a.addEventListener('click',event=>{event.stopPropagation();if(topic.live&&!event.ctrlKey&&!event.metaKey&&!event.shiftKey&&!event.altKey){event.preventDefault();open(topic,a);}});
        a.addEventListener('keydown',event=>{if(['Enter',' '].includes(event.key)){event.stopPropagation();if(event.key===' '&&topic.live){event.preventDefault();open(topic,a);}}});
        inner.replaceChildren(a);
      }
    }
  }
  new MutationObserver(sync).observe(document.body,{childList:true,subtree:true});sync();
  document.getElementById('demo-back').addEventListener('click',close);
  dialog.addEventListener('close',()=>{if(!dialog.open)cleanup();});
  dialog.addEventListener('cancel',event=>{event.preventDefault();close();});
  window.addEventListener('message',event=>{
    if(event.origin!==location.origin||event.source!==frame?.contentWindow||!dialog.open)return;
    if(event.data?.type==='jpda-ready'){clearTimeout(loadTimer);loading.hidden=true;frame.style.visibility='visible';}
    if(event.data?.type==='jpda-back')close();
  });
  // Keep the parent presentation still while using controls in its modal.
  window.addEventListener('keydown',event=>{if(!dialog.open)return;event.stopImmediatePropagation();if(event.key==='Escape'){event.preventDefault();close();}},true);
  window.addEventListener('pagehide',()=>{clearTimeout(loadTimer);if(frame)frame.remove();});
})();
