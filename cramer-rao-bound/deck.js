(() => {
  'use strict';
  const doc=JSON.parse(document.getElementById('bento-doc').textContent);
  const topics=doc.slides.flatMap(slide=>slide.elements.filter(e=>e.type==='text'&&/class="deck-link"/.test(e.html)).map(e=>{
    const template=document.createElement('template');template.innerHTML=e.html;
    const source=template.content.querySelector('a');
    return {slide:slide.id,element:e.id,href:source.getAttribute('href'),label:source.firstElementChild.textContent};
  }));
  const labs=doc.slides.flatMap((slide,index)=>slide.id.endsWith('-lab')?[{slide,index,lab:slide.id.slice(2)}]:[]);
  const narrow=matchMedia('(max-width:700px)');
  const staticExport=new URLSearchParams(location.search).has('static');
  let active=null,queued=false;
  function focusDeck(){
    const reveal=document.querySelector('.bento-present-overlay .reveal');
    if(reveal){reveal.tabIndex=-1;reveal.focus({preventScroll:true});}
  }
  function cleanup(){
    if(!active)return;
    const focused=active.host.contains(document.activeElement);
    clearTimeout(active.timer);active.host.remove();active=null;
    if(focused)focusDeck();
  }
  function navigate(direction){
    if(!active)return;
    location.hash='#/'+Math.max(0,Math.min(doc.slides.length-1,active.entry.index+direction));
    focusDeck();
  }
  function overview(){
    focusDeck();
    // Bento reserves Escape for leaving presentation mode; O opens its overview.
    document.dispatchEvent(new KeyboardEvent('keydown',{key:'o',code:'KeyO',keyCode:79,which:79,bubbles:true}));
  }
  function mount(entry,root){
    const host=document.createElement('div');host.className='crb-inline-lab';
    host.setAttribute('role','region');host.setAttribute('aria-label','Interactive experiment');
    const header=document.createElement('header');header.className='crb-lab-heading';
    for(const [id,tag] of [['kicker','p'],['title','h1'],['subtitle','p']]){
      const node=document.createElement(tag);node.textContent=entry.slide.elements.find(e=>e.id===id).html;
      node.className='crb-lab-'+id;header.append(node);
    }
    const content=document.createElement('div');content.className='crb-lab-content';
    const loading=document.createElement('div');loading.className='crb-lab-loading';
    const status=document.createElement('p');status.setAttribute('role','status');status.textContent='Loading the experiment…';
    const retry=document.createElement('button');retry.type='button';retry.textContent='Retry';retry.hidden=true;
    retry.addEventListener('click',()=>{cleanup();schedule();});loading.append(status,retry);
    const frame=document.createElement('iframe');frame.className='crb-lab-frame';
    frame.title=entry.slide.elements.find(e=>e.id==='title').html;frame.hidden=true;
    frame.setAttribute('sandbox','allow-scripts allow-same-origin');
    content.append(loading,frame);
    const footer=document.createElement('nav');footer.className='crb-lab-navigation';footer.setAttribute('aria-label','Slide navigation');
    for(const [label,action] of [['← Previous',()=>navigate(-1)],['Overview',overview],['Next →',()=>navigate(1)]]){
      const button=document.createElement('button');button.type='button';button.textContent=label;button.addEventListener('click',action);footer.append(button);
    }
    const counter=document.createElement('span');counter.textContent=`${entry.index+1} / ${doc.slides.length}`;footer.append(counter);
    host.append(header,content,footer);
    if(narrow.matches){host.classList.add('crb-mobile-lab');root.closest('.bento-present-overlay').append(host);}
    else root.append(host);
    active={entry,root,host,frame,loading,narrow:narrow.matches,timer:setTimeout(()=>{
      status.textContent='The experiment is taking longer to load.';retry.hidden=false;
    },12000)};
    const url=new URL('./live/index.html',location.href);url.searchParams.set('lab',entry.lab);url.searchParams.set('embed','slide');frame.src=url.href;
  }
  function sync(){
    queued=false;
    for(const topic of topics){
      const selector=`.bento-slide[data-slide-id="${CSS.escape(topic.slide)}"] [data-el-id="${CSS.escape(topic.element)}"] .bento-text-inner`;
      for(const inner of document.querySelectorAll(selector)){
        if(inner.querySelector('a[href]'))continue;
        const a=document.createElement('a');a.className='deck-link';a.href=topic.href;a.target='_self';
        const label=document.createElement('span');label.textContent=topic.label;
        const arrow=document.createElement('span');arrow.className='reference-arrow';arrow.setAttribute('aria-hidden','true');arrow.textContent='→';a.append(label,arrow);
        a.addEventListener('click',event=>event.stopPropagation());
        a.addEventListener('keydown',event=>{if(event.key==='Enter')event.stopPropagation();});
        inner.replaceChildren(a);
      }
    }
    const root=document.querySelector('.bento-present-overlay .reveal:not(.overview) section.present .bento-slide');
    const entry=staticExport?null:labs.find(item=>item.slide.id===root?.dataset.slideId);
    if(active&&(active.root!==root||!entry||active.narrow!==narrow.matches))cleanup();
    if(entry&&!active)mount(entry,root);
  }
  function schedule(){if(!queued){queued=true;queueMicrotask(sync);}}
  new MutationObserver(schedule).observe(document.body,{childList:true,subtree:true,attributes:true,attributeFilter:['class']});
  narrow.addEventListener('change',schedule);
  window.addEventListener('message',event=>{
    const sameOrigin=event.origin===location.origin||(location.protocol==='file:'&&event.origin==='null');
    if(!sameOrigin||event.source!==active?.frame.contentWindow)return;
    if(event.data?.type==='crb-ready'){clearTimeout(active.timer);active.loading.hidden=true;active.frame.hidden=false;}
    if(event.data?.type==='crb-nav'&&[-1,1].includes(event.data.direction))navigate(event.data.direction);
    if(event.data?.type==='crb-overview')overview();
  });
  window.addEventListener('pagehide',cleanup);
  window.addEventListener('pageshow',schedule);
  sync();
})();
