(() => {
 // Reveal's phone scroll layout can settle one page before a deep link. Align
 // explicit hashes after layout, using the actual page positions.
 let pending=Number(location.hash.match(/^#\/(\d+)$/)?.[1]),queued=false;
 function alignRoute(){
  const root=document.querySelector('.reveal.reveal-scroll.ready');
  if(!root||!Number.isInteger(pending)||queued)return;
  const pages=root.querySelectorAll('.scroll-page');
  if(!pages[pending]||document.querySelectorAll('.companion-demo-slide').length!==5)return;
  queued=true;
  requestAnimationFrame(()=>requestAnimationFrame(()=>{
   queued=false;
   const target=pages[pending];pending=NaN;
   if(target?.isConnected)root.scrollTo({top:root.scrollTop+target.getBoundingClientRect().top-root.getBoundingClientRect().top,behavior:'instant'});
  }));
 }
 new MutationObserver(alignRoute).observe(document.body,{childList:true,subtree:true,attributes:true,attributeFilter:['class']});
 addEventListener('hashchange',()=>{pending=Number(location.hash.match(/^#\/(\d+)$/)?.[1]);alignRoute();});
 alignRoute();
 // Keep external Bento links in the current tab, matching this site.
 document.addEventListener('click',e=>{const link=e.target.closest?.('[data-link]');if(link&&/^https?:/.test(link.dataset.link)){e.preventDefault();e.stopImmediatePropagation();location.href=link.dataset.link;}},true);
 const nav=document.createElement('nav');nav.id='deck-shortcuts';nav.setAttribute('aria-label','Presentation resources');nav.innerHTML='<a href="../grassmannian/">Guide &amp; labs ↗</a><a href="print.html">Print edition ↗</a><a href="grassmannian.pdf">PDF ↓</a>';document.body.append(nav);
})();
