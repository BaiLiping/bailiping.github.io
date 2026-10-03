(() => {
 const labs=JSON.parse(document.getElementById('grassmann-labs').textContent),dialog=document.getElementById('lab-dialog'),stage=document.getElementById('lab-stage'),back=document.getElementById('lab-back'),status=document.getElementById('lab-status');
 let frame=null,trigger=null;
 function cleanup(){if(frame){frame.contentWindow?.postMessage({type:'grassmann-pause'},location.origin);frame.remove();frame=null;}if(dialog.open)dialog.close();const focus=trigger;trigger=null;requestAnimationFrame(()=>focus?.isConnected&&focus.focus({preventScroll:true}));}
 function open(lab,control){if(dialog.open)cleanup();trigger=control;document.getElementById('lab-title').textContent=lab.title;document.getElementById('lab-full').href='../grassmannian/lab.html?lab='+lab.lab;status.hidden=false;status.textContent='Loading the experiment…';frame=document.createElement('iframe');frame.title=lab.title;frame.sandbox='allow-scripts allow-same-origin allow-top-navigation-by-user-activation';frame.addEventListener('error',()=>{status.textContent='The experiment could not load. Open the full lab using the link above.';});stage.append(frame);dialog.showModal();back.focus();frame.src='../grassmannian/lab.html?lab='+lab.lab+'&embed=slides';}
 back.addEventListener('click',cleanup);dialog.addEventListener('cancel',e=>{e.preventDefault();cleanup();});dialog.addEventListener('close',()=>{if(frame)cleanup();});
 addEventListener('message',e=>{if(e.origin!==location.origin||e.source!==frame?.contentWindow)return;if(e.data?.type==='grassmann-close')cleanup();if(e.data?.type==='grassmann-ready'){frame.dataset.ready='true';status.hidden=true;}});
 addEventListener('hashchange',()=>{if(dialog.open)cleanup();});addEventListener('pagehide',cleanup);
 function enhance(){
  for(const lab of labs)for(const slide of document.querySelectorAll(`[data-slide-id="${lab.slide}"]`)){
   const button=slide.querySelector(`[data-el-id="try-${lab.lab}-hit"]`),label=slide.querySelector(`[data-el-id="try-${lab.lab}-label"]`);
   if(label){label.dataset.labLabel='';label.setAttribute('aria-hidden','true');}
   if(!button)continue;
   if(!button.hasAttribute('data-lab-launch')){button.dataset.labLaunch=lab.lab;button.setAttribute('role','button');button.setAttribute('aria-label','Try live: '+lab.title);button.addEventListener('click',e=>{e.preventDefault();e.stopPropagation();open(lab,button);});button.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();e.stopPropagation();open(lab,button);}});}
   button.tabIndex=slide.closest('section')?.classList.contains('present')?0:-1;
  }
 }
 new MutationObserver(enhance).observe(document.body,{childList:true,subtree:true,attributes:true,attributeFilter:['class']});enhance();
 // Native external Bento links are kept in the current tab, matching this site.
 document.addEventListener('click',e=>{const link=e.target.closest?.('[data-link]');if(link&&/^https?:/.test(link.dataset.link)){e.preventDefault();e.stopImmediatePropagation();location.href=link.dataset.link;}},true);
 const nav=document.createElement('nav');nav.id='deck-shortcuts';nav.setAttribute('aria-label','Presentation resources');nav.innerHTML='<a href="../grassmannian/">Guide &amp; labs ↗</a><a href="print.html">Print edition ↗</a><a href="grassmannian.pdf">PDF ↓</a>';document.body.append(nav);
})();
